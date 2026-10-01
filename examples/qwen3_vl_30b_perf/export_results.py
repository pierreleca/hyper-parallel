# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Distil a host-swap campaign into one small JSON file, everything the guide's figures need.

A campaign directory (``a3_campaign.sh``) is too large to copy off the cluster;
this keeps, rounded, what the results and the figures are drawn from:

* ``runs``: per run its state, overrides, budget and text layers, and its sweep row
  (memory peaks, step time, bytes moved, copy back waited for, pinned memory, routing);
* ``baseline_tables``: from the no-swap run, received pairs and memory peaks per
  (step, rank); at the step and rank that received the most, every rank's pairs and
  retained MoE memory per layer;
* ``swaps``: per swap run, the decisions and bytes per layer of the rank and step
  where the baseline received the most, the bytes a received pair saves;
* ``profiles``: per profiled run, the trace report's per-rank summary
  (``analyze_npu_trace.py --ranks``, gathered as ``<run>/profile/node*.json``).

Stdlib only, so it runs on the control node::

    python3 examples/qwen3_vl_30b_perf/export_results.py $OUT_BASE/<campaign> [--out results.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from typing import Any

from analyze_ep_instrument import call_loads, layer_memory, load_records, sweep_row

GIB = 1024 ** 3
MIB = 1024 ** 2


def _round(value: Any, digits: int = 3) -> Any:
    """Round every float in a JSON-like value, to keep the file small."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def read_plan(campaign: str) -> tuple[str, dict[str, list[str]]]:
    """Return the baseline's name and each run's overrides, as the campaign ran them."""
    plan = os.path.join(campaign, "plan.sh")
    script = f'source "{plan}"; echo "$BASELINE"; for e in "${{RUNS[@]}}"; do echo "$e"; done'
    lines = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()
    runs = {}
    for line in lines[1:]:
        words = line.split()
        if words:
            runs[words[0]] = words[1:]
    return lines[0] if lines else "", runs


def read_states(campaign: str) -> dict[str, str]:
    """Each run's state line from SUMMARY.txt: finished, or FAILED with its first error."""
    states = {}
    path = os.path.join(campaign, "SUMMARY.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                match = re.match(r"  (\S+): (finished|FAILED.*|launch failed)$", line.rstrip())
                if match:
                    states[match.group(1)] = match.group(2)[:240]
    return states


def override(words: list[str], key: str) -> str | None:
    """The value a run's overrides give to one key, if any."""
    for word in words:
        if word.startswith(f"--{key}="):
            return word.split("=", 1)[1]
    return None


def baseline_tables(run_dir: str, skip: int) -> dict[str, Any]:
    """The no-swap run: per (step, rank) its received pairs and peaks; per layer, at the step and
    rank that received the most, every rank's pairs and retained MoE memory."""
    _headers, steps = load_records(os.path.join(run_dir, "instrument"), skip)
    ranks = sorted(steps)
    by_key = {(record["step"], rank): record for rank in ranks for record in steps[rank]}
    numbers = sorted({step for step, _rank in by_key})
    loads = {key: call_loads(record) for key, record in by_key.items()}
    layers = sorted({layer for load in loads.values() for layer in load})
    totals = {key: sum(entry["recv"] for entry in load.values()) for key, load in loads.items()}
    step, rank = max(totals, key=lambda key: (totals[key], key))

    def per_step_rank(value) -> dict[str, list]:
        return {str(number): [value(number, column) if (number, column) in by_key else None for column in ranks]
                for number in numbers}

    def peak(name: str):
        return lambda number, column: round(by_key[(number, column)].get("memory", {}).get(name, 0) / GIB, 3) or None

    worst = {}
    for column in ranks:
        memory = layer_memory(by_key[(step, column)]) if (step, column) in by_key else {}
        load = loads.get((step, column), {})
        worst[column] = {
            "recv": [load.get(layer, {}).get("recv", 0) for layer in layers],
            "sent": [load.get(layer, {}).get("send", 0) for layer in layers],
            "retained_mib": [round(memory[layer][0] / MIB, 1) if layer in memory else None for layer in layers],
        }
    return {
        "ranks": ranks, "layers": layers, "steps": numbers, "worst": {"step": step, "rank": rank},
        "recv_total": per_step_rank(lambda number, column: totals[(number, column)]),
        "peak_reserved_gib": per_step_rank(peak("step_peak_reserved")),
        "peak_allocated_gib": per_step_rank(peak("step_peak_allocated")),
        "at_worst_step": {name: [worst[column][name] for column in ranks] for name in ("recv", "sent", "retained_mib")},
    }


def swap_at(swap_dir: str, rank: int, step: int) -> dict[str, Any] | None:
    """One rank's swap record of one step: bytes per layer, decisions, and the bytes a received pair saves."""
    path = os.path.join(swap_dir, f"host_swap_rank{rank}.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    records = [record for record in records if not record.get("header")]
    pair = [sum(saved[2] for saved in layer["saved"]) / layer["rows"]
            for record in records for layer in record["layers"] if layer["rows"] and layer["saved"]]
    matches = [record for record in records if record["step"] == step]
    if not matches:
        return None
    record = matches[-1]
    return {
        "rank": rank, "step": step, "pair_bytes": pair[0] if pair else None,
        "swapped_gib": {str(layer["index"]): layer["swapped_bytes"] / GIB for layer in record["layers"]},
        "stall_ms": {str(layer["index"]): layer["stall_ms"] for layer in record["layers"]},
        "evictions": record.get("evictions", []),
        "d2h_gib": record["d2h_gib"], "pinned_gib": record.get("pinned_gib"),
    }


def profile_summary(run_dir: str) -> dict[str, Any]:
    """The per-rank trace summaries every node wrote, merged."""
    merged: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "profile", "*.json"))):
        with open(path, encoding="utf-8") as stream:
            merged[os.path.basename(path)[:-5]] = json.load(stream)
    return merged


def export(campaign: str, skip: int) -> dict[str, Any]:
    """Everything the figures and the results need from one campaign."""
    baseline, plan = read_plan(campaign)
    states = read_states(campaign)
    code = ""
    log = os.path.join(campaign, "campaign.log")
    if os.path.exists(log):
        with open(log, encoding="utf-8") as stream:
            code = next((line.strip() for line in stream if line.startswith("code: ")), "")
    result: dict[str, Any] = {"campaign": os.path.basename(campaign.rstrip("/")), "code": code,
                              "baseline": baseline, "runs": {}, "swaps": {}, "profiles": {}}
    for name, words in plan.items():
        run_dir = os.path.join(campaign, name)
        entry: dict[str, Any] = {
            "state": states.get(name, "unknown"), "overrides": " ".join(words),
            "layers": override(words, "model.num_hidden_layers"),
            "budget_layers": None if override(words, "ep_host_swap.enabled") == "false"
            else override(words, "ep_host_swap.budget_layers"),
        }
        if os.path.isdir(os.path.join(run_dir, "instrument")):
            entry["sweep"] = sweep_row(run_dir, skip)
        result["runs"][name] = entry
        profile = profile_summary(run_dir)
        if profile:
            result["profiles"][name] = profile
    if baseline and os.path.isdir(os.path.join(campaign, baseline, "instrument")):
        base = baseline_tables(os.path.join(campaign, baseline), skip)
        result["baseline_tables"] = base
        step, rank = base["worst"]["step"], base["worst"]["rank"]
        base_layers = override(plan[baseline], "model.num_hidden_layers")
        for name, words in plan.items():
            if override(words, "model.num_hidden_layers") != base_layers:
                continue  # another depth routes differently
            record = swap_at(os.path.join(campaign, name, "ep_host_swap"), rank, step)
            if record is not None:
                result["swaps"][name] = record
    return _round(result)


def main() -> int:
    """Write the campaign's results file and say how large it is."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("campaign", help="the campaign directory a3_campaign.sh wrote")
    parser.add_argument("--skip", type=int, default=2, help="warm-up steps to drop, as the reports do")
    parser.add_argument("--out", default=None, help="file to write (default: <campaign>/results.json)")
    args = parser.parse_args()
    out = args.out or os.path.join(args.campaign, "results.json")
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(export(args.campaign, args.skip), stream, separators=(",", ":"))
    print(f"wrote {out}: {os.path.getsize(out) / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
