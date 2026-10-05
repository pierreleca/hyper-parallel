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
"""Compare a candidate with a baseline: the speedup and its interval, where it comes from, and whether numerics held.

A claim of "20% faster" needs more than two step times. This script reads the per-rank records of
``hetero_profile`` (``hooks: false`` is enough: it costs two device events and one synchronization per step),
pools the repeats of each arm, and reports

- **speedup:** the ratio of work per second, end to end (the step plus the host's gap before it), with a
  bootstrap confidence interval over steps. When both arms ran the same samples in the same steps (found by the
  fingerprint of each micro-batch) the comparison is also made step by step, which removes the variance that the
  data itself brings; that paired interval decides the verdict against ``--target``;
- **throughput in three units:** samples, tokens and work per second, where work is ``tokens + cost_visual *
  visual_tokens``, so an arm that was handed other samples (a regrouped order) is not credited for lighter ones;
- **where the time went:** with module hooks on in both arms, each component's cost per 1k tokens (per 1k patches
  for the vision tower), the imbalance between ranks, and what the new design adds (the ``custom`` part);
- **numerics:** the loss and the gradient norm of the paired steps; a speedup that moved them is not a speedup;
- **memory:** the peak allocated bytes.

A log of the trainer (``log.txt`` of a campaign run) stands in for an arm that ran without the recorder: it gives
the step time the trainer measured, the loss and the gradient norm, not the tokens.

    python examples/qwen3_vl_30b_perf/compare_runs.py --baseline runA/hetero runB/hetero --candidate runC/hetero

Two runs of the same configuration measure the noise floor: their "speedup" is what no change can claim.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import re
import statistics
from typing import Any, Optional, Sequence

TARGET = 0.20
COST_VISUAL = 0.5
BOOTSTRAPS = 2000
_STEP_LINE = re.compile(r"\bstep=(\d+)\b")
_FIELD = re.compile(r"\b([A-Za-z_]+/[A-Za-z_]+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


# -- loading ----------------------------------------------------------------------------------------------

def _load_module(name: str) -> Any:
    """Import a script that sits beside this one."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records_dir(path: str) -> Optional[str]:
    """Return the directory of the per-rank record files of a run, or None when it has none."""
    for candidate in (path, os.path.join(path, "hetero")):
        if glob.glob(os.path.join(candidate, "rank*.jsonl")):
            return candidate
    return None


def steps_from_records(directory: str, skip: int, cost_visual: float) -> tuple[list[dict[str, Any]], bool]:
    """Return one dict per step, joined over the ranks, and whether the module hooks were on.

    The step's time is the slowest rank's, as the trainer measures it; end to end adds the slowest rank's gap
    before the step (data loading, callbacks). ``samples`` fingerprints identify the step's samples.
    """
    per_step: dict[int, list[dict]] = {}
    expected = 0
    hooks = False
    for path in sorted(glob.glob(os.path.join(directory, "rank*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            lines = [json.loads(line) for line in stream if line.strip()]
        expected += 1
        hooks = hooks or bool(lines[0].get("hooks", True) and lines[0].get("modules"))
        for record in [line for line in lines[1:] if line.get("kind") == "step"][skip:]:
            per_step.setdefault(record["step"], []).append(record)
    steps = []
    for step, records in sorted(per_step.items()):
        if len(records) < expected:
            continue
        micro = [batch for record in records for batch in record["micro_batches"]]
        tokens = sum(batch.get("real_tokens", 0) for batch in micro)
        visual = sum(batch.get("visual_tokens", 0) for batch in micro)
        step_ms = max(record["wall_ms"] for record in records)
        end_to_end = max(record["wall_ms"] + (record.get("inter_step_ms") or 0.0) for record in records)
        losses = [record["loss"] for record in records if record.get("loss") is not None]
        norms = [record["grad_norm"] for record in records if record.get("grad_norm") is not None]
        peaks = [record["peak_allocated"] for record in records if record.get("peak_allocated")]
        steps.append({
            "step": step, "step_ms": step_ms, "e2e_ms": end_to_end,
            "samples": sum(b.get("batch_size", 1) for b in micro), "tokens": tokens, "visual": visual,
            "work": tokens + cost_visual * visual,
            "samples_key": tuple(sorted(batch.get("fingerprint", 0) for batch in micro)),
            "loss": losses[0] if losses else None, "grad_norm": norms[0] if norms else None,
            "peak": max(peaks) if peaks else None,
        })
    return steps, hooks


def steps_from_log(path: str, skip: int) -> list[dict[str, Any]]:
    """Return one dict per step from a trainer log: the trainer's step time, loss and gradient norm."""
    steps: dict[int, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            found = _STEP_LINE.search(line)
            if found is None:
                continue
            fields = {name: float(value) for name, value in _FIELD.findall(line)}
            if "performance/step_time" in fields:
                steps.setdefault(int(found.group(1)), {}).update(fields)
    result = []
    for step in sorted(steps)[skip:]:
        fields = steps[step]
        step_ms = fields["performance/step_time"] * 1e3
        peak = fields.get("memory/device_max_allocated_gb")
        result.append({
            "step": step, "step_ms": step_ms, "e2e_ms": step_ms, "samples": int(fields.get("data/step_samples", 0)),
            "tokens": None, "visual": None, "work": None, "samples_key": None,
            "loss": fields.get("training/total_loss"), "grad_norm": fields.get("training/grad_norm"),
            "peak": peak * 2 ** 30 if peak else None,
        })
    return result


class Arm:
    """The steps of one side of a comparison, pooled over its runs."""

    def __init__(self, paths: Sequence[str], skip: int, cost_visual: float = COST_VISUAL) -> None:
        """Load every run; a run is a record directory, a campaign run directory, or a trainer log."""
        self.paths = list(paths)
        self.runs: list[list[dict[str, Any]]] = []
        self.hooked_dirs: list[str] = []
        for path in paths:
            directory = records_dir(path)
            if directory is not None:
                steps, hooks = steps_from_records(directory, skip, cost_visual)
                if hooks:
                    self.hooked_dirs.append(directory)
            elif os.path.isfile(path) or os.path.isfile(os.path.join(path, "log.txt")):
                steps = steps_from_log(path if os.path.isfile(path) else os.path.join(path, "log.txt"), skip)
            else:
                raise SystemExit(f"{path}: no rank*.jsonl records and no log.txt")
            if not steps:
                raise SystemExit(f"{path}: no complete step after skipping {skip}")
            self.runs.append(steps)
        self.steps = [step for run in self.runs for step in run]

    @property
    def name(self) -> str:
        """The arm's runs, by directory name."""
        names = []
        for path in self.paths:
            base = os.path.abspath(path)
            names.append(os.path.basename(os.path.dirname(base)) if os.path.basename(base) == "hetero"
                         else os.path.basename(base))
        return "+".join(names)

    @property
    def has_tokens(self) -> bool:
        """Whether every step knows its tokens (records, not a log)."""
        return all(step["tokens"] is not None for step in self.steps)


# -- statistics -------------------------------------------------------------------------------------------

def mean(values: Sequence[float]) -> float:
    """Return the mean, or 0 for an empty list."""
    return statistics.fmean(values) if values else 0.0


def throughput(steps: Sequence[dict], quantity: str) -> Optional[float]:
    """Return a quantity per second over the steps, end to end (None if a step lacks it)."""
    if any(step[quantity] is None for step in steps):
        return None
    seconds = sum(step["e2e_ms"] for step in steps) / 1e3
    return sum(step[quantity] for step in steps) / seconds if seconds else None


def percentile(values: Sequence[float], share: float) -> float:
    """Return the value at this share of the sorted list."""
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


def bootstrap_speedup(baseline: Sequence[dict], candidate: Sequence[dict], quantity: Optional[str],
                      draws: int = BOOTSTRAPS, seed: int = 0) -> tuple[float, float, float]:
    """Return (speedup, low, high): the ratio of the candidate's to the baseline's rate, minus one, 95% interval.

    The rate is ``quantity`` per second end to end, or the inverse of the step time when ``quantity`` is None.
    Steps are resampled with replacement within each arm.
    """
    def rate(steps: Sequence[dict]) -> float:
        """The arm's rate over these steps."""
        if quantity is None:
            return len(steps) / (sum(step["e2e_ms"] for step in steps) / 1e3)
        return sum(step[quantity] for step in steps) / (sum(step["e2e_ms"] for step in steps) / 1e3)

    point = rate(candidate) / rate(baseline) - 1.0
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        first = [baseline[rng.randrange(len(baseline))] for _ in baseline]
        second = [candidate[rng.randrange(len(candidate))] for _ in candidate]
        ratios.append(rate(second) / rate(first) - 1.0)
    return point, percentile(ratios, 0.025), percentile(ratios, 0.975)


def pair_steps(baseline: Sequence[dict], candidate: Sequence[dict]) -> list[tuple[float, float, float]]:
    """Return (baseline ms, candidate ms, work) for every set of samples that both arms ran in one step.

    A set seen in several steps (a repeated run) is averaged within each arm.
    """
    def by_key(steps: Sequence[dict]) -> dict[Any, list[dict]]:
        """Group the steps by the fingerprints of their samples."""
        grouped: dict[Any, list[dict]] = {}
        for step in steps:
            if step["samples_key"] is not None:
                grouped.setdefault(step["samples_key"], []).append(step)
        return grouped

    first, second = by_key(baseline), by_key(candidate)
    return [(mean([s["e2e_ms"] for s in steps]), mean([s["e2e_ms"] for s in second[key]]),
             mean([s["work"] or 0.0 for s in steps])) for key, steps in first.items() if key in second]


def bootstrap_paired(pairs: Sequence[tuple[float, float, float]], draws: int = BOOTSTRAPS,
                     seed: int = 0) -> tuple[float, float, float]:
    """Return (speedup, low, high) over paired steps: total baseline time over total candidate time, minus one."""
    def ratio(chosen: Sequence[tuple[float, float, float]]) -> float:
        """Time saved over the chosen pairs."""
        return sum(p[0] for p in chosen) / sum(p[1] for p in chosen) - 1.0

    rng = random.Random(seed)
    draws_list = [ratio([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(draws)]
    return ratio(pairs), percentile(draws_list, 0.025), percentile(draws_list, 0.975)


# -- the report -------------------------------------------------------------------------------------------

def _component_costs(arm: Arm, skip: int) -> Optional[dict[str, Any]]:
    """Per-1k-token cost of each component, and the imbalance, over the arm's runs with module hooks."""
    if not arm.hooked_dirs:
        return None
    analyzer = _load_module("analyze_hetero")
    rows: list[dict] = []
    imbalance: list[dict] = []
    for directory in arm.hooked_dirs:
        run = analyzer.Run(directory, skip)
        run_rows = analyzer.build_rows(run)
        rows += run_rows
        imbalance.append(analyzer.report_imbalance(run_rows, {}, []))
    tokens = mean([row.get("real_tokens", 0) for row in rows]) or 1.0
    patches = mean([row.get("patches", 0) for row in rows]) or 1.0
    costs = {}
    for part in (*analyzer.WORK_PARTS, "text_attn", "text_experts", "text_moe", "ep_exchange"):
        total = mean([analyzer.component_time(row, part) for row in rows])
        costs[part] = total / (patches if part == "vision" else tokens) * 1e3
    names: dict[str, float] = {}
    for row in rows:
        for name, phases in row.get("custom_by_name", {}).items():
            names[name] = names.get(name, 0.0) + sum(phases.values())
    return {"per_1k": costs, "custom": {name: value / len(rows) for name, value in names.items()},
            "busiest_over_mean": mean([i["busiest_over_mean"] for i in imbalance]),
            "excess_share": mean([i["excess_share_of_step"] for i in imbalance])}


def _numerics(baseline: Arm, candidate: Arm, tolerance: float, out: list[str]) -> dict[str, Any]:
    """Compare loss and gradient norm over the steps both arms ran on the same samples."""
    def lookup(arm: Arm) -> dict[Any, list[dict]]:
        """Index the steps by their samples."""
        grouped: dict[Any, list[dict]] = {}
        for step in arm.steps:
            if step["samples_key"] is not None:
                grouped.setdefault(step["samples_key"], []).append(step)
        return grouped

    first, second = lookup(baseline), lookup(candidate)
    shared = [key for key in first if key in second]
    summary: dict[str, Any] = {"paired_steps": len(shared)}
    if not shared:
        out.append("  numerics: the arms ran no step on the same samples, so loss and gradient norm are not "
                   "comparable step by step; compare the loss curves over consumed samples")
        return summary
    for quantity, label in (("loss", "loss"), ("grad_norm", "gradient norm")):
        differences = []
        for key in shared:
            base = [s[quantity] for s in first[key] if s[quantity] is not None]
            cand = [s[quantity] for s in second[key] if s[quantity] is not None]
            if base and cand:
                differences.append(abs(mean(cand) - mean(base)) / max(abs(mean(base)), 1e-12))
        if differences:
            worst, average = max(differences), mean(differences)
            verdict = "ok" if worst <= tolerance else "DIFFERS"
            out.append(f"  numerics, {label}: {len(differences)} paired steps, relative difference mean "
                       f"{average:.2%}, worst {worst:.2%} (tolerance {tolerance:.0%}) -> {verdict}")
            summary[quantity] = {"mean": average, "worst": worst, "ok": worst <= tolerance}
    return summary


def compare(baseline: Arm, candidate: Arm, target: float = TARGET, skip: int = 0, tolerance: float = 0.03,
            draws: int = BOOTSTRAPS) -> tuple[list[str], dict[str, Any]]:
    """Build the comparison report; return its lines and a JSON-friendly summary."""
    out: list[str] = []
    summary: dict[str, Any] = {"baseline": baseline.name, "candidate": candidate.name, "target": target}
    out.append(f"A/B: baseline {baseline.name} ({len(baseline.runs)} run(s), {len(baseline.steps)} steps) against "
               f"candidate {candidate.name} ({len(candidate.runs)} run(s), {len(candidate.steps)} steps)")
    rows = [("step time (slowest rank)", "step_ms"), ("end to end (+ gap before the step)", "e2e_ms")]
    out.append("                                     baseline      candidate      speedup (95% interval)")
    for label, key in rows:
        base, cand = mean([s[key] for s in baseline.steps]), mean([s[key] for s in candidate.steps])
        point, low, high = bootstrap_speedup(
            [{**s, "e2e_ms": s[key]} for s in baseline.steps], [{**s, "e2e_ms": s[key]} for s in candidate.steps],
            None, draws)
        out.append(f"  {label:34s} {base:9.1f} ms   {cand:9.1f} ms   {point:+7.1%}  [{low:+.1%}, {high:+.1%}]")
        summary[key] = {"baseline": base, "candidate": cand, "speedup": point, "low": low, "high": high}
    primary = None
    if baseline.has_tokens and candidate.has_tokens:
        for label, quantity in (("samples per second", "samples"), ("tokens per second", "tokens"),
                                (f"work per second (tokens + {COST_VISUAL} visual)", "work")):
            base, cand = throughput(baseline.steps, quantity), throughput(candidate.steps, quantity)
            point, low, high = bootstrap_speedup(baseline.steps, candidate.steps, quantity, draws)
            out.append(f"  {label:34s} {base:9.1f}      {cand:9.1f}      {point:+7.1%}  [{low:+.1%}, {high:+.1%}]")
            summary[quantity] = {"baseline": base, "candidate": cand, "speedup": point, "low": low, "high": high}
        primary = summary["work"]
    else:
        out.append("  (a trainer log carries no token counts: only the step time is compared)")
    pairs = pair_steps(baseline.steps, candidate.steps)
    decisive = primary or summary["e2e_ms"]
    label = "work per second, unpaired"
    if len(pairs) >= 5:
        point, low, high = bootstrap_paired(pairs, draws)
        out.append(f"  paired steps (same samples in the same step): {len(pairs)}; end-to-end time saved "
                   f"{point:+.1%} [{low:+.1%}, {high:+.1%}]")
        summary["paired"] = {"steps": len(pairs), "speedup": point, "low": low, "high": high}
        decisive, label = summary["paired"], "paired steps"
    else:
        out.append(f"  paired steps: {len(pairs)} (fewer than 5): the arms ran other samples in the steps, so the "
                   "comparison rests on work per second")
    if primary is None and len(pairs) < 5:
        label = "step time, unpaired"
    verdict = ("MET" if decisive["low"] >= target else "NOT MET" if decisive["high"] < target else "INCONCLUSIVE")
    out.append(f"  verdict against +{target:.0%} ({label}): {decisive['speedup']:+.1%} "
               f"[{decisive['low']:+.1%}, {decisive['high']:+.1%}] -> {verdict}")
    summary["verdict"] = verdict
    runs = [mean([s["e2e_ms"] for s in run]) for run in baseline.runs]
    if len(runs) > 1:
        spread = (max(runs) - min(runs)) / mean(runs)
        out.append(f"  noise: the baseline's {len(runs)} runs differ by {spread:.1%} in mean end-to-end step time")
        summary["noise"] = spread
    summary["numerics"] = _numerics(baseline, candidate, tolerance, out)
    peaks = [mean([s["peak"] for s in arm.steps if s["peak"]]) for arm in (baseline, candidate)]
    if all(peaks):
        out.append(f"  memory: mean per-step peak allocated {peaks[0] / 2 ** 30:.1f} GiB -> "
                   f"{peaks[1] / 2 ** 30:.1f} GiB")
    first, second = _component_costs(baseline, skip), _component_costs(candidate, skip)
    if first and second:
        out.append("")
        out.append("  where the time moved (ms per 1k tokens of the micro-batch; the vision tower per 1k patches)")
        out.append("    component        baseline   candidate     change")
        for part, base in first["per_1k"].items():
            cand = second["per_1k"][part]
            if base or cand:
                out.append(f"    {part:14s} {base:9.2f}   {cand:9.2f}   {(cand - base) / base if base else 0:+9.1%}")
        for name, value in second["custom"].items():
            out.append(f"    custom '{name}': {value:.1f} ms per micro-batch added by the candidate")
        out.append(f"    busiest rank over the mean: {first['busiest_over_mean']:.3f} -> "
                   f"{second['busiest_over_mean']:.3f}; its excess as a share of the step: "
                   f"{first['excess_share']:.1%} -> {second['excess_share']:.1%}")
        summary["components"] = {"baseline": first, "candidate": second}
    return out, summary


def main() -> int:
    """Parse the command line and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", nargs="+", required=True, help="baseline runs (repeats are pooled)")
    parser.add_argument("--candidate", nargs="+", required=True, help="candidate runs (repeats are pooled)")
    parser.add_argument("--target", type=float, default=TARGET, help="speedup to certify (0.20 is 20%%)")
    parser.add_argument("--skip", type=int, default=1,
                        help="recorded steps to drop from each run (the first has no gap before it)")
    parser.add_argument("--cost-visual", type=float, default=COST_VISUAL,
                        help="extra work of a visual token over a text token")
    parser.add_argument("--loss-tol", type=float, default=0.03, help="relative loss difference that still passes")
    parser.add_argument("--draws", type=int, default=BOOTSTRAPS, help="bootstrap draws")
    parser.add_argument("--json", default=None, help="also write the summary here")
    args = parser.parse_args()
    baseline = Arm(args.baseline, args.skip, args.cost_visual)
    candidate = Arm(args.candidate, args.skip, args.cost_visual)
    lines, summary = compare(baseline, candidate, args.target, args.skip, args.loss_tol, args.draws)
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
