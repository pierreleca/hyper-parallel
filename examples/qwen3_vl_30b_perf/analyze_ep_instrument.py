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
"""Summarize the expert-parallel imbalance records of one run.

Reads the per-rank JSON Lines files written by
``hyper_parallel.distributed.expert_parallel.instrument`` and reports

- **routing:** the load each rank receives per layer, its imbalance
  ``lambda = max / mean``, which rank is the busiest and how persistent that
  is across layers and steps;
- **time:** the device time of every MoE phase per rank, the imbalance of the
  expert GEMM, and the idle time the other ranks spend waiting for the
  busiest one;
- **memory:** each rank's peak, the phase that holds it, and how far the
  ranks differ;
- **offload headroom:** the activation bytes the busiest rank holds above the
  mean, against the time between MoE blocks that a transfer could hide in.

It writes CSV files beside the records and, with ``--trace``, a Chrome trace
of one step that opens in chrome://tracing or Perfetto.

    python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py <record-dir> --skip 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from typing import Any

FWD_PHASES = (
    ("start", "routed", "router+counts"),
    ("routed", "dispatched", "dispatch a2a"),
    ("dispatched", "experts", "experts"),
    ("experts", "combined", "combine a2a"),
    ("combined", "end", "aggregate"),
)
BWD_PHASES = (
    ("start", "aggregate", "aggregate bwd"),
    ("aggregate", "combine", "combine a2a bwd"),
    ("combine", "experts", "experts bwd"),
    ("experts", "dispatch", "dispatch a2a bwd"),
    ("dispatch", "end", "router+gather bwd"),
)
GIB = 1024 ** 3


def load_records(record_dir: str, skip: int) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    """Read every rank file; return the headers and the kept step records."""
    headers: dict[int, dict] = {}
    steps: dict[int, list[dict]] = {}
    paths = sorted(glob.glob(os.path.join(record_dir, "rank*.jsonl")))
    if not paths:
        raise SystemExit(f"no rank*.jsonl files in {record_dir}")
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            lines = [json.loads(line) for line in stream if line.strip()]
        if not lines or lines[0].get("kind") != "header":
            raise SystemExit(f"{path}: first line is not a header")
        rank = int(lines[0]["rank"])
        headers[rank] = lines[0]
        records = [line for line in lines[1:] if line.get("kind") == "step"]
        steps[rank] = records[skip:]
    return headers, steps


def mark_spans(record: dict) -> dict[tuple[int, str, int], tuple[float, float]]:
    """Return {(layer, pass, occurrence): (first mark, last mark)} in ms.

    A checkpointed block stops recomputing as soon as its last saved tensor
    is back, so a pass is bounded by the marks it did reach, not by ``end``.
    """
    spans: dict[tuple[int, str, int], tuple[float, float]] = {}
    for layer, pass_name, occurrence, _name, time_ms, _alloc, _peak in record["marks"]:
        if layer < 0:
            continue
        key = (layer, pass_name, occurrence)
        first, last = spans.get(key, (time_ms, time_ms))
        spans[key] = (min(first, time_ms), max(last, time_ms))
    return spans


def _overlap(begin: float, end: float, spans: list[tuple[float, float]]) -> float:
    """Return how much of [begin, end) the given spans cover."""
    return sum(
        max(0.0, min(end, span_end) - max(begin, span_begin))
        for span_begin, span_end in spans
    )


def phase_times(record: dict) -> dict[tuple[int, str, str], float]:
    """Return {(layer, pass, phase): milliseconds} for one step record.

    A checkpointed block is recomputed inside its own backward pass, so the
    recompute's time is subtracted from the backward phase that contains it;
    it stays visible under the ``recompute`` pass.
    """
    stamps: dict[tuple[int, str, int, str], float] = {}
    for layer, pass_name, occurrence, name, time_ms, _alloc, _peak in record["marks"]:
        stamps[(layer, pass_name, occurrence, name)] = time_ms
    recompute_spans = [
        span for (_layer, pass_name, _occurrence), span in mark_spans(record).items()
        if pass_name == "recompute"
    ]
    times: dict[tuple[int, str, str], float] = {}
    for (layer, pass_name, occurrence, name) in list(stamps):
        if name != "start":
            continue
        phases = BWD_PHASES if pass_name == "bwd" else FWD_PHASES
        for first, second, label in phases:
            begin = stamps.get((layer, pass_name, occurrence, first))
            end = stamps.get((layer, pass_name, occurrence, second))
            if begin is None or end is None:
                continue
            duration = end - begin
            if pass_name == "bwd":
                duration -= _overlap(begin, end, recompute_spans)
            key = (layer, pass_name, label)
            times[key] = times.get(key, 0.0) + duration
    return times


def call_loads(record: dict) -> dict[int, dict[str, Any]]:
    """Return {layer: routing counts} for the forward pass of one step."""
    loads: dict[int, dict[str, Any]] = {}
    for call in record["calls"]:
        if call["pass"] != "fwd" or "recv" not in call:
            continue
        layer = call["layer"]
        entry = loads.setdefault(
            layer, {"recv": 0, "send": 0, "tokens": 0, "expert_counts": None}
        )
        entry["recv"] += sum(call["recv"])
        entry["send"] += sum(call["send"])
        entry["tokens"] += call.get("tokens", 0)
        counts = call.get("expert_counts")
        if counts is not None:
            if entry["expert_counts"] is None:
                entry["expert_counts"] = list(counts)
            else:
                entry["expert_counts"] = [
                    a + b for a, b in zip(entry["expert_counts"], counts)
                ]
    return loads


def moe_gaps(record: dict) -> list[float]:
    """Return the milliseconds between consecutive MoE blocks in one step."""
    spans = sorted(mark_spans(record).values())
    return [
        later[0] - earlier[1]
        for earlier, later in zip(spans, spans[1:])
        if later[0] > earlier[1]
    ]


def imbalance(values: list[float]) -> float:
    """Return max / mean, the imbalance factor used throughout the report."""
    mean = statistics.fmean(values) if values else 0.0
    return max(values) / mean if mean > 0 else 1.0


def collect(headers: dict, steps: dict) -> dict[str, Any]:
    """Build the per-(step, layer, rank) table every section reads."""
    ranks = sorted(headers)
    rows: list[dict[str, Any]] = []
    per_step_rank: dict[tuple[int, int], dict[str, Any]] = {}
    for rank in ranks:
        for record in steps[rank]:
            step = record["step"]
            times = phase_times(record)
            loads = call_loads(record)
            memory = record.get("memory", {})
            per_step_rank[(step, rank)] = {
                "wall_s": record.get("wall_s"),
                "peak_allocated": memory.get("step_peak_allocated"),
                "peak_reserved": memory.get("step_peak_reserved"),
                "recv_total": sum(load["recv"] for load in loads.values()),
                "gaps_ms": moe_gaps(record),
                "marks": record["marks"],
            }
            for layer, load in sorted(loads.items()):
                row = {
                    "step": step,
                    "layer": layer,
                    "rank": rank,
                    "recv": load["recv"],
                    "send": load["send"],
                    "tokens": load["tokens"],
                    "expert_counts": load["expert_counts"],
                }
                for pass_name, phases in (("fwd", FWD_PHASES), ("recompute", FWD_PHASES),
                                          ("bwd", BWD_PHASES)):
                    for _first, _second, label in phases:
                        row[f"{pass_name}:{label}"] = times.get((layer, pass_name, label))
                rows.append(row)
    return {"ranks": ranks, "rows": rows, "per_step_rank": per_step_rank}


def _by_step_layer(rows: list[dict]) -> dict[tuple[int, int], dict[int, dict]]:
    """Index the rows as {(step, layer): {rank: row}}."""
    table: dict[tuple[int, int], dict[int, dict]] = {}
    for row in rows:
        table.setdefault((row["step"], row["layer"]), {})[row["rank"]] = row
    return table


def report_routing(table: dict, ranks: list[int], out: list[str]) -> dict[str, Any]:
    """Report the token imbalance per layer and how persistent it is."""
    lambdas: list[float] = []
    hot_counts: dict[int, int] = {rank: 0 for rank in ranks}
    per_layer: dict[int, list[float]] = {}
    for (_step, layer), per_rank in sorted(table.items()):
        loads = [per_rank[rank]["recv"] for rank in ranks if rank in per_rank]
        if len(loads) < len(ranks):
            continue
        factor = imbalance(loads)
        lambdas.append(factor)
        per_layer.setdefault(layer, []).append(factor)
        hot_counts[ranks[loads.index(max(loads))]] += 1
    if not lambdas:
        out.append("routing: no complete layer records")
        return {}
    ordered = sorted(lambdas)
    summary = {
        "lambda_mean": statistics.fmean(lambdas),
        "lambda_median": statistics.median(lambdas),
        "lambda_p90": ordered[int(0.9 * (len(ordered) - 1))],
        "lambda_max": max(lambdas),
    }
    out.append("ROUTING IMBALANCE (per layer, per step: max / mean received tokens)")
    out.append(
        f"  lambda: mean {summary['lambda_mean']:.3f}  median {summary['lambda_median']:.3f}"
        f"  p90 {summary['lambda_p90']:.3f}  max {summary['lambda_max']:.3f}"
    )
    out.append("  busiest rank, share of (step, layer) pairs: " + ", ".join(
        f"r{rank} {count / len(lambdas):.0%}" for rank, count in sorted(hot_counts.items())
    ))
    worst = sorted(per_layer.items(), key=lambda item: -statistics.fmean(item[1]))[:5]
    out.append("  layers with the highest mean lambda: " + ", ".join(
        f"L{layer} {statistics.fmean(values):.2f}" for layer, values in worst
    ))
    return summary


def report_persistence(collected: dict, out: list[str]) -> dict[str, Any]:
    """Report the imbalance of each rank's load summed over all layers."""
    ranks = collected["ranks"]
    steps = sorted({step for step, _rank in collected["per_step_rank"]})
    factors = []
    for step in steps:
        totals = [
            collected["per_step_rank"][(step, rank)]["recv_total"]
            for rank in ranks
            if (step, rank) in collected["per_step_rank"]
        ]
        if len(totals) == len(ranks) and sum(totals):
            factors.append(imbalance(totals))
    if not factors:
        return {}
    summary = {"lambda_step_mean": statistics.fmean(factors), "lambda_step_max": max(factors)}
    out.append("PERSISTENCE (load summed over all layers, per rank)")
    out.append(
        f"  step-level lambda: mean {summary['lambda_step_mean']:.3f}"
        f"  max {summary['lambda_step_max']:.3f}"
    )
    out.append(
        "  A step-level lambda near 1 means the busy rank changes between layers, so"
    )
    out.append(
        "  saved activations even out; a high one means one rank is hot all the way down."
    )
    return summary


def report_time(table: dict, ranks: list[int], out: list[str]) -> dict[str, Any]:
    """Report per-phase device time, its imbalance and the implied idle time."""
    phase_keys = [f"fwd:{label}" for *_x, label in FWD_PHASES]
    phase_keys += [f"recompute:{label}" for *_x, label in FWD_PHASES]
    phase_keys += [f"bwd:{label}" for *_x, label in BWD_PHASES]
    totals: dict[str, dict[int, float]] = {key: {rank: 0.0 for rank in ranks} for key in phase_keys}
    idle_ms = {rank: 0.0 for rank in ranks}
    expert_factors: list[float] = []
    for (_step, _layer), per_rank in table.items():
        for key in phase_keys:
            for rank, row in per_rank.items():
                value = row.get(key)
                if value is not None:
                    totals[key][rank] += value
        for key in ("fwd:experts", "bwd:experts bwd"):
            values = [per_rank[rank].get(key) for rank in ranks if rank in per_rank]
            if any(value is None for value in values) or len(values) < len(ranks):
                continue
            slowest = max(values)
            if key == "fwd:experts" and statistics.fmean(values) > 0:
                expert_factors.append(slowest / statistics.fmean(values))
            for rank, value in zip(ranks, values):
                idle_ms[rank] += slowest - value
    steps = max(1, len({step for step, _layer in table}))
    out.append("TIME (device ms per step, summed over layers)")
    out.append("  phase                     " + "".join(f"  r{rank:<8}" for rank in ranks))
    for key in phase_keys:
        values = [totals[key][rank] / steps for rank in ranks]
        if max(values) < 0.05:
            continue
        out.append(f"  {key:<26}" + "".join(f"{value:10.2f}" for value in values))
    summary: dict[str, Any] = {}
    if expert_factors:
        summary["expert_time_lambda_mean"] = statistics.fmean(expert_factors)
        out.append(
            f"  expert GEMM time imbalance (max/mean per layer): "
            f"mean {summary['expert_time_lambda_mean']:.3f}"
        )
    waiting = [idle_ms[rank] / steps for rank in ranks]
    summary["idle_ms_per_step"] = dict(zip(map(str, ranks), waiting))
    out.append("  waiting for the busiest rank's experts, ms/step: " + ", ".join(
        f"r{rank} {value:.1f}" for rank, value in zip(ranks, waiting)
    ))
    return summary


def report_memory(collected: dict, out: list[str]) -> dict[str, Any]:
    """Report each rank's peak and the phase boundaries that hold it."""
    ranks = collected["ranks"]
    peaks: dict[int, list[int]] = {rank: [] for rank in ranks}
    segment_peak: dict[str, list[int]] = {}
    for (_step, rank), entry in collected["per_step_rank"].items():
        if entry["peak_allocated"]:
            peaks[rank].append(entry["peak_allocated"])
        previous = "step:start"
        for layer, pass_name, _occurrence, name, _time_ms, _alloc, peak in entry["marks"]:
            label = f"{'step' if layer < 0 else 'L*'}:{pass_name}:{name}"
            if peak:
                segment_peak.setdefault(f"{previous} -> {label}", []).append(peak)
            previous = label
    if not any(peaks.values()):
        out.append("MEMORY: no allocator readings (CPU run)")
        return {}
    summary = {
        "peak_allocated_gib": {
            str(rank): statistics.fmean(values) / GIB for rank, values in peaks.items() if values
        }
    }
    out.append("MEMORY (per rank, mean over steps)")
    out.append("  peak allocated, GiB: " + ", ".join(
        f"r{rank} {value:.2f}" for rank, value in sorted(summary["peak_allocated_gib"].items())
    ))
    spread = max(summary["peak_allocated_gib"].values()) - min(summary["peak_allocated_gib"].values())
    summary["peak_spread_gib"] = spread
    out.append(f"  spread between ranks: {spread:.2f} GiB")
    ranked = sorted(segment_peak.items(), key=lambda item: -max(item[1]))[:6]
    out.append("  segments holding the highest peaks:")
    for label, values in ranked:
        out.append(f"    {max(values) / GIB:7.2f} GiB  {label}")
    return summary


def report_offload(collected: dict, headers: dict, table: dict, out: list[str]) -> dict[str, Any]:
    """Report the bytes above the mean on the busiest rank, and the idle windows."""
    header = headers[min(headers)]
    hidden = header.get("hidden")
    intermediate = header.get("intermediate")
    element = header.get("expert_element_size") or header.get("element_size") or 2
    if not hidden or not intermediate:
        return {}
    per_pair = (hidden + 3 * intermediate) * element
    ranks = collected["ranks"]
    excess_pairs = []
    for (_step, _layer), per_rank in table.items():
        loads = [per_rank[rank]["recv"] for rank in ranks if rank in per_rank]
        if len(loads) < len(ranks) or not sum(loads):
            continue
        excess_pairs.append(max(loads) - statistics.fmean(loads))
    if not excess_pairs:
        return {}
    gaps = [gap for entry in collected["per_step_rank"].values() for gap in entry["gaps_ms"]]
    summary = {
        "bytes_per_pair": per_pair,
        "excess_mib_per_layer": statistics.fmean(excess_pairs) * per_pair / (1024 ** 2),
        "median_gap_ms": statistics.median(gaps) if gaps else 0.0,
    }
    out.append("OFFLOAD HEADROOM (what a transfer would have to move, and when)")
    out.append(
        f"  saved activation bytes per routed pair: {per_pair} B"
        f"  (hidden {hidden} + 3 x intermediate {intermediate}, {element} B each)"
    )
    out.append(
        f"  busiest rank above the mean: {summary['excess_mib_per_layer']:.1f} MiB per layer"
    )
    out.append(
        f"  median time between MoE blocks: {summary['median_gap_ms']:.2f} ms"
        f"  ({len(gaps)} windows)"
    )
    for bandwidth in (6.5, 26.0, 137.0):
        movable = bandwidth * 1e9 * summary["median_gap_ms"] * 1e-3 / (1024 ** 2)
        out.append(f"    at {bandwidth:>5.1f} GB/s a window moves {movable:8.1f} MiB")
    return summary


def write_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    """Write one CSV with the given column order."""
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(collected: dict, out_dir: str) -> None:
    """Write the per-layer, per-rank and per-expert tables."""
    os.makedirs(out_dir, exist_ok=True)
    rows = collected["rows"]
    columns = [key for key in rows[0] if key != "expert_counts"]
    write_csv(os.path.join(out_dir, "layer_step.csv"), rows, columns)

    rank_rows = [
        {"step": step, "rank": rank, **{key: value for key, value in entry.items()
                                        if key not in ("marks", "gaps_ms")}}
        for (step, rank), entry in sorted(collected["per_step_rank"].items())
    ]
    write_csv(
        os.path.join(out_dir, "rank_step.csv"),
        rank_rows,
        ["step", "rank", "wall_s", "recv_total", "peak_allocated", "peak_reserved"],
    )

    expert_rows = [
        {"step": row["step"], "layer": row["layer"], "rank": row["rank"],
         "expert": expert, "tokens": tokens}
        for row in rows if row["expert_counts"]
        for expert, tokens in enumerate(row["expert_counts"])
    ]
    if expert_rows:
        write_csv(
            os.path.join(out_dir, "expert_load.csv"),
            expert_rows,
            ["step", "layer", "rank", "expert", "tokens"],
        )


def write_trace(collected: dict, headers: dict, path: str, step: int | None) -> None:
    """Write a Chrome trace of one step: one process per rank, phases as slices."""
    events: list[dict[str, Any]] = []
    steps = sorted({key[0] for key in collected["per_step_rank"]})
    chosen = step if step is not None else steps[len(steps) // 2]
    for rank in collected["ranks"]:
        entry = collected["per_step_rank"].get((chosen, rank))
        if entry is None:
            continue
        events.append({"ph": "M", "pid": rank, "name": "process_name",
                       "args": {"name": f"rank {rank} ({headers[rank].get('host', '')})"}})
        stamps: dict[tuple[int, str, int, str], float] = {}
        for layer, pass_name, occurrence, name, time_ms, _alloc, _peak in entry["marks"]:
            stamps[(layer, pass_name, occurrence, name)] = time_ms
        for (layer, pass_name, occurrence, name), begin in stamps.items():
            if name != "start" or layer < 0:
                continue
            phases = BWD_PHASES if pass_name == "bwd" else FWD_PHASES
            for first, second, label in phases:
                start = stamps.get((layer, pass_name, occurrence, first))
                end = stamps.get((layer, pass_name, occurrence, second))
                if start is None or end is None:
                    continue
                events.append({
                    "ph": "X", "pid": rank, "tid": pass_name, "name": f"L{layer} {label}",
                    "ts": start * 1e3, "dur": max(end - start, 0.0) * 1e3,
                    "args": {"layer": layer, "occurrence": occurrence},
                })
        for layer, pass_name, _occurrence, name, time_ms, alloc, _peak in entry["marks"]:
            if alloc is not None:
                events.append({"ph": "C", "pid": rank, "name": "allocated_MiB",
                               "ts": time_ms * 1e3, "args": {"MiB": alloc / (1024 ** 2)}})
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, stream)


def main() -> int:
    """Read the records, print the report and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir", help="directory holding rank*.jsonl")
    parser.add_argument("--skip", type=int, default=2, help="warm-up steps to drop")
    parser.add_argument("--out-dir", default=None, help="where to write the CSVs")
    parser.add_argument("--trace", action="store_true", help="also write a Chrome trace")
    parser.add_argument("--trace-step", type=int, default=None, help="step to trace")
    args = parser.parse_args()

    headers, steps = load_records(args.record_dir, args.skip)
    collected = collect(headers, steps)
    if not collected["rows"]:
        raise SystemExit("no step records left after --skip")
    table = _by_step_layer(collected["rows"])

    header = headers[min(headers)]
    out = [
        f"records: {len(headers)} ranks, "
        f"{sum(len(records) for records in steps.values())} step records kept, "
        f"time source {header.get('time_source')}",
        f"model: {header.get('num_experts')} experts, top-{header.get('top_k')}, "
        f"hidden {header.get('hidden')}, intermediate {header.get('intermediate')}, "
        f"{len(header.get('layers', {}))} MoE blocks",
        "",
    ]
    summary = {"header": header}
    summary["routing"] = report_routing(table, collected["ranks"], out)
    out.append("")
    summary["persistence"] = report_persistence(collected, out)
    out.append("")
    summary["time"] = report_time(table, collected["ranks"], out)
    out.append("")
    summary["memory"] = report_memory(collected, out)
    out.append("")
    summary["offload"] = report_offload(collected, headers, table, out)
    print("\n".join(out))

    out_dir = args.out_dir or os.path.join(args.record_dir, "analysis")
    write_outputs(collected, out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if args.trace:
        write_trace(collected, headers, os.path.join(out_dir, "trace.json"), args.trace_step)
    print(f"\nwrote {out_dir}/{{layer_step,rank_step,expert_load}}.csv and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
