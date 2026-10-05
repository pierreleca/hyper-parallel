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


def marks(record: dict) -> list[tuple]:
    """Return a record's marks as 9-tuples; older records lack the reserved fields.

    Fields: layer, pass, occurrence, name, ms, allocated, allocated peak,
    reserved, reserved peak.
    """
    return [tuple(mark) + (None,) * (9 - len(mark)) for mark in record["marks"]]


def mark_spans(record: dict) -> dict[tuple[int, str, int], tuple[float, float]]:
    """Return {(layer, pass, occurrence): (first mark, last mark)} in ms.

    A checkpointed block stops recomputing as soon as its last saved tensor
    is back, so a pass is bounded by the marks it did reach, not by ``end``.
    """
    spans: dict[tuple[int, str, int], tuple[float, float]] = {}
    for layer, pass_name, occurrence, _name, time_ms, _alloc, _peak, *_reserved in marks(record):
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
    for layer, pass_name, occurrence, name, time_ms, _alloc, _peak, *_reserved in marks(record):
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


def layer_memory(record: dict) -> dict[int, tuple[int, int]]:
    """Return {layer: (retained, transient)} bytes of the forward pass of one step.

    ``retained`` is what the MoE block leaves allocated: its output, plus the
    tensors saved for backward when the block is not recomputed. ``transient``
    is the highest the allocator went above the block's starting point while
    the block ran.
    """
    starts: dict[int, int] = {}
    highest: dict[int, int] = {}
    result: dict[int, tuple[int, int]] = {}
    for layer, pass_name, occurrence, name, _time_ms, alloc, peak, *_reserved in marks(record):
        if layer < 0 or pass_name != "fwd" or occurrence != 0 or alloc is None:
            continue
        if name == "start":
            starts[layer] = alloc
            highest[layer] = alloc
        elif layer in starts:
            highest[layer] = max(highest[layer], peak or alloc, alloc)
            if name == "end":
                result[layer] = (alloc - starts[layer], highest[layer] - starts[layer])
    return result


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
            memory_by_layer = layer_memory(record)
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
                    "retained": memory_by_layer.get(layer, (None, None))[0],
                    "transient": memory_by_layer.get(layer, (None, None))[1],
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
    first, last = _thirds(lambdas)
    summary["lambda_first_third"], summary["lambda_last_third"] = first, last
    out.append(
        f"  drift: mean lambda {first:.3f} in the first third of the steps, {last:.3f} in the last"
    )
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
        for layer, pass_name, _occurrence, name, _time_ms, _alloc, peak, *_reserved in marks(entry):
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
    summary.update(_report_reserved(collected, out))
    return summary


def _thirds(values: list[float]) -> tuple[float, float]:
    """Return the means of the first and the last third of a series."""
    third = max(1, len(values) // 3)
    return statistics.fmean(values[:third]), statistics.fmean(values[-third:])


def _report_reserved(collected: dict, out: list[str]) -> dict[str, Any]:
    """Report the reserved peak and the reserve held above the allocated peak.

    Reserved minus allocated at a given moment says little: the cache keeps
    memory freed late in the backward. What fragmentation changes is how much
    more the cache must reserve than the step's live peak, so the report
    compares each step's reserved peak with its allocated peak, and how that
    gap moves over the run.
    """
    ranks = collected["ranks"]
    reserved_peak: dict[int, list[int]] = {rank: [] for rank in ranks}
    overhead: dict[int, list[int]] = {rank: [] for rank in ranks}
    for (_step, rank), entry in sorted(collected["per_step_rank"].items()):
        if entry.get("peak_reserved"):
            reserved_peak[rank].append(entry["peak_reserved"])
            if entry.get("peak_allocated"):
                overhead[rank].append(entry["peak_reserved"] - entry["peak_allocated"])
    if not any(reserved_peak.values()):
        return {}
    summary = {
        "peak_reserved_gib": {
            str(rank): statistics.fmean(values) / GIB for rank, values in reserved_peak.items() if values
        }
    }
    out.append("  peak reserved, GiB: " + ", ".join(
        f"r{rank} {value:.2f}" for rank, value in sorted(summary["peak_reserved_gib"].items())
    ))
    if any(overhead.values()):
        trend = {}
        for rank, series in overhead.items():
            if series:
                first, last = _thirds([gap / GIB for gap in series])
                trend[str(rank)] = {"first_third": first, "last_third": last}
        summary["reserve_above_peak_gib"] = trend
        out.append("  reserved peak - allocated peak, per step, GiB, first -> last third:")
        out.append("    " + ", ".join(
            f"r{rank} {values['first_third']:.2f} -> {values['last_third']:.2f}"
            for rank, values in sorted(trend.items())
        ))
        out.append("    a gap that grows over the steps is fragmentation building up")
    return summary


def load_swaps(swap_dir: str) -> dict[tuple[int, int], list[int]]:
    """Read the host-swap records: {(step, rank): [swapped layer indices]}."""
    swaps: dict[tuple[int, int], list[int]] = {}
    for path in sorted(glob.glob(os.path.join(swap_dir, "host_swap_rank*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("header") or not record.get("layers"):
                    continue
                swaps[(record["step"], record["rank"])] = [layer["index"] for layer in record["layers"]]
    return swaps


def report_reserved_growth(
        steps: dict[int, list[dict]],
        out: list[str],
        swaps: dict[tuple[int, int], list[int]] | None = None,
        threshold_mib: float = 32.0,
        listed: int = 12,
) -> dict[str, Any]:
    """Report where the allocator's reserve grows: which step, which segment, by how much.

    The reserve is a high-water mark: the cache keeps what it maps. With
    ``segment_peaks`` the recorder resets the peak counters at every mark, so
    each mark holds the reserved peak of the segment that ends there; the
    growth is how far that peak passes every earlier one. Every step counts,
    warm-up included, since the reserve settles there. With ``swaps``, the
    steps where the rank swapped are tagged with the swapped layers.
    """
    summary: dict[str, Any] = {}
    out.append(f"RESERVED GROWTH (where the allocator's reserve rises; all steps, events over {threshold_mib:.0f} MiB)")
    for rank in sorted(steps):
        high = None
        events = []
        first = None
        for record in steps[rank]:
            previous = "step:start"
            for layer, pass_name, _occurrence, name, _time_ms, _alloc, _peak, reserved, reserved_peak in marks(record):
                label = f"{'step' if layer < 0 else f'L{layer}'}:{pass_name}:{name}"
                value = max(reserved or 0, reserved_peak or 0)
                if value:
                    if high is None:
                        high = first = value
                    elif value > high:
                        events.append((record["step"], f"{previous} -> {label}", value - high, value))
                        high = value
                previous = label
        if high is None:
            continue
        total = high - first
        big = [event for event in events if event[2] >= threshold_mib * 2 ** 20]
        summary[str(rank)] = {
            "first_gib": first / GIB, "final_gib": high / GIB, "growth_gib": total / GIB,
            "events": [{"step": step, "segment": segment, "mib": growth / 2 ** 20}
                       for step, segment, growth, _value in events],
        }
        out.append(f"  r{rank}: {first / GIB:.2f} -> {high / GIB:.2f} GiB (+{total / GIB:.2f}), {len(events)} rises,"
                   f" {len(big)} over {threshold_mib:.0f} MiB:")
        for step, segment, growth, value in sorted(big, key=lambda item: -item[2])[:listed]:
            tag = ""
            if swaps is not None:
                swapped = swaps.get((step, rank))
                tag = f"  [swapped L{', L'.join(map(str, swapped))}]" if swapped else "  [no swap]"
            out.append(f"    step {step:3d}  +{growth / 2 ** 20:7.1f} MiB -> {value / GIB:6.2f} GiB  {segment}{tag}")
    return summary


def report_offload(
        collected: dict,
        headers: dict,
        table: dict,
        out: list[str],
        intermediate: int | None = None,
) -> dict[str, Any]:
    """Report the bytes above the mean on the busiest rank, and the idle windows.

    ``intermediate`` overrides the header's expert intermediate size, which
    earlier records read from the wrong axis of the expert weight.
    """
    header = headers[min(headers)]
    hidden = header.get("hidden")
    intermediate = intermediate or header.get("intermediate")
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


def report_layer_memory(table: dict, ranks: list[int], out: list[str]) -> dict[str, Any]:
    """Report each MoE block's own memory cost, per rank.

    With the block recomputed, ``retained`` is little more than its output;
    without, it is the activations the block saves, which scale with the
    tokens the rank received in that layer.
    """
    per_layer: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
    for (_step, layer), per_rank in table.items():
        for rank, row in per_rank.items():
            if row.get("retained") is not None:
                per_layer.setdefault(layer, {}).setdefault(rank, []).append(
                    (row["retained"], row["transient"], row["recv"])
                )
    if not per_layer:
        return {}
    mib = 1024 ** 2
    out.append("MEMORY PER MoE BLOCK (forward pass, MiB, mean over steps)")
    out.append("  layer  retained per rank" + " " * 26 + "max/mean  transient max  B/pair")
    retained_totals = {rank: 0.0 for rank in ranks}
    factors, bytes_per_pair = [], []
    for layer in sorted(per_layer):
        retained = {
            rank: statistics.fmean(entry[0] for entry in per_layer[layer].get(rank, [(0, 0, 0)]))
            for rank in ranks
        }
        transient = max(entry[1] for entries in per_layer[layer].values() for entry in entries)
        pairs = [entry for entries in per_layer[layer].values() for entry in entries if entry[2]]
        per_pair = statistics.median(entry[0] / entry[2] for entry in pairs) if pairs else 0.0
        for rank in ranks:
            retained_totals[rank] += retained[rank]
        factor = imbalance(list(retained.values()))
        factors.append(factor)
        bytes_per_pair.append(per_pair)
        out.append(
            f"  L{layer:<5}" + "".join(f"{retained[rank] / mib:9.1f}" for rank in ranks)
            + f"{factor:12.2f}{transient / mib:14.1f}{per_pair:9.0f}"
        )
    summary = {
        "retained_lambda_per_layer_mean": statistics.fmean(factors),
        "retained_lambda_summed": imbalance(list(retained_totals.values())),
        "retained_bytes_per_pair": statistics.median(bytes_per_pair),
    }
    out.append(
        f"  per-layer max/mean {summary['retained_lambda_per_layer_mean']:.2f}, "
        f"summed over layers {summary['retained_lambda_summed']:.2f}"
    )
    fit = _fit_receive_side(
        [entry for entries in per_layer.values() for rows in entries.values() for entry in rows]
    )
    if fit is not None:
        summary.update(fit)
        out.append(
            f"  fit retained = a x received pairs + b: a = {fit['bytes_per_received_pair']:.0f} B, "
            f"b = {fit['fixed_bytes'] / mib:.0f} MiB, R^2 = {fit['r_squared']:.2f}"
        )
        out.append(
            "  Only a follows the routing: the rest is sized by the rank's own tokens"
        )
        out.append(
            "  (the combined output, the send buffer), which every rank has the same of."
        )
    return summary


def _fit_receive_side(entries: list[tuple[int, int, int]]) -> dict[str, float] | None:
    """Fit retained bytes = a x received pairs + b by least squares.

    The block keeps two kinds of tensors: those sized by the tokens the rank
    received (expert inputs and intermediates), which follow the routing, and
    those sized by its own tokens (the combined output), which do not. The
    slope ``a`` is what a capacity limit acts on.
    """
    points = [(entry[2], entry[0]) for entry in entries if entry[2]]
    if len(points) < 3:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    var_x = sum((x - mean_x) ** 2 for x, _ in points)
    if var_x == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var_x
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for _, y in points)
    residual = sum((y - slope * x - intercept) ** 2 for x, y in points)
    return {
        "bytes_per_received_pair": slope,
        "fixed_bytes": intercept,
        "r_squared": 1.0 - residual / total if total else 0.0,
    }


def report_step_budget(table: dict, out: list[str]) -> dict[str, Any]:
    """Compare one budget for all MoE layers with one budget per layer.

    Per-layer loads swing more than a rank's total over the layers, so a
    budget on the total needs less reserve for the same safety. For every
    (step, rank) the total is the sum of what its blocks retained; the sweep
    reports how often a budget of f times the mean total is exceeded, and by
    how much, which is what would have to be evicted to host.
    """
    per_step: dict[tuple[int, int], list[int]] = {}
    per_layer_max: dict[int, int] = {}
    for (step, layer), per_rank in table.items():
        for rank, row in per_rank.items():
            if row.get("retained") is None:
                continue
            per_step.setdefault((step, rank), []).append(row["retained"])
            per_layer_max[layer] = max(per_layer_max.get(layer, 0), row["retained"])
    layers = len(per_layer_max)
    totals = [sum(values) for values in per_step.values() if len(values) == layers]
    if not totals or not layers:
        return {}
    mean_total = statistics.fmean(totals)
    summary: dict[str, Any] = {
        "mean_total_gib": mean_total / GIB,
        "max_total_gib": max(totals) / GIB,
        "per_layer_worst_sum_gib": sum(per_layer_max.values()) / GIB,
    }
    out.append("ONE BUDGET FOR ALL MoE LAYERS (per rank and step, MoE block memory summed over layers)")
    out.append(
        f"  total: mean {summary['mean_total_gib']:.2f} GiB, worst {summary['max_total_gib']:.2f} GiB"
        f" ({max(totals) / mean_total:.3f}x the mean)"
    )
    out.append(
        f"  reserve with no eviction: one budget {summary['max_total_gib']:.2f} GiB, per-layer budgets"
        f" {summary['per_layer_worst_sum_gib']:.2f} GiB (each layer at its own worst)"
    )
    out.append("  budget (x mean)   GiB    (step, rank) over   worst eviction GiB   host ms")
    rows = []
    for factor in (1.0, 1.02, 1.05, 1.1, 1.15):
        budget = factor * mean_total
        overs = [total - budget for total in totals if total > budget]
        row = {
            "factor": factor,
            "budget_gib": budget / GIB,
            "over_share": len(overs) / len(totals),
            "worst_eviction_gib": max(overs) / GIB if overs else 0.0,
        }
        row["host_ms"] = row["worst_eviction_gib"] * GIB / 26e9 * 1e3
        rows.append(row)
        out.append(
            f"  {factor:<16.2f}{row['budget_gib']:7.2f}{row['over_share']:17.0%}"
            f"{row['worst_eviction_gib']:18.2f}{row['host_ms']:12.1f}"
        )
    summary["sweep"] = rows
    return summary


def report_capacity(
        table: dict,
        ranks: list[int],
        per_pair: float,
        per_pair_source: str,
        out: list[str],
        fixed_bytes: float = 0.0,
) -> list[dict[str, float]]:
    """Sweep a per-rank capacity: memory reserved against tokens over capacity.

    A capacity factor C sizes every rank's MoE buffers for C times the mean
    load of a layer. Tokens above it are dropped, or with host overflow, keep
    their activations in host memory; the sweep reports both sides.
    """
    loads_all = [row["recv"] for per_rank in table.values() for row in per_rank.values()]
    if not loads_all:
        return []
    mean_load = statistics.fmean(loads_all)
    mib = 1024 ** 2
    host_bandwidth = 26e9  # A2, one NPU's host link used alone (only the hot rank spills)
    out.append(
        f"CAPACITY SWEEP (per rank and layer; {per_pair:.0f} B per received pair"
        f" + {fixed_bytes / mib:.0f} MiB fixed, {per_pair_source})"
    )
    out.append("  C     reserved MiB   layers over   tokens over   hot-rank spill MiB   host ms")
    out.append("                                                    mean      max          max")
    rows = []
    for factor in (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0):
        over, total, spilling, spills = 0.0, 0.0, 0, []
        for per_rank in table.values():
            loads = [per_rank[rank]["recv"] for rank in ranks if rank in per_rank]
            if len(loads) < len(ranks) or not sum(loads):
                continue
            capacity = factor * statistics.fmean(loads)
            excess = [max(0.0, load - capacity) for load in loads]
            over += sum(excess)
            total += sum(loads)
            spilling += any(excess)
            spills.append(max(excess) * per_pair)
        row = {
            "capacity_factor": factor,
            "reserved_mib": (fixed_bytes + factor * mean_load * per_pair) / mib,
            "layers_over_share": spilling / max(1, len(spills)),
            "tokens_over_share": over / total if total else 0.0,
            "hot_spill_mean_mib": statistics.fmean(spills) / mib if spills else 0.0,
            "hot_spill_max_mib": max(spills) / mib if spills else 0.0,
        }
        row["host_ms_max"] = row["hot_spill_max_mib"] * mib / host_bandwidth * 1e3
        rows.append(row)
        out.append(
            f"  {factor:<5.2f}{row['reserved_mib']:10.0f}{row['layers_over_share']:13.0%}"
            f"{row['tokens_over_share']:14.2%}{row['hot_spill_mean_mib']:11.1f}"
            f"{row['hot_spill_max_mib']:9.1f}{row['host_ms_max']:13.1f}"
        )
    return rows


def _routing_by_step(steps: dict[int, list[dict]]) -> dict[tuple[int, int, int], tuple]:
    """Return {(step, rank, layer): (per-rank counts, per-expert counts)} of the forward passes."""
    routing = {}
    for rank, records in steps.items():
        for record in records:
            for call in record["calls"]:
                if call["pass"] == "fwd" and "recv" in call:
                    routing[(record["step"], rank, call["layer"])] = (
                        tuple(call["recv"]), tuple(call.get("expert_counts") or ()),
                    )
    return routing


def compare_runs(first_dir: str, second_dir: str, out: list[str], skip: int = 2) -> dict[str, Any]:
    """Compare the routing of two runs step by step, then their step times.

    With full determinism, step i routes the same tokens to the same experts
    in every run; the first step where the counts differ is where the runs
    diverge. The step time is the slowest rank's wall time, over the steps
    both runs kept after ``skip`` warm-up steps.
    """
    _headers_a, steps_a = load_records(first_dir, 0)
    _headers_b, steps_b = load_records(second_dir, 0)
    routing_a, routing_b = _routing_by_step(steps_a), _routing_by_step(steps_b)
    common = sorted(set(routing_a) & set(routing_b))
    if not common:
        out.append("COMPARE: the runs share no (step, rank, layer) record")
        return {}
    differing = [key for key in common if routing_a[key] != routing_b[key]]
    steps = sorted({step for step, _rank, _layer in common})
    summary = {
        "steps_compared": len(steps),
        "records_compared": len(common),
        "records_differing": len(differing),
        "first_differing_step": min(step for step, _rank, _layer in differing) if differing else None,
    }
    out.append(f"COMPARE {first_dir}  vs  {second_dir}")
    out.append(
        f"  {len(common)} (step, rank, layer) routing records over steps {steps[0]}-{steps[-1]}: "
        f"{len(differing)} differ"
    )
    if differing:
        step, rank, layer = min(differing)
        counts_a, counts_b = routing_a[(step, rank, layer)][1], routing_b[(step, rank, layer)][1]
        moved = sum(abs(a - b) for a, b in zip(counts_a, counts_b)) // 2
        out.append(
            f"  first difference: step {step}, rank {rank}, layer {layer}"
            f" ({moved} token-expert assignments moved)"
        )
    else:
        out.append("  the runs route identically: step i gives the same result in both")
    _headers_a, kept_a = load_records(first_dir, skip)
    _headers_b, kept_b = load_records(second_dir, skip)
    wall_a, wall_b = _step_walls(kept_a), _step_walls(kept_b)
    both = sorted(set(wall_a) & set(wall_b))
    if both:
        this = [wall_a[step] for step in both]
        base = [wall_b[step] for step in both]
        for name, pick in (("mean", statistics.fmean), ("median", statistics.median)):
            before, after = pick(base), pick(this)
            summary[f"step_s_{name}"] = (before, after)
            out.append(
                f"  step time, slowest rank, {name} over steps {both[0]}-{both[-1]}: {before:.3f} -> {after:.3f} s"
                f" ({(after - before) * 1e3:+.0f} ms, {(after / before - 1) * 100:+.1f}%), second run -> first"
            )
    return summary


def _step_walls(steps: dict[int, list[dict]]) -> dict[int, float]:
    """Return {step: the slowest rank's wall time in seconds}."""
    walls: dict[int, float] = {}
    for records in steps.values():
        for record in records:
            if record.get("wall_s") is not None:
                walls[record["step"]] = max(walls.get(record["step"], 0.0), record["wall_s"])
    return walls


def report_swap_activity(swap_dir: str, skip: int, out: list[str]) -> dict[str, Any]:
    """Summarize the host-swap records per rank: what moved, how fast, and what compute waited for.

    Steps up to ``skip`` are warm-up and left out, as in the rest of the report.
    """
    per_rank: dict[int, list[dict]] = {}
    budget = None
    for path in sorted(glob.glob(os.path.join(swap_dir, "host_swap_rank*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("header"):
                    # Older records give a capacity factor: the budget in mean layers per MoE layer.
                    budget = record.get("budget_layers", record.get("capacity_factor", budget))
                elif record["step"] > skip:
                    per_rank.setdefault(record["rank"], []).append(record)
    if not per_rank:
        return {}
    out.append(f"SWAP ACTIVITY (budget {budget}, per rank, mean over steps after {skip})")
    out.append("  rank  steps swapping  layers/step  GiB/step  D2H GB/s  H2D GB/s  copy back hidden ms"
               "  exposed ms  evictions/step")
    rows = []
    for rank, records in sorted(per_rank.items()):
        swapping = [record for record in records if record["swapped_layers"]]
        rates_out = [record["d2h_gbps"] for record in swapping if record["d2h_gbps"]]
        rates_in = [record["h2d_gbps"] for record in swapping if record["h2d_gbps"]]
        row = {
            "rank": rank,
            "steps": len(records),
            "steps_swapping": len(swapping),
            "layers_per_step": statistics.fmean(record["swapped_layers"] for record in records),
            "gib_per_step": statistics.fmean(record["d2h_gib"] for record in records),
            "d2h_gbps": statistics.fmean(rates_out) if rates_out else None,
            "h2d_gbps": statistics.fmean(rates_in) if rates_in else None,
            "hidden_ms": statistics.fmean(record["h2d_hidden_ms"] for record in records),
            "exposed_ms": statistics.fmean(record["stall_ms"] for record in records),
            "evictions_per_step": statistics.fmean(len(record.get("evictions", [])) for record in records),
        }
        rows.append(row)
        out.append(
            f"  r{rank:<4}{len(swapping):>7}/{len(records):<8}{row['layers_per_step']:11.1f}"
            f"{row['gib_per_step']:10.2f}{_rate(row['d2h_gbps']):>10}{_rate(row['h2d_gbps']):>10}"
            f"{row['hidden_ms']:21.1f}{row['exposed_ms']:12.1f}{row['evictions_per_step']:16.1f}"
        )
    out.append(
        f"  all ranks: {statistics.fmean(row['gib_per_step'] for row in rows):.2f} GiB per rank and step,"
        f" copy back exposed {statistics.fmean(row['exposed_ms'] for row in rows):.1f} ms per step on average,"
        f" {max(row['exposed_ms'] for row in rows):.1f} ms at worst"
    )
    return {"budget": budget, "ranks": rows}


def sweep_row(run_dir: str, skip: int) -> dict[str, Any]:
    """Summarize one run: its memory peaks, its step time and what its swap moved.

    ``run_dir`` holds ``instrument/`` and, when the swap ran, ``ep_host_swap/``;
    a directory of ``rank*.jsonl`` is taken as the records themselves.
    """
    inner = os.path.join(run_dir, "instrument")
    record_dir = inner if os.path.isdir(inner) else run_dir
    headers, steps = load_records(record_dir, skip)
    collected = collect(headers, steps)
    allocated: dict[int, list[float]] = {}
    reserved: dict[int, list[float]] = {}
    walls: dict[int, float] = {}
    for (step, rank), entry in collected["per_step_rank"].items():
        if entry.get("peak_allocated"):
            allocated.setdefault(rank, []).append(entry["peak_allocated"] / GIB)
        if entry.get("peak_reserved"):
            reserved.setdefault(rank, []).append(entry["peak_reserved"] / GIB)
        if entry.get("wall_s") is not None:
            walls[step] = max(walls.get(step, 0.0), entry["wall_s"])
    alloc = [statistics.fmean(values) for values in allocated.values()]
    resv = [statistics.fmean(values) for values in reserved.values()]
    # Routing health: max / mean received pairs per (step, layer); about 16 when one rank per
    # EP group of 16 gets everything, which is what a router scoring every expert alike does.
    received: dict[tuple[int, int], list[int]] = {}
    for item in collected["rows"]:
        received.setdefault((item["step"], item["layer"]), []).append(item["recv"])
    ratios = [imbalance(values) for values in received.values() if sum(values)]
    row = {
        "run": os.path.basename(run_dir.rstrip("/")) or run_dir,
        "steps": len(walls),
        "alloc_mean": statistics.fmean(alloc) if alloc else 0.0,
        "alloc_worst": max(alloc) if alloc else 0.0,
        "reserved_mean": statistics.fmean(resv) if resv else 0.0,
        "reserved_worst": max(resv) if resv else 0.0,
        "step_mean": statistics.fmean(walls.values()) if walls else 0.0,
        "step_median": statistics.median(walls.values()) if walls else 0.0,
        "routing_max_over_mean": statistics.fmean(ratios) if ratios else 0.0,
    }
    row.update(swap_totals(os.path.join(run_dir, "ep_host_swap"), skip))
    return row


def swap_totals(swap_dir: str, skip: int) -> dict[str, Any]:
    """What a run's swap records add up to: bytes moved, copy rates, exposure, pinned memory."""
    empty = {"moved_gib": 0.0, "d2h_min": None, "d2h_max": None, "exposed_max": 0.0, "pinned_max": 0.0}
    per_rank: dict[int, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(swap_dir, "host_swap_rank*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if not record.get("header") and record["step"] > skip:
                    per_rank.setdefault(record["rank"], []).append(record)
    if not per_rank:
        return empty
    moved, rates, exposed, pinned = [], [], [], []
    for records in per_rank.values():
        moved.append(statistics.fmean(record["d2h_gib"] for record in records))
        seen = [record["d2h_gbps"] for record in records if record["d2h_gbps"]]
        if seen:
            rates.append(statistics.fmean(seen))
        exposed.append(statistics.fmean(record["stall_ms"] for record in records))
        pinned.append(max(record.get("pinned_gib", 0.0) for record in records))
    return {"moved_gib": statistics.fmean(moved), "d2h_min": min(rates) if rates else None,
            "d2h_max": max(rates) if rates else None, "exposed_max": max(exposed),
            "pinned_max": max(pinned)}


def report_sweep(run_dirs: list[str], skip: int, out: list[str]) -> list[dict[str, Any]]:
    """One row per run, so a factor sweep can be read at a glance."""
    rows = [sweep_row(run_dir, skip) for run_dir in run_dirs]
    out.append(f"SWEEP ({len(rows)} runs, per-rank means over the steps kept after {skip})")
    out.append("  run                              steps   moved  alloc  alloc  reserved  reserved   step s   step s"
               "   D2H GB/s   exposed  pinned  routing")
    out.append("                                           GiB/st   mean  worst      mean     worst     mean   median"
               "    min-max     ms max     GiB  max/mean")
    for row in rows:
        rate = "-" if row["d2h_min"] is None else f"{row['d2h_min']:.0f}-{row['d2h_max']:.0f}"
        out.append(
            f"  {row['run']:<32}{row['steps']:5d}{row['moved_gib']:8.2f}{row['alloc_mean']:7.2f}"
            f"{row['alloc_worst']:7.2f}{row['reserved_mean']:10.2f}{row['reserved_worst']:10.2f}"
            f"{row['step_mean']:9.3f}{row['step_median']:9.3f}{rate:>11}{row['exposed_max']:11.1f}"
            f"{row['pinned_max']:8.2f}{row['routing_max_over_mean']:10.2f}"
        )
    for row in rows:
        if row["routing_max_over_mean"] > 4:
            out.append(f"  WARNING {row['run']}: routing max/mean {row['routing_max_over_mean']:.1f}, the router looks"
                       " collapsed (every token to the same experts); its memory and time say nothing about the swap")
    if len(rows) < 2:
        return rows
    base = rows[0]
    out.append(f"  against {base['run']}: step time and memory as deltas")
    for row in rows[1:]:
        out.append(
            f"  {row['run']:<32}     {row['moved_gib']:8.2f} moved,"
            f" reserved mean {row['reserved_mean'] - base['reserved_mean']:+6.2f},"
            f" worst {row['reserved_worst'] - base['reserved_worst']:+6.2f} GiB,"
            f" step mean {(row['step_mean'] / base['step_mean'] - 1) * 100:+5.1f}%,"
            f" median {(row['step_median'] / base['step_median'] - 1) * 100:+5.1f}%"
        )
    return rows


def _rate(value: float | None) -> str:
    """A bandwidth for a table cell, or '-' when nothing was timed."""
    return "-" if value is None else f"{value:.1f}"


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
        for layer, pass_name, occurrence, name, time_ms, _alloc, _peak, *_reserved in marks(entry):
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
        for layer, pass_name, _occurrence, name, time_ms, alloc, _peak, *_reserved in marks(entry):
            if alloc is not None:
                events.append({"ph": "C", "pid": rank, "name": "allocated_MiB",
                               "ts": time_ms * 1e3, "args": {"MiB": alloc / (1024 ** 2)}})
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, stream)


def main() -> int:
    """Read the records, print the report and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir", nargs="?", help="directory holding rank*.jsonl")
    parser.add_argument("--skip", type=int, default=2, help="warm-up steps to drop")
    parser.add_argument(
        "--sweep", nargs="+", default=None, metavar="RUN_DIR",
        help="compare whole runs, one row each, and stop: every directory holding instrument/ and"
             " ep_host_swap/; the first is the baseline the deltas are taken against",
    )
    parser.add_argument("--out-dir", default=None, help="where to write the CSVs")
    parser.add_argument("--trace", action="store_true", help="also write a Chrome trace")
    parser.add_argument("--trace-step", type=int, default=None, help="step to trace")
    parser.add_argument(
        "--compare", default=None, metavar="OTHER_DIR",
        help="compare the routing of this run with another, step by step, and stop",
    )
    parser.add_argument(
        "--swap-dir", default=None,
        help="ep_host_swap output directory: tag the reserved-memory rises with the steps' swaps",
    )
    parser.add_argument(
        "--intermediate", type=int, default=None,
        help="expert intermediate size, overriding the recorded one (768 for Qwen3-VL-30B)",
    )
    args = parser.parse_args()

    if args.sweep:
        out = []
        report_sweep(args.sweep, args.skip, out)
        print("\n".join(out))
        return 0
    if not args.record_dir:
        raise SystemExit("record_dir is required unless --sweep is given")

    if args.compare:
        out = []
        compare_runs(args.record_dir, args.compare, out, args.skip)
        print("\n".join(out))
        return 0

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
        f"hidden {header.get('hidden')}, "
        f"intermediate {args.intermediate or header.get('intermediate')}, "
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
    _headers, all_steps = load_records(args.record_dir, 0)
    swaps = load_swaps(args.swap_dir) if args.swap_dir else None
    summary["reserved_growth"] = report_reserved_growth(all_steps, out, swaps)
    out.append("")
    if args.swap_dir:
        summary["swap_activity"] = report_swap_activity(args.swap_dir, args.skip, out)
        out.append("")
    summary["offload"] = report_offload(collected, headers, table, out, args.intermediate)
    out.append("")
    summary["layer_memory"] = report_layer_memory(table, collected["ranks"], out)
    out.append("")
    estimate = summary["offload"].get("bytes_per_pair", 0)
    layer_memory_summary = summary["layer_memory"]
    slope = layer_memory_summary.get("bytes_per_received_pair", 0)
    # A fitted slope close to the shape estimate means the block was not
    # recomputed and its saved activations are what the allocator saw; with
    # the block recomputed only its output remains, which the routing does
    # not size.
    if slope and slope >= 0.5 * estimate and layer_memory_summary.get("r_squared", 0) > 0.5:
        per_pair, fixed, source = slope, layer_memory_summary["fixed_bytes"], "fitted, MoE kept"
    else:
        per_pair, fixed, source = estimate, 0.0, "shape estimate, MoE recomputed"
    if per_pair:
        summary["capacity"] = report_capacity(
            table, collected["ranks"], per_pair, source, out, fixed_bytes=fixed
        )
    out.append("")
    summary["step_budget"] = report_step_budget(table, out)
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
