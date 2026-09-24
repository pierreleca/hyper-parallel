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
"""Summarize the device side of an Ascend profiler trace (trace_view.json).

Reports, for one profiled step or all of them:

- which threads the trace holds, and which device stream computes;
- how that stream spends the step: computing, waiting on another stream
  (EVENT WAIT tasks), or idle with nothing queued;
- the kernels and kernel categories that take the time;
- the idle time between tasks, counted by gap length;
- the stream waits, by the type of collective each one waited for, and the
  longest of them;
- the collectives (hcom events of the "Communication" process), by type, and
  the activation swap copies (MEMCPY_ASYNC on the swap streams): their time in
  flight and how much of it compute hides. Stream waits are attributed to the
  collective or swap copy that ends with them.
- optionally, the kernels around every match of a name, with the gaps between
  them (``--around aclnnGroupedMatmul``), to read what one phase launches.

    python examples/qwen3_vl_30b_perf/analyze_npu_trace.py <trace dir or trace_view.json>

With the traces of every rank (profiling.rank: -1), ``--ranks`` compares them:
per-rank step breakdown and routed-token work, and the k-th alltoallv matched
across ranks, split into the wait for the last rank to arrive and the transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Optional

# Run as a script, Python puts this directory first on the import path.
from ascend_trace import (
    SWAP_LABEL, SWAP_PATTERN, Trace, around, attribute_waits, base_name, busy_time, category, comm_exposure,
    comm_type, is_sync, split_sync,
    find_rank_traces, step_breakdown, summarize, window,
)

MS = 1e3


def report_inventory(trace: Trace, out: list[str]) -> None:
    """List the busiest threads, so the trace's layout is visible."""
    out.append(f"trace: {trace.path}")
    out.append(f"  {len(trace.events)} complete events, {len(trace.processes)} processes")
    out.append("  busiest threads (events, summed duration ms):")
    for label, count, total in trace.inventory(12):
        out.append(f"    {count:8d} {total / MS:12.1f}  {label}")


def report_streams(trace: Trace, key: tuple, swap_keys: list, out: list[str]) -> None:
    """The device streams: tasks, kernels, sync tasks, copies, busy time and main operators."""
    out.append("  device streams (* compute, S swap; events, aclnn kernels, sync tasks, MEMCPY_ASYNC and their share"
               " without / with sync tasks, busy ms without sync; top operators by time):")
    for row in trace.streams():
        marker = "*" if row["key"] == key else "S" if row["key"] in swap_keys else " "
        compute, _sync = split_sync(trace.thread_events(row["key"]))
        top = ", ".join(f"{item['name'][:28]} {item['total_us'] / MS:.0f}" for item in summarize(compute)[:3])
        out.append(f"   {marker}{row['events']:8d} {row['kernels']:8d} {row['sync']:8d} {row['copies']:8d}"
                   f" {row['copy_share']:5.0%} {row['copy_share_all']:5.0%} {row['busy_us'] / MS:10.1f}"
                   f"  {row['label']}: {top}")


def swap_setup(trace: Trace, key: tuple, args: argparse.Namespace) -> tuple[list, list]:
    """The swap streams (by the MEMCPY_ASYNC share, or --swap-stream) and their copies."""
    keys = trace.swap_threads(compute=key, min_share=args.swap_min_share, name=args.swap_stream)
    return keys, trace.swaps(keys)


def waitable_label(event: Any) -> str:
    """What a stream wait waited for: a collective type, or the activation swap."""
    return SWAP_LABEL if SWAP_PATTERN.match(event.name) else comm_type(event.name)


def report_step(label: str, tasks: list, span: tuple[float, float], out: list[str]) -> dict:
    """How the compute stream spends one window: compute, waits on other streams, idle, edges."""
    parts = step_breakdown(tasks, span[0], span[1])
    total = parts["span"] or 1.0

    def share(name: str) -> str:
        """One part in ms and as a share of the step."""
        return f"{parts[name] / MS:7.1f} ({parts[name] / total:4.0%})"

    out.append(
        f"  {label}: {parts['span'] / MS:8.1f} ms = compute {share('compute')} + stream wait {share('wait')}"
        f" + idle {share('idle')} + before first / after last task {share('edges')}; {parts['tasks']} tasks"
        + (f"; last task runs {parts['overrun'] / MS:.1f} ms past the step" if parts["overrun"] > 0 else "")
    )
    return parts


def report_mean(breakdowns: list[dict], out: list[str]) -> None:
    """The mean of the per-step breakdowns, and how far the steps spread."""
    names = ("span", "compute", "wait", "idle", "edges")
    mean = {name: sum(parts[name] for parts in breakdowns) / len(breakdowns) for name in names}
    spans = [parts["span"] for parts in breakdowns]
    out.append(
        f"  mean of {len(breakdowns)}: {mean['span'] / MS:6.1f} ms = compute {mean['compute'] / MS:7.1f}"
        f" + stream wait {mean['wait'] / MS:7.1f} + idle {mean['idle'] / MS:7.1f}"
        f" + before first / after last task {mean['edges'] / MS:7.1f}"
        f"; steps range {min(spans) / MS:.1f}-{max(spans) / MS:.1f} ms"
    )


def report_idle(parts: dict, out: list[str]) -> None:
    """The idle gaps between the stream's tasks, counted by length."""
    out.append(f"  idle between tasks: {parts['idle'] / MS:.1f} ms, by gap length (gaps, total ms, share of idle):")
    for row in parts["idle_buckets"]:
        out.append(f"    {row['count']:8d} {row['total_us'] / MS:10.2f} {row['total_us'] / (parts['idle'] or 1.0):6.1%}"
                   f"  {row['bucket']}")


def report_kernels(tasks: list, top: int, by_task: bool, out: list[str]) -> list[dict]:
    """The kernels and categories that take the stream's compute time."""
    compute, _sync = split_sync(tasks)
    busy = busy_time(compute) or 1.0
    rows = summarize(compute, key=(lambda name: name) if by_task else base_name)
    unit = "device tasks" if by_task else "operators (an aclnn operator may run several device tasks)"
    out.append(f"  top {top} {unit}:")
    out.append("     tasks   total ms    mean us  share")
    for row in rows[:top]:
        out.append(
            f"    {row['count']:6d} {row['total_us'] / MS:10.2f} {row['mean_us']:10.1f}"
            f" {row['total_us'] / busy:6.1%}  {row['name'][:100]}"
        )
    out.append("  by category (tasks, total ms, share of compute):")
    for row in summarize(compute, key=category):
        out.append(f"    {row['count']:6d} {row['total_us'] / MS:10.2f} {row['total_us'] / busy:6.1%}  {row['name']}")
    return rows


def report_waits(trace: Trace, tasks: list, start: float, count: int, swaps: list, out: list[str]) -> None:
    """Stream waits by what they waited for (a collective type or the swap), and the longest ones."""
    waits = [event for event in sorted(tasks) if is_sync(event) and event.dur > 0]
    causes = attribute_waits(waits, trace.communications() + swaps)
    by_type: dict[str, list[float]] = {}
    for wait, cause in zip(waits, causes):
        by_type.setdefault(waitable_label(cause) if cause else "(nothing ends there)", []).append(wait.dur)
    total = busy_time(waits)
    out.append(f"  stream waits: {len(waits)}, {total / MS:.1f} ms; by the collective or swap copy ending with the"
               " wait (waits, ms, share):")
    for name, durations in sorted(by_type.items(), key=lambda item: -sum(item[1])):
        share = sum(durations) / (total or 1.0)
        out.append(f"    {len(durations):6d} {sum(durations) / MS:10.2f} {share:6.1%}  {name}")
    if count <= 0:
        return
    ordered = sorted(tasks)
    position = {id(event): index for index, event in enumerate(ordered)}
    out.append(f"  longest {min(count, len(waits))} waits"
               " (us, at ms, collective or swap copy waited for and its duration us, next task):")
    for wait, cause in sorted(zip(waits, causes), key=lambda item: -item[0].dur)[:count]:
        following = next((item.name for item in ordered[position[id(wait)] + 1:] if not is_sync(item)), "-")
        waited = f"{waitable_label(cause)} {cause.dur:.0f}" if cause else "-"
        out.append(f"    {wait.dur:9.1f} @ {(wait.ts - start) / MS:8.2f}  {waited:24s}  then {following[:50]}")


def exposure_rows(comms: list, swaps: list, parts: dict) -> list[dict]:
    """Collectives by type, all of them, and the swap copies: time in flight, hidden, exposed."""
    rows = comm_exposure(comms, parts["compute_union"], parts["sync_union"]) if comms else []
    if swaps:
        rows += comm_exposure(swaps, parts["compute_union"], parts["sync_union"], key=lambda name: SWAP_LABEL,
                              total_label=None)
    return rows


def report_collectives(trace: Trace, start: float, end: float, parts: dict, swaps: list,
                       out: list[str]) -> list[dict]:
    """Collectives by type and swap copies in the window: their time, and how much of it compute hides."""
    comms = window(trace.communications(), start, end)
    rows = exposure_rows(comms, window(swaps, start, end), parts)
    if not rows:
        out.append("  collectives: no hcom event in the Communication process, and no swap stream")
        return []
    out.append("  collectives by type, and swap copies: time in flight (union of their intervals), hidden under")
    out.append("  compute on the compute stream, exposed (during a stream wait / otherwise); ms:")
    out.append("     count   in flight     hidden    exposed    in wait  hidden%")
    for row in rows:
        out.append(
            f"    {row['count']:6d} {row['total_us'] / MS:11.2f} {row['hidden_us'] / MS:10.2f}"
            f" {row['exposed_us'] / MS:10.2f} {row['in_wait_us'] / MS:10.2f}"
            f" {row['hidden_us'] / (row['total_us'] or 1.0):7.0%}  {row['name']}"
        )
    return rows


def report_around(kernels: list, args: argparse.Namespace, start: float, out: list[str]) -> None:
    """Kernels around the first matches of a name, with the gaps between them."""
    contexts = around(kernels, args.around, args.before, args.after, args.limit, args.skip)
    out.append(f"  around '{args.around}': {len(contexts)} match(es) shown")
    for number, rows in enumerate(contexts):
        out.append(f"   match {number}:")
        for offset, gap, event in rows:
            marker = ">>" if offset == 0 else "  "
            out.append(
                f"    {marker}{offset:+4d} gap {gap:8.1f} us  dur {event.dur:9.1f} us"
                f"  @ {(event.ts - start) / MS:8.2f} ms  {event.name[:90]}"
            )


def write_kernels(tasks: list, start: float, path: str) -> None:
    """Write the window's device tasks with their gaps, for a spreadsheet."""
    ordered = sorted(tasks)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_ms", "dur_us", "gap_before_us", "category", "name"])
        previous_end = None
        for event in ordered:
            gap = event.ts - previous_end if previous_end is not None else 0.0
            label = "stream wait" if is_sync(event) else category(event.name)
            writer.writerow([f"{(event.ts - start) / MS:.4f}", f"{event.dur:.2f}", f"{max(gap, 0.0):.2f}",
                             label, event.name])
            previous_end = event.end if previous_end is None else max(previous_end, event.end)


def _rank_view(trace: Trace, step: Optional[int], args: argparse.Namespace) -> dict:
    """One rank's compute stream over one profiled step."""
    key = trace.compute_thread()
    _swap_keys, swaps = swap_setup(trace, key, args)
    tasks = trace.thread_events(key)
    steps = {number: (start, end) for number, start, end in trace.steps()}
    if not steps:
        raise ValueError(f"{trace.path}: no ProfilerStep range")
    number = step if step in steps else max(steps)
    start, end = steps[number]
    selected = window(tasks, start, end)
    parts = step_breakdown(selected, start, end)
    compute = split_sync(selected)[0]
    categories = {row["name"]: row["total_us"] for row in summarize(compute, key=category)}
    comms = window(trace.communications(), start, end)
    return {"step": number, "start": start, "end": end, "parts": parts, "categories": categories,
            "operators": summarize(compute),
            "comms": comms, "exposure": {row["name"]: row for row in exposure_rows(
                comms, window(swaps, start, end), parts)}}


def report_rank_table(views: dict[int, dict], out: list[str]) -> None:
    """Per rank: how the compute stream spends the step, and the work that scales with routed tokens."""
    out.append("PER RANK (ms): step = compute + stream wait + idle + edges; routed-token work; collectives; swap")
    out.append("  rank  step    span  compute     wait     idle    edges  grouped mm  sort/index"
               "  a2av flight  a2av exposed  all-comm exposed  swap flight  swap exposed")
    for rank, view in views.items():
        parts, cats, exposure = view["parts"], view["categories"], view["exposure"]
        empty = {"total_us": 0.0, "exposed_us": 0.0}
        a2av = exposure.get("alltoallv", empty)
        swap = exposure.get(SWAP_LABEL, empty)
        out.append(
            f"  {rank:4d} {view['step']:5d} {parts['span'] / MS:7.1f} {parts['compute'] / MS:8.1f}"
            f" {parts['wait'] / MS:8.1f} {parts['idle'] / MS:8.1f} {parts['edges'] / MS:8.1f}"
            f" {cats.get('grouped matmul', 0.0) / MS:11.1f} {cats.get('sort / index', 0.0) / MS:11.1f}"
            f" {a2av['total_us'] / MS:12.1f} {a2av['exposed_us'] / MS:13.1f}"
            f" {exposure.get('all collectives', empty)['exposed_us'] / MS:17.1f}"
            f" {swap['total_us'] / MS:12.1f} {swap['exposed_us'] / MS:13.1f}"
        )


def report_rank_categories(views: dict[int, dict], top: int, out: list[str]) -> None:
    """Compute per category on every rank, the categories that differ most first."""
    ranks = list(views)
    names = sorted({name for view in views.values() for name in view["categories"]})
    rows = []
    for name in names:
        values = [views[rank]["categories"].get(name, 0.0) for rank in ranks]
        rows.append((max(values) - min(values), name, values))
    out.append("COMPUTE BY CATEGORY (ms), widest spread across ranks first:")
    out.append("    spread  " + "  ".join(f"  rank {rank}" for rank in ranks) + "  category")
    for spread, name, values in sorted(rows, reverse=True)[:top]:
        out.append(f"  {spread / MS:8.1f}  " + "  ".join(f"{value / MS:8.1f}" for value in values) + f"  {name}")


def report_rank_operators(views: dict[int, dict], top: int, out: list[str]) -> None:
    """Compute per operator on every rank, the operators that differ most first."""
    ranks = list(views)
    totals = {rank: {row["name"]: row["total_us"] for row in views[rank]["operators"]} for rank in ranks}
    names = {name for table in totals.values() for name in table}
    rows = []
    for name in names:
        values = [totals[rank].get(name, 0.0) for rank in ranks]
        rows.append((max(values) - min(values), name, values))
    out.append("COMPUTE BY OPERATOR (ms), widest spread across ranks first:")
    out.append("    spread  " + "  ".join(f"  rank {rank}" for rank in ranks) + "  operator")
    for spread, name, values in sorted(rows, reverse=True)[:top]:
        out.append(f"  {spread / MS:8.1f}  " + "  ".join(f"{value / MS:8.1f}" for value in values) + f"  {name}")


def report_rank_exposure(views: dict[int, dict], out: list[str]) -> None:
    """Exposed collective time per type on every rank (in flight in brackets)."""
    ranks = list(views)
    names = sorted({name for view in views.values() for name in view["exposure"]},
                   key=lambda name: (name == "all collectives", name))
    out.append("EXPOSED COLLECTIVES AND SWAP (ms): exposed / in flight, per rank")
    out.append("  " + "".join(f"{f'rank {rank}':>20s}" for rank in ranks) + "  type")
    for name in names:
        cells = []
        for rank in ranks:
            row = views[rank]["exposure"].get(name)
            cells.append(f"{row['exposed_us'] / MS:9.1f} / {row['total_us'] / MS:8.1f}" if row else f"{'-':>20s}")
        out.append("  " + "".join(f"{cell:>20s}" for cell in cells) + f"  {name}")


def report_matched(views: dict[int, dict], kind: str, list_rows: bool, out: list[str]) -> dict:
    """Match the k-th collective of one type across ranks: arrival skew and transfer.

    A collective ends on every rank once the last rank has joined and the data
    has moved, so on each rank its duration is the wait for the last rank
    (last start - own start) plus the transfer (end - last start). The ranks
    run on one host and share its clock.
    """
    lists = {rank: sorted(event for event in view["comms"] if comm_type(event.name) == kind)
             for rank, view in views.items()}
    counts = {rank: len(events) for rank, events in lists.items()}
    count = min(counts.values()) if counts else 0
    out.append(f"MATCHED {kind}: {count} per rank" + ("" if len(set(counts.values())) <= 1 else
                                                      f" (counts differ: {counts}; matched by order up to {count})"))
    if count == 0:
        return {}
    origin = min(view["start"] for view in views.values())
    ranks = list(lists)
    waited = {rank: 0.0 for rank in ranks}
    own = {rank: 0.0 for rank in ranks}
    last_count = {rank: 0 for rank in ranks}
    total_skew = total_transfer = 0.0
    rows = []
    for index in range(count):
        events = {rank: lists[rank][index] for rank in ranks}
        last_start = max(event.ts for event in events.values())
        last_rank = max(ranks, key=lambda rank, events=events: events[rank].ts)
        transfer = max(max(event.end for event in events.values()) - last_start, 0.0)
        skew = last_start - min(event.ts for event in events.values())
        total_skew += skew
        total_transfer += transfer
        last_count[last_rank] += 1
        for rank, event in events.items():
            waited[rank] += last_start - event.ts
            own[rank] += event.dur
        rows.append((index, (min(event.ts for event in events.values()) - origin) / MS, skew, transfer, last_rank,
                     [events[rank].dur for rank in ranks]))
    if list_rows:
        out.append("     k      at ms   skew us  transfer us  last  " + "  ".join(f"r{rank} dur us" for rank in ranks))
        for index, at, skew, transfer, last_rank, durations in rows:
            out.append(f"  {index:4d} {at:10.2f} {skew:9.0f} {transfer:12.0f}  {last_rank:4d}  "
                       + "  ".join(f"{duration:9.0f}" for duration in durations))
    out.append(f"  sum over the {count}: widest skew {total_skew / MS:.1f} ms, transfer after the last arrival"
               f" {total_transfer / MS:.1f} ms")
    out.append("  per rank (ms): time in the collective = waiting for the last rank + transfer; times last")
    for rank in ranks:
        out.append(f"    rank {rank}: {own[rank] / MS:8.1f} = {waited[rank] / MS:8.1f} waiting"
                   f" + {(own[rank] - waited[rank]) / MS:8.1f}; last {last_count[rank]} times")
    return {"count": count, "skew_ms": total_skew / MS, "transfer_ms": total_transfer / MS,
            "waited_ms": {rank: value / MS for rank, value in waited.items()},
            "own_ms": {rank: value / MS for rank, value in own.items()}, "last": last_count}


def report_ranks(args: argparse.Namespace) -> int:
    """Compare the traces of every rank under one trace_dir."""
    files = find_rank_traces(args.path)
    if len(files) < 2:
        raise SystemExit(f"--ranks needs traces of several ranks under {args.path}; found {files}")
    out: list[str] = [f"traces: {len(files)} ranks"]
    views = {}
    for rank, path in files.items():
        out.append(f"  rank {rank}: {path}")
        views[rank] = _rank_view(Trace.load(path), args.step, args)
    if len({view["step"] for view in views.values()}) > 1:
        out.append(f"  warning: ranks show different steps: { {r: v['step'] for r, v in views.items()} }")
    out.append("")
    report_rank_table(views, out)
    out.append("")
    report_rank_categories(views, 12, out)
    out.append("")
    report_rank_operators(views, args.top, out)
    out.append("")
    report_rank_exposure(views, out)
    summary = {"ranks": {rank: {"step": view["step"], **_scalars(view["parts"])} for rank, view in views.items()}}
    for kind in args.match:
        out.append("")
        summary[kind] = report_matched(views, kind, args.list_matched, out)
    print("\n".join(out))
    out_dir = args.out_dir or os.path.join(args.path if os.path.isdir(args.path) else os.path.dirname(args.path),
                                           "analysis_ranks")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ranks.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"\nwrote {out_dir}/ranks.json")
    return 0


def _scalars(parts: dict) -> dict:
    """The JSON-friendly part of a step breakdown, in ms."""
    return {name: parts[name] / MS for name in ("span", "compute", "wait", "idle", "edges", "overrun")}


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="trace_view.json, or a directory holding one")
    parser.add_argument("--step", type=int, default=None, help="profiled step to detail (default: the last)")
    parser.add_argument("--top", type=int, default=25, help="kernels to list")
    parser.add_argument("--by-task", action="store_true",
                        help="list device tasks by full name instead of grouping them by aclnn operator")
    parser.add_argument("--waits", type=int, default=10, help="longest stream waits to list")
    parser.add_argument("--around", default=None, help="regex of task names to show in context")
    parser.add_argument("--before", type=int, default=25, help="tasks shown before each match")
    parser.add_argument("--after", type=int, default=5, help="tasks shown after each match")
    parser.add_argument("--limit", type=int, default=3, help="matches shown")
    parser.add_argument("--skip", type=int, default=0, help="matches skipped before the first one shown")
    parser.add_argument("--out-dir", default=None, help="where to write kernels.csv and summary.json")
    parser.add_argument("--rank", type=int, default=0,
                        help="the rank to detail when the path holds the traces of several ranks")
    parser.add_argument("--swap-stream", default=None,
                        help="device stream(s) carrying the swap copies, by label substring (e.g. 'Stream 49'),"
                             " instead of the MEMCPY_ASYNC share rule")
    parser.add_argument("--swap-min-share", type=float, default=0.5,
                        help="a stream is a swap stream when MEMCPY_ASYNC tasks exceed this share of its"
                             " non-sync tasks (and number at least 4)")
    parser.add_argument("--ranks", action="store_true",
                        help="compare the traces of every rank under the path (profiling.rank: -1)")
    parser.add_argument("--match", nargs="+", default=["alltoallv", "alltoall"],
                        help="collective types matched across ranks in --ranks mode")
    parser.add_argument("--list-matched", action="store_true", help="list every matched collective")
    return parser.parse_args()


def main() -> int:
    """Load the trace and print the report."""
    args = parse_args()
    if args.ranks:
        return report_ranks(args)
    ranked = find_rank_traces(args.path)
    trace = Trace.load(ranked.get(args.rank, args.path) if ranked else args.path)
    out: list[str] = []
    report_inventory(trace, out)
    key = trace.compute_thread()
    tasks = trace.thread_events(key)
    swap_keys, swaps = swap_setup(trace, key, args)
    report_streams(trace, key, swap_keys, out)
    out.append(f"compute stream: {trace.thread_label(key)}, {len(tasks)} tasks")
    out.append("swap streams: " + (", ".join(trace.thread_label(item) for item in swap_keys) or "none")
               + f", {len(swaps)} MEMCPY_ASYNC")
    out.append("")

    steps = trace.steps()
    summary = {"compute_stream": trace.thread_label(key), "steps": []}
    out.append("STEPS (ProfilerStep numbers)")
    breakdowns = {}
    if steps:
        for step, start, end in steps:
            breakdowns[step] = report_step(f"step {step}", window(tasks, start, end), (start, end), out)
        if len(steps) > 1:
            report_mean(list(breakdowns.values()), out)
        chosen = next((item for item in steps if item[0] == args.step), steps[-1])
        label, start, end = f"step {chosen[0]}", chosen[1], chosen[2]
        parts = breakdowns[chosen[0]]
    else:
        start, end = tasks[0].ts, tasks[-1].end
        label = "whole trace (no ProfilerStep range found)"
        parts = breakdowns[0] = report_step(label, tasks, (start, end), out)
    summary["steps"] = [{"step": step, **_scalars(value)} for step, value in breakdowns.items()]
    selected = window(tasks, start, end)
    out.append("")
    out.append(f"DETAIL: {label}")
    summary["kernels"] = report_kernels(selected, args.top, args.by_task, out)
    report_idle(parts, out)
    report_waits(trace, selected, start, args.waits, swaps, out)
    summary["collectives"] = report_collectives(trace, start, end, parts, swaps, out)
    if args.around:
        report_around(selected, args, start, out)
    print("\n".join(out))

    out_dir = args.out_dir or os.path.join(os.path.dirname(trace.path), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    write_kernels(selected, start, os.path.join(out_dir, "kernels.csv"))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"\nwrote {out_dir}/kernels.csv and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
