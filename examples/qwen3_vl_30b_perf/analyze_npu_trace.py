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
- the stream waits, by the type of collective each one waited for, and the
  longest of them;
- the collectives (hcom events of the "Communication" process), by type;
- optionally, the kernels around every match of a name, with the gaps between
  them (``--around aclnnGroupedMatmul``), to read what one phase launches.

    python examples/qwen3_vl_30b_perf/analyze_npu_trace.py <trace dir or trace_view.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import os

# Run as a script, Python puts this directory first on the import path.
from ascend_trace import (
    Trace, around, attribute_waits, base_name, busy_time, category, comm_type, gaps, is_sync, split_sync,
    summarize, window,
)

MS = 1e3


def report_inventory(trace: Trace, out: list[str]) -> None:
    """List the busiest threads, so the trace's layout is visible."""
    out.append(f"trace: {trace.path}")
    out.append(f"  {len(trace.events)} complete events, {len(trace.processes)} processes")
    out.append("  busiest threads (events, summed duration ms):")
    for label, count, total in trace.inventory(12):
        out.append(f"    {count:8d} {total / MS:12.1f}  {label}")


def report_streams(trace: Trace, key: tuple, out: list[str]) -> None:
    """The device streams: tasks, aclnn kernels, sync tasks, busy time and main operators."""
    out.append("  device streams (events, aclnn kernels, sync tasks, busy ms without sync; top operators by time):")
    for row in trace.streams():
        marker = "*" if row["key"] == key else " "
        compute, _sync = split_sync(trace.thread_events(row["key"]))
        top = ", ".join(f"{item['name'][:28]} {item['total_us'] / MS:.0f}" for item in summarize(compute)[:3])
        out.append(f"   {marker}{row['events']:8d} {row['kernels']:8d} {row['sync']:8d} {row['busy_us'] / MS:10.1f}"
                   f"  {row['label']}: {top}")


def report_step(label: str, tasks: list, span: tuple[float, float], out: list[str]) -> dict:
    """How the compute stream spends one window: compute, waits on other streams, idle."""
    compute, sync = split_sync(tasks)
    length = span[1] - span[0]
    busy = busy_time(compute)
    waiting = busy_time(sync)
    idle = [gap for gap, _before, _after in gaps(tasks)]
    out.append(
        f"  {label}: {length / MS:8.1f} ms = compute {busy / MS:7.1f} ({busy / length:4.0%})"
        f" + wait on other streams {waiting / MS:6.1f} ({waiting / length:4.0%})"
        f" + idle {sum(idle) / MS:6.1f} ({sum(idle) / length:4.0%});"
        f" {len(compute)} tasks"
    )
    return {"label": label, "span_ms": length / MS, "compute_ms": busy / MS, "wait_ms": waiting / MS,
            "idle_ms": sum(idle) / MS, "tasks": len(compute)}


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


def report_waits(trace: Trace, tasks: list, start: float, end: float, count: int, out: list[str]) -> None:
    """Stream waits by the collective type they waited for, and the longest ones."""
    waits = [event for event in sorted(tasks) if is_sync(event) and event.dur > 0]
    causes = attribute_waits(waits, trace.communications())
    by_type: dict[str, list[float]] = {}
    for wait, cause in zip(waits, causes):
        by_type.setdefault(comm_type(cause.name) if cause else "(no collective ends there)", []).append(wait.dur)
    total = busy_time(waits)
    out.append(f"  stream waits: {len(waits)}, {total / MS:.1f} ms; by the collective ending with the wait"
               " (waits, ms, share):")
    for name, durations in sorted(by_type.items(), key=lambda item: -sum(item[1])):
        out.append(f"    {len(durations):6d} {sum(durations) / MS:10.2f} {sum(durations) / (total or 1.0):6.1%}  {name}")
    if count <= 0:
        return
    ordered = sorted(tasks)
    position = {id(event): index for index, event in enumerate(ordered)}
    out.append(f"  longest {min(count, len(waits))} waits (us, at ms, collective waited for and its duration us, next task):")
    for wait, cause in sorted(zip(waits, causes), key=lambda item: -item[0].dur)[:count]:
        following = next((item.name for item in ordered[position[id(wait)] + 1:] if not is_sync(item)), "-")
        waited = f"{comm_type(cause.name)} {cause.dur:.0f}" if cause else "-"
        out.append(f"    {wait.dur:9.1f} @ {(wait.ts - start) / MS:8.2f}  {waited:24s}  then {following[:50]}")


def report_collectives(trace: Trace, start: float, end: float, out: list[str]) -> None:
    """Device collectives in the window, by type."""
    comms = window(trace.communications(), start, end)
    if not comms:
        out.append("  collectives: no hcom event in the Communication process")
        return
    out.append("  collectives by type (count, total ms, mean us, max us):")
    for row in summarize(comms, key=comm_type):
        out.append(f"    {row['count']:6d} {row['total_us'] / MS:10.2f} {row['mean_us']:10.1f} {row['max_us']:10.1f}"
                   f"  {row['name']}")


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
    return parser.parse_args()


def main() -> int:
    """Load the trace and print the report."""
    args = parse_args()
    trace = Trace.load(args.path)
    out: list[str] = []
    report_inventory(trace, out)
    key = trace.compute_thread()
    tasks = trace.thread_events(key)
    report_streams(trace, key, out)
    out.append(f"compute stream: {trace.thread_label(key)}, {len(tasks)} tasks")
    out.append("")

    steps = trace.steps()
    summary = {"compute_stream": trace.thread_label(key), "steps": []}
    out.append("STEPS (ProfilerStep numbers)")
    if steps:
        for step, start, end in steps:
            summary["steps"].append(report_step(f"step {step}", window(tasks, start, end), (start, end), out))
        chosen = next((item for item in steps if item[0] == args.step), steps[-1])
        label, start, end = f"step {chosen[0]}", chosen[1], chosen[2]
    else:
        start, end = tasks[0].ts, tasks[-1].end
        label = "whole trace (no ProfilerStep range found)"
        summary["steps"].append(report_step(label, tasks, (start, end), out))
    selected = window(tasks, start, end)
    out.append("")
    out.append(f"DETAIL: {label}")
    summary["kernels"] = report_kernels(selected, args.top, args.by_task, out)
    report_waits(trace, selected, start, end, args.waits, out)
    report_collectives(trace, start, end, out)
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
