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

- which threads the trace holds, and which one is the compute stream;
- how busy that stream is, and the idle time between its kernels;
- the kernels and kernel categories that take the time;
- the longest idle gaps, with the kernels on either side: a stream that waits
  for the host (a ``.tolist()``, an ``.item()``) shows up here;
- the collectives, by thread;
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
from ascend_trace import Trace, around, busy_time, category, gaps, summarize, window

MS = 1e3


def report_inventory(trace: Trace, out: list[str]) -> None:
    """List the busiest threads, so the trace's layout is visible."""
    out.append(f"trace: {trace.path}")
    out.append(f"  {len(trace.events)} complete events, {len(trace.processes)} processes")
    out.append("  busiest threads (events, summed duration ms):")
    for label, count, total in trace.inventory(12):
        out.append(f"    {count:8d} {total / MS:12.1f}  {label}")


def report_step(label: str, kernels: list, span: tuple[float, float], out: list[str]) -> dict:
    """Busy and idle time of the compute stream over one window."""
    busy = busy_time(kernels)
    length = span[1] - span[0]
    idle = [gap for gap, _before, _after in gaps(kernels)]
    out.append(
        f"  {label}: {length / MS:9.1f} ms, compute stream busy {busy / MS:9.1f} ms"
        f" ({busy / length:.0%}), {len(kernels)} kernels, idle gaps {sum(idle) / MS:.1f} ms"
        f" ({sum(1 for gap in idle if gap > 100)} over 100 us)"
    )
    return {"label": label, "span_ms": length / MS, "busy_ms": busy / MS, "kernels": len(kernels),
            "idle_ms": sum(idle) / MS}


def report_kernels(kernels: list, top: int, out: list[str]) -> list[dict]:
    """The kernels and categories that take the stream's time."""
    busy = busy_time(kernels) or 1.0
    rows = summarize(kernels)
    out.append(f"  top {top} kernels (count, total ms, mean us, share of busy time):")
    for row in rows[:top]:
        out.append(
            f"    {row['count']:6d} {row['total_us'] / MS:10.2f} {row['mean_us']:10.1f}"
            f" {row['total_us'] / busy:6.1%}  {row['name']}"
        )
    categories = summarize(kernels, key=category)
    out.append("  by category (count, total ms, share):")
    for row in categories:
        out.append(f"    {row['count']:6d} {row['total_us'] / MS:10.2f} {row['total_us'] / busy:6.1%}  {row['name']}")
    return rows


def report_gaps(kernels: list, start: float, count: int, out: list[str]) -> None:
    """The longest idle gaps and the kernels on either side."""
    ranked = sorted(gaps(kernels), key=lambda item: -item[0])[:count]
    out.append(f"  longest {count} idle gaps (us, at ms into the window, kernel before -> kernel after):")
    for gap, before, after in ranked:
        out.append(f"    {gap:9.1f} @ {(before.end - start) / MS:8.2f}  {before.name[:60]}  ->  {after.name[:60]}")


def report_collectives(trace: Trace, start: float, end: float, out: list[str]) -> None:
    """Collectives in the window, by thread and name."""
    rows: dict[tuple[str, str], list[float]] = {}
    for event in window(trace.collectives(), start, end):
        key = (trace.thread_label((event.pid, event.tid)), event.name.split("_")[0])
        rows.setdefault(key, []).append(event.dur)
    if not rows:
        out.append("  collectives: none matched")
        return
    out.append("  collectives (count, total ms, thread, name):")
    for (label, name), durations in sorted(rows.items(), key=lambda item: -sum(item[1]))[:15]:
        out.append(f"    {len(durations):6d} {sum(durations) / MS:10.2f}  {label}  {name}")


def report_around(kernels: list, args: argparse.Namespace, start: float, out: list[str]) -> None:
    """Kernels around the first matches of a name, with the gaps between them."""
    contexts = around(kernels, args.around, args.before, args.after, args.limit)
    out.append(f"  around '{args.around}': {len(contexts)} match(es) shown")
    for number, rows in enumerate(contexts):
        out.append(f"   match {number}:")
        for offset, gap, event in rows:
            marker = ">>" if offset == 0 else "  "
            out.append(
                f"    {marker}{offset:+4d} gap {gap:8.1f} us  dur {event.dur:9.1f} us"
                f"  @ {(event.ts - start) / MS:8.2f} ms  {event.name[:90]}"
            )


def write_kernels(kernels: list, start: float, path: str) -> None:
    """Write the window's kernels with their gaps, for a spreadsheet."""
    ordered = sorted(kernels)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_ms", "dur_us", "gap_before_us", "category", "name"])
        previous_end = None
        for event in ordered:
            gap = event.ts - previous_end if previous_end is not None else 0.0
            writer.writerow([f"{(event.ts - start) / MS:.4f}", f"{event.dur:.2f}", f"{max(gap, 0.0):.2f}",
                             category(event.name), event.name])
            previous_end = event.end if previous_end is None else max(previous_end, event.end)


def main() -> int:
    """Load the trace and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="trace_view.json, or a directory holding one")
    parser.add_argument("--step", type=int, default=None, help="profiled step to detail (default: the last)")
    parser.add_argument("--top", type=int, default=25, help="kernels to list")
    parser.add_argument("--gaps", type=int, default=15, help="idle gaps to list")
    parser.add_argument("--around", default=None, help="regex of kernel names to show in context")
    parser.add_argument("--before", type=int, default=25, help="kernels shown before each match")
    parser.add_argument("--after", type=int, default=5, help="kernels shown after each match")
    parser.add_argument("--limit", type=int, default=3, help="matches shown")
    parser.add_argument("--out-dir", default=None, help="where to write kernels.csv and summary.json")
    args = parser.parse_args()

    trace = Trace.load(args.path)
    out: list[str] = []
    report_inventory(trace, out)
    key = trace.compute_thread()
    kernels = trace.thread_events(key)
    out.append(f"compute stream: {trace.thread_label(key)}, {len(kernels)} kernels")
    out.append("")

    steps = trace.steps()
    summary = {"compute_stream": trace.thread_label(key), "steps": []}
    out.append("STEPS")
    if steps:
        for step, start, end in steps:
            summary["steps"].append(report_step(f"step {step}", window(kernels, start, end), (start, end), out))
        chosen = next((item for item in steps if item[0] == args.step), steps[-1])
        label, start, end = f"step {chosen[0]}", chosen[1], chosen[2]
    else:
        start, end = kernels[0].ts, kernels[-1].end
        label = "whole trace (no ProfilerStep range found)"
        summary["steps"].append(report_step(label, kernels, (start, end), out))
    selected = window(kernels, start, end)
    out.append("")
    out.append(f"DETAIL: {label}")
    summary["kernels"] = report_kernels(selected, args.top, out)
    report_gaps(selected, start, args.gaps, out)
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
