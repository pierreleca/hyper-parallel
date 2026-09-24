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
"""Read the Chrome trace that the Ascend profiler writes (trace_view.json).

torch_npu's profiler writes one directory per run,
``<trace_dir>/<host>_<pid>_<time>_ascend_pt/ASCEND_PROFILER_OUTPUT/``, whose
``trace_view.json`` is a Chrome trace: complete events (``ph: "X"``) with
microsecond ``ts`` and ``dur``, and metadata events naming every process and
thread. The device kernels sit in the process named "Ascend Hardware", one
thread per stream; the compute stream is the thread with the most kernels whose
name starts with ``aclnn``.

The helpers here stay generic: load the trace, name its processes and threads,
pick the compute stream, cut it into profiler steps, and measure kernels, the
idle time between them, the stream waits and the collectives they wait for. ``analyze_npu_trace.py`` builds its
report from them.
"""

from __future__ import annotations

import bisect
import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

COMPUTE_PROCESS = "Ascend Hardware"
KERNEL_PREFIX = "aclnn"
STEP_PATTERN = re.compile(r"ProfilerStep#(\d+)")
# Stream synchronization tasks on the device: a stream that waits for an event
# recorded by another stream (the communication stream, usually) shows an
# EVENT WAIT that lasts until the other stream gets there. They are not compute.
# CANN versions differ in the separator: "EVENT WAIT" in some, "EVENT_WAIT" in others.
SYNC_PATTERN = re.compile(r"^(EVENT|NOTIFY)[ _](WAIT|RECORD)", re.IGNORECASE)
COMM_PROCESS = "Communication"
COMM_PREFIX = "hcom"
# Activation swap copies are MEMCPY_ASYNC tasks on a stream of the device
# process; one swap may be split into several. Other streams run a few copies
# for other reasons, so a stream counts as a swap stream only when copies make
# up more than SWAP_MIN_SHARE of its tasks, synchronization tasks left out.
SWAP_PATTERN = re.compile(r"^MEMCPY_ASYNC", re.IGNORECASE)
SWAP_MIN_SHARE = 0.5
SWAP_MIN_COUNT = 4
SWAP_LABEL = "swap"
# hcom_alltoallv__909_502_1 -> alltoallv: the numbers after the type change with every instance.
COMM_TYPE_PATTERN = re.compile(r"^hcom_?([A-Za-z]+)")

# First match wins; the names are the aclnn/ATen operator names seen in traces.
CATEGORIES = (
    ("grouped matmul", re.compile(r"GroupedMatmul|GroupMatmul|Gmm", re.IGNORECASE)),
    ("attention", re.compile(r"FlashAttention|FusionAttention|FlashAttn|Attention", re.IGNORECASE)),
    ("matmul", re.compile(r"Matmul|MatMul|Gemm|BatchMatMul|Mm\b|Addmm", re.IGNORECASE)),
    ("moe routing", re.compile(r"MoeTokenPermute|MoeTokenUnpermute|MoeInitRouting|MoeGating|TopK|Topk",
                               re.IGNORECASE)),
    ("sort / index", re.compile(r"Sort|Argsort|Index|Gather|Scatter|Bincount|Histc|Cumsum|RepeatInterleave|"
                                r"Nonzero|Unique|Masked", re.IGNORECASE)),
    ("norm / activation", re.compile(r"RmsNorm|LayerNorm|Norm|Swiglu|SwiGlu|Silu|Gelu|Softmax|Rope|Rotary",
                                     re.IGNORECASE)),
    ("copy / cast", re.compile(r"Copy|Cast|Memcpy|MEMCPY|Contiguous|Transpose|Permute|Concat|Cat\b|Slice|"
                               r"Split|Fill|Zero|Ones|Empty", re.IGNORECASE)),
    ("elementwise", re.compile(r"Add|Mul|Sub|Div|Exp|Log|Pow|Sqrt|Neg|Abs|Maximum|Minimum|Where|Equal|"
                               r"Greater|Less|Clamp", re.IGNORECASE)),
    ("optimizer", re.compile(r"Adam|Lamb|Foreach", re.IGNORECASE)),
)


@dataclass(order=True)
class Event:
    """One complete event: a kernel, a host operator or a range."""

    ts: float
    dur: float
    name: str = field(compare=False)
    pid: Any = field(compare=False, default=None)
    tid: Any = field(compare=False, default=None)
    args: dict = field(compare=False, default_factory=dict)

    @property
    def end(self) -> float:
        """Return the end time, in microseconds."""
        return self.ts + self.dur


RANK_DIR_PATTERN = re.compile(r"(?:^|[/\\])rank(\d+)_[^/\\]*_ascend_pt(?:[/\\]|$)")
PROFILER_INFO_PATTERN = re.compile(r"profiler_info_(\d+)\.json$")
IDLE_BUCKETS = ((10.0, "< 10 us"), (100.0, "10-100 us"), (1e3, "0.1-1 ms"), (1e4, "1-10 ms"),
                (float("inf"), ">= 10 ms"))


def find_trace_files(path: str) -> list[str]:
    """Return the trace_view.json files under ``path``, or ``path`` itself."""
    if os.path.isfile(path):
        return [path]
    files = glob.glob(os.path.join(path, "**", "trace_view.json"), recursive=True)
    return sorted(files, key=os.path.getmtime)


def trace_rank(trace_file: str) -> Optional[int]:
    """Return the rank a trace belongs to, from its run directory, or None.

    The trainer names run directories rank<N>_<time>_ascend_pt; torch_npu also
    writes profiler_info_<N>.json next to ASCEND_PROFILER_OUTPUT when the
    process group is up.
    """
    match = RANK_DIR_PATTERN.search(trace_file)
    if match:
        return int(match.group(1))
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(trace_file)))
    for name in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        match = PROFILER_INFO_PATTERN.match(name)
        if match:
            return int(match.group(1))
    return None


def find_rank_traces(path: str) -> dict[int, str]:
    """Return {rank: newest trace_view.json of that rank} under ``path``."""
    found: dict[int, str] = {}
    for trace_file in find_trace_files(path):  # oldest first: newer runs overwrite
        rank = trace_rank(trace_file)
        if rank is not None:
            found[rank] = trace_file
    return dict(sorted(found.items()))


def _number(value: Any) -> Optional[float]:
    """Parse a numeric field that the Ascend exporter may write as a string."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Trace:
    """A loaded Chrome trace: named processes and threads, and complete events."""

    def __init__(self, raw: list[dict]) -> None:
        """Index the raw event list."""
        self.path = ""
        self.processes: dict[Any, str] = {}
        self.threads: dict[tuple[Any, Any], str] = {}
        self.events: list[Event] = []
        for item in raw:
            phase = item.get("ph")
            if phase == "M":
                name = (item.get("args") or {}).get("name")
                if item.get("name") == "process_name":
                    self.processes[item.get("pid")] = str(name)
                elif item.get("name") == "thread_name":
                    self.threads[(item.get("pid"), item.get("tid"))] = str(name)
            elif phase == "X":
                ts, dur = _number(item.get("ts")), _number(item.get("dur"))
                if ts is None or dur is None:
                    continue
                self.events.append(Event(ts, dur, str(item.get("name", "")), item.get("pid"), item.get("tid"),
                                         item.get("args") or {}))
        self.events.sort()

    @classmethod
    def load(cls, path: str) -> "Trace":
        """Load a trace_view.json, or the newest one under a directory."""
        files = find_trace_files(path)
        if not files:
            raise FileNotFoundError(f"no trace_view.json under {path}")
        with open(files[-1], encoding="utf-8") as stream:
            data = json.load(stream)
        raw = data.get("traceEvents", []) if isinstance(data, dict) else data
        trace = cls(raw)
        trace.path = files[-1]
        return trace

    # -- inventory -----------------------------------------------------------

    def by_thread(self) -> dict[tuple[Any, Any], list[Event]]:
        """Group the complete events by (pid, tid), in time order."""
        groups: dict[tuple[Any, Any], list[Event]] = defaultdict(list)
        for event in self.events:
            groups[(event.pid, event.tid)].append(event)
        return groups

    def thread_label(self, key: tuple[Any, Any]) -> str:
        """Return 'process / thread' for a (pid, tid) pair."""
        pid, tid = key
        return f"{self.processes.get(pid, pid)} / {self.threads.get(key, tid)}"

    def inventory(self, limit: int = 20) -> list[tuple[str, int, float]]:
        """Return the busiest threads: label, event count, summed duration (us)."""
        rows = [
            (self.thread_label(key), len(events), sum(event.dur for event in events))
            for key, events in self.by_thread().items()
        ]
        return sorted(rows, key=lambda row: -row[1])[:limit]

    def find_processes(self, pattern: str) -> list[Any]:
        """Return the pids whose process name contains ``pattern``."""
        return [pid for pid, name in self.processes.items() if pattern.lower() in name.lower()]

    # -- device kernels ------------------------------------------------------

    def compute_thread(self, process: str = COMPUTE_PROCESS, prefix: str = KERNEL_PREFIX) -> tuple[Any, Any]:
        """Return the (pid, tid) of the stream with the most ``prefix`` kernels."""
        pids = set(self.find_processes(process))
        counts = Counter(
            (event.pid, event.tid)
            for event in self.events
            if event.pid in pids and event.name.startswith(prefix)
        )
        if not counts:
            raise ValueError(f"no '{prefix}' events in a process named like '{process}'")
        return counts.most_common(1)[0][0]

    def streams(self, process: str = COMPUTE_PROCESS, prefix: str = KERNEL_PREFIX) -> list[dict[str, Any]]:
        """Return, per device stream: label, events, ``prefix`` kernels, sync tasks, busy us."""
        pids = set(self.find_processes(process))
        rows = []
        for key, events in self.by_thread().items():
            if key[0] not in pids:
                continue
            compute, sync = split_sync(events)
            copies = sum(1 for event in compute if SWAP_PATTERN.match(event.name))
            rows.append({"key": key, "label": self.thread_label(key), "events": len(events),
                         "kernels": sum(1 for event in events if event.name.startswith(prefix)),
                         "sync": len(sync), "busy_us": busy_time(compute), "copies": copies,
                         "copy_share": copies / len(compute) if compute else 0.0,
                         "copy_share_all": copies / len(events) if events else 0.0})
        return sorted(rows, key=lambda row: -row["kernels"])

    def swap_threads(self, compute: Optional[tuple[Any, Any]] = None, min_share: float = SWAP_MIN_SHARE,
                     min_count: int = SWAP_MIN_COUNT, name: Optional[str] = None) -> list[tuple[Any, Any]]:
        """Return the device streams that carry the activation swap copies.

        Args:
            compute: The compute stream, never a swap stream.
            min_share: MEMCPY_ASYNC tasks must be more than this share of the
                stream's tasks, synchronization tasks left out.
            min_count: And at least this many.
            name: When given, the streams whose label contains it, instead of the rule.
        """
        rows = [row for row in self.streams() if row["key"] != compute]
        if name is not None:
            return [row["key"] for row in rows if name in row["label"]]
        return [row["key"] for row in rows if row["copies"] >= min_count and row["copy_share"] > min_share]

    def swaps(self, threads: list[tuple[Any, Any]]) -> list[Event]:
        """Return the MEMCPY_ASYNC tasks of the swap streams."""
        keys = set(threads)
        return [event for event in self.events
                if (event.pid, event.tid) in keys and SWAP_PATTERN.match(event.name)]

    def communications(self, process: str = COMM_PROCESS, prefix: str = COMM_PREFIX) -> list[Event]:
        """Return the device collectives: ``prefix`` events of the communication process."""
        pids = set(self.find_processes(process))
        return [event for event in self.events if event.pid in pids and event.name.startswith(prefix)]

    def thread_events(self, key: tuple[Any, Any]) -> list[Event]:
        """Return the events of one thread, in time order."""
        return [event for event in self.events if (event.pid, event.tid) == key]

    def steps(self, pattern: re.Pattern = STEP_PATTERN) -> list[tuple[int, float, float]]:
        """Return (step, start, end) of the profiler step ranges, any thread."""
        found = {}
        for event in self.events:
            match = pattern.search(event.name)
            if match:
                step = int(match.group(1))
                start, end = found.get(step, (event.ts, event.end))
                found[step] = (min(start, event.ts), max(end, event.end))
        return [(step, start, end) for step, (start, end) in sorted(found.items())]


# -- measurements over event lists --------------------------------------------

def window(events: Iterable[Event], start: float, end: float) -> list[Event]:
    """Return the events that start inside [start, end)."""
    return [event for event in events if start <= event.ts < end]


def busy_time(events: list[Event]) -> float:
    """Return the length of the union of the events' intervals, in us."""
    total, current_start, current_end = 0.0, None, None
    for event in sorted(events):
        if current_end is None or event.ts > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = event.ts, event.end
        else:
            current_end = max(current_end, event.end)
    if current_end is not None:
        total += current_end - current_start
    return total


def merge(events: Iterable[Event]) -> list[tuple[float, float]]:
    """Return the union of the events' intervals, as sorted disjoint (start, end) pairs."""
    merged: list[list[float]] = []
    for event in sorted(events):
        if merged and event.ts <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], event.end)
        else:
            merged.append([event.ts, event.end])
    return [(interval[0], interval[1]) for interval in merged]


def length(intervals: list[tuple[float, float]]) -> float:
    """Return the total length of disjoint intervals."""
    return sum(end - start for start, end in intervals)


def intersection(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> float:
    """Return the length of the intersection of two sorted disjoint interval lists."""
    total, i, j = 0.0, 0, 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            total += end - start
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return total


def step_breakdown(tasks: list[Event], start: float, end: float) -> dict[str, Any]:
    """Split a window of one stream into compute, stream waits, idle and edges, in us.

    ``tasks`` are the stream's tasks that start in [start, end). Compute is
    the union of the non-synchronization tasks; wait is the part of the
    EVENT WAIT tasks that no compute task covers; idle is the time between
    the first task's start and the last task's end that no task covers; the
    edges are the rest of the window, before the first task and after the
    last one. The four add up to the window length, less any tail of the last
    task past ``end``. ``idle_buckets`` counts the idle gaps by length.
    """
    compute, sync = split_sync(tasks)
    compute_union, all_union = merge(compute), merge(tasks)
    sync_union = merge(sync)
    wait = length(sync_union) - intersection(sync_union, compute_union)
    idle_gaps = [following[0] - previous[1] for previous, following in zip(all_union, all_union[1:])]
    first = all_union[0][0] if all_union else start
    last = all_union[-1][1] if all_union else start
    buckets = []
    lower = 0.0
    for upper, label in IDLE_BUCKETS:
        chosen = [gap for gap in idle_gaps if lower <= gap < upper]
        buckets.append({"bucket": label, "count": len(chosen), "total_us": sum(chosen)})
        lower = upper
    return {"span": end - start, "compute": length(compute_union), "wait": wait, "idle": sum(idle_gaps),
            "edges": (first - start) + max(end - last, 0.0), "overrun": max(last - end, 0.0),
            "tasks": len(compute), "idle_buckets": buckets, "compute_union": compute_union,
            "sync_union": sync_union}


def comm_exposure(comms: list[Event], compute_union: list[tuple[float, float]],
                  sync_union: list[tuple[float, float]],
                  key: Optional[Callable[[str], str]] = None,
                  total_label: Optional[str] = "all collectives") -> list[dict[str, Any]]:
    """Return, per collective type and for all of them, how much of their time compute hides.

    total: union of the collectives' intervals; hidden: the part during which
    the compute stream computes; exposed: the rest, split into the part the
    stream spends in an EVENT WAIT and the part it sits idle or before/after
    its tasks. Times in us. ``total_label`` names the row over all of them,
    left out when None.
    """
    key = key or comm_type
    groups: dict[str, list[Event]] = defaultdict(list)
    for event in comms:
        groups[key(event.name)].append(event)
    rows = []
    totals = [(total_label, comms)] if total_label else []
    for name, events in sorted(groups.items()) + totals:
        union = merge(events)
        total = length(union)
        hidden = intersection(union, compute_union)
        in_wait = min(intersection(union, sync_union), total - hidden)
        rows.append({"name": name, "count": len(events), "total_us": total, "hidden_us": hidden,
                     "exposed_us": total - hidden, "in_wait_us": in_wait})
    return rows


def gaps(events: list[Event]) -> list[tuple[float, Event, Event]]:
    """Return (idle us, kernel before, kernel after) between consecutive events."""
    result, last = [], None
    for event in sorted(events):
        if last is not None and event.ts > last.end:
            result.append((event.ts - last.end, last, event))
        if last is None or event.end > last.end:
            last = event
    return result


def is_sync(event: Event) -> bool:
    """Return whether an event is a stream synchronization task, not compute."""
    return bool(SYNC_PATTERN.match(event.name))


def split_sync(events: list[Event]) -> tuple[list[Event], list[Event]]:
    """Split a stream's events into (compute tasks, synchronization tasks)."""
    compute, sync = [], []
    for event in events:
        (sync if is_sync(event) else compute).append(event)
    return compute, sync


def overlapping(events: Iterable[Event], start: float, end: float) -> list[Event]:
    """Return the events whose interval intersects [start, end)."""
    return [event for event in events if event.ts < end and event.end > start]


def comm_name(name: str) -> str:
    """Return a collective's name without the numbers that make every instance unique."""
    return re.sub(r"[_#:]*\d[\w.]*", "", name).strip("_ ") or name


def comm_type(name: str) -> str:
    """Return the type of an hcom collective: alltoallv, allGather, reduceScatter..."""
    match = COMM_TYPE_PATTERN.match(name)
    return match.group(1) if match else comm_name(name)


def attribute_waits(waits: list[Event], comms: list[Event], tolerance: float = 50.0) -> list[Optional[Event]]:
    """Return, for each stream wait, the collective it most likely waited for.

    A stream wait ends when the stream it waits on records the event, which
    the communication stream does right after its collective: the candidate
    is the collective whose end is closest to the wait's end, among those
    ending inside the wait or at most ``tolerance`` us after it. None when no
    collective ends there (the wait is on another compute stream, or on an
    event recorded before the wait started).
    """
    ends = sorted(comms, key=lambda event: event.end)
    keys = [event.end for event in ends]
    result = []
    for wait in waits:
        low = bisect.bisect_left(keys, wait.ts)
        high = bisect.bisect_right(keys, wait.end + tolerance)
        candidates = ends[low:high]
        result.append(min(candidates, key=lambda event: abs(event.end - wait.end)) if candidates else None)
    return result


def base_name(name: str) -> str:
    """Return an operator name without the aclnn launcher's repeated suffixes."""
    return name.split("_", 1)[0] if name.startswith(KERNEL_PREFIX) else name


def category(name: str) -> str:
    """Return the coarse category of a kernel name."""
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def summarize(events: list[Event], key: Callable[[str], str] = base_name) -> list[dict[str, Any]]:
    """Aggregate events by ``key``: count, total, mean and max duration in us."""
    groups: dict[str, list[float]] = defaultdict(list)
    for event in events:
        groups[key(event.name)].append(event.dur)
    rows = [
        {"name": name, "count": len(durations), "total_us": sum(durations),
         "mean_us": sum(durations) / len(durations), "max_us": max(durations)}
        for name, durations in groups.items()
    ]
    return sorted(rows, key=lambda row: -row["total_us"])


def around(events: list[Event], pattern: str, before: int, after: int, limit: int,
           skip: int = 0) -> list[list[tuple]]:
    """Return the tasks around the matches of ``pattern``, skipping the first ``skip``.

    Each context is a list of (offset, gap before in us, event), offset 0
    being the match; the gap shows where the stream sat idle. A match that
    falls inside the previous context is not shown again.
    """
    regex = re.compile(pattern)
    ordered = sorted(events)
    contexts, seen, last_shown = [], 0, -1
    for index, event in enumerate(ordered):
        if not regex.search(event.name):
            continue
        seen += 1
        if seen <= skip or index <= last_shown:
            continue
        rows = []
        last_shown = min(len(ordered), index + after + 1) - 1
        for position in range(max(0, index - before), last_shown + 1):
            previous = ordered[position - 1] if position > 0 else None
            gap = ordered[position].ts - previous.end if previous is not None else 0.0
            rows.append((position - index, max(gap, 0.0), ordered[position]))
        contexts.append(rows)
        if len(contexts) >= limit:
            break
    return contexts


__all__ = [
    "CATEGORIES", "SWAP_LABEL", "SWAP_MIN_COUNT", "SWAP_MIN_SHARE", "SWAP_PATTERN",
    "COMM_PREFIX", "COMM_PROCESS", "COMPUTE_PROCESS", "Event", "IDLE_BUCKETS", "KERNEL_PREFIX",
    "STEP_PATTERN", "SYNC_PATTERN", "Trace", "around", "attribute_waits", "base_name", "busy_time", "category",
    "comm_exposure", "comm_name", "comm_type", "find_rank_traces", "find_trace_files", "gaps", "intersection",
    "is_sync", "length", "merge", "overlapping", "split_sync", "step_breakdown", "summarize", "trace_rank",
    "window",
]
