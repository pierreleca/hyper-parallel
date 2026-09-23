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
idle gaps between them and the collectives. ``analyze_npu_trace.py`` builds its
report from them.
"""

from __future__ import annotations

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
COMM_PATTERN = re.compile(r"alltoall|all_to_all|allreduce|all_reduce|allgather|all_gather|reduce_?scatter|"
                          r"broadcast|hcom|hccl|send|recv", re.IGNORECASE)

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


def find_trace_files(path: str) -> list[str]:
    """Return the trace_view.json files under ``path``, or ``path`` itself."""
    if os.path.isfile(path):
        return [path]
    files = glob.glob(os.path.join(path, "**", "trace_view.json"), recursive=True)
    return sorted(files, key=os.path.getmtime)


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

    def collectives(self) -> list[Event]:
        """Return every complete event whose name looks like a collective."""
        return [event for event in self.events if COMM_PATTERN.search(event.name)]


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


def gaps(events: list[Event]) -> list[tuple[float, Event, Event]]:
    """Return (idle us, kernel before, kernel after) between consecutive events."""
    result, last = [], None
    for event in sorted(events):
        if last is not None and event.ts > last.end:
            result.append((event.ts - last.end, last, event))
        if last is None or event.end > last.end:
            last = event
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


def around(events: list[Event], pattern: str, before: int, after: int, limit: int) -> list[list[tuple]]:
    """Return the kernels around the first ``limit`` matches of ``pattern``.

    Each context is a list of (offset, gap before in us, event), offset 0
    being the match; the gap shows where the stream waited, for the host or
    for another stream.
    """
    regex = re.compile(pattern)
    ordered = sorted(events)
    contexts = []
    for index, event in enumerate(ordered):
        if not regex.search(event.name):
            continue
        rows = []
        for position in range(max(0, index - before), min(len(ordered), index + after + 1)):
            previous = ordered[position - 1] if position > 0 else None
            gap = ordered[position].ts - previous.end if previous is not None else 0.0
            rows.append((position - index, max(gap, 0.0), ordered[position]))
        contexts.append(rows)
        if len(contexts) >= limit:
            break
    return contexts


__all__ = [
    "CATEGORIES", "COMM_PATTERN", "COMPUTE_PROCESS", "Event", "KERNEL_PREFIX", "STEP_PATTERN", "Trace",
    "around", "base_name", "busy_time", "category", "find_trace_files", "gaps", "summarize", "window",
]
