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
"""The components of a step, read from the Ascend profiler's trace alone.

``hetero_profile`` says which module ran when. The profiler says what the device did. This module uses only the
profiler's ``trace_view.json`` of one rank and groups what it holds into the components the study talks about, by what
the events are called, with no hook record involved:

- the kernels of the compute stream by class: attention (FlashAttention*), expert GEMMs (GroupedMatmul*), routing, sort
  and index kernels (the MoE preamble), dense matmul, norm and activation, and the rest;
- the compute stream's state: computing, waiting for a collective (of which kind), or idle with nothing queued;
- the collectives by kind: the MoE token exchange (alltoallv), the equal-split all-to-all, all-gather, reduce-scatter...

It numbers them per profiled step (``step_numbers``), lists what the trace holds so that a name that falls in the wrong
class is easy to spot (``inventory``), and draws them as a process of the same trace (``profile_lanes`` and
``process_events``) on the profiler's own timeline. ``write_integrated`` writes the original events untouched followed
by the new ones, so the file opens in Perfetto, chrome://tracing or MindStudio Insight as the original trace with the
detected components above it.

The classes come from the kernel names. When a real trace names something otherwise, ``--class attention=REGEX`` of
``component_trace.py`` (``parse_rules``) adds a rule that wins over the defaults, and the inventory shows which kernels
fell in "other".
"""

from __future__ import annotations

import bisect
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

# Run as a script, Python puts this directory first on the import path.
from ascend_trace import (
    SYNC_PATTERN, Event, Trace, attribute_waits, base_name, category, comm_exposure, comm_type, find_trace_files,
    is_sync, step_breakdown, window,
)

# The classes of kernels, as (key, title); the first is drawn first.
CLASSES = (
    ("attention", "attention kernels"),
    ("experts", "expert GEMM kernels (grouped matmul)"),
    ("routing", "routing, sort, index kernels"),
    ("dense", "dense matmul kernels"),
    ("norm", "norm and activation kernels"),
    ("other", "copy, cast, elementwise and other kernels"),
)
CLASS_KEYS = tuple(key for key, _ in CLASSES)
SHORT_NAMES = {"attention": "attention", "experts": "expert GEMM", "routing": "routing, sort, index",
               "dense": "dense matmul", "norm": "norm, activation", "other": "copy, cast, elementwise, other"}
CLASS_OF_CATEGORY = {
    "attention": "attention", "grouped matmul": "experts", "moe routing": "routing", "sort / index": "routing",
    "matmul": "dense", "norm / activation": "norm",
}
# Names the shared categories of ascend_trace miss: separators between the words, abbreviations.
LENIENT_RULES = (
    ("experts", re.compile(r"grouped[_ ]?mat[_ ]?mul|group[_ ]?mat[_ ]?mul|(?<![a-z])gmm(?![a-z])", re.IGNORECASE)),
    ("attention", re.compile(r"flash[_ ]?attention|fusion[_ ]?attention|fused[_ ]?infer[_ ]?attention", re.IGNORECASE)),
)
# The lanes of the process drawn from the profile: kernel classes, the compute stream's state, the collectives.
STATE_LANE = "state"
COLLECTIVE_LANES = (
    ("coll:alltoallv", "alltoallv: the MoE token exchange"),
    ("coll:alltoall", "alltoall: equal splits (counts)"),
    ("coll:allgather", "allGather: weights"),
    ("coll:reducescatter", "reduceScatter: gradients"),
    ("coll:other", "other collectives"),
)
COLLECTIVE_LANE_OF = {"alltoallv": "coll:alltoallv", "alltoall": "coll:alltoall", "allgather": "coll:allgather",
                      "reducescatter": "coll:reducescatter"}
PROFILE_LANES = tuple((f"class:{key}", f"{letter} {title}") for letter, (key, title) in zip("ABCDEF", CLASSES)) + (
    (STATE_LANE, "G compute stream: computing / waiting / idle"),
) + tuple((key, f"{letter} {title}") for letter, (key, title) in zip("HIJKL", COLLECTIVE_LANES))
MERGE_US = 100.0                 # same-class kernels this close are drawn as one slice
IDLE_US = 20.0                   # a gap this long on the compute stream is drawn as idle
MIN_DRAW_US = 1.0                # slices shorter than this are counted but not drawn
MIN_STATE_US = 5.0               # on the state lane, waits and idle gaps shorter than this are not drawn
MAX_ROWS = 8                     # overlapping collectives of one kind are spread over this many rows
MS = 1e3


# -- intervals --------------------------------------------------------------------------------------------

def merged(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the union of intervals as sorted, disjoint (start, end) pairs."""
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def subtract(base: tuple[float, float], holes: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return ``base`` without the (merged, sorted) ``holes``, as disjoint intervals."""
    pieces, cursor = [], base[0]
    for start, end in holes:
        if end <= cursor or start >= base[1]:
            continue
        if start > cursor:
            pieces.append((cursor, min(start, base[1])))
        cursor = max(cursor, end)
    if cursor < base[1]:
        pieces.append((cursor, base[1]))
    return pieces


def total(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the summed length of intervals."""
    return sum(end - start for start, end in intervals)


class Busy:
    """The time a set of intervals covers, with prefix sums so that the overlap with any interval costs O(log n)."""

    def __init__(self, intervals: Iterable[tuple[float, float]]) -> None:
        """Merge the intervals and index them."""
        self.intervals = merged(intervals)
        self.starts = [a for a, _ in self.intervals]
        self.cumulative: list[float] = []
        running = 0.0
        for start, end in self.intervals:
            running += end - start
            self.cumulative.append(running)

    def before(self, x: float) -> float:
        """Return the covered time up to ``x``."""
        index = bisect.bisect_right(self.starts, x) - 1
        if index < 0:
            return 0.0
        start, end = self.intervals[index]
        return (self.cumulative[index - 1] if index else 0.0) + min(x, end) - start

    def overlap(self, start: float, end: float) -> float:
        """Return the covered time inside [start, end]."""
        return self.before(end) - self.before(start) if end > start else 0.0


# -- the classes of kernels -------------------------------------------------------------------------------

def parse_rules(specs: Sequence[str]) -> list[tuple[str, re.Pattern]]:
    """Parse ``CLASS=REGEX`` rules; CLASS is one of ``CLASS_KEYS``. Raises ValueError on a bad rule."""
    rules = []
    for spec in specs:
        key, separator, pattern = spec.partition("=")
        if not separator or key not in CLASS_KEYS or not pattern:
            raise ValueError(f"a class rule is CLASS=REGEX with CLASS one of {', '.join(CLASS_KEYS)}: {spec!r}")
        rules.append((key, re.compile(pattern, re.IGNORECASE)))
    return rules


def make_classifier(rules: Sequence[tuple[str, re.Pattern]] = ()) -> Callable[[str], str]:
    """Return a function from a kernel name to its class key: the given rules, then the defaults."""
    cache: dict[str, str] = {}

    def classify(name: str) -> str:
        """The class key of a kernel name."""
        found = cache.get(name)
        if found is None:
            found = next((key for key, pattern in rules if pattern.search(name)), None) or next(
                (key for key, pattern in LENIENT_RULES if pattern.search(name)), None) or CLASS_OF_CATEGORY.get(
                    category(name), "other")
            cache[name] = found
        return found

    return classify


# -- the trace and what is read from it -------------------------------------------------------------------

@dataclass
class Capture:
    """A trace file as it was written, and the index of it that the analysis reads."""

    path: str
    container: Any
    events: list[dict]
    trace: Trace
    size_bytes: int = 0


def load_capture(path: str) -> Capture:
    """Read a trace_view.json (or the newest one under a directory) once, keeping every event as it was written."""
    files = find_trace_files(path)
    if not files:
        raise FileNotFoundError(f"no trace_view.json under {path}")
    with open(files[-1], encoding="utf-8") as stream:
        container = json.load(stream)
    events = container.get("traceEvents", []) if isinstance(container, dict) else container
    trace = Trace(events)
    trace.path = files[-1]
    return Capture(files[-1], container, events, trace, os.path.getsize(files[-1]))


@dataclass
class Signals:
    """What the Ascend trace says about a rank: the compute stream, the collectives, the step ranges (microseconds)."""

    trace: Trace
    classify: Callable[[str], str]
    compute_key: tuple
    compute: list[Event]
    comms: list[Event]
    by_class: dict[str, Busy]
    by_category: dict[str, Busy]
    busy: Busy
    waits: Busy
    comm: dict[str, Busy]
    all_to_all: Busy
    steps: list[tuple[int, float, float]]
    extent: tuple[float, float]
    records: list[float] = field(default_factory=list)


def read_signals(trace: Trace, rules: Sequence[tuple[str, re.Pattern]] = ()) -> Signals:
    """Split the compute stream by kernel class and category, its waits, the collectives, the record tasks."""
    classify = make_classifier(rules)
    key = trace.compute_thread()
    compute = trace.thread_events(key)
    classes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    categories: dict[str, list[tuple[float, float]]] = defaultdict(list)
    busy, waits = [], []
    for event in compute:
        if is_sync(event):
            if "WAIT" in event.name.upper() and SYNC_PATTERN.match(event.name):
                waits.append((event.ts, event.end))
            continue
        busy.append((event.ts, event.end))
        classes[classify(event.name)].append((event.ts, event.end))
        categories[category(event.name)].append((event.ts, event.end))
    comms = trace.communications()
    comm: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in comms:
        comm[comm_type(event.name).lower()].append((event.ts, event.end))
    all_to_all = Busy(interval for name, items in comm.items() if "alltoall" in name for interval in items)
    everything = busy + waits
    extent = (min((a for a, _ in everything), default=0.0), max((b for _, b in everything), default=0.0))
    hardware = set(trace.find_processes("Ascend Hardware"))
    record_pattern = re.compile(r"^(EVENT|NOTIFY)[ _]?RECORD", re.IGNORECASE)
    records = sorted(event.ts for event in trace.events if event.pid in hardware and record_pattern.match(event.name))
    return Signals(trace, classify, key, compute, comms, {name: Busy(items) for name, items in classes.items()},
                   {name: Busy(items) for name, items in categories.items()}, Busy(busy), Busy(waits),
                   {name: Busy(items) for name, items in comm.items()}, all_to_all, trace.steps(), extent, records)


# -- the numbers of a profiled step -----------------------------------------------------------------------

def wait_kinds(waits: Sequence[Event], comms: Sequence[Event]) -> list[str]:
    """Return, for each stream wait, the kind of collective that released it, or "unattributed"."""
    return [comm_type(cause.name) if cause is not None else "unattributed"
            for cause in attribute_waits(list(waits), list(comms))]


def step_numbers(signals: Signals, number: int, start: float, end: float) -> dict[str, Any]:
    """Return the compute stream's accounting for one profiled step, in microseconds.

    ``busy_by_class`` is the summed duration of the kernels of each class; ``waits`` the stream waits by the kind of
    collective that released them; ``idle`` the gaps between tasks, with their count by length; ``collectives`` the
    time each kind is in flight and how much of it the compute stream does not hide.
    """
    tasks = window(signals.compute, start, end)
    parts = step_breakdown(tasks, start, end)
    by_class: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in tasks:
        if not is_sync(event):
            klass = signals.classify(event.name)
            by_class[klass] += event.dur
            counts[klass] += 1
    waiting = [event for event in tasks if is_sync(event) and "WAIT" in event.name.upper() and event.dur > 0]
    waited: dict[str, float] = defaultdict(float)
    for event, kind in zip(waiting, wait_kinds(waiting, signals.comms)):
        waited[kind] += event.dur
    exposure = comm_exposure(window(signals.comms, start, end), parts["compute_union"], parts["sync_union"])
    return {
        "step": number, "start": start, "end": end, "span": parts["span"], "compute": parts["compute"],
        "wait": parts["wait"], "idle": parts["idle"], "edges": parts["edges"], "tasks": parts["tasks"],
        "busy_by_class": dict(by_class), "kernels_by_class": dict(counts), "waits": dict(waited),
        "idle_buckets": parts["idle_buckets"], "collectives": exposure,
    }


def describe_step(numbers: dict[str, Any]) -> list[str]:
    """Return the text lines for one profiled step's numbers."""
    span = numbers["span"]

    def ms(value: float) -> str:
        """Milliseconds with a thousands separator, no decimals."""
        return f"{value / MS:,.0f}"

    lines = [f"ProfilerStep {numbers['step']}, {ms(span)} ms: the compute stream is computing "
             f"{ms(numbers['compute'])} ms ({numbers['compute'] / span:.0%}), waiting for another stream "
             f"{ms(numbers['wait'])} ms ({numbers['wait'] / span:.0%}), idle between tasks {ms(numbers['idle'])} ms "
             f"({numbers['idle'] / span:.0%}), outside its first and last task {ms(numbers['edges'])} ms"]
    lines.append("    computing, by class of kernel: " + " | ".join(
        f"{SHORT_NAMES[key]} {ms(numbers['busy_by_class'].get(key, 0.0))}" for key in CLASS_KEYS))
    if numbers["waits"]:
        lines.append("    waiting, by what released it: " + " | ".join(
            f"{kind} {ms(value)}" for kind, value in sorted(numbers["waits"].items(), key=lambda item: -item[1])))
    buckets = [bucket for bucket in numbers["idle_buckets"] if bucket["count"]]
    if buckets:
        lines.append("    idle gaps: " + " | ".join(
            f"{bucket['bucket']} x{bucket['count']} {ms(bucket['total_us'])} ms" for bucket in reversed(buckets)))
    rows = [row for row in numbers["collectives"] if row["name"] != "all collectives"]
    if rows:
        lines.append("    collectives in flight / not hidden by compute (ms): " + " | ".join(
            f"{row['name']} {ms(row['total_us'])} / {ms(row['exposed_us'])}" for row in rows))
    return lines


def inventory(capture: Capture, signals: Signals, limit: int = 6) -> list[str]:
    """Return text lines that say what the trace holds and how its kernels were classed, to find a wrong class."""
    trace = signals.trace
    lines = [f"the trace: {capture.path} ({capture.size_bytes / 2 ** 20:,.0f} MiB, {len(capture.events):,} events)"]
    streams = trace.streams()
    if streams:
        compute_label = trace.thread_label(signals.compute_key)
        lines.append("  streams of the device: " + "; ".join(
            f"{row['label'].split(' / ')[-1]} {row['events']:,} tasks" + (" (compute)" if row["label"] == compute_label
                                                                          else "") for row in streams[:6]))
    per_class: dict[str, Counter] = defaultdict(Counter)
    time_of: dict[str, float] = defaultdict(float)
    count_of: dict[str, int] = defaultdict(int)
    for event in signals.compute:
        if is_sync(event):
            continue
        klass = signals.classify(event.name)
        per_class[klass][base_name(event.name)] += event.dur
        time_of[klass] += event.dur
        count_of[klass] += 1
    titles = dict(CLASSES)
    for key in CLASS_KEYS:
        if not count_of[key]:
            lines.append(f"  {titles[key]}: none")
            continue
        shown = limit if key == "other" else 2
        lines.append(f"  {titles[key]}: {count_of[key]:,} kernels, {time_of[key] / MS:,.0f} ms; largest: " + ", ".join(
            f"{name} {value / MS:,.0f} ms" for name, value in per_class[key].most_common(shown)))
    kinds = Counter(comm_type(event.name) for event in signals.comms)
    if kinds:
        lines.append("  collectives (hcom): " + ", ".join(f"{kind} x{count}" for kind, count in kinds.most_common()))
    sync_names = Counter(re.sub(r"[ _]", "_", event.name.upper()) for event in signals.compute if is_sync(event))
    lines.append("  sync tasks on the compute stream: " + (", ".join(
        f"{name} x{count:,}" for name, count in sync_names.most_common(4)) or "none")
        + f"; event-record tasks on the device: {len(signals.records):,}")
    lines.append("  profiler steps: " + (", ".join(
        f"{number} ({(end - start) / MS:,.0f} ms)" for number, start, end in signals.steps) or "none found"))
    return lines


# -- the lanes --------------------------------------------------------------------------------------------

@dataclass
class Piece:
    """One slice of a lane; the unit of ``start`` and ``end`` is the caller's (microseconds for the profile)."""

    start: float
    end: float
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    row: int = 0


def class_pieces(signals: Signals, merge_us: float = MERGE_US) -> dict[str, list[Piece]]:
    """Return, per class, the runs of its kernels that lie within ``merge_us`` of each other, as slices."""
    runs: dict[str, list[list[float]]] = {key: [] for key in CLASS_KEYS}
    for event in signals.compute:
        if is_sync(event):
            continue
        klass = runs[signals.classify(event.name)]
        if klass and event.ts - klass[-1][1] <= merge_us:
            klass[-1][1] = max(klass[-1][1], event.end)
            klass[-1][2] += 1
            klass[-1][3] += event.dur
        else:
            klass.append([event.ts, event.end, 1, event.dur])
    return {f"class:{key}": [Piece(start, end, SHORT_NAMES[key], {
        "kernels": int(count), "computing_ms": round(busy / MS, 3), "span_ms": round((end - start) / MS, 3)})
        for start, end, count, busy in rows if end - start >= MIN_DRAW_US] for key, rows in runs.items()}


def state_pieces(signals: Signals, idle_us: float = IDLE_US) -> list[Piece]:
    """Partition the compute stream into computing, waiting (by what released it) and idle slices.

    Gaps shorter than ``idle_us`` are absorbed in the computing slice around them.
    """
    waiting = [event for event in signals.compute if is_sync(event) and "WAIT" in event.name.upper() and event.dur > 0]
    kind_of = {id(event): kind for event, kind in zip(waiting, wait_kinds(waiting, signals.comms))}
    pieces: list[Piece] = []
    run: Optional[list[float]] = None
    cursor: Optional[float] = None

    def flush() -> None:
        """Close the computing slice in progress."""
        nonlocal run
        if run is not None:
            pieces.append(Piece(run[0], run[1], "computing", {"kernels": int(run[2]), "computing_ms": round(
                run[3] / MS, 3), "span_ms": round((run[1] - run[0]) / MS, 3)}))
            run = None

    for event in signals.compute:
        if is_sync(event) and event.dur < MIN_STATE_US:      # a record, or a wait too short to draw: part of the run
            cursor = event.end if cursor is None else max(cursor, event.end)
            continue
        if cursor is not None and event.ts - cursor > idle_us:
            flush()
            pieces.append(Piece(cursor, event.ts, "idle", {"span_ms": round((event.ts - cursor) / MS, 3)}))
        if is_sync(event):
            flush()
            kind = kind_of.get(id(event))
            pieces.append(Piece(event.ts, event.end, f"waiting: {kind}" if kind else "waiting: other event", {
                "task": event.name, "span_ms": round(event.dur / MS, 3)}))
        elif run is None:
            run = [event.ts, event.end, 1, event.dur]
        else:
            run[1], run[2], run[3] = max(run[1], event.end), run[2] + 1, run[3] + event.dur
        cursor = event.end if cursor is None else max(cursor, event.end)
    flush()
    merged_pieces: list[Piece] = []
    for piece in pieces:                       # consecutive waits of one kind and idle gaps read better as one slice
        last = merged_pieces[-1] if merged_pieces else None
        if (last is not None and last.name == piece.name and last.name != "computing"
                and piece.start - last.end <= idle_us):
            last.end = piece.end
            last.args["span_ms"] = round((last.end - last.start) / MS, 3)
        else:
            merged_pieces.append(piece)
    return [piece for piece in merged_pieces
            if piece.end - piece.start >= (MIN_DRAW_US if piece.name == "computing" else MIN_STATE_US)]


def collective_pieces(signals: Signals) -> dict[str, list[Piece]]:
    """Return the collectives as slices, one lane per kind, overlapping ones on further rows."""
    by_lane: dict[str, list[Event]] = defaultdict(list)
    for event in signals.comms:
        by_lane[COLLECTIVE_LANE_OF.get(comm_type(event.name).lower(), "coll:other")].append(event)
    result: dict[str, list[Piece]] = {key: [] for key, _ in COLLECTIVE_LANES}
    for lane, events in by_lane.items():
        row_ends: list[float] = []
        for event in sorted(events):
            row = next((index for index, end in enumerate(row_ends) if event.ts >= end), None)
            if row is None:
                row = len(row_ends)
                row_ends.append(event.end)
            else:
                row_ends[row] = event.end
            result[lane].append(Piece(event.ts, event.end, comm_type(event.name), {
                "event": event.name, "span_ms": round(event.dur / MS, 3)}, min(row, MAX_ROWS - 1)))
    return result


def profile_lanes(signals: Signals, merge_us: float = MERGE_US, idle_us: float = IDLE_US) -> dict[str, list[Piece]]:
    """Return every lane of the process drawn from the profile, keyed as in ``PROFILE_LANES``; times in microseconds."""
    lanes = class_pieces(signals, merge_us)
    lanes[STATE_LANE] = state_pieces(signals, idle_us)
    lanes.update(collective_pieces(signals))
    return lanes


# -- the events written into the trace --------------------------------------------------------------------

def free_pids(events: Sequence[dict], count: int, base: int = 9_000_000) -> list[int]:
    """Return ``count`` integer process ids that no event of the trace uses."""
    used = {event.get("pid") for event in events}
    found: list[int] = []
    candidate = base
    while len(found) < count:
        if candidate not in used and str(candidate) not in used:
            found.append(candidate)
        candidate += 1
    return found


def process_events(pid: Any, name: str, sort_index: int, lanes: Sequence[tuple[str, str]],
                   pieces: dict[str, list[Piece]], offset_us: float = 0.0, scale: float = 1.0) -> list[dict]:
    """Return the Chrome-trace events of one process: its name, its lanes as threads, and the slices.

    A slice at ``start`` .. ``end`` (in the pieces' unit) is written at ``offset_us + start * scale`` microseconds, so
    pieces in milliseconds since a step's start use ``scale=1000``. Pieces of a lane that overlap use ``Piece.row``.
    """
    events: list[dict[str, Any]] = [
        {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}},
        {"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": sort_index}},
    ]
    for number, (lane, title) in enumerate(lanes, start=1):
        rows = sorted({piece.row for piece in pieces.get(lane, [])} | {0})
        for row in rows:
            tid = number * 10 + row
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                           "args": {"name": title + (f" (row {row + 1})" if row else "")}})
            events.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": tid}})
        for piece in pieces.get(lane, []):
            if (piece.end - piece.start) * scale < MIN_DRAW_US:
                continue
            events.append({"ph": "X", "name": piece.name, "cat": lane, "pid": pid, "tid": number * 10 + piece.row,
                           "ts": offset_us + piece.start * scale, "dur": (piece.end - piece.start) * scale,
                           "args": piece.args})
    return events


def clip_events(events: Sequence[dict], span: tuple[float, float]) -> list[dict]:
    """Return the events of a window: the metadata whole, a slice clipped to it, anything else dropped.

    Used for the events this module adds, whose slices (a step, a layer) are longer than the window one wants to
    look at; clipping them keeps the view as narrow as the window instead of as wide as the step.
    """
    kept: list[dict] = []
    for event in events:
        if event.get("ph") != "X":
            kept.append(event)
            continue
        start = float(event["ts"])
        end = start + float(event.get("dur", 0.0))
        if end < span[0] or start > span[1]:
            continue
        low, high = max(start, span[0]), min(end, span[1])
        kept.append({**event, "ts": low, "dur": high - low})
    return kept


def in_window(events: Sequence[dict], span: tuple[float, float]) -> list[dict]:
    """Return the metadata events and the events that start inside a window, each exactly as it was written."""
    return [event for event in events
            if event.get("ph") == "M" or span[0] <= float(event.get("ts", 0.0)) <= span[1]]


def write_integrated(capture: Capture, extra: Sequence[dict], path: str,
                     span: Optional[tuple[float, float]] = None) -> int:
    """Write the original events untouched, then ``extra``, in the original container; return the file's size.

    A trace that is a bare list of events (what the Ascend exporter writes) is extended in the text of the file, so the
    original stays byte for byte and a trace of hundreds of megabytes is not serialized again; a trace in an object
    with a ``traceEvents`` key is written again with the new events added to that list.

    With ``span`` (microseconds on the trace's timeline) only the original events that START inside it are written,
    each still exactly as it was: a trace of gigabytes becomes a file a viewer opens at once, holding the stretch of
    the step one wants to look at.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if span is not None:
        events = in_window(capture.events, span) + list(extra)
        data: Any = {**capture.container, "traceEvents": events} if isinstance(capture.container, dict) else events
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        return os.path.getsize(path)
    if isinstance(capture.container, list):
        with open(capture.path, encoding="utf-8") as source:
            text = source.read().rstrip()
        if text.endswith("]"):
            added = ",".join(json.dumps(event, separators=(",", ":")) for event in extra)
            body = text[:-1].rstrip()
            separator = "" if (body.endswith("[") or not added) else ","
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(body + separator + added + "]\n")
            return os.path.getsize(path)
    events = capture.events + list(extra)
    data: Any = {**capture.container, "traceEvents": events} if isinstance(capture.container, dict) else events
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, separators=(",", ":"))
    return os.path.getsize(path)
