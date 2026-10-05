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
"""Draw the recorder's components as a trace, and check them against the Ascend profiler's kernels.

The numbers of ``analyze_hetero.py`` do not come from the profiler. ``hetero_profile`` stamps a device event at the
entry and the exit of each hooked module, and a component is a module, told by the role its path matches:
``language_model.layers.N.self_attn`` is attention, ``...mlp.experts`` the expert GEMMs, ``visual.blocks.N`` a vision
block. Some components are derived from those spans, and those are the ones to doubt: the MoE exchange is the MoE
block's span minus the experts' and the attention's inside it (the router, the dispatch and combine all-to-alls and the
wait for the rest of the EP group), the gaps are what lies between two spans, the wait for the EP group is the spread of
the exchange over the ranks of a group. This script draws the partition the report is built on, so that it can be looked
at, and, when the Ascend profiler ran on the same steps, lays the real kernels, stream waits and collectives of the
rank under it and measures how well the two agree.

For each rank drawn it writes a Chrome trace (open it in https://ui.perfetto.dev, chrome://tracing or MindStudio
Insight) with a process ``components`` whose lanes partition the step by the recorder's labels, one lane per component:

    1 step and phases            the step, each micro-batch's forward and backward, the tail (clip, optimizer, sync)
    2 layers and vision blocks   one slice per decoder layer and vision block and pass, to find a place by index
    3 vision tower               patch embedding, blocks, deepstack mergers, merger
    4 attention                  the decoder layers' self-attention
    5 expert GEMMs               the experts' span (the local experts' GEMMs; no collective inside)
    6 MoE exchange               the MoE block minus the above: router, dispatch and combine all-to-all, waits
    7 embedding, head, loss      the embedding, the vocabulary projection and the loss
    8 inside layers              what a layer holds besides attention and MoE: norms, residuals
    9 between modules            the weights' wait, launch gaps, glue; named by the slice that follows
    10 custom regions            modules of ``extra_roles`` and ``HETERO_PROFILE.region`` spans

Lanes 3 to 9 never overlap: every instant between the first and the last boundary is in exactly one of them (lane 10
belongs to a design and may wrap other modules). A slice is named after its component and pass (``attention fwd``,
``experts recompute``, ``exchange bwd``) so a viewer colours the same component alike; the layer, the module path and
the duration are in the slice's arguments.

With a profiler trace of the same rank it first ties the two clocks together. The recorder's stamps are device event
records and the profiler lists a record task for each, so the offset at which the stamps coincide with those tasks (to
15 us) gives the start of the step on the trace's timeline, and pairs the record step with the profiler step, without
using a single label. Where the trace lists no such tasks, it slides the slices over the kernels instead, scoring how
much of the grouped-matmul time falls in the expert slices and of the attention time in the attention and vision slices
and how busy the compute stream is inside them; that peak must stand out from the best place elsewhere. Then it
measures what the compute stream really did inside the slices:

- the recall of the three kernel classes the labels predict: the GroupedMatmul kernels inside the expert slices, the
  attention kernels inside the attention and vision slices, the all-to-all collectives inside the exchange slices;
- the compute stream's busy, waiting and idle share inside each lane, and the kernels and collectives it met there;
  the exchange lane should be a stalled stream and an ``alltoallv``, the gaps an all-gather wait, and the stream should
  not wait inside the attention, expert and vision slices (the weights' wait belongs outside: the hook-order check);
- the sum of each lane against the report's component (``analyze_hetero.build_rows``), which must be the same number.

The lanes alone (small, for every rank drawn) go to ``<run>/profile/components/components.json`` with the summary and
the text report; the lanes over that rank's real events of the step, which are large, go to
``<run>/profile/components_full/components_rank<N>_with_trace.json``.

    python examples/qwen3_vl_30b_perf/component_trace.py <run dir>            # <run>/hetero and <run>/profile
    python examples/qwen3_vl_30b_perf/component_trace.py <run dir> --rank 0 17 --no-original

It needs nothing but the standard library, so it runs on the nodes (``cluster exec``) where the traces are. Without
``<run>/profile`` it draws the lanes alone, on the recorder's clock.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

# Run as a script, Python puts this directory first on the import path.
from analyze_hetero import build_rows, spans_of as record_spans
from ascend_trace import SYNC_PATTERN, Trace, category, comm_type, find_rank_traces, is_sync

PHASES = ("fwd", "recompute", "bwd")
LANES = (
    ("steps", "1 step and phases"),
    ("layers", "2 layers and vision blocks"),
    ("vision", "3 vision tower"),
    ("attention", "4 attention"),
    ("experts", "5 expert GEMMs"),
    ("exchange", "6 MoE exchange: router, all-to-all, wait"),
    ("head", "7 embedding, head, loss"),
    ("inside", "8 inside layers: norms, residuals"),
    ("between", "9 between modules: weights, launch gaps"),
    ("custom", "10 custom regions"),
)
# The lanes that partition the step: each instant of it is in exactly one of them.
PARTITION = ("vision", "attention", "experts", "exchange", "head", "inside", "between", "custom")
LEAF_LANES = ("vision", "attention", "experts", "exchange", "head", "custom")
COMPUTE_LANES = ("vision", "attention", "experts")
VISION_ROLES = ("vision.patch_embed", "vision.block", "vision.deepstack", "vision.merger")
HEAD_ROLES = ("lm_head", "text.embed")
STOCK_ROLES = frozenset({
    "root", "vision.root", "text.root", "text.router", "text.norm", "text.layer", "text.moe", "text.attn",
    "text.experts", *VISION_ROLES, *HEAD_ROLES,
})
SLICE_NAMES = {
    "vision.patch_embed": "patch embed", "vision.block": "vision block", "vision.deepstack": "deepstack merger",
    "vision.merger": "merger", "text.attn": "attention", "text.experts": "experts", "lm_head": "lm head",
    "text.embed": "embedding",
}
# What the labels predict, as (what, kernel class, lanes whose slices should hold it, least share that must).
CHECKS = (
    ("GroupedMatmul kernels inside the expert slices", "grouped matmul", ("experts",), 0.90),
    ("attention kernels inside the attention and vision slices", "attention", ("attention", "vision"), 0.90),
    ("all-to-all collectives inside the exchange slices", "alltoall", ("exchange",), 0.80),
)
ALIGN_CLASSES = (("grouped matmul", ("experts",)), ("attention", ("attention", "vision")))
PROCESS_BASE = 9_000_000
MIN_SLICE_MS = 0.001
COARSE_US, WIDE_US, FINE_US, POLISH_US = 2000.0, 10_000.0, 100.0, 10.0
MARGIN_BEFORE_US, MARGIN_AFTER_US = 300_000.0, 600_000.0
SEPARATE_US = 20_000.0           # peaks closer than this are one peak
CLEAR_PEAK = 1.3                 # the best alignment must beat any other by this factor
HOOK_ORDER_WAIT_SHARE = 0.10     # stream waits inside compute slices above this share: the hooks sit elsewhere
RECONCILE_TOLERANCE_MS = 0.01
# The recorder stamps its events with a device event record; the profiler lists such a task on the stream, so the
# stamps are anchors that pin the clocks together to a few microseconds.
RECORD_PATTERN = re.compile(r"^(EVENT|NOTIFY)[ _]?RECORD", re.IGNORECASE)
ANCHOR_US = 15.0                 # a stamp and a record task this close are one event
ANCHOR_HYPOTHESES = 3            # the first stamps, each tried against every record task of the window
ANCHOR_FIRST, ANCHOR_FIRST_NEED = 8, 6   # a candidate offset must first match 6 of 8 stamps
ANCHOR_PROBE, ANCHOR_PROBE_SHARE = 40, 0.6
ANCHOR_SAMPLE = 400
MIN_STAMPS = 20
MIN_ANCHORED = 0.4               # share of the stamps that must coincide with a record task for the anchors to count
PARTIAL_ANCHORED = 0.9           # below this the match is partial: clocks drifting over the step, or foreign tasks


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


# -- the recorder's side ----------------------------------------------------------------------------------

@dataclass
class Span:
    """One module's span in one pass and micro-batch, in milliseconds since the step's start event."""

    role: str
    index: Optional[int]
    name: str
    phase: str
    occ: int
    start: float
    end: float


@dataclass
class Piece:
    """One slice of a lane, in milliseconds since the step's start event."""

    start: float
    end: float
    name: str
    args: dict[str, Any] = field(default_factory=dict)


def load_rank(directory: str, rank: int) -> Optional[tuple[dict, list[dict]]]:
    """Return a rank's header (modules added later merged in) and step records, or None without a file."""
    for path in sorted(glob.glob(os.path.join(directory, "rank*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            first = stream.readline()
        if not first.strip() or json.loads(first).get("rank") != rank:
            continue
        with open(path, encoding="utf-8") as stream:
            lines = [json.loads(line) for line in stream if line.strip()]
        header = lines[0]
        records = [line for line in lines[1:] if line.get("kind") == "step"]
        known = {module["id"] for module in header["modules"]}
        for record in records:
            for module in record.get("modules_added", []):
                if module["id"] not in known:
                    header["modules"].append(module)
                    known.add(module["id"])
        return header, records
    return None


def ranks_with_records(directory: str) -> list[int]:
    """Return the ranks that have a record file in ``directory``."""
    ranks = []
    for path in glob.glob(os.path.join(directory, "rank*.jsonl")):
        with open(path, encoding="utf-8") as stream:
            first = stream.readline()
        if first.strip():
            ranks.append(int(json.loads(first)["rank"]))
    return sorted(ranks)


def spans_of(header: dict, record: dict) -> list[Span]:
    """Return the closed spans of a step record, in time order (the boundaries are read as the report reads them)."""
    modules = {module["id"]: module for module in header["modules"]}
    spans = []
    for (module_id, phase, occ), pair in record_spans(record).items():
        module = modules.get(module_id)
        if module is not None and "in" in pair and "out" in pair and phase in PHASES:
            spans.append(Span(module["role"], module["index"], module["name"], phase, occ, pair["in"], pair["out"]))
    return sorted(spans, key=lambda span: (span.start, span.end))


def slice_name(span: Span) -> str:
    """Return a slice's name: the component and the pass, so a viewer colours a component alike in every layer."""
    return f"{SLICE_NAMES.get(span.role, span.name)} {span.phase}"


def build_lanes(spans: list[Span], step_ms: float, step: int) -> dict[str, list[Piece]]:
    """Cut the step into the lanes: every instant between the first and the last boundary is in one partition lane.

    The MoE exchange of a block is its span minus the attention and expert spans of the same layer that lie inside
    it: the router, the dispatch and combine all-to-alls and the wait for the EP group. With non-reentrant
    checkpointing the block's backward span also holds the layer's recompute (the block's own recompute span never
    closes), so the recomputed attention and experts are carved out of it, and what is left of the recompute, its
    router and all-to-alls, stays in the exchange, exactly as ``analyze_hetero`` books it.
    """
    lanes: dict[str, list[Piece]] = {lane: [] for lane, _ in LANES}

    def add(lane: str, start: float, end: float, name: str, **args: Any) -> None:
        """Append a slice if it has a length."""
        if end - start >= MIN_SLICE_MS:
            lanes[lane].append(Piece(start, end, name, {**args, "ms": round(end - start, 3)}))

    def facts(span: Span) -> dict[str, Any]:
        """The arguments every slice of a span carries."""
        return {"module": span.name, "layer": span.index, "pass": span.phase, "micro_batch": span.occ}

    by_layer: dict[tuple[Optional[int], int], list[Span]] = {}
    for span in spans:
        if span.role in ("text.attn", "text.experts"):
            by_layer.setdefault((span.index, span.occ), []).append(span)
    for span in spans:
        if span.role in VISION_ROLES:
            add("vision", span.start, span.end, slice_name(span), **facts(span))
        elif span.role == "text.attn":
            add("attention", span.start, span.end, slice_name(span), **facts(span))
        elif span.role == "text.experts":
            add("experts", span.start, span.end, slice_name(span), **facts(span))
        elif span.role in HEAD_ROLES:
            add("head", span.start, span.end, slice_name(span), **facts(span))
        elif span.role == "text.moe":
            inside = [child for child in by_layer.get((span.index, span.occ), [])
                      if child.start >= span.start - 1e-9 and child.end <= span.end + 1e-9]
            experts = [child for child in inside if child.role == "text.experts"]
            first_expert = min((child.start for child in experts), default=None)
            last_expert = max((child.end for child in experts), default=None)
            for start, end in subtract((span.start, span.end), merged((c.start, c.end) for c in inside)):
                if first_expert is not None and end <= first_expert + 1e-9:
                    where = "before the experts: router, dispatch all-to-all, wait for the group"
                elif last_expert is not None and start >= last_expert - 1e-9:
                    where = "after the experts: combine all-to-all, wait for the group, aggregate"
                else:
                    where = "between the recomputed pieces"
                add("exchange", start, end, f"exchange {span.phase}", where=where, **facts(span))
        elif span.role not in STOCK_ROLES:
            add("custom", span.start, span.end, f"{span.name} {span.phase}", **facts(span))
        if span.role in ("text.layer", "vision.block"):
            kind = "layer" if span.role == "text.layer" else "vision block"
            add("layers", span.start, span.end, f"{kind} {span.index} {span.phase}", **facts(span))
    roots = [span for span in spans if span.role == "root" and span.phase == "fwd"]
    heads = [span for span in spans if span.role == "lm_head" and span.phase == "bwd"]
    for root in roots:                      # the loss: from the end of the forward to the logits' gradient arriving
        for head in (h for h in heads if h.occ == root.occ):
            add("head", root.end, head.start, "loss", module="loss", micro_batch=root.occ)
    leaf = merged((p.start, p.end) for lane in LEAF_LANES for p in lanes[lane])
    for span in spans:                      # what a layer holds besides the leaf lanes: norms, residuals
        if span.role == "text.layer":
            for start, end in subtract((span.start, span.end), leaf):
                add("inside", start, end, f"layer glue {span.phase}", **facts(span))
    if spans:
        covered = merged((p.start, p.end) for lane in PARTITION if lane != "between" for p in lanes[lane])
        window = (min(span.start for span in spans), max(span.end for span in spans))
        followers = sorted((p.start, p.name) for lane in PARTITION if lane != "between" for p in lanes[lane])
        starts = [start for start, _ in followers]
        for start, end in subtract(window, covered):
            index = bisect.bisect_left(starts, end - 1e-6)
            add("between", start, end, "between modules",
                before=followers[index][1] if index < len(followers) else "the tail of the step")
    last = max((span.end for span in spans), default=0.0)
    add("steps", 0.0, step_ms, f"step {step}")
    for root in roots:
        add("steps", root.start, root.end, f"forward {root.occ}")
    for head in heads:                      # a micro-batch's backward: the logits' gradient to its last boundary
        ends = [span.end for span in spans if span.occ == head.occ and span.phase in ("bwd", "recompute")]
        add("steps", head.start, max(ends, default=head.start), f"backward {head.occ} (recompute inside)")
    add("steps", last, step_ms, "after the last boundary: clip, optimizer, sync")
    return lanes


def lane_ms(lanes: dict[str, list[Piece]]) -> dict[str, float]:
    """Return the milliseconds each lane holds."""
    return {lane: total((p.start, p.end) for p in pieces) for lane, pieces in lanes.items()}


def reconcile(header: dict, record: dict, lanes: dict[str, list[Piece]]) -> dict[str, tuple[float, float]]:
    """Return {lane: (milliseconds of the lane, milliseconds of the report's component)} for one rank and step.

    The report (``analyze_hetero.build_rows``) books a step's spans into components by sums; the lanes book the same
    spans as intervals. They must carry the same milliseconds, or one of the two bookings has a bug.
    """
    rank = header["rank"]
    rows = build_rows(SimpleNamespace(ranks=[rank], headers={rank: header}, steps={rank: [record]}))

    def summed(*groups: str) -> float:
        """Sum components over the micro-batches and passes."""
        return sum(row["t"][group][phase] for row in rows for group in groups for phase in PHASES)

    spent = lane_ms(lanes)
    pairs = {
        "vision": summed("vision"), "attention": summed("text_attn"), "experts": summed("text_experts"),
        "exchange": summed("ep_exchange"), "head": summed("embed", "head", "loss"), "custom": summed("custom"),
    }
    if summed("text_layer") > 0:
        pairs["inside"] = summed("text_layer") - summed("text_attn") - summed("text_experts")
    return {lane: (spent[lane], value) for lane, value in pairs.items()}


# -- the profiler's side ----------------------------------------------------------------------------------

@dataclass
class Signals:
    """What the Ascend trace says about a rank, as interval sets (microseconds on the trace's timeline)."""

    trace: Trace
    by_category: dict[str, Busy]
    busy: Busy
    waits: Busy
    comm: dict[str, Busy]
    all_to_all: Busy
    steps: list[tuple[int, float, float]]
    extent: tuple[float, float]
    records: list[float]


def read_signals(path: str) -> Signals:
    """Load a rank's trace and split its compute stream by kernel category, waits and collectives."""
    trace = Trace.load(path)
    key = trace.compute_thread()
    categories: dict[str, list[tuple[float, float]]] = {}
    busy, waits = [], []
    for event in trace.thread_events(key):
        if is_sync(event):
            if "WAIT" in event.name.upper() and SYNC_PATTERN.match(event.name):
                waits.append((event.ts, event.end))
            continue
        busy.append((event.ts, event.end))
        categories.setdefault(category(event.name), []).append((event.ts, event.end))
    comm: dict[str, list[tuple[float, float]]] = {}
    for event in trace.communications():
        comm.setdefault(comm_type(event.name).lower(), []).append((event.ts, event.end))
    all_to_all = Busy(interval for name, items in comm.items() if "alltoall" in name for interval in items)
    everything = busy + waits
    extent = (min((a for a, _ in everything), default=0.0), max((b for _, b in everything), default=0.0))
    hardware = set(trace.find_processes("Ascend Hardware"))
    records = sorted(event.ts for event in trace.events if event.pid in hardware and RECORD_PATTERN.match(event.name))
    return Signals(trace, {name: Busy(items) for name, items in categories.items()}, Busy(busy), Busy(waits),
                   {name: Busy(items) for name, items in comm.items()}, all_to_all, trace.steps(), extent, records)


# -- finding where the recorder's clock starts on the trace's timeline -------------------------------------

class Aligner:
    """Scores an offset by how well kernels sit inside the slices that carry their name.

    The score is the mean share of each kernel class (grouped matmul, attention) that falls in its slices, times the
    share of the slices that the compute stream keeps busy. Recall alone has a flat top (a shift that keeps the
    kernels inside slightly longer slices costs nothing); the busy share is what makes the peak sharp.
    """

    def __init__(self, lanes: dict[str, list[Piece]], signals: Signals) -> None:
        """Take the slices in microseconds, for the kernel classes that the trace has."""
        self.signals = signals
        self.classes: dict[str, list[tuple[float, float]]] = {}
        for name, names in ALIGN_CLASSES:
            if name in signals.by_category:
                spans = [(p.start * 1000.0, p.end * 1000.0) for lane in names for p in lanes[lane]]
                if spans:
                    self.classes[name] = spans
        self.span_us = sum(b - a for spans in self.classes.values() for a, b in spans)

    def usable(self) -> bool:
        """Return whether there is a kernel class to align on."""
        return bool(self.classes)

    def denominators(self, low: float, high: float) -> dict[str, float]:
        """Return each class's kernel time inside [low, high], the reference for the shares."""
        return {name: self.signals.by_category[name].overlap(low, high) for name in self.classes}

    def score(self, offset: float, totals: dict[str, float]) -> float:
        """Return the alignment score of an offset (microseconds on the trace's timeline)."""
        recalls_, covered = [], 0.0
        for name, spans in self.classes.items():
            busy = self.signals.by_category[name]
            recalls_.append(sum(busy.overlap(offset + a, offset + b) for a, b in spans) / totals[name]
                            if totals[name] else 0.0)
            covered += sum(self.signals.busy.overlap(offset + a, offset + b) for a, b in spans)
        return (sum(recalls_) / len(recalls_)) * (covered / self.span_us) if recalls_ and self.span_us else 0.0


@dataclass
class Fit:
    """Where a step starts on the trace's timeline, and how sure that is."""

    offset_us: float
    score: float
    elsewhere: float

    @property
    def clear(self) -> bool:
        """Return whether the peak stands out: every place away from it scores ``CLEAR_PEAK`` times lower."""
        return self.score > 0 and self.score >= CLEAR_PEAK * self.elsewhere


def scan(aligner: Aligner, totals: dict[str, float], low: float, high: float, step: float
         ) -> list[tuple[float, float]]:
    """Return (score, offset) on a grid."""
    result, offset = [], low
    while offset <= high:
        result.append((aligner.score(offset, totals), offset))
        offset += step
    return result


def best_of(aligner: Aligner, totals: dict[str, float], center: float, half: float, step: float
            ) -> tuple[float, float]:
    """Return (score, offset) of the best point of a fine grid around ``center``."""
    best = (-1.0, center)
    offset = center - half
    while offset <= center + half + 1e-6:
        best = max(best, (aligner.score(offset, totals), offset))
        offset += step
    return best


def refine(aligner: Aligner, totals: dict[str, float], coarse: list[tuple[float, float]], grid: float) -> Fit:
    """Refine the best peaks of a coarse scan, centre the plateau of the best, and measure the best elsewhere."""
    peaks: list[tuple[float, float]] = []
    for value, offset in sorted(coarse, reverse=True):
        if all(abs(offset - other) > SEPARATE_US for _, other in peaks):
            peaks.append((value, offset))
        if len(peaks) == 4:
            break
    best = max(best_of(aligner, totals, offset, grid, FINE_US) for _, offset in peaks)
    points = [(aligner.score(best[1] + k * POLISH_US, totals), best[1] + k * POLISH_US) for k in range(-40, 41)]
    top = max(value for value, _ in points)
    near = [offset for value, offset in points if value >= top - 0.002]
    final = near[len(near) // 2]
    elsewhere = max((value for value, offset in coarse if abs(offset - final) > SEPARATE_US), default=0.0)
    return Fit(final, top, elsewhere)


def search_window(window: tuple[float, float], step_ms: float) -> tuple[float, float]:
    """Return the offsets in which the step can start, given the profiler step's range (microseconds)."""
    low = window[0] - MARGIN_BEFORE_US
    return low, max(window[1] - step_ms * 1000.0 + MARGIN_AFTER_US, low + COARSE_US)


def spread(values: Sequence[float], count: int) -> list[float]:
    """Return about ``count`` values spread evenly over a sorted sequence."""
    return list(values[::max(len(values) // count, 1)])


def coincides(stamp_us: float, records: Sequence[float], offset: float) -> bool:
    """Return whether an event-record task lies within ``ANCHOR_US`` of a stamp placed at an offset."""
    index = bisect.bisect_left(records, offset + stamp_us - ANCHOR_US)
    return index < len(records) and records[index] <= offset + stamp_us + ANCHOR_US


def anchored_share(stamps_us: Sequence[float], records: Sequence[float], offset: float) -> float:
    """Return the share of the recorder's stamps that have an event-record task within ``ANCHOR_US`` at an offset."""
    return sum(coincides(stamp, records, offset) for stamp in stamps_us) / len(stamps_us) if stamps_us else 0.0


@dataclass
class Anchor:
    """Where the recorder's stamps coincide with the trace's event-record tasks."""

    offset_us: float
    share: float
    stamps: int


def anchor_search(record: dict, signals: Signals, low: float, high: float) -> Optional[Anchor]:
    """Find the offset in [low, high] at which the recorder's stamps coincide with the trace's event-record tasks.

    The stamps are the device events the recorder took at the module boundaries, and the profiler lists a record task
    on the stream for each. The first stamps are tried against every task of the window as the start of the step; a
    candidate that fails a cheap test on a few stamps is dropped, the others are scored on a spread of stamps. The
    result does not depend on the labels, so it pins the clocks together before the labels are checked, and it checks
    the recorder's clock as a side effect.
    """
    records = signals.records
    stamps = sorted({mark[4] * 1000.0 for mark in record["marks"]})
    if len(stamps) < MIN_STAMPS or not records:
        return None
    first, probe, sample = spread(stamps, ANCHOR_FIRST), spread(stamps, ANCHOR_PROBE), spread(stamps, ANCHOR_SAMPLE)
    candidates: set[int] = set()
    for hypothesis in stamps[:ANCHOR_HYPOTHESES]:
        window = records[bisect.bisect_left(records, low + hypothesis):bisect.bisect_right(records, high + hypothesis)]
        for task in window:
            offset = task - hypothesis
            if sum(coincides(stamp, records, offset) for stamp in first) >= ANCHOR_FIRST_NEED \
                    and anchored_share(probe, records, offset) >= ANCHOR_PROBE_SHARE:
                candidates.add(round(offset))
    scored = sorted(((anchored_share(sample, records, candidate), candidate) for candidate in candidates), reverse=True)
    if not scored:
        return None
    top, where = scored[0]
    near = sorted(candidate for value, candidate in scored
                  if value >= top - 0.005 and abs(candidate - where) <= 4 * ANCHOR_US)
    best = float(near[len(near) // 2])
    return Anchor(best, anchored_share(stamps, records, best), len(stamps))


# -- the cross-check --------------------------------------------------------------------------------------

def pieces_us(lanes: dict[str, list[Piece]], names: Sequence[str], offset: float) -> list[tuple[float, float]]:
    """Return the slices of some lanes on the trace's timeline (microseconds)."""
    return [(offset + p.start * 1000.0, offset + p.end * 1000.0) for lane in names for p in lanes[lane]]


def composition(lanes: dict[str, list[Piece]], signals: Signals, offset: float) -> dict[str, dict[str, Any]]:
    """Return, per partition lane, what the compute stream and the collectives really did inside its slices."""
    result: dict[str, dict[str, Any]] = {}
    for lane in PARTITION:
        slices = pieces_us(lanes, (lane,), offset)
        length = total(slices)
        if not length:
            continue
        busy = sum(signals.busy.overlap(a, b) for a, b in slices)
        waits = sum(signals.waits.overlap(a, b) for a, b in slices)
        kernels = {name: sum(items.overlap(a, b) for a, b in slices) for name, items in signals.by_category.items()}
        collectives = {name: sum(items.overlap(a, b) for a, b in slices) / 1000.0
                       for name, items in signals.comm.items()}
        result[lane] = {
            "ms": length / 1000.0, "busy": busy / length, "wait": waits / length,
            "idle": max(1.0 - (busy + waits) / length, 0.0),
            "kernels": {name: value / busy for name, value in sorted(kernels.items(), key=lambda item: -item[1])
                        if value > 0 and busy > 0},
            "collectives_ms": {name: value for name, value in sorted(collectives.items(), key=lambda item: -item[1])
                               if value > 0},
        }
    return result


def recalls(lanes: dict[str, list[Piece]], signals: Signals, offset: float, step_ms: float) -> list[dict[str, Any]]:
    """Return the share of each kernel class the labels predict that really lies in the slices that carry its name."""
    step = (offset, offset + step_ms * 1000.0)
    found = []
    for what, kind, names, floor in CHECKS:
        busy = signals.all_to_all if kind == "alltoall" else signals.by_category.get(kind)
        inside_step = busy.overlap(*step) if busy is not None else 0.0
        share = (sum(busy.overlap(a, b) for a, b in merged(pieces_us(lanes, names, offset))) / inside_step
                 if inside_step > 0 else None)
        found.append({"what": what, "share": share, "floor": floor, "ms": inside_step / 1000.0})
    return found


def hook_order(shape: dict[str, dict[str, Any]]) -> Optional[float]:
    """Return the share of the compute slices' time that the compute stream spends waiting (it belongs outside them)."""
    length = sum(entry["ms"] for lane, entry in shape.items() if lane in COMPUTE_LANES)
    waits = sum(entry["ms"] * entry["wait"] for lane, entry in shape.items() if lane in COMPUTE_LANES)
    return waits / length if length else None


def verdict(checks: list[dict[str, Any]], certain: bool, waiting: Optional[float]) -> str:
    """Say in a few words whether the kernels agree with the labels (``certain``: the clocks are surely aligned)."""
    known = [check for check in checks if check["share"] is not None]
    if not known:
        return "NOTHING TO COMPARE: no kernel of the checked classes in the step"
    failing = [check for check in known if check["share"] < check["floor"]]
    notes = []
    if waiting is not None and waiting > HOOK_ORDER_WAIT_SHARE:
        notes.append(f"the stream waits {waiting:.0%} of the time inside compute slices: the hooks may sit before the "
                     "weights' unshard wait")
    tail = ("; " + "; ".join(notes)) if notes else ""
    if failing and not certain:
        return ("ALIGNMENT UNCERTAIN: another place fits almost as well (see the best score elsewhere), so the low "
                "shares above may be the clocks; give the start with --offset-ms after looking at the file" + tail)
    if failing:
        return ("DISAGREE: " + "; ".join(f"{check['what']}: {check['share']:.0%} (expected >= {check['floor']:.0%})"
                                         for check in failing) + ". The clocks are aligned, so the labels are suspect "
                "there" + tail)
    return ("AGREE: every checked kernel class lies in the slices that carry its name"
            + ("" if certain else " (the alignment is not unique, but every class still falls in its slices)") + tail)


# -- the output trace -------------------------------------------------------------------------------------

def lane_events(lanes: dict[str, list[Piece]], pid: int, offset_us: float, rank: int, synthetic: bool) -> list[dict]:
    """Return the Chrome-trace events of one rank's simplified process."""
    tag = "SYNTHETIC " if synthetic else ""
    events: list[dict[str, Any]] = [
        {"ph": "M", "name": "process_name", "pid": pid,
         "args": {"name": f"{tag}rank {rank} | components (hetero_profile)"}},
        {"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": -1000 + pid % 1000}},
    ]
    for number, (lane, title) in enumerate(LANES, start=1):
        events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": number, "args": {"name": title}})
        events.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": number,
                       "args": {"sort_index": number}})
        for piece in lanes[lane]:
            events.append({"ph": "X", "name": piece.name, "cat": lane, "pid": pid, "tid": number,
                           "ts": offset_us + piece.start * 1000.0, "dur": (piece.end - piece.start) * 1000.0,
                           "args": piece.args})
    return events


def original_events(signals: Signals, first_pid: int, window: tuple[float, float], rank: int, host: bool,
                    min_us: float) -> tuple[list[dict], int]:
    """Return the rank's real events inside the window, in processes of their own, and how many were left out."""
    trace = signals.trace
    wanted = set(trace.processes) if host else set(trace.find_processes("Ascend Hardware")) | set(
        trace.find_processes("Communication"))
    mapping = {pid: first_pid + index for index, pid in enumerate(sorted(wanted, key=str))}
    events: list[dict[str, Any]] = []
    for pid, new in mapping.items():
        events.append({"ph": "M", "name": "process_name", "pid": new,
                       "args": {"name": f"rank {rank} | {trace.processes.get(pid, pid)}"}})
        events.append({"ph": "M", "name": "process_sort_index", "pid": new, "args": {"sort_index": new % 1000}})
    for (pid, tid), name in trace.threads.items():
        if pid in mapping:
            events.append({"ph": "M", "name": "thread_name", "pid": mapping[pid], "tid": tid, "args": {"name": name}})
    skipped = 0
    for event in trace.events:
        if event.pid in mapping and event.end >= window[0] and event.ts <= window[1]:
            if event.dur < min_us:
                skipped += 1
                continue
            events.append({"ph": "X", "name": event.name, "pid": mapping[event.pid], "tid": event.tid, "ts": event.ts,
                           "dur": event.dur, "args": event.args})
    return events, skipped


# -- one rank ---------------------------------------------------------------------------------------------

@dataclass
class Choice:
    """The pairing of a record step with a profiler step, and where the step starts on the trace."""

    step: int
    profiler_step: int
    offset_us: float
    anchor: Optional[Anchor] = None
    fit: Optional[Fit] = None


def pairings(by_step: dict[int, dict], signals: Signals, args: argparse.Namespace
             ) -> list[tuple[int, int, float, float]]:
    """Return the (record step, profiler step, start, end) pairs worth trying; the profiler may number otherwise."""
    windows = signals.steps or [(-1, signals.extent[0], signals.extent[1])]
    if args.search_all:
        windows = [(-1, signals.extent[0], signals.extent[1])]
    pairs = []
    for number, start, end in windows:
        if args.profiler_step is not None and number != args.profiler_step:
            continue
        for step in sorted(by_step):
            if args.step is not None and step != args.step:
                continue
            if args.step is None and number >= 0 and step not in (number - 1, number, number + 1, number + 2):
                continue
            pairs.append((step, number, start, end))
    return pairs


def offsets_of(args: argparse.Namespace, step_ms: float, start: float, end: float) -> tuple[float, float]:
    """Return the offsets to search for a step in a profiler step's range (the whole range with --search-all)."""
    if args.search_all:
        return start, max(end - step_ms * 1000.0, start)
    return search_window((start, end), step_ms)


def choose(by_step: dict[int, dict], header: dict, signals: Signals, args: argparse.Namespace, out: list[str]
           ) -> Optional[Choice]:
    """Find the record step, the profiler step and the offset that fit best; None if nothing can be paired.

    First the recorder's own event-record tasks: every pairing is searched for the offset at which its stamps coincide
    with them, which needs no label. Without such anchors, every plausible pairing is scanned on a coarse grid by how
    well the kernels sit in the slices that carry their name (the profiler may number its steps otherwise than the
    optimizer), the two best are refined, and the one with the higher score wins.
    """
    pairs = pairings(by_step, signals, args)
    anchors = []
    for step, number, start, end in pairs:
        found = anchor_search(by_step[step], signals, *offsets_of(args, by_step[step]["device_ms"], start, end))
        if found is not None:
            anchors.append((found.share, step, number, found))
    if anchors:
        anchors.sort(key=lambda item: -item[0])
        out.append("  event-record anchors (record step <-> profiler step: share of stamps matched): " + ", ".join(
            f"{step}<->{number}: {share:.0%}" for share, step, number, _ in anchors[:8]))
        share, step, number, found = anchors[0]
        if share >= MIN_ANCHORED:
            return Choice(step, number, found.offset_us, anchor=found)
    scanned = []
    for step, number, start, end in pairs:
        record = by_step[step]
        aligner = Aligner(build_lanes(spans_of(header, record), record["device_ms"], step), signals)
        if not aligner.usable():
            continue
        grid = WIDE_US if args.search_all else COARSE_US
        low, high = offsets_of(args, record["device_ms"], start, end)
        totals = aligner.denominators(low, high + record["device_ms"] * 1000.0)
        scores = scan(aligner, totals, low, max(high, low + grid), grid)
        scanned.append((max(scores)[0], step, number, aligner, totals, scores, grid))
    if not scanned:
        out.append("  no record step pairs with a profiler step, or the trace has no grouped matmul or attention "
                   "kernel")
        return None
    scanned.sort(key=lambda item: -item[0])
    out.append("  no event-record anchors" + ("" if not anchors else f" (best {anchors[0][0]:.0%})")
               + "; kernel alignment, pairings (record step <-> profiler step: coarse score): " + ", ".join(
                   f"{step}<->{number}: {value:.2f}" for value, step, number, *_ in scanned[:8]))
    fits = [(refine(aligner, totals, scores, grid), step, number) for _, step, number, aligner, totals, scores, grid
            in scanned[:2]]
    fit, step, number = max(fits, key=lambda item: item[0].score)
    return Choice(step, number, fit.offset_us, fit=fit)


def process_rank(rank: int, hetero_dir: str, profile_dir: str, args: argparse.Namespace, index: int,
                 out: list[str], full_dir: Optional[str]) -> Optional[tuple[list[dict], dict[str, Any]]]:
    """Build one rank's events and findings; None if the rank has no usable record."""
    loaded = load_rank(hetero_dir, rank)
    if loaded is None:
        out.append(f"rank {rank}: no record file in {hetero_dir}")
        return None
    header, records = loaded
    by_step = {record["step"]: record for record in records if record["marks"]}
    if not by_step:
        out.append(f"rank {rank}: no step with module boundaries (was hetero_profile.hooks off?)")
        return None
    traces = find_rank_traces(profile_dir) if os.path.isdir(profile_dir) and not args.no_trace else {}
    signals = read_signals(traces[rank]) if rank in traces else None
    choice: Optional[Choice] = None
    number = -1
    offset = 0.0
    if signals is None:
        step = args.step if args.step is not None else max(by_step)
        out.append(f"rank {rank}: no Ascend trace here, so the lanes alone, on the recorder's clock")
    elif args.offset_ms is not None:
        step = args.step if args.step is not None else max(by_step)
        offset = args.offset_ms * 1000.0
        out.append(f"rank {rank}: the step starts at {args.offset_ms} ms on the trace, as given")
    else:
        out.append(f"rank {rank}:")
        choice = choose(by_step, header, signals, args, out)
        if choice is None:
            return None
        step, number, offset = choice.step, choice.profiler_step, choice.offset_us
    if step not in by_step:
        out.append(f"rank {rank}: no step {step} with module boundaries; steps are {sorted(by_step)}")
        return None
    record = by_step[step]
    step_ms = record["device_ms"]
    lanes = build_lanes(spans_of(header, record), step_ms, step)
    spent = lane_ms(lanes)
    finding: dict[str, Any] = {"rank": rank, "step": step, "profiler_step": number, "step_ms": step_ms,
                               "lanes_ms": spent, "synthetic": header.get("time_source") == "synthetic"}
    out.append(f"  record step {step}, {step_ms:.1f} ms of device time; the lanes partition it:")
    for lane in PARTITION:
        if spent[lane]:
            out.append(f"    {lane:10s} {spent[lane]:10.1f} ms  {spent[lane] / step_ms:6.1%}")
    pairs = reconcile(header, record, lanes)
    worst = max((abs(mine - theirs) for mine, theirs in pairs.values()), default=0.0)
    finding["reconcile"] = {lane: {"lane_ms": mine, "report_ms": theirs} for lane, (mine, theirs) in pairs.items()}
    out.append("  the lanes against the report's components (analyze_hetero): " + (
        f"the same milliseconds (largest difference {worst:.4f} ms)" if worst <= RECONCILE_TOLERANCE_MS else
        "DIFFERENT, " + ", ".join(f"{lane} {mine:.1f} vs {theirs:.1f}" for lane, (mine, theirs) in pairs.items()
                                  if abs(mine - theirs) > RECONCILE_TOLERANCE_MS)))
    events = lane_events(lanes, PROCESS_BASE + 1000 * index, offset, rank, finding["synthetic"])
    if signals is None:
        return events, finding
    shape = composition(lanes, signals, offset)
    checks = recalls(lanes, signals, offset, step_ms)
    waiting = hook_order(shape)
    finding.update(offset_us=offset, composition=shape, checks=checks, compute_wait_share=waiting)
    certain = choice is None
    if choice is not None and choice.anchor is not None:
        certain = True
        finding.update(anchored=choice.anchor.share)
        aligner = Aligner(lanes, signals)
        score = (aligner.score(offset, aligner.denominators(offset, offset + step_ms * 1000.0))
                 if aligner.usable() else 0.0)
        out.append(f"  the step starts at {offset / 1000.0:.3f} ms on the trace (profiler step {number}), found on the "
                   f"recorder's own event-record tasks: {choice.anchor.share:.0%} of its {choice.anchor.stamps} stamps "
                   f"coincide with one within {ANCHOR_US:g} us"
                   + ("" if choice.anchor.share >= PARTIAL_ANCHORED else " (partial: the clocks drift over the step, "
                                                                        "or some tasks are not the recorder's)")
                   + f"; the kernels score {score:.2f} there")
    elif choice is not None and choice.fit is not None:
        certain = choice.fit.clear
        finding.update(alignment=choice.fit.score, elsewhere=choice.fit.elsewhere)
        out.append(f"  the step starts at {offset / 1000.0:.3f} ms on the trace (profiler step {number}), found on the "
                   f"kernels alone: score {choice.fit.score:.2f}, best elsewhere {choice.fit.elsewhere:.2f}")
    for lane in PARTITION:
        if lane not in shape:
            continue
        entry = shape[lane]
        kernels = ", ".join(f"{name} {share:.0%}" for name, share in list(entry["kernels"].items())[:3]) or "none"
        comms = ", ".join(f"{name} {ms:.1f} ms" for name, ms in list(entry["collectives_ms"].items())[:2]) or "none"
        out.append(f"    {lane:10s} stream busy {entry['busy']:4.0%} waiting {entry['wait']:4.0%} idle "
                   f"{entry['idle']:4.0%} | kernels: {kernels} | collectives: {comms}")
    for check in checks:
        share = check["share"]
        out.append(f"    {check['what']}: " + ("n/a (none in the step)" if share is None else f"{share:.1%}"))
    if waiting is not None:
        out.append(f"    stream waits inside the attention, expert and vision slices: {waiting:.1%} of their time")
    finding["verdict"] = verdict(checks, certain, waiting)
    out.append(f"  {finding['verdict']}")
    if args.original and full_dir is not None:
        window = (offset - args.margin_ms * 1000.0, offset + step_ms * 1000.0 + args.margin_ms * 1000.0)
        extra, skipped = original_events(signals, PROCESS_BASE + 1000 * index + 1, window, rank,
                                         args.include_host, args.min_us)
        os.makedirs(full_dir, exist_ok=True)
        path = os.path.join(full_dir, f"components_rank{rank}_with_trace.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({"traceEvents": events + extra, "displayTimeUnit": "ms"}, stream)
        out.append(f"  wrote {path} ({len(extra)} real events, {skipped} shorter than {args.min_us:g} us left out)")
    return events, finding


# -- the command ------------------------------------------------------------------------------------------

def leaf_work(header: dict, record: dict) -> float:
    """Return the compute a rank's step holds: the vision tower, the attention, the experts, the head and loss."""
    spent = lane_ms(build_lanes(spans_of(header, record), record["device_ms"], record["step"]))
    return sum(spent[lane] for lane in ("vision", "attention", "experts", "head"))


def pick_ranks(hetero_dir: str, profile_dir: str, out: list[str]) -> list[int]:
    """Return the idlest and the busiest rank of the last recorded step among those that have a record and a trace."""
    traces = find_rank_traces(profile_dir) if os.path.isdir(profile_dir) else {}
    local = ranks_with_records(hetero_dir)
    candidates = [rank for rank in local if rank in traces] or local
    scored = []
    for rank in candidates:
        loaded = load_rank(hetero_dir, rank)
        hooked = [record for record in (loaded[1] if loaded else []) if record["marks"]]
        if hooked:
            scored.append((leaf_work(loaded[0], hooked[-1]), rank))
    scored.sort()
    chosen = sorted({scored[0][1], scored[-1][1]}) if scored else candidates[:1]
    out.append(f"ranks drawn: the idlest and the busiest of the last recorded step, among {len(candidates)}: {chosen}")
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the command line, build the traces and print the findings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run directory holding hetero/ (records) and profile/ (Ascend traces)")
    parser.add_argument("--hetero-dir", default=None, help="the records (default: <run>/hetero, or <run> itself)")
    parser.add_argument("--profile-dir", default=None, help="the Ascend traces (default: <run>/profile)")
    parser.add_argument("--rank", type=int, nargs="+", default=None,
                        help="ranks to draw (default: the idlest and the busiest of those that have both files)")
    parser.add_argument("--step", type=int, default=None, help="the record step (default: found by alignment)")
    parser.add_argument("--profiler-step", type=int, default=None, help="the ProfilerStep number to use")
    parser.add_argument("--offset-ms", type=float, default=None,
                        help="where the step starts on the trace's timeline, instead of searching (one --rank)")
    parser.add_argument("--search-all", action="store_true",
                        help="search the whole trace for the step, not the profiler step's range (slower)")
    parser.add_argument("--out", default=None, help="small files: lanes, summary, report (<run>/profile/components)")
    parser.add_argument("--full-out", default=None,
                        help="large files: lanes and the real events (<run>/profile/components_full)")
    parser.add_argument("--no-trace", action="store_true", help="draw the lanes alone, from the records")
    parser.add_argument("--no-original", dest="original", action="store_false",
                        help="do not write the file that holds the real device events under the lanes")
    parser.add_argument("--include-host", action="store_true", help="copy the host processes of the trace as well")
    parser.add_argument("--margin-ms", type=float, default=50.0, help="real events kept around the step")
    parser.add_argument("--min-us", type=float, default=5.0, help="real events shorter than this are left out")
    args = parser.parse_args(argv)
    default = os.path.join(args.run, "hetero")
    hetero_dir = args.hetero_dir or (default if os.path.isdir(default) else args.run)
    profile_dir = args.profile_dir or os.path.join(args.run, "profile")
    out_dir = args.out or os.path.join(args.run, "profile", "components")
    full_dir = args.full_out or os.path.join(args.run, "profile", "components_full")
    out: list[str] = []
    if not glob.glob(os.path.join(hetero_dir, "rank*.jsonl")):
        print(f"no rank*.jsonl in {hetero_dir}: pass the run directory (it holds hetero/ and profile/) or --hetero-dir")
        return 1
    ranks = args.rank if args.rank is not None else pick_ranks(hetero_dir, profile_dir, out)
    events: list[dict[str, Any]] = []
    findings = []
    for index, rank in enumerate(ranks):
        built = process_rank(rank, hetero_dir, profile_dir, args, index, out, full_dir)
        if built is not None:
            events += built[0]
            findings.append(built[1])
    if not findings:
        print("\n".join(out))
        return 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "components.json"), "w", encoding="utf-8") as stream:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, stream)
    with open(os.path.join(out_dir, "components_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(findings, stream, indent=2)
    out.append(f"wrote {os.path.join(out_dir, 'components.json')} ({len(events)} events: the lanes alone) and "
               "components_summary.json; open a trace in https://ui.perfetto.dev, chrome://tracing or MindStudio "
               "Insight, the process 'components' sits above the real streams")
    report = "\n".join(out)
    print(report)
    with open(os.path.join(out_dir, "components.txt"), "w", encoding="utf-8") as stream:
        stream.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
