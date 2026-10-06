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
"""Put the detected components into the Ascend profiler's trace, to check them by eye.

The numbers of ``analyze_hetero.py`` come from hooks that stamp a device event at the boundaries of modules. This
command lays what it detects over the profiler's own record of the same rank. For each rank it writes ONE trace file:
the original ``trace_view.json`` with every event untouched (host and device, flows and counters included), followed
by new processes drawn on the same timeline, so it opens in https://ui.perfetto.dev, chrome://tracing or MindStudio
Insight as the original trace with the components above it:

- ``components from the profile alone``: no hook record is used. The kernels of the compute stream by class (attention,
  expert GEMMs, routing/sort/index, dense matmul, norm/activation, other), the stream's state (computing, waiting for
  which collective, idle), and the collectives by kind (the MoE token exchange, the counts exchange, all-gather,
  reduce-scatter). ``profile_components.py`` holds this part and works on any trace_view.json;
- ``components from the hooks``: the lanes of the heterogeneity report (vision, attention, expert GEMMs, MoE exchange,
  head and loss, glue inside layers, the gaps between modules, custom regions, the step and its phases), each module
  boundary placed on the profiler's own timestamp of the device event the recorder took there. Needs the rank's
  ``hetero/`` records.

    python examples/qwen3_vl_30b_perf/component_trace.py <run dir>            # <run>/profile and <run>/hetero
    python examples/qwen3_vl_30b_perf/component_trace.py <trace_view.json> --rank 3   # the profile alone, any trace

The report (``components.txt``) says what the trace holds and how its kernels were classed (so a name in the wrong
class is easy to see; ``--class attention=REGEX`` adds a rule), numbers each profiled step from the trace alone, and,
with records, how well the two agree. How the hooks are tied to the profile:

1. The recorder's stamps are device event records, and the profiler lists a record task for each. The offset at which
   they coincide with those tasks (to 15 us, found without any label) gives the start of the step on the trace's
   timeline and pairs the record step with the profiler step. Without such tasks the slices are slid over the kernels
   instead, which is sharper if the labels are right and ambiguous if they are not (the report says which).
2. Each stamp is then moved onto the timestamp of its own record task, so the lanes follow the profiler's clock stamp
   by stamp; the median and largest shift and the drift over the step show whether the two clocks agree.
3. The recall of the three kernel classes the labels predict (GroupedMatmul kernels inside the expert slices, attention
   kernels inside the attention and vision slices, all-to-alls inside the exchange slices), the compute stream's busy,
   waiting and idle share inside each lane, and the sum of each lane against the report's component must agree.

It needs nothing but the standard library, so it runs on the nodes (``cluster exec``) where the traces are.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

# Run as a script, Python puts this directory first on the import path.
from analyze_hetero import build_rows, exchange_cells, spans_of as record_spans
from ascend_trace import find_rank_traces, trace_rank
from profile_components import (
    PROFILE_LANES, Capture, Piece, Signals, clip_events, describe_step, free_pids, inventory, load_capture, merged,
    parse_rules, process_events, profile_lanes, read_signals, step_numbers, subtract, total, write_integrated,
)

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
# What the labels predict, as (what, kernel class or "alltoall", lanes whose slices should hold it, least share).
CHECKS = (
    ("GroupedMatmul kernels inside the expert slices", "experts", ("experts",), 0.90),
    ("attention kernels inside the attention and vision slices", "attention", ("attention", "vision"), 0.90),
    ("all-to-all collectives inside the exchange slices", "alltoall", ("exchange",), 0.80),
)
ALIGN_CLASSES = (("experts", ("experts",)), ("attention", ("attention", "vision")))
PROCESS_BASE = 9_000_000
MIN_SLICE_MS = 0.001             # slices shorter than this are counted in the lanes but not drawn (1 us)
COARSE_US, WIDE_US, FINE_US, POLISH_US = 2000.0, 10_000.0, 100.0, 10.0
MARGIN_BEFORE_US, MARGIN_AFTER_US = 300_000.0, 600_000.0
# The offsets to search always span at least this much: a recorder step longer than the profiler's range of it would
# otherwise leave no window at all, and the anchors could not be found.
MIN_SEARCH_US = 2_000_000.0
SEPARATE_US = 20_000.0           # peaks closer than this are one peak
CLEAR_PEAK = 1.3                 # the best alignment must beat any other by this factor
HOOK_ORDER_WAIT_SHARE = 0.10     # stream waits inside compute slices above this share: the hooks sit elsewhere
RECONCILE_TOLERANCE_MS = 0.05    # the lanes and the report may differ by this much, or by RECONCILE_RELATIVE of a lane,
RECONCILE_RELATIVE = 5e-3        # for device events that come out of order by a microsecond or two (a misbooking is
                                 # a whole span, far above that)
# The recorder stamps its events with a device event record; the profiler lists such a task on the stream, so the
# stamps are anchors that pin the clocks together to a few microseconds.
ANCHOR_US = 15.0                 # a stamp and a record task this close are one event
ANCHOR_HYPOTHESES = 3            # the first stamps, each tried against every record task of the window
ANCHOR_FIRST, ANCHOR_FIRST_NEED = 8, 6   # a candidate offset must first match 6 of 8 stamps
ANCHOR_PROBE, ANCHOR_PROBE_SHARE = 40, 0.6
ANCHOR_SAMPLE = 400
MIN_STAMPS = 20
MIN_ANCHORED = 0.4               # share of the stamps that must coincide with a record task for the anchors to count
PARTIAL_ANCHORED = 0.9           # below this the match is partial: clocks drifting over the step, or foreign tasks
MATCH_US = 25.0                  # a stamp is moved onto the record task nearest to it if that is this close
DESCRIBED_STEPS = 3              # profiled steps whose numbers are printed (the last ones)


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
        """Append a slice if it has a length (even a sub-microsecond one, so that the lanes add up exactly)."""
        if end > start:
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


def reconcile_tolerance(mine: float, theirs: float) -> float:
    """Return how far apart a lane and the report's component may be before the two bookings are called different."""
    return max(RECONCILE_TOLERANCE_MS, RECONCILE_RELATIVE * max(abs(mine), abs(theirs)))


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


def ep_group_waits(hetero_dir: str, rank: int, step: int, ep_size: int) -> Optional[dict[str, Any]]:
    """Return the report's floor and waiting of one rank's MoE exchange, from the records of its whole EP group.

    ``analyze_hetero`` defines them per micro-batch and layer: a rank's exchange (the MoE block's span minus the
    experts', over the passes) minus the smallest exchange among the ``ep_size`` consecutive ranks of its group is the
    time it waited for the slowest rank of the group, and the smallest is the floor, the exchange with nobody to wait
    for. None when a rank of the group has no record of the step (then the group is not whole).
    """
    first = rank // ep_size * ep_size
    members = list(range(first, first + ep_size))
    headers, steps = {}, {}
    for member in members:
        loaded = load_rank(hetero_dir, member)
        record = next((r for r in (loaded[1] if loaded else []) if r["step"] == step and r["marks"]), None)
        if loaded is None or record is None:
            return None
        headers[member], steps[member] = loaded[0], [record]
    cells = exchange_cells(SimpleNamespace(ranks=members, headers=headers, steps=steps), ep_size)
    layers: dict[tuple[int, int], dict[str, Any]] = {}
    for (_step, occurrence, layer, _group), by_rank in cells.items():
        low = min(by_rank.values())
        layers[(occurrence, layer)] = {"exchange_ms": by_rank[rank], "floor_ms": low, "wait_ms": by_rank[rank] - low,
                                       "last_to_arrive": min(by_rank, key=lambda member: by_rank[member])}
    if not layers:
        return None
    return {"ep_size": ep_size, "layers": layers,
            "exchange_ms": sum(cell["exchange_ms"] for cell in layers.values()),
            "floor_ms": sum(cell["floor_ms"] for cell in layers.values()),
            "wait_ms": sum(cell["wait_ms"] for cell in layers.values())}


def annotate_exchange(lanes: dict[str, list[Piece]], waits: dict[str, Any]) -> None:
    """Put the layer's floor and waiting in the arguments of every exchange slice of that layer."""
    for piece in lanes["exchange"]:
        cell = waits["layers"].get((piece.args.get("micro_batch"), piece.args.get("layer")))
        if cell is not None:
            piece.args.update(layer_exchange_ms=round(cell["exchange_ms"], 3),
                              layer_floor_ms=round(cell["floor_ms"], 3), layer_wait_ms=round(cell["wait_ms"], 3),
                              last_to_arrive_rank=cell["last_to_arrive"])


# -- finding where the recorder's clock starts on the trace's timeline -------------------------------------

class Aligner:
    """Scores an offset by how well kernels sit inside the slices that carry their name.

    The score is the mean share of each kernel class (expert GEMMs, attention) that falls in its slices, times the
    share of the slices that the compute stream keeps busy. Recall alone has a flat top (a shift that keeps the
    kernels inside slightly longer slices costs nothing); the busy share is what makes the peak sharp.
    """

    def __init__(self, lanes: dict[str, list[Piece]], signals: Signals) -> None:
        """Take the slices in microseconds, for the kernel classes that the trace has."""
        self.signals = signals
        self.classes: dict[str, list[tuple[float, float]]] = {}
        for name, names in ALIGN_CLASSES:
            if name in signals.by_class:
                spans = [(p.start * 1000.0, p.end * 1000.0) for lane in names for p in lanes[lane]]
                if spans:
                    self.classes[name] = spans
        self.span_us = sum(b - a for spans in self.classes.values() for a, b in spans)

    def usable(self) -> bool:
        """Return whether there is a kernel class to align on."""
        return bool(self.classes)

    def denominators(self, low: float, high: float) -> dict[str, float]:
        """Return each class's kernel time inside [low, high], the reference for the shares."""
        return {name: self.signals.by_class[name].overlap(low, high) for name in self.classes}

    def score(self, offset: float, totals: dict[str, float]) -> float:
        """Return the alignment score of an offset (microseconds on the trace's timeline)."""
        recalls_, covered = [], 0.0
        for name, spans in self.classes.items():
            busy = self.signals.by_class[name]
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
    return low, max(window[1] - step_ms * 1000.0 + MARGIN_AFTER_US, low + MIN_SEARCH_US)


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


@dataclass
class Placement:
    """A step record whose stamps were moved onto the profiler's timestamps, and how far they moved."""

    record: dict
    matched: int
    stamps: int
    median_us: float
    worst_us: float
    drift_us_per_s: float


def place_on_profile(record: dict, signals: Signals, offset: float) -> Placement:
    """Move every stamp of a step record onto the timestamp of its own event-record task in the trace.

    A stamp predicted at ``offset + t`` is matched to the record task nearest to it, within ``MATCH_US``. Stamps
    without a task take the shift of their neighbours. The shifts say whether the recorder's clock and the profiler's
    agree: a median near zero, a small worst case and no drift over the step mean they do.
    """
    marks = record["marks"]
    tasks = signals.records
    predicted = [offset + mark[4] * 1000.0 for mark in marks]
    shifts: list[Optional[float]] = []
    for time_us in predicted:
        index = bisect.bisect_left(tasks, time_us)
        near = [task - time_us for task in tasks[max(index - 1, 0):index + 1] if abs(task - time_us) <= MATCH_US]
        shifts.append(min(near, key=abs) if near else None)
    matched = [(time_us, shift) for time_us, shift in zip(predicted, shifts) if shift is not None]
    if not matched:
        return Placement(record, 0, len(marks), 0.0, 0.0, 0.0)
    filled: list[float] = []
    previous: Optional[int] = None
    following = [None] * len(shifts)
    upcoming: Optional[int] = None
    for index in range(len(shifts) - 1, -1, -1):
        if shifts[index] is not None:
            upcoming = index
        following[index] = upcoming
    for index, shift in enumerate(shifts):
        if shift is not None:
            previous = index
            filled.append(shift)
            continue
        after = following[index]
        if previous is None:
            filled.append(shifts[after])
        elif after is None:
            filled.append(shifts[previous])
        else:
            span = predicted[after] - predicted[previous]
            weight = (predicted[index] - predicted[previous]) / span if span > 0 else 0.0
            filled.append(shifts[previous] * (1 - weight) + shifts[after] * weight)
    values = sorted(shift for _, shift in matched)
    drift = 0.0
    if len(matched) >= 5:
        mean_time = sum(time_us for time_us, _ in matched) / len(matched)
        mean_shift = sum(shift for _, shift in matched) / len(matched)
        variance = sum((time_us - mean_time) ** 2 for time_us, _ in matched)
        if variance > 0:
            drift = sum((time_us - mean_time) * (shift - mean_shift) for time_us, shift in matched) / variance * 1e6
    moved = [[*mark[:4], round(mark[4] + shift / 1000.0, 6), *mark[5:]] for mark, shift in zip(marks, filled)]
    return Placement({**record, "marks": moved}, len(matched), len(marks), values[len(values) // 2],
                     max(abs(value) for value in values), drift)


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
        busy = signals.all_to_all if kind == "alltoall" else signals.by_class.get(kind)
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


# -- the inputs of one command ----------------------------------------------------------------------------

@dataclass
class Job:
    """What one run of the command works from."""

    args: argparse.Namespace
    traces: dict[int, str]
    hetero_dir: Optional[str]
    rules: list
    out_dir: str
    full_dir: str


def has_records(directory: str) -> bool:
    """Return whether a directory holds hetero_profile record files."""
    return os.path.isdir(directory) and bool(glob.glob(os.path.join(directory, "rank*.jsonl")))


def resolve(args: argparse.Namespace) -> Job:
    """Work out the traces, the records and the output directories from the path given."""
    path = os.path.abspath(args.path)
    rules = parse_rules(args.class_rules)
    if os.path.isfile(path):
        rank = args.rank[0] if args.rank else trace_rank(path)
        traces = {rank if rank is not None else 0: path}
        profile = next((parent for parent in Path(path).parents if parent.name == "profile"), None)
        base = str(profile) if profile is not None else os.path.dirname(path)
        guess = os.path.join(os.path.dirname(base), "hetero") if profile is not None else ""
        hetero = args.hetero_dir or (guess if has_records(guess) else None)
        out_dir, full_dir = os.path.join(base, "components"), os.path.join(base, "components_full")
    else:
        traces = find_rank_traces(path)
        hetero = args.hetero_dir or next((d for d in (os.path.join(path, "hetero"), path) if has_records(d)), None)
        base = os.path.join(path, "profile") if (os.path.isdir(os.path.join(path, "profile")) or has_records(
            os.path.join(path, "hetero"))) else path
        out_dir, full_dir = os.path.join(base, "components"), os.path.join(base, "components_full")
    return Job(args, {} if args.no_trace else traces, hetero, rules, args.out or out_dir, args.full_out or full_dir)


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
        out.append("    event-record anchors (record step <-> profiler step: share of stamps matched): " + ", ".join(
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
        out.append("    no record step pairs with a profiler step, or the trace has no expert GEMM or attention kernel "
                   "(see the classes above; --class adds a rule)")
        return None
    scanned.sort(key=lambda item: -item[0])
    out.append("    no event-record anchors" + ("" if not anchors else f" (best {anchors[0][0]:.0%})")
               + "; kernel alignment, pairings (record step <-> profiler step: coarse score): " + ", ".join(
                   f"{step}<->{number}: {value:.2f}" for value, step, number, *_ in scanned[:8]))
    fits = [(refine(aligner, totals, scores, grid), step, number) for _, step, number, aligner, totals, scores, grid
            in scanned[:2]]
    fit, step, number = max(fits, key=lambda item: item[0].score)
    return Choice(step, number, fit.offset_us, fit=fit)


def hook_part(rank: int, job: Job, signals: Optional[Signals], out: list[str], finding: dict[str, Any]
              ) -> Optional[tuple[dict[str, list[Piece]], float]]:
    """Align and check the hooks' lanes of one rank; return the lanes to draw and where they start (microseconds)."""
    args, hetero_dir = job.args, job.hetero_dir
    loaded = load_rank(hetero_dir, rank) if hetero_dir else None
    if loaded is None:
        out.append(f"  hooks: no record file for rank {rank}" + (f" in {hetero_dir}" if hetero_dir else ""))
        return None
    header, records = loaded
    by_step = {record["step"]: record for record in records if record["marks"]}
    if not by_step:
        out.append("  hooks: no step with module boundaries (was hetero_profile.hooks off?)")
        return None
    choice: Optional[Choice] = None
    number, offset = -1, 0.0
    out.append("  THE HOOKS (hetero_profile)" + (" AGAINST THE PROFILE" if signals is not None else ""))
    if signals is None:
        step = args.step if args.step is not None else max(by_step)
        out.append("    no Ascend trace for this rank: the lanes alone, on the recorder's clock")
    elif args.offset_ms is not None:
        step = args.step if args.step is not None else max(by_step)
        offset = args.offset_ms * 1000.0
        out.append(f"    the step starts at {args.offset_ms} ms on the trace, as given")
    else:
        choice = choose(by_step, header, signals, args, out)
        if choice is None:
            return None
        step, number, offset = choice.step, choice.profiler_step, choice.offset_us
    if step not in by_step:
        out.append(f"    no step {step} with module boundaries; steps are {sorted(by_step)}")
        return None
    record = by_step[step]
    step_ms = record["device_ms"]
    lanes_report = build_lanes(spans_of(header, record), step_ms, step)
    spent = lane_ms(lanes_report)
    finding.update(step=step, profiler_step=number, step_ms=step_ms, lanes_ms=spent,
                   synthetic=header.get("time_source") == "synthetic")
    out.append(f"    record step {step}, {step_ms:.1f} ms of device time; the lanes partition it:")
    for lane in PARTITION:
        if spent[lane]:
            out.append(f"      {lane:10s} {spent[lane]:10.1f} ms  {spent[lane] / step_ms:6.1%}")
    certain = choice is None
    lanes = lanes_report
    if signals is not None and choice is not None:
        if choice.anchor is not None:
            certain = True
            finding.update(anchored=choice.anchor.share)
            placement = place_on_profile(record, signals, offset)
            lanes = build_lanes(spans_of(header, placement.record), step_ms, step)
            aligner = Aligner(lanes, signals)
            score = (aligner.score(offset, aligner.denominators(offset, offset + step_ms * 1000.0))
                     if aligner.usable() else 0.0)
            partial = "" if choice.anchor.share >= PARTIAL_ANCHORED else (
                " (partial: the clocks drift over the step, or some tasks are not the recorder's)")
            out.append(f"    the step starts at {offset / 1000.0:.3f} ms on the trace (profiler step {number}), found "
                       f"on the recorder's own event-record tasks: {choice.anchor.share:.0%} of its "
                       f"{choice.anchor.stamps} stamps coincide with one within {ANCHOR_US:g} us{partial}; the "
                       f"kernels score {score:.2f} there")
            finding.update(placement={"matched": placement.matched, "stamps": placement.stamps,
                                      "median_shift_us": placement.median_us, "worst_shift_us": placement.worst_us,
                                      "drift_us_per_s": placement.drift_us_per_s})
            out.append(f"    each stamp moved onto its own record task: {placement.matched} of {placement.stamps} "
                       f"matched, median shift {placement.median_us:+.1f} us, largest {placement.worst_us:.1f} us, "
                       f"drift {placement.drift_us_per_s:+.2f} us per second of step")
        elif choice.fit is not None:
            certain = choice.fit.clear
            finding.update(alignment=choice.fit.score, elsewhere=choice.fit.elsewhere)
            out.append(f"    the step starts at {offset / 1000.0:.3f} ms on the trace (profiler step {number}), found "
                       f"on the kernels alone: score {choice.fit.score:.2f}, best elsewhere {choice.fit.elsewhere:.2f}"
                       "; the lanes are placed by that offset, not stamp by stamp")
    waits = ep_group_waits(hetero_dir, rank, step, args.ep_size)
    group = f"{rank // args.ep_size * args.ep_size}-{rank // args.ep_size * args.ep_size + args.ep_size - 1}"
    if waits is None:
        out.append(f"    EP wait not computed: a rank of the group of {args.ep_size} (ranks {group}) has no record of "
                   f"step {step}, or there are no expert spans (--ep-size is the ranks per EP group)")
    else:
        annotate_exchange(lanes, waits)
        last = Counter(cell["last_to_arrive"] for cell in waits["layers"].values()).most_common(3)
        finding["ep_wait"] = {key: waits[key] for key in ("ep_size", "exchange_ms", "floor_ms", "wait_ms")}
        share = waits["wait_ms"] / max(waits["exchange_ms"], 1e-9)
        out.append(f"    EP wait (the report's definition: a layer's exchange above the smallest in the group of "
                   f"{args.ep_size}, ranks {group}): exchange {waits['exchange_ms']:.0f} ms = floor "
                   f"{waits['floor_ms']:.0f} ms + waiting {waits['wait_ms']:.0f} ms ({share:.0%})")
        out.append("    last to arrive, the rank that waits least, most often: " + ", ".join(
            f"rank {member} ({count} of {len(waits['layers'])} layers)" for member, count in last))
    pairs = reconcile(header, record, lanes_report)
    worst = max((abs(mine - theirs) for mine, theirs in pairs.values()), default=0.0)
    different = {lane: (mine, theirs) for lane, (mine, theirs) in pairs.items()
                 if abs(mine - theirs) > reconcile_tolerance(mine, theirs)}
    finding["reconcile"] = {lane: {"lane_ms": mine, "report_ms": theirs} for lane, (mine, theirs) in pairs.items()}
    out.append("    the lanes against the report's components (analyze_hetero): " + (
        f"the same milliseconds (largest difference {worst:.4f} ms)" if not different else
        "DIFFERENT, " + ", ".join(f"{lane} {mine:.3f} vs {theirs:.3f} ms" for lane, (mine, theirs)
                                  in different.items())))
    if signals is None:
        return lanes, 0.0
    shape = composition(lanes, signals, offset)
    checks = recalls(lanes, signals, offset, step_ms)
    waiting = hook_order(shape)
    finding.update(offset_us=offset, composition=shape, checks=checks, compute_wait_share=waiting)
    for lane in PARTITION:
        if lane not in shape:
            continue
        entry = shape[lane]
        kernels = ", ".join(f"{name} {share:.0%}" for name, share in list(entry["kernels"].items())[:3]) or "none"
        comms = ", ".join(f"{name} {ms:.1f} ms" for name, ms in list(entry["collectives_ms"].items())[:2]) or "none"
        out.append(f"      {lane:10s} stream busy {entry['busy']:4.0%} waiting {entry['wait']:4.0%} idle "
                   f"{entry['idle']:4.0%} | kernels: {kernels} | collectives: {comms}")
    for check in checks:
        share = check["share"]
        out.append(f"      {check['what']}: " + ("n/a (none in the step)" if share is None else f"{share:.1%}"))
    if waiting is not None:
        out.append(f"      stream waits inside the attention, expert and vision slices: {waiting:.1%} of their time")
    finding["verdict"] = verdict(checks, certain, waiting)
    out.append(f"    {finding['verdict']}")
    return lanes, offset


def profile_part(capture: Capture, signals: Signals, job: Job, out: list[str], finding: dict[str, Any]
                 ) -> dict[str, list[Piece]]:
    """Describe what the trace holds and each profiled step from the profile alone; return the lanes to draw."""
    out.append("  THE PROFILE ALONE (no hook record used)")
    out.extend("    " + line for line in inventory(capture, signals))
    steps = signals.steps[-DESCRIBED_STEPS:] or [(-1, signals.extent[0], signals.extent[1])]
    finding["profile"] = []
    for number, start, end in steps:
        numbers = step_numbers(signals, number, start, end)
        finding["profile"].append(numbers)
        out.extend("    " + line for line in describe_step(numbers))
    return profile_lanes(signals, job.args.merge_us, job.args.idle_us)


def process_rank(rank: int, job: Job, out: list[str]) -> Optional[dict[str, Any]]:
    """Analyse one rank, write its trace file, and return its findings; None if there is nothing to draw."""
    args = job.args
    out.append(f"rank {rank}:")
    finding: dict[str, Any] = {"rank": rank}
    capture = signals = None
    if rank in job.traces:
        capture = load_capture(job.traces[rank])
        signals = read_signals(capture.trace, job.rules)
    profile = profile_part(capture, signals, job, out, finding) if capture and signals else None
    hooked = hook_part(rank, job, signals, out, finding) if job.hetero_dir else None
    if profile is None and hooked is None:
        out.append("  nothing to draw: no trace and no hooks record for this rank")
        return None
    tag = "SYNTHETIC " if finding.get("synthetic") else ""
    new_events: list[dict[str, Any]] = []
    pids = free_pids(capture.events if capture else [], 2, PROCESS_BASE)
    if hooked is not None:
        lanes, offset = hooked
        placed = "placed on the profiler's timestamps" if capture and finding.get("anchored") else (
            "placed by an offset" if capture else "on the recorder's clock")
        new_events += process_events(
            pids[0], f"{tag}rank {rank} | components from the hooks (hetero_profile), {placed}", -1000, LANES, lanes,
            offset, 1000.0)
    if profile is not None:
        new_events += process_events(pids[1], f"rank {rank} | components from the profile alone (kernel classes, "
                                              "stream state, collectives)", -999, PROFILE_LANES, profile)
    if capture is not None and args.original:
        window = None
        name = f"rank{rank}_trace_with_components"
        if args.window_ms is not None:
            start = finding.get("offset_us", signals.extent[0]) + args.window_ms[0] * 1000.0
            window = (start, start + args.window_ms[1] * 1000.0)
            new_events = clip_events(new_events, window)
            name += f"_at{args.window_ms[0]:.0f}ms_for{args.window_ms[1]:.0f}ms"
        path = os.path.join(job.full_dir, f"{name}.json")
        size = write_integrated(capture, new_events, path, window)
        processes = int(hooked is not None) + int(profile is not None)
        kept = ("the whole trace" if window is None else
                f"{args.window_ms[1]:.0f} ms of the step, {args.window_ms[0]:.0f} ms after its start")
        out.append(f"  wrote {path}: {size / 2 ** 20:,.1f} MiB, {kept}, the original events untouched, then "
                   f"{len(new_events):,} new ones in {processes} process(es)")
        finding["trace_file"] = path
    elif capture is None and hooked is not None:
        path = os.path.join(job.out_dir, f"rank{rank}_lanes.json")
        os.makedirs(job.out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({"traceEvents": new_events, "displayTimeUnit": "ms"}, stream)
        out.append(f"  wrote {path} ({len(new_events):,} events: the lanes alone)")
        finding["trace_file"] = path
    return finding


# -- the command ------------------------------------------------------------------------------------------

def leaf_work(header: dict, record: dict) -> float:
    """Return the compute a rank's step holds: the vision tower, the attention, the experts, the head and loss."""
    spent = lane_ms(build_lanes(spans_of(header, record), record["device_ms"], record["step"]))
    return sum(spent[lane] for lane in ("vision", "attention", "experts", "head"))


def pick_ranks(job: Job, out: list[str]) -> list[int]:
    """Return the ranks to draw: with records the idlest and the busiest that have a trace, else the first trace."""
    if not job.hetero_dir:
        chosen = sorted(job.traces)[:1]
        out.append(f"no hetero records: the profile alone for rank {chosen[0] if chosen else '-'} "
                   f"of {len(job.traces)} trace(s); --rank chooses others")
        return chosen
    local = ranks_with_records(job.hetero_dir)
    candidates = [rank for rank in local if rank in job.traces] or local
    scored = []
    for rank in candidates:
        loaded = load_rank(job.hetero_dir, rank)
        hooked = [record for record in (loaded[1] if loaded else []) if record["marks"]]
        if hooked:
            scored.append((leaf_work(loaded[0], hooked[-1]), rank))
    scored.sort()
    chosen = sorted({scored[0][1], scored[-1][1]}) if scored else candidates[:1]
    if len(scored) > 1:
        out.append(f"ranks drawn, among {len(candidates)}: the idlest, rank {scored[0][1]} ({scored[0][0]:.0f} ms of "
                   f"vision, attention, expert and head work in the last recorded step), and the busiest, rank "
                   f"{scored[-1][1]} ({scored[-1][0]:.0f} ms); one trace file each")
    else:
        out.append(f"rank drawn: {chosen}")
    return chosen


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="a run directory (hetero/ records and profile/ traces), a directory of traces, "
                                     "or one trace_view.json")
    parser.add_argument("--hetero-dir", default=None, help="the hetero_profile records (default: <run>/hetero)")
    parser.add_argument("--rank", type=int, nargs="+", default=None,
                        help="ranks to process (default: with records the idlest and the busiest that have a trace, "
                             "else the first trace)")
    parser.add_argument("--class", dest="class_rules", action="append", default=[], metavar="CLASS=REGEX",
                        help="classify the kernels whose name matches REGEX as CLASS (attention, experts, routing, "
                             "dense, norm, other), before the defaults; repeatable")
    parser.add_argument("--step", type=int, default=None, help="the record step (default: found by alignment)")
    parser.add_argument("--profiler-step", type=int, default=None, help="the ProfilerStep number to pair with")
    parser.add_argument("--ep-size", type=int, default=16, help="ranks per expert-parallel group (for the EP wait)")
    parser.add_argument("--offset-ms", type=float, default=None,
                        help="where the record step starts on the trace's timeline, instead of searching (one --rank)")
    parser.add_argument("--search-all", action="store_true",
                        help="search the whole trace for the step, not the profiler step's range (slower)")
    parser.add_argument("--merge-us", type=float, default=100.0,
                        help="kernels of one class this close are drawn as one slice in the profile lanes")
    parser.add_argument("--idle-us", type=float, default=20.0,
                        help="a gap on the compute stream this long is drawn as idle in the profile lanes")
    parser.add_argument("--window-ms", type=float, nargs=2, default=None, metavar=("OFFSET", "LENGTH"),
                        help="write only this stretch of the step into the trace file: OFFSET ms after the step's "
                             "start, LENGTH ms long. A trace of gigabytes becomes a file a viewer opens at once; "
                             "without it the whole trace is written")
    parser.add_argument("--out", default=None, help="small files: report and summary (<run>/profile/components)")
    parser.add_argument("--full-out", default=None,
                        help="large files: the trace of each rank with the components (<run>/profile/components_full)")
    parser.add_argument("--no-trace", action="store_true", help="ignore the traces: the hooks' lanes alone")
    parser.add_argument("--no-original", dest="original", action="store_false",
                        help="do not write the trace files (report only)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the command line, process the ranks, write the trace files and print the findings."""
    args = parse_args(argv)
    try:
        job = resolve(args)
    except ValueError as error:
        print(error)
        return 2
    if not job.traces and not job.hetero_dir:
        print(f"no trace_view.json and no rank*.jsonl under {args.path}: pass a run directory (it holds hetero/ and "
              "profile/), a directory of traces, or one trace_view.json")
        return 1
    out: list[str] = []
    ranks = args.rank if args.rank is not None else pick_ranks(job, out)
    findings = []
    for rank in ranks:
        finding = process_rank(rank, job, out)
        if finding is not None:
            findings.append(finding)
    print("\n".join(out))
    if not findings:
        return 1
    os.makedirs(job.out_dir, exist_ok=True)
    with open(os.path.join(job.out_dir, "components_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(findings, stream, indent=2, default=str)
    with open(os.path.join(job.out_dir, "components.txt"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(out) + "\n")
    print(f"report: {os.path.join(job.out_dir, 'components.txt')} (and components_summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
