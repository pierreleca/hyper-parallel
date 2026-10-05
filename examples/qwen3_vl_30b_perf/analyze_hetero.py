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
"""Explain a run's step time by the heterogeneity of the model and of the data.

Reads the per-rank files of ``hyper_parallel.trainer.runtime.hetero_profile``
(``<run>/hetero``) and reports

- **data:** how much the samples of one step differ, in tokens, images and image
  size, and how the ranks' work follows from them;
- **model:** where the time of a micro-batch goes (vision tower, decoder
  attention, MoE blocks, vocabulary projection and loss), in the forward pass,
  the recompute and the backward pass, per layer, and the gaps between modules
  (the wait for the weights, the exposed communication);
- **cost model:** a least-squares fit of each component's time on the features of
  the sample (tokens, patches, vision attention), and how well the data explains
  the time;
- **imbalance:** per step, how far the slowest rank is above the mean, which
  component the excess comes from, and how much of it the data predicts;
- **what-ifs:** the step time if the samples were grouped by cost, and how evenly
  the model's components can be cut into pipeline stages;
- **routing:** per MoE layer, whether the image tokens and the text tokens choose
  the same experts, and which of them overloads an expert-parallel rank.

It needs nothing but the Python standard library, so it runs on a control node.

    python examples/qwen3_vl_30b_perf/analyze_hetero.py <run dir>        # <run>/hetero
    python examples/qwen3_vl_30b_perf/analyze_hetero.py --sweep <run dir> <run dir> ...

A span of a module is its own work and any collective inside it; the wait for its
weights, and the exposed communication between modules, are the gaps. See the
header of ``hetero_profile.py`` for where the boundaries sit.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from typing import Any, Optional, Sequence

PHASES = ("fwd", "recompute", "bwd")
# Component -> the roles whose spans add up to it.
GROUPS: dict[str, tuple[str, ...]] = {
    "vision": ("vision.patch_embed", "vision.block", "vision.deepstack", "vision.merger"),
    "text_layer": ("text.layer",),
    "text_attn": ("text.attn",),
    "text_moe": ("text.moe",),
    "embed": ("text.embed",),
    "head": ("lm_head",),
}
# What adds up to a micro-batch's work: no component here contains another.
WORK_PARTS = ("vision", "text_layer", "embed", "head", "loss")
FEATURES_VISION = ("patches", "vision_attn_pairs")
FEATURES_TEXT = ("real_tokens",)
MS_PER_TOKEN_UNIT = 1e3


# -- loading ----------------------------------------------------------------------------------------------

class Run:
    """The per-rank records of one run."""

    def __init__(self, directory: str, skip: int = 0) -> None:
        """Read every ``rank*.jsonl`` under ``directory``, dropping the first ``skip`` steps of each rank."""
        paths = sorted(glob.glob(os.path.join(directory, "rank*.jsonl")))
        if not paths:
            raise SystemExit(f"no rank*.jsonl files in {directory}")
        self.directory = directory
        self.headers: dict[int, dict] = {}
        self.steps: dict[int, list[dict]] = {}
        for path in paths:
            with open(path, encoding="utf-8") as stream:
                lines = [json.loads(line) for line in stream if line.strip()]
            if not lines or lines[0].get("kind") != "header":
                raise SystemExit(f"{path}: the first line is not a header")
            rank = int(lines[0]["rank"])
            self.headers[rank] = lines[0]
            self.steps[rank] = [line for line in lines[1:] if line.get("kind") == "step"][skip:]
        self.ranks = sorted(self.headers)

    @property
    def name(self) -> str:
        """The run's name: its directory, or the one above a ``hetero`` directory."""
        path = os.path.abspath(self.directory)
        return os.path.basename(os.path.dirname(path)) if os.path.basename(path) == "hetero" else os.path.basename(path)


def spans_of(record: dict) -> dict[tuple[int, str, int], dict[str, float]]:
    """Return {(module id, pass, occurrence): {"in": ms, "out": ms}} for one step record.

    ``in`` and ``out`` are the entry and exit of the module in the forward and recompute passes, and
    the arrival of the output's gradient and the production of the input's in the backward pass.
    """
    spans: dict[tuple[int, str, int], dict[str, float]] = {}
    for module_id, pass_name, occurrence, kind, time_ms, *_ in record["marks"]:
        spans.setdefault((module_id, pass_name, occurrence), {})[kind] = time_ms
    return spans


def _span_ms(spans: dict, module_id: int, pass_name: str, occurrence: int) -> Optional[float]:
    """Return the duration of one span, or None when either end was not recorded."""
    span = spans.get((module_id, pass_name, occurrence))
    if span is None or "in" not in span or "out" not in span:
        return None
    return span["out"] - span["in"]


def _allocated(record: dict, module_id: int, pass_name: str, occurrence: int, kind: str) -> Optional[int]:
    """Return the allocated bytes at one boundary, or None."""
    for mark in record["marks"]:
        if mark[0] == module_id and mark[1] == pass_name and mark[2] == occurrence and mark[3] == kind:
            return mark[5] if len(mark) > 5 else None
    return None


def build_rows(run: Run) -> list[dict[str, Any]]:
    """Return one row per rank, step and micro-batch: its workload, and the time of each component.

    ``t[component][phase]`` is the summed duration in milliseconds of the component's modules in the phase;
    ``gap[sequence][phase]`` is the idle time between consecutive modules of the vision blocks and of the
    decoder layers; ``loss`` runs from the end of the model's forward to the gradient reaching the logits.
    """
    rows = []
    for rank in run.ranks:
        modules = {m["id"]: m for m in run.headers[rank]["modules"]}
        by_role: dict[str, list[dict]] = {}
        for module in modules.values():
            by_role.setdefault(module["role"], []).append(module)
        for role_modules in by_role.values():
            role_modules.sort(key=lambda m: (m["index"] is None, m["index"] or 0, m["id"]))
        root = (by_role.get("root") or [None])[0]
        for record in run.steps[rank]:
            spans = spans_of(record)
            tail_start = max((mark[4] for mark in record["marks"]), default=0.0)
            for occurrence, batch in enumerate(record["micro_batches"]):
                row: dict[str, Any] = {
                    "rank": rank, "step": record["step"], "mb": occurrence, "step_ms": record["device_ms"],
                    "wall_ms": record["wall_ms"], "inter_step_ms": record.get("inter_step_ms"),
                    "peak_allocated": record.get("peak_allocated"), "tail_ms": record["device_ms"] - tail_start,
                }
                row.update({key: value for key, value in batch.items() if isinstance(value, (int, float))})
                row["t"] = {group: {phase: _sum_spans(spans, by_role, roles, phase, occurrence) for phase in PHASES}
                            for group, roles in GROUPS.items()}
                row["gap"] = {
                    "vision": _gaps(spans, by_role.get("vision.block", []), occurrence),
                    "text": _gaps(spans, by_role.get("text.layer", []), occurrence),
                }
                row["t"]["loss"] = {"fwd": 0.0, "recompute": 0.0, "bwd": _loss_ms(spans, root, by_role, occurrence)}
                row["wall_fwd"] = _root_span(spans, by_role, "vision.root", occurrence), _root_span(
                    spans, by_role, "text.root", occurrence)
                if root is not None:
                    row["mb_ms"] = _mb_ms(record, root["id"], occurrence)
                row["act"] = _activation_bytes(record, by_role, occurrence)
                rows.append(row)
    return rows


def _sum_spans(spans: dict, by_role: dict, roles: Sequence[str], phase: str, occurrence: int) -> float:
    """Sum the durations of every module of the roles in one phase of one micro-batch."""
    total = 0.0
    for role in roles:
        for module in by_role.get(role, []):
            value = _span_ms(spans, module["id"], phase, occurrence)
            if value is not None:
                total += value
    return total


def _gaps(spans: dict, blocks: list[dict], occurrence: int) -> dict[str, float]:
    """Idle milliseconds between consecutive blocks: forward in index order, backward in reverse order."""
    forward = backward = 0.0
    for earlier, later in zip(blocks, blocks[1:]):
        before = spans.get((earlier["id"], "fwd", occurrence), {})
        after = spans.get((later["id"], "fwd", occurrence), {})
        if "out" in before and "in" in after:
            forward += max(after["in"] - before["out"], 0.0)
        first = spans.get((later["id"], "bwd", occurrence), {})   # the later block's backward runs first
        second = spans.get((earlier["id"], "bwd", occurrence), {})
        if "out" in first and "in" in second:
            backward += max(second["in"] - first["out"], 0.0)
    return {"fwd": forward, "bwd": backward}


def _loss_ms(spans: dict, root: Optional[dict], by_role: dict, occurrence: int) -> float:
    """Milliseconds from the end of the model's forward to the gradient of the logits arriving."""
    heads = by_role.get("lm_head", [])
    if root is None or not heads:
        return 0.0
    forward = spans.get((root["id"], "fwd", occurrence), {})
    head = spans.get((heads[0]["id"], "bwd", occurrence), {})
    if "out" in forward and "in" in head:
        return max(head["in"] - forward["out"], 0.0)
    return 0.0


def _root_span(spans: dict, by_role: dict, role: str, occurrence: int) -> Optional[float]:
    """Return the forward duration of a tower's root module, which holds its glue and waits."""
    modules = by_role.get(role, [])
    return _span_ms(spans, modules[0]["id"], "fwd", occurrence) if modules else None


def _mb_ms(record: dict, root_id: int, occurrence: int) -> float:
    """Return the milliseconds from the start of a micro-batch's forward to its last backward boundary."""
    start = end = None
    for module_id, pass_name, occ, kind, time_ms, *_ in record["marks"]:
        if occ != occurrence:
            continue
        if module_id == root_id and pass_name == "fwd" and kind == "in":
            start = time_ms
        elif pass_name == "bwd":
            end = time_ms if end is None else max(end, time_ms)
    return (end - start) if start is not None and end is not None else 0.0


def _activation_bytes(record: dict, by_role: dict, occurrence: int) -> dict[str, float]:
    """Return the bytes each tower leaves allocated at the end of its forward pass."""
    result = {}
    for name, role in (("vision", "vision.root"), ("text", "text.root"), ("head", "lm_head")):
        modules = by_role.get(role, [])
        if not modules:
            continue
        before = _allocated(record, modules[0]["id"], "fwd", occurrence, "in")
        after = _allocated(record, modules[0]["id"], "fwd", occurrence, "out")
        if before is not None and after is not None:
            result[name] = float(after - before)
    return result


# -- small statistics -------------------------------------------------------------------------------------

def mean(values: Sequence[float]) -> float:
    """Return the mean, or 0 for an empty list."""
    return statistics.fmean(values) if values else 0.0


def quantile(values: Sequence[float], share: float) -> float:
    """Return the value at this share of the sorted list."""
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def cv(values: Sequence[float]) -> float:
    """Return the standard deviation over the mean."""
    average = mean(values)
    return statistics.pstdev(values) / average if average and len(values) > 1 else 0.0


def ranks_of(values: Sequence[float]) -> list[float]:
    """Return the ranks (1-based, ties averaged) of the values."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Pearson's correlation, or 0 when either list does not vary."""
    if len(first) < 2 or statistics.pstdev(first) == 0 or statistics.pstdev(second) == 0:
        return 0.0
    return statistics.correlation(first, second)


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman's rank correlation."""
    return pearson(ranks_of(first), ranks_of(second))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small linear system by Gauss-Jordan elimination with partial pivoting."""
    size = len(vector)
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            continue
        augmented[column] = [value / divisor for value in augmented[column]]
        for other in range(size):
            if other != column:
                factor = augmented[other][column]
                augmented[other] = [a - factor * b for a, b in zip(augmented[other], augmented[column])]
    return [augmented[i][size] for i in range(size)]


def fit(features: list[list[float]], targets: list[float]) -> dict[str, Any]:
    """Least-squares fit of ``target = c0 + sum(c_i * feature_i)``; columns that do not vary are dropped.

    Returns the coefficients (``intercept`` and one per kept column, by position), the R squared and
    the root-mean-square residual.
    """
    count = len(targets)
    columns = [index for index in range(len(features[0])) if statistics.pstdev([row[index] for row in features]) > 0]
    design = [[1.0] + [row[index] for index in columns] for row in features]
    size = len(columns) + 1
    normal = [[sum(row[i] * row[j] for row in design) + (1e-9 if i == j else 0.0) for j in range(size)]
              for i in range(size)]
    right = [sum(row[i] * target for row, target in zip(design, targets)) for i in range(size)]
    beta = solve(normal, right)
    predicted = [sum(b * x for b, x in zip(beta, row)) for row in design]
    total = sum((target - mean(targets)) ** 2 for target in targets)
    residual = sum((target - guess) ** 2 for target, guess in zip(targets, predicted))
    coefficients = [0.0] * (len(features[0]) + 1)
    coefficients[0] = beta[0]
    for position, column in enumerate(columns):
        coefficients[column + 1] = beta[position + 1]
    return {"coefficients": coefficients, "r2": 1.0 - residual / total if total > 0 else 1.0,
            "rmse": math.sqrt(residual / count) if count else 0.0, "n": count}


def predict(model: dict[str, Any], features: Sequence[float]) -> float:
    """Evaluate a fitted model on one feature vector."""
    coefficients = model["coefficients"]
    return coefficients[0] + sum(c * x for c, x in zip(coefficients[1:], features))


# -- cost model -------------------------------------------------------------------------------------------

def feature_vector(row: dict, component: str) -> list[float]:
    """Return the features a component's time is fitted on: patches for the vision tower, tokens for the rest.

    Tokens and patches are in thousands; the quadratic terms are in millions, which keeps the fit well conditioned.
    """
    tokens = row.get("real_tokens", 0) / 1e3
    if component == "vision":
        return [row.get("patches", 0) / 1e3, row.get("vision_attn_pairs", 0) / 1e6]
    if component in ("text_layer", "text_attn"):
        return [tokens, tokens * tokens]
    return [tokens]


def component_time(row: dict, component: str) -> float:
    """Return a component's time in the micro-batch: forward, recompute and backward together."""
    return sum(row["t"][component][phase] for phase in PHASES)


def fit_components(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Fit each component's time on the features of the micro-batch."""
    models = {}
    for component in ("vision", "text_layer", "text_attn", "text_moe", "embed", "head", "loss"):
        times = [component_time(row, component) for row in rows]
        if not any(times):
            continue
        models[component] = fit([feature_vector(row, component) for row in rows], times)
    return models


def modeled_cost(models: dict[str, dict], row: dict) -> float:
    """Return the work the cost model predicts for a micro-batch, in milliseconds."""
    return sum(predict(models[part], feature_vector(row, part)) for part in WORK_PARTS if part in models)


# -- reports ----------------------------------------------------------------------------------------------

def _by_step(rows: list[dict]) -> dict[int, dict[int, dict[str, Any]]]:
    """Group micro-batch rows into {step: {rank: summed row}}: a rank's work is the sum over its micro-batches."""
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        entry = grouped.setdefault(row["step"], {}).setdefault(row["rank"], {
            "work": 0.0, "parts": {part: 0.0 for part in WORK_PARTS}, "step_ms": row["step_ms"], "rows": [],
            "inter_step_ms": row["inter_step_ms"], "mb_ms": 0.0,
        })
        for part in WORK_PARTS:
            value = component_time(row, part)
            entry["parts"][part] += value
            entry["work"] += value
        entry["mb_ms"] += row.get("mb_ms", 0.0)
        entry["rows"].append(row)
    return grouped


def report_overview(run: Run, rows: list[dict], out: list[str]) -> dict[str, Any]:
    """The run: ranks, steps, micro-batches, step time and the host gap between steps."""
    steps = _by_step(rows)
    step_ms = [mean([entry["step_ms"] for entry in ranks.values()]) for ranks in steps.values()]
    gaps = [entry["inter_step_ms"] for ranks in steps.values() for entry in ranks.values()
            if entry["inter_step_ms"] is not None]
    micro = len({(row["step"], row["mb"]) for row in rows}) // max(len(steps), 1)
    tails = [row["tail_ms"] for row in rows]
    out.append(f"RUN {run.name}: {len(run.ranks)} ranks, {len(steps)} steps analysed, {micro} micro-batch(es) per rank "
               f"and step; time source {run.headers[run.ranks[0]].get('time_source')}")
    out.append(f"  step {mean(step_ms):9.1f} ms (steps {min(step_ms):.1f} to {max(step_ms):.1f}); after the last "
               f"backward boundary (grad clip, optimizer, sync) {mean(tails):.1f} ms; host gap between steps "
               f"{mean(gaps):.1f} ms (max {max(gaps, default=0.0):.1f})")
    return {"ranks": len(run.ranks), "steps": len(steps), "micro_batches": micro, "step_ms": mean(step_ms),
            "step_ms_cv": cv(step_ms), "tail_ms": mean(tails), "host_gap_ms": mean(gaps),
            "host_gap_max_ms": max(gaps, default=0.0)}


def report_data(rows: list[dict], out: list[str]) -> dict[str, Any]:
    """How much the samples differ: across all micro-batches, and across the ranks of one step."""
    summary: dict[str, Any] = {}
    steps: dict[int, dict[int, dict[str, float]]] = {}
    for row in rows:
        entry = steps.setdefault(row["step"], {}).setdefault(row["rank"], {})
        for key in ("real_tokens", "visual_tokens", "images", "patches", "vision_attn_pairs", "label_tokens"):
            entry[key] = entry.get(key, 0.0) + row.get(key, 0)
    out.append("DATA: per micro-batch (all ranks and steps), and the slowest-looking rank of a step against the mean")
    out.append("  feature             mean      cv      min      p50      p90      max   "
               "max/mean over ranks (mean of steps)")
    for key in ("real_tokens", "visual_tokens", "images", "patches", "vision_attn_pairs", "label_tokens"):
        values = [row.get(key, 0) for row in rows]
        factors = []
        for ranks in steps.values():
            per_rank = [entry[key] for entry in ranks.values()]
            if mean(per_rank) > 0:
                factors.append(max(per_rank) / mean(per_rank))
        out.append(f"  {key:18s} {mean(values):9.1f} {cv(values):6.2f} {min(values):8.0f} {quantile(values, .5):8.0f} "
                   f"{quantile(values, .9):8.0f} {max(values):8.0f}   {mean(factors):6.2f}")
        summary[key] = {"mean": mean(values), "cv": cv(values), "max_over_mean": mean(factors)}
    tokens = [row.get("real_tokens", 0) for row in rows]
    visual = [row.get("visual_tokens", 0) for row in rows]
    share = [v / t if t else 0.0 for v, t in zip(visual, tokens)]
    out.append(f"  visual share of the tokens: mean {mean(share):.2f}, cv {cv(share):.2f}; correlation of tokens and "
               f"visual tokens {pearson(tokens, visual):+.2f}")
    summary["visual_share_mean"] = mean(share)
    return summary


def report_components(rows: list[dict], out: list[str]) -> dict[str, Any]:
    """Where a micro-batch's time goes, by component and phase, and the cost per unit of work."""
    count = len(rows)
    total_mb = mean([row.get("mb_ms", 0.0) for row in rows])
    tokens = mean([row.get("real_tokens", 0) for row in rows]) or 1.0
    patches = mean([row.get("patches", 0) for row in rows]) or 1.0
    out.append("")
    out.append(f"MODEL: time of a micro-batch ({total_mb:.1f} ms from forward start to last backward boundary), "
               f"mean over {count} micro-batches; ms")
    out.append("  component        forward  recompute  backward    total   share  "
               "per 1k tokens (vision: per 1k patches)")
    summary: dict[str, Any] = {"mb_ms": total_mb}
    for component in ("vision", "text_layer", "text_attn", "text_moe", "embed", "head", "loss"):
        phase_means = {phase: mean([row["t"][component][phase] for row in rows]) for phase in PHASES}
        total = sum(phase_means.values())
        if total == 0:
            continue
        unit = patches if component == "vision" else tokens
        nested = "  (inside text_layer)" if component in ("text_attn", "text_moe") else ""
        out.append(f"  {component:14s} {phase_means['fwd']:9.1f} {phase_means['recompute']:10.1f} "
                   f"{phase_means['bwd']:9.1f} {total:9.1f} {total / (total_mb or 1.0):6.1%}  "
                   f"{total / unit * 1e3:9.2f}{nested}")
        summary[component] = {**phase_means, "total": total, "per_1k": total / unit * 1e3}
    gaps = {f"{kind} {phase}": mean([row["gap"][kind][phase] for row in rows])
            for kind in ("vision", "text") for phase in ("fwd", "bwd")}
    out.append("  idle between consecutive modules (the wait for weights, exposed communication, launch gaps): "
               + ", ".join(f"{name} {value:.1f}" for name, value in gaps.items()))
    summary["gaps"] = gaps
    accounted = sum(summary[c]["total"] for c in WORK_PARTS if c in summary) + sum(gaps.values())
    out.append(f"  components + gaps account for {accounted:.1f} ms of {total_mb:.1f} ms; the rest is glue between "
               "the towers, the loss inputs and the optimizer-free parts of the step")
    wall = [row["wall_fwd"] for row in rows if row["wall_fwd"][0] is not None]
    if wall:
        out.append(f"  forward wall of the vision tower {mean([w[0] for w in wall]):.1f} ms and of the decoder "
                   f"{mean([w[1] or 0.0 for w in wall]):.1f} ms, including their glue and gaps")
    return summary


def report_layers(run: Run, rows: list[dict], out: list[str], top: int) -> dict[str, Any]:
    """Per decoder layer and per vision block: forward, recompute and backward time, the slowest indices first."""
    del rows
    summary: dict[str, Any] = {}
    for title, roles in (("decoder layer", ("text.layer", "text.attn", "text.moe")),
                         ("vision block", ("vision.block",))):
        table = layer_table(run, roles)
        present = [role for role in roles if any(role in entry for entry in table.values())]
        if not present:
            continue
        lead = present[0]
        totals = {index: sum(entry.get(lead, {}).values()) for index, entry in table.items()}
        out.append("")
        out.append(f"{title.upper()}S: mean ms per micro-batch over ranks and steps, forward / recompute / backward of "
                   f"each index; the {top} slowest first")
        out.append("  index  " + "  ".join(f"{role:>24s}" for role in present))
        for index in sorted(totals, key=lambda i: -totals[i])[:top]:
            cells = []
            for role in present:
                phases = table[index].get(role, {})
                cells.append(f"{phases.get('fwd', 0.0):6.2f}/{phases.get('recompute', 0.0):5.2f}/"
                             f"{phases.get('bwd', 0.0):6.2f}")
            out.append(f"  {index:5d}  " + "  ".join(f"{cell:>24s}" for cell in cells))
        ordered = [totals[index] for index in sorted(totals)]
        out.append(f"  {len(ordered)} indices; index 0 takes {ordered[0]:.2f} ms, the last {ordered[-1]:.2f} ms, "
                   f"mean {mean(ordered):.2f} ms, widest {max(ordered):.2f} ms")
        summary[title] = {"mean_ms": mean(ordered), "first_ms": ordered[0], "last_ms": ordered[-1],
                          "max_ms": max(ordered)}
    return summary


def layer_table(run: Run, roles: Sequence[str]) -> dict[int, dict[str, dict[str, float]]]:
    """Return {index: {role: {phase: mean ms}}} over the ranks, steps and micro-batches of the run."""
    sums: dict[tuple[int, str, str], list[float]] = {}
    for rank in run.ranks:
        modules = {m["id"]: m for m in run.headers[rank]["modules"] if m["role"] in roles and m["index"] is not None}
        for record in run.steps[rank]:
            spans = spans_of(record)
            for (module_id, pass_name, _occurrence), span in spans.items():
                if module_id in modules and pass_name in PHASES and "in" in span and "out" in span:
                    key = (modules[module_id]["index"], modules[module_id]["role"], pass_name)
                    sums.setdefault(key, []).append(span["out"] - span["in"])
    table: dict[int, dict[str, dict[str, float]]] = {}
    for (index, role, phase), values in sums.items():
        table.setdefault(index, {}).setdefault(role, {})[phase] = mean(values)
    return table


def report_fit(rows: list[dict], out: list[str]) -> dict[str, dict]:
    """How well the sample's features explain each component's time, and the fitted cost per feature."""
    models = fit_components(rows)
    out.append("")
    out.append("COST MODEL: component time (ms) = c0 + c1 * x1 + c2 * x2, fitted over all micro-batches. vision: x1 = "
               "patches (1k), x2 = vision attention pairs (1M); text parts: x1 = tokens (1k), x2 = tokens^2 (1M)")
    out.append("  component            c0       c1       c2     R^2   rmse ms")
    for component, model in models.items():
        c = model["coefficients"] + [0.0] * (3 - len(model["coefficients"]))
        out.append(f"  {component:14s} {c[0]:9.2f} {c[1]:8.2f} {c[2]:8.2f} {model['r2']:7.2f} {model['rmse']:8.2f}")
    explained = [row for row in rows if "mb_ms" in row]
    if explained and models:
        cost = [modeled_cost(models, row) for row in explained]
        work = [sum(component_time(row, part) for part in WORK_PARTS) for row in explained]
        out.append(f"  whole micro-batch: modeled work vs measured work, Pearson {pearson(cost, work):+.3f}, Spearman "
                   f"{spearman(cost, work):+.3f}")
    return models


def report_imbalance(rows: list[dict], models: dict[str, dict], out: list[str]) -> dict[str, Any]:
    """Per step: how far the busiest rank is above the mean, from which component, and what the data predicts."""
    steps = _by_step(rows)
    factors, excess_share, idle_share, predictable = [], [], [], []
    parts_excess = {part: 0.0 for part in WORK_PARTS}
    straggler_hits: dict[int, int] = {}
    excess_total = step_total = 0.0
    for ranks in steps.values():
        work = {rank: entry["work"] for rank, entry in ranks.items()}
        mean_work, top_rank = mean(list(work.values())), max(work, key=work.get)
        step_ms = mean([entry["step_ms"] for entry in ranks.values()])
        factors.append(work[top_rank] / mean_work if mean_work else 1.0)
        excess = work[top_rank] - mean_work
        excess_total += excess
        step_total += step_ms
        excess_share.append(excess / step_ms if step_ms else 0.0)
        idle_share.append(1.0 - mean_work / work[top_rank] if work[top_rank] else 0.0)
        straggler_hits[top_rank] = straggler_hits.get(top_rank, 0) + 1
        for part in WORK_PARTS:
            part_mean = mean([entry["parts"][part] for entry in ranks.values()])
            parts_excess[part] += ranks[top_rank]["parts"][part] - part_mean
        if models and len(ranks) > 2:
            modeled = [sum(modeled_cost(models, row) for row in ranks[rank]["rows"]) for rank in ranks]
            predictable.append(spearman(modeled, [work[rank] for rank in ranks]))
    out.append("")
    out.append("IMBALANCE: a rank's work is the sum of its modules' own time (waits for weights and gaps excluded)")
    out.append(f"  busiest rank / mean rank: {mean(factors):.3f} on average over the steps (worst step "
               f"{max(factors):.3f}); its excess is {mean(excess_share):.1%} of the step, i.e. "
               f"{mean(idle_share):.1%} of the ranks' compute time is spent waiting for it")
    total_excess = sum(parts_excess.values()) or 1.0
    out.append("  where the busiest rank's excess comes from: "
               + ", ".join(f"{part} {parts_excess[part] / total_excess:.0%}" for part in WORK_PARTS))
    if predictable:
        out.append(f"  the data predicts who is busiest: rank correlation of modeled and measured work within a step "
                   f"{mean(predictable):+.2f} (1 = the samples explain the whole order)")
    busiest = sorted(straggler_hits.items(), key=lambda item: -item[1])[:5]
    out.append("  most often the busiest rank (steps): "
               + ", ".join(f"rank {rank} x{count}" for rank, count in busiest))
    return {"busiest_over_mean": mean(factors), "busiest_over_mean_worst": max(factors),
            "excess_share_of_step": mean(excess_share), "idle_share": mean(idle_share),
            "excess_by_part": {part: parts_excess[part] / total_excess for part in WORK_PARTS},
            "predictability": mean(predictable) if predictable else None,
            "excess_ms_total": excess_total, "step_ms_total": step_total}


def bucketed_step_costs(costs: Sequence[float], group: int) -> float:
    """Sum, over steps of ``group`` samples, the cost of the costliest, when samples are sorted into steps."""
    ordered = sorted(costs)
    return sum(max(ordered[start:start + group]) for start in range(0, len(ordered) - group + 1, group))


def report_whatif_balancing(rows: list[dict], models: dict[str, dict], imbalance: dict[str, Any],
                            out: list[str]) -> dict[str, Any]:
    """The step time if the samples were grouped by cost, on the cost model; a ceiling the measured excess bounds."""
    if not models:
        return {}
    steps = _by_step(rows)
    ranks_per_step = max(len(ranks) for ranks in steps.values())
    per_rank = [sum(modeled_cost(models, row) for row in entry["rows"]) for ranks in steps.values()
                for entry in ranks.values()]
    actual_max = sum(max(sum(modeled_cost(models, row) for row in entry["rows"]) for entry in ranks.values())
                     for ranks in steps.values())
    actual_mean = sum(mean([sum(modeled_cost(models, row) for row in entry["rows"]) for entry in ranks.values()])
                      for ranks in steps.values())
    bucketed = bucketed_step_costs(per_rank, ranks_per_step)
    removable = max(actual_max - bucketed, 0.0) / (actual_max - actual_mean) if actual_max > actual_mean else 0.0
    saved_ms = removable * imbalance["excess_ms_total"]
    share = saved_ms / imbalance["step_ms_total"] if imbalance["step_ms_total"] else 0.0
    out.append("")
    out.append("WHAT IF the samples of a step were alike (sorted by modeled cost into steps of "
               f"{ranks_per_step} samples; same samples, same total work)")
    out.append(f"  modeled busiest-rank time of the steps: {actual_max:.0f} ms as run, {bucketed:.0f} ms bucketed, "
               f"{actual_mean:.0f} ms if every rank carried exactly the mean")
    out.append(f"  bucketing removes {removable:.0%} of the modeled excess; on the measured excess that is "
               f"{saved_ms:.0f} ms over the analysed steps, {share:.1%} of their time")
    return {"modeled_actual": actual_max, "modeled_bucketed": bucketed, "modeled_mean": actual_mean,
            "removable_share_of_excess": removable, "saved_share_of_step": share}


def partition(costs: Sequence[float], stages: int) -> tuple[float, list[int]]:
    """Cut a sequence into ``stages`` contiguous groups minimizing the heaviest; return it and the cut positions."""
    count = len(costs)
    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    best = [[math.inf] * (count + 1) for _ in range(stages + 1)]
    cut = [[0] * (count + 1) for _ in range(stages + 1)]
    best[0][0] = 0.0
    for stage in range(1, stages + 1):
        for end in range(stage, count + 1):
            for start in range(stage - 1, end):
                value = max(best[stage - 1][start], prefix[end] - prefix[start])
                if value < best[stage][end]:
                    best[stage][end], cut[stage][end] = value, start
    positions, end = [], count
    for stage in range(stages, 0, -1):
        end = cut[stage][end]
        positions.append(end)
    return best[stages][count], sorted(positions)[1:]


def pipeline_sequence(run: Run, rows: list[dict]) -> list[tuple[str, float]]:
    """Return the model's pieces in order with their mean time per micro-batch (all phases)."""
    totals: dict[tuple[str, Optional[int]], float] = {}
    micro_batches = 0
    for rank in run.ranks:
        modules = {m["id"]: m for m in run.headers[rank]["modules"]}
        for record in run.steps[rank]:
            micro_batches += len(record["micro_batches"])
            for (module_id, pass_name, _occurrence), span in spans_of(record).items():
                module = modules.get(module_id)
                if module is not None and pass_name in PHASES and "in" in span and "out" in span:
                    key = (module["role"], module["index"])
                    totals[key] = totals.get(key, 0.0) + span["out"] - span["in"]
    per_batch = {key: value / max(micro_batches, 1) for key, value in totals.items()}
    sequence: list[tuple[str, float]] = []
    layered = any(role == "text.layer" for role, _ in per_batch)
    for role, label in (("vision.patch_embed", "vision.patch_embed"), ("vision.block", "vision.block"),
                        ("vision.deepstack", "vision.deepstack"), ("vision.merger", "vision.merger"),
                        ("text.embed", "embed"), ("text.layer", "layer"), ("text.attn", "attn"), ("text.moe", "moe"),
                        ("lm_head", "lm_head")):
        if role in ("text.attn", "text.moe") and layered:
            continue
        ordered = sorted(per_batch.items(), key=lambda item: (item[0][1] is None, item[0][1] or 0))
        for (key_role, index), value in ordered:
            if key_role == role:
                sequence.append((f"{label}{'' if index is None else index}", value))
    sequence.append(("loss", mean([row["t"]["loss"]["bwd"] for row in rows])))
    return sequence


def report_pipeline(run: Run, rows: list[dict], stages: Sequence[int], out: list[str]) -> dict[str, Any]:
    """How evenly the model's pieces cut into pipeline stages: the vision tower in front, the head and loss behind."""
    sequence = pipeline_sequence(run, rows)
    costs = [cost for _, cost in sequence]
    total = sum(costs)
    if total <= 0:
        return {}
    vision = sum(cost for label, cost in sequence if label.startswith("vision"))
    tail = sum(cost for label, cost in sequence if label in ("lm_head", "loss"))
    out.append("")
    out.append(f"PIPELINE what-if (mean costs of the model's {len(sequence)} pieces, forward + recompute + backward): "
               f"the vision tower is {vision / total:.1%} of the work, the head and loss {tail / total:.1%}")
    summary: dict[str, Any] = {"vision_share": vision / total, "head_loss_share": tail / total}
    layers = [(label, cost) for label, cost in sequence if label.startswith("layer")]
    for count in stages:
        heaviest, cuts = partition(costs, count)
        ideal = total / count
        # The obvious cut: the layers split evenly, the vision tower with the first stage, the head with the last.
        even = _even_layer_cut(sequence, layers, count)
        names = [sequence[position][0] for position in cuts]
        out.append(f"  {count} stages: best contiguous cut has the heaviest stage at {heaviest / ideal:.3f}x the mean "
                   f"(cuts before {', '.join(names)}); even layers per stage: {even / ideal:.3f}x")
        summary[str(count)] = {"best_over_mean": heaviest / ideal, "even_layers_over_mean": even / ideal}
    return summary


def _even_layer_cut(sequence: list[tuple[str, float]], layers: list[tuple[str, float]], stages: int) -> float:
    """Return the heaviest stage when the layers are split evenly and the rest stays with the first and last stage."""
    if not layers:
        return sum(cost for _, cost in sequence)
    first = [cost for label, cost in sequence if label.startswith(("vision", "embed"))]
    last = [cost for label, cost in sequence if label in ("lm_head", "loss")]
    per_stage = [0.0] * stages
    per_stage[0] += sum(first)
    per_stage[-1] += sum(last)
    size = len(layers) / stages
    for position, (_label, cost) in enumerate(layers):
        per_stage[min(int(position / size), stages - 1)] += cost
    return max(per_stage)


def report_memory(rows: list[dict], out: list[str]) -> dict[str, Any]:
    """Peak memory against the workload: the biggest sample sets the die's limit."""
    kept = [row for row in rows if row.get("peak_allocated")]
    if not kept:
        return {}
    peaks = [row["peak_allocated"] / 2 ** 30 for row in kept]
    per_rank: dict[int, float] = {}
    for row in rows:
        if row.get("peak_allocated"):
            per_rank[row["rank"]] = max(per_rank.get(row["rank"], 0.0), row["peak_allocated"] / 2 ** 30)
    out.append("")
    out.append(f"MEMORY: peak allocated per rank and step {mean(peaks):.1f} GiB mean, {max(peaks):.1f} GiB max; the "
               f"busiest rank is {max(per_rank.values()) / mean(list(per_rank.values())):.2f}x the mean rank")
    if len({row.get("real_tokens", 0) for row in kept}) > 2:
        model = fit([[row.get("real_tokens", 0) / 1e3, row.get("patches", 0) / 1e3] for row in kept], peaks)
        longest = max(kept, key=lambda row: row.get("real_tokens", 0))
        out.append(f"  per-step peak ~ {model['coefficients'][0]:.1f} GiB + {model['coefficients'][1]:.2f} GiB per "
                   f"1k tokens + {model['coefficients'][2]:.2f} GiB per 1k patches (R^2 {model['r2']:.2f}); the "
                   f"longest sample ({longest.get('real_tokens', 0):.0f} tokens) peaked at "
                   f"{longest['peak_allocated'] / 2 ** 30:.1f} GiB")
    activations = {name: mean([row["act"][name] for row in rows if name in row["act"]]) / 2 ** 20
                   for name in ("vision", "text", "head") if any(name in row["act"] for row in rows)}
    if activations:
        out.append("  left allocated by each part's forward (MiB, mean): "
                   + ", ".join(f"{name} {value:.0f}" for name, value in activations.items()))
    return {"peak_gib_mean": mean(peaks), "peak_gib_max": max(peaks),
            "busiest_rank_over_mean": max(per_rank.values()) / mean(list(per_rank.values()))}


def divergence(first: Sequence[float], second: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits between two count vectors (0: the same distribution)."""
    total_a, total_b = sum(first), sum(second)
    if not total_a or not total_b:
        return 0.0
    value = 0.0
    for a, b in zip(first, second):
        p, q = a / total_a, b / total_b
        m = (p + q) / 2
        if p > 0:
            value += p * math.log2(p / m) / 2
        if q > 0:
            value += q * math.log2(q / m) / 2
    return value


def report_routing(run: Run, ep_size: int, out: list[str], top: int) -> dict[str, Any]:
    """Per MoE layer: do image and text tokens choose the same experts, and which one loads an EP rank unevenly."""
    cells: dict[tuple[int, int, int, int], tuple[list[int], list[int]]] = {}
    for rank in run.ranks:
        for record in run.steps[rank]:
            for entry in record.get("routing", []):
                cells[(record["step"], entry["mb"], entry["layer"], rank)] = (entry["visual"], entry["text"])
    if not cells:
        return {}
    layers = sorted({key[2] for key in cells})
    experts = len(next(iter(cells.values()))[0])
    local = max(experts // ep_size, 1)
    totals: dict[int, list[list[float]]] = {layer: [[0.0] * experts, [0.0] * experts] for layer in layers}
    imbalance: dict[int, dict[str, list[float]]] = {layer: {"all": [], "visual": [], "text": []} for layer in layers}
    groups: dict[tuple[int, int, int, int], list[tuple[list[int], list[int]]]] = {}
    for (step, mb, layer, rank), pair in cells.items():
        groups.setdefault((step, mb, layer, rank // ep_size), []).append(pair)
        for expert in range(experts):
            totals[layer][0][expert] += pair[0][expert]
            totals[layer][1][expert] += pair[1][expert]
    for (_step, _mb, layer, _group), pairs in groups.items():
        loads = {"all": [0.0] * ep_size, "visual": [0.0] * ep_size, "text": [0.0] * ep_size}
        for visual, text in pairs:
            for expert in range(experts):
                destination = min(expert // local, ep_size - 1)
                loads["visual"][destination] += visual[expert]
                loads["text"][destination] += text[expert]
                loads["all"][destination] += visual[expert] + text[expert]
        for name, values in loads.items():
            if mean(values) > 0:
                imbalance[layer][name].append(max(values) / mean(values))
    out.append("")
    out.append(f"ROUTING by modality ({experts} experts, EP groups of {ep_size} consecutive ranks): per layer, the "
               "load of the busiest EP rank over the mean (all tokens / image tokens only / text tokens only), the "
               "image tokens' share of the choices, and how far the two kinds' expert choices differ "
               "(Jensen-Shannon, bits)")
    out.append("  layer   all  image  text  image share  JS bits  hottest image experts      hottest text experts")
    rows_out = []
    for layer in layers:
        visual, text = totals[layer]
        js = divergence(visual, text)
        share = sum(visual) / ((sum(visual) + sum(text)) or 1.0)
        hot_v = sorted(range(experts), key=lambda e: -visual[e])[:3]
        hot_t = sorted(range(experts), key=lambda e: -text[e])[:3]
        row = {"layer": layer, "all": mean(imbalance[layer]["all"]), "visual": mean(imbalance[layer]["visual"]),
               "text": mean(imbalance[layer]["text"]), "visual_share": share, "js_bits": js,
               "hot_visual": hot_v, "hot_text": hot_t}
        rows_out.append(row)
    for row in sorted(rows_out, key=lambda r: -r["all"])[:top]:
        out.append(f"  {row['layer']:5d} {row['all']:5.2f} {row['visual']:6.2f} {row['text']:5.2f} "
                   f"{row['visual_share']:11.1%} {row['js_bits']:8.3f}  {str(row['hot_visual']):25s}  "
                   f"{row['hot_text']}")
    out.append(f"  all layers: busiest EP rank {mean([r['all'] for r in rows_out]):.2f}x the mean "
               f"(image tokens alone {mean([r['visual'] for r in rows_out]):.2f}x, text tokens alone "
               f"{mean([r['text'] for r in rows_out]):.2f}x); mean JS divergence "
               f"{mean([r['js_bits'] for r in rows_out]):.3f} bits")
    return {"layers": rows_out, "mean_all": mean([r["all"] for r in rows_out]),
            "mean_visual": mean([r["visual"] for r in rows_out]), "mean_text": mean([r["text"] for r in rows_out]),
            "mean_js_bits": mean([r["js_bits"] for r in rows_out])}


# -- outputs ----------------------------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict], models: dict[str, dict]) -> None:
    """Write one line per micro-batch: workload, component times, modeled work."""
    columns = ["rank", "step", "mb", "real_tokens", "visual_tokens", "images", "patches", "vision_attn_pairs",
               "label_tokens", "step_ms", "mb_ms"] + [f"{c}_ms" for c in (*WORK_PARTS, "text_attn", "text_moe")] \
        + ["gap_vision_fwd", "gap_text_fwd", "gap_vision_bwd", "gap_text_bwd", "modeled_ms"]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            cells = [row.get(c, "") for c in columns[:11]]
            cells += [round(component_time(row, c), 3) for c in (*WORK_PARTS, "text_attn", "text_moe")]
            cells += [round(row["gap"]["vision"]["fwd"], 3), round(row["gap"]["text"]["fwd"], 3),
                      round(row["gap"]["vision"]["bwd"], 3), round(row["gap"]["text"]["bwd"], 3),
                      round(modeled_cost(models, row), 3)]
            writer.writerow(cells)


def analyse(directory: str, skip: int, ep_size: int, stages: Sequence[int], top: int,
            out_dir: Optional[str]) -> tuple[list[str], dict[str, Any]]:
    """Run every report on one run; return the text lines and the JSON-friendly summary."""
    run = Run(directory, skip)
    rows = build_rows(run)
    if not rows:
        raise SystemExit(f"{directory}: no micro-batch record after skipping {skip} steps")
    out: list[str] = []
    summary: dict[str, Any] = {"run": run.name}
    summary["overview"] = report_overview(run, rows, out)
    summary["data"] = report_data(rows, out)
    summary["model"] = report_components(rows, out)
    summary["layers"] = report_layers(run, rows, out, top)
    models = report_fit(rows, out)
    summary["fit"] = {name: {"coefficients": m["coefficients"], "r2": m["r2"], "rmse": m["rmse"]}
                      for name, m in models.items()}
    summary["imbalance"] = report_imbalance(rows, models, out)
    summary["balancing"] = report_whatif_balancing(rows, models, summary["imbalance"], out)
    summary["pipeline"] = report_pipeline(run, rows, stages, out)
    summary["memory"] = report_memory(rows, out)
    summary["routing"] = report_routing(run, ep_size, out, top)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_csv(os.path.join(out_dir, "microbatches.csv"), rows, models)
        with open(os.path.join(out_dir, "hetero_report.json"), "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        with open(os.path.join(out_dir, "hetero_report.txt"), "w", encoding="utf-8") as stream:
            stream.write("\n".join(out) + "\n")
        out.append(f"\nwrote microbatches.csv, hetero_report.json and hetero_report.txt to {out_dir}")
    return out, summary


def sweep_row(summary: dict[str, Any]) -> list[Any]:
    """One line of the sweep table from a run's summary."""
    model, imbalance = summary["model"], summary["imbalance"]
    total = model.get("mb_ms") or 1.0
    return [summary["run"], summary["overview"]["step_ms"], summary["data"]["real_tokens"]["cv"],
            summary["data"]["visual_tokens"]["cv"], imbalance["busiest_over_mean"], imbalance["idle_share"],
            model.get("vision", {}).get("total", 0.0) / total, model.get("text_moe", {}).get("total", 0.0) / total,
            model.get("text_attn", {}).get("total", 0.0) / total,
            (model.get("head", {}).get("total", 0.0) + model.get("loss", {}).get("total", 0.0)) / total,
            summary["overview"]["host_gap_ms"],
            (summary.get("balancing") or {}).get("saved_share_of_step", 0.0)]


def report_sweep(summaries: list[dict[str, Any]], out: list[str]) -> None:
    """One row per run: the step time and the heterogeneity measures side by side."""
    out.append("SWEEP: one row per run (shares are of the micro-batch's time)")
    out.append("  run                              step ms  tokens cv  visual cv  busiest/mean  idle   vision    moe   "
               "attn  head+loss  host gap ms  bucketing saves")
    for summary in summaries:
        row = sweep_row(summary)
        out.append(f"  {row[0][:30]:30s} {row[1]:9.1f} {row[2]:10.2f} {row[3]:10.2f} {row[4]:13.3f} {row[5]:5.1%} "
                   f"{row[6]:7.1%} {row[7]:6.1%} {row[8]:6.1%} {row[9]:9.1%} {row[10]:12.1f} {row[11]:15.1%}")


def default_out_dir(records: str) -> str:
    """Return where the outputs go by default: beside a ``hetero`` directory of records, else inside the directory."""
    records = os.path.abspath(records)
    base = os.path.dirname(records) if os.path.basename(records) == "hetero" else records
    return os.path.join(base, "analysis_hetero")


def _resolve(path: str) -> str:
    """Return the directory of the per-rank files: the path itself, or its ``hetero`` subdirectory."""
    return path if glob.glob(os.path.join(path, "rank*.jsonl")) else os.path.join(path, "hetero")


def main() -> int:
    """Parse the command line and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="*",
                        help="run directories (each holds rank*.jsonl, or a hetero/ directory of them)")
    parser.add_argument("--skip", type=int, default=0, help="recorded steps to drop from the start of each rank")
    parser.add_argument("--ep-size", type=int, default=16, help="ranks per expert-parallel group (consecutive ranks)")
    parser.add_argument("--pp-stages", type=int, nargs="+", default=[2, 4, 8],
                        help="stage counts of the pipeline what-if")
    parser.add_argument("--top", type=int, default=8, help="rows listed in the layer and routing tables")
    parser.add_argument("--out-dir", default=None, help="where to write microbatches.csv and hetero_report.*")
    parser.add_argument("--sweep", action="store_true", help="compare several runs, one row each")
    args = parser.parse_args()
    if not args.run:
        parser.error("give at least one run directory")
    if args.sweep:
        summaries = []
        for directory in args.run:
            _, summary = analyse(_resolve(directory), args.skip, args.ep_size, args.pp_stages, args.top, None)
            summaries.append(summary)
        lines: list[str] = []
        report_sweep(summaries, lines)
        print("\n".join(lines))
        return 0
    for directory in args.run:
        out_dir = args.out_dir or default_out_dir(_resolve(directory))
        lines, _ = analyse(_resolve(directory), args.skip, args.ep_size, args.pp_stages, args.top, out_dir)
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
