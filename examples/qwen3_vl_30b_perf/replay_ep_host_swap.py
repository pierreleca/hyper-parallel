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
"""Replay candidate eviction rules of the MoE host swap on a no-swap run's routing.

Every (step, rank) of the no-swap records gives the pairs each MoE layer received
and sent. Each layer saves three tensors per received pair (grouped-GEMM input,
SwiGLU input, SwiGLU output); after each layer a rule sets a threshold on what the
rank may hold, and the rank evicts whole tensors, earliest layers first, down to it,
exactly as ``host_swap.py`` picks them. With m the rank's mean bytes per layer (its
sent pairs), L layers and f the factor, the budget is B = f·L·m and the threshold
after layer i (1-based) is:

- ``projection``: B − (L − i)·m, what the swap does today;
- ``angled``: f·i·m, the budget spread evenly over the layers;
- ``combined``: the larger of the two, which is ``angled`` for f ≤ 1 and
  ``projection`` for f ≥ 1.

``--no-last`` skips the decision after the last layer, so nothing is still being
copied when forward ends. Per rule and factor the report gives, over every
(step, rank): bytes moved, the largest single decision (a burst on the host link),
bytes decided after the last layer (still in flight at the step's peak), the
highest end-of-forward total against the budget, and the share of rank-steps that
evicted although their whole pass fit the budget (bytes moved for nothing)::

    python examples/qwen3_vl_30b_perf/replay_ep_host_swap.py $R/a3_16dev_8l_noswap/instrument
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

from analyze_ep_instrument import call_loads, load_records
from hyper_parallel.distributed.expert_parallel.host_swap import choose_offload

GIB = 1024 ** 3
# Bytes each saved tensor keeps per received pair, Qwen3-VL-30B-A3B (hidden 2048, expert intermediate 768, bf16).
TENSOR_BYTES = (4096, 3072, 1536)
RULES = ("projection", "angled", "combined")


def threshold(rule: str, factor: float, layer: int, layers: int, mean: float) -> float:
    """What a rank may hold after ``layer`` (1-based) of ``layers``."""
    projection = factor * layers * mean - (layers - layer) * mean
    angled = factor * layer * mean
    if rule == "projection":
        return projection
    if rule == "angled":
        return angled
    return max(projection, angled)


@dataclass
class Pass:
    """One rank's forward pass under one rule: what it held, moved and decided."""

    decisions: list[tuple[int, float]] = field(default_factory=list)  # (layer, bytes evicted), 0-based
    held_end: float = 0.0

    @property
    def moved(self) -> float:
        """Bytes sent to host over the pass."""
        return sum(amount for _layer, amount in self.decisions)


def replay_pass(received: list[int], sent: list[int], rule: str, factor: float, no_last: bool) -> Pass:
    """Apply one rule to one rank's pass, earliest tensors first, as the swap picks them."""
    layers = len(received)
    mean = sum(TENSOR_BYTES) * statistics.fmean(sent)
    on_device = [[size * pairs for size in TENSOR_BYTES] for pairs in received]
    result = Pass()
    for layer in range(layers):
        if no_last and layer == layers - 1:
            break
        held = sum(sum(sizes) for sizes in on_device[:layer + 1])
        need = held - threshold(rule, factor, layer + 1, layers, mean)
        if need <= 0:
            continue
        evicted = 0.0
        for sizes in on_device[:layer + 1]:
            candidates = [index for index, size in enumerate(sizes) if size > 0]
            for choice in choose_offload([sizes[index] for index in candidates], int(need - evicted) + 1):
                evicted += sizes[candidates[choice]]
                sizes[candidates[choice]] = 0
            if evicted >= need:
                break
        result.decisions.append((layer, evicted))
    result.held_end = sum(sum(sizes) for sizes in on_device)
    return result


def load_passes(instrument_dir: str, skip: int) -> list[tuple[list[int], list[int]]]:
    """(received, sent) pairs per layer for every kept (step, rank) of a no-swap run."""
    _headers, steps = load_records(instrument_dir, skip)
    passes = []
    for records in steps.values():
        for record in records:
            loads = call_loads(record)
            layers = sorted(loads)
            passes.append(([loads[layer]["recv"] for layer in layers], [loads[layer]["send"] for layer in layers]))
    return passes


def report(passes: list[tuple[list[int], list[int]]], factors: list[float], rules: list[str],
           window_gib: float) -> list[str]:
    """One row per (rule, factor, last-layer policy)."""
    layers = len(passes[0][0])
    out = [f"{len(passes)} (step, rank) passes of {layers} MoE layers; a copy window carries about "
           f"{window_gib:.1f} GiB",
           "  rule        factor  layers kept  last   moved GiB mean/max  largest decision GiB  over window"
           "  decided last GiB mean/max  end/budget max  moved for nothing"]
    for rule in rules:
        for factor in factors:
            for no_last in (False, True):
                runs = [(received, sent, replay_pass(received, sent, rule, factor, no_last))
                        for received, sent in passes]
                moved = [run.moved / GIB for _r, _s, run in runs]
                largest = [max((amount for _l, amount in run.decisions), default=0.0) / GIB for _r, _s, run in runs]
                last = [sum(amount for layer, amount in run.decisions if layer == layers - 1) / GIB
                        for _r, _s, run in runs]
                ends = [run.held_end / (factor * layers * sum(TENSOR_BYTES) * statistics.fmean(sent))
                        for _r, sent, run in runs]
                budget = [factor * layers * sum(TENSOR_BYTES) * statistics.fmean(sent) for _r, sent, _run in runs]
                totals = [sum(TENSOR_BYTES) * sum(received) for received, _s, _run in runs]
                wasted = sum(1 for (_r, _s, run), total, cap in zip(runs, totals, budget)
                             if run.decisions and total <= cap)
                out.append(
                    f"  {rule:<11}{factor:6.2f}{factor * layers:12.1f}  {'skip' if no_last else 'keep':<5}"
                    f"{statistics.fmean(moved):9.2f} / {max(moved):5.2f}{max(largest):21.2f}"
                    f"{sum(1 for value in largest if value > window_gib) / len(runs):13.0%}"
                    f"{statistics.fmean(last):14.2f} / {max(last):5.2f}{max(ends):16.3f}"
                    f"{wasted / len(runs):18.0%}"
                )
    return out


def main() -> int:
    """Parse, replay, print."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("record_dir", help="the no-swap run's instrument directory (rank*.jsonl)")
    parser.add_argument("--skip", type=int, default=2, help="warm-up records to drop, as the report does")
    parser.add_argument("--factors", type=float, nargs="+", default=[1.2, 1.1, 1.0, 0.9, 0.7, 0.5, 0.3])
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=RULES)
    parser.add_argument("--window-gib", type=float, default=2.0,
                        help="what one copy window carries (74 ms at 30 GB/s is about 2 GiB)")
    args = parser.parse_args()
    print("\n".join(report(load_passes(args.record_dir, args.skip), args.factors, args.rules, args.window_gib)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
