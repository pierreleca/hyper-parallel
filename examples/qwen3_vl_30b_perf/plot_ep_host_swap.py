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
"""Draw the MoE host swap's figures: the problem, the mechanism and the factor sweep.

Two modes:

``figures`` writes the three PNGs of ``docs/guide/ep_host_swap.md`` from the
numbers measured on 16 A3 dies (``EP_HOST_SWAP_RESULTS.md``). The mechanism
figure replays the swap's own eviction rule (``choose_offload`` over the three
saved tensors of each layer) on one rank's measured per-layer memory::

    python examples/qwen3_vl_30b_perf/plot_ep_host_swap.py figures --out-dir docs/images

``records`` draws the mechanism figure from run records instead: the no-swap
run's received pairs per layer (the swap runs route identically step by step)
and each swap run's eviction decisions. Run from the repository root with this
checkout importable (``pip install -e .`` or ``PYTHONPATH=.``)::

    python examples/qwen3_vl_30b_perf/plot_ep_host_swap.py records $R/a3_16dev_8l_noswap/instrument \\
        --swap 1.0=$R/a3_16dev_8l_step100/ep_host_swap --swap 0.5=$R/a3_16dev_8l_step050/ep_host_swap \\
        --out ep_host_swap_mechanism.png

In the mechanism figure the x axis counts MoE blocks, forward then backward, not
time; evictions are drawn at the decision, and a swapped layer comes back while
the layer above it runs backward. Only the routed part of the MoE activations is
drawn, the part the budget governs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from analyze_ep_instrument import call_loads, load_records  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel.distributed.expert_parallel.host_swap import choose_offload  # noqa: E402  pylint: disable=wrong-import-position

GIB = 1024 ** 3
MIB = 1024 ** 2

# Measured on 16 A3 dies, Qwen3-VL-30B-A3B, 6 text layers, no swap (EP_HOST_SWAP_RESULTS.md).
FIXED_PER_LAYER = 1645 * MIB  # held by every rank alike
LAYER_SPREAD = [  # MoE memory per rank and layer, GiB: min, mean, max over the 16 ranks
    ("L0", 2.35, 2.68, 3.00), ("L1", 2.32, 2.68, 3.08), ("L2", 2.08, 2.68, 3.51),
    ("L3", 2.32, 2.68, 3.48), ("L4", 2.16, 2.68, 3.49), ("L5", 2.18, 2.64, 3.25),
]
MEAN_TOTAL = 16.02  # GiB per rank over the six layers
RANK_VS_MEAN = {  # GiB against the mean total, per rank
    "r6": 1.13, "r10": 0.96, "r11": 0.93, "r12": 0.72, "r14": 0.62, "r7": 0.62, "r9": 0.56, "r3": 0.42,
    "r13": 0.06, "r5": -0.09, "r0": -0.52, "r15": -0.56, "r8": -0.62, "r1": -0.80, "r4": -1.57, "r2": -1.83,
}
RANK6_LAYERS = [2.35, 2.39, 2.89, 3.48, 2.97, 3.07]  # rank 6's MoE memory per layer, GiB
# The three saved tensors' share of the bytes per pair: grouped-GEMM input, SwiGLU input, SwiGLU output.
TENSOR_SHARES = [4096, 3072, 1536]
# Factor sweep at 8 text layers: factor, worst reserved, mean reserved (GiB), moved (GiB per rank and
# step), median step time against no swap, copy back exposed on the worst rank (ms).
SWEEP_BASE = (59.44, 58.27)
SWEEP = [
    (1.0, 58.56, 57.65, 0.73, 1.1, 0.0), (0.9, 57.74, 56.81, 1.59, 2.1, 0.0),
    (0.8, 56.95, 56.04, 2.40, 3.2, 0.0), (0.7, 56.62, 55.25, 3.18, 3.5, 0.0),
    (0.6, 56.12, 54.51, 3.96, 3.1, 0.0), (0.5, 55.41, 53.77, 4.77, 3.7, 0.0),
    (0.4, 54.63, 52.98, 5.64, 3.7, 0.0), (0.3, 54.06, 52.32, 6.44, 3.2, 0.0),
    (0.2, 53.07, 51.96, 7.10, 3.5, 21.9), (0.1, 53.46, 51.64, 7.78, 5.9, 99.1),
]

GREY, BLUE, RED = "#8c8c8c", "#1f5fa8", "#c0392b"


@dataclass
class SwapStep:
    """What one swap did on one rank in one step: bytes swapped per layer and its decisions."""

    factor: float
    swapped: dict[int, float] = field(default_factory=dict)
    evictions: list[dict] = field(default_factory=list)


def _style(axis) -> None:
    """Drop the top and right spines."""
    axis.spines[["top", "right"]].set_visible(False)


def _save(fig, out: str) -> None:
    """Write the figure; the format follows the extension (PNG for the docs, which ignore SVG)."""
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# -- the swap's rule, replayed ---------------------------------------------------


def replay(saved: list[float], mean: float, factor: float) -> SwapStep:
    """Apply the swap's rule to one pass: after each layer, evict the earliest tensors while projected > budget.

    Args:
        saved: Routed bytes each layer saves.
        mean: Routed bytes of a layer at the rank's mean load.
        factor: ``capacity_factor``.
    """
    layers = len(saved)
    budget = factor * mean * layers
    on_device = [[share * size / sum(TENSOR_SHARES) for share in TENSOR_SHARES] for size in saved]
    result = SwapStep(factor=factor)
    for layer in range(layers):
        held = sum(sum(tensors) for tensors in on_device[:layer + 1])
        projected = held + (layers - layer - 1) * mean
        need = projected - budget
        if need <= 0:
            continue
        evicted = 0.0
        for item in range(layer + 1):
            sizes = on_device[item]
            candidates = [index for index, size in enumerate(sizes) if size > 0]
            if not candidates:
                continue
            for choice in choose_offload([int(sizes[index]) for index in candidates], int(need - evicted) + 1):
                index = candidates[choice]
                evicted += sizes[index]
                result.swapped[item] = result.swapped.get(item, 0.0) + sizes[index]
                sizes[index] = 0.0
            if evicted >= need:
                break
        result.evictions.append({"after_layer": layer, "budget_gib": budget / GIB,
                                 "projected_gib": projected / GIB, "evicted_gib": evicted / GIB})
    return result


def curve(saved: list[float], swap: SwapStep | None) -> list[float]:
    """Held bytes after each forward block, then during each backward block (top layer first)."""
    layers = len(saved)
    evicted_at = [0.0] * layers
    swapped = swap.swapped if swap is not None else {}
    for decision in swap.evictions if swap is not None else []:
        evicted_at[min(decision["after_layer"], layers - 1)] += decision["evicted_gib"] * GIB
    held, points = 0.0, []
    for layer in range(layers):
        held += saved[layer] - evicted_at[layer]
        points.append(held)
    device = [saved[layer] - swapped.get(layer, 0.0) for layer in range(layers)]
    for layer in reversed(range(layers)):
        # The layer below comes back while this one runs backward; this one is then freed.
        if layer > 0 and swapped.get(layer - 1):
            held += swapped[layer - 1]
            device[layer - 1] = saved[layer - 1]
        points.append(held)
        held -= device[layer]
    return points


def draw_mechanism(saved: list[float], mean: float, swaps: list[SwapStep], title: str, out: str) -> None:
    """One panel per factor: the pass without swap, with swap, the budget and each eviction decision."""
    count = len(saved)
    base = [value / GIB for value in curve(saved, None)]
    xs = list(range(2 * count))
    labels = [f"F{layer}" for layer in range(count)] + [f"B{layer}" for layer in reversed(range(count))]
    top = max(base) * 1.3
    fig, axes = plt.subplots(1, len(swaps), figsize=(4.4 * len(swaps), 3.6), sharey=True, squeeze=False)
    for axis, swap in zip(axes[0], swaps):
        budget = swap.factor * mean * count / GIB
        held = [value / GIB for value in curve(saved, swap)]
        axis.axvspan(count - 0.5, 2 * count - 0.5, color="#f2f2f2", zorder=0)
        axis.text((count - 1) / 2, top * 0.99, "forward", ha="center", va="top", fontsize=8, color="#555555")
        axis.text(count + (count - 1) / 2, top * 0.99, "backward", ha="center", va="top", fontsize=8,
                  color="#555555")
        axis.fill_between(xs, held, base, step="mid", color=BLUE, alpha=0.12, linewidth=0, label="on host")
        axis.step(xs, base, where="mid", color=GREY, linestyle="--", linewidth=1.4, label="no swap")
        axis.step(xs, held, where="mid", color=BLUE, linewidth=2.0, label="with swap")
        # The rule after layer l: evict while held + remaining layers x mean > budget, so the
        # threshold on what is held rises by one mean load per layer and meets the budget at the end.
        threshold = [max(budget - (count - 1 - layer) * mean / GIB, 0.0) for layer in range(count)]
        axis.plot(range(count), threshold, color=RED, linewidth=1.4, linestyle=(0, (4, 2)),
                  label="eviction threshold: budget − remaining layers × mean load")
        axis.plot([count - 1], [budget], marker="_", markersize=14, color=RED, linestyle="none")
        axis.annotate(f"budget {budget:.1f} GiB", (count - 1, budget), textcoords="offset points",
                      xytext=(4, -14), ha="left", fontsize=7, color=RED)
        after = {decision["after_layer"] for decision in swap.evictions}
        for layer in sorted(after):
            before = (held[layer - 1] if layer else 0.0) + saved[layer] / GIB
            axis.annotate("", xy=(layer, held[layer]), xytext=(layer, before),
                          arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.0})
            axis.plot([layer], [before], marker="o", markersize=4, markerfacecolor="white",
                      markeredgecolor=RED, linestyle="none")
        axis.set_title(f"capacity_factor {swap.factor:g}: peak {max(base):.1f} → {max(held):.1f} GiB, "
                       f"{sum(swap.swapped.values()) / GIB:.1f} GiB to host", fontsize=9)
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=7)
        axis.set_ylim(0, top)
        _style(axis)
    axes[0][0].set_ylabel("routed MoE activations held, GiB")
    axes[0][0].plot([], [], marker="o", markersize=4, markerfacecolor="white", markeredgecolor=RED,
                    linestyle="none", label="held before eviction: over the threshold, earliest layers evicted")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, out)


# -- the documentation's figures ------------------------------------------------


def draw_problem(out: str) -> None:
    """Per layer the busiest rank sits far above the mean; over the step the ranks sit much closer."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for index, (_name, low, mean, high) in enumerate(LAYER_SPREAD):
        left.plot([index, index], [low, high], color=GREY, linewidth=1, zorder=1)
        left.text(index, high + 0.1, f"{high / mean:.2f}×", ha="center", fontsize=8, color=RED)
    xs = range(len(LAYER_SPREAD))
    left.scatter(xs, [row[3] for row in LAYER_SPREAD], marker="^", color=RED, s=36, zorder=2,
                 label="busiest rank (label: × the mean)")
    left.scatter(xs, [row[2] for row in LAYER_SPREAD], marker="_", color=BLUE, s=160, linewidths=2, zorder=2,
                 label="mean over the 16 ranks")
    left.scatter(xs, [row[1] for row in LAYER_SPREAD], marker="v", color=GREY, s=36, zorder=2,
                 label="lightest rank")
    left.legend(fontsize=7, frameon=False, loc="lower left")
    left.set_xticks(range(len(LAYER_SPREAD)))
    left.set_xticklabels([row[0] for row in LAYER_SPREAD])
    left.set_ylim(0, 4)
    left.set_xlim(-0.5, len(LAYER_SPREAD) - 0.1)
    left.set_ylabel("MoE memory of one layer, GiB")
    left.set_title("Per layer: busiest, mean and lightest rank", fontsize=9)
    _style(left)

    ranks = list(RANK_VS_MEAN)
    totals = [MEAN_TOTAL + RANK_VS_MEAN[rank] for rank in ranks]
    # Same colours as the left panel: red the busiest, blue the mean, grey the others.
    right.bar(range(len(ranks)), totals, color=[RED if total == max(totals) else GREY for total in totals],
              alpha=0.6, width=0.7)
    right.axhline(MEAN_TOTAL, color=BLUE, linewidth=2)
    right.text(len(ranks) - 0.5, MEAN_TOTAL + 0.08, f"mean over the 16 ranks, {MEAN_TOTAL:.2f} GiB", ha="right",
               fontsize=8, color=BLUE)
    right.set_xticks(range(len(ranks)))
    right.set_xticklabels(ranks, fontsize=7)
    right.set_ylim(13, 18)
    right.set_ylabel("MoE memory over the step, GiB")
    right.set_title(f"Per rank over all layers: the peak, {max(totals):.2f} GiB, "
                    f"is {max(totals) / MEAN_TOTAL:.2f}× the mean", fontsize=9)
    _style(right)
    fig.suptitle("MoE activation memory under EP imbalance (16 A3 dies, 6 text layers, no swap)", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def draw_sweep(out: str) -> None:
    """Memory falls with the factor while the step time stays near 3%, until the copies stop hiding."""
    labels = ["no\nswap"] + [f"{row[0]:g}" for row in SWEEP]
    xs = list(range(len(labels)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    left.plot(xs, [SWEEP_BASE[0]] + [row[1] for row in SWEEP], marker="o", color=RED, label="worst rank")
    left.plot(xs, [SWEEP_BASE[1]] + [row[2] for row in SWEEP], marker="o", color=BLUE, label="mean over ranks")
    left.axhline(61.3, color=GREY, linestyle=":", linewidth=1)
    left.text(0, 61.0, "die capacity 61.3 GiB", fontsize=8, color="#555555", va="top")
    left.set_ylabel("peak reserved memory, GiB")
    left.set_title("Memory falls with the factor", fontsize=9)
    left.legend(fontsize=8, frameon=False, loc="lower left")

    moved = [0.0] + [row[3] for row in SWEEP]
    right.bar(xs, moved, color=BLUE, alpha=0.35, label="moved to host, GiB per rank and step")
    right.set_ylabel("moved to host, GiB")
    twin = right.twinx()
    twin.plot(xs, [0.0] + [row[4] for row in SWEEP], marker="o", color=RED, label="median step time")
    twin.set_ylabel("median step time vs no swap, %")
    twin.set_ylim(0, 7)
    for x, row in zip(xs[1:], SWEEP):
        if row[5]:
            twin.annotate(f"{row[5]:.0f} ms\nwaited", (x, row[4]), textcoords="offset points", xytext=(-16, 4),
                          ha="right", fontsize=7, color=RED)
    right.set_title("Cost stays near 3%; copies stop hiding below 0.3", fontsize=9)
    handles = right.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    names = right.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    right.legend(handles, names, fontsize=8, frameon=False, loc="upper left")
    for axis in (left, right):
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_xlabel("capacity_factor")
        _style(axis)
    twin.spines[["top"]].set_visible(False)
    fig.suptitle("Factor sweep (16 A3 dies, 8 text layers, eleven runs that route identically)", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def figures(out_dir: str) -> None:
    """Write the three figures of the guide."""
    os.makedirs(out_dir, exist_ok=True)
    draw_problem(os.path.join(out_dir, "ep_host_swap_problem.png"))
    saved = [size * GIB - FIXED_PER_LAYER for size in RANK6_LAYERS]
    mean = MEAN_TOTAL * GIB / len(RANK6_LAYERS) - FIXED_PER_LAYER
    draw_mechanism(saved, mean, [replay(saved, mean, 1.0), replay(saved, mean, 0.5)],
                   "The swap on rank 6's measured step (6 text layers; eviction rule replayed)",
                   os.path.join(out_dir, "ep_host_swap_mechanism.png"))
    draw_sweep(os.path.join(out_dir, "ep_host_swap_sweep.png"))


# -- from run records -------------------------------------------------------------


def load_routing(instrument_dir: str, skip: int) -> dict[tuple[int, int], dict[int, tuple[int, int]]]:
    """Return {(step, rank): {layer: (received pairs, sent pairs)}} from a no-swap run."""
    _headers, steps = load_records(instrument_dir, skip)
    routing = {}
    for rank, records in steps.items():
        for record in records:
            routing[(record["step"], rank)] = {
                layer: (load["recv"], load["send"]) for layer, load in call_loads(record).items()
            }
    return routing


def load_swap(swap_dir: str, rank: int, step: int, factor: float) -> tuple[SwapStep, float | None]:
    """Read one rank's record of one step; also return the bytes a received pair saves, if a layer shows it."""
    path = os.path.join(swap_dir, f"host_swap_rank{rank}.jsonl")
    with open(path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    records = [record for record in records if not record.get("header")]
    pair_bytes = next((sum(saved[2] for saved in layer["saved"]) / layer["rows"]
                       for record in records for layer in record["layers"] if layer["rows"] and layer["saved"]),
                      None)
    matches = [record for record in records if record["step"] == step]
    if not matches:
        raise SystemExit(f"{path}: no record for step {step}")
    record = matches[-1]
    swap = SwapStep(factor=factor, swapped={layer["index"]: float(layer["swapped_bytes"])
                                            for layer in record["layers"]},
                    evictions=record.get("evictions", []))
    return swap, pair_bytes


def from_records(args: argparse.Namespace) -> None:
    """Draw the mechanism figure from a no-swap run and swap runs of the same routing."""
    routing = load_routing(args.noswap, args.skip)
    keys = [key for key in routing
            if (args.step is None or key[0] == args.step) and (args.rank is None or key[1] == args.rank)]
    if not keys:
        raise SystemExit("no (step, rank) in the no-swap records matches --step / --rank")
    step, rank = max(keys, key=lambda key: sum(recv for recv, _sent in routing[key].values()))
    swaps, pair_bytes = [], args.pair_bytes
    for spec in args.swap:
        factor, _sep, swap_dir = spec.partition("=")
        swap, seen = load_swap(swap_dir, rank, step, float(factor))
        swaps.append(swap)
        pair_bytes = pair_bytes or seen
    if not pair_bytes:
        raise SystemExit("no swapped layer to read the bytes per pair from; pass --pair-bytes")
    row = routing[(step, rank)]
    layers = sorted(row)
    saved = [row[layer][0] * pair_bytes for layer in layers]
    mean = sum(row[layer][1] for layer in layers) * pair_bytes / len(layers)
    draw_mechanism(saved, mean, swaps, f"The swap on rank {rank}, step {step}, {len(layers)} MoE layers",
                   args.out)


def main() -> int:
    """Parse the mode and draw."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    doc = modes.add_parser("figures", help="the guide's three figures, from the measured numbers")
    doc.add_argument("--out-dir", default="docs/images", help="where the PNGs go")
    run = modes.add_parser("records", help="the mechanism figure from run records")
    run.add_argument("noswap", help="the no-swap run's instrument directory (rank*.jsonl)")
    run.add_argument("--swap", action="append", required=True, metavar="FACTOR=DIR",
                     help="a swap run's ep_host_swap directory and its capacity_factor; repeat")
    run.add_argument("--step", type=int, default=None, help="step to draw (default: the worst)")
    run.add_argument("--rank", type=int, default=None, help="rank to draw (default: the worst)")
    run.add_argument("--skip", type=int, default=2, help="warm-up records to drop, as the report does")
    run.add_argument("--pair-bytes", type=float, default=None,
                     help="bytes saved per received pair (default: read from the swap records)")
    run.add_argument("--out", default="ep_host_swap_mechanism.png", help="SVG to write")
    args = parser.parse_args()
    if args.mode == "figures":
        figures(args.out_dir)
    else:
        from_records(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
