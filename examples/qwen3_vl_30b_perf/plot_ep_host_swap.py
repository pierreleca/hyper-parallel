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
"""Draw the MoE host swap's figures from a campaign's results.json (export_results.py).

    python examples/qwen3_vl_30b_perf/plot_ep_host_swap.py sweep.json [--depth depth.json] --out-dir docs/images

writes the guide's figures: the problem (per layer and per rank, the no-swap run's
MoE memory at the step where one rank received the most), the mechanism (that
rank and step under two budgets: what it held, the threshold, each decision),
the budget sweep and, with ``--depth``, how deep the model fits with and without
the swap.

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
from matplotlib.patches import Patch  # noqa: E402  pylint: disable=wrong-import-position

from replay_ep_host_swap import threshold  # noqa: E402  pylint: disable=wrong-import-position

GIB = 1024 ** 3
CAPACITY_GIB = 61.28  # an A3 die as torch_npu reports it

GREY, BLUE, RED = "#8c8c8c", "#1f5fa8", "#c0392b"


@dataclass
class SwapStep:
    """What one swap did on one rank in one step: its budget, bytes swapped per layer and its decisions."""

    budget_layers: float
    rule: str = "projection"
    swapped: dict[int, float] = field(default_factory=dict)
    decisions: list[tuple[int, float]] = field(default_factory=list)  # (layer after which, bytes evicted)


def _style(axis) -> None:
    """Drop the top and right spines."""
    axis.spines[["top", "right"]].set_visible(False)


def _save(fig, out: str) -> None:
    """Write the figure; the format follows the extension (PNG for the docs, which ignore SVG)."""
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# -- the mechanism ------------------------------------------------------------------


def curve(saved: list[float], swap: SwapStep | None) -> list[float]:
    """Held bytes after each forward block, then during each backward block (top layer first)."""
    layers = len(saved)
    evicted_at = [0.0] * layers
    swapped = swap.swapped if swap is not None else {}
    for layer, amount in swap.decisions if swap is not None else []:
        evicted_at[min(layer, layers - 1)] += amount
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
    """One panel per budget: the pass without swap, with swap, the eviction threshold and each decision."""
    count = len(saved)
    base = [value / GIB for value in curve(saved, None)]
    xs = list(range(2 * count))
    labels = [f"F{layer}" for layer in range(count)] + [f"B{layer}" for layer in reversed(range(count))]
    top = max(base) * 1.3
    fig, axes = plt.subplots(1, len(swaps), figsize=(4.4 * len(swaps), 3.6), sharey=True, squeeze=False)
    for axis, swap in zip(axes[0], swaps):
        budget = swap.budget_layers * mean / GIB
        held = [value / GIB for value in curve(saved, swap)]
        axis.axvspan(count - 0.5, 2 * count - 0.5, color="#f2f2f2", zorder=0)
        axis.text((count - 1) / 2, top * 0.99, "forward", ha="center", va="top", fontsize=8, color="#555555")
        axis.text(count + (count - 1) / 2, top * 0.99, "backward", ha="center", va="top", fontsize=8,
                  color="#555555")
        axis.fill_between(xs, held, base, step="mid", color=BLUE, alpha=0.12, linewidth=0, label="on host")
        axis.step(xs, base, where="mid", color=GREY, linestyle="--", linewidth=1.4, label="no swap")
        axis.step(xs, held, where="mid", color=BLUE, linewidth=2.0, label="with swap")
        # What the rank may hold after each layer; the rank evicts down to it.
        line = [max(threshold(swap.rule, swap.budget_layers / count, layer + 1, count, mean), 0.0) / GIB
                for layer in range(count)]
        axis.plot(range(count), line, color=RED, linewidth=1.4, linestyle=(0, (4, 2)),
                  label="eviction threshold after each layer")
        axis.plot([count - 1], [budget], marker="_", markersize=14, color=RED, linestyle="none")
        axis.annotate(f"budget {budget:.1f} GiB", (count - 1, budget), textcoords="offset points",
                      xytext=(4, -14), ha="left", fontsize=7, color=RED)
        after = {layer for layer, _amount in swap.decisions}
        for layer in sorted(after):
            before = (held[layer - 1] if layer else 0.0) + saved[layer] / GIB
            axis.annotate("", xy=(layer, held[layer]), xytext=(layer, before),
                          arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.0})
            axis.plot([layer], [before], marker="o", markersize=4, markerfacecolor="white",
                      markeredgecolor=RED, linestyle="none")
        axis.set_title(f"budget_layers {swap.budget_layers:g} of {count}: peak {max(base):.1f} → {max(held):.1f} GiB, "
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


# -- the figures, from results.json --------------------------------------------------


def _size(results: dict) -> str:
    """The run shape the titles quote."""
    base = results["runs"][results["baseline"]]
    ranks = len(results["baseline_tables"]["ranks"])
    return f"{ranks} A3 dies, {base['layers']} text layers, no swap"


def draw_problem(results: dict, out: str) -> None:
    """Per layer the busiest rank sits far above the mean; over the step the ranks sit much closer."""
    tables = results["baseline_tables"]
    step = tables["worst"]["step"]
    retained = [[value / 1024 for value in row] for row in tables["at_worst_step"]["retained_mib"]]  # rank x layer
    layers = tables["layers"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for index in range(len(layers)):
        column = [row[index] for row in retained]
        low, mean, high = min(column), sum(column) / len(column), max(column)
        left.plot([index, index], [low, high], color=GREY, linewidth=1, zorder=1)
        left.scatter([index], [high], marker="^", color=RED, s=36, zorder=2,
                     label="busiest rank (label: × the mean)" if index == 0 else None)
        left.scatter([index], [mean], marker="_", color=BLUE, s=160, linewidths=2, zorder=2,
                     label="mean over the ranks" if index == 0 else None)
        left.scatter([index], [low], marker="v", color=GREY, s=36, zorder=2,
                     label="lightest rank" if index == 0 else None)
        left.text(index, high * 1.03, f"{high / mean:.2f}×", ha="center", fontsize=7, color=RED)
    left.legend(fontsize=7, frameon=False, loc="lower left")
    left.set_xticks(range(len(layers)))
    left.set_xticklabels([f"L{layer}" for layer in layers])
    left.set_ylim(0, max(max(row) for row in retained) * 1.2)
    left.set_ylabel("MoE memory of one layer, GiB")
    left.set_title("Per layer: busiest, mean and lightest rank", fontsize=9)
    _style(left)

    totals = sorted((sum(row) for row in retained), reverse=True)
    mean_total = sum(totals) / len(totals)
    right.bar(range(len(totals)), totals, color=[RED] + [GREY] * (len(totals) - 1), alpha=0.6, width=0.8)
    right.axhline(mean_total, color=BLUE, linewidth=2)
    right.text(len(totals) - 0.5, mean_total * 1.01, f"mean, {mean_total:.2f} GiB", ha="right", fontsize=8,
               color=BLUE, va="bottom")
    right.set_xticks([])
    right.set_xlabel("ranks, fullest first")
    right.set_ylim(min(totals) * 0.9, max(totals) * 1.05)
    right.set_ylabel("MoE memory over the step, GiB")
    right.set_title(f"Per rank over all layers: the peak, {totals[0]:.2f} GiB, is {totals[0] / mean_total:.2f}×"
                    " the mean", fontsize=9)
    _style(right)
    fig.suptitle(f"MoE activation memory under EP imbalance ({_size(results)}, step {step})", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def mechanism(results: dict, budgets: list[float], out: str) -> None:
    """The baseline's fullest rank and step, under the swap runs closest to the budgets asked for."""
    tables = results["baseline_tables"]
    worst = tables["worst"]
    step, column = worst["step"], tables["ranks"].index(worst["rank"])
    swaps = {name: record for name, record in results["swaps"].items() if record.get("pair_bytes")}
    if not swaps:
        raise SystemExit("results.json holds no swap record of the fullest rank and step")
    pair_bytes = next(iter(swaps.values()))["pair_bytes"]
    received, sent = tables["at_worst_step"]["recv"][column], tables["at_worst_step"]["sent"][column]
    saved = [pairs * pair_bytes for pairs in received]
    mean = sum(sent) * pair_bytes / len(sent)
    chosen = []
    for budget in budgets:
        name = min(swaps, key=lambda run: abs(float(results["runs"][run]["budget_layers"]) - budget))
        record = swaps[name]
        chosen.append(SwapStep(
            budget_layers=float(results["runs"][name]["budget_layers"]),
            swapped={int(layer): amount * GIB for layer, amount in record["swapped_gib"].items()},
            decisions=[(decision["after_layer"], decision["evicted_gib"] * GIB) for decision in record["evictions"]],
        ))
    draw_mechanism(saved, mean, chosen, f"The swap on rank {worst['rank']}, step {step}, the fullest of the"
                   f" no-swap run ({len(saved)} MoE layers, the runs route identically)", out)


def draw_sweep(results: dict, out: str) -> None:
    """Memory falls with the budget; the step time shows what the copies cost."""
    base = results["runs"][results["baseline"]]
    rows = sorted(((float(run["budget_layers"]), run["sweep"]) for name, run in results["runs"].items()
                   if run["budget_layers"] and run.get("sweep") and run["layers"] == base["layers"]
                   and "profiling.enabled=true" not in run["overrides"]), reverse=True)
    base_row = base["sweep"]
    labels = ["no\nswap"] + [f"{budget:g}" for budget, _row in rows]
    xs = list(range(len(labels)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    left.plot(xs, [base_row["reserved_worst"]] + [row["reserved_worst"] for _b, row in rows], marker="o", color=RED,
              label="worst rank")
    left.plot(xs, [base_row["reserved_mean"]] + [row["reserved_mean"] for _b, row in rows], marker="o", color=BLUE,
              label="mean over ranks")
    left.axhline(CAPACITY_GIB, color=GREY, linestyle=":", linewidth=1)
    left.text(0, CAPACITY_GIB - 0.3, f"die capacity {CAPACITY_GIB} GiB", fontsize=8, color="#555555", va="top")
    left.set_ylabel("peak reserved memory, GiB")
    left.set_title("Memory falls with the budget", fontsize=9)
    left.legend(fontsize=8, frameon=False, loc="lower left")

    right.bar(xs, [0.0] + [row["moved_gib"] for _b, row in rows], color=BLUE, alpha=0.35,
              label="moved to host, GiB per rank and step")
    right.set_ylabel("moved to host, GiB")
    twin = right.twinx()
    times = [0.0] + [(row["step_median"] / base_row["step_median"] - 1) * 100 for _b, row in rows]
    twin.plot(xs, times, marker="o", color=RED, label="median step time")
    twin.set_ylabel("median step time vs no swap, %")
    twin.set_ylim(min(0.0, min(times)) - 1, max(times) * 1.4 + 1)
    for x, (_budget, row) in zip(xs[1:], rows):
        if row["exposed_max"] >= 1.0:
            twin.annotate(f"{row['exposed_max']:.0f} ms\nwaited", (x, times[x]), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=7, color=RED)
    right.set_title("What the copies cost", fontsize=9)
    handles = right.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    names = right.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    right.legend(handles, names, fontsize=8, frameon=False, loc="upper left")
    for axis in (left, right):
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_xlabel(f"budget_layers, of {base['layers']} MoE layers")
        _style(axis)
    twin.spines[["top"]].set_visible(False)
    fig.suptitle(f"Budget sweep ({len(results['baseline_tables']['ranks'])} A3 dies, {base['layers']} text layers,"
                 f" {len(rows) + 1} runs that route identically)", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def draw_depth(results: dict, out: str) -> None:
    """Per depth: no swap, the swap as a safety net and as a reduction; worst reserved, or out of memory."""
    runs = [run for run in results["runs"].values() if run["layers"]]
    depths = sorted({int(run["layers"]) for run in runs})
    kinds = [("no swap", GREY), ("budget = mean", BLUE), ("budget = 0.2 × mean", RED)]

    def kind(run: dict) -> int:
        if run["budget_layers"] is None:
            return 0
        return 1 if float(run["budget_layers"]) >= int(run["layers"]) else 2

    fig, axis = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for run in runs:
        index, depth = kind(run), int(run["layers"])
        x = depths.index(depth) + (index - 1) * width
        if run.get("sweep") and run["state"] == "finished":
            value = run["sweep"]["reserved_worst"]
            axis.bar(x, value, width, color=kinds[index][1], alpha=0.7)
        else:
            oom = "OutOfMemory" in run["state"]
            axis.bar(x, CAPACITY_GIB, width, color="white", edgecolor=kinds[index][1], hatch="//")
            axis.text(x, CAPACITY_GIB + 0.5, "OOM" if oom else "failed", ha="center", fontsize=7,
                      color=kinds[index][1], rotation=90, va="bottom")
    handles = [Patch(color=color, alpha=0.7, label=name) for name, color in kinds]
    handles.append(Patch(facecolor="white", edgecolor=GREY, hatch="//", label="did not run to the end"))
    axis.axhline(CAPACITY_GIB, color=GREY, linestyle=":", linewidth=1)
    axis.set_xticks(range(len(depths)))
    axis.set_xticklabels([f"{depth} layers" for depth in depths])
    axis.set_ylabel("peak reserved memory, worst rank, GiB")
    axis.set_ylim(0, CAPACITY_GIB * 1.25)
    axis.legend(handles=handles, fontsize=8, frameon=False, loc="upper left", ncol=2)
    axis.set_title("How deep the model fits, with and without the swap", fontsize=10)
    _style(axis)
    fig.tight_layout()
    _save(fig, out)


def main() -> int:
    """Read the results and draw."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("results", help="the sweep campaign's results.json")
    parser.add_argument("--depth", default=None, help="the depth campaign's results.json")
    parser.add_argument("--budgets", default="8,3.2", help="budget_layers of the two mechanism panels")
    parser.add_argument("--out-dir", default="docs/images", help="where the PNGs go")
    args = parser.parse_args()
    with open(args.results, encoding="utf-8") as stream:
        results = json.load(stream)
    os.makedirs(args.out_dir, exist_ok=True)
    draw_problem(results, os.path.join(args.out_dir, "ep_host_swap_problem.png"))
    mechanism(results, [float(value) for value in args.budgets.split(",")],
              os.path.join(args.out_dir, "ep_host_swap_mechanism.png"))
    draw_sweep(results, os.path.join(args.out_dir, "ep_host_swap_sweep.png"))
    if args.depth:
        with open(args.depth, encoding="utf-8") as stream:
            draw_depth(json.load(stream), os.path.join(args.out_dir, "ep_host_swap_depth.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
