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
"""Write SYNTHETIC per-rank records in the format of ``hetero_profile``, to try the reports before a run.

Nothing here was measured. The times come from the constants of :class:`Costs`, which are rough FLOP-count
estimates of Qwen3-VL-30B-A3B on an A3 die, and from a timeline simulation with the structure of the real
run: every rank computes its own sample; the weights of each module are gathered by a collective that waits
for the slowest rank (the gap before the module); the experts' all-to-alls wait for the slowest rank of the
expert-parallel group; the step ends when the last rank does. What the reports show about *those* mechanisms is
therefore real in structure and invented in size. The header says ``time_source: synthetic`` and the reports
print a banner.

    python examples/qwen3_vl_30b_perf/synthetic_hetero_records.py --out /tmp/demo/hetero --scenario both

Uses: learning to read the report; checking the analysis at the scale of 32 ranks and 48 layers; the tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

# Run as a script, Python puts this directory first on the import path.
from hetero_sampling import (
    SCENARIOS, arrange, draw_targets, grid_for_tokens, plan_image_count, sample_cost, split_visual,
)

FWD, RECOMPUTE, BWD = "fwd", "recompute", "bwd"
IN, OUT = "in", "out"
DEEPSTACK_AFTER = (8, 16, 24)


@dataclass
class Costs:
    """Assumed costs in milliseconds; see the module docstring. Backward is twice the forward."""

    vision_fixed: float = 0.08
    vision_per_patch: float = 3.4e-4        # 30 MFLOP per patch per block at ~90 TFLOPS
    vision_per_pair: float = 7.7e-8         # full attention inside an image, per patch pair
    merger_per_patch: float = 2.0e-5
    embed_per_patch: float = 6.0e-5
    attn_fixed: float = 0.12
    attn_per_token: float = 4.2e-4          # 38 MFLOP per token
    attn_per_token_sq: float = 1.17e-7      # causal: 2 L^2 d
    router: float = 0.2
    expert_per_pair: float = 1.57e-4        # 9.4 MFLOP per routed pair at ~60 TFLOPS
    a2a_fixed: float = 0.25
    a2a_per_pair: float = 2.7e-5            # 4 KB per pair at ~150 GB/s
    aggregate_per_token: float = 2.0e-5
    norms_per_token: float = 6.0e-5
    head_per_token: float = 6.2e-3          # 622 MFLOP per token at ~100 TFLOPS
    loss_per_token: float = 4.5e-3
    ag_vision: float = 0.5                  # weight gather of one vision block over 32 ranks
    ag_text: float = 1.6                    # weight gather of one decoder layer, dense and experts
    ag_head: float = 4.0
    tail: float = 80.0                      # gradient clip, optimizer, final sync
    bwd_factor: float = 2.0
    jitter: float = 0.01


class Simulator:
    """Plays the ranks of a run through forward, recompute and backward, with the collectives that couple them."""

    def __init__(self, *, ranks: int, ep_size: int, layers: int, blocks: int, experts: int, top_k: int,
                 costs: Costs, seed: int, slow_rank: Optional[int] = None, slow_factor: float = 1.06) -> None:
        """Set up the ranks, their speeds and the module table."""
        self.ranks, self.ep_size, self.layers, self.blocks = ranks, ep_size, layers, blocks
        self.experts, self.top_k, self.costs = experts, top_k, costs
        self.rng = random.Random(seed)
        self.t = [0.0] * ranks
        self.entry = [0.0] * ranks
        self.live = [0.0] * ranks
        self.peak = [0] * ranks
        self.marks: list[list[list[Any]]] = [[] for _ in range(ranks)]
        self.routing_records: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
        self.speed = [1.0 + self.rng.gauss(0.0, 0.008) for _ in range(ranks)]
        if slow_rank is not None:
            self.speed[slow_rank] *= slow_factor
        table = [("root", None), ("vision.root", None), ("vision.patch_embed", None)]
        table += [("vision.block", b) for b in range(blocks)]
        table += [("vision.deepstack", k) for k in range(len(DEEPSTACK_AFTER))] + [("vision.merger", None)]
        table += [("text.root", None), ("text.embed", None)] + [("text.layer", i) for i in range(layers)]
        table += [("text.attn", i) for i in range(layers)] + [("text.moe", i) for i in range(layers)]
        table += [("text.experts", i) for i in range(layers)]
        table += [("text.router", i) for i in range(layers)] + [("lm_head", None)]
        self.modules = [{"id": i, "name": f"{role}.{index}" if index is not None else role, "role": role,
                         "index": index} for i, (role, index) in enumerate(table)]
        self.ids: dict[tuple[str, Optional[int]], int] = {(m["role"], m["index"]): m["id"] for m in self.modules}
        # Expert popularity per layer: the text tokens spread widely, the image tokens pile onto a few experts.
        self.text_p, self.visual_p = [], []
        for _ in range(layers):
            text = [self.rng.gammavariate(1.5, 1.0) for _ in range(experts)]
            visual = [self.rng.gammavariate(0.35, 1.0) for _ in range(experts)]
            text_total, visual_total = sum(text), sum(visual)
            self.text_p.append([x / text_total for x in text])
            self.visual_p.append([0.65 * v / visual_total + 0.35 * t / text_total for v, t in zip(visual, text)])

    # -- helpers -----------------------------------------------------------------------------------------

    def noise(self, rank: int) -> float:
        """Return a rank's speed factor for one stretch of work."""
        return self.speed[rank] * (1.0 + self.rng.gauss(0.0, self.costs.jitter))

    def mark(self, rank: int, module: int, phase: str, occ: int, kind: str) -> None:
        """Record one boundary at the rank's clock."""
        self.marks[rank].append([module, phase, occ, kind, round(self.t[rank], 4), self.allocated(rank)])

    def allocated(self, rank: int) -> int:
        """Return the bytes allocated on a rank now (weights and optimizer state plus what is live)."""
        return int(12.5 * 2 ** 30 + self.live[rank])

    def gate(self, cost: float) -> None:
        """Wait for the weights of the next module: a collective that the slowest rank's arrival releases.

        With one module prefetched ahead, the collective starts when each rank enters the previous module, so
        it completes ``cost`` after the last of them; a rank that is early waits, and that wait is the gap.
        """
        ready = max(self.entry) + cost
        for rank in range(self.ranks):
            self.t[rank] = max(self.t[rank], ready)
            self.entry[rank] = self.t[rank]

    def group_barrier(self, pairs: list[float]) -> None:
        """An expert-parallel all-to-all: every rank of a group leaves after the last arrives, plus the transfer."""
        for start in range(0, self.ranks, self.ep_size):
            group = range(start, min(start + self.ep_size, self.ranks))
            done = max(self.t[r] for r in group) + self.costs.a2a_fixed + self.costs.a2a_per_pair * max(
                pairs[r] for r in group)
            for rank in group:
                self.t[rank] = done

    # -- the work of one micro-batch -------------------------------------------------------------------------

    def routing(self, occ: int, samples: list[dict]) -> list[list[tuple[list[int], list[int]]]]:
        """Return, per layer and rank, the experts chosen by the image tokens and by the text tokens (and log them)."""
        result = []
        for layer in range(self.layers):
            per_rank = []
            for sample in samples:
                visual_pairs = sample["visual_tokens"] * self.top_k
                text_pairs = (sample["real_tokens"] - sample["visual_tokens"]) * self.top_k
                visual = [max(int(round(visual_pairs * p + self.rng.gauss(0, math.sqrt(visual_pairs * p + 1)))), 0)
                          for p in self.visual_p[layer]]
                text = [max(int(round(text_pairs * p + self.rng.gauss(0, math.sqrt(text_pairs * p + 1)))), 0)
                        for p in self.text_p[layer]]
                per_rank.append((visual, text))
            for rank, (visual, text) in enumerate(per_rank):
                self.routing_records[rank].append({"layer": layer, "mb": occ, "visual": visual, "text": text})
            result.append(per_rank)
        return result

    def received(self, routing: list, layer: int) -> list[float]:
        """Return the routed pairs each rank receives in a layer: its group's choices for the experts it owns."""
        local = self.experts // self.ep_size
        received = [0.0] * self.ranks
        for start in range(0, self.ranks, self.ep_size):
            members = range(start, min(start + self.ep_size, self.ranks))
            totals = [0] * self.experts
            for rank in members:
                visual, text = routing[layer][rank]
                for expert in range(self.experts):
                    totals[expert] += visual[expert] + text[expert]
            for position, rank in enumerate(members):
                received[rank] = float(sum(totals[position * local:(position + 1) * local]))
        return received

    def layer_pass(self, phase: str, occ: int, layer: int, samples: list[dict], routing: list,
                   factor: float = 1.0) -> None:
        """One decoder layer, forward or recompute: attention, then the MoE block with its two all-to-alls."""
        c, ids = self.costs, self.ids
        recv = self.received(routing, layer)
        for r in range(self.ranks):
            self.mark(r, ids["text.layer", layer], phase, occ, IN)
            self.t[r] += c.norms_per_token * samples[r]["real_tokens"] * 0.5 * factor
            self.mark(r, ids["text.attn", layer], phase, occ, IN)
            tokens = samples[r]["real_tokens"]
            self.t[r] += (c.attn_fixed + c.attn_per_token * tokens + c.attn_per_token_sq * tokens * tokens) \
                * self.noise(r) * factor
            self.mark(r, ids["text.attn", layer], phase, occ, OUT)
            self.mark(r, ids["text.moe", layer], phase, occ, IN)
            self.t[r] += c.router * self.noise(r) * factor
        self.group_barrier(recv)
        for r in range(self.ranks):
            self.mark(r, ids["text.experts", layer], phase, occ, IN)
            self.t[r] += c.expert_per_pair * recv[r] * self.noise(r) * factor
            self.mark(r, ids["text.experts", layer], phase, occ, OUT)
        self.group_barrier(recv)
        for r in range(self.ranks):
            self.t[r] += c.aggregate_per_token * samples[r]["real_tokens"] * factor
            self.mark(r, ids["text.moe", layer], phase, occ, OUT)
            self.t[r] += c.norms_per_token * samples[r]["real_tokens"] * 0.5 * factor
            self.mark(r, ids["text.layer", layer], phase, occ, OUT)

    def block_pass(self, phase: str, occ: int, block: int, samples: list[dict], factor: float = 1.0) -> None:
        """One vision block."""
        c, ids = self.costs, self.ids
        for r in range(self.ranks):
            self.mark(r, ids["vision.block", block], phase, occ, IN)
            self.t[r] += (c.vision_fixed + c.vision_per_patch * samples[r]["patches"]
                          + c.vision_per_pair * samples[r]["vision_attn_pairs"]) * self.noise(r) * factor
            self.mark(r, ids["vision.block", block], phase, occ, OUT)

    def micro_batch(self, occ: int, samples: list[dict]) -> None:
        """Forward, then backward with recompute, for one micro-batch on every rank."""
        c, ids = self.costs, self.ids
        routing = self.routing(occ, samples)
        for r in range(self.ranks):
            self.live[r] = 0.0
            self.mark(r, ids["root", None], FWD, occ, IN)
            self.mark(r, ids["vision.root", None], FWD, occ, IN)
            self.mark(r, ids["vision.patch_embed", None], FWD, occ, IN)
            self.t[r] += (0.1 + c.embed_per_patch * samples[r]["patches"]) * self.noise(r)
            self.mark(r, ids["vision.patch_embed", None], FWD, occ, OUT)
        self.entry = list(self.t)
        for block in range(self.blocks):
            self.gate(c.ag_vision)
            self.block_pass(FWD, occ, block, samples)
            if block in DEEPSTACK_AFTER:
                index = DEEPSTACK_AFTER.index(block)
                for r in range(self.ranks):
                    self.mark(r, ids["vision.deepstack", index], FWD, occ, IN)
                    self.t[r] += (0.15 + c.merger_per_patch * samples[r]["patches"]) * self.noise(r)
                    self.mark(r, ids["vision.deepstack", index], FWD, occ, OUT)
        for r in range(self.ranks):
            self.mark(r, ids["vision.merger", None], FWD, occ, IN)
            self.t[r] += (0.15 + c.merger_per_patch * samples[r]["patches"]) * self.noise(r)
            self.mark(r, ids["vision.merger", None], FWD, occ, OUT)
            self.mark(r, ids["vision.root", None], FWD, occ, OUT)
            self.mark(r, ids["text.root", None], FWD, occ, IN)
            self.mark(r, ids["text.embed", None], FWD, occ, IN)
            self.t[r] += 0.1 + 1e-4 * samples[r]["real_tokens"]
            self.mark(r, ids["text.embed", None], FWD, occ, OUT)
        self.entry = list(self.t)
        for layer in range(self.layers):
            self.gate(c.ag_text)
            self.layer_pass(FWD, occ, layer, samples, routing)
            for r in range(self.ranks):
                self.live[r] += samples[r]["real_tokens"] * 4096.0
        for r in range(self.ranks):
            self.mark(r, ids["text.root", None], FWD, occ, OUT)
        self.gate(c.ag_head)
        for r in range(self.ranks):
            tokens = samples[r]["real_tokens"]
            self.mark(r, ids["lm_head", None], FWD, occ, IN)
            self.t[r] += (0.2 + c.head_per_token * tokens) * self.noise(r)
            self.live[r] += tokens * 151936 * 6.0
            self.mark(r, ids["lm_head", None], FWD, occ, OUT)
            self.mark(r, ids["root", None], FWD, occ, OUT)
            self.t[r] += (0.3 + c.loss_per_token * tokens) * self.noise(r)
            self.live[r] += tokens * 151936 * 4.0
            self.peak[r] = max(self.peak[r], self.allocated(r))
            self.mark(r, ids["lm_head", None], BWD, occ, IN)
            self.t[r] += c.bwd_factor * (0.2 + c.head_per_token * tokens) * self.noise(r)
            self.live[r] -= tokens * 151936 * 10.0
            self.mark(r, ids["lm_head", None], BWD, occ, OUT)
        self.entry = list(self.t)
        for layer in reversed(range(self.layers)):
            self.gate(c.ag_text)
            self.layer_pass(RECOMPUTE, occ, layer, samples, routing)
            recv = self.received(routing, layer)
            for r in range(self.ranks):
                self.mark(r, ids["text.layer", layer], BWD, occ, IN)
                self.mark(r, ids["text.moe", layer], BWD, occ, IN)
                self.t[r] += c.aggregate_per_token * samples[r]["real_tokens"] * c.bwd_factor
            self.group_barrier(recv)
            for r in range(self.ranks):
                self.mark(r, ids["text.experts", layer], BWD, occ, IN)
                self.t[r] += c.bwd_factor * c.expert_per_pair * recv[r] * self.noise(r)
                self.mark(r, ids["text.experts", layer], BWD, occ, OUT)
            self.group_barrier(recv)
            for r in range(self.ranks):
                self.t[r] += c.bwd_factor * c.router * self.noise(r)
                self.mark(r, ids["text.moe", layer], BWD, occ, OUT)
                self.mark(r, ids["text.attn", layer], BWD, occ, IN)
                tokens = samples[r]["real_tokens"]
                self.t[r] += c.bwd_factor * (c.attn_fixed + c.attn_per_token * tokens
                                             + c.attn_per_token_sq * tokens * tokens) * self.noise(r)
                self.mark(r, ids["text.attn", layer], BWD, occ, OUT)
                self.t[r] += c.norms_per_token * tokens * c.bwd_factor
                self.mark(r, ids["text.layer", layer], BWD, occ, OUT)
                self.live[r] -= tokens * 4096.0
        for r in range(self.ranks):
            self.mark(r, ids["vision.merger", None], BWD, occ, IN)
            self.t[r] += c.bwd_factor * (0.15 + c.merger_per_patch * samples[r]["patches"]) * self.noise(r)
            self.mark(r, ids["vision.merger", None], BWD, occ, OUT)
        self.entry = list(self.t)
        for block in reversed(range(self.blocks)):
            self.gate(c.ag_vision)
            self.block_pass(RECOMPUTE, occ, block, samples)
            for r in range(self.ranks):
                self.mark(r, ids["vision.block", block], BWD, occ, IN)
                self.t[r] += c.bwd_factor * (c.vision_fixed + c.vision_per_patch * samples[r]["patches"]
                                             + c.vision_per_pair * samples[r]["vision_attn_pairs"]) * self.noise(r)
                self.mark(r, ids["vision.block", block], BWD, occ, OUT)

    def step(self, batches: list[list[dict]]) -> list[dict[str, Any]]:
        """Run one optimizer step (``batches[m][rank]`` is rank's sample in micro-batch m); return a record per rank."""
        self.t = [0.0] * self.ranks
        self.marks = [[] for _ in range(self.ranks)]
        self.routing_records = [[] for _ in range(self.ranks)]
        self.live = [0.0] * self.ranks
        self.peak = [0] * self.ranks
        self.entry = [0.0] * self.ranks
        for occ, samples in enumerate(batches):
            self.micro_batch(occ, samples)
        finish = max(self.t) + self.costs.tail
        return [{"device_ms": round(finish, 3), "wall_ms": round(finish + self.rng.uniform(0.5, 2.0), 3),
                 "marks": self.marks[r], "routing": self.routing_records[r], "peak": self.peak[r]}
                for r in range(self.ranks)]


def draw_samples(count: int, scenario: str, *, seed: int, mean_len: int = 8192, len_cv: float = 0.8,
                 mean_visual: int = 2048, visual_cv: float = 1.0, max_len: int = 16384) -> list[dict[str, Any]]:
    """Draw ``count`` samples of a scenario: tokens, visual tokens, images, patches and the vision attention cost."""
    rng = random.Random(seed)
    spec = SCENARIOS[scenario]
    targets = draw_targets(
        count, mean_len=mean_len, len_cv=len_cv if spec.len_cv is None else spec.len_cv, mean_visual=mean_visual,
        visual_cv=visual_cv if spec.visual_cv is None else spec.visual_cv, corr=spec.corr, min_len=1024,
        max_len=max_len, min_visual=64, rng=rng)
    samples = []
    for index, (length, visual) in enumerate(targets):
        images = plan_image_count(visual, 64, 2048, 16, rng)
        tokens = split_visual(visual, images, 64, 2048, rng)
        grids = [grid_for_tokens(n, rng.uniform(0.5, 2.0)) for n in tokens]
        patches = [4 * rows * cols for rows, cols in grids]
        samples.append({
            "tokens": length, "real_tokens": length, "label_tokens": rng.randint(32, 600), "batch_size": 1,
            "images": images, "videos": 0, "patches": sum(patches), "visual_tokens": sum(p // 4 for p in patches),
            "image_tokens": sum(p // 4 for p in patches), "vision_attn_pairs": sum(p * p for p in patches),
            "pixel_rows": sum(patches), "grids": [[1, 2 * rows, 2 * cols] for rows, cols in grids][:64],
            "fingerprint": 10_000_000 + index * 7919 + rng.randint(0, 6000),
        })
    return samples


def write_run(out: str, *, scenario: str, ranks: int, steps: int, accumulation: int, layers: int, blocks: int,
              ep_size: int, experts: int, seed: int, slow_rank: Optional[int], first_step: int = 3,
              order: str = "random") -> None:
    """Simulate a run and write one ``rank*.jsonl`` per rank into ``out``.

    ``order`` is how the drawn samples are dealt to the steps: ``random`` (the draw order) or ``balanced`` (sorted by
    cost into steps of alike samples, as ``prepare_hetero_data.py --arrange balanced`` writes them).
    """
    simulator = Simulator(ranks=ranks, ep_size=ep_size, layers=layers, blocks=blocks, experts=experts, top_k=8,
                          costs=Costs(), seed=seed, slow_rank=slow_rank)
    samples = draw_samples(ranks * accumulation * steps, scenario, seed=seed)
    ranked = arrange([sample_cost({"tokens": x["real_tokens"], "visual_tokens": x["visual_tokens"]}, 0.5)
                      for x in samples], ranks, order, random.Random(seed + 1))
    samples = [samples[index] for index in ranked]
    os.makedirs(out, exist_ok=True)
    header = {"kind": "header", "world_size": ranks, "host": "synthetic", "device_type": "npu",
              "time_source": "synthetic", "spatial_merge_size": 2, "hooks": True, "modules": simulator.modules}
    files = []
    for rank in range(ranks):
        stream = open(os.path.join(out, f"rank{rank:03d}.jsonl"), "w", encoding="utf-8")  # pylint: disable=R1732
        stream.write(json.dumps({**header, "rank": rank}) + "\n")
        files.append(stream)
    rng = random.Random(seed + 1)
    for step in range(steps):
        batches = []
        for occ in range(accumulation):
            offset = (step * accumulation + occ) * ranks
            batches.append(samples[offset:offset + ranks])      # the sampler deals consecutive samples to the ranks
        records = simulator.step(batches)
        loss = 2.2 - 0.02 * step + rng.gauss(0.0, 0.01)
        for rank, record in enumerate(records):
            mine = [batches[occ][rank] for occ in range(accumulation)]
            line = {
                "kind": "step", "step": first_step + step, "wall_ms": record["wall_ms"],
                "device_ms": record["device_ms"],
                "inter_step_ms": None if step == 0 else round(6.0 + 0.003 * mine[0]["visual_tokens"]
                                                              + (150.0 if rng.random() < 0.02 else 0.0), 3),
                "micro_batches": mine, "marks": record["marks"], "routing": record["routing"],
                "peak_allocated": record["peak"], "peak_reserved": None, "loss": round(loss, 5), "grad_norm": 1.0,
            }
            files[rank].write(json.dumps(line) + "\n")
    for stream in files:
        stream.close()
    with open(os.path.join(out, "SYNTHETIC.txt"), "w", encoding="utf-8") as note:
        note.write("These records were generated by synthetic_hetero_records.py from assumed costs.\n"
                   "Nothing in them was measured on a device.\n")


def main() -> int:
    """Parse the command line and write the records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory of the rank files (a run's hetero/ directory)")
    parser.add_argument("--scenario", choices=sorted(set(SCENARIOS) - {"natural"}), default="both")
    parser.add_argument("--ranks", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--accumulation", type=int, default=1, help="micro-batches per rank and step")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=27)
    parser.add_argument("--ep-size", type=int, default=16)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slow-rank", type=int, default=None, help="make one die 6%% slower")
    parser.add_argument("--order", choices=("random", "balanced"), default="random",
                        help="deal the samples in the draw order, or sorted by cost into steps of alike samples")
    args = parser.parse_args()
    write_run(args.out, scenario=args.scenario, ranks=args.ranks, steps=args.steps, accumulation=args.accumulation,
              layers=args.layers, blocks=args.blocks, ep_size=args.ep_size, experts=args.experts, seed=args.seed,
              slow_rank=args.slow_rank, order=args.order)
    print(f"wrote {args.ranks} rank files of {args.steps} steps to {args.out}: SYNTHETIC, nothing measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
