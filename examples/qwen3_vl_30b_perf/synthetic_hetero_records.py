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

With ``--ascend-ranks`` it also writes, for the last step of those ranks, an Ascend-profiler-like ``trace_view.json``
(kernels, stream waits, collectives, a ProfilerStep range) at an arbitrary time offset, so that ``component_trace.py``
can be tried end to end: it has to find the offset again and draw the components over the kernels.

Uses: learning to read the report; checking the analysis at the scale of 32 ranks and 48 layers; the tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

# Run as a script, Python puts this directory first on the import path.
from hetero_sampling import (
    SCENARIOS, arrange, draw_targets, grid_for_tokens, plan_image_count, sample_cost, split_visual,
)

FWD, RECOMPUTE, BWD = "fwd", "recompute", "bwd"
IN, OUT = "in", "out"
DEEPSTACK_AFTER = (8, 16, 24)
# The kernels a stretch of work runs, as (name, share of the stretch), one after the other; the rest is idle.
# Only for the Ascend-like trace that the simulator can write alongside the records.
KERNELS = {
    "attn": (("aclnnFlashAttentionScore", 0.70), ("aclnnMatmul", 0.14), ("aclnnRmsNorm", 0.06)),
    "experts": (("aclnnGroupedMatmulV4", 0.26), ("aclnnGroupedMatmulV4", 0.26), ("aclnnGroupedMatmulV4", 0.26),
                ("aclnnSwiGlu", 0.08), ("aclnnMoeTokenPermute", 0.05)),
    "vision": (("aclnnMatmul", 0.40), ("aclnnFlashAttentionVarLenScore", 0.30), ("aclnnLayerNorm", 0.08),
               ("aclnnGelu", 0.06)),
    "head": (("aclnnMatmul", 0.92),),
    "loss": (("aclnnCrossEntropyLoss", 0.85),),
    "small": (("aclnnMatmul", 0.60),),
    "router": (("aclnnMoeInitRouting", 0.80),),
    "aggregate": (("aclnnIndexAdd", 0.80),),
    "norm": (("aclnnRmsNorm", 0.80),),
}


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
                 costs: Costs, seed: int, slow_rank: Optional[int] = None, slow_factor: float = 1.06,
                 checkpoint: str = "reentrant") -> None:
        """Set up the ranks, their speeds and the module table.

        ``checkpoint`` is how the layers are recomputed: ``reentrant`` runs the whole layer again before its
        backward; ``nonreentrant`` (what the Trainer's wrapper does) recomputes it lazily inside the backward of
        its last module, and stops as soon as its last saved tensor is back, i.e. inside the MoE block.
        """
        self.checkpoint = checkpoint
        self.ranks, self.ep_size, self.layers, self.blocks = ranks, ep_size, layers, blocks
        self.experts, self.top_k, self.costs = experts, top_k, costs
        self.rng = random.Random(seed)
        self.t = [0.0] * ranks
        self.entry = [0.0] * ranks
        self.live = [0.0] * ranks
        self.peak = [0] * ranks
        self.marks: list[list[list[Any]]] = [[] for _ in range(ranks)]
        self.routing_records: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
        # What each rank's device ran in the last step, for the trace: (kind, name, start ms, duration ms).
        self.kernels: list[list[tuple[str, str, float, float]]] = [[] for _ in range(ranks)]
        self.collectives = 0
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

    def work(self, rank: int, duration: float, spec: Optional[str] = None) -> None:
        """Advance a rank's clock by ``duration`` of compute, and note the kernels it ran (``spec`` names them)."""
        start = self.t[rank]
        self.t[rank] += duration
        if spec is not None and duration > 0:
            at = start
            for name, share in KERNELS[spec]:
                self.kernels[rank].append(("kernel", name, at, duration * share))
                at += duration * share

    def wait(self, rank: int, until: float, collective: str, issued: float) -> None:
        """Hold a rank until ``until`` and note the stream wait and the collective it waited for."""
        start = self.t[rank]
        self.collectives += 1
        self.kernels[rank].append(("comm", f"hcom_{collective}__{self.collectives}_{rank}_1", issued,
                                   max(until - issued, 0.0)))
        if until > start:
            self.kernels[rank].append(("wait", "EVENT_WAIT", start, until - start))
        self.t[rank] = max(start, until)

    def gate(self, cost: float) -> None:
        """Wait for the weights of the next module: a collective that the slowest rank's arrival releases.

        With one module prefetched ahead, the collective starts when each rank enters the previous module, so
        it completes ``cost`` after the last of them; a rank that is early waits, and that wait is the gap.
        """
        ready = max(self.entry) + cost
        for rank in range(self.ranks):
            self.wait(rank, ready, "allGather", self.entry[rank])
            self.entry[rank] = self.t[rank]

    def group_barrier(self, pairs: list[float]) -> None:
        """An expert-parallel all-to-all: every rank of a group leaves after the last arrives, plus the transfer."""
        for start in range(0, self.ranks, self.ep_size):
            group = range(start, min(start + self.ep_size, self.ranks))
            done = max(self.t[r] for r in group) + self.costs.a2a_fixed + self.costs.a2a_per_pair * max(
                pairs[r] for r in group)
            for rank in group:
                self.wait(rank, done, "alltoallv", self.t[rank])

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
                   factor: float = 1.0, interrupted: bool = False) -> None:
        """One decoder layer, forward or recompute: attention, then the MoE block with its two all-to-alls.

        ``interrupted`` is the lazy recompute of non-reentrant checkpointing: the layer's own hooks sit outside the
        checkpoint and do not see it, and the MoE block is abandoned before it returns, so it never closes.
        """
        c, ids = self.costs, self.ids
        recv = self.received(routing, layer)
        for r in range(self.ranks):
            if not interrupted:
                self.mark(r, ids["text.layer", layer], phase, occ, IN)
            self.work(r, c.norms_per_token * samples[r]["real_tokens"] * 0.5 * factor, "norm")
            self.mark(r, ids["text.attn", layer], phase, occ, IN)
            tokens = samples[r]["real_tokens"]
            self.work(r, (c.attn_fixed + c.attn_per_token * tokens + c.attn_per_token_sq * tokens * tokens)
                      * self.noise(r) * factor, "attn")
            self.mark(r, ids["text.attn", layer], phase, occ, OUT)
            self.mark(r, ids["text.moe", layer], phase, occ, IN)
            self.work(r, c.router * self.noise(r) * factor, "router")
        self.group_barrier(recv)
        for r in range(self.ranks):
            self.mark(r, ids["text.experts", layer], phase, occ, IN)
            self.work(r, c.expert_per_pair * recv[r] * self.noise(r) * factor, "experts")
            self.mark(r, ids["text.experts", layer], phase, occ, OUT)
        self.group_barrier(recv)
        for r in range(self.ranks):
            self.work(r, c.aggregate_per_token * samples[r]["real_tokens"] * factor, "aggregate")
            if interrupted:
                continue
            self.mark(r, ids["text.moe", layer], phase, occ, OUT)
            self.work(r, c.norms_per_token * samples[r]["real_tokens"] * 0.5 * factor, "norm")
            self.mark(r, ids["text.layer", layer], phase, occ, OUT)

    def block_pass(self, phase: str, occ: int, block: int, samples: list[dict], factor: float = 1.0) -> None:
        """One vision block."""
        c, ids = self.costs, self.ids
        for r in range(self.ranks):
            self.mark(r, ids["vision.block", block], phase, occ, IN)
            self.work(r, (c.vision_fixed + c.vision_per_patch * samples[r]["patches"]
                          + c.vision_per_pair * samples[r]["vision_attn_pairs"]) * self.noise(r) * factor, "vision")
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
            self.work(r, (0.1 + c.embed_per_patch * samples[r]["patches"]) * self.noise(r), "small")
            self.mark(r, ids["vision.patch_embed", None], FWD, occ, OUT)
        self.entry = list(self.t)
        for block in range(self.blocks):
            self.gate(c.ag_vision)
            self.block_pass(FWD, occ, block, samples)
            if block in DEEPSTACK_AFTER:
                index = DEEPSTACK_AFTER.index(block)
                for r in range(self.ranks):
                    self.mark(r, ids["vision.deepstack", index], FWD, occ, IN)
                    self.work(r, (0.15 + c.merger_per_patch * samples[r]["patches"]) * self.noise(r), "small")
                    self.mark(r, ids["vision.deepstack", index], FWD, occ, OUT)
        for r in range(self.ranks):
            self.mark(r, ids["vision.merger", None], FWD, occ, IN)
            self.work(r, (0.15 + c.merger_per_patch * samples[r]["patches"]) * self.noise(r), "small")
            self.mark(r, ids["vision.merger", None], FWD, occ, OUT)
            self.mark(r, ids["vision.root", None], FWD, occ, OUT)
            self.mark(r, ids["text.root", None], FWD, occ, IN)
            self.mark(r, ids["text.embed", None], FWD, occ, IN)
            self.work(r, 0.1 + 1e-4 * samples[r]["real_tokens"], "small")
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
            self.work(r, (0.2 + c.head_per_token * tokens) * self.noise(r), "head")
            self.live[r] += tokens * 151936 * 6.0
            self.mark(r, ids["lm_head", None], FWD, occ, OUT)
            self.mark(r, ids["root", None], FWD, occ, OUT)
            self.work(r, (0.3 + c.loss_per_token * tokens) * self.noise(r), "loss")
            self.live[r] += tokens * 151936 * 4.0
            self.peak[r] = max(self.peak[r], self.allocated(r))
            self.mark(r, ids["lm_head", None], BWD, occ, IN)
            self.work(r, c.bwd_factor * (0.2 + c.head_per_token * tokens) * self.noise(r), "head")
            self.live[r] -= tokens * 151936 * 10.0
            self.mark(r, ids["lm_head", None], BWD, occ, OUT)
        self.entry = list(self.t)
        lazy = self.checkpoint == "nonreentrant"
        for layer in reversed(range(self.layers)):
            self.gate(c.ag_text)
            if lazy:           # the gradient reaches the MoE block first; the recompute runs inside its backward
                for r in range(self.ranks):
                    self.mark(r, ids["text.layer", layer], BWD, occ, IN)
                    self.mark(r, ids["text.moe", layer], BWD, occ, IN)
            self.layer_pass(RECOMPUTE, occ, layer, samples, routing, interrupted=lazy)
            recv = self.received(routing, layer)
            for r in range(self.ranks):
                if not lazy:
                    self.mark(r, ids["text.layer", layer], BWD, occ, IN)
                    self.mark(r, ids["text.moe", layer], BWD, occ, IN)
                self.work(r, c.aggregate_per_token * samples[r]["real_tokens"] * c.bwd_factor, "aggregate")
            self.group_barrier(recv)
            for r in range(self.ranks):
                self.mark(r, ids["text.experts", layer], BWD, occ, IN)
                self.work(r, c.bwd_factor * c.expert_per_pair * recv[r] * self.noise(r), "experts")
                self.mark(r, ids["text.experts", layer], BWD, occ, OUT)
            self.group_barrier(recv)
            for r in range(self.ranks):
                self.work(r, c.bwd_factor * c.router * self.noise(r), "router")
                self.mark(r, ids["text.moe", layer], BWD, occ, OUT)
                self.mark(r, ids["text.attn", layer], BWD, occ, IN)
                tokens = samples[r]["real_tokens"]
                self.work(r, c.bwd_factor * (c.attn_fixed + c.attn_per_token * tokens
                                             + c.attn_per_token_sq * tokens * tokens) * self.noise(r), "attn")
                self.mark(r, ids["text.attn", layer], BWD, occ, OUT)
                self.work(r, c.norms_per_token * tokens * c.bwd_factor, "norm")
                self.mark(r, ids["text.layer", layer], BWD, occ, OUT)
                self.live[r] -= tokens * 4096.0
        for r in range(self.ranks):
            self.mark(r, ids["vision.merger", None], BWD, occ, IN)
            self.work(r, c.bwd_factor * (0.15 + c.merger_per_patch * samples[r]["patches"]) * self.noise(r), "small")
            self.mark(r, ids["vision.merger", None], BWD, occ, OUT)
        self.entry = list(self.t)
        for block in reversed(range(self.blocks)):
            self.gate(c.ag_vision)
            self.block_pass(RECOMPUTE, occ, block, samples)
            for r in range(self.ranks):
                self.mark(r, ids["vision.block", block], BWD, occ, IN)
                self.work(r, c.bwd_factor * (c.vision_fixed + c.vision_per_patch * samples[r]["patches"]
                                             + c.vision_per_pair * samples[r]["vision_attn_pairs"]) * self.noise(r),
                          "vision")
                self.mark(r, ids["vision.block", block], BWD, occ, OUT)

    def ascend_events(self, rank: int, offset_us: float, profiler_step: int, finish_ms: float) -> list[dict[str, Any]]:
        """Return the events of an Ascend-profiler-like ``trace_view.json`` for one rank's last step.

        The device clock of the records starts at ``offset_us`` on the trace's timeline. The compute stream carries
        the kernels, the stream waits and one event record per boundary; the collectives sit in the process
        ``Communication``; a ``ProfilerStep#N`` range on the host spans the step with the gap before it.
        """
        hardware, comm, host = 1, 2, 3
        events: list[dict[str, Any]] = [
            {"ph": "M", "name": "process_name", "pid": hardware, "args": {"name": "Ascend Hardware"}},
            {"ph": "M", "name": "thread_name", "pid": hardware, "tid": 7, "args": {"name": "Stream 7"}},
            {"ph": "M", "name": "process_name", "pid": comm, "args": {"name": "Communication"}},
            {"ph": "M", "name": "thread_name", "pid": comm, "tid": 1, "args": {"name": "Group_0 Communication"}},
            {"ph": "M", "name": "process_name", "pid": host, "args": {"name": "CANN"}},
            {"ph": "M", "name": "thread_name", "pid": host, "tid": 1, "args": {"name": "python"}},
            {"ph": "X", "name": f"ProfilerStep#{profiler_step}", "pid": host, "tid": 1,
             "ts": offset_us - 80_000.0, "dur": finish_ms * 1000.0 + 100_000.0, "args": {}},
        ]
        for kind, name, start, duration in self.kernels[rank]:
            if duration <= 0:
                continue
            event = {"ph": "X", "name": name, "ts": offset_us + start * 1000.0, "dur": duration * 1000.0, "args": {}}
            if kind == "comm":
                event.update(pid=comm, tid=1)
            else:
                event.update(pid=hardware, tid=7)
                if kind == "kernel":
                    event["args"] = {"Task Type": "AI_CORE"}
            events.append(event)
        for mark in self.marks[rank]:                       # the recorder's own device events
            events.append({"ph": "X", "name": "EVENT_RECORD", "pid": hardware, "tid": 7,
                           "ts": offset_us + mark[4] * 1000.0, "dur": 1.0, "args": {}})
        return events

    def step(self, batches: list[list[dict]]) -> list[dict[str, Any]]:
        """Run one optimizer step (``batches[m][rank]`` is rank's sample in micro-batch m); return a record per rank."""
        self.t = [0.0] * self.ranks
        self.marks = [[] for _ in range(self.ranks)]
        self.kernels = [[] for _ in range(self.ranks)]
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
              order: str = "random", checkpoint: str = "reentrant", ascend_dir: Optional[str] = None,
              ascend_ranks: Sequence[int] = (), ascend_offset_ms: float = 4321.987,
              ascend_step_shift: int = 1) -> None:
    """Simulate a run and write one ``rank*.jsonl`` per rank into ``out``.

    ``order`` is how the drawn samples are dealt to the steps: ``random`` (the draw order) or ``balanced`` (sorted by
    cost into steps of alike samples, as ``prepare_hetero_data.py --arrange balanced`` writes them). With
    ``ascend_ranks`` the last step of those ranks is also written as an Ascend-like trace under ``ascend_dir``: its
    time origin is at ``ascend_offset_ms`` on the trace's timeline, and its ProfilerStep is numbered
    ``ascend_step_shift`` below the record's step (the mapping ``component_trace.py`` has to find).
    """
    simulator = Simulator(ranks=ranks, ep_size=ep_size, layers=layers, blocks=blocks, experts=experts, top_k=8,
                          costs=Costs(), seed=seed, slow_rank=slow_rank, checkpoint=checkpoint)
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
        if ascend_dir and step == steps - 1:
            for rank in ascend_ranks:
                events = simulator.ascend_events(rank, ascend_offset_ms * 1000.0 + 37.0 * rank,
                                                 first_step + step - ascend_step_shift, records[rank]["device_ms"])
                directory = os.path.join(ascend_dir, f"rank{rank}_20261005_synthetic_ascend_pt",
                                         "ASCEND_PROFILER_OUTPUT")
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, "trace_view.json"), "w", encoding="utf-8") as stream:
                    json.dump({"traceEvents": events}, stream)
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
    parser.add_argument("--checkpoint", choices=("reentrant", "nonreentrant"), default="reentrant",
                        help="how the layers are recomputed (nonreentrant is what the Trainer's wrapper does)")
    parser.add_argument("--order", choices=("random", "balanced"), default="random",
                        help="deal the samples in the draw order, or sorted by cost into steps of alike samples")
    parser.add_argument("--ascend-ranks", type=int, nargs="*", default=[],
                        help="also write an Ascend-like trace of the last step of these ranks")
    parser.add_argument("--ascend-dir", default=None, help="where the traces go (a run's profile/ directory)")
    parser.add_argument("--ascend-offset-ms", type=float, default=4321.987,
                        help="where the step starts on the trace's timeline (component_trace.py has to find it)")
    args = parser.parse_args()
    write_run(args.out, scenario=args.scenario, ranks=args.ranks, steps=args.steps, accumulation=args.accumulation,
              layers=args.layers, blocks=args.blocks, ep_size=args.ep_size, experts=args.experts, seed=args.seed,
              slow_rank=args.slow_rank, order=args.order, checkpoint=args.checkpoint, ascend_dir=args.ascend_dir,
              ascend_ranks=args.ascend_ranks, ascend_offset_ms=args.ascend_offset_ms)
    print(f"wrote {args.ranks} rank files of {args.steps} steps to {args.out}: SYNTHETIC, nothing measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
