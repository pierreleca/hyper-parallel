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
"""MoE activation budget for a forward pass, earliest layers swapped to host (CPU path)."""

import importlib.util
import json
import pathlib
import random
import socket
import sys
from typing import Any, Iterator

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward
from hyper_parallel.distributed.expert_parallel.host_swap import HOST_SWAP, EPHostSwap, choose_offload
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS

HIDDEN, INTER = 16, 8


@pytest.fixture(name="swap")
def fixture_swap(tmp_path: pathlib.Path) -> Iterator[EPHostSwap]:
    """Enable the swap at 2.7 mean layers (below 3 layers at the mean), small row threshold; disable it after."""
    HOST_SWAP.configure(enabled=True, budget_layers=2.7, min_row_bytes=16, output_dir=str(tmp_path))
    HOST_SWAP.begin_step(1)
    yield HOST_SWAP
    HOST_SWAP.close()
    HOST_SWAP.configure(enabled=False, budget_layers=0.0, min_row_bytes=1024, output_dir="")
    HOST_SWAP.begin_step(0)


def _block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """A SwiGLU expert block that saves three tensors with one row per routed pair."""
    gate, up = (x @ w1).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w2


def _run(rows_per_layer: list[int], sent: int, swap_on: bool) -> tuple[list[torch.Tensor], list]:
    """Run a stack of blocks with the given received rows; return the input gradients and layers."""
    torch.manual_seed(0)
    weights = [(torch.randn(HIDDEN, 2 * INTER, requires_grad=True), torch.randn(INTER, HIDDEN, requires_grad=True))
               for _ in rows_per_layer]
    inputs = [torch.randn(rows, HIDDEN, requires_grad=True) for rows in rows_per_layer]
    total = 0
    for x, (w1, w2) in zip(inputs, weights):
        if swap_on:
            with HOST_SWAP.layer(x.shape[0], sent):
                y = _block(x, w1, w2)
        else:
            y = _block(x, w1, w2)
        total = total + y.square().sum()
    layers = list(HOST_SWAP._swapped) if swap_on else []  # pylint: disable=protected-access
    total.backward()
    return [x.grad for x in inputs] + [w.grad for pair in weights for w in pair], layers


def test_choose_offload_takes_the_smallest_covering_set():
    """The chosen tensors hold at least what must leave, as little more as possible."""
    sizes = [4096, 3072, 1536]
    assert choose_offload(sizes, 0) == []
    assert choose_offload(sizes, 1000) == [2]
    assert choose_offload(sizes, 2000) == [1]
    assert choose_offload(sizes, 3500) == [0]
    assert choose_offload(sizes, 4200) == [1, 2]  # 4608 bytes, less than 4096 + 1536
    assert choose_offload(sizes, 9000) == [0, 1, 2]


def test_pinned_pool_hands_out_the_best_fit_among_mixed_sizes():
    """Buffers of several sizes come back to the pool and go out again by best fit."""
    from hyper_parallel.distributed.expert_parallel.host_swap import _PinnedPool  # pylint: disable=import-outside-toplevel

    pool = _PinnedPool(pin=False)
    small, large = pool.take(1), pool.take(3 * 64 * 1024 * 1024)
    assert small.numel() < large.numel()
    pool.give(large)
    pool.give(small)
    assert pool.take(10) is small, "the smallest buffer that fits"
    assert pool.take(10) is large
    assert not pool.free


def test_swapped_layers_give_the_same_gradients(swap):
    """Layers over the budget move tensors to host and back; the gradients do not change."""
    rows = [12, 6, 15]  # 33 rows against a budget of 2.7 x 8
    expected, _ = _run(rows, sent=8, swap_on=False)
    swap.begin_step(1)
    got, layers = _run(rows, sent=8, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    assert layers and layers[0].index == 0, "the earliest layer goes first"
    for layer in layers:
        assert all(item.device is None for item in layer.swapped), "released after backward"


def test_record_counts_the_moved_bytes(swap, tmp_path):
    """The step record lists every swapped layer and the bytes both ways."""
    swap.configure(enabled=True, budget_layers=1.0, min_row_bytes=16, output_dir=str(tmp_path))
    _run([12, 6], sent=8, swap_on=True)  # layer 0 alone is over one mean layer
    record = swap.end_step(rank=0)
    assert record["moe_layers"] == 2 and record["swapped_layers"] == 1
    assert record["d2h_gib"] == record["h2d_gib"] > 0
    lines = (tmp_path / "host_swap_rank0.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["header"] and json.loads(lines[1])["layers"][0]["index"] == 0


def test_nothing_is_tracked_without_grad(swap):
    """Inference and recompute-free no-grad passes are left alone."""
    with torch.no_grad(), swap.layer(12, 8):
        _block(torch.randn(12, HIDDEN), torch.randn(HIDDEN, 2 * INTER), torch.randn(INTER, HIDDEN))
    assert swap.end_step(rank=0) is None


class _Experts(nn.Module):
    """Stacked experts with an expert-major grouped path that saves per-pair tensors."""

    def __init__(self, num_experts: int = 8) -> None:
        """Create expert weights small enough for a CPU test."""
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * INTER, HIDDEN) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, HIDDEN, INTER) * 0.1)

    def forward_expert_major(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Per-pair expert weights gathered by expert id, as a grouped GEMM would apply them."""
        expert_ids = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        gate_up = torch.einsum("nh,nih->ni", x, self.gate_up_proj[expert_ids])
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.einsum("ni,nhi->nh", F.silu(gate) * up, self.down_proj[expert_ids])


class _Moe(nn.Module):
    """Minimal MoE block with the interface ep_routed_forward expects."""

    def __init__(self) -> None:
        """Build the router and the stacked experts."""
        super().__init__()
        self.gate = nn.Linear(HIDDEN, 8, bias=False)
        self.experts = _Experts()
        self.top_k = 2


@pytest.fixture(name="world_one_group")
def fixture_world_one_group() -> Iterator[Any]:
    """Provide a single-process gloo group for the all-to-all primitives."""
    created = False
    if not dist.is_initialized():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
        created = True
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


def _ep_grads(world_one_group, swap_on: bool) -> list[torch.Tensor]:
    """Two EP MoE blocks forward and backward; return the input and parameter gradients."""
    torch.manual_seed(1)
    model = nn.ModuleList([_Moe(), _Moe()])
    for block in model:
        bind_local_expert_forward(block, ep_size=1)
        block.experts.ep_use_grouped_gemm = True
    hidden = torch.randn(1, 6, HIDDEN, requires_grad=True)
    out = hidden
    for block in model:
        out = ep_routed_forward(block, out, router_fn=MOE_ROUTER_ADAPTERS["default"], ep_group=world_one_group)
    out.square().sum().backward()
    del swap_on
    return [hidden.grad] + [param.grad for param in model.parameters()]


def test_ep_blocks_swap_and_keep_their_gradients(world_one_group, swap):
    """Through ep_routed_forward, a budget below the load swaps and changes nothing."""
    swap.configure(enabled=False, budget_layers=0.0, min_row_bytes=16, output_dir="")
    expected = _ep_grads(world_one_group, swap_on=False)
    # One rank receives what it sends; one mean layer for two layers puts the pass over it.
    swap.configure(enabled=True, budget_layers=1.0, min_row_bytes=16, output_dir="")
    swap.begin_step(1)
    got = _ep_grads(world_one_group, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    record = swap.end_step(rank=0)
    assert record["moe_layers"] == 2 and record["swapped_layers"] >= 1
    assert all(layer["swapped_bytes"] > 0 for layer in record["layers"])


def test_a_budget_under_the_layers_swaps_balanced_layers_and_keeps_gradients(swap):
    """Under one mean layer per layer, the pass swaps even when every layer is at the mean."""
    rows = [8, 8, 8]  # every layer exactly at the mean load
    expected, _ = _run(rows, sent=8, swap_on=False)
    swap.begin_step(1)
    got, layers = _run(rows, sent=8, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    assert layers and min(layer.index for layer in layers) == 0, "the earliest layer goes first"
    record = swap.end_step(rank=0)
    assert record["evictions"]
    for decision in record["evictions"]:
        assert decision["held_gib"] - decision["evicted_gib"] <= decision["threshold_gib"] + 1e-12


def test_a_budget_under_the_layers_spreads_the_evictions(swap):
    """Once the pass length is known, the deficit starts leaving after the first layer, not at the end."""
    _run([8, 8, 8], sent=8, swap_on=True)
    first = swap.end_step(rank=0)["evictions"]
    assert [decision["after_layer"] for decision in first] == [2], "the first pass budgets the layers it has seen"
    swap.begin_step(2)
    _run([8, 8, 8], sent=8, swap_on=True)
    second = swap.end_step(rank=0)["evictions"]
    assert second[0]["after_layer"] == 0 and second[0]["expected_layers"] == 3
    # 2.7 of 3 mean layers: after the first layer the threshold is 0.9 of it, the budget spread evenly.
    assert second[0]["threshold_gib"] == pytest.approx(0.9 * second[0]["budget_gib"] / 2.7)


def test_a_budget_over_the_layers_lets_an_early_peak_through(swap):
    """Over one mean layer per layer, a rank heavy early but light later moves nothing."""
    swap.configure(enabled=True, budget_layers=3.3, min_row_bytes=16, output_dir="")
    for step in (1, 2):  # the second pass knows it has three layers
        swap.begin_step(step)
        # 10 rows first is over 3.3 / 3 of the budget, but not over what it leaves for two mean layers.
        _grads, layers = _run([10, 6, 7], sent=8, swap_on=True)  # 23 rows against 3.3 x 8 = 26.4
    assert not layers, "ahead of the mean after the first layer, yet within the budget at the end"


def test_budget_above_the_spread_swaps_nothing(swap):
    """A budget over every rank's total leaves the pass on device."""
    swap.configure(enabled=True, budget_layers=4.5, min_row_bytes=16, output_dir="")
    swap.begin_step(1)
    _grads, layers = _run([12, 6, 9], sent=8, swap_on=True)  # total 27 rows against 4.5 x 8
    assert not layers


def test_a_layer_comes_back_only_from_the_layer_above(swap, monkeypatch):
    """Swapped early layers come back one layer ahead of their backward, not at its start."""
    swap.configure(enabled=True, budget_layers=2.0, min_row_bytes=16, output_dir="")
    swap.begin_step(1)
    trace = []
    unpack, load = swap._unpack, swap._load  # pylint: disable=protected-access
    monkeypatch.setattr(swap, "_unpack", lambda packed: (
        trace.append(("unpack", packed.layer.index)) if hasattr(packed, "layer") else None, unpack(packed))[1])
    monkeypatch.setattr(swap, "_load", lambda layer: (trace.append(("load", layer.index)), load(layer))[0])
    _grads, layers = _run([8, 8, 8, 8], sent=8, swap_on=True)
    for layer in layers:
        first_unpack_above = trace.index(("unpack", layer.index + 1))
        assert trace.index(("load", layer.index)) > first_unpack_above, f"layer {layer.index} loaded early"


def test_budget_layers_must_be_positive_when_enabled():
    """A budget of zero or less could hold nothing."""
    with pytest.raises(ValueError, match="positive"):
        EPHostSwap().configure(enabled=True, budget_layers=0.0, min_row_bytes=16, output_dir="")


def test_the_replay_script_picks_tensors_as_the_swap_does(monkeypatch):
    """replay_ep_host_swap.py carries its own copy of choose_offload; it must agree with the swap's."""
    examples = pathlib.Path(__file__).resolve().parents[3] / "examples" / "qwen3_vl_30b_perf"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("replay_ep_host_swap", examples / "replay_ep_host_swap.py")
    replay = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "replay_ep_host_swap", replay)  # its dataclasses look the module up
    spec.loader.exec_module(replay)
    generator = random.Random(0)
    for _ in range(200):
        sizes = [generator.randint(1, 5000) for _ in range(generator.randint(1, 6))]
        need = generator.randint(-10, sum(sizes) + 10)
        assert replay.choose_offload(sizes, need) == choose_offload(sizes, need), (sizes, need)
