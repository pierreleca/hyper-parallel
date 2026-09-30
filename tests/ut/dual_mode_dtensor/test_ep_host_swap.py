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

import socket
from typing import Any, Iterator

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward
from hyper_parallel.distributed.expert_parallel.host_swap import (  # pylint: disable=protected-access
    HOST_SWAP,
    EPHostSwap,
    _PinnedPool,
    choose_offload,
)
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.trainer.config.parser import parse_training_args

HIDDEN, INTER = 16, 8


@pytest.fixture(name="swap")
def fixture_swap() -> Iterator[EPHostSwap]:
    """Enable the swap at 2.7 mean layers (below 3 layers at the mean); disable it after."""
    HOST_SWAP.configure(enabled=True, budget_layers=2.7)
    HOST_SWAP.begin_step()
    yield HOST_SWAP
    HOST_SWAP.configure(enabled=False, budget_layers=0.0)
    HOST_SWAP.begin_step()


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
    swap.begin_step()
    got, layers = _run(rows, sent=8, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    assert layers and layers[0].index == 0, "the earliest layer goes first"
    for layer in layers:
        assert all(item.device is None for item in layer.swapped), "released after backward"


def test_step_summary_counts_the_moved_bytes(swap):
    """The step summary counts the MoE layers, the swapped ones and the bytes sent to host."""
    swap.configure(enabled=True, budget_layers=1.0)
    _run([12, 6], sent=8, swap_on=True)  # layer 0 alone is over one mean layer
    summary = swap.end_step()
    assert summary["moe_layers"] == 2 and summary["swapped_layers"] == 1
    assert summary["d2h_gib"] > 0


def test_nothing_is_tracked_without_grad(swap):
    """Inference and recompute-free no-grad passes are left alone."""
    with torch.no_grad(), swap.layer(12, 8):
        _block(torch.randn(12, HIDDEN), torch.randn(HIDDEN, 2 * INTER), torch.randn(INTER, HIDDEN))
    assert swap.end_step() is None


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
    swap.configure(enabled=False, budget_layers=0.0)
    expected = _ep_grads(world_one_group, swap_on=False)
    # One rank receives what it sends; one mean layer for two layers puts the pass over it.
    swap.configure(enabled=True, budget_layers=1.0)
    swap.begin_step()
    got = _ep_grads(world_one_group, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    summary = swap.end_step()
    assert summary["moe_layers"] == 2 and summary["swapped_layers"] >= 1 and summary["d2h_gib"] > 0


def _spy_decisions(swap: EPHostSwap, monkeypatch) -> list[tuple[int, int, int]]:
    """After each MoE layer: (layer index, bytes held on device, bytes swapped so far) of the pass."""
    decisions = []
    enforce = swap._enforce_budget  # pylint: disable=protected-access

    def enforce_noted(layer: Any) -> None:
        """Apply the rule, then note what the pass holds and has swapped."""
        enforce(layer)
        current = swap._layers[swap._pass_start:]  # pylint: disable=protected-access
        held = sum(saved.nbytes for item in current for saved in item.saved if saved.device is not None)
        decisions.append((layer.index, held, sum(item.swapped_bytes for item in current)))

    monkeypatch.setattr(swap, "_enforce_budget", enforce_noted)
    return decisions


def test_a_budget_under_the_layers_swaps_balanced_layers_and_keeps_gradients(swap, monkeypatch):
    """Under one mean layer per layer, the pass swaps even when every layer is at the mean."""
    rows = [8, 8, 8]  # every layer exactly at the mean load
    expected, _ = _run(rows, sent=8, swap_on=False)
    _run(rows, sent=8, swap_on=True)  # the first pass teaches the swap the pass has three layers
    swap.begin_step()
    decisions = _spy_decisions(swap, monkeypatch)
    got, layers = _run(rows, sent=8, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    assert layers and min(layer.index for layer in layers) == 0, "the earliest layer goes first"
    mean = 8 * swap._pair_bytes  # pylint: disable=protected-access
    for index, held, _swapped in decisions:
        # What may be held after this layer, with the remaining layers at the mean.
        assert held <= 2.7 * mean - (len(rows) - index - 1) * mean


def test_a_budget_under_the_layers_acts_from_the_first_layer(swap, monkeypatch):
    """Once the pass length is known, the projection starts evicting after the first layer, not at the end."""
    decisions = _spy_decisions(swap, monkeypatch)
    _run([8, 8, 8], sent=8, swap_on=True)
    assert [swapped > 0 for _index, _held, swapped in decisions] == [False, False, True], \
        "the first pass budgets the layers it has seen"
    swap.begin_step()
    decisions.clear()
    _run([8, 8, 8], sent=8, swap_on=True)
    index, held, swapped = decisions[0]
    assert index == 0 and swapped > 0
    # 2.7 of 3 mean layers: after the first layer, room is kept for two more at the mean.
    assert held <= (2.7 - 2) * 8 * swap._pair_bytes  # pylint: disable=protected-access


def test_a_budget_over_the_layers_lets_an_early_peak_through(swap):
    """Over one mean layer per layer, a rank heavy early but light later moves nothing."""
    swap.configure(enabled=True, budget_layers=3.3)
    for step in (1, 2):  # the second pass knows it has three layers
        swap.begin_step()
        # 10 rows first is ahead of the mean, but within what the budget leaves after two mean layers.
        _grads, layers = _run([10, 6, 7], sent=8, swap_on=True)  # 23 rows against 3.3 x 8 = 26.4
    assert not layers, "ahead of the mean after the first layer, yet within the budget at the end"


def test_budget_above_the_spread_swaps_nothing(swap):
    """A budget over every rank's total leaves the pass on device."""
    swap.configure(enabled=True, budget_layers=4.5)
    swap.begin_step()
    _grads, layers = _run([12, 6, 9], sent=8, swap_on=True)  # total 27 rows against 4.5 x 8
    assert not layers


def test_a_layer_comes_back_only_from_the_layer_above(swap, monkeypatch):
    """Swapped early layers come back one layer ahead of their backward, not at its start."""
    swap.configure(enabled=True, budget_layers=2.0)
    swap.begin_step()
    trace = []
    unpack, load = swap._unpack, swap._load  # pylint: disable=protected-access
    monkeypatch.setattr(swap, "_unpack", lambda packed: (
        trace.append(("unpack", packed.layer.index)) if hasattr(packed, "layer") else None, unpack(packed))[1])
    monkeypatch.setattr(swap, "_load", lambda layer: (trace.append(("load", layer.index)), load(layer))[0])
    _grads, layers = _run([8, 8, 8, 8], sent=8, swap_on=True)
    for layer in layers:
        first_unpack_above = trace.index(("unpack", layer.index + 1))
        assert trace.index(("load", layer.index)) > first_unpack_above, f"layer {layer.index} loaded early"


def test_every_layer_brought_back_is_waited_for(swap, monkeypatch):
    """A prefetched layer is waited for too: its device memory exists before the copy back fills it."""
    swap.configure(enabled=True, budget_layers=2.0)
    swap.begin_step()
    returned = []
    unpack = swap._unpack  # pylint: disable=protected-access

    def unpack_checked(packed: Any) -> torch.Tensor:
        """Unpack, noting whether a swapped tensor's layer was waited for first."""
        tensor = unpack(packed)
        if hasattr(packed, "layer") and packed.on_host:
            returned.append(packed.layer.waited)
        return tensor

    monkeypatch.setattr(swap, "_unpack", unpack_checked)
    _grads, layers = _run([8, 8, 8, 8], sent=8, swap_on=True)
    assert any(layer.prefetched for layer in layers), "the scenario needs a prefetched layer"
    assert returned and all(returned), "a swapped tensor went back to autograd before its layer was waited for"


def test_budget_layers_must_be_positive_when_enabled():
    """A budget of zero or less could hold nothing."""
    with pytest.raises(ValueError, match="positive"):
        EPHostSwap().configure(enabled=True, budget_layers=0.0)


def _sample_model(width: int) -> None:  # pylint: disable=unused-argument
    """Model target for the config tests."""


def _sample_optimizer(learning_rate: float) -> None:  # pylint: disable=unused-argument
    """Optimizer target for the config tests."""


@pytest.mark.parametrize("override, message", [
    ("--activation_checkpoint.mode=full", "activation_checkpoint"),
    ("--activation_checkpoint.mode=selective", "activation_checkpoint"),
    ("--compile.enabled=true", "compile"),
    ("--accelerator.pp_size=2", "pipeline"),
])
def test_the_trainer_refuses_what_the_swap_cannot_budget(tmp_path, override, message):
    """The swap needs every MoE block's activations kept in one forward pass at a time."""
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(
        f"model:\n  _target_: {__name__}._sample_model\n  width: 8\n"
        f"optimizer:\n  _target_: {__name__}._sample_optimizer\n  learning_rate: 0.01\n"
        "ep_host_swap:\n  enabled: true\n  budget_layers: 4.0\n", encoding="utf-8")
    assert parse_training_args([str(yaml_path)]).ep_host_swap.enabled
    with pytest.raises(ValueError, match=message):
        parse_training_args([str(yaml_path), override])
