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
"""Per-layer MoE activation budget with the excess swapped to host (CPU path)."""

import json
import pathlib
import socket
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
    """Enable the swap with a small row threshold; disable it after the test."""
    HOST_SWAP.configure(enabled=True, capacity_factor=1.0, min_row_bytes=16, output_dir=str(tmp_path))
    HOST_SWAP.begin_step(1)
    yield HOST_SWAP
    HOST_SWAP.close()
    HOST_SWAP.configure(enabled=False, capacity_factor=1.2, min_row_bytes=1024, output_dir="")
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


def test_swapped_layers_give_the_same_gradients(swap):
    """Layers over budget move tensors to host and back; the gradients do not change."""
    rows = [12, 6, 15]  # budget 8 rows: layers 0 and 2 are over it
    expected, _ = _run(rows, sent=8, swap_on=False)
    swap.begin_step(1)
    got, layers = _run(rows, sent=8, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    assert [layer.index for layer in layers] == [0, 2]
    for layer in layers:
        excess_bytes = (layer.rows - layer.capacity) * sum(item.nbytes for item in layer.saved) / layer.rows
        assert layer.swapped_bytes >= excess_bytes, "the layer ends within its budget"
        assert all(item.device is None for item in layer.swapped), "nothing kept on device after backward"
    assert layers[0].prefetched and not layers[1].prefetched, "layer 0 came back while layer 2 ran backward"


def test_record_counts_the_moved_bytes(swap, tmp_path):
    """The step record lists every swapped layer and the bytes both ways."""
    _run([12, 6], sent=8, swap_on=True)
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
    """Through ep_routed_forward, a budget below the load swaps every layer and changes nothing."""
    swap.configure(enabled=False, capacity_factor=1.0, min_row_bytes=16, output_dir="")
    expected = _ep_grads(world_one_group, swap_on=False)
    # One rank receives what it sends; a budget of half of that puts every layer over it.
    swap.configure(enabled=True, capacity_factor=0.5, min_row_bytes=16, output_dir="")
    swap.begin_step(1)
    got = _ep_grads(world_one_group, swap_on=True)
    for want, have in zip(expected, got):
        torch.testing.assert_close(have, want)
    record = swap.end_step(rank=0)
    assert record["moe_layers"] == 2 and record["swapped_layers"] == 2
    assert all(layer["swapped_bytes"] > 0 for layer in record["layers"])
