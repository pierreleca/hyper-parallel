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
"""Bound the MoE activation memory of a forward pass by swapping its earliest layers to host.

Under expert parallelism a rank's MoE block keeps, for backward, a few
tensors with one row per routed pair it received: the grouped-GEMM input, the
SwiGLU input and the SwiGLU output (8.7 KB per pair for Qwen3-VL-30B-A3B).
The number of pairs follows the routing, so what a rank holds at the end of
forward changes from step to step and from rank to rank, and a bad step can
run out of memory.

Each rank may keep at most ``budget_layers`` mean layers of these tensors,
where a mean layer is what one MoE layer saves when the rank receives exactly
the pairs it sends (its own tokens times top-k). The pairs a rank sends are
known before the routing runs, so no routing can move the budget: whatever the
routing does, what the MoE layers keep for backward ends the forward pass within
``budget_layers`` mean layers.

After each MoE layer the rank projects its end-of-forward total, what it holds
plus the remaining layers at one mean layer each, and while the projection is
over the budget it copies whole saved tensors of its earliest layers, whose
backward comes last, to pinned host memory on a side stream. Acting on the
projection rather than on the budget itself settles the evictions early in the
pass, away from the step's peak at the end of forward, and keeps room for the
remaining layers so that little is left to evict after the last one. Under one
mean layer per layer the rank has a deficit to move whatever the routing does;
over it, a rank moves bytes only when it is on course to end the pass over the
budget, so a rank that is heavy early but light later moves nothing.

The copy to host is never waited for; the allocator keeps each evicted block
until the copy stream is done with it. In backward, a swapped layer comes back
while the layer just above it runs its backward, so the copy back of the
earliest layers does not land at the peak either; the compute stream waits only
for what has not arrived.

The number of layers of a pass is learned from the previous passes; the first
pass budgets only the layers seen so far. Passes must not interleave, so no
pipeline schedule, and the MoE blocks must keep their activations, so no
activation checkpointing: a checkpointed block keeps none to swap, and this
module's hooks inside a checkpointed region would keep what checkpointing
drops. The trainer refuses both.

Only the activations move: floating-point tensors of at least two dimensions, one
row of features per received pair. The index tensors stay on device.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch

from hyper_parallel.models.build_options import get_device_type, get_torch_device

GIB = 1024 ** 3
# Beyond this many candidates, a greedy choice replaces the exhaustive one.
_EXHAUSTIVE_LIMIT = 12
# Pinned buffers are allocated in multiples of this size, so the pool can serve
# the slightly different sizes of later steps.
_PIN_GRANULE = 64 * 1024 * 1024


def _in_backward() -> bool:
    """Return whether the caller runs inside the autograd engine (activation recompute)."""
    current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)  # pylint: disable=protected-access
    return current_graph_task_id is not None and current_graph_task_id() != -1


def choose_offload(sizes: list[int], need: int) -> list[int]:
    """Return the indices of the smallest-total subset of ``sizes`` holding at least ``need``.

    Args:
        sizes: Size in bytes of every candidate tensor.
        need: Bytes that must leave the device.

    Returns:
        Indices into ``sizes``; empty when ``need`` is not positive, every index
        when even all of them fall short.
    """
    if need <= 0 or not sizes:
        return []
    if sum(sizes) <= need:
        return list(range(len(sizes)))
    if len(sizes) <= _EXHAUSTIVE_LIMIT:
        best: Optional[tuple[int, ...]] = None
        best_total = 0
        for count in range(1, len(sizes) + 1):
            for subset in itertools.combinations(range(len(sizes)), count):
                total = sum(sizes[index] for index in subset)
                if total >= need and (best is None or total < best_total):
                    best, best_total = subset, total
        return list(best or ())
    chosen, total = [], 0
    for index in sorted(range(len(sizes)), key=lambda item: -sizes[item]):
        if total >= need:
            break
        chosen.append(index)
        total += sizes[index]
    return sorted(chosen)


class _PinnedPool:
    """Reusable pinned host buffers, best fit by size."""

    def __init__(self, pin: bool) -> None:
        """Pin the buffers when there is a device to copy from."""
        self.pin = pin
        self.free: list[torch.Tensor] = []
        self.allocated_bytes = 0

    def take(self, nbytes: int) -> torch.Tensor:
        """Return a byte buffer of at least ``nbytes``."""
        fitting = [index for index, buffer in enumerate(self.free) if buffer.numel() >= nbytes]
        if fitting:
            # By position: list.remove would compare tensors with ==, element by element.
            return self.free.pop(min(fitting, key=lambda index: self.free[index].numel()))
        size = -(-nbytes // _PIN_GRANULE) * _PIN_GRANULE
        self.allocated_bytes += size
        return torch.empty(size, dtype=torch.uint8, pin_memory=self.pin)

    def give(self, buffer: torch.Tensor) -> None:
        """Return a buffer to the pool."""
        self.free.append(buffer)


class _Saved:
    """One tensor the expert block saved for backward, on device or on host."""

    __slots__ = ("device", "host", "buffer", "shape", "dtype", "nbytes", "layer", "uses", "on_host")

    def __init__(self, tensor: torch.Tensor, layer: "_Layer") -> None:
        """Hold the device tensor until the budget decides to swap it."""
        self.device: Optional[torch.Tensor] = tensor
        self.host: Optional[torch.Tensor] = None
        self.buffer: Optional[torch.Tensor] = None
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.nbytes = tensor.numel() * tensor.element_size()
        self.layer = layer
        # Times autograd saved it; the device copy is dropped after the last unpack.
        self.uses = 0
        self.on_host = False

    @property
    def row_bytes(self) -> int:
        """Bytes per row, one row per received pair."""
        return self.nbytes // self.shape[0]


@dataclass
class _Layer:
    """One MoE block call of the forward pass."""

    index: int
    rows: int
    sent: int = 0
    saved: list[_Saved] = field(default_factory=list)
    keys: dict = field(default_factory=dict)
    swapped: list[_Saved] = field(default_factory=list)
    swapped_bytes: int = 0
    # Recorded on the copy stream once the layer's copy back is enqueued.
    loaded: Any = None
    loading: bool = False
    waited: bool = False
    prefetch_done: bool = False
    prefetched: bool = False


class EPHostSwap:
    """A budget in mean layers for a forward pass's MoE activations, its earliest layers swapped to host."""

    def __init__(self) -> None:
        """Start disabled; ``configure`` turns it on."""
        self.enabled = False
        self.budget_layers = 0.0
        self._device = None
        self._copy_stream = None
        self._pool: Optional[_PinnedPool] = None
        self._layers: list[_Layer] = []
        self._swapped: list[_Layer] = []
        self._current: Optional[_Layer] = None
        # Where the current forward pass starts in ``_layers``, whether a backward ran
        # since (the next layer then starts a pass), the longest pass seen and the last
        # bytes per pair.
        self._pass_start = 0
        self._backward_seen = False
        self._expected_layers = 0
        self._pair_bytes = 0

    # -- configuration and steps ---------------------------------------------

    def configure(self, *, enabled: bool, budget_layers: float) -> None:
        """Turn the swap on or off and set the budget in mean layers."""
        if enabled and budget_layers <= 0:
            raise ValueError("ep_host_swap.budget_layers must be positive")
        self.enabled = enabled
        self._expected_layers = 0
        self.budget_layers = budget_layers
        if not enabled:
            return
        self._device = None if get_device_type() == "cpu" else get_torch_device()
        self._copy_stream = self._device.Stream() if self._device is not None else None
        self._pool = _PinnedPool(pin=self._device is not None)

    def begin_step(self) -> None:
        """Forget the previous step's layers."""
        self._layers = []
        self._swapped = []
        self._current = None
        self._pass_start = 0
        self._backward_seen = False

    def end_step(self) -> Optional[dict]:
        """Return what the step moved: its MoE layers, the layers swapped and the bytes sent to host."""
        if not self.enabled or not self._layers:
            return None
        summary = {
            "moe_layers": len(self._layers),
            "swapped_layers": len(self._swapped),
            "d2h_gib": sum(layer.swapped_bytes for layer in self._swapped) / GIB,
        }
        self._layers, self._swapped = [], []
        return summary

    # -- forward ---------------------------------------------------------------

    @contextmanager
    def layer(self, received_rows: int, sent_rows: int) -> Iterator[None]:
        """Track what one MoE block saves; enforce the budget when the block returns.

        Args:
            received_rows: Routed pairs this rank received for its experts.
            sent_rows: Routed pairs this rank sent, its tokens times top-k.
        """
        if not self.enabled or not torch.is_grad_enabled() or _in_backward():
            yield
            return
        if self._backward_seen:
            self._pass_start, self._backward_seen = len(self._layers), False
        current = _Layer(index=len(self._layers), rows=received_rows, sent=sent_rows)
        self._layers.append(current)
        self._current = current
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self._current = None
        self._enforce_budget(current)

    def _candidate(self, tensor: torch.Tensor, layer: _Layer) -> bool:
        """Whether a saved tensor scales with the received pairs and can move."""
        return (
            type(tensor) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
            and tensor.is_floating_point()
            and tensor.dim() >= 2
            and tensor.shape[0] == layer.rows
            and layer.rows > 0
            and tensor.is_contiguous()
            and not (tensor.requires_grad and tensor.is_leaf)
            and (self._device is None or tensor.device.type != "cpu")
        )

    def _pack(self, tensor: torch.Tensor) -> Any:
        """Wrap the tensors that scale with the received pairs; pass the rest through."""
        layer = self._current
        if layer is None or not self._candidate(tensor, layer):
            return tensor
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tensor.dtype)
        saved = layer.keys.get(key)
        if saved is None:
            saved = _Saved(tensor, layer)
            layer.keys[key] = saved
            layer.saved.append(saved)
        saved.uses += 1
        return saved

    def _enforce_budget(self, layer: _Layer) -> None:
        """Swap the earliest layers' tensors while the projected end-of-forward total is over the budget.

        A mean layer is the pass's mean sent pairs times the bytes a received
        pair saves; the pass's number of layers is the longest seen so far, so
        the first pass budgets only the layers it has reached.
        """
        layer.keys = {}
        if layer.saved:
            self._pair_bytes = sum(saved.row_bytes for saved in layer.saved)
        current = self._layers[self._pass_start:]
        self._expected_layers = max(self._expected_layers, len(current))
        done, layers = len(current), self._expected_layers
        mean = self._pair_bytes * sum(item.sent for item in current) / len(current)
        budget = self.budget_layers * mean
        # What may be held now if every remaining layer comes at the mean.
        threshold = budget - (layers - done) * mean
        held = sum(saved.nbytes for item in current for saved in item.saved if saved.device is not None)
        need = held - threshold
        if need <= 0:
            return
        evicted = 0
        for item in current:
            candidates = [index for index, saved in enumerate(item.saved)
                          if saved.device is not None and not saved.on_host]
            if not candidates:
                continue
            picked = choose_offload([item.saved[index].nbytes for index in candidates], int(need - evicted) + 1)
            chosen = [item.saved[candidates[choice]] for choice in picked]
            evicted += sum(saved.nbytes for saved in chosen)
            self._evict(item, chosen)
            if evicted >= need:
                break

    def _evict(self, layer: _Layer, chosen: list[_Saved]) -> None:
        """Copy whole saved tensors of one layer to host and release their device memory."""
        if not chosen:
            return
        for saved in chosen:
            saved.on_host = True
        layer.swapped.extend(chosen)
        layer.swapped_bytes += sum(saved.nbytes for saved in chosen)
        if self._copy_stream is None:
            for saved in chosen:
                self._to_host(saved)
        else:
            self._copy_stream.wait_stream(self._device.current_stream())
            with self._device.stream(self._copy_stream):
                for saved in chosen:
                    self._to_host(saved)
                    # The allocator keeps the block until the copy stream is done with it.
                    saved.device.record_stream(self._copy_stream)
        for saved in chosen:
            saved.device = None
        if layer not in self._swapped:
            self._swapped.append(layer)

    def _to_host(self, saved: _Saved) -> None:
        """Copy a saved tensor into a pinned buffer."""
        saved.buffer = self._pool.take(saved.nbytes)
        saved.host = saved.buffer[:saved.nbytes].view(saved.dtype).view(saved.shape)
        saved.host.copy_(saved.device, non_blocking=self._copy_stream is not None)

    # -- backward --------------------------------------------------------------

    def _unpack(self, packed: Any) -> torch.Tensor:
        """Return a saved tensor, bringing its layer back first if it was swapped."""
        if not isinstance(packed, _Saved):
            return packed
        self._backward_seen = True
        layer = packed.layer
        # Keyed on the swap, not on the device tensor: a prefetch gives the tensor its
        # device memory at once, before the copy back has filled it.
        if packed.on_host:
            if not layer.loading:
                self._load(layer)
            self._wait(layer)
        self._prefetch_below(layer)
        tensor = packed.device
        packed.uses -= 1
        if packed.uses <= 0:
            packed.device = None
        return tensor

    def _load(self, layer: _Layer) -> None:
        """Rebuild a swapped layer's tensors in fresh device memory."""
        layer.loading = True
        for saved in layer.swapped:
            saved.device = torch.empty(saved.shape, dtype=saved.dtype, device=self._device_name())
        if self._copy_stream is None:
            for saved in layer.swapped:
                self._from_host(saved)
            return
        # The copy stream writes memory the compute stream allocated: order it after
        # the compute stream's earlier work on those blocks.
        self._copy_stream.wait_stream(self._device.current_stream())
        with self._device.stream(self._copy_stream):
            for saved in layer.swapped:
                self._from_host(saved)
            layer.loaded = self._device.Event()
            layer.loaded.record()

    def _from_host(self, saved: _Saved) -> None:
        """Copy a tensor back from host and return the pinned buffer to the pool.

        Every copy runs on the one copy stream, so a later offload into the same
        buffer is ordered after this load.
        """
        saved.device.copy_(saved.host, non_blocking=self._copy_stream is not None)
        self._pool.give(saved.buffer)
        saved.buffer = None
        saved.host = None

    def _wait(self, layer: _Layer) -> None:
        """Make the compute stream wait for a layer's copy back, once."""
        if layer.waited:
            return
        layer.waited = True
        if self._copy_stream is not None:
            self._device.current_stream().wait_event(layer.loaded)

    def _prefetch_below(self, layer: _Layer) -> None:
        """Start copying back the layer just below this one, if it was swapped.

        Backward runs the MoE layers from the last to the first, so the layer
        below comes back while this one's backward computes. Only the layer
        just below does: bringing the earliest layers back one swapped layer
        ahead would land their copies at the peak.
        """
        if layer.prefetch_done:
            return
        layer.prefetch_done = True
        for other in self._swapped:
            if other.index == layer.index - 1 and not other.loading:
                other.prefetched = True
                self._load(other)

    # -- helpers ---------------------------------------------------------------

    def _device_name(self) -> str:
        """The device new tensors go to."""
        return "cpu" if self._device is None else f"{get_device_type()}:{self._device.current_device()}"


HOST_SWAP = EPHostSwap()

__all__ = ["EPHostSwap", "HOST_SWAP", "choose_offload"]
