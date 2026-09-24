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
"""Cap the activation memory of each MoE block by swapping its excess to host.

Under expert parallelism a rank's MoE block keeps, for backward, a few
tensors with one row per routed pair it received: the grouped-GEMM input, the
SwiGLU input and the SwiGLU output (8.7 KB per pair for Qwen3-VL-30B-A3B).
The number of pairs follows the routing, so the busiest rank of a layer holds
more than the others and a bad step can run out of memory.

Each MoE layer gets a budget of ``capacity_factor`` times the pairs the rank
sends, which is the mean number of pairs a rank receives when every rank holds
the same number of tokens. When a layer receives more, the pairs beyond the
budget have to leave the device: at the end of the layer's local expert
computation this module picks the smallest set of the saved tensors that holds
at least that many bytes and copies them to pinned host memory on a side
stream; the device memory is released as soon as the copy is done. Whole
tensors move, never row slices, so neither direction needs an extra device
copy, and the layer then holds at most its budget.

In backward, when a MoE layer first needs its saved tensors, the copy back of
the nearest swapped layer below it starts on the side stream, so it runs while
this layer's backward computes. The topmost swapped layer is copied back on
demand; the compute stream waits for it, and that wait is measured.

Each rank writes one JSON Lines file with, per step and per swapped layer, the
bytes moved, the device time of both copies and the time the compute stream
waited, so the host-link bandwidth can be read off.

Activation recompute runs the block again inside the autograd engine; nothing
is swapped then, since the recomputed tensors are consumed at once.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
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
        fitting = [buffer for buffer in self.free if buffer.numel() >= nbytes]
        if fitting:
            buffer = min(fitting, key=lambda item: item.numel())
            self.free.remove(buffer)
            return buffer
        size = -(-nbytes // _PIN_GRANULE) * _PIN_GRANULE
        self.allocated_bytes += size
        return torch.empty(size, dtype=torch.uint8, pin_memory=self.pin)

    def give(self, buffer: torch.Tensor) -> None:
        """Return a buffer to the pool."""
        self.free.append(buffer)


class _Saved:
    """One tensor the expert block saved for backward, on device or on host."""

    __slots__ = ("device", "host", "buffer", "shape", "dtype", "nbytes", "layer", "uses")

    def __init__(self, tensor: torch.Tensor, layer: "_Layer") -> None:
        """Hold the device tensor until the layer decides to swap it."""
        self.device: Optional[torch.Tensor] = tensor
        self.host: Optional[torch.Tensor] = None
        self.buffer: Optional[torch.Tensor] = None
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.nbytes = tensor.numel() * tensor.element_size()
        self.layer = layer
        # Times autograd saved it; the device copy is dropped after the last unpack.
        self.uses = 0


@dataclass
class _Layer:
    """One MoE block call of the forward pass."""

    index: int
    rows: int
    capacity: int
    saved: list[_Saved] = field(default_factory=list)
    keys: dict = field(default_factory=dict)
    swapped: list[_Saved] = field(default_factory=list)
    swapped_bytes: int = 0
    events: dict = field(default_factory=dict)
    loading: bool = False
    waited: bool = False
    prefetch_done: bool = False
    prefetched: bool = False


class EPHostSwap:
    """Per-layer budget for the MoE activations, with the excess swapped to host."""

    def __init__(self) -> None:
        """Start disabled; ``configure`` turns it on."""
        self.enabled = False
        self.capacity_factor = 1.2
        self.min_row_bytes = 1024
        self.output_dir = ""
        self._device = None
        self._copy_stream = None
        self._pool: Optional[_PinnedPool] = None
        self._layers: list[_Layer] = []
        self._swapped: list[_Layer] = []
        self._current: Optional[_Layer] = None
        self._step = 0
        self._file = None

    # -- configuration and steps ---------------------------------------------

    def configure(self, *, enabled: bool, capacity_factor: float, min_row_bytes: int, output_dir: str) -> None:
        """Set the budget and where the per-rank records go."""
        if capacity_factor <= 0:
            raise ValueError("ep_host_swap.capacity_factor must be positive")
        self.enabled = enabled
        self.capacity_factor = capacity_factor
        self.min_row_bytes = min_row_bytes
        self.output_dir = output_dir
        if not enabled:
            return
        self._device = None if get_device_type() == "cpu" else get_torch_device()
        self._copy_stream = self._device.Stream() if self._device is not None else None
        self._pool = _PinnedPool(pin=self._device is not None)

    def begin_step(self, step: int) -> None:
        """Forget the previous step's layers."""
        self._step = step
        self._layers = []
        self._swapped = []
        self._current = None

    def end_step(self, rank: int) -> Optional[dict]:
        """Resolve the step's copy timings, append its record and return it."""
        if not self.enabled or not self._layers:
            return None
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
            self._device.current_stream().synchronize()
        layers = [self._layer_record(layer) for layer in self._swapped]
        d2h_bytes = sum(layer["swapped_bytes"] for layer in layers)
        h2d_bytes = sum(layer["swapped_bytes"] for layer in layers if layer["loaded"])
        d2h_ms = sum(layer["d2h_ms"] or 0.0 for layer in layers)
        h2d_ms = sum(layer["h2d_ms"] or 0.0 for layer in layers)
        record = {
            "step": self._step,
            "rank": rank,
            "moe_layers": len(self._layers),
            "swapped_layers": len(layers),
            "d2h_gib": d2h_bytes / GIB,
            "h2d_gib": h2d_bytes / GIB,
            "d2h_gbps": d2h_bytes / d2h_ms / 1e6 if d2h_ms else None,
            "h2d_gbps": h2d_bytes / h2d_ms / 1e6 if h2d_ms else None,
            "stall_ms": sum(layer["stall_ms"] or 0.0 for layer in layers),
            "pinned_gib": self._pool.allocated_bytes / GIB if self._pool else 0.0,
            "layers": layers,
        }
        self._write(record, rank)
        self._layers, self._swapped = [], []
        return record

    def close(self) -> None:
        """Close this rank's file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    # -- forward ---------------------------------------------------------------

    @contextmanager
    def layer(self, received_rows: int, sent_rows: int) -> Iterator[None]:
        """Track what one MoE block saves; swap its excess when the block returns.

        Args:
            received_rows: Routed pairs this rank received for its experts.
            sent_rows: Routed pairs this rank sent, its tokens times top-k.
        """
        if not self.enabled or not torch.is_grad_enabled() or _in_backward():
            yield
            return
        capacity = int(self.capacity_factor * sent_rows)
        current = _Layer(index=len(self._layers), rows=received_rows, capacity=capacity)
        self._layers.append(current)
        self._current = current
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self._current = None
        self._swap_excess(current)

    def _candidate(self, tensor: torch.Tensor, layer: _Layer) -> bool:
        """Whether a saved tensor scales with the received pairs and can move."""
        return (
            type(tensor) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
            and tensor.dim() >= 1
            and tensor.shape[0] == layer.rows
            and layer.rows > 0
            and tensor.is_contiguous()
            and not (tensor.requires_grad and tensor.is_leaf)
            and tensor.numel() // layer.rows * tensor.element_size() >= self.min_row_bytes
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

    def _swap_excess(self, layer: _Layer) -> None:
        """Copy the smallest set of saved tensors that covers the layer's excess to host."""
        layer.keys = {}
        excess_rows = layer.rows - layer.capacity
        if excess_rows <= 0 or not layer.saved:
            return
        row_bytes = sum(saved.nbytes for saved in layer.saved) / layer.rows
        chosen = choose_offload([saved.nbytes for saved in layer.saved], int(excess_rows * row_bytes))
        layer.swapped = [layer.saved[index] for index in chosen]
        layer.swapped_bytes = sum(saved.nbytes for saved in layer.swapped)
        if self._copy_stream is None:
            for saved in layer.swapped:
                self._to_host(saved)
        else:
            compute = self._device.current_stream()
            self._copy_stream.wait_stream(compute)
            with self._device.stream(self._copy_stream):
                layer.events["d2h_start"] = self._event()
                for saved in layer.swapped:
                    self._to_host(saved)
                    # The allocator keeps the block until the copy stream is done with it.
                    saved.device.record_stream(self._copy_stream)
                layer.events["d2h_end"] = self._event()
        for saved in layer.swapped:
            saved.device = None
        self._swapped.append(layer)

    def _to_host(self, saved: _Saved) -> None:
        """Copy one saved tensor into a pinned buffer."""
        saved.buffer = self._pool.take(saved.nbytes)
        saved.host = saved.buffer[:saved.nbytes].view(saved.dtype).view(saved.shape)
        saved.host.copy_(saved.device, non_blocking=self._copy_stream is not None)

    # -- backward --------------------------------------------------------------

    def _unpack(self, packed: Any) -> torch.Tensor:
        """Return a saved tensor, bringing its layer back first if it was swapped."""
        if not isinstance(packed, _Saved):
            return packed
        layer = packed.layer
        if packed.device is None:
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
        """Copy a swapped layer's tensors back into fresh device memory."""
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
            layer.events["h2d_start"] = self._event()
            for saved in layer.swapped:
                self._from_host(saved)
            layer.events["h2d_end"] = self._event()

    def _from_host(self, saved: _Saved) -> None:
        """Copy one tensor back and return its pinned buffer to the pool.

        Every copy runs on the one copy stream, so a later offload into the same
        buffer is ordered after this load.
        """
        saved.device.copy_(saved.host, non_blocking=self._copy_stream is not None)
        self._pool.give(saved.buffer)
        saved.buffer = None
        saved.host = None

    def _wait(self, layer: _Layer) -> None:
        """Make the compute stream wait for a layer's copy back, once, timing the wait."""
        if layer.waited:
            return
        layer.waited = True
        if self._copy_stream is None:
            return
        compute = self._device.current_stream()
        layer.events["stall_start"] = self._event(compute)
        compute.wait_event(layer.events["h2d_end"])
        layer.events["stall_end"] = self._event(compute)

    def _prefetch_below(self, layer: _Layer) -> None:
        """Start copying back the nearest swapped layer that backward reaches after this one.

        Backward runs the MoE layers from the last to the first, so while one
        layer's backward computes, the swapped layer with the next lower index
        comes back.
        """
        if layer.prefetch_done:
            return
        layer.prefetch_done = True
        below = [other for other in self._swapped if other.index < layer.index and not other.loading]
        if below:
            target = max(below, key=lambda other: other.index)
            target.prefetched = True
            self._load(target)

    # -- helpers ---------------------------------------------------------------

    def _device_name(self) -> str:
        """The device new tensors go to."""
        return "cpu" if self._device is None else f"{get_device_type()}:{self._device.current_device()}"

    def _event(self, stream: Any = None) -> Any:
        """Record a timing event on ``stream``, or on the current stream."""
        event = self._device.Event(enable_timing=True)
        event.record(stream) if stream is not None else event.record()  # pylint: disable=expression-not-assigned
        return event

    @staticmethod
    def _elapsed(events: dict, start: str, end: str) -> Optional[float]:
        """Milliseconds between two recorded events, or None when either is missing."""
        if start not in events or end not in events:
            return None
        return events[start].elapsed_time(events[end])

    def _layer_record(self, layer: _Layer) -> dict:
        """The JSON-friendly summary of one swapped layer."""
        return {
            "index": layer.index,
            "rows": layer.rows,
            "capacity": layer.capacity,
            "excess_rows": layer.rows - layer.capacity,
            "saved": [[list(saved.shape), str(saved.dtype), saved.nbytes] for saved in layer.saved],
            "swapped_bytes": layer.swapped_bytes,
            "d2h_ms": self._elapsed(layer.events, "d2h_start", "d2h_end"),
            "h2d_ms": self._elapsed(layer.events, "h2d_start", "h2d_end"),
            "stall_ms": self._elapsed(layer.events, "stall_start", "stall_end"),
            "loaded": layer.loading,
            "prefetched": layer.prefetched,
        }

    def _write(self, record: dict, rank: int) -> None:
        """Append one record to this rank's file."""
        if not self.output_dir:
            return
        if self._file is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"host_swap_rank{rank}.jsonl")
            self._file = open(path, "a", encoding="utf-8")  # pylint: disable=consider-using-with
            header = {"header": True, "host": socket.gethostname(), "rank": rank,
                      "capacity_factor": self.capacity_factor, "min_row_bytes": self.min_row_bytes}
            self._file.write(json.dumps(header) + "\n")
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()


HOST_SWAP = EPHostSwap()

__all__ = ["EPHostSwap", "HOST_SWAP", "choose_offload"]
