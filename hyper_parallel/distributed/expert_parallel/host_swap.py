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
budget have to leave the device. At the end of the layer's local expert
computation this module copies saved tensors to pinned host memory on a side
stream, and the device memory is released as soon as the copy is done; the
layer then holds at most its budget. Two granularities:

- ``tensors``: the smallest set of whole saved tensors that holds the excess.
  No device copy, but the host link carries up to a whole tensor more than
  needed (the smallest tensor, the SwiGLU output, is 1536 bytes per pair).
- ``rows``: about the excess only. Whole tensors when they fit, and for the
  rest the last rows of one tensor (the one whose kept rows are smallest).
  The rows that stay are copied into a compact device tensor, and back into
  place when the tensor is rebuilt: two device copies on the side stream,
  and a transient device buffer for the kept rows while they run.

In backward, when a MoE layer first needs its saved tensors, the copy back of
the nearest swapped layer below it starts on the side stream, so it runs while
this layer's backward computes. The topmost swapped layer is copied back on
demand; the compute stream waits for it, and that wait is measured.

Each rank writes one JSON Lines file with, per step and per swapped layer, the
bytes moved, the device time of both copies and how much of it the compute
stream did not wait for, so the host-link bandwidth and the overlap can be read
off. The copy to host is never waited for: the compute stream goes on, and only
the release of the device memory follows the copy. The copy back is exposed
for the time the compute stream waits on it, and hidden for the rest.

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


def plan_rows(tensors: list[tuple[int, int]], need: int) -> list[tuple[int, int]]:
    """Return how many rows of which tensors to move so that at least ``need`` bytes leave.

    At most one tensor is split: the others move whole. Among the plans, the
    one moving the fewest host bytes wins, then the one whose split tensor
    keeps the fewest bytes on device, which is the device copy the split
    costs. Splitting the tensor with the fewest bytes per row rounds the
    least, so it usually wins both.

    Args:
        tensors: (rows, bytes per row) of every candidate tensor.
        need: Bytes that must leave the device.

    Returns:
        (index, rows moved) pairs; rows moved equals the tensor's rows for a
        whole tensor. Empty when ``need`` is not positive.
    """
    if need <= 0 or not tensors:
        return []
    sizes = [rows * row_bytes for rows, row_bytes in tensors]
    if sum(sizes) <= need:
        return [(index, tensors[index][0]) for index in range(len(tensors))]
    best_key, best_plan = None, None
    indices = range(len(tensors))
    for count in range(len(tensors) + 1):
        for whole in itertools.combinations(indices, count):
            moved = sum(sizes[index] for index in whole)
            plan = [(index, tensors[index][0]) for index in whole]
            copied = 0
            if moved < need:
                remaining = need - moved
                options = []
                for split in indices:
                    rows, row_bytes = tensors[split]
                    if split in whole or rows * row_bytes < remaining:
                        continue
                    taken = -(-remaining // row_bytes)
                    options.append(((rows - taken) * row_bytes, split, taken))
                if not options:
                    continue
                copied, split, taken = min(options)
                plan.append((split, taken))
                moved += taken * tensors[split][1]
            key = (moved, copied)
            if best_key is None or key < best_key:
                best_key, best_plan = key, plan
    return sorted(best_plan)


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

    __slots__ = ("device", "host", "buffer", "shape", "dtype", "nbytes", "layer", "uses", "rows_out", "kept")

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
        # Rows swapped to host: all of them, or the last rows_out in ``rows`` granularity.
        self.rows_out = 0
        # The rows that stay on device while the others are on host.
        self.kept: Optional[torch.Tensor] = None

    @property
    def rows(self) -> int:
        """Number of rows, one per received pair."""
        return self.shape[0]

    @property
    def row_bytes(self) -> int:
        """Bytes per row."""
        return self.nbytes // self.shape[0]

    @property
    def whole(self) -> bool:
        """Whether every row goes to host."""
        return self.rows_out >= self.rows

    @property
    def host_bytes(self) -> int:
        """Bytes that go to host."""
        return self.rows_out * self.row_bytes


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
    device_copy_bytes: int = 0
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
        self.granularity = "tensors"
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

    def configure(self, *, enabled: bool, capacity_factor: float, min_row_bytes: int, output_dir: str,
                  granularity: str = "tensors") -> None:
        """Set the budget, what moves (``rows`` or ``tensors``) and where the per-rank records go."""
        if capacity_factor <= 0:
            raise ValueError("ep_host_swap.capacity_factor must be positive")
        if granularity not in ("rows", "tensors"):
            raise ValueError(f"ep_host_swap.granularity must be 'rows' or 'tensors', not {granularity!r}")
        self.enabled = enabled
        self.granularity = granularity
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
        stall_ms = sum(layer["stall_ms"] or 0.0 for layer in layers)
        load_ms = sum(layer["load_ms"] or 0.0 for layer in layers)
        record = {
            "step": self._step,
            "rank": rank,
            "granularity": self.granularity,
            "moe_layers": len(self._layers),
            "swapped_layers": len(layers),
            "d2h_gib": d2h_bytes / GIB,
            "h2d_gib": h2d_bytes / GIB,
            "d2h_gbps": d2h_bytes / d2h_ms / 1e6 if d2h_ms else None,
            "h2d_gbps": h2d_bytes / h2d_ms / 1e6 if h2d_ms else None,
            "d2h_ms": d2h_ms,
            "h2d_ms": h2d_ms,
            # rows granularity: the kept rows, copied out in forward and back in backward.
            "device_copy_gib": sum(layer["device_copy_bytes"] for layer in layers) / GIB,
            "d2d_out_ms": sum(layer["d2d_out_ms"] or 0.0 for layer in layers),
            "d2d_in_ms": sum(layer["d2d_in_ms"] or 0.0 for layer in layers),
            # The whole copy back (host copy and device copy); the part the compute
            # stream waited for, and the rest, which ran under compute.
            "load_ms": load_ms,
            "stall_ms": stall_ms,
            "h2d_hidden_ms": max(load_ms - stall_ms, 0.0),
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
        """Copy the layer's excess to host: whole tensors, or rows of them."""
        layer.keys = {}
        excess_rows = layer.rows - layer.capacity
        if excess_rows <= 0 or not layer.saved:
            return
        row_bytes = sum(saved.row_bytes for saved in layer.saved)
        need = excess_rows * row_bytes
        if self.granularity == "tensors":
            plan = [(index, layer.saved[index].rows)
                    for index in choose_offload([saved.nbytes for saved in layer.saved], need)]
        else:
            plan = plan_rows([(saved.rows, saved.row_bytes) for saved in layer.saved], need)
        for index, rows in plan:
            layer.saved[index].rows_out = rows
        layer.swapped = [layer.saved[index] for index, _rows in plan]
        layer.swapped_bytes = sum(saved.host_bytes for saved in layer.swapped)
        split = [saved for saved in layer.swapped if not saved.whole]
        # The kept rows' buffers are allocated on the compute stream, before the copies.
        for saved in split:
            saved.kept = torch.empty((saved.rows - saved.rows_out,) + saved.shape[1:], dtype=saved.dtype,
                                     device=saved.device.device)
        layer.device_copy_bytes = sum(saved.kept.numel() * saved.kept.element_size() for saved in split)
        if self._copy_stream is None:
            for saved in split:
                saved.kept.copy_(saved.device[:saved.rows - saved.rows_out])
            for saved in layer.swapped:
                self._to_host(saved)
        else:
            compute = self._device.current_stream()
            self._copy_stream.wait_stream(compute)
            with self._device.stream(self._copy_stream):
                layer.events["d2d_out_start"] = self._event()
                for saved in split:
                    saved.kept.copy_(saved.device[:saved.rows - saved.rows_out], non_blocking=True)
                    saved.kept.record_stream(self._copy_stream)
                layer.events["d2d_out_end"] = self._event()
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
        """Copy a saved tensor, or its last ``rows_out`` rows, into a pinned buffer."""
        saved.buffer = self._pool.take(saved.host_bytes)
        shape = (saved.rows_out,) + saved.shape[1:]
        saved.host = saved.buffer[:saved.host_bytes].view(saved.dtype).view(shape)
        source = saved.device if saved.whole else saved.device[saved.rows - saved.rows_out:]
        saved.host.copy_(source, non_blocking=self._copy_stream is not None)

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
        """Rebuild a swapped layer's tensors in fresh device memory: host rows, then kept rows."""
        layer.loading = True
        for saved in layer.swapped:
            saved.device = torch.empty(saved.shape, dtype=saved.dtype, device=self._device_name())
        split = [saved for saved in layer.swapped if not saved.whole]
        if self._copy_stream is None:
            for saved in layer.swapped:
                self._from_host(saved)
            for saved in split:
                self._from_kept(saved)
            return
        # The copy stream writes memory the compute stream allocated: order it after
        # the compute stream's earlier work on those blocks.
        self._copy_stream.wait_stream(self._device.current_stream())
        with self._device.stream(self._copy_stream):
            layer.events["h2d_start"] = self._event()
            for saved in layer.swapped:
                self._from_host(saved)
            layer.events["h2d_end"] = self._event()
            layer.events["d2d_in_start"] = self._event()
            for saved in split:
                self._from_kept(saved)
            layer.events["loaded"] = self._event()

    def _from_host(self, saved: _Saved) -> None:
        """Copy the host rows back into place and return the pinned buffer to the pool.

        Every copy runs on the one copy stream, so a later offload into the same
        buffer is ordered after this load.
        """
        target = saved.device if saved.whole else saved.device[saved.rows - saved.rows_out:]
        target.copy_(saved.host, non_blocking=self._copy_stream is not None)
        self._pool.give(saved.buffer)
        saved.buffer = None
        saved.host = None

    def _from_kept(self, saved: _Saved) -> None:
        """Copy the kept rows back into place and drop their buffer."""
        saved.device[:saved.rows - saved.rows_out].copy_(saved.kept, non_blocking=self._copy_stream is not None)
        saved.kept = None

    def _wait(self, layer: _Layer) -> None:
        """Make the compute stream wait for a layer's copy back, once, timing the wait."""
        if layer.waited:
            return
        layer.waited = True
        if self._copy_stream is None:
            return
        compute = self._device.current_stream()
        layer.events["stall_start"] = self._event(compute)
        compute.wait_event(layer.events["loaded"])
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

    def _hidden(self, events: dict) -> Optional[float]:
        """Copy-back time that ran under compute: the whole load's duration less the compute stream's wait."""
        load = self._elapsed(events, "h2d_start", "loaded")
        stall = self._elapsed(events, "stall_start", "stall_end")
        if load is None or stall is None:
            return None
        return max(load - stall, 0.0)

    def _layer_record(self, layer: _Layer) -> dict:
        """The JSON-friendly summary of one swapped layer."""
        return {
            "index": layer.index,
            "rows": layer.rows,
            "capacity": layer.capacity,
            "excess_rows": layer.rows - layer.capacity,
            "saved": [[list(saved.shape), str(saved.dtype), saved.nbytes] for saved in layer.saved],
            # (shape, rows sent to host) of every swapped tensor.
            "moved": [[list(saved.shape), saved.rows_out] for saved in layer.swapped],
            "need_bytes": (layer.rows - layer.capacity) * sum(saved.row_bytes for saved in layer.saved),
            "swapped_bytes": layer.swapped_bytes,
            "device_copy_bytes": layer.device_copy_bytes,
            "d2d_out_ms": self._elapsed(layer.events, "d2d_out_start", "d2d_out_end"),
            "d2h_ms": self._elapsed(layer.events, "d2h_start", "d2h_end"),
            "h2d_ms": self._elapsed(layer.events, "h2d_start", "h2d_end"),
            "d2d_in_ms": self._elapsed(layer.events, "d2d_in_start", "loaded"),
            "load_ms": self._elapsed(layer.events, "h2d_start", "loaded"),
            "stall_ms": self._elapsed(layer.events, "stall_start", "stall_end"),
            "h2d_hidden_ms": self._hidden(layer.events),
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

__all__ = ["EPHostSwap", "HOST_SWAP", "choose_offload", "plan_rows"]
