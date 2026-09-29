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

After each MoE layer the rank compares what it holds with a threshold that
grows with the layers seen so far, and copies whole saved tensors of its
earliest layers, whose backward comes last, to pinned host memory on a side
stream until it is back under. With B the budget, L the layers of a pass, i
the layers done and m a mean layer, the threshold is the larger of

- ``B * i / L``: the budget spread evenly over the layers, and
- ``B - (L - i) * m``: what may be held now if every remaining layer comes at
  the mean.

Below one mean layer per layer (B < L m) the first term is the larger: the rank
has a deficit to move whatever the routing does, and moves it evenly across the
pass. Above, the second is: the rank moves bytes only when it is on course to
end the pass over the budget, so a rank that is heavy early and light later
moves nothing. Both end at B after the last layer.

The copy to host is never waited for; the allocator keeps each evicted block
until the copy stream is done with it. In backward, a swapped layer comes back
while the layer just above it runs its backward, so the copy back of the
earliest layers does not land at the peak either; the compute stream waits only
for what has not arrived, and that wait is measured.

The number of layers of a pass is learned from the previous passes; the first
pass budgets only the layers seen so far. Passes must not interleave (no
pipeline schedule). Activation recompute runs the block again inside the
autograd engine; nothing is swapped then, since the recomputed tensors are
consumed at once.

Each rank writes one JSON Lines file with, per step, the eviction decisions and,
per swapped layer, the bytes moved, the device time of both copies and how much
of the copy back the compute stream did not wait for.
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
    events: dict = field(default_factory=dict)
    # (start, end) events of every copy to host; a layer can be swapped in several rounds.
    d2h_rounds: list = field(default_factory=list)
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
        self.min_row_bytes = 1024
        self.output_dir = ""
        self._device = None
        self._copy_stream = None
        self._pool: Optional[_PinnedPool] = None
        self._layers: list[_Layer] = []
        self._swapped: list[_Layer] = []
        self._current: Optional[_Layer] = None
        # Where the current forward pass starts in ``_layers``, whether a backward ran
        # since (the next layer then starts a pass), the longest pass seen, the last
        # bytes per pair and the step's eviction decisions.
        self._pass_start = 0
        self._backward_seen = False
        self._expected_layers = 0
        self._pair_bytes = 0
        self._evictions: list[dict] = []
        self._step = 0
        self._file = None

    # -- configuration and steps ---------------------------------------------

    def configure(self, *, enabled: bool, budget_layers: float, min_row_bytes: int, output_dir: str) -> None:
        """Set the budget in mean layers, which saved tensors may move and where the per-rank records go."""
        if enabled and budget_layers <= 0:
            raise ValueError("ep_host_swap.budget_layers must be positive")
        self.enabled = enabled
        self._expected_layers = 0
        self.budget_layers = budget_layers
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
        self._pass_start = 0
        self._backward_seen = False
        self._evictions = []

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
        record = {
            "step": self._step,
            "rank": rank,
            "moe_layers": len(self._layers),
            "swapped_layers": len(layers),
            "d2h_gib": d2h_bytes / GIB,
            "h2d_gib": h2d_bytes / GIB,
            "d2h_gbps": d2h_bytes / d2h_ms / 1e6 if d2h_ms else None,
            "h2d_gbps": h2d_bytes / h2d_ms / 1e6 if h2d_ms else None,
            "d2h_ms": d2h_ms,
            "h2d_ms": h2d_ms,
            # The copy back: the part the compute stream waited for, and the rest, which ran under compute.
            "stall_ms": stall_ms,
            "h2d_hidden_ms": max(h2d_ms - stall_ms, 0.0),
            "pinned_gib": self._pool.allocated_bytes / GIB if self._pool else 0.0,
            "layers": layers,
            # One entry per decision that swapped something.
            "evictions": self._evictions,
        }
        self._write(record, rank)
        self._layers, self._swapped, self._evictions = [], [], []
        return record

    def close(self) -> None:
        """Close this rank's file."""
        if self._file is not None:
            self._file.close()
            self._file = None

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

    def _enforce_budget(self, layer: _Layer) -> None:
        """Swap the earliest layers' tensors while what the pass holds is over its threshold.

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
        threshold = max(budget * done / layers, budget - (layers - done) * mean)
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
        self._evictions.append({
            "after_layer": layer.index,
            "expected_layers": self._expected_layers,
            "budget_gib": budget / GIB,
            "threshold_gib": threshold / GIB,
            "held_gib": held / GIB,
            "need_gib": need / GIB,
            "evicted_gib": evicted / GIB,
        })

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
                start = self._event()
                for saved in chosen:
                    self._to_host(saved)
                    # The allocator keeps the block until the copy stream is done with it.
                    saved.device.record_stream(self._copy_stream)
                layer.d2h_rounds.append((start, self._event()))
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
            layer.events["h2d_start"] = self._event()
            for saved in layer.swapped:
                self._from_host(saved)
            layer.events["loaded"] = self._event()

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

    @staticmethod
    def _d2h_ms(layer: _Layer) -> Optional[float]:
        """Milliseconds of every copy of the layer to host, or None when none was timed."""
        if not layer.d2h_rounds:
            return None
        return sum(start.elapsed_time(end) for start, end in layer.d2h_rounds)

    def _layer_record(self, layer: _Layer) -> dict:
        """The JSON-friendly summary of one swapped layer."""
        h2d_ms = self._elapsed(layer.events, "h2d_start", "loaded")
        stall_ms = self._elapsed(layer.events, "stall_start", "stall_end")
        return {
            "index": layer.index,
            "rows": layer.rows,
            "saved": [[list(saved.shape), str(saved.dtype), saved.nbytes] for saved in layer.saved],
            "moved": [list(saved.shape) for saved in layer.swapped],
            "swapped_bytes": layer.swapped_bytes,
            "d2h_ms": self._d2h_ms(layer),
            "h2d_ms": h2d_ms,
            "stall_ms": stall_ms,
            "h2d_hidden_ms": None if h2d_ms is None or stall_ms is None else max(h2d_ms - stall_ms, 0.0),
            "loaded": layer.loading,
            "prefetched": layer.prefetched,
        }

    def _write(self, record: dict, rank: int) -> None:
        """Append one record to this rank's file, which a run starts afresh, as the EP instrument does."""
        if not self.output_dir:
            return
        if self._file is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"host_swap_rank{rank}.jsonl")
            self._file = open(path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
            header = {"header": True, "host": socket.gethostname(), "rank": rank,
                      "budget_layers": self.budget_layers, "min_row_bytes": self.min_row_bytes}
            self._file.write(json.dumps(header) + "\n")
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()


HOST_SWAP = EPHostSwap()

__all__ = ["EPHostSwap", "HOST_SWAP", "choose_offload"]
