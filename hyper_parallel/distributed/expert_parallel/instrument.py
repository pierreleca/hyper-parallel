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
"""Per-rank measurement of expert-parallel load imbalance.

``ep_routed_forward`` calls this module around every phase of every MoE
block. For each block and each pass -- forward, activation recompute and
backward -- it records

- the routed token counts (per destination rank, per source rank and, on the
  forward pass, per expert), the raw load every downstream number derives
  from;
- a device event at each phase boundary, so the phase durations of the ranks
  can be compared and the wait at the combine all-to-all read off;
- the allocator's current and peak bytes at the same boundaries, which locate
  the memory peak in the step and show how far the ranks differ.

Each rank writes one JSON Lines file: a header, then one record per step.
``examples/qwen3_vl_30b_perf/analyze_ep_instrument.py`` merges the files.

Everything is off unless ``ep_instrument.enabled`` is set in the trainer
configuration, and the hooks then cost one attribute read per MoE call.
Peak-per-segment reading resets the allocator's peak counters, so the
trainer's own peak metric folds in :meth:`EPInstrument.fold_peaks`.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Optional

import torch
import torch.distributed as dist

from hyper_parallel.models.build_options import get_device_type, get_torch_device

FWD = "fwd"
RECOMPUTE = "recompute"
BWD = "bwd"

# Phase boundaries, in the order ep_routed_forward records them.
FWD_MARKS = ("start", "routed", "dispatched", "experts", "combined", "end")
BWD_MARKS = ("start", "aggregate", "combine", "experts", "dispatch", "end")


def _in_backward() -> bool:
    """Return whether the caller runs inside the autograd engine.

    Activation recompute is driven by the engine, so a MoE forward that runs
    with a live graph task is the recompute of a checkpointed block rather
    than the model's own forward pass.
    """
    current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)  # pylint: disable=protected-access
    if current_graph_task_id is None:
        return False
    return current_graph_task_id() != -1


class _DeviceProbe:
    """Device timing and allocator readings, with a host-clock fallback."""

    def __init__(self) -> None:
        """Bind the accelerator namespace, or None when running on CPU."""
        self.device = None if get_device_type() == "cpu" else get_torch_device()
        self.uses_events = self.device is not None and hasattr(self.device, "Event")

    def stamp(self) -> Any:
        """Return a timing stamp: a recorded device event, or a host time."""
        if self.uses_events:
            event = self.device.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def elapsed_ms(self, start: Any, stamp: Any) -> float:
        """Return the milliseconds between two stamps of the same kind."""
        if self.uses_events:
            return float(start.elapsed_time(stamp))
        return float((stamp - start) * 1e3)

    def synchronize(self) -> None:
        """Wait for the device queue, so every event carries a time."""
        if self.device is not None:
            self.device.synchronize()

    def allocated(self) -> Optional[int]:
        """Return the bytes the caching allocator holds for live tensors."""
        if self.device is None:
            return None
        return int(self.device.memory_allocated())

    def reserved(self) -> Optional[int]:
        """Return the bytes the caching allocator holds from the device."""
        if self.device is None:
            return None
        return int(self.device.memory_reserved())

    def peaks(self) -> tuple[Optional[int], Optional[int]]:
        """Return the allocated and reserved peaks since the last reset."""
        if self.device is None:
            return None, None
        return (
            int(self.device.max_memory_allocated()),
            int(self.device.max_memory_reserved()),
        )

    def reset_peaks(self) -> None:
        """Restart the allocator's peak counters at the current occupancy."""
        if self.device is not None:
            self.device.reset_peak_memory_stats()


class _BackwardMark(torch.autograd.Function):  # pylint: disable=abstract-method
    """Identity in the forward pass; records a boundary in the backward one.

    The forward pass inserts one of these on each tensor that separates two
    backward phases, so the backward timeline is built from the same kind of
    boundary as the forward one, without touching the autograd engine.
    """

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
            ctx: Any,
            tensor: torch.Tensor,
            recorder: "EPInstrument",
            layer: int,
            occurrence: int,
            name: str,
    ) -> torch.Tensor:
        """Return the tensor unchanged and retain where to record."""
        ctx.boundary = (recorder, layer, occurrence, name)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
            ctx: Any,
            grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """Record the backward boundary and pass the gradient through."""
        recorder, layer, occurrence, name = ctx.boundary
        recorder.mark(layer, BWD, occurrence, name)
        return grad, None, None, None, None


class _CallProbe:
    """Handle for one MoE block call: marks boundaries and keeps its counts."""

    __slots__ = ("recorder", "layer", "pass_name", "occurrence", "record")

    def __init__(
            self,
            recorder: "EPInstrument",
            layer: int,
            pass_name: str,
            occurrence: int,
    ) -> None:
        """Open one call record on the recorder."""
        self.recorder = recorder
        self.layer = layer
        self.pass_name = pass_name
        self.occurrence = occurrence
        self.record: dict[str, Any] = {
            "layer": layer,
            "pass": pass_name,
            "occurrence": occurrence,
        }

    def mark(self, name: str) -> None:
        """Record a forward phase boundary."""
        self.recorder.mark(self.layer, self.pass_name, self.occurrence, name)

    def wrap(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Attach a backward boundary to ``tensor``, when one can run.

        Exactly one pass builds the graph the backward pass walks: the
        model's own forward pass under non-reentrant checkpointing (the
        recompute's graph is then discarded, so its boundaries never fire),
        or the recompute under reentrant checkpointing (whose forward runs
        without grad). Marking every pass that records a graph covers both.
        """
        if not torch.is_grad_enabled():
            return tensor
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return tensor
        return _BackwardMark.apply(  # pylint: disable=not-callable
            tensor, self.recorder, self.layer, self.occurrence, name
        )

    def routing(
            self,
            *,
            topk_indices: torch.Tensor,
            hidden_states: torch.Tensor,
            send_counts: list[int],
            receive_counts: list[int],
            global_expert_count: int,
    ) -> None:
        """Keep the token counts of this call.

        The per-expert histogram is a device tensor here; the recorder drains
        every histogram of the step in one transfer when the step ends.
        """
        self.record.update(
            tokens=int(hidden_states.shape[0] * hidden_states.shape[1])
            if hidden_states.dim() == 3 else int(hidden_states.shape[0]),
            top_k=int(topk_indices.shape[-1]),
            hidden=int(hidden_states.shape[-1]),
            send=list(send_counts),
            recv=list(receive_counts),
        )
        self.recorder.note_model(
            global_expert_count=global_expert_count,
            top_k=int(topk_indices.shape[-1]),
            hidden=int(hidden_states.shape[-1]),
            element_size=int(hidden_states.dtype.itemsize),
        )
        if self.pass_name == FWD and self.recorder.record_counts:
            self.recorder.hold_expert_counts(
                self.record,
                torch.bincount(
                    topk_indices.reshape(-1),
                    minlength=global_expert_count,
                ),
            )


class EPInstrument:
    """Collects the per-rank record and writes it as JSON Lines."""

    def __init__(self) -> None:
        """Create the recorder in its disabled state."""
        self.enabled = False
        self.record_counts = True
        self.segment_peaks = True
        self.align_steps = True
        self.output_dir = ""
        self._probe: Optional[_DeviceProbe] = None
        self._names: dict[int, str] = {}
        self._layers: dict[int, int] = {}
        self._model: dict[str, Any] = {}
        self._experts: dict[str, Any] = {}
        self._stream = None
        self._header_written = False
        self._active = False
        self._step = 0
        self._wall_start = 0.0
        self._start_stamp: Any = None
        self._marks: list[list[Any]] = []
        self._calls: list[dict[str, Any]] = []
        self._counts: list[tuple[dict[str, Any], torch.Tensor]] = []
        self._occurrences: dict[tuple[int, str], int] = {}
        self._run_peaks = [0, 0]
        self._step_peaks = [0, 0]

    # -- configuration -----------------------------------------------------

    def configure(
            self,
            *,
            enabled: bool,
            output_dir: str,
            segment_peaks: bool = True,
            align_steps: bool = True,
            record_counts: bool = True,
    ) -> None:
        """Enable recording and choose where the per-rank files are written."""
        self.enabled = enabled
        self.output_dir = output_dir
        self.segment_peaks = segment_peaks
        self.align_steps = align_steps
        self.record_counts = record_counts
        if enabled:
            self._probe = _DeviceProbe()

    def register_modules(self, model: Any) -> None:
        """Name the MoE blocks, so records carry module paths, not order.

        A block is one the EP binder has prepared: its experts hold a local
        expert count.
        """
        if not self.enabled or model is None:
            return
        for name, module in model.named_modules():
            experts = getattr(module, "experts", None)
            if experts is not None and hasattr(experts, "local_expert_count"):
                self._names[id(module)] = name
                self._note_experts(experts)

    def _note_experts(self, experts: Any) -> None:
        """Keep the expert shapes that turn token counts into bytes."""
        if "intermediate" in self._experts:
            return
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        if gate_up is None or down is None or gate_up.dim() != 3 or down.dim() != 3:
            return
        # Layouts differ ([E, H, 2I] in GroupedExperts, [E, 2I, H] in HF), but
        # the hidden size is the dimension gate_up and down share, and the
        # intermediate size is the other dimension of down.
        down_dims = (int(down.shape[1]), int(down.shape[2]))
        shared = {int(gate_up.shape[1]), int(gate_up.shape[2])} & set(down_dims)
        if len(shared) == 1:
            hidden = shared.pop()
            intermediate = down_dims[0] if down_dims[1] == hidden else down_dims[1]
        else:
            intermediate = min(down_dims)
        self._experts = {
            "local_experts": int(experts.local_expert_count),
            "intermediate": intermediate,
            "expert_element_size": int(gate_up.dtype.itemsize),
        }

    def note_model(
            self,
            *,
            global_expert_count: int,
            top_k: int,
            hidden: int,
            element_size: int,
    ) -> None:
        """Keep the model shapes the analysis needs, from the first call."""
        if self._model:
            return
        self._model = {
            "num_experts": global_expert_count,
            "top_k": top_k,
            "hidden": hidden,
            "element_size": element_size,
        }

    # -- step lifecycle ----------------------------------------------------

    def begin_step(self, step: int) -> None:
        """Start recording one optimizer step.

        With ``align_steps``, the ranks meet at a barrier first, so their
        timelines start together and can be read side by side.
        """
        if not self.enabled:
            return
        if self.align_steps and dist.is_available() and dist.is_initialized():
            dist.barrier()
        self._probe.synchronize()
        self._step = step
        self._marks = []
        self._calls = []
        self._counts = []
        self._occurrences = {}
        self._step_peaks = [0, 0]
        self._probe.reset_peaks()
        self._wall_start = time.perf_counter()
        self._start_stamp = self._probe.stamp()
        self._active = True
        self.mark(-1, "step", 0, "start")

    def end_step(self) -> Optional[dict[str, Any]]:
        """Close the step, resolve the stamps and write one record."""
        if not self._active:
            return None
        self.mark(-1, "step", 0, "end")
        self._active = False
        wall_s = time.perf_counter() - self._wall_start
        self._probe.synchronize()
        self._drain_expert_counts()
        record = {
            "kind": "step",
            "step": self._step,
            "wall_s": round(wall_s, 6),
            "memory": {
                "step_peak_allocated": self._step_peaks[0] or None,
                "step_peak_reserved": self._step_peaks[1] or None,
            },
            "marks": [self._resolve(mark) for mark in self._marks],
            "calls": self._calls,
        }
        self._write(record)
        return record

    @staticmethod
    def step_summary(record: dict[str, Any]) -> dict[str, float]:
        """Condense one step record into the few numbers worth logging live."""
        pairs = sum(sum(call.get("recv", ())) for call in record["calls"]
                    if call["pass"] == FWD)
        # A recompute stops as soon as the last saved tensor is back, so a
        # pass is measured from its first to its last mark, not start to end.
        spans: dict[tuple[int, str, int], list[float]] = {}
        for layer, pass_name, occurrence, _name, time_ms, *_memory in record["marks"]:
            if layer < 0:
                continue
            spans.setdefault((layer, pass_name, occurrence), []).append(time_ms)
        duration = {FWD: 0.0, RECOMPUTE: 0.0, BWD: 0.0}
        for (_layer, pass_name, _occurrence), times in spans.items():
            if pass_name in duration:
                duration[pass_name] += max(times) - min(times)
        peak = record.get("memory", {}).get("step_peak_allocated") or 0
        return {
            "pairs": pairs,
            "fwd_ms": duration[FWD],
            "recompute_ms": duration[RECOMPUTE],
            "bwd_ms": duration[BWD],
            "peak_gib": peak / (1024 ** 3),
        }

    def mark(self, layer: int, pass_name: str, occurrence: int, name: str) -> None:
        """Record one phase boundary: a time stamp and the allocator state.

        The allocator state is the allocated and reserved bytes now and, with
        ``segment_peaks``, their peaks since the previous boundary. Reserved
        minus allocated is memory the cache holds but no tensor uses: when it
        grows over the steps, the cache is fragmenting.
        """
        if not self._active:
            return
        stamp = self._probe.stamp()
        allocated = self._probe.allocated()
        reserved_now = self._probe.reserved()
        peak = reserved = None
        if self.segment_peaks:
            peak, reserved = self._probe.peaks()
            if peak is not None:
                self._run_peaks = [
                    max(self._run_peaks[0], peak),
                    max(self._run_peaks[1], reserved),
                ]
                self._step_peaks = [
                    max(self._step_peaks[0], peak),
                    max(self._step_peaks[1], reserved),
                ]
            self._probe.reset_peaks()
        self._marks.append(
            [layer, pass_name, occurrence, name, stamp, allocated, peak, reserved_now, reserved]
        )

    def open_call(self, module: Any, ep_group: Any) -> Optional[_CallProbe]:
        """Return a probe for one MoE block call, or None when inactive."""
        if not self._active:
            return None
        del ep_group
        layer = self._layer_of(module)
        pass_name = RECOMPUTE if _in_backward() else FWD
        key = (layer, pass_name)
        occurrence = self._occurrences.get(key, 0)
        self._occurrences[key] = occurrence + 1
        probe = _CallProbe(self, layer, pass_name, occurrence)
        self._calls.append(probe.record)
        return probe

    def hold_expert_counts(self, record: dict[str, Any], counts: torch.Tensor) -> None:
        """Keep a per-expert histogram until the step ends."""
        self._counts.append((record, counts))

    def fold_peaks(self, allocated: int, reserved: int) -> tuple[int, int]:
        """Return peaks that survive this module's peak-counter resets.

        Reading a peak per segment resets the allocator's counters, so the
        trainer's own metric would otherwise report the last segment only.
        """
        if not self.enabled or not self.segment_peaks:
            return allocated, reserved
        return (
            max(allocated, self._run_peaks[0]),
            max(reserved, self._run_peaks[1]),
        )

    def close(self) -> None:
        """Close the per-rank file."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # -- internals ---------------------------------------------------------

    def _layer_of(self, module: Any) -> int:
        """Return a stable index for one MoE block, in first-seen order."""
        key = id(module)
        layer = self._layers.get(key)
        if layer is None:
            layer = len(self._layers)
            self._layers[key] = layer
        return layer

    def _drain_expert_counts(self) -> None:
        """Move every per-expert histogram of the step to host in one copy."""
        if not self._counts:
            return
        stacked = torch.stack([counts for _, counts in self._counts]).cpu()
        for (record, _), row in zip(self._counts, stacked.tolist()):
            record["expert_counts"] = row
        self._counts = []

    def _resolve(self, mark: list[Any]) -> list[Any]:
        """Turn one recorded boundary into times and bytes.

        Fields: layer, pass, occurrence, name, ms since the step start,
        allocated bytes, allocated peak since the previous boundary, reserved
        bytes, reserved peak since the previous boundary.
        """
        layer, pass_name, occurrence, name, stamp, allocated, peak, reserved, reserved_peak = mark
        return [
            layer,
            pass_name,
            occurrence,
            name,
            round(self._probe.elapsed_ms(self._start_stamp, stamp), 4),
            allocated,
            peak,
            reserved,
            reserved_peak,
        ]

    def _header(self) -> dict[str, Any]:
        """Describe the rank and the model, once per file."""
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        layers = {
            str(index): self._names.get(key, f"moe.{index}")
            for key, index in self._layers.items()
        }
        return {
            "kind": "header",
            "rank": rank,
            "world_size": world,
            "host": socket.gethostname(),
            "device_type": get_device_type(),
            "time_source": "device" if self._probe.uses_events else "host",
            "segment_peaks": self.segment_peaks,
            "layers": layers,
            **self._model,
            **self._experts,
        }

    def _write(self, record: dict[str, Any]) -> None:
        """Append one record to this rank's file, opening it on first use."""
        if self._stream is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"rank{rank:03d}.jsonl")
            self._stream = open(path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
        if not self._header_written:
            self._stream.write(json.dumps(self._header()) + "\n")
            self._header_written = True
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()


EP_INSTRUMENT = EPInstrument()

__all__ = ["BWD", "EP_INSTRUMENT", "EPInstrument", "FWD", "FWD_MARKS", "BWD_MARKS", "RECOMPUTE"]
