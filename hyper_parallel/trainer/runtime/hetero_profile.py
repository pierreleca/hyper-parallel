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
"""Per-component device timing and per-micro-batch workload, to explain imbalance.

A multimodal MoE model is heterogeneous in two ways that a step time alone
hides. The model is: a dense vision tower, a text decoder whose attention and
routed experts scale differently with the sequence, a vocabulary projection.
The data is: samples differ in length, in images and in image size, so under
data parallelism the ranks of one step do different amounts of every part.

This recorder keeps, per rank and optimizer step,

- the workload of every micro-batch: tokens, label tokens, images, vision
  patches, the vision attention cost and the image grids;
- a device event at the boundaries of each hooked module, in the forward pass,
  in the recompute of a checkpointed block and in the backward pass, so the
  time of every component can be set against the workload that caused it;
- with ``routing_by_modality``, the expert histogram of the image tokens and
  of the text tokens of every MoE layer, to see whether the two route alike.

Where the boundaries sit matters. A module's entry stamp is taken *after* the
sharding hooks have unsharded its weights, and its exit stamp *before* they
reshard, so a span holds the module's own work (and any collective inside it,
such as the expert all-to-alls) but not the wait for its weights. That wait is
the gap between one module's exit and the next one's entry. The backward
boundaries are autograd identity functions on the module's output and input,
placed on the same side of the sharding hooks.

Each rank writes one JSON Lines file, a header and then one record per step.
``examples/qwen3_vl_30b_perf/analyze_hetero.py`` merges the files.

Everything is off unless ``hetero_profile.enabled`` is set. With the recorder
inactive the hooks cost one attribute read per module call.
"""

from __future__ import annotations

import functools
import json
import os
import re
import socket
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from hyper_parallel.models.build_options import get_device_type, get_torch_device
from hyper_parallel.trainer.runtime.logging import create_logger

logger = create_logger(__name__)

FWD = "fwd"
RECOMPUTE = "recompute"
BWD = "bwd"
# Boundary kinds. In the forward and recompute passes: entering and leaving the
# module. In the backward pass: the gradient of the module's output arriving,
# and the gradient of its input being produced.
ENTER = "in"
EXIT = "out"

IGNORE_INDEX = -100
# Path segments that wrappers add; the role of a module does not depend on them.
_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod", "_wrapped_module"}
)
# Roles that need the finer module levels, switched by ``vision_blocks`` and ``sublayers``.
VISION_DETAIL_ROLES = frozenset({"vision.patch_embed", "vision.block", "vision.deepstack", "vision.merger"})
TEXT_DETAIL_ROLES = frozenset({"text.attn", "text.moe"})
# The router is only read for its routing decisions, never timed.
ROUTER_ROLE = "text.router"
ROOT_ROLE = "root"

# (role, regex over the module path with wrapper segments removed). The first
# group, when there is one, is the index of a repeated block. Module paths of the
# Hugging Face Qwen3-VL-MoE model; the roles are what the analysis reads.
DEFAULT_ROLES: tuple[tuple[str, str], ...] = (
    ("vision.root", r"(?:^|\.)visual$"),
    ("vision.patch_embed", r"(?:^|\.)visual\.patch_embed$"),
    ("vision.block", r"(?:^|\.)visual\.blocks\.(\d+)$"),
    ("vision.deepstack", r"(?:^|\.)visual\.deepstack_merger_list\.(\d+)$"),
    ("vision.merger", r"(?:^|\.)visual\.merger$"),
    ("text.root", r"(?:^|\.)language_model$"),
    ("text.embed", r"(?:^|\.)language_model\.embed_tokens$"),
    ("text.layer", r"(?:^|\.)language_model\.layers\.(\d+)$"),
    ("text.attn", r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn$"),
    ("text.moe", r"(?:^|\.)language_model\.layers\.(\d+)\.mlp$"),
    (ROUTER_ROLE, r"(?:^|\.)language_model\.layers\.(\d+)\.mlp\.gate$"),
    ("text.norm", r"(?:^|\.)language_model\.norm$"),
    ("lm_head", r"(?:^|\.)lm_head$"),
)


def normalize_module_path(name: str) -> str:
    """Return a module path without the segments that wrappers insert."""
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def _in_backward() -> bool:
    """Return whether the caller runs inside the autograd engine.

    Activation recompute is driven by the engine, so a module that runs with a
    live graph task is being recomputed rather than running the model's own
    forward pass.
    """
    current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)  # pylint: disable=protected-access
    if current_graph_task_id is None:
        return False
    return current_graph_task_id() != -1


def _scalar(value: Any) -> int:
    """Return a device or host scalar tensor as an int."""
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def batch_workload(batch: Mapping[str, Any], spatial_merge_size: int = 2, max_grids: int = 64) -> dict[str, Any]:
    """Describe the work of one micro-batch from its model inputs.

    Args:
        batch: Model inputs as the trainer passes them to the model; any field
            may be absent. Tensors may be on the device (reading them waits for
            it, which is cheap between steps).
        spatial_merge_size: Side of the patch merge of the vision tower; four
            patches become one visual token with 2.
        max_grids: At most this many image grids are listed.

    Returns:
        ``tokens`` (the padded sequence positions), ``real_tokens`` (attention
        mask), ``label_tokens`` (positions the loss reads), ``images`` and
        ``videos``, ``patches`` (vision patches, the vision tower's rows),
        ``visual_tokens`` (patches after the merge), ``image_tokens`` (positions
        marked as media in the sequence), ``vision_attn_pairs`` (the vision
        attention cost, the sum over frames of the squared patches of a frame)
        and ``grids`` (the first ``max_grids`` image grids as [t, h, w]).
    """
    result: dict[str, Any] = {}
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        result["batch_size"] = int(input_ids.shape[0]) if input_ids.dim() > 1 else 1
        result["tokens"] = int(input_ids.numel())
    attention_mask = batch.get("attention_mask")
    result["real_tokens"] = (
        _scalar(attention_mask.sum()) if isinstance(attention_mask, torch.Tensor) else result.get("tokens", 0)
    )
    labels = batch.get("labels")
    if isinstance(labels, torch.Tensor):
        result["label_tokens"] = _scalar((labels != IGNORE_INDEX).sum())
    media = batch.get("mm_token_type_ids")
    if isinstance(media, torch.Tensor):
        result["image_tokens"] = _scalar((media > 0).sum())

    patches = attn_pairs = images = videos = 0
    grids: list[list[int]] = []
    for field, kind in (("image_grid_thw", "images"), ("video_grid_thw", "videos")):
        grid = batch.get(field)
        if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
            continue
        rows = grid.detach().cpu().reshape(-1, 3).tolist()
        for frames, height, width in rows:
            patches += frames * height * width
            attn_pairs += frames * (height * width) ** 2
            if len(grids) < max_grids and kind == "images":
                grids.append([int(frames), int(height), int(width)])
        if kind == "images":
            images += len(rows)
        else:
            videos += len(rows)
    result.update(
        images=images,
        videos=videos,
        patches=int(patches),
        visual_tokens=int(patches) // (spatial_merge_size ** 2),
        vision_attn_pairs=int(attn_pairs),
        grids=grids,
    )
    pixel_values = batch.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor):
        result["pixel_rows"] = int(pixel_values.shape[0])
    return result


def _guarded(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a recorder method; if it raises, switch the recorder off instead of failing the training step."""

    @functools.wraps(method)
    def wrapper(self: "HeteroProfiler", *args: Any, **kwargs: Any) -> Any:
        """Call the method unless the recorder already failed."""
        if self._failed:  # pylint: disable=protected-access
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._fail(method.__name__, exc)  # pylint: disable=protected-access
            return None

    return wrapper


class _DeviceClock:
    """Device events and allocator readings, with a host-clock fallback."""

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
        return None if self.device is None else int(self.device.memory_allocated())

    def peaks(self) -> tuple[Optional[int], Optional[int]]:
        """Return the allocated and reserved peaks since the last reset."""
        if self.device is None:
            return None, None
        return int(self.device.max_memory_allocated()), int(self.device.max_memory_reserved())

    def reset_peaks(self) -> None:
        """Restart the allocator's peak counters at the current occupancy."""
        if self.device is not None:
            self.device.reset_peak_memory_stats()


class _Boundary(torch.autograd.Function):  # pylint: disable=abstract-method
    """Identity in the forward pass; records a boundary in the backward one."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
            ctx: Any,
            tensor: torch.Tensor,
            profiler: "HeteroProfiler",
            module_id: int,
            occurrence: int,
            kind: str,
    ) -> torch.Tensor:
        """Return the tensor unchanged and retain where to record."""
        ctx.boundary = (profiler, module_id, occurrence, kind)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
            ctx: Any,
            grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """Record the backward boundary and pass the gradient through."""
        profiler, module_id, occurrence, kind = ctx.boundary
        profiler.mark(module_id, BWD, occurrence, kind)
        return grad, None, None, None, None


class HeteroProfiler:
    """Hooks a model and writes the per-step component and workload records."""

    def __init__(self, roles: Sequence[tuple[str, str]] = DEFAULT_ROLES) -> None:
        """Create the recorder in its disabled state.

        Args:
            roles: ``(role, regex)`` pairs matched against normalized module
                paths; the first matching pair names a module's role.
        """
        self.enabled = False
        self._failed = False
        self.output_dir = ""
        self.vision_blocks = True
        self.sublayers = True
        self.routing_by_modality = True
        self.memory = True
        self.step_peaks = True
        self.spatial_merge_size = 2
        self._run_peaks = [0, 0]
        self._roles = [(role, re.compile(pattern)) for role, pattern in roles]
        self._clock: Optional[_DeviceClock] = None
        self._modules: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._root_id: Optional[int] = None
        self._stream = None
        self._header_written = False
        self._active = False
        self._step = 0
        self._wall_start = 0.0
        self._last_end: Optional[float] = None
        self._inter_step_ms: Optional[float] = None
        self._start_stamp: Any = None
        self._marks: list[tuple[Any, ...]] = []
        self._occurrences: dict[tuple[int, str], int] = {}
        self._open: dict[int, list[tuple[str, int]]] = {}
        self._micro_batches: list[dict[str, Any]] = []
        self._visual_mask: Optional[torch.Tensor] = None
        self._routing: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        self._extras: dict[str, Any] = {}

    # -- configuration -------------------------------------------------------

    def configure(
            self,
            *,
            enabled: bool,
            output_dir: str,
            vision_blocks: bool = True,
            sublayers: bool = True,
            routing_by_modality: bool = True,
            memory: bool = True,
            step_peaks: bool = True,
            spatial_merge_size: int = 2,
    ) -> None:
        """Enable recording and choose where the per-rank files are written."""
        self.enabled = enabled
        self.output_dir = output_dir
        self.vision_blocks = vision_blocks
        self.sublayers = sublayers
        self.routing_by_modality = routing_by_modality
        self.memory = memory
        self.step_peaks = step_peaks
        self.spatial_merge_size = spatial_merge_size
        if enabled:
            self._clock = _DeviceClock()

    @property
    def modules(self) -> list[dict[str, Any]]:
        """The hooked modules: id, normalized path, role and block index."""
        return self._modules

    def match_role(self, name: str) -> tuple[Optional[str], Optional[int]]:
        """Return the role and block index of a module path, or (None, None)."""
        path = normalize_module_path(name)
        for role, pattern in self._roles:
            found = pattern.search(path)
            if found:
                return role, int(found.group(1)) if found.groups() else None
        return None, None

    def _wanted(self, role: str) -> bool:
        """Return whether a role is recorded under the current detail switches."""
        if role in VISION_DETAIL_ROLES:
            return self.vision_blocks
        if role in TEXT_DETAIL_ROLES:
            return self.sublayers
        if role == ROUTER_ROLE:
            return self.routing_by_modality
        return True

    @_guarded
    def attach(self, model: torch.nn.Module) -> dict[str, int]:
        """Hook the model's modules; return how many of each role were found.

        Call it after the model is sharded and its activation checkpointing is
        applied, so the hooks sit outside the sharding hooks. A module behind
        a wrapper is hooked once, at the outermost module of that path.
        """
        if not self.enabled or model is None:
            return {}
        self.detach()
        seen: set[str] = set()
        counts: dict[str, int] = {}
        for name, module in model.named_modules():
            path = normalize_module_path(name)
            if path in seen:
                continue
            if name == "":
                role, index = ROOT_ROLE, None
            else:
                role, index = self.match_role(name)
            if role is None or not self._wanted(role):
                continue
            seen.add(path)
            module_id = len(self._modules)
            self._modules.append({"id": module_id, "name": path, "role": role, "index": index})
            counts[role] = counts.get(role, 0) + 1
            if role == ROOT_ROLE:
                self._root_id = module_id
                self._handles.append(
                    module.register_forward_pre_hook(self._root_pre_hook(module_id), with_kwargs=True)
                )
                self._handles.append(
                    module.register_forward_hook(self._root_post_hook(module_id), with_kwargs=True)
                )
            elif role == ROUTER_ROLE:
                self._handles.append(module.register_forward_hook(self._router_hook(module_id, index)))
            else:
                # Appended after the sharding hooks, so entry follows the weights' unshard; prepended on the
                # way out, so the module's output is marked before the sharding hooks see it.
                self._handles.append(
                    module.register_forward_pre_hook(self._pre_hook(module_id), with_kwargs=True)
                )
                self._handles.append(
                    module.register_forward_hook(self._post_hook(module_id), with_kwargs=True, prepend=True)
                )
        return counts

    def detach(self) -> None:
        """Remove every hook, and forget the modules."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._modules = []
        self._root_id = None

    # -- step lifecycle ------------------------------------------------------

    @_guarded
    def begin_step(self, step: int, micro_batches: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        """Start recording one optimizer step.

        The workload of every micro-batch is read first (it may wait for the
        device, which is idle between steps), then the step's start stamp is
        taken.
        """
        if not self.enabled or not self._modules:
            return
        workloads = [batch_workload(batch, self.spatial_merge_size) for batch in micro_batches or []]
        self._clock.synchronize()
        self._step = step
        self._marks = []
        self._occurrences = {}
        self._open = {}
        self._micro_batches = workloads
        self._routing = []
        self._visual_mask = None
        self._extras = {}
        if self.step_peaks:
            self._clock.reset_peaks()
        started = time.perf_counter()
        self._inter_step_ms = None if self._last_end is None else (started - self._last_end) * 1e3
        self._wall_start = started
        self._start_stamp = self._clock.stamp()
        self._active = True

    @_guarded
    def end_step(self, **extras: Any) -> Optional[dict[str, Any]]:
        """Close the step, resolve the stamps and write one record."""
        if not self._active:
            return None
        end_stamp = self._clock.stamp()
        self._active = False
        wall_ms = (time.perf_counter() - self._wall_start) * 1e3
        self._clock.synchronize()
        self._last_end = time.perf_counter()
        peak_allocated, peak_reserved = self._clock.peaks() if self.step_peaks else (None, None)
        if peak_allocated is not None:
            self._run_peaks = [max(self._run_peaks[0], peak_allocated), max(self._run_peaks[1], peak_reserved)]
        record: dict[str, Any] = {
            "kind": "step",
            "step": self._step,
            "wall_ms": round(wall_ms, 3),
            "device_ms": round(self._clock.elapsed_ms(self._start_stamp, end_stamp), 3),
            "inter_step_ms": None if self._inter_step_ms is None else round(self._inter_step_ms, 3),
            "micro_batches": self._micro_batches,
            "marks": [self._resolve(mark) for mark in self._marks],
            "routing": self._drain_routing(),
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
        }
        record.update(extras)
        self._write(record)
        return record

    def fold_peaks(self, allocated: int, reserved: int) -> tuple[int, int]:
        """Return peaks that survive this recorder's resets of the allocator's counters.

        Reading a peak per step resets the allocator's counters, so the trainer's own
        peak metric would otherwise report the last step only.
        """
        if not self.enabled or not self.step_peaks:
            return allocated, reserved
        return max(allocated, self._run_peaks[0]), max(reserved, self._run_peaks[1])

    def close(self) -> None:
        """Close the per-rank file and remove the hooks."""
        self.detach()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # -- recording -----------------------------------------------------------

    @_guarded
    def mark(self, module_id: int, pass_name: str, occurrence: int, kind: str) -> None:
        """Record one boundary: a time stamp and the allocator's allocated bytes."""
        if not self._active:
            return
        stamp = self._clock.stamp()
        allocated = self._clock.allocated() if self.memory else None
        self._marks.append((module_id, pass_name, occurrence, kind, stamp, allocated))

    def _next_occurrence(self, module_id: int, pass_name: str) -> int:
        """Count the calls of one module in one pass during this step."""
        key = (module_id, pass_name)
        occurrence = self._occurrences.get(key, 0)
        self._occurrences[key] = occurrence + 1
        return occurrence

    def _protect(self, hook: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a module hook so that a failure of the recorder switches it off and leaves the model alone."""

        @functools.wraps(hook)
        def protected(*args: Any, **kwargs: Any) -> Any:
            """Call the hook unless the recorder already failed."""
            if self._failed:
                return None
            try:
                return hook(*args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._fail("module hook", exc)
                return None

        return protected

    def _fail(self, where: str, exc: BaseException) -> None:
        """Switch the recorder off after an error in it, and say so once."""
        self._failed = True
        self._active = False
        logger.warning("hetero profile switched off after an error in %s: %r; training goes on without it", where, exc)

    @staticmethod
    def _wrap_tensor(tensor: Any, profiler: "HeteroProfiler", module_id: int, occurrence: int, kind: str) -> Any:
        """Attach a backward boundary to a tensor that takes part in the graph."""
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            return _Boundary.apply(tensor, profiler, module_id, occurrence, kind)  # pylint: disable=not-callable
        return tensor

    def _pre_hook(self, module_id: int) -> Callable[..., Any]:
        """Return the forward pre-hook of one module."""

        def hook(_module: Any, args: tuple, kwargs: dict) -> Optional[tuple[tuple, dict]]:
            """Stamp the entry, and mark the input's gradient for the backward pass."""
            if not self._active:
                return None
            pass_name = RECOMPUTE if _in_backward() else FWD
            occurrence = self._next_occurrence(module_id, pass_name)
            self._open.setdefault(module_id, []).append((pass_name, occurrence))
            self.mark(module_id, pass_name, occurrence, ENTER)
            if not torch.is_grad_enabled():
                return None
            # The backward boundary of the module's input: its gradient is complete
            # once the module's backward has produced it.
            if args and isinstance(args[0], torch.Tensor):
                wrapped = self._wrap_tensor(args[0], self, module_id, occurrence, EXIT)
                return (wrapped,) + tuple(args[1:]), kwargs
            if isinstance(kwargs.get("hidden_states"), torch.Tensor):
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = self._wrap_tensor(kwargs["hidden_states"], self, module_id, occurrence, EXIT)
                return args, kwargs
            return None

        return self._protect(hook)

    def _post_hook(self, module_id: int) -> Callable[..., Any]:
        """Return the forward hook of one module."""

        def hook(_module: Any, _args: tuple, _kwargs: dict, output: Any) -> Any:
            """Stamp the exit, and mark the output's gradient for the backward pass."""
            if not self._active:
                return None
            opened = self._open.get(module_id)
            if not opened:
                return None
            pass_name, occurrence = opened.pop()
            self.mark(module_id, pass_name, occurrence, EXIT)
            if not torch.is_grad_enabled():
                return None
            # The backward boundary of the module's output: its gradient has just arrived.
            if isinstance(output, torch.Tensor):
                return self._wrap_tensor(output, self, module_id, occurrence, ENTER)
            if isinstance(output, tuple) and not hasattr(output, "_fields"):
                for position, item in enumerate(output):
                    if isinstance(item, torch.Tensor) and item.requires_grad:
                        wrapped = self._wrap_tensor(item, self, module_id, occurrence, ENTER)
                        return output[:position] + (wrapped,) + output[position + 1:]
            return None

        return self._protect(hook)

    def _root_pre_hook(self, module_id: int) -> Callable[..., Any]:
        """Return the pre-hook of the model: one call per micro-batch."""

        def hook(_module: Any, _args: tuple, kwargs: dict) -> None:
            """Stamp the start of a micro-batch's forward pass and keep its image-token mask."""
            if not self._active:
                return
            pass_name = RECOMPUTE if _in_backward() else FWD
            occurrence = self._next_occurrence(module_id, pass_name)
            self.mark(module_id, pass_name, occurrence, ENTER)
            media = kwargs.get("mm_token_type_ids")
            self._visual_mask = (media > 0).reshape(-1) if isinstance(media, torch.Tensor) else None

        return self._protect(hook)

    def _root_post_hook(self, module_id: int) -> Callable[..., Any]:
        """Return the hook that closes a micro-batch's forward pass."""

        def hook(_module: Any, _args: tuple, _kwargs: dict, _output: Any) -> None:
            """Stamp the end of a micro-batch's forward pass."""
            if not self._active:
                return
            occurrence = self._occurrences.get((module_id, FWD), 1) - 1
            self.mark(module_id, FWD, occurrence, EXIT)

        return self._protect(hook)

    def _router_hook(self, module_id: int, layer: Optional[int]) -> Callable[..., Any]:
        """Return the hook that splits a router's expert choices by token modality."""
        del module_id

        def hook(_module: Any, _args: tuple, output: Any) -> None:
            """Count the experts chosen for the image tokens and for the text tokens."""
            mask = self._visual_mask
            if not self._active or mask is None or _in_backward():
                return
            if not (isinstance(output, (tuple, list)) and len(output) == 3):
                return
            logits, _scores, indices = output
            if indices.shape[0] != mask.shape[0]:
                return
            experts = int(logits.shape[-1])
            occurrence = max(self._occurrences.get((self._root_id, FWD), 1) - 1, 0)
            with torch.no_grad():
                # Static shapes only (no boolean indexing, whose size the host would have to wait for):
                # the image tokens' choices are counted with a weight of 1, every choice with a weight of 1.
                flat = indices.reshape(-1)
                weight = mask.reshape(-1, 1).expand(-1, indices.shape[-1]).reshape(-1).to(torch.float32)
                visual = torch.zeros(experts, dtype=torch.float32, device=flat.device).index_add_(0, flat, weight)
                total = torch.zeros(experts, dtype=torch.float32, device=flat.device).index_add_(
                    0, flat, torch.ones_like(weight))
            self._routing.append((int(layer if layer is not None else -1), occurrence, visual, total - visual))

        return self._protect(hook)

    # -- output --------------------------------------------------------------

    def _drain_routing(self) -> list[dict[str, Any]]:
        """Move every expert histogram of the step to the host in one copy."""
        if not self._routing:
            return []
        stacked = torch.stack([torch.stack([visual, text]) for _, _, visual, text in self._routing]).cpu()
        rows = []
        for (layer, occurrence, _visual, _text), counts in zip(self._routing, stacked.tolist()):
            rows.append({"layer": layer, "mb": occurrence, "visual": [int(round(count)) for count in counts[0]],
                         "text": [int(round(count)) for count in counts[1]]})
        self._routing = []
        return rows

    def _resolve(self, mark: tuple[Any, ...]) -> list[Any]:
        """Turn one recorded boundary into a time since the step start.

        Fields: module id, pass, occurrence, kind, milliseconds since the step
        start, allocated bytes.
        """
        module_id, pass_name, occurrence, kind, stamp, allocated = mark
        return [module_id, pass_name, occurrence, kind,
                round(self._clock.elapsed_ms(self._start_stamp, stamp), 4), allocated]

    def _header(self) -> dict[str, Any]:
        """Describe the rank and the hooked modules, once per file."""
        ready = dist.is_available() and dist.is_initialized()
        return {
            "kind": "header",
            "rank": dist.get_rank() if ready else 0,
            "world_size": dist.get_world_size() if ready else 1,
            "host": socket.gethostname(),
            "device_type": get_device_type(),
            "time_source": "device" if self._clock.uses_events else "host",
            "spatial_merge_size": self.spatial_merge_size,
            "modules": self._modules,
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


HETERO_PROFILE = HeteroProfiler()


def step_digest(record: Mapping[str, Any], modules: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Condense one step record into the few numbers worth logging live.

    Args:
        record: A step record as :meth:`HeteroProfiler.end_step` returns it.
        modules: The header's module table, to name the roots.

    Returns:
        Workload sums over the micro-batches, the forward time of the vision
        tower, the text decoder and the vocabulary projection, and the span of
        the backward boundaries (first to last), in milliseconds.
    """
    roles = {module["id"]: module["role"] for module in modules}
    opened: dict[tuple[int, str, int], float] = {}
    forward = {"vision.root": 0.0, "text.root": 0.0, "lm_head": 0.0}
    backward_times: list[float] = []
    for module_id, pass_name, occurrence, kind, time_ms, *_ in record["marks"]:
        if pass_name == BWD:
            backward_times.append(time_ms)
            continue
        key = (module_id, pass_name, occurrence)
        if kind == ENTER:
            opened[key] = time_ms
        elif key in opened and pass_name == FWD and roles.get(module_id) in forward:
            forward[roles[module_id]] += time_ms - opened.pop(key)
    batches = record["micro_batches"]
    return {
        "micro_batches": float(len(batches)),
        "real_tokens": float(sum(batch.get("real_tokens", 0) for batch in batches)),
        "visual_tokens": float(sum(batch.get("visual_tokens", 0) for batch in batches)),
        "images": float(sum(batch.get("images", 0) for batch in batches)),
        "vision_fwd_ms": forward["vision.root"],
        "text_fwd_ms": forward["text.root"],
        "head_fwd_ms": forward["lm_head"],
        "bwd_ms": (max(backward_times) - min(backward_times)) if backward_times else 0.0,
        "step_ms": float(record["device_ms"]),
    }


__all__ = [
    "BWD", "DEFAULT_ROLES", "ENTER", "EXIT", "FWD", "HETERO_PROFILE", "HeteroProfiler", "RECOMPUTE", "batch_workload",
    "normalize_module_path", "step_digest",
]
