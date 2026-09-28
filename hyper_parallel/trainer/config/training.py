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
"""Training-loop, debug, wandb, profiling and EP-instrument configuration sections.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class names, fields and defaults are unchanged.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from hyper_parallel.components.quantization.config import LowPrecisionConfig


@dataclass
class TrainingConfig:
    """Training-loop parameters exposed by the initial YAML schema."""

    train_iters: Optional[int] = None
    train_samples: Optional[int] = None
    eval_iters: int = 0

    global_batch_size: int = 8
    micro_batch_size: int = 1

    backend: Literal["nccl", "hccl", "gloo"] = "nccl"
    max_grad_norm: float = 1.0
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"
    seed: Optional[int] = None
    enable_full_determinism: bool = False
    gc_steps: int = 0
    empty_cache_steps: int = 0
    empty_cache_before_backward: bool = False
    eval_steps: int = 0
    eval_epochs: int = 0
    logging_steps: int = 1
    low_precision: LowPrecisionConfig = field(default_factory=LowPrecisionConfig)


@dataclass
class DebugConfig:
    """Debug parameters exposed by the initial YAML schema."""

    check_dataset: Optional[Literal["debug", "info", "warn"]] = None
    check_nan_inf: bool = False


@dataclass
class WandbConfig:
    """WandB remote-logging parameters (03 §4.2.5: read by build_callback_manager)."""

    enabled: bool = False
    project: str = ""
    entity: Optional[str] = None


@dataclass
class EPInstrumentConfig:
    """Expert-parallel imbalance measurement settings.

    Read by ``EPInstrumentCallback``; the recording itself lives in
    ``hyper_parallel.distributed.expert_parallel.instrument``.
    """

    enabled: bool = False
    output_dir: str = "./outputs/ep_instrument"
    start_step: int = 1
    end_step: int = 0
    segment_peaks: bool = True
    align_steps: bool = True
    record_counts: bool = True

    def __post_init__(self) -> None:
        """Reject a recording window that can never open."""
        if self.start_step < 1:
            raise ValueError("ep_instrument.start_step must be at least 1")
        if self.end_step and self.end_step <= self.start_step:
            raise ValueError(
                "ep_instrument.end_step must be greater than start_step, or 0 for every step"
            )


@dataclass
class EPHostSwapConfig:
    """Budget for the MoE activations, per layer or per step, the excess swapped to host.

    Read by ``EPHostSwapCallback``; the swap itself lives in
    ``hyper_parallel.distributed.expert_parallel.host_swap``. With ``budget:
    layer`` each MoE layer keeps at most ``capacity_factor`` times the routed
    pairs a rank sends; the saved tensors beyond that go to pinned host memory
    in forward and come back in backward. With ``budget: step`` a rank keeps at
    most ``capacity_factor`` times its mean load summed over all its MoE layers,
    and swaps its earliest layers as soon as its projected total goes over; the
    factor may then be below 1, making every rank swap every step.
    """

    enabled: bool = False
    capacity_factor: float = 1.2
    # "layer": a budget per MoE layer. "step": one budget for all MoE layers of a
    # forward pass, earliest layers swapped first ("tensors" granularity only).
    budget: str = "layer"
    # What goes to host: "tensors" moves whole saved tensors, with no device copy
    # but up to a whole tensor more host traffic; "rows" moves about the excess
    # only, but its kept-rows buffers (a new size every step, allocated near the
    # peak) fragment the allocator: on 4 A2 dies the reserve grew 0.5-1.2 GiB,
    # more than the swap saved.
    granularity: str = "tensors"
    # Saved tensors with fewer bytes per routed pair (indices) stay on device.
    min_row_bytes: int = 1024
    # One JSON Lines file per rank: bytes moved, copy times and waits per step.
    output_dir: str = "./outputs/ep_host_swap"

    def __post_init__(self) -> None:
        """Reject a budget that cannot hold anything."""
        if self.capacity_factor <= 0:
            raise ValueError("ep_host_swap.capacity_factor must be positive")
        if self.granularity not in ("rows", "tensors"):
            raise ValueError(f"ep_host_swap.granularity must be 'rows' or 'tensors', not {self.granularity!r}")
        if self.budget not in ("layer", "step"):
            raise ValueError(f"ep_host_swap.budget must be 'layer' or 'step', not {self.budget!r}")
        if self.budget == "step" and self.granularity != "tensors":
            raise ValueError("ep_host_swap.budget 'step' supports 'tensors' granularity only")


@dataclass
class ProfilingConfig:
    """Lightweight per-step profiler settings."""

    enabled: bool = False
    start_step: int = 3
    end_step: int = 4
    trace_dir: str = "./outputs/profiling"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_modules: bool = False
    # The rank to profile, or -1 for every rank (one rank<N>_<time>_ascend_pt directory each).
    rank: int = 0
