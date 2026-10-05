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
"""Training-loop, debug, wandb, profiling, EP-instrument and heterogeneity-profile configuration sections.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class names, fields and defaults are unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

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
class HeteroProfileConfig:
    """Per-component timing and per-micro-batch workload recording settings.

    Read by ``HeteroProfileCallback``; the recording itself lives in
    ``hyper_parallel.trainer.runtime.hetero_profile``. It explains how the
    imbalance between ranks arises: which model component takes the time, and
    which property of the samples (tokens, images, image size) makes it take
    more on one rank than on another.
    """

    enabled: bool = False
    output_dir: str = "./outputs/hetero_profile"
    start_step: int = 1
    end_step: int = 0
    # Hook the model's modules. Off, only the step record is written (wall and device time of the step,
    # the gap since the previous one, the workload and fingerprint of each micro-batch, loss and gradient
    # norm): cheap enough to time a baseline and a candidate with the recorder on.
    hooks: bool = True
    # Further modules to hook, as "role=regex" (matched against module paths without wrapper segments; the
    # first group of the regex is the block index). Unknown roles are summed into the report's "custom" part.
    extra_roles: List[str] = field(default_factory=list)
    # Time each vision block, merger and patch embedding, not only the vision tower.
    vision_blocks: bool = True
    # Time the attention and the MoE block of each decoder layer, not only the layer.
    sublayers: bool = True
    # Per MoE layer, the expert histogram of the image tokens and of the text tokens.
    routing_by_modality: bool = True
    # Read the allocator's allocated bytes at every boundary.
    memory: bool = True
    # Record each step's peak allocated and reserved bytes, which resets the allocator's peak counters
    # at every step; the trainer's own peak metric folds the recorder's running peak back in. Leave it
    # off together with ep_instrument.segment_peaks, which resets them in the middle of the step.
    step_peaks: bool = True

    def __post_init__(self) -> None:
        """Reject a recording window that can never open."""
        if self.start_step < 1:
            raise ValueError("hetero_profile.start_step must be at least 1")
        if self.end_step and self.end_step <= self.start_step:
            raise ValueError(
                "hetero_profile.end_step must be greater than start_step, or 0 for every step"
            )


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
