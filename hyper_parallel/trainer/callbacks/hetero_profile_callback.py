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
"""Drive the heterogeneity profiler from the training loop."""

from typing import Any, Dict, List, Optional

from hyper_parallel.trainer.runtime.hetero_profile import HETERO_PROFILE, step_digest
from hyper_parallel.trainer.runtime.logging import create_logger

from .base import Callback, TrainerState

logger = create_logger(__name__)


class HeteroProfileCallback(Callback):
    """Open and close one recording window per optimizer step.

    Steps are counted from one, as the training log counts them, and the
    window is ``[start_step, end_step)``; ``end_step = 0`` records until the
    run ends. The step's micro-batches are read before the recording starts, so
    reading them is not part of what the step is measured to take.
    """

    def __init__(self, trainer: Any) -> None:
        """Configure the profiler from ``TrainerConfig.hetero_profile``."""
        super().__init__(trainer)
        self.config = trainer.config.hetero_profile
        self.profiler = HETERO_PROFILE
        self.profiler.configure(
            enabled=self.config.enabled,
            output_dir=self.config.output_dir,
            vision_blocks=self.config.vision_blocks,
            sublayers=self.config.sublayers,
            routing_by_modality=self.config.routing_by_modality,
            memory=self.config.memory,
            step_peaks=self.config.step_peaks,
            hooks=self.config.hooks,
            extra_roles=self.config.extra_roles,
        )

    def _records(self, state: TrainerState) -> bool:
        """Return whether the step that is starting is inside the window."""
        step = state.global_step + 1
        if step < self.config.start_step:
            return False
        return self.config.end_step == 0 or step < self.config.end_step

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook the model's modules and announce where the files go."""
        del state, kwargs
        if not self.config.enabled:
            return
        counts = self.profiler.attach(self.trainer.model) or {}
        logger.info(
            "Hetero profile: %s; writing per-rank records to %s",
            ("hooked " + ", ".join(f"{count} {role}" for role, count in sorted(counts.items())))
            if self.config.hooks else "step records only (hooks off)",
            self.config.output_dir,
        )

    def on_step_begin(  # pylint: disable=arguments-differ
            self,
            state: TrainerState,
            micro_batches: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
    ) -> None:
        """Start recording when the step is inside the window."""
        del kwargs
        if self.config.enabled and self._records(state):
            self.profiler.begin_step(state.global_step + 1, micro_batches)

    def on_step_end(  # pylint: disable=arguments-differ
            self,
            state: TrainerState,
            loss: Optional[float] = None,
            grad_norm: Optional[float] = None,
            **kwargs: Any,
    ) -> None:
        """Resolve the step's stamps, append the record and log its digest."""
        del kwargs
        if not self.config.enabled:
            return
        record = self.profiler.end_step(loss=loss, grad_norm=grad_norm)
        if record is None:
            return
        digest = step_digest(record, self.profiler.modules)
        logger.info(
            "Hetero profile rank%s step %s: %d micro-batch(es), %.0f tokens, %.0f visual tokens, %.0f images; "
            "vision fwd %.1f ms, text fwd %.1f ms, head fwd %.1f ms, bwd %.1f ms, step %.1f ms",
            self.trainer.global_rank,
            state.global_step + 1,
            digest["micro_batches"],
            digest["real_tokens"],
            digest["visual_tokens"],
            digest["images"],
            digest["vision_fwd_ms"],
            digest["text_fwd_ms"],
            digest["head_fwd_ms"],
            digest["bwd_ms"],
            digest["step_ms"],
        )

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Close this rank's file."""
        del state, kwargs
        if self.config.enabled:
            self.profiler.close()


__all__ = ["HeteroProfileCallback"]
