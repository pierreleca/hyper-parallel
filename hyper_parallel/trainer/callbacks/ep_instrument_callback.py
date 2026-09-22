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
"""Drive the expert-parallel imbalance recorder from the training loop."""

from typing import Any

from hyper_parallel.distributed.expert_parallel.instrument import EP_INSTRUMENT
from hyper_parallel.trainer.runtime.logging import create_logger

from .base import Callback, TrainerState

logger = create_logger(__name__)


class EPInstrumentCallback(Callback):
    """Open and close one recording window per optimizer step.

    Steps are counted from one, as the training log counts them, and the
    window is ``[start_step, end_step)``; ``end_step = 0`` records until the
    run ends.
    """

    def __init__(self, trainer: Any) -> None:
        """Configure the recorder from ``TrainerConfig.ep_instrument``."""
        super().__init__(trainer)
        self.config = trainer.config.ep_instrument
        EP_INSTRUMENT.configure(
            enabled=self.config.enabled,
            output_dir=self.config.output_dir,
            segment_peaks=self.config.segment_peaks,
            align_steps=self.config.align_steps,
            record_counts=self.config.record_counts,
        )

    def _records(self, state: TrainerState) -> bool:
        """Return whether the step that is starting is inside the window."""
        step = state.global_step + 1
        if step < self.config.start_step:
            return False
        return self.config.end_step == 0 or step < self.config.end_step

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Name the MoE blocks and announce where the files go."""
        del state, kwargs
        if not self.config.enabled:
            return
        EP_INSTRUMENT.register_modules(self.trainer.model)
        logger.info(
            "EP instrument: writing per-rank records to %s",
            self.config.output_dir,
        )

    def on_step_begin(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Start recording when the step is inside the window."""
        del kwargs
        if self.config.enabled and self._records(state):
            EP_INSTRUMENT.begin_step(state.global_step + 1)

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Resolve the step's stamps, append the record and log its digest."""
        del kwargs
        if not self.config.enabled:
            return
        record = EP_INSTRUMENT.end_step()
        if record is None:
            return
        digest = EP_INSTRUMENT.step_summary(record)
        logger.info(
            "EP instrument rank%s step %s: %.0fk routed pairs, MoE fwd %.1f ms, "
            "recompute %.1f ms, bwd %.1f ms, peak %.2f GiB",
            self.trainer.global_rank,
            state.global_step + 1,
            digest["pairs"] / 1e3,
            digest["fwd_ms"],
            digest["recompute_ms"],
            digest["bwd_ms"],
            digest["peak_gib"],
        )

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Close this rank's file."""
        del state, kwargs
        if self.config.enabled:
            EP_INSTRUMENT.close()


__all__ = ["EPInstrumentCallback"]
