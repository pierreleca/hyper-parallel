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
"""Drive the MoE activation budget and its host swap from the training loop."""

from typing import Any

from hyper_parallel.distributed.expert_parallel.host_swap import HOST_SWAP
from hyper_parallel.trainer.runtime.logging import create_logger

from .base import Callback, TrainerState

logger = create_logger(__name__)


class EPHostSwapCallback(Callback):
    """Reset the swap at each step and log what it moved at the end."""

    def __init__(self, trainer: Any) -> None:
        """Configure the swap from ``TrainerConfig.ep_host_swap``."""
        super().__init__(trainer)
        self.config = trainer.config.ep_host_swap
        HOST_SWAP.configure(enabled=self.config.enabled, budget_layers=self.config.budget_layers)

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Announce the budget."""
        del state, kwargs
        if self.config.enabled:
            logger.info("EP host swap: MoE activations of a pass kept within %.2f mean layers",
                        self.config.budget_layers)

    def on_step_begin(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Start a new step's layer list."""
        del state, kwargs
        if self.config.enabled:
            HOST_SWAP.begin_step()

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Log what the step moved to host."""
        del kwargs
        if not self.config.enabled:
            return
        summary = HOST_SWAP.end_step()
        if summary is None or not summary["swapped_layers"]:
            return
        logger.info(
            "EP host swap rank%s step %s: %d of %d MoE layers swapped, %.2f GiB to host",
            self.trainer.global_rank,
            state.global_step + 1,
            summary["swapped_layers"],
            summary["moe_layers"],
            summary["d2h_gib"],
        )


__all__ = ["EPHostSwapCallback"]
