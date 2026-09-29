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
    """Reset the swap at each step and record its copies at the end."""

    def __init__(self, trainer: Any) -> None:
        """Configure the swap from ``TrainerConfig.ep_host_swap``."""
        super().__init__(trainer)
        self.config = trainer.config.ep_host_swap
        HOST_SWAP.configure(
            enabled=self.config.enabled,
            budget_layers=self.config.budget_layers,
            min_row_bytes=self.config.min_row_bytes,
            output_dir=self.config.output_dir,
        )

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Announce the budget and where the records go."""
        del state, kwargs
        if self.config.enabled:
            logger.info(
                "EP host swap: MoE activations of a pass kept within %.2f mean layers, records in %s",
                self.config.budget_layers,
                self.config.output_dir,
            )

    def on_step_begin(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Start a new step's layer list."""
        del kwargs
        if self.config.enabled:
            HOST_SWAP.begin_step(state.global_step + 1)

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Resolve the copy timings, append the record and log its digest."""
        del kwargs
        if not self.config.enabled:
            return
        record = HOST_SWAP.end_step(self.trainer.global_rank)
        if record is None or not record["swapped_layers"]:
            return
        logger.info(
            "EP host swap rank%s step %s: %d of %d MoE layers swapped, %.2f GiB to host in %.1f ms"
            " at %s GB/s (never waited for), back in %.1f ms at %s GB/s; copy back %.1f ms hidden"
            " under compute, %.1f ms exposed",
            self.trainer.global_rank,
            state.global_step + 1,
            record["swapped_layers"],
            record["moe_layers"],
            record["d2h_gib"],
            record["d2h_ms"],
            _rate(record["d2h_gbps"]),
            record["h2d_ms"],
            _rate(record["h2d_gbps"]),
            record["h2d_hidden_ms"],
            record["stall_ms"],
        )

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Close this rank's file."""
        del state, kwargs
        if self.config.enabled:
            HOST_SWAP.close()


def _rate(value: Any) -> str:
    """A bandwidth for the log, or '-' when no copy was timed."""
    return "-" if value is None else f"{value:.1f}"


__all__ = ["EPHostSwapCallback"]
