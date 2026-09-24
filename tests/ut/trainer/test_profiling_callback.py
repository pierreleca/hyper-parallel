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
"""Profiling callback: which ranks record a trace."""

from types import SimpleNamespace

import pytest

from hyper_parallel.trainer.callbacks.profiling_callback import ProfilingCallback
from hyper_parallel.trainer.config.training import ProfilingConfig


def _trainer(rank: int, global_rank: int, world_size: int = 4) -> SimpleNamespace:
    """A trainer stand-in carrying only what the callback reads."""
    config = SimpleNamespace(profiling=ProfilingConfig(enabled=True, start_step=2, end_step=3, rank=rank))
    return SimpleNamespace(config=config, mesh=None, global_rank=global_rank, world_size=world_size)


@pytest.mark.parametrize("global_rank", [0, 1, 3])
def test_rank_minus_one_profiles_every_rank(global_rank: int) -> None:
    """rank -1 records on every rank."""
    assert ProfilingCallback(_trainer(-1, global_rank)).enabled


def test_one_rank_profiles_only_that_rank() -> None:
    """A rank index records on that rank alone."""
    assert ProfilingCallback(_trainer(2, 2)).enabled
    assert not ProfilingCallback(_trainer(2, 1)).enabled


@pytest.mark.parametrize("rank", [-2, 4])
def test_out_of_range_rank_is_rejected(rank: int) -> None:
    """Anything but -1 or a rank of the job is a configuration error."""
    with pytest.raises(ValueError, match="profiling.rank"):
        ProfilingCallback(_trainer(rank, 0))
