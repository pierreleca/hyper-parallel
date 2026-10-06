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

from hyper_parallel.trainer.callbacks.profiling_callback import ProfilingCallback, profiled_ranks
from hyper_parallel.trainer.config.training import ProfilingConfig


def _trainer(rank: int, global_rank: int, world_size: int = 4, ranks: list = None) -> SimpleNamespace:
    """A trainer stand-in carrying only what the callback reads."""
    config = SimpleNamespace(
        profiling=ProfilingConfig(enabled=True, start_step=2, end_step=3, rank=rank, ranks=list(ranks or []))
    )
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


@pytest.mark.parametrize(("global_rank", "recorded"), [(0, True), (1, False), (3, True)])
def test_a_list_of_ranks_records_on_those_ranks_alone(global_rank: int, recorded: bool) -> None:
    """``ranks`` names the few ranks a study compares, so a profiled run writes a trace per named rank, not per rank."""
    assert ProfilingCallback(_trainer(-1, global_rank, ranks=[0, 3])).enabled is recorded


def test_a_list_of_ranks_wins_over_rank() -> None:
    """With ``ranks`` set, ``rank`` is not read: neither its value nor its range matters."""
    assert ProfilingCallback(_trainer(2, 3, ranks=[3])).enabled
    assert not ProfilingCallback(_trainer(2, 2, ranks=[3])).enabled
    assert ProfilingCallback(_trainer(-99, 3, ranks=[3])).enabled, "an unread rank is not validated"


@pytest.mark.parametrize("ranks", [[0, 4], [-1], [1, 2, 9]])
def test_ranks_outside_the_job_are_rejected(ranks: list) -> None:
    """A rank the job does not have would silently record nothing."""
    with pytest.raises(ValueError, match="profiling.ranks"):
        ProfilingCallback(_trainer(0, 0, ranks=ranks))


def test_which_ranks_are_profiled() -> None:
    """The rule the callback applies, as a set: the list when given, else the rank, -1 meaning every rank."""
    assert profiled_ranks(ProfilingConfig(rank=-1), 4) == {0, 1, 2, 3}
    assert profiled_ranks(ProfilingConfig(rank=2), 4) == {2}
    assert profiled_ranks(ProfilingConfig(rank=-1, ranks=[1, 3]), 4) == {1, 3}


def test_the_raw_collection_directory_is_kept_unless_asked_otherwise() -> None:
    """The profiler keeps its raw data by default; a study short of disk sets data_simplification."""
    assert ProfilingConfig().data_simplification is False
    assert ProfilingConfig(data_simplification=True).data_simplification is True
