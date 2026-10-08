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
"""The VLM trainer's step loop: reading a step's micro-batches, and stopping on the same step.

The methods under test touch only ``self.base``, so they are exercised unbound against a stand-in
rather than a built trainer, which would need a model and a device.
"""

import importlib.machinery
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


def _vlm_trainer():
    """Import the trainer class, standing torch_npu in for the absent accelerator.

    Imported inside the tests rather than at module scope, and skipped rather than failed when the
    import does not succeed. Reaching this class reaches Transformers, and a whole-directory run of
    ``tests/ut/trainer`` currently cannot load it: torch's inductor test operators get registered
    twice across the session and Transformers' lazy loader then raises. That is a suite-ordering
    defect of the tree, not of this file -- ``test_text_trainer.py`` fails to collect for the same
    reason -- so these tests run on the file and stand aside in the sweep rather than reporting a
    failure they did not cause.
    """
    import hyper_parallel.models.build_options  # pylint: disable=C0415,unused-import

    # build_options is imported first, so its NPU probe resolves to absent before the stub arrives.
    stub = mock.MagicMock()
    stub.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    sys.modules.setdefault("torch_npu", stub)
    try:
        from hyper_parallel.trainer.vlm_trainer import VLMTrainer  # pylint: disable=C0415
    except ImportError as exc:
        pytest.skip(f"the trainer class is not importable in this session: {exc}")
    return VLMTrainer


def _trainer(num_micro_batches, batches, group=None):
    """Return a stand-in carrying only what the step-loop methods read."""
    source = iter(batches)
    mesh = SimpleNamespace(dp_cp_mesh=None if group is None else SimpleNamespace(get_group=lambda: group))
    return SimpleNamespace(base=SimpleNamespace(
        get_batch=lambda _: next(source),
        num_micro_batches=num_micro_batches,
        mesh=mesh,
    ))


def test_prefetch_reads_exactly_one_step_of_micro_batches():
    """A step reads num_micro_batches items and leaves the rest for the next one."""
    VLMTrainer = _vlm_trainer()
    trainer = _trainer(2, ["a", "b", "c", "d"])

    first = VLMTrainer.prefetch_micro_batches(trainer, None)
    second = VLMTrainer.prefetch_micro_batches(trainer, None)

    assert first == ["a", "b"], f"first step mismatch: expected=['a', 'b'], got={first}"
    assert second == ["c", "d"], f"second step mismatch: expected=['c', 'd'], got={second}"


def test_prefetch_raises_when_the_data_runs_out():
    """Exhaustion surfaces as StopIteration from the collective-free read, not mid-step."""
    VLMTrainer = _vlm_trainer()
    trainer = _trainer(2, ["a"])

    try:
        VLMTrainer.prefetch_micro_batches(trainer, None)
    except StopIteration:
        return
    raise AssertionError("prefetch did not raise StopIteration on a short step")


def test_exhaustion_is_local_without_a_data_parallel_group():
    """A single process answers for itself."""
    VLMTrainer = _vlm_trainer()
    trainer = _trainer(1, [])

    for exhausted in (True, False):
        answer = VLMTrainer.data_exhausted_anywhere(trainer, exhausted)
        assert answer is exhausted, f"single-process answer mismatch: expected={exhausted}, got={answer}"


def test_one_rank_running_out_stops_every_rank():
    """A rank with data left still stops, because a rank that has left will join no collective.

    With a fixed batch size every rank runs out on the same step. With a token budget they do not: an
    equal number of samples packs into an unequal number of rows, so the ranks reach the end several
    steps apart and the first to leave would strand the others in the next step's collectives.
    """
    VLMTrainer = _vlm_trainer()
    group = object()
    trainer = _trainer(1, [], group=group)
    calls = []

    def _max_over_group(value, op=None, group=None):
        """Stand in for the real all-reduce: another rank reports exhaustion."""
        calls.append((value, op, group))
        return max(value, 1.0)

    with mock.patch("hyper_parallel.trainer.vlm_trainer.all_reduce", _max_over_group):
        answer = VLMTrainer.data_exhausted_anywhere(trainer, False)

    assert answer is True, f"a rank with data left did not stop: got={answer}"
    assert calls and calls[0][1] == "max", f"the flag must be reduced with max: got={calls}"
    assert calls[0][2] is group, f"the flag must be reduced over the data-parallel group: got={calls}"


def test_no_rank_running_out_lets_the_step_proceed():
    """When every rank has data, the loop carries on."""
    VLMTrainer = _vlm_trainer()
    group = object()
    trainer = _trainer(1, [], group=group)

    with mock.patch("hyper_parallel.trainer.vlm_trainer.all_reduce", lambda value, op=None, group=None: value):
        answer = VLMTrainer.data_exhausted_anywhere(trainer, False)

    assert answer is False, f"the loop stopped with data left everywhere: got={answer}"
