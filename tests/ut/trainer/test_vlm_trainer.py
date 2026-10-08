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
        answer = VLMTrainer.data_exhausted_everywhere(trainer, exhausted)
        assert answer is exhausted, f"single-process answer mismatch: expected={exhausted}, got={answer}"


def test_one_rank_running_out_does_not_end_the_epoch():
    """A rank that finishes early keeps going, so the data the others still hold is not thrown away.

    With a fixed batch size every rank runs out on the same step. With a token budget they do not: an
    equal number of samples packs into an unequal number of rows, so stopping at the first rank to
    finish would drop what the others still hold -- nearly a third of this study's epoch.
    """
    VLMTrainer = _vlm_trainer()
    group = object()
    trainer = _trainer(1, [], group=group)
    calls = []

    def _min_over_group(value, op=None, group=None):
        """Stand in for the real all-reduce: another rank still has data."""
        calls.append((value, op, group))
        return min(value, 0.0)

    with mock.patch("hyper_parallel.trainer.vlm_trainer.all_reduce", _min_over_group):
        answer = VLMTrainer.data_exhausted_everywhere(trainer, True)

    assert answer is False, f"the epoch ended with data left on another rank: got={answer}"
    assert calls and calls[0][1] == "min", f"the flag must be reduced with min: got={calls}"
    assert calls[0][2] is group, f"the flag must be reduced over the data-parallel group: got={calls}"


def test_the_epoch_ends_when_every_rank_has_run_out():
    """Once no rank has data, the loop stops."""
    VLMTrainer = _vlm_trainer()
    group = object()
    trainer = _trainer(1, [], group=group)

    with mock.patch("hyper_parallel.trainer.vlm_trainer.all_reduce", lambda value, op=None, group=None: value):
        answer = VLMTrainer.data_exhausted_everywhere(trainer, True)

    assert answer is True, f"the loop carried on with no data anywhere: got={answer}"


def test_padded_work_supervises_no_token_at_all():
    """A finished rank replays its last micro-batch supervising nothing, so no sample trains twice.

    Keeping even one token would train on a replayed sample a second time. Masking every one of them
    is exact rather than merely small: the cross-entropy writes a gradient only where it supervises a
    token, so the parameter gradients stay at zero -- ``check_padded_work.py`` asserts that on the
    real model, including with the router's load-balancing term on.
    """
    import torch  # pylint: disable=C0415

    VLMTrainer = _vlm_trainer()
    labels = torch.tensor([[-100, -100, 7, 8, 9]])
    template = [({"input_ids": torch.ones(1, 5, dtype=torch.long), "labels": labels},
                 {"labels": labels, "loss_mask": labels >= 0})]

    padded = VLMTrainer.padding_micro_batches(_trainer(1, []), template)

    model_inputs, loss_inputs = padded[0]
    supervised = int((loss_inputs["labels"] != -100).sum())
    assert supervised == 0, f"padded work must supervise no token: got={supervised}"
    assert int(loss_inputs["loss_mask"].sum()) == 0, \
        f"the loss mask must follow the labels: got={int(loss_inputs['loss_mask'].sum())}"
    # The model computes the loss from the labels it is handed, so they must agree with the weighting.
    assert int((model_inputs["labels"] != -100).sum()) == 0, \
        f"the model's labels must be masked too: got={int((model_inputs['labels'] != -100).sum())}"
    assert model_inputs["input_ids"].shape == (1, 5), \
        f"padded work must keep the shape of real work: got={tuple(model_inputs['input_ids'].shape)}"
    # The template is what the rank really read; padding must not alter it in place.
    assert int((labels != -100).sum()) == 3, \
        f"the template was masked in place: got={int((labels != -100).sum())}"


def test_padded_work_counts_no_token_for_the_loss_weighting():
    """A padded micro-batch must not dilute the step's token denominator either."""
    import torch  # pylint: disable=C0415

    from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token  # pylint: disable=C0415

    VLMTrainer = _vlm_trainer()
    labels = torch.tensor([[-100, -100, 7, 8, 9]])
    template = [({"input_ids": torch.ones(1, 5, dtype=torch.long), "labels": labels},
                 {"labels": labels, "loss_mask": labels >= 0})]

    _, loss_inputs = VLMTrainer.padding_micro_batches(_trainer(1, []), template)[0]

    assert int(count_loss_token(template[0][1])["foundation_tokens"]) > 0, \
        "the template counted no token, so this test proves nothing"
    counted = int(count_loss_token(loss_inputs)["foundation_tokens"])
    assert counted == 0, f"padded work must count no supervised token: got={counted}"


def test_padded_work_without_a_template_is_refused():
    """A rank that never read a micro-batch was handed no data, which is not a ragged epoch."""
    VLMTrainer = _vlm_trainer()

    try:
        VLMTrainer.padding_micro_batches(_trainer(1, []), None)
    except ValueError as error:
        assert "configuration" in str(error), f"the error should name the cause: got={error}"
        return
    raise AssertionError("a missing template was accepted")
