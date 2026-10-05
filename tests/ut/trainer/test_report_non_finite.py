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
"""debug.check_nan_inf names the parameters whose gradient is not finite."""

import logging
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.trainer.base import BaseTrainer


def _trainer(model: nn.Module) -> SimpleNamespace:
    """The attributes the report reads, without building a trainer."""
    return SimpleNamespace(model=model, state=SimpleNamespace(global_step=3), global_rank=5)


def test_names_the_parameters_with_a_non_finite_gradient(caplog):
    """A NaN gradient and an infinite loss are logged with the parameter's name, the step and the rank."""
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[0].weight.grad = torch.zeros(2, 2)
    model[1].weight.grad = torch.tensor([[1.0, float("nan")], [0.0, 0.0]])
    with caplog.at_level(logging.WARNING):
        BaseTrainer.report_non_finite(_trainer(model), float("inf"))
    assert "non-finite at step 3 rank 5" in caplog.text and "1.weight" in caplog.text
    assert "0.weight" not in caplog.text


def test_says_nothing_when_everything_is_finite(caplog):
    """Finite gradients and loss log nothing."""
    model = nn.Linear(2, 2)
    model.weight.grad = torch.ones(2, 2)
    with caplog.at_level(logging.WARNING):
        BaseTrainer.report_non_finite(_trainer(model), 1.5)
    assert "non-finite" not in caplog.text
