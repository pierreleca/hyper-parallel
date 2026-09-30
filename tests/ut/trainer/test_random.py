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
"""Full-determinism setup: Ascend collectives read HCCL_DETERMINISTIC."""

from types import SimpleNamespace

import pytest
import torch

from hyper_parallel.trainer.runtime import random as trainer_random

_TOUCHED = (
    "HCCL_DETERMINISTIC", "NCCL_DETERMINISTIC", "CLOSE_MATMUL_K_SHIFT", "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG", "FLASH_ATTENTION_DETERMINISTIC",
)


@pytest.fixture(name="npu_like")
def fixture_npu_like(monkeypatch):
    """Take the Ascend branch on a CPU host and restore the global state after."""
    for name in _TOUCHED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(trainer_random, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(manual_seed=lambda seed: None, manual_seed_all=lambda seed: None),
        raising=False,
    )
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled = cudnn


def test_full_determinism_sets_the_hccl_variable(npu_like):
    """HCCL reads HCCL_DETERMINISTIC; NCCL_DETERMINISTIC alone left its reductions free."""
    del npu_like
    trainer_random.enable_full_determinism(42)
    assert trainer_random.os.environ["HCCL_DETERMINISTIC"] == "true"
    assert torch.are_deterministic_algorithms_enabled()


def test_full_determinism_keeps_an_exported_hccl_mode(npu_like, monkeypatch):
    """A stricter mode the user exported is not overwritten."""
    del npu_like
    monkeypatch.setenv("HCCL_DETERMINISTIC", "strict")
    trainer_random.enable_full_determinism(42)
    assert trainer_random.os.environ["HCCL_DETERMINISTIC"] == "strict"
