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
"""Build options of ``HyperAutoModel*.from_pretrained`` and ``from_config``.

- ``allow_uncovered_params`` reaches the sharding planner through the
  ``DistributedSetup`` that infrastructure is built from.

Infrastructure and the model build are mocked, so no checkpoint, Hub access or
device is needed.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers import auto_model
from tests.common.mark_utils import arg_mark

_PATH = "/checkpoints/model"


class _MockedBuildTestCase(unittest.TestCase):
    """Mock infrastructure creation, config resolution and the model build."""

    def setUp(self) -> None:
        """Start the mocks and register their cleanup."""
        self.read_config = SimpleNamespace(architectures=["ReadFromPath"], num_hidden_layers=48)
        patches = (
            mock.patch.object(auto_model, "instantiate_infrastructure", return_value=(None, None)),
            mock.patch.object(auto_model, "get_hf_config", return_value=self.read_config),
            mock.patch.object(auto_model.HyperAutoModelForCausalLM, "_build_model"),
        )
        self.instantiate, self.get_hf_config, self.build_model = (patcher.start() for patcher in patches)
        for patcher in patches:
            self.addCleanup(patcher.stop)

    def planner_setup(self) -> DistributedSetup:
        """Return the setup that infrastructure, and so the planner, was built from."""
        return self.instantiate.call_args.kwargs["distributed_setup"]


class TestAllowUncoveredParams(_MockedBuildTestCase):
    """``allow_uncovered_params`` reaches the planner's setup on both entry points."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_from_pretrained_sets_setup_flag(self):
        """from_pretrained marks the caller's setup before infrastructure is built."""
        setup = DistributedSetup()

        auto_model.HyperAutoModelForCausalLM.from_pretrained(
            _PATH, distributed_setup=setup, allow_uncovered_params=True
        )

        self.assertIs(self.planner_setup(), setup,
                      f"infrastructure must be built from the caller's setup, got {self.planner_setup()}")
        flag = getattr(setup, "allow_uncovered_params", False)
        self.assertTrue(flag, f"expected allow_uncovered_params on the setup, got {flag}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_from_config_sets_setup_flag(self):
        """from_config marks the caller's setup before infrastructure is built."""
        setup = DistributedSetup()

        auto_model.HyperAutoModelForCausalLM.from_config(
            SimpleNamespace(architectures=["Supplied"]), distributed_setup=setup, allow_uncovered_params=True
        )

        self.assertIs(self.planner_setup(), setup,
                      f"infrastructure must be built from the caller's setup, got {self.planner_setup()}")
        flag = getattr(setup, "allow_uncovered_params", False)
        self.assertTrue(flag, f"expected allow_uncovered_params on the setup, got {flag}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_flag_defaults_off(self):
        """Without the option the planner keeps its fail-fast coverage check."""
        setup = DistributedSetup()

        auto_model.HyperAutoModelForCausalLM.from_pretrained(_PATH, distributed_setup=setup)

        flag = getattr(setup, "allow_uncovered_params", False)
        self.assertFalse(flag, f"allow_uncovered_params must default to off, got {flag}")


if __name__ == "__main__":
    unittest.main()
