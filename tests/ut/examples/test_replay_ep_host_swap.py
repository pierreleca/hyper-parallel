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
"""The host-swap replay script picks tensors as the swap does."""

import importlib.util
import pathlib
import random
import sys

from hyper_parallel.distributed.expert_parallel.host_swap import choose_offload

_EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "examples" / "qwen3_vl_30b_perf"


def test_the_replay_script_picks_tensors_as_the_swap_does(monkeypatch):
    """replay_ep_host_swap.py carries its own copy of choose_offload; it must agree with the swap's."""
    monkeypatch.syspath_prepend(str(_EXAMPLES))
    spec = importlib.util.spec_from_file_location("replay_ep_host_swap", _EXAMPLES / "replay_ep_host_swap.py")
    replay = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "replay_ep_host_swap", replay)  # its dataclasses look the module up
    spec.loader.exec_module(replay)
    generator = random.Random(0)
    for _ in range(200):
        sizes = [generator.randint(1, 5000) for _ in range(generator.randint(1, 6))]
        need = generator.randint(-10, sum(sizes) + 10)
        assert replay.choose_offload(sizes, need) == choose_offload(sizes, need), (sizes, need)
