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
"""The 32-die heterogeneity configuration resolves with the Trainer's own parser, targets and overrides included."""

import pathlib
import subprocess
import sys
import textwrap

_ROOT = pathlib.Path(__file__).parents[3]
_CONFIG = _ROOT / "examples" / "qwen3_vl_30b_perf" / "train_32dev_a3_hetero.yaml"

# Run in a subprocess: resolving the configuration imports its targets, which need a torch_npu stand-in, and that
# must not leak into the other tests of the process.
_SCRIPT = textwrap.dedent("""
    import importlib.machinery, sys
    from unittest import mock
    import hyper_parallel.models.build_options
    from transformers.utils import import_utils
    import_utils.is_torch_npu_available()
    stub = mock.MagicMock()
    stub.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    sys.modules["torch_npu"] = stub
    sys.argv = ["train_vl.py", sys.argv[1], *sys.argv[2:]]
    from hyper_parallel.trainer.config.parser import parse_training_args
    config = parse_training_args()
    print(config.fsdp_config.dp_shard_size, config.fsdp_config.edp_shard_size, config.accelerator.ep_size,
          config.training.micro_batch_size, config.training.global_batch_size)
    print(config.hetero_profile.enabled, config.hetero_profile.output_dir, config.hetero_profile.start_step)
    print(config.ep_instrument.enabled, config.ep_instrument.segment_peaks, config.activation_checkpoint.mode)
    print(config.dataset.data_path)
    print(config.dataset.data_transform._target_path)
    print(config.model._target_path)
    print(config.hetero_profile.hooks, config.hetero_profile.enabled, config.ep_instrument.enabled,
          config.training.train_iters, config.training.global_batch_size)
""")


def _resolve(*overrides: str) -> list:
    """Resolve the configuration with these overrides in a clean interpreter; return its printed lines."""
    env = {"HYPER_PARALLEL_PLATFORM": "torch", "PYTHONPATH": str(_ROOT), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(_CONFIG), *overrides], cwd=_ROOT, env=env, capture_output=True,
        text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-7:]


def test_configuration_resolves_with_the_defaults():
    """2 nodes x 16 dies: FSDP 32, EP 16, experts FSDP-sharded over the nodes; one sample per rank per step."""
    lines = _resolve()
    assert lines[0] == "32 2 16 1 32"
    assert lines[1].startswith("True /home/pl/runs/qwen3_vl_30b_perf/a3_32dev_hetero/hetero 3")
    assert lines[2] == "True False full"
    assert "variable_length_transform" in lines[4]
    assert "cropped_qwen3_vl" in lines[5]


def test_overrides_reach_the_new_sections():
    """The campaign sets the record directories, the dataset and the recompute mode on the command line."""
    lines = _resolve(
        "--hetero_profile.output_dir=/tmp/run/hetero", "--hetero_profile.start_step=2",
        "--dataset.data_path=/data/hetero_both_n640/vlm_conversations.balanced.json",
        "--activation_checkpoint.mode=selective", "--model.num_hidden_layers=8",
    )
    assert lines[1].startswith("True /tmp/run/hetero 2")
    assert lines[2] == "True False selective"
    assert lines[3].endswith("vlm_conversations.balanced.json")


def test_the_flags_of_the_plans_are_accepted():
    """The A/B and baseline plans switch the recorders and the length of the run with these overrides."""
    light = _resolve("--hetero_profile.hooks=false", "--ep_instrument.enabled=false", "--training.train_iters=20")
    assert light[6] == "False True False 20 32"
    off = _resolve("--hetero_profile.enabled=false", "--ep_instrument.enabled=false",
                   "--training.global_batch_size=64", "--training.train_iters=10")
    assert off[6] == "True False False 10 64"
