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
_PACKING_CONFIG = _ROOT / "examples" / "qwen3_vl_30b_perf" / "train_32dev_a3_packing.yaml"
# The packing configuration is a copy, because the config system loads one file and refuses to
# change a _target_ from the command line. These are the only keys it is allowed to differ in.
_PACKING_DELTAS = {
    "dataloader._target_": ("hyper_parallel.data.batching.FixedBatchDataLoader",
                            "hyper_parallel.data.batching.DynamicBatchDataLoader"),
    "dataloader.min_buffered_samples": (None, 2),
    "dataloader.collate_fn.packing": (False, True),
    "model.packed_position_ids": (False, True),
    "dataset.data_transform.max_seq_len": (20000, 16384),
}

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
    print(config.dataset.data_transform._target_path, config.dataset.data_transform.padding,
          config.dataset.data_transform.text_only)
    print(config.model._target_path)
    print(config.hetero_profile.hooks, config.hetero_profile.enabled, config.ep_instrument.enabled,
          config.training.train_iters, config.training.global_batch_size)
    print(config.profiling.enabled, config.profiling.ranks, config.profiling.start_step,
          config.profiling.end_step, config.profiling.data_simplification)
""")


def _resolve(*overrides: str) -> list:
    """Resolve the configuration with these overrides in a clean interpreter; return its printed lines."""
    env = {"HYPER_PARALLEL_PLATFORM": "torch", "PYTHONPATH": str(_ROOT), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(_CONFIG), *overrides], cwd=_ROOT, env=env, capture_output=True,
        text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-8:]


def _flatten(node, prefix=""):
    """Return the configuration as a flat mapping of dotted path to leaf value."""
    flat = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            flat.update(_flatten(value, f"{prefix}{index}."))
    else:
        flat[prefix.rstrip(".")] = node
    return flat


def test_the_packing_config_is_the_study_config_plus_the_documented_deltas():
    """The copied packing configuration may differ only in the keys its header names."""
    import yaml  # pylint: disable=C0415

    base = _flatten(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))
    packing = _flatten(yaml.safe_load(_PACKING_CONFIG.read_text(encoding="utf-8")))

    differing = {key for key in set(base) | set(packing) if base.get(key) != packing.get(key)}
    expected = set(_PACKING_DELTAS)
    assert differing == expected, (f"the packing config drifted: unexpected={sorted(differing - expected)}, "
                                  f"missing={sorted(expected - differing)}")
    for key, (was, now) in _PACKING_DELTAS.items():
        assert base.get(key) == was, f"{key} in the study config: expected={was}, got={base.get(key)}"
        assert packing.get(key) == now, f"{key} in the packing config: expected={now}, got={packing.get(key)}"


def test_configuration_resolves_with_the_defaults():
    """2 nodes x 16 dies: FSDP 32, EP 16, experts FSDP-sharded over the nodes; one sample per rank per step."""
    lines = _resolve()
    assert lines[0] == "32 2 16 1 32"
    assert lines[1].startswith("True /home/pl/runs/qwen3_vl_30b_perf/a3_32dev_hetero/hetero 3")
    assert lines[2] == "True False full"
    assert "build_vlm_data_transform" in lines[4], lines[4]
    # padding none is what makes a rank's work depend on its sample; the default would hide it.
    assert lines[4].endswith(" none keep"), lines[4]
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


def test_the_data_transform_plans_switch_their_policies():
    """The padding and text-only plans reach the sample transform through the command line."""
    padded = _resolve("--dataset.data_transform.padding=max_length",
                      "--dataset.data_transform.max_seq_len=16384")
    assert padded[4].endswith(" max_length keep"), padded[4]

    placeholder = _resolve("--dataset.data_transform.text_only=placeholder")
    assert placeholder[4].endswith(" none placeholder"), placeholder[4]


def test_the_profile_plan_names_the_ranks_it_profiles():
    """A trace per rank is gigabytes, so the profile plan profiles four ranks, two per node, and keeps no raw data."""
    default = _resolve()
    assert default[7] == "False [] 6 8 True", "off by default, no rank named, the raw collection dropped"
    lines = _resolve("--profiling.enabled=true", "--profiling.ranks=[6,13,22,29]",
                     "--profiling.start_step=6", "--profiling.end_step=8")
    assert lines[7] == "True [6, 13, 22, 29] 6 8 True"
