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
"""A/B comparison: the speedup and its interval, pairing by samples, work normalisation, numerics, logs."""

import importlib.util
import json
import pathlib
import random
import sys
from types import ModuleType

import pytest

_DIR = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf"


def _load(name: str, path: pathlib.Path) -> ModuleType:
    """Import a script by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _load("compare_runs", _DIR / "compare_runs.py")
helper = _load("analyze_hetero_helper_tests", pathlib.Path(__file__).parent / "test_analyze_hetero.py")


def _write(directory: pathlib.Path, *, scale: float = 1.0, ranks: int = 4, steps: int = 12, seed: int = 0,
           loss_shift: float = 0.0, hooks: bool = False, token_scale: float = 1.0, noise: float = 0.0,
           noise_seed: int = 99) -> pathlib.Path:
    """Write light-mode records: per step, every rank's time, workload and fingerprint (and loss on all ranks)."""
    noisy = random.Random(noise_seed)
    directory.mkdir(parents=True, exist_ok=True)
    for rank in range(ranks):
        lines = [{"kind": "header", "rank": rank, "world_size": ranks, "hooks": hooks, "modules": [],
                  "time_source": "device"}]
        local = random.Random(seed)
        for step in range(1, steps + 1):
            tokens = local.randint(2000, 12000)
            visual = local.randint(200, 3000)
            ident = local.randint(1, 10 ** 9)
            cost = (tokens + 0.5 * visual) * 0.4 * scale * (1.0 + noisy.uniform(-noise, noise))
            lines.append({
                "kind": "step", "step": step, "wall_ms": cost, "device_ms": cost,
                "inter_step_ms": 20.0 if step > 1 else None,
                "micro_batches": [{"batch_size": 1, "tokens": int(tokens * token_scale),
                                   "real_tokens": int(tokens * token_scale), "visual_tokens": visual,
                                   "fingerprint": ident, "images": 2, "patches": 4 * visual,
                                   "vision_attn_pairs": 1000}],
                "marks": [], "routing": [], "peak_allocated": 2 ** 30 * (30 + step % 3),
                "loss": 2.0 - 0.01 * step + loss_shift, "grad_norm": 1.0,
            })
        (directory / f"rank{rank:03d}.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return directory


def test_a_20_percent_faster_candidate_on_the_same_samples_meets_the_target(tmp_path):
    """Times scaled by 0.78 on identical samples: 28% more work per second, the paired interval clears 20%."""
    base = _write(tmp_path / "base")
    cand = _write(tmp_path / "cand", scale=0.78)
    lines, summary = compare.compare(compare.Arm([str(base)], 1), compare.Arm([str(cand)], 1), draws=400)
    assert summary["paired"]["steps"] == 11
    assert summary["paired"]["speedup"] == pytest.approx(0.2822, abs=0.03)
    assert summary["verdict"] == "MET"
    assert summary["numerics"]["loss"]["ok"] and summary["numerics"]["grad_norm"]["ok"]
    assert any("verdict against +20%" in line for line in lines)


def test_a_5_percent_gain_does_not_meet_a_20_percent_target(tmp_path):
    """A small, real gain is reported as such, and the verdict says it falls short."""
    base = _write(tmp_path / "base")
    cand = _write(tmp_path / "cand", scale=0.95)
    _, summary = compare.compare(compare.Arm([str(base)], 1), compare.Arm([str(cand)], 1), draws=400)
    assert 0.03 < summary["paired"]["speedup"] < 0.09
    assert summary["verdict"] == "NOT MET"


def test_two_identical_runs_are_the_noise_floor(tmp_path):
    """The same configuration twice (timing jitter only) claims about nothing; the interval contains zero."""
    first = _write(tmp_path / "a", noise=0.03, noise_seed=1)
    second = _write(tmp_path / "b", noise=0.03, noise_seed=2)
    _, summary = compare.compare(compare.Arm([str(first)], 1), compare.Arm([str(second)], 1), draws=400)
    assert abs(summary["paired"]["speedup"]) < 0.04
    assert summary["paired"]["low"] < 0 < summary["paired"]["high"]
    assert summary["verdict"] == "NOT MET"


def test_repeats_of_the_baseline_give_a_noise_estimate(tmp_path):
    """Pooled baseline runs report how far apart they ran."""
    runs = [str(_write(tmp_path / "base0")), str(_write(tmp_path / "base1", scale=1.04))]
    cand = _write(tmp_path / "cand", scale=0.8)
    _, summary = compare.compare(compare.Arm(runs, 1), compare.Arm([str(cand)], 1), draws=200)
    assert summary["noise"] == pytest.approx(0.04, abs=0.03)


def test_other_samples_are_compared_on_work_not_on_time(tmp_path):
    """A candidate handed half the tokens and finishing in half the time is not faster per unit of work."""
    base = _write(tmp_path / "base")
    cand = _write(tmp_path / "cand", scale=0.5, token_scale=0.5, seed=0)
    _, summary = compare.compare(compare.Arm([str(base)], 1), compare.Arm([str(cand)], 1), draws=200)
    assert summary["e2e_ms"]["speedup"] > 0.5          # the clock says twice as fast...
    assert summary["tokens"]["speedup"] == pytest.approx(0.0, abs=0.12)    # ...the tokens say it is not


def test_unpaired_arms_fall_back_to_work_per_second(tmp_path):
    """Different samples in the steps: no pairing, the verdict rests on work per second."""
    base = _write(tmp_path / "base", seed=0)
    cand = _write(tmp_path / "cand", seed=5, scale=0.7)
    lines, summary = compare.compare(compare.Arm([str(base)], 1), compare.Arm([str(cand)], 1), draws=200)
    assert "paired" not in summary
    assert any("fewer than 5" in line for line in lines)


def test_a_candidate_that_moved_the_loss_is_flagged(tmp_path):
    """Faster but with another loss on the same samples: the numerics line says it differs."""
    base = _write(tmp_path / "base")
    cand = _write(tmp_path / "cand", scale=0.7, loss_shift=0.5)
    lines, summary = compare.compare(compare.Arm([str(base)], 1), compare.Arm([str(cand)], 1), draws=200)
    assert not summary["numerics"]["loss"]["ok"]
    assert any("DIFFERS" in line for line in lines)


def test_a_trainer_log_stands_in_for_an_arm_without_the_recorder(tmp_path):
    """Step time, loss and gradient norm are read from the metric lines; there are no token counts."""
    log = tmp_path / "log.txt"
    log.write_text("\n".join(
        f"[node0] step={step} epoch=0 data/step_samples=32 memory/device_max_allocated_gb=40.5 "
        f"performance/step_time={20.0 - 0.1 * step} training/grad_norm=1.0 training/total_loss={2.0 - 0.01 * step}"
        for step in range(1, 11)) + "\nnoise line\n")
    cand = _write(tmp_path / "cand", scale=0.8)
    arm = compare.Arm([str(log)], 2)
    assert len(arm.steps) == 8 and not arm.has_tokens
    lines, summary = compare.compare(arm, compare.Arm([str(cand)], 1), draws=100)
    assert "tokens" not in summary
    assert any("only the step time is compared" in line for line in lines)


def test_components_are_compared_when_both_arms_hooked_the_modules(tmp_path):
    """With module hooks on in both arms, the cost per 1k tokens of each component and the imbalance are listed."""
    helper._write_run(tmp_path / "base" / "hetero", ranks=4, steps=8, seed=1)       # pylint: disable=protected-access
    helper._write_run(tmp_path / "cand" / "hetero", ranks=4, steps=8, seed=1)       # pylint: disable=protected-access
    lines, summary = compare.compare(compare.Arm([str(tmp_path / "base")], 1), compare.Arm([str(tmp_path / "cand")], 1),
                                     skip=1, draws=100)
    assert "components" in summary
    assert any("where the time moved" in line for line in lines)
    assert summary["components"]["baseline"]["per_1k"]["text_layer"] > 0
