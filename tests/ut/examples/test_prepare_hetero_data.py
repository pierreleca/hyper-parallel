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
"""Data builder for the heterogeneity experiments: distributions, image sizes, samples, arrangements."""

import importlib.util
import io
import math
import pathlib
import random
import statistics
import sys
from types import ModuleType

import pytest
from PIL import Image

_DIR = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf"


def _load(name: str) -> ModuleType:
    """Import an example script by path; the scripts import their siblings as top-level modules."""
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("prepare_cauldron_data")  # the sibling that prepare_hetero_data imports by name
data = _load("prepare_hetero_data")


class _Tokenizer:
    """One token per whitespace-separated word; decode joins the words back."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        """Tokenize by words."""
        del add_special_tokens
        return {"input_ids": text.split()}

    @staticmethod
    def decode(ids: list, skip_special_tokens: bool = True) -> str:
        """Join the words back."""
        del skip_special_tokens
        return " ".join(ids)


def _png(width: int, height: int) -> bytes:
    """Encode a plain image of this size."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _words(count: int, tag: str) -> str:
    """Return ``count`` distinct words."""
    return " ".join(f"{tag}{i}" for i in range(count))


def _pool() -> list:
    """Four conversations of two subsets, with images of different shapes."""
    conversations = []
    for row, (subset, size) in enumerate((("vsr", (640, 480)), ("vsr", (500, 700)),
                                          ("finqa", (800, 600)), ("finqa", (1200, 400)))):
        conversations.append(data.Conversation(
            subset, row, [_png(*size)],
            [(_words(12, f"q{row}_"), _words(40, f"a{row}_")), (_words(8, f"r{row}_"), _words(90, f"b{row}_"))],
        ))
    return conversations


def test_lognormal_parameters_give_the_requested_mean():
    """The log-normal with these parameters has the mean and spread asked for."""
    mu, sigma = data.lognormal_parameters(8192, 0.8)
    assert math.exp(mu + sigma ** 2 / 2) == pytest.approx(8192, rel=1e-9)
    assert math.sqrt(math.expm1(sigma ** 2)) == pytest.approx(0.8, rel=1e-9)


def test_targets_keep_the_mean_and_the_bounds():
    """Clipped draws are scaled back to the requested means, and no sample leaves its bounds."""
    targets = data.draw_targets(2000, mean_len=8192, len_cv=0.8, mean_visual=2048, visual_cv=1.0, corr=0.0,
                                min_len=1024, max_len=16384, min_visual=64, rng=random.Random(0))
    lengths = [length for length, _ in targets]
    visuals = [visual for _, visual in targets]
    assert statistics.fmean(lengths) == pytest.approx(8192, rel=0.05)
    assert statistics.fmean(visuals) == pytest.approx(2048, rel=0.05)
    assert min(lengths) >= 1024 and max(lengths) <= 16384 and min(visuals) >= 64
    assert all(length >= visual + data.MIN_TEXT_TOKENS for length, visual in targets)
    assert statistics.pstdev(lengths) / statistics.fmean(lengths) > 0.4


def test_fixed_targets_do_not_vary():
    """A coefficient of variation of 0 on both axes gives one shape for every sample."""
    targets = data.draw_targets(50, mean_len=8192, len_cv=0.0, mean_visual=2048, visual_cv=0.0, corr=0.0,
                                min_len=1024, max_len=16384, min_visual=64, rng=random.Random(0))
    assert set(targets) == {(8192, 2048)}


def test_constant_visual_survives_short_lengths():
    """With constant visual tokens a short draw lengthens the sample; the images keep their size."""
    targets = data.draw_targets(500, mean_len=3000, len_cv=1.2, mean_visual=2048, visual_cv=0.0, corr=0.0,
                                min_len=1024, max_len=16384, min_visual=64, rng=random.Random(1))
    assert {visual for _, visual in targets} == {2048}
    assert all(length >= 2048 + data.MIN_TEXT_TOKENS for length, _ in targets)


def test_correlation_ties_length_and_visual_tokens():
    """A positive copula correlation makes long samples the image-heavy ones."""
    targets = data.draw_targets(3000, mean_len=8192, len_cv=1.0, mean_visual=2048, visual_cv=1.0, corr=0.9,
                                min_len=1024, max_len=16384, min_visual=64, rng=random.Random(2))
    lengths = [length for length, _ in targets]
    visuals = [visual for _, visual in targets]
    mean_l, mean_v = statistics.fmean(lengths), statistics.fmean(visuals)
    covariance = statistics.fmean((length - mean_l) * (visual - mean_v) for length, visual in targets)
    assert covariance / (statistics.pstdev(lengths) * statistics.pstdev(visuals)) > 0.5


def test_split_visual_adds_up_within_bounds():
    """The images share the visual tokens exactly when the bounds leave room."""
    rng = random.Random(3)
    for total, count in ((2048, 3), (500, 5), (6000, 4)):
        parts = data.split_visual(total, count, 64, 2048, rng)
        assert len(parts) == count and sum(parts) == total
        assert all(64 <= part <= 2048 for part in parts)


def test_image_count_respects_the_token_bounds():
    """Enough images to stay under the per-image maximum, never more than the cap."""
    rng = random.Random(4)
    counts = {data.plan_image_count(5000, 64, 2048, 16, rng) for _ in range(200)}
    assert min(counts) >= 3 and max(counts) <= 16
    assert data.plan_image_count(10 ** 6, 64, 2048, 16, rng) == 16


@pytest.mark.parametrize("tokens,aspect", [(64, 1.0), (1000, 0.75), (2048, 4.0), (300, 0.1)])
def test_grid_is_inside_the_processor_bounds(tokens, aspect):
    """A grid has at least 64 tokens, sides in 32-pixel multiples, and the asked aspect within the clamp."""
    rows, columns = data.grid_for_tokens(tokens, aspect)
    assert rows * columns >= data.MIN_IMAGE_TOKENS
    assert rows <= data.MAX_SIDE_TOKENS and columns <= data.MAX_SIDE_TOKENS
    assert rows * columns == pytest.approx(max(tokens, data.MIN_IMAGE_TOKENS), rel=0.35)


def test_resized_image_is_a_whole_grid_of_tokens():
    """The resized image has 32-pixel sides, so the processor leaves it alone."""
    image = data.resize_to_grid(_png(640, 480), 1000)
    assert image.width % 32 == 0 and image.height % 32 == 0
    assert data.visual_tokens(image) == pytest.approx(1000, rel=0.1)
    assert image.width * image.height >= 65536


def test_balanced_order_equalizes_the_ranks_of_a_step():
    """Same samples, but the steps hold alike ones: the slowest rank costs about the mean."""
    rng = random.Random(5)
    costs = [rng.lognormvariate(8.0, 0.8) for _ in range(640)]
    random_order = data.arrange(costs, 32, "random", random.Random(6))
    balanced = data.arrange(costs, 32, "balanced", random.Random(6))
    assert sorted(balanced) == sorted(random_order) == list(range(640))
    spread = data.step_imbalance(costs, random_order, 32)
    flat = data.step_imbalance(costs, balanced, 32)
    # The step of the 32 largest samples still spreads (the tail of the draw), the others hardly do.
    assert spread["max_over_mean"] > 1.5
    assert flat["max_over_mean"] < 1.25
    assert flat["idle_share"] < spread["idle_share"] / 2


def test_balanced_order_keeps_a_remainder():
    """Samples that do not fill a step still come out, at the end."""
    assert sorted(data.arrange([float(i) for i in range(70)], 32, "balanced", random.Random(0))) == list(range(70))


def test_synthetic_sample_hits_its_targets(tmp_path):
    """The record has the images asked for, their tokens add up, and the text fills the rest."""
    composer = data.Composer(tmp_path, _Tokenizer(), _pool(), random.Random(7), max_len=4096, message_overhead=5)
    record, stats = composer.synthetic(length=2000, visual=600)
    assert record["messages"][0]["content"].startswith("<image>" * len(record["images"]))
    assert record["messages"][-1]["role"] == "assistant"
    assert stats["images"] == len(record["images"]) >= 1
    assert stats["visual_tokens"] == pytest.approx(600, rel=0.25)
    assert stats["tokens"] == pytest.approx(2000, rel=0.1)
    assert stats["tokens"] == stats["text_tokens"] + stats["visual_tokens"] + 2 * stats["images"]
    for name in record["images"]:
        assert (tmp_path / name).is_file()


def test_synthetic_samples_follow_different_targets(tmp_path):
    """Two targets give two different sizes: the builder does not collapse to one shape."""
    composer = data.Composer(tmp_path, _Tokenizer(), _pool(), random.Random(8), max_len=8192, message_overhead=5)
    small = composer.synthetic(length=1200, visual=200)[1]
    large = composer.synthetic(length=6000, visual=3000)[1]
    assert large["tokens"] > 3 * small["tokens"]
    assert large["visual_tokens"] > 5 * small["visual_tokens"]


def test_natural_sample_keeps_one_subset_and_an_image(tmp_path):
    """A natural sample is made of real conversations of a single subset, and never lacks an image."""
    composer = data.Composer(tmp_path, _Tokenizer(), _pool(), random.Random(9), max_len=4096, message_overhead=5)
    for _ in range(5):
        record, stats = composer.natural(length=600)
        assert record["subset"] in {"vsr", "finqa"}
        assert stats["images"] == len(record["images"]) >= 1
        assert stats["tokens"] <= 4096


def test_scenarios_set_the_spread_they_isolate():
    """text varies only the length, vision only the visual tokens, fixed neither."""
    assert (data.SCENARIOS["text"].len_cv, data.SCENARIOS["text"].visual_cv) == (None, 0.0)
    assert (data.SCENARIOS["vision"].len_cv, data.SCENARIOS["vision"].visual_cv) == (0.0, None)
    assert (data.SCENARIOS["fixed"].len_cv, data.SCENARIOS["fixed"].visual_cv) == (0.0, 0.0)
    assert data.SCENARIOS["natural"].natural
