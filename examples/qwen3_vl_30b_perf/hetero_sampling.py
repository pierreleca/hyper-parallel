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
"""The arithmetic of the heterogeneity datasets, with nothing but the standard library.

How many tokens and visual tokens each sample gets in each scenario, how the visual tokens split over images, how
big an image of so many tokens is, how samples are ordered into steps, and how to describe the result. The data
builder (``prepare_hetero_data.py``) and the synthetic-record generator (``synthetic_hetero_records.py``) share it.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Optional, Sequence

# The processor's pixel bounds (size.shortest_edge / longest_edge of the checkpoint), in merged tokens of
# 32 x 32 pixels: no image below 64 tokens, none above 16384.
MIN_IMAGE_TOKENS = 64
MAX_SIDE_TOKENS = 128
# Text kept for the last assistant turn, the only one the loss reads, whatever the images take.
MIN_TEXT_TOKENS = 256
MIN_ANSWER_TOKENS = 16
ARRANGEMENTS = ("random", "balanced")


@dataclass(frozen=True)
class Scenario:
    """How the total tokens and the visual tokens of the samples are drawn."""

    len_cv: Optional[float]
    visual_cv: Optional[float]
    corr: float = 0.0
    natural: bool = False


# None: take --len-cv / --visual-cv.
SCENARIOS: dict[str, Scenario] = {
    "fixed": Scenario(0.0, 0.0),
    "text": Scenario(None, 0.0),
    "vision": Scenario(0.0, None),
    "both": Scenario(None, None),
    "longtail": Scenario(1.5, 1.5, corr=0.7),
    "natural": Scenario(None, None, natural=True),
}


# -- distributions ----------------------------------------------------------------------------------------

def lognormal_parameters(mean: float, cv: float) -> tuple[float, float]:
    """Return (mu, sigma) of the log-normal distribution with this mean and coefficient of variation."""
    sigma_squared = math.log1p(cv * cv)
    return math.log(mean) - sigma_squared / 2, math.sqrt(sigma_squared)


def rescale_to_mean(values: list[float], mean: float, low: float, high: float, tolerance: float = 0.002) -> list[float]:
    """Scale values until their mean, after clipping to [low, high], is ``mean`` (as near as the bounds allow)."""
    scale = 1.0
    clipped = [min(max(value, low), high) for value in values]
    for _ in range(60):
        current = statistics.fmean(clipped)
        if abs(current - mean) <= tolerance * mean:
            break
        scale *= mean / current
        clipped = [min(max(value * scale, low), high) for value in values]
    return clipped


def draw_targets(
        count: int,
        *,
        mean_len: int,
        len_cv: float,
        mean_visual: int,
        visual_cv: float,
        corr: float,
        min_len: int,
        max_len: int,
        min_visual: int,
        rng: random.Random,
) -> list[tuple[int, int]]:
    """Draw (total tokens, visual tokens) targets for ``count`` samples.

    Log-normal margins joined by a Gaussian copula of correlation ``corr``;
    a coefficient of variation of 0 gives a constant. The visual tokens leave
    ``MIN_TEXT_TOKENS`` of the total for text.
    """
    normals = []
    for _ in range(count):
        first = rng.gauss(0.0, 1.0)
        normals.append((first, corr * first + math.sqrt(max(1.0 - corr * corr, 0.0)) * rng.gauss(0.0, 1.0)))

    def margin(index: int, mean: float, cv: float, low: float, high: float) -> list[float]:
        """Draw one margin: constant for a spread of 0, else log-normal, scaled back to its mean after clipping."""
        if cv <= 0:
            return [min(max(mean, low), high)] * count
        mu, sigma = lognormal_parameters(mean, cv)
        return rescale_to_mean([math.exp(mu + sigma * pair[index]) for pair in normals], mean, low, high)

    lengths = margin(0, mean_len, len_cv, min_len, max_len)
    visuals = margin(1, mean_visual, visual_cv, min_visual, max_len - MIN_TEXT_TOKENS)
    # A sample is never shorter than its images plus the text the loss needs; the length gives way, not the images.
    return [
        (min(max(int(round(length)), int(round(visual)) + MIN_TEXT_TOKENS), max_len), int(round(visual)))
        for length, visual in zip(lengths, visuals)
    ]


def split_visual(total: int, count: int, low: int, high: int, rng: random.Random) -> list[int]:
    """Split ``total`` visual tokens into ``count`` images of ``low`` to ``high`` tokens each, as far as that allows."""
    if count <= 1:
        return [min(max(total, low), high)]
    weights = [rng.gammavariate(2.0, 1.0) for _ in range(count)]
    scale = total / sum(weights)
    parts = [min(max(int(round(weight * scale)), low), high) for weight in weights]
    remainder = total - sum(parts)
    for index in rng.sample(range(count), count):
        if remainder == 0:
            break
        room = (high - parts[index]) if remainder > 0 else (low - parts[index])
        step = min(remainder, room) if remainder > 0 else max(remainder, room)
        parts[index] += step
        remainder -= step
    return parts


def plan_image_count(total: int, low: int, high: int, max_images: int, rng: random.Random) -> int:
    """Pick how many images share ``total`` visual tokens, each within [low, high]."""
    fewest = max(1, math.ceil(total / high))
    if fewest >= max_images:
        return max_images
    most = max(fewest, min(max_images, total // low))
    candidates = list(range(fewest, most + 1))
    return rng.choices(candidates, weights=[1.0 / count for count in candidates])[0]


def grid_for_tokens(tokens: int, aspect: float, min_tokens: int = MIN_IMAGE_TOKENS) -> tuple[int, int]:
    """Return the (rows, columns) of merged tokens of an image of about ``tokens`` tokens and this height/width."""
    tokens = max(tokens, min_tokens)
    aspect = min(max(aspect, 0.25), 4.0)
    rows = min(max(int(round(math.sqrt(tokens * aspect))), 1), MAX_SIDE_TOKENS)
    columns = min(max(int(round(tokens / rows)), 1), MAX_SIDE_TOKENS)
    while rows * columns < min_tokens and columns < MAX_SIDE_TOKENS:
        columns += 1
    return rows, columns


# -- arrangements -----------------------------------------------------------------------------------------

def sample_cost(stats: dict[str, Any], cost_visual: float) -> float:
    """Estimated cost of a sample: its positions, plus ``cost_visual`` per visual token for the vision tower."""
    return stats["tokens"] + cost_visual * stats["visual_tokens"]


def arrange(costs: Sequence[float], group: int, mode: str, rng: random.Random) -> list[int]:
    """Return the order in which to write the samples, given the cost of each and the ranks per step."""
    count = len(costs)
    if mode == "random":
        return list(range(count))
    if mode != "balanced":
        raise ValueError(f"unknown arrangement {mode!r}")
    ranked = sorted(range(count), key=lambda index: costs[index])
    whole = count - count % group
    groups = [ranked[start:start + group] for start in range(0, whole, group)]
    rng.shuffle(groups)
    return [index for members in groups for index in members] + ranked[whole:]


def step_imbalance(costs: Sequence[float], order: Sequence[int], group: int) -> dict[str, float]:
    """Return the mean of max / mean cost over the steps of an order, and the work the slowest rank adds."""
    factors = []
    for start in range(0, len(order) - group + 1, group):
        step = [costs[index] for index in order[start:start + group]]
        factors.append(max(step) / statistics.fmean(step))
    mean_factor = statistics.fmean(factors) if factors else 1.0
    return {"steps": len(factors), "max_over_mean": round(mean_factor, 4),
            "idle_share": round(1.0 - 1.0 / mean_factor, 4)}


def describe(values: Sequence[float]) -> dict[str, float]:
    """Return the mean, spread and quantiles of a list of numbers."""
    ordered = sorted(values)
    mean = statistics.fmean(ordered)

    def quantile(share: float) -> float:
        """Return the value at this share of the sorted list."""
        return ordered[min(int(share * len(ordered)), len(ordered) - 1)]

    return {"mean": round(mean, 2), "cv": round(statistics.pstdev(ordered) / mean, 4) if mean else 0.0,
            "min": ordered[0], "p50": quantile(0.5), "p90": quantile(0.9), "max": ordered[-1]}
