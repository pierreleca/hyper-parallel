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
"""Build Qwen3-VL samples whose length and image content differ in a controlled way.

``prepare_cauldron_data.py`` gives every sample the same length, which is right
for measuring a kernel and wrong for measuring imbalance: with identical shapes
the data-parallel ranks do identical work. This script builds the opposite, from
the same four cauldron subsets (real images, real text): samples whose total
tokens ``L`` and visual tokens ``V`` follow a distribution you choose, so the
effect of each kind of heterogeneity on the step can be measured on its own,
at the same mean tokens.

Scenarios (``--scenario``):

- ``fixed``     L and V constant. The reference: no difference in shape between ranks.
- ``text``      L varies, V constant. The text decoder's work differs, the vision tower's does not.
- ``vision``    V varies, L constant (the text shrinks as the images grow). Only the vision
                tower's share of the work differs; the decoder sees the same length.
- ``both``      L and V vary independently.
- ``longtail``  L and V vary a lot and together (long documents have many images): the
                few biggest samples of a step set its time.
- ``natural``   one run of real conversations of a single subset per sample, at the images'
                natural sizes: the mix the data has, not a designed one.

Every scenario keeps the mean of L and V (``--mean-len``, ``--mean-visual``), so scenarios
compare at equal total work; their spread is ``--len-cv`` and ``--visual-cv`` (standard deviation over
mean) unless the scenario sets it. Every sample holds at least one image: with the vision tower
sharded over all ranks, a rank without an image would skip its all-gather and stall the others.

The order of the samples decides which ranks work on which in each step (the sampler gives
step ``s`` the samples ``s * dp_size`` to ``(s + 1) * dp_size - 1``, one per rank). ``--arrange``
writes the same samples in several orders, each as its own JSON file next to the first:

- ``random``    ``vlm_conversations.json``: the draw order, an unbiased mix in every step;
- ``balanced``  ``vlm_conversations.balanced.json``: samples sorted by estimated cost and cut into
                steps, so the ranks of a step get alike samples while steps differ. The total work is
                the same as ``random``; whatever time it saves is what the imbalance between ranks cost.

Images are resized to 32-pixel multiples inside the Qwen3-VL processor's pixel bounds, so the
processor does not resize them again and the visual tokens of a sample are known in advance.

    python examples/qwen3_vl_30b_perf/prepare_hetero_data.py \\
      --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_both_n640 \\
      --scenario both --num-samples 640 --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct \\
      --download-dir /home/pl/data/the_cauldron --offline

The result is deterministic in ``--seed``, so every node that builds it gets the same files.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

# Run as a script, Python puts this directory first on the import path.
from prepare_cauldron_data import (
    _IMAGE_OVERHEAD, _SUBSETS, _TOKEN_EDGE, Conversation, download, fit_image, read_conversations, visual_tokens,
)

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


def resize_to_grid(payload: bytes, tokens: int) -> Image.Image:
    """Decode an image and resize it to about ``tokens`` merged tokens, keeping its aspect ratio."""
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    rows, columns = grid_for_tokens(tokens, image.height / image.width)
    return image.resize((columns * _TOKEN_EDGE, rows * _TOKEN_EDGE), Image.Resampling.BICUBIC)


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


# -- samples ----------------------------------------------------------------------------------------------

class Composer:
    """Builds records from the conversation pool, writing their images to ``staging/images``."""

    def __init__(
            self,
            staging: Path,
            tokenizer: Any,
            conversations: list[Conversation],
            rng: random.Random,
            *,
            max_len: int,
            min_image_tokens: int = MIN_IMAGE_TOKENS,
            max_image_tokens: int = 2048,
            max_images: int = 16,
            message_overhead: Optional[int] = None,
    ) -> None:
        """Bind the output directory, the tokenizer and the pool.

        Args:
            staging: Directory the dataset is written to.
            tokenizer: Anything callable as ``tokenizer(text, add_special_tokens=False)["input_ids"]``
                with a ``decode`` method.
            conversations: The pool of real conversations, each with images.
            rng: The source of every random choice.
            max_len: Longest sample, in tokens.
            min_image_tokens: Fewest visual tokens of an image.
            max_image_tokens: Most visual tokens of an image.
            max_images: Most images of a sample.
            message_overhead: Template tokens around one message; measured from the tokenizer when omitted.
        """
        self.staging = staging
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.rng = rng
        self.max_len = max_len
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.max_images = max_images
        self.by_subset: dict[str, list[Conversation]] = {}
        self.images: list[bytes] = []
        for conversation in conversations:
            self.by_subset.setdefault(conversation.subset, []).append(conversation)
            self.images.extend(conversation.images)
        self.message_overhead = (
            message_overhead if message_overhead is not None else self._tokens("<|im_start|>user\n<|im_end|>\n")
        )
        self.counter = 0
        (staging / "images").mkdir(parents=True, exist_ok=True)

    def _tokens(self, text: str) -> int:
        """Count the tokens of a text, without special tokens."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _clip(self, text: str, limit: int) -> str:
        """Cut a text to at most ``limit`` tokens."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= limit else self.tokenizer.decode(ids[:limit], skip_special_tokens=True)

    def _pair(self, pool: Optional[list[Conversation]] = None) -> tuple[str, str]:
        """Pick one user-assistant turn pair of a random conversation."""
        conversation = self.rng.choice(pool or self.conversations)
        return self.rng.choice(conversation.turns)

    def _save(self, image: Image.Image, index: int) -> str:
        """Write the ``index``-th image of the current sample and return its path relative to the dataset."""
        name = f"images/s{self.counter:06d}_{index:02d}.jpg"
        image.save(self.staging / name, quality=90)
        return name

    def _text(self, budget: int, first_images: int) -> tuple[list[dict[str, str]], int]:
        """Fill ``budget`` tokens with turn pairs; the first user message carries the image placeholders.

        Whole pairs are added while they fit; the pair that does not fit is cut to the room left, and is the
        last one, so the sample ends on an assistant turn of real text.
        """
        messages: list[dict[str, str]] = []
        used = 0
        prefix = "<image>" * first_images
        overhead = 2 * self.message_overhead
        while True:
            user, answer = self._pair()
            room = budget - used - overhead
            user_tokens, answer_tokens = self._tokens(user), self._tokens(answer)
            last = False
            if user_tokens + answer_tokens > room:
                if messages and room < 2 * MIN_ANSWER_TOKENS:
                    break
                user = self._clip(user, max(room // 2, 1))
                user_tokens = self._tokens(user)
                answer = self._clip(answer, max(room - user_tokens, 1))
                answer_tokens = self._tokens(answer)
                last = True
            messages.append({"role": "user", "content": (prefix if not messages else "") + user})
            messages.append({"role": "assistant", "content": answer})
            used += user_tokens + answer_tokens + overhead
            if last or budget - used < overhead + 2 * MIN_ANSWER_TOKENS:
                break
        return messages, used

    def synthetic(self, length: int, visual: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a sample of about ``length`` tokens, ``visual`` of them visual, from images resized to fit."""
        count = plan_image_count(visual, self.min_image_tokens, self.max_image_tokens, self.max_images, self.rng)
        parts = split_visual(visual, count, self.min_image_tokens, self.max_image_tokens, self.rng)
        names, actual = [], 0
        for tokens in parts:
            image = resize_to_grid(self.rng.choice(self.images), tokens)
            names.append(self._save(image, len(names)))
            actual += visual_tokens(image)
        text_budget = length - actual - _IMAGE_OVERHEAD * len(names)
        messages, used = self._text(max(text_budget, MIN_TEXT_TOKENS), len(names))
        record = {"messages": messages, "images": names}
        stats = {"tokens": used + actual + _IMAGE_OVERHEAD * len(names), "visual_tokens": actual,
                 "images": len(names), "text_tokens": used, "turns": len(messages) // 2}
        self.counter += 1
        return record, stats

    def natural(self, length: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a sample of about ``length`` tokens from consecutive conversations of one subset."""
        subset = self.rng.choice(sorted(self.by_subset))
        pool = self.by_subset[subset]
        messages: list[dict[str, str]] = []
        names: list[str] = []
        used = visual = 0
        ceiling = min(self.max_len, int(length * 1.25) + 64)
        for _ in range(64):
            conversation = self.rng.choice(pool)
            fitted = [fit_image(payload) for payload in conversation.images[: self.max_images]]
            image_tokens = sum(visual_tokens(image) + _IMAGE_OVERHEAD for image in fitted)
            turns = list(conversation.turns)
            text = sum(self._tokens(user) + self._tokens(answer) + 2 * self.message_overhead for user, answer in turns)
            if used + image_tokens + text > ceiling and messages:
                if used >= length // 2:
                    break
                continue
            if len(names) + len(fitted) > self.max_images:
                break
            saved = [self._save(image, len(names) + offset) for offset, image in enumerate(fitted)]
            for turn, (user, answer) in enumerate(turns):
                messages.append({"role": "user", "content": ("<image>" * len(saved) if turn == 0 else "") + user})
                messages.append({"role": "assistant", "content": answer})
            names.extend(saved)
            visual += sum(visual_tokens(image) for image in fitted)
            used += image_tokens + text
            if used >= length:
                break
        if not names:
            raise RuntimeError("a natural sample came out without an image; lower --min-len or check the pool")
        record = {"messages": messages, "images": names, "subset": subset}
        stats = {"tokens": used, "visual_tokens": visual, "images": len(names), "text_tokens": used - visual,
                 "turns": len(messages) // 2, "subset": subset}
        self.counter += 1
        return record, stats


# -- verification -----------------------------------------------------------------------------------------

def measure(json_path: Path, processor_path: str, max_seq_len: int, indices: Sequence[int]) -> list[dict[str, Any]]:
    """Encode records with the training transform, and report what the model would receive."""
    # Heavy optional imports, only needed for the verification.
    from hyper_parallel.data.vlm import build_processor  # pylint: disable=C0415
    from hyper_parallel.data.vlm.dataset import VLMDataset  # pylint: disable=C0415
    from variable_length_transform import build_variable_length_vlm_transform  # pylint: disable=C0415

    transform = build_variable_length_vlm_transform(processor=build_processor(processor_path), max_seq_len=max_seq_len)
    dataset = VLMDataset(str(json_path))
    reports = []
    for index in indices:
        record = dataset[index]
        sample = transform(record)
        grid = sample["image_grid_thw"]
        reports.append({
            "record": index,
            "tokens": int(sample["attention_mask"].sum()),
            "images": int(grid.shape[0]),
            "images_in_record": len(record["images"]),
            "visual_tokens": int((sample["mm_token_type_ids"] == 1).sum()),
            "patches": int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum()) if grid.numel() else 0,
            "label_tokens": int((sample["labels"] >= 0).sum()),
        })
    return reports


def verify(json_path: Path, processor_path: str, max_seq_len: int, samples: list[dict[str, Any]],
           count: int) -> dict[str, Any]:
    """Compare the estimated tokens of the first records with the processor's, and require every image kept."""
    indices = list(range(min(count, len(samples))))
    reports = measure(json_path, processor_path, max_seq_len, indices)
    errors, mismatches = [], []
    for report in reports:
        estimate = samples[report["record"]]
        if report["images"] != report["images_in_record"] or report["images"] == 0:
            raise RuntimeError(f"record {report['record']} lost an image or has none: {report}")
        errors.append((report["tokens"] - estimate["tokens"]) / max(report["tokens"], 1))
        if report["visual_tokens"] != estimate["visual_tokens"]:
            mismatches.append((report["record"], report["visual_tokens"], estimate["visual_tokens"]))
    summary = {"records": len(reports), "token_error_mean": round(statistics.fmean(errors), 4) if errors else 0.0,
               "token_error_max": round(max((abs(error) for error in errors), default=0.0), 4),
               "visual_token_mismatches": mismatches}
    print(f"verified {summary['records']} records against the processor: every image kept, token estimate off by "
          f"{summary['token_error_mean']:+.2%} on average (worst {summary['token_error_max']:.2%})")
    if mismatches:
        print(f"warning: the processor counts other visual tokens than the builder in {len(mismatches)} record(s) "
              f"(record, processor, builder): {mismatches[:5]}; the samples are valid, "
              "the statistics use the builder's")
    return summary


# -- build ------------------------------------------------------------------------------------------------

def build_samples(args: argparse.Namespace, composer: Composer, rng: random.Random) -> tuple[list, list]:
    """Draw the targets of the scenario and compose every sample."""
    scenario = SCENARIOS[args.scenario]
    len_cv = args.len_cv if scenario.len_cv is None else scenario.len_cv
    visual_cv = args.visual_cv if scenario.visual_cv is None else scenario.visual_cv
    targets = draw_targets(
        args.num_samples, mean_len=args.mean_len, len_cv=len_cv, mean_visual=args.mean_visual, visual_cv=visual_cv,
        corr=scenario.corr if args.corr is None else args.corr, min_len=args.min_len, max_len=args.max_len,
        min_visual=args.min_image_tokens, rng=rng,
    )
    records, stats = [], []
    for index, (length, visual) in enumerate(targets):
        record, record_stats = composer.natural(length) if scenario.natural else composer.synthetic(length, visual)
        record["sample_id"] = index
        record_stats.update(sample_id=index, target_tokens=length, target_visual_tokens=visual)
        records.append(record)
        stats.append(record_stats)
    return records, stats


def prepare(args: argparse.Namespace) -> None:
    """Download what is missing, build the samples and publish the dataset atomically."""
    output_dir: Path = args.output_dir
    if (output_dir / "vlm_conversations.json").is_file():
        print(f"dataset already present: {output_dir}")
        return
    from transformers import AutoTokenizer  # pylint: disable=C0415

    tokenizer = AutoTokenizer.from_pretrained(args.processor_path, local_files_only=True)
    conversations = read_conversations(download(args.download_dir, insecure=args.insecure, offline=args.offline))
    rng = random.Random(args.seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        started = time.perf_counter()
        composer = Composer(staging, tokenizer, conversations, rng, max_len=args.max_len,
                            min_image_tokens=args.min_image_tokens, max_image_tokens=args.max_image_tokens,
                            max_images=args.max_images)
        records, stats = build_samples(args, composer, rng)
        costs = [sample_cost(item, args.cost_visual) for item in stats]
        orders, balance = {}, {}
        args.arrange = ["random"] + [mode for mode in args.arrange if mode != "random"]
        for mode in args.arrange:
            orders[mode] = arrange(costs, args.dp_size, mode, random.Random(args.seed + 1))
            balance[mode] = step_imbalance(costs, orders[mode], args.dp_size)
            name = "vlm_conversations.json" if mode == "random" else f"vlm_conversations.{mode}.json"
            with (staging / name).open("w", encoding="utf-8") as handle:
                json.dump([records[index] for index in orders[mode]], handle)
        with (staging / "samples.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle)
        meta = {
            "source": "HuggingFaceM4/the_cauldron", "subsets": [subset for subset, _ in _SUBSETS],
            "scenario": args.scenario, "num_samples": args.num_samples, "seed": args.seed, "dp_size": args.dp_size,
            "cost": f"tokens + {args.cost_visual} * visual_tokens", "max_len": args.max_len,
            "tokens": describe([item["tokens"] for item in stats]),
            "visual_tokens": describe([item["visual_tokens"] for item in stats]),
            "images": describe([item["images"] for item in stats]),
            "cost_summary": describe(costs),
            "arrangements": balance,
            "build_seconds": round(time.perf_counter() - started, 1),
        }
        meta["verified"] = (verify(staging / "vlm_conversations.json", args.processor_path, args.max_len,
                                   stats, args.verify) if args.verify else None)
        with (staging / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"dataset ready: {output_dir}")
    print(f"tokens/sample mean {meta['tokens']['mean']:.0f} (cv {meta['tokens']['cv']:.2f}), visual "
          f"{meta['visual_tokens']['mean']:.0f} (cv {meta['visual_tokens']['cv']:.2f}), "
          f"{meta['images']['mean']:.1f} images")
    for mode, row in balance.items():
        print(f"  {mode:9s}: slowest rank of a step costs {row['max_over_mean']:.2f}x the mean "
              f"({row['idle_share']:.0%} of the ranks' time idle)")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--download-dir", type=Path, default=Path("/home/pl/data/the_cauldron"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mean-len", type=int, default=8192, help="mean tokens per sample")
    parser.add_argument("--len-cv", type=float, default=0.8, help="spread of the tokens (std / mean)")
    parser.add_argument("--min-len", type=int, default=1024)
    parser.add_argument("--max-len", type=int, default=16384, help="longest sample; the data transform must allow it")
    parser.add_argument("--mean-visual", type=int, default=2048, help="mean visual tokens per sample")
    parser.add_argument("--visual-cv", type=float, default=1.0, help="spread of the visual tokens (std / mean)")
    parser.add_argument("--corr", type=float, default=None,
                        help="correlation of length and visual tokens (default: the scenario's)")
    parser.add_argument("--min-image-tokens", type=int, default=MIN_IMAGE_TOKENS)
    parser.add_argument("--max-image-tokens", type=int, default=2048,
                        help="largest image: 2048 tokens is 1448 x 1448 px")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--arrange", nargs="+", choices=ARRANGEMENTS, default=list(ARRANGEMENTS),
                        help="orders to write; random is vlm_conversations.json, "
                             "the others vlm_conversations.<name>.json")
    parser.add_argument("--dp-size", type=int, default=32, help="ranks that share a step (the sampler's block)")
    parser.add_argument("--cost-visual", type=float, default=0.5,
                        help="extra cost of a visual token over a text token, for ordering the balanced arrangement")
    parser.add_argument("--offline", action="store_true",
                        help="never download: stop with the missing file names instead")
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    parser.add_argument("--verify", type=int, default=8,
                        help="records checked with the training transform (0: none); needs the processor")
    return parser.parse_args(argv)


if __name__ == "__main__":
    prepare(_parse_args())
