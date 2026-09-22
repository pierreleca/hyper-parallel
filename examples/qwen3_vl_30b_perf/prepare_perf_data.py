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
"""Generate padding-free synthetic Qwen3-VL conversations of a target length.

Every conversation holds ``images_per_sample`` images followed by a question
and a long assistant answer. The answer is sized so the rendered sample is
longer than ``seq_len`` tokens; the master VLM data transform then truncates
every sample to exactly ``seq_len`` tokens, so no batch carries padding. This
keeps every profiled token real work, and it keeps the text attention
mask-free: with ``attn_implementation: flash_attention_2`` Transformers drops
an all-ones mask, and the fused text attention then runs the NPU causal sparse
mode.

Images come from a small shared pool so the dataset stays compact on disk. The
output directory is published atomically, so several nodes may run this
script against the same shared path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

# Common English words that the Qwen tokenizer encodes as one token each when
# preceded by a space, so an answer's word count bounds its token count below.
_WORDS = (
    "the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "with", "was", "on", "be", "by",
    "this", "are", "from", "at", "or", "an", "which", "have", "not", "has", "but", "were", "can", "all",
    "their", "more", "one", "also", "other", "time", "there", "been", "first", "into", "new", "some",
    "would", "two", "only", "when", "used", "these", "may", "most", "many", "over", "such", "after",
    "like", "then", "them", "than", "well", "very", "part", "image", "color", "left", "right",
)
_BASE_COLORS = ((200, 40, 40), (40, 180, 40), (40, 60, 200), (220, 210, 50), (200, 50, 200), (60, 200, 210))
# Qwen3-VL: 16-pixel patches merged 2x2 -> one visual token per 32x32 pixels.
_PIXELS_PER_VISUAL_TOKEN_EDGE = 32
# Covers the chat-template tokens and the question on top of the answer words.
_TEXT_MARGIN_WORDS = 512


def visual_tokens_per_image(image_size: int) -> int:
    """Return the merged visual token count of one square image."""
    edge = image_size // _PIXELS_PER_VISUAL_TOKEN_EDGE
    return edge * edge


def _write_image_pool(root: Path, pool_size: int, image_size: int, seed: int) -> list[str]:
    """Write deterministic noisy solid-color images and return their relative paths."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(pool_size):
        rng = np.random.default_rng(seed + index)
        base = np.array(_BASE_COLORS[index % len(_BASE_COLORS)], dtype=np.int16)
        noise = rng.integers(-30, 30, size=(image_size, image_size, 3), dtype=np.int16)
        pixels = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
        name = f"images/pool_{index:03d}.png"
        Image.fromarray(pixels).save(root / name)
        names.append(name)
    return names


def _build_records(
    image_names: list[str],
    *,
    seq_len: int,
    num_samples: int,
    images_per_sample: int,
    image_size: int,
    seed: int,
) -> list[dict]:
    """Build LLaVA-style records whose rendered length exceeds ``seq_len``."""
    visual_tokens = images_per_sample * visual_tokens_per_image(image_size)
    if visual_tokens >= seq_len:
        raise ValueError(
            f"{images_per_sample} images of {image_size} px take {visual_tokens} visual tokens, "
            f"which leaves no text in a {seq_len}-token sequence"
        )
    answer_words = seq_len - visual_tokens + _TEXT_MARGIN_WORDS
    words = np.array(_WORDS)
    rng = np.random.default_rng(seed)
    records = []
    for sample_index in range(num_samples):
        images = [
            image_names[(sample_index * images_per_sample + slot) % len(image_names)]
            for slot in range(images_per_sample)
        ]
        answer = " ".join(words[rng.integers(0, len(words), size=answer_words)].tolist())
        records.append({
            "messages": [
                {
                    "role": "user",
                    "content": "<image>" * images_per_sample + f"Describe the images of sample {sample_index}.",
                },
                {"role": "assistant", "content": answer},
            ],
            "images": images,
        })
    return records


def _verify_first_sample(json_path: Path, processor_path: str, seq_len: int, images_per_sample: int) -> dict:
    """Encode the first record with the Trainer transform and check it is padding-free."""
    # Heavy optional imports: only needed when verification is requested.
    from hyper_parallel.data.vlm import build_processor, build_vlm_data_transform  # pylint: disable=C0415
    from hyper_parallel.data.vlm.dataset import VLMDataset  # pylint: disable=C0415

    transform = build_vlm_data_transform(processor=build_processor(processor_path), max_seq_len=seq_len)
    record = VLMDataset(str(json_path))[0]
    start = time.perf_counter()
    sample = transform(record)
    report = {
        "tokens": int(sample["input_ids"].shape[0]),
        "real_tokens": int(sample["attention_mask"].sum()),
        "images": int(sample["image_grid_thw"].shape[0]),
        "visual_tokens": int((sample["mm_token_type_ids"] == 1).sum()),
        "trainable_labels": int((sample["labels"] >= 0).sum()),
        "transform_seconds": round(time.perf_counter() - start, 2),
    }
    if report["real_tokens"] != seq_len:
        raise RuntimeError(f"first sample would be padded: {report}")
    if report["images"] != images_per_sample:
        raise RuntimeError(f"truncation dropped images: {report}")
    print(f"verified first sample: {report}")
    return report


def prepare_perf_data(
    output_dir: Path,
    *,
    seq_len: int,
    num_samples: int,
    images_per_sample: int,
    image_size: int = 1024,
    pool_size: int = 16,
    seed: int = 1234,
    processor_path: str | None = None,
) -> None:
    """Write ``vlm_conversations.json`` plus its image pool into ``output_dir``.

    Args:
        output_dir: Final dataset directory; left untouched when it already exists.
        seq_len: Trainer ``max_seq_len`` every sample must fill.
        num_samples: Number of conversations.
        images_per_sample: Images referenced by each conversation.
        image_size: Square image edge in pixels (a multiple of 32).
        pool_size: Number of distinct images shared by all conversations.
        seed: Seed of the deterministic content generator.
        processor_path: Optional model directory; when given, the first sample is
            encoded with the Trainer transform to prove it is padding-free.
    """
    if min(seq_len, num_samples, images_per_sample, pool_size) <= 0:
        raise ValueError("seq_len, num_samples, images_per_sample and pool_size must be positive")
    if image_size <= 0 or image_size % _PIXELS_PER_VISUAL_TOKEN_EDGE:
        raise ValueError(f"image_size must be a positive multiple of {_PIXELS_PER_VISUAL_TOKEN_EDGE}")
    if (output_dir / "vlm_conversations.json").is_file():
        print(f"dataset already present: {output_dir}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        image_names = _write_image_pool(staging, pool_size, image_size, seed)
        records = _build_records(
            image_names,
            seq_len=seq_len,
            num_samples=num_samples,
            images_per_sample=images_per_sample,
            image_size=image_size,
            seed=seed,
        )
        with (staging / "vlm_conversations.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle)
        meta = {
            "seq_len": seq_len,
            "num_samples": num_samples,
            "images_per_sample": images_per_sample,
            "image_size": image_size,
            "visual_tokens_per_sample": images_per_sample * visual_tokens_per_image(image_size),
            "pool_size": pool_size,
            "seed": seed,
        }
        if processor_path is not None:
            meta["first_sample"] = _verify_first_sample(
                staging / "vlm_conversations.json", processor_path, seq_len, images_per_sample
            )
        with (staging / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        try:
            os.rename(staging, output_dir)
        except OSError:
            # Another node published the same deterministic dataset first.
            if not (output_dir / "vlm_conversations.json").is_file():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"dataset ready: {output_dir}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--images-per-sample", required=True, type=int)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--processor-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the dataset described by the command line."""
    args = _parse_args(argv)
    prepare_perf_data(
        args.output_dir,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        images_per_sample=args.images_per_sample,
        image_size=args.image_size,
        pool_size=args.pool_size,
        seed=args.seed,
        processor_path=args.processor_path,
    )


if __name__ == "__main__":
    main()
