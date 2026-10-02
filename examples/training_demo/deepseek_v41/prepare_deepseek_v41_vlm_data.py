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
"""Build a local Online VLM JSONL dataset for the V4.1 multimodal smoke.

The emitted records follow the OpenAI-style ``messages`` contract described in
``docs/guide/data/deepseek_v41_vlm_online_data_guide.md``: one image content
block per sample, a short question, and a final assistant message that carries
the only supervised tokens. Image paths stay relative to the JSONL file so the
native Online Mapping source resolves them from its own directory, which keeps
the dataset movable between hosts.

This generator is offline and stdlib-only. It copies the source images next to
the JSONL instead of re-encoding them, because the model image processor owns
resizing, patching and placeholder expansion at training time.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Short answers keep the supervised span small and stable across samples.
QUESTION_ANSWER_TEMPLATES = (
    ("What is shown in this image?", "A photograph of produce."),
    ("Describe the image in one word.", "Vegetables."),
    ("Is the subject of this image edible?", "Yes."),
    ("How many distinct colours stand out in this image?", "Two."),
    ("Does this image contain any text?", "No."),
    ("Name the dominant colour in this image.", "Orange."),
    ("Is this image a photograph or a diagram?", "A photograph."),
    ("Would this image appear in a cookbook?", "Yes."),
)


def _collect_images(sources: Sequence[str]) -> list[Path]:
    """Resolve the configured sources into an ordered list of image files.

    Args:
        sources: Directories, globs or explicit image paths.

    Returns:
        Deterministically ordered image paths.

    Raises:
        ValueError: If no readable image file is found.
    """
    collected: list[Path] = []
    for source in sources:
        candidate = Path(source).expanduser()
        if candidate.is_dir():
            matches = sorted(
                path for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif candidate.is_file():
            matches = [candidate]
        else:
            parent = candidate.parent if str(candidate.parent) else Path(".")
            matches = sorted(
                path for path in parent.glob(candidate.name)
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        collected.extend(match.resolve() for match in matches)
    unique = list(dict.fromkeys(collected))
    if not unique:
        raise ValueError(f"No image files found in: {', '.join(sources)}")
    return unique


def _build_record(sample_index: int, relative_image: str, dataset_name: str, split: str) -> dict:
    """Build one OpenAI-style multimodal SFT record.

    Args:
        sample_index: Global sample index, used for the id and the template.
        relative_image: Image path relative to the JSONL file.
        dataset_name: Tracking name written into ``source``.
        split: Split name written into ``source``.

    Returns:
        A JSON-serialisable record ending with an assistant message.
    """
    question, answer = QUESTION_ANSWER_TEMPLATES[sample_index % len(QUESTION_ANSWER_TEMPLATES)]
    return {
        "id": f"{split}_{sample_index}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": relative_image}},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ],
        "source": {"dataset": dataset_name, "split": split, "row_index": sample_index},
    }


def prepare_vlm_data(
        output_dir: Path,
        image_sources: Sequence[str],
        *,
        num_train_samples: int = 512,
        num_valid_samples: int = 128,
        dataset_name: str = "local/deepseek_v41_smoke",
) -> dict[str, Path]:
    """Write the train and valid JSONL files and stage their images.

    Source images are cycled, so a handful of files is enough to build any
    sample count. Each split gets its own JSONL beside a shared ``images``
    directory.

    Args:
        output_dir: Directory that receives the JSONL files and ``images/``.
        image_sources: Directories, globs or explicit image paths.
        num_train_samples: Number of training records.
        num_valid_samples: Number of validation records.
        dataset_name: Tracking name written into every ``source`` field.

    Returns:
        Mapping from split name to the written JSONL path.

    Raises:
        ValueError: If a requested sample count is negative or train is empty.
    """
    if num_train_samples <= 0:
        raise ValueError("num_train_samples must be positive")
    if num_valid_samples < 0:
        raise ValueError("num_valid_samples must not be negative")
    images = _collect_images(image_sources)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for image_index, image_path in enumerate(images):
        target = image_dir / f"image_{image_index:04d}{image_path.suffix.lower()}"
        if not target.exists() or target.stat().st_size != image_path.stat().st_size:
            shutil.copyfile(image_path, target)
        staged.append(f"images/{target.name}")

    written: dict[str, Path] = {}
    sample_index = 0
    for split, count in (("train", num_train_samples), ("valid", num_valid_samples)):
        if count == 0:
            continue
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as split_file:
            for _ in range(count):
                record = _build_record(sample_index, staged[sample_index % len(staged)], dataset_name, split)
                split_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                sample_index += 1
        written[split] = split_path
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare V4.1 Online VLM smoke data")
    parser.add_argument("--output-dir", required=True, help="Directory for the JSONL files and images/")
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        help="Image directories, globs or files; cycled to reach the sample counts",
    )
    parser.add_argument("--num-train-samples", type=int, default=512)
    parser.add_argument("--num-valid-samples", type=int, default=128)
    parser.add_argument("--dataset-name", default="local/deepseek_v41_smoke")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the requested multimodal JSONL dataset."""
    args = _parse_args(argv)
    written = prepare_vlm_data(
        Path(args.output_dir).expanduser().resolve(),
        args.images,
        num_train_samples=args.num_train_samples,
        num_valid_samples=args.num_valid_samples,
        dataset_name=args.dataset_name,
    )
    for split, path in written.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
