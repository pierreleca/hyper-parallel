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
"""Build fixed-length Qwen3-VL samples from real image-text conversations.

Routing depends on content, so load-imbalance numbers measured on synthetic
text and solid-colour images say little about real use. This script downloads
four single-file subsets of HuggingFaceM4/the_cauldron (about 0.8 GB), chosen
to cover different kinds of images and text:

- vsr: natural photos, short spatial-reasoning questions;
- infographic_vqa: dense infographics, questions about their text;
- scienceqa: science diagrams and photos, multiple-choice with explanations;
- finqa: financial report tables, long reasoning answers.

Real conversations are short (a few hundred tokens with their image), so each
output sample joins consecutive conversations, shuffled across the subsets,
into one multi-turn conversation longer than ``--seq-len``. The Trainer
transform then truncates it to exactly ``seq_len`` tokens: every step carries
the same number of tokens and no padding. All images sit in the early turns;
the last assistant turn is real text only and long enough that the truncation
always falls inside it, because the transform drops any image it cuts and
trains only on the last assistant turn.

Images are resized to multiples of 32 pixels, at most 1024 x 1024 pixels in
area, so each takes at most 1024 visual tokens and the Qwen3-VL processor does
not resize them again.

    python examples/qwen3_vl_30b_perf/prepare_cauldron_data.py \\
      --output-dir /home/pl/data/qwen3_vl_30b_perf/cauldron_seq16384_n240 \\
      --seq-len 16384 --num-samples 240 \\
      --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct

Downloads go to ``--download-dir`` and are reused; ``HF_ENDPOINT`` selects a
mirror of huggingface.co. Behind a proxy that re-signs TLS traffic, point
``SSL_CERT_FILE`` at its CA certificate, or pass ``--insecure`` to skip the
verification.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import ssl
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

_REPO = "HuggingFaceM4/the_cauldron"
# (subset, file) pairs: one Parquet file each, 107-292 MB, 0.8 GB together.
_SUBSETS = (
    ("vsr", "vsr/train-00000-of-00001-b56e9224d46b0ed3.parquet"),
    ("infographic_vqa", "infographic_vqa/train-00000-of-00001-9187ab6377a43fd2.parquet"),
    ("scienceqa", "scienceqa/train-00000-of-00001-c411546b9bc4df22.parquet"),
    ("finqa", "finqa/train-00000-of-00001-4eb0e3dd12354fba.parquet"),
)
# Qwen3-VL: 16-pixel patches merged 2x2, so one visual token per 32 x 32 pixels.
_TOKEN_EDGE = 32
_MAX_PIXELS = 1024 * 1024
_MIN_PIXELS = 256 * 256
# Chat-template tokens around one turn, and around one image's token run.
_TURN_OVERHEAD = 12
_IMAGE_OVERHEAD = 2
# Tokens the final text-only turn must reach past seq_len, so estimation error
# cannot move the truncation into an image or into the image-bearing turns.
_TEXT_TAIL = 2048


@dataclass
class Conversation:
    """One real conversation: its images and its question-answer turns."""

    subset: str
    row: int
    images: list[bytes]
    turns: list[tuple[str, str]]


def download(download_dir: Path, insecure: bool = False) -> list[tuple[str, Path]]:
    """Fetch the Parquet files that are not already in ``download_dir``.

    ``insecure`` skips TLS certificate verification, for a proxy that
    re-signs traffic with a certificate the environment does not trust.
    Pointing ``SSL_CERT_FILE`` at the proxy's CA certificate keeps
    verification on instead.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    context = None
    if insecure:
        print("warning: TLS certificate verification is disabled (--insecure)")
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    download_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for subset, remote in _SUBSETS:
        target = download_dir / remote.replace("/", "__")
        if not target.is_file():
            url = f"{endpoint}/datasets/{_REPO}/resolve/main/{remote}"
            print(f"downloading {url}")
            partial = target.with_suffix(".part")
            with urllib.request.urlopen(url, context=context) as response, \
                    partial.open("wb") as handle:  # nosec B310
                shutil.copyfileobj(response, handle, length=1 << 20)
            partial.rename(target)
        files.append((subset, target))
    return files


def _plain(text: str) -> str:
    """Strip a text and defuse media placeholders the transform would expand."""
    return text.strip().replace("<image>", "[image]").replace("<video>", "[video]")


def read_conversations(files: list[tuple[str, Path]]) -> list[Conversation]:
    """Load every conversation of the downloaded subsets.

    pyarrow is imported here, after the tokenizer has imported torch: a pip
    pyarrow wheel loads the system libstdc++, and once that older copy is in
    the process, torch_npu's own imports (sqlite3 through the environment's
    ICU) fail to find the C++ ABI they need.
    """
    import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

    conversations = []
    for subset, path in files:
        table = pq.read_table(path, columns=["images", "texts"])
        images_column = table.column("images").to_pylist()
        texts_column = table.column("texts").to_pylist()
        for row, (images, texts) in enumerate(zip(images_column, texts_column)):
            turns = [
                (_plain(turn["user"]), _plain(turn["assistant"]))
                for turn in texts or []
                if turn.get("user") and turn.get("assistant")
            ]
            payloads = [image["bytes"] for image in images or [] if image and image.get("bytes")]
            if turns and payloads:
                conversations.append(Conversation(subset, row, payloads, turns))
        print(f"{subset}: {sum(c.subset == subset for c in conversations)} conversations")
    return conversations


def fit_image(payload: bytes) -> Image.Image:
    """Decode and resize an image to 32-pixel multiples within the pixel bounds."""
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    width, height = image.size
    scale = 1.0
    if width * height > _MAX_PIXELS:
        scale = math.sqrt(_MAX_PIXELS / (width * height))
    elif width * height < _MIN_PIXELS:
        scale = math.sqrt(_MIN_PIXELS / (width * height))
    new_width = max(_TOKEN_EDGE, int(width * scale) // _TOKEN_EDGE * _TOKEN_EDGE)
    new_height = max(_TOKEN_EDGE, int(height * scale) // _TOKEN_EDGE * _TOKEN_EDGE)
    while new_width * new_height < _MIN_PIXELS:
        new_width += _TOKEN_EDGE
        new_height += _TOKEN_EDGE
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def visual_tokens(image: Image.Image) -> int:
    """Return the merged visual tokens of an image already fitted by fit_image."""
    return (image.width // _TOKEN_EDGE) * (image.height // _TOKEN_EDGE)


class SampleBuilder:
    """Join shuffled conversations into records longer than ``seq_len``."""

    def __init__(self, staging: Path, tokenizer: Any, seq_len: int) -> None:
        """Bind the output directory, the tokenizer and the target length."""
        self.staging = staging
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        (staging / "images").mkdir(parents=True, exist_ok=True)

    def _tokens(self, text: str) -> int:
        """Count the tokens of a text, without special tokens."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _text_tokens(self, conversation: Conversation) -> int:
        """Estimate a conversation's text tokens, template included."""
        return sum(
            self._tokens(user) + self._tokens(answer) + 2 * _TURN_OVERHEAD
            for user, answer in conversation.turns
        )

    def _save_images(self, conversation: Conversation, fitted: list[Image.Image]) -> list[str]:
        """Write a conversation's fitted images and return their relative paths."""
        names = []
        for index, image in enumerate(fitted):
            name = f"images/{conversation.subset}_{conversation.row:05d}_{index}.jpg"
            image.save(self.staging / name, quality=90)
            names.append(name)
        return names

    def build(self, stream: Iterator[Conversation]) -> tuple[dict, dict]:
        """Build one record and its statistics from the next conversations.

        Image-bearing conversations are added while they fit ``seq_len`` minus
        the text tail; the first one that does not fit starts the text-only
        tail instead, so no image can reach the truncation point.
        """
        messages, images = [], []
        estimate, visual = 0, 0
        budget = self.seq_len - _TEXT_TAIL
        overflow = None
        for conversation in stream:
            fitted = [fit_image(payload) for payload in conversation.images]
            image_tokens = sum(visual_tokens(image) + _IMAGE_OVERHEAD for image in fitted)
            tokens = image_tokens + self._text_tokens(conversation)
            if messages and estimate + tokens > budget:
                overflow = conversation
                break
            names = self._save_images(conversation, fitted)
            images.extend(names)
            visual += image_tokens
            estimate += tokens
            for turn_index, (user, answer) in enumerate(conversation.turns):
                prefix = "<image>" * len(names) if turn_index == 0 else ""
                messages.append({"role": "user", "content": prefix + user})
                messages.append({"role": "assistant", "content": answer})
        if overflow is None:
            raise RuntimeError("ran out of conversations; lower --num-samples or --seq-len")

        # The text tail: the next conversations' text, without their images.
        question, tail, tail_tokens = None, [], 0
        pending = [overflow]
        while estimate + tail_tokens < self.seq_len + _TEXT_TAIL:
            if not pending:
                conversation = next(stream, None)
                if conversation is None:
                    raise RuntimeError("ran out of conversations; lower --num-samples or --seq-len")
                pending.append(conversation)
            for user, answer in pending.pop().turns:
                if question is None:
                    question = user
                    tail.append(answer)
                else:
                    tail.append(f"{user}\n{answer}")
                tail_tokens += self._tokens(tail[-1]) + 1
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": "\n\n".join(tail)})
        record = {"messages": messages, "images": images}
        stats = {"images": len(images), "visual_tokens": visual, "estimated_tokens": estimate + tail_tokens}
        return record, stats


def verify(json_path: Path, processor_path: str, seq_len: int, count: int) -> list[dict]:
    """Encode the first records with the Trainer transform: full length, no image lost."""
    # Heavy optional imports, only needed for the verification.
    from hyper_parallel.data.vlm import build_processor, build_vlm_data_transform  # pylint: disable=C0415
    from hyper_parallel.data.vlm.dataset import VLMDataset  # pylint: disable=C0415

    transform = build_vlm_data_transform(processor=build_processor(processor_path), max_seq_len=seq_len)
    dataset = VLMDataset(str(json_path))
    reports = []
    for index in range(len(dataset) if count <= 0 else min(count, len(dataset))):
        record = dataset[index]
        sample = transform(record)
        report = {
            "record": index,
            "real_tokens": int(sample["attention_mask"].sum()),
            "images": int(sample["image_grid_thw"].shape[0]),
            "images_in_record": len(record["images"]),
            "visual_tokens": int((sample["mm_token_type_ids"] == 1).sum()),
            "trainable_labels": int((sample["labels"] >= 0).sum()),
        }
        if report["real_tokens"] != seq_len or report["images"] != report["images_in_record"]:
            raise RuntimeError(f"record {index} would be padded or lose an image: {report}")
        reports.append(report)
        if index < 3:
            print(f"verified record {index}: {report}")
    print(f"verified {len(reports)} records: full length, every image kept")
    return reports[:3]


def prepare(args: argparse.Namespace) -> None:
    """Download, join and publish the dataset atomically."""
    output_dir: Path = args.output_dir
    if (output_dir / "vlm_conversations.json").is_file():
        print(f"dataset already present: {output_dir}")
        return
    from transformers import AutoTokenizer  # pylint: disable=C0415

    tokenizer = AutoTokenizer.from_pretrained(args.processor_path, local_files_only=True)
    conversations = read_conversations(download(args.download_dir, insecure=args.insecure))
    random.Random(args.seed).shuffle(conversations)
    stream = iter(conversations)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        builder = SampleBuilder(staging, tokenizer, args.seq_len)
        records, stats = [], []
        start = time.perf_counter()
        for _ in range(args.num_samples):
            record, record_stats = builder.build(stream)
            records.append(record)
            stats.append(record_stats)
        with (staging / "vlm_conversations.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle)
        meta = {
            "source": _REPO,
            "subsets": [subset for subset, _ in _SUBSETS],
            "seq_len": args.seq_len,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "conversations_available": len(conversations),
            "images_per_sample_mean": sum(s["images"] for s in stats) / len(stats),
            "visual_token_share_mean": sum(s["visual_tokens"] for s in stats) / (len(stats) * args.seq_len),
            "build_seconds": round(time.perf_counter() - start, 1),
            "verified": verify(staging / "vlm_conversations.json", args.processor_path, args.seq_len,
                               args.verify),
        }
        with (staging / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"dataset ready: {output_dir}")
    print(
        f"{meta['images_per_sample_mean']:.1f} images per sample, "
        f"{meta['visual_token_share_mean']:.0%} of the tokens visual"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument(
        "--download-dir", type=Path,
        default=Path("/home/pl/data/the_cauldron"),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--insecure", action="store_true",
        help="skip TLS certificate verification (a proxy with its own certificate)",
    )
    parser.add_argument(
        "--verify", type=int, default=0,
        help="records checked with the Trainer transform; 0 checks every record (about 1 s each)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    prepare(_parse_args())
