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
"""Convert the_cauldron Parquet subsets into a DeepSeek-V4.1 Online VLM dataset.

Unlike a fixed-length perf dataset, this converter **preserves** the natural
heterogeneity of the corpus: image bytes are copied unmodified, every
conversation turn is kept, and nothing is padded or packed. Sample cost
therefore varies the way real data varies, which is what a load-balancing or
packing experiment needs in order to have imbalance to recover.

Alongside the JSONL it writes a cost manifest and a summary. Cost is predicted
with the model's own sizing rules (``plan_image_grid`` and the chat template),
so the prediction matches what the training transform will produce without
decoding a single pixel — image dimensions come from the file header. The
summary reports the straggler-bound headroom, ``E[max over W] / E[mean] - 1``,
under two cost models: LLM tokens and ViT patches. Those bound what any
balancing change can win at one sample per rank per step, so they say up front
whether a target speedup is available in the data at all.

Parquet layout (the_cauldron): column ``images`` is a list of structs with a
``bytes`` field, column ``texts`` a list of structs with ``user`` and
``assistant`` fields.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from hyper_parallel.models.deepseek_v41.adapter.data.image_processor import (
    num_image_tokens,
    plan_image_grid,
)
from hyper_parallel.models.deepseek_v41.adapter.data.processor import build_deepseek_v41_processor

# One Parquet file per subset, named as `prepare_cauldron_data.py` stores them:
# the remote path with "/" replaced by "__".
_SUBSET_FILES = {
    "vsr": "vsr/train-00000-of-00001-b56e9224d46b0ed3.parquet",
    "infographic_vqa": "infographic_vqa/train-00000-of-00001-9187ab6377a43fd2.parquet",
    "scienceqa": "scienceqa/train-00000-of-00001-c411546b9bc4df22.parquet",
    "finqa": "finqa/train-00000-of-00001-4eb0e3dd12354fba.parquet",
}
_MANIFEST_FIELDS = (
    "id",
    "subset",
    "row",
    "num_images",
    "num_turns",
    "text_tokens",
    "image_tokens",
    "total_tokens",
    "vit_patches",
)


@dataclass
class Conversation:
    """One corpus row: its image payloads and its question/answer turns."""

    subset: str
    row: int
    images: list[bytes]
    turns: list[tuple[str, str]]


@dataclass
class SampleCost:
    """Predicted encoding cost of one converted sample."""

    text_tokens: int
    image_tokens: int
    vit_patches: int

    @property
    def total_tokens(self) -> int:
        """Return the predicted encoded sequence length."""
        return self.text_tokens + self.image_tokens


@dataclass
class ConversionStats:
    """Counters explaining what the conversion kept and dropped."""

    rows_read: int = 0
    kept: int = 0
    dropped_no_turns: int = 0
    dropped_oversize: int = 0
    images_dropped: int = 0
    costs: list[SampleCost] = field(default_factory=list)


def _plain(text: str) -> str:
    """Strip text and defuse media placeholders the transform would expand."""
    return text.strip().replace("<image>", "[image]").replace("<video>", "[video]")


def read_conversations(parquet_dir: Path, subsets: Sequence[str]) -> Iterator[Conversation]:
    """Stream conversations from the configured Parquet subsets.

    Args:
        parquet_dir: Directory holding the downloaded Parquet files.
        subsets: Subset names to read, in order.

    Yields:
        One `Conversation` per corpus row that carries usable turns.

    Raises:
        SystemExit: If a configured subset's Parquet file is absent.
    """
    # pyarrow must be imported after torch, which the model imports above pull in.
    import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

    for subset in subsets:
        remote = _SUBSET_FILES[subset]
        path = parquet_dir / remote.replace("/", "__")
        if not path.is_file():
            held = sorted(item.name for item in parquet_dir.iterdir()) if parquet_dir.is_dir() else None
            raise SystemExit(
                f"missing Parquet file for subset {subset!r}: {path}\n"
                + (f"the directory holds: {held}" if held is not None else "the directory does not exist")
                + "\ncopy it there from a machine with internet access (name as above)"
            )
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
            yield Conversation(subset=subset, row=row, images=payloads, turns=turns)


def build_messages(conversation: Conversation, image_urls: Sequence[str]) -> list[dict[str, Any]]:
    """Build the V4.1 message list for one conversation.

    Every image is a content block on the first user turn, in corpus order, so
    the placeholder order the template emits matches the image order the
    processor consumes. Later turns are plain strings, and the final message is
    the last assistant answer — the only supervised span.

    Args:
        conversation: Source conversation.
        image_urls: Image paths relative to the JSONL file, in corpus order.

    Returns:
        A message list satisfying the Online VLM contract.
    """
    messages: list[dict[str, Any]] = []
    for turn_index, (user, answer) in enumerate(conversation.turns):
        if turn_index == 0:
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": url}} for url in image_urls
            ]
            content.append({"type": "text", "text": user})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": answer})
    return messages


def predict_cost(
        processor: Any,
        messages: Sequence[dict[str, Any]],
        payloads: Sequence[bytes],
        *,
        thinking_mode: str = "chat",
        drop_thinking: bool = True,
) -> SampleCost:
    """Predict one sample's encoded cost without decoding image pixels.

    The chat template counts one placeholder token per image; the processor
    later expands each into ``n_llm_h * (n_llm_w + 1) + 2`` tokens. Reproducing
    that arithmetic over the header dimensions gives the exact encoded length.

    Args:
        processor: Built `DeepseekV41Processor`.
        messages: Message list for the sample.
        payloads: Encoded image bytes, in the order they appear.
        thinking_mode: Must match the training transform, which defaults to
            ``chat``; a different mode changes the prompt length.
        drop_thinking: Must likewise match the training transform.

    Returns:
        The predicted cost of the sample.
    """
    prompt, _media = processor.chat_template(
        list(messages),
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        return_multi_modal_data=True,
    )
    prompt_tokens = len(processor.tokenizer.encode(prompt))
    image_tokens = 0
    vit_patches = 0
    patch_size = processor.vision_patch_size
    for payload in payloads:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
        grid_h, grid_w, resized_h, resized_w = plan_image_grid(width, height, processor)
        image_tokens += num_image_tokens(grid_h, grid_w)
        vit_patches += (resized_h // patch_size) * (resized_w // patch_size)
    return SampleCost(
        text_tokens=prompt_tokens - len(payloads),
        image_tokens=image_tokens,
        vit_patches=vit_patches,
    )


def _imbalance_ceiling(costs: Sequence[int], world_size: int, trials: int, rng: random.Random) -> float:
    """Estimate ``E[max over world_size] / E[mean] - 1`` for a cost sample.

    At one sample per rank per step, a step costs the slowest rank, so this is
    the fraction a perfect balancer could remove. Estimated by sampling the
    empirical distribution, which needs no closed form for its shape.

    Args:
        costs: Observed per-sample costs.
        world_size: Ranks drawing one sample each per step.
        trials: Monte-Carlo draws.
        rng: Seeded generator, so the reported number is reproducible.

    Returns:
        The headroom as a fraction, or 0.0 when the costs are degenerate.
    """
    mean_cost = statistics.fmean(costs)
    if mean_cost <= 0:
        return 0.0
    expected_max = statistics.fmean(
        max(rng.choices(costs, k=world_size)) for _ in range(trials)
    )
    return expected_max / mean_cost - 1.0


def summarize(stats: ConversionStats, world_sizes: Sequence[int], seed: int, trials: int = 4000) -> dict[str, Any]:
    """Build the cost summary, including the balancing headroom per world size.

    Args:
        stats: Populated conversion statistics.
        world_sizes: Rank counts to report headroom for.
        seed: Seed for the Monte-Carlo estimate.
        trials: Monte-Carlo draws per world size.

    Returns:
        A JSON-serialisable summary.
    """
    token_costs = [cost.total_tokens for cost in stats.costs]
    patch_costs = [cost.vit_patches for cost in stats.costs]
    summary: dict[str, Any] = {
        "samples": len(stats.costs),
        "rows_read": stats.rows_read,
        "dropped_no_turns": stats.dropped_no_turns,
        "dropped_oversize": stats.dropped_oversize,
        "images_dropped_to_fit": stats.images_dropped,
    }
    if not stats.costs:
        return summary
    for name, costs in (("total_tokens", token_costs), ("vit_patches", patch_costs)):
        ordered = sorted(costs)
        mean_cost = statistics.fmean(costs)
        summary[name] = {
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[int(len(ordered) * 0.9)],
            "p99": ordered[int(len(ordered) * 0.99)],
            "max": ordered[-1],
            "mean": round(mean_cost, 1),
            # Coefficient of variation: scale-free heterogeneity of this corpus.
            "cv": round(statistics.pstdev(costs) / mean_cost, 3) if mean_cost else 0.0,
            "imbalance_ceiling": {
                str(world): round(_imbalance_ceiling(costs, world, trials, random.Random(seed)), 3)
                for world in world_sizes
            },
        }
    return summary


def convert(args: argparse.Namespace) -> ConversionStats:
    """Convert the configured subsets into JSONL splits plus a cost manifest.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Statistics describing the conversion.

    Raises:
        SystemExit: If no sample survived the length constraint.
    """
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    processor = build_deepseek_v41_processor(config_path=args.model_dir)

    stats = ConversionStats()
    # Only an explicit train count caps the read; otherwise every row is
    # converted and the validation split is carved out of the result.
    wanted = (args.num_train_samples + args.num_valid_samples) if args.num_train_samples else 0
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for conversation in read_conversations(Path(args.parquet_dir).expanduser().resolve(), args.subsets):
        stats.rows_read += 1
        if wanted and len(records) >= wanted:
            break
        if not conversation.turns:
            stats.dropped_no_turns += 1
            continue

        payloads = list(conversation.images[: args.max_images_per_sample])
        sample_id = f"{conversation.subset}_{conversation.row}"
        # Trim images only if the sample would otherwise exceed max_seq_len:
        # the transform raises on an over-long sample rather than truncating.
        while True:
            urls = [f"images/{sample_id}_{index}.jpg" for index in range(len(payloads))]
            cost = predict_cost(processor, build_messages(conversation, urls), payloads)
            if cost.total_tokens <= args.max_seq_len:
                break
            if not payloads or not args.trim_images:
                break
            payloads.pop()
            stats.images_dropped += 1
        if cost.total_tokens > args.max_seq_len:
            stats.dropped_oversize += 1
            continue

        for index, payload in enumerate(payloads):
            (image_dir / f"{sample_id}_{index}.jpg").write_bytes(payload)
        urls = [f"images/{sample_id}_{index}.jpg" for index in range(len(payloads))]
        records.append({
            "id": sample_id,
            "messages": build_messages(conversation, urls),
            "source": {"dataset": f"HuggingFaceM4/the_cauldron/{conversation.subset}", "row_index": conversation.row},
        })
        manifest.append({
            "id": sample_id,
            "subset": conversation.subset,
            "row": conversation.row,
            "num_images": len(payloads),
            "num_turns": len(conversation.turns),
            "text_tokens": cost.text_tokens,
            "image_tokens": cost.image_tokens,
            "total_tokens": cost.total_tokens,
            "vit_patches": cost.vit_patches,
        })
        stats.costs.append(cost)
        stats.kept += 1

    if not records:
        raise SystemExit("no sample survived conversion; check --parquet-dir and --max-seq-len")

    # Shuffle before splitting so each split spans every subset; the sampler's
    # own order is a separate, configured concern.
    random.Random(args.seed).shuffle(records)
    # Without an explicit train count, the validation split is reserved first
    # and training takes the remainder.
    train_count = args.num_train_samples or max(0, len(records) - args.num_valid_samples)
    splits = {"train": records[:train_count]}
    if args.num_valid_samples:
        splits["valid"] = records[train_count: train_count + args.num_valid_samples]
    for split, split_records in splits.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as split_file:
            for record in split_records:
                split_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {output_dir / f'{split}.jsonl'} ({len(split_records)} samples)")

    with (output_dir / "cost_manifest.csv").open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    summary = summarize(stats, args.world_sizes, args.seed)
    (output_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return stats


def verify(args: argparse.Namespace, count: int) -> None:
    """Re-encode the first samples through the real transform and compare costs.

    Args:
        args: Parsed command-line arguments.
        count: Number of samples to re-encode.

    Raises:
        SystemExit: If a predicted length does not match the encoded length.
    """
    # Imported here so a conversion-only run does not pay for the transform.
    from hyper_parallel.models.deepseek_v41.adapter.data.transform_fn import (  # pylint: disable=C0415
        build_deepseek_v41_omni_transform,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    jsonl_path = output_dir / "train.jsonl"
    processor = build_deepseek_v41_processor(config_path=args.model_dir)
    transform = build_deepseek_v41_omni_transform(processor=processor, max_seq_len=args.max_seq_len)
    with (output_dir / "cost_manifest.csv").open(encoding="utf-8", newline="") as manifest_file:
        costs = {row["id"]: int(row["total_tokens"]) for row in csv.DictReader(manifest_file)}
    with jsonl_path.open(encoding="utf-8") as jsonl_file:
        for line_index, line in enumerate(jsonl_file):
            if line_index >= count:
                break
            record = json.loads(line)
            record["__online_source_path__"] = str(jsonl_path)
            encoded = transform.encode_sample(record)
            actual = int(encoded["input_ids"].shape[-1])
            predicted = costs[record["id"]]
            if actual != predicted:
                raise SystemExit(
                    f"cost prediction mismatch for {record['id']}: predicted {predicted}, encoded {actual}"
                )
            print(f"verified {record['id']}: {actual} tokens")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert the_cauldron into a V4.1 Online VLM dataset")
    parser.add_argument("--parquet-dir", default="/home/pl/data/the_cauldron")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="DeepSeek-V4.1-Flash assets (config + tokenizer)")
    parser.add_argument("--subsets", nargs="+", default=sorted(_SUBSET_FILES), choices=sorted(_SUBSET_FILES))
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-train-samples", type=int, default=0, help="0 keeps every converted sample")
    parser.add_argument("--num-valid-samples", type=int, default=0)
    parser.add_argument("--max-images-per-sample", type=int, default=8)
    parser.add_argument(
        "--trim-images",
        action="store_true",
        help="drop trailing images from an over-long sample instead of dropping the sample",
    )
    parser.add_argument("--world-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", type=int, default=0, help="re-encode N samples and check the predicted cost")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Convert the corpus and optionally verify the cost prediction."""
    args = _parse_args(argv)
    convert(args)
    if args.verify:
        verify(args, args.verify)


if __name__ == "__main__":
    main()
