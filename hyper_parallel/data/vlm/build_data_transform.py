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
"""Build the VLM multimodal sample transform."""

__all__ = ["VLMChatTransform", "build_vlm_data_transform"]

import re
from typing import Any, Optional

import torch

from hyper_parallel.data.constants import IGNORE_INDEX

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")
PADDING_MODES = ("max_length", "none")
TEXT_ONLY_MODES = ("keep", "placeholder")
_MODAL_FIELDS = ("pixel_values", "image_grid_thw")
# Images are RGB; the processor carries the patch geometry but not the channel count.
_IMAGE_CHANNELS = 3


def _template_reads(chat_template: Any, variable: str) -> bool:
    """Return whether a chat template, or any template of a named set, mentions ``variable``."""
    templates = chat_template.values() if isinstance(chat_template, dict) else (chat_template,)
    return any(isinstance(template, str) and variable in template for template in templates)


class VLMChatTransform:
    """Encode one multimodal conversation into one padded model sample.

    ``messages`` follows the Qwen3-VL content-list format
    (``{"type": "image", "url": ...}`` / ``{"type": "text", "text": ...}``), and
    ``<image>`` string placeholders are also accepted. The final ``assistant``
    turn is the training target; the preceding prompt is masked with -100.

    Args:
        processor: Qwen3-VL processor (tokenizer + image/video processors).
        max_seq_len: Target sequence length for truncation and padding.
        padding: ``"max_length"`` pads every sample to ``max_seq_len``; ``"none"`` leaves each sample at
            its own length and makes ``max_seq_len`` a ceiling. Padding to a constant costs the ratio of
            that constant to the mean sample, paid in full by the decoder and the experts, because the
            router scores a pad position like any other; ``"none"`` needs a collator that pads a
            micro-batch to its longest sample.

    Raises:
        ValueError: If ``padding`` is not one of :data:`PADDING_MODES`.
    """

    def __init__(
            self,
            processor: Any,
            *,
            max_seq_len: int = 256,
            padding: str = "max_length",
            text_only: str = "keep",
    ) -> None:
        """Store the processor, the target sequence length and the padding and text-only policies."""
        if padding not in PADDING_MODES:
            raise ValueError(f"padding must be one of {PADDING_MODES}, but got {padding!r}")
        if text_only not in TEXT_ONLY_MODES:
            raise ValueError(f"text_only must be one of {TEXT_ONLY_MODES}, but got {text_only!r}")
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.text_only = text_only

    def _image_token_id(self) -> int:
        """Return the id of the image placeholder token.

        Raises:
            ValueError: If the processor exposes neither the id nor the token to look it up with.
        """
        token_id = getattr(self.processor, "image_token_id", None)
        if token_id is not None:
            return int(token_id)
        tokenizer = getattr(self.processor, "tokenizer", None)
        image_token = getattr(self.processor, "image_token", None)
        if tokenizer is None or image_token is None:
            raise ValueError("text_only='placeholder' needs the processor's image token to build one")
        return int(tokenizer.convert_tokens_to_ids(image_token))

    def _vision_placeholder(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the smallest image the vision tower accepts: one merged token of blank patches.

        Raises:
            ValueError: If the processor carries no image processor to read the patch geometry from.
        """
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("text_only='placeholder' needs the processor's image_processor for the geometry")
        merge = int(image_processor.merge_size)
        patch = int(image_processor.patch_size)
        temporal = int(image_processor.temporal_patch_size)
        # A grid of one merge block is the smallest the tower's patch merger accepts; the tower reads
        # pixel_values as (-1, channels, temporal_patch, patch, patch), so a row is their product.
        grid = torch.tensor([[1, merge, merge]], dtype=torch.long)
        pixel_values = torch.zeros(merge * merge, _IMAGE_CHANNELS * temporal * patch * patch)
        return pixel_values, grid

    def _add_vision_placeholder(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepend one blank image to an image-free sample so its rank still runs the vision tower.

        With the tower sharded over every rank, a rank whose sample holds no image would not join the
        all-gather of the tower's weights and the others would wait for it. One blank image keeps every
        rank on the same path, at the cost of a single token that is masked out of the loss.
        """
        pixel_values, grid = self._vision_placeholder()
        heads = {
            "input_ids": self._image_token_id(),
            "attention_mask": 1,
            "mm_token_type_ids": 1,
            "labels": IGNORE_INDEX,
        }
        for field, value in heads.items():
            column = sample[field]
            sample[field] = torch.cat([column.new_full((1,), value), column])
        sample["pixel_values"] = pixel_values
        sample["image_grid_thw"] = grid
        return sample

    @staticmethod
    def _normalize_messages(messages: Any, images: Any = None) -> Any:
        """Split ``<image>``/``<video>`` string placeholders into content-list parts."""
        if images is None:
            images = []
        image_queue = list(images)
        normalized = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                parts = []
                for token in re.split(r"(<image>|<video>)", content):
                    if token in ("<image>", "<video>"):
                        if not image_queue:
                            raise ValueError(
                                f"message content contains a {token} placeholder but no matching media is available"
                            )
                        media_type = "image" if token == "<image>" else "video"
                        parts.append({"type": media_type, "url": image_queue.pop(0)})
                    elif token:
                        parts.append({"type": "text", "text": token})
                content = parts
            normalized.append({"role": message.get("role"), "content": content})
        return normalized

    def _encode(self, messages: Any, *, add_generation_prompt: bool) -> dict[str, Any]:
        """Render and encode one conversation with the processor."""
        chat_template = getattr(self.processor, "chat_template", None)
        if not chat_template:
            tokenizer = getattr(self.processor, "tokenizer", None)
            chat_template = getattr(tokenizer, "chat_template", None)
        # Only templates that read enable_thinking (Qwen3 hybrid thinking) get it:
        # Transformers treats any other extra keyword as a processor argument,
        # ignores it and logs two warnings per call.
        template_kwargs = {}
        if _template_reads(chat_template, "enable_thinking"):
            template_kwargs["enable_thinking"] = False
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            **template_kwargs,
        )

    @staticmethod
    def _modal_runs(mm: torch.Tensor) -> list[tuple[int, int]]:
        """Return inclusive ``(start, end)`` positions of contiguous image-token runs."""
        is_modal = mm == 1
        runs: list[tuple[int, int]] = []
        start: Optional[int] = None
        for pos in range(int(mm.shape[0])):
            if bool(is_modal[pos]):
                if start is None:
                    start = pos
            elif start is not None:
                runs.append((start, pos - 1))
                start = None
        if start is not None:
            runs.append((start, int(mm.shape[0]) - 1))
        return runs

    def _drop_truncated_images(
            self, sample: dict[str, torch.Tensor], max_len: int
    ) -> dict[str, torch.Tensor]:
        """Drop images whose vision-token run extends past ``max_len``."""
        grid = sample.get("image_grid_thw")
        # A text-only conversation makes the processor emit no modality outputs, so there is nothing to drop.
        if grid is None or grid.numel() == 0:
            return sample

        runs = self._modal_runs(sample["mm_token_type_ids"])
        keep = 0
        for run_idx, (_, end) in enumerate(runs):
            if run_idx >= int(grid.shape[0]) or end >= max_len:
                break
            keep += 1
        if keep == int(grid.shape[0]):
            return sample

        mm = sample["mm_token_type_ids"].clone()
        input_ids = sample["input_ids"].clone()
        attention_mask = sample["attention_mask"].clone()
        labels = sample["labels"].clone()
        for start, end in runs[keep:]:
            mm[start:end + 1] = 0
            input_ids[start:end + 1] = 0
            attention_mask[start:end + 1] = 0
            labels[start:end + 1] = IGNORE_INDEX
        sample["mm_token_type_ids"] = mm
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["labels"] = labels

        pixel_rows = sum(int(g[0]) * int(g[1]) * int(g[2]) for g in grid[:keep])
        sample["pixel_values"] = sample["pixel_values"][:pixel_rows]
        sample["image_grid_thw"] = grid[:keep]
        return sample

    def _truncate(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Cut a sample longer than ``max_seq_len`` down to it, dropping the images whose run is cut."""
        max_len = self.max_seq_len
        if int(sample["input_ids"].shape[0]) <= max_len:
            return sample
        sample = self._drop_truncated_images(sample, max_len)
        for field in _SEQ_FIELDS:
            sample[field] = sample[field][:max_len]
        return sample

    def _pad_to_max_len(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extend every sequence field of a short sample up to ``max_seq_len``."""
        max_len = self.max_seq_len
        for field in _SEQ_FIELDS:
            value = sample[field]
            if value.shape[0] < max_len:
                pad_value = IGNORE_INDEX if field == "labels" else 0
                sample[field] = torch.cat([
                    value,
                    torch.full((max_len - value.shape[0],), pad_value, dtype=value.dtype),
                ])
        return sample

    def _truncate_and_pad(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Truncate to ``max_seq_len``, then pad up to it unless ``padding`` is ``"none"``."""
        sample = self._truncate(sample)
        return self._pad_to_max_len(sample) if self.padding == "max_length" else sample

    def __call__(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode one record into a padded model sample."""
        messages = self._normalize_messages(record["messages"], record.get("images"))
        full = self._encode(messages, add_generation_prompt=False)
        prompt = self._encode(messages[:-1], add_generation_prompt=True)
        prompt_len = len(prompt["input_ids"][0])

        input_ids = torch.tensor(full["input_ids"][0], dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        sample = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(full["attention_mask"][0], dtype=torch.long),
            "mm_token_type_ids": torch.tensor(full["mm_token_type_ids"][0], dtype=torch.long),
            "labels": labels,
        }
        # The processor returns no modality outputs at all for a conversation without an image, so a
        # text-only record yields a sample with no pixel_values rather than an empty one. An empty
        # image_grid_thw is not a usable stand-in: the vision tower concatenates per-image lists and
        # raises on an empty grid.
        for field in _MODAL_FIELDS:
            if field in full:
                sample[field] = full[field]
        if self.text_only == "placeholder" and "pixel_values" not in sample:
            sample = self._add_vision_placeholder(sample)
        return self._truncate_and_pad(sample)


def build_vlm_data_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 256,
        padding: str = "max_length",
        text_only: str = "keep",
        **transform_options: Any,
) -> VLMChatTransform:
    """Build the VLM sample transform.

    Args:
        processor: Qwen3-VL processor used to render and encode conversations.
        max_seq_len: Target sequence length for truncation and padding.
        padding: ``"max_length"`` pads every sample to ``max_seq_len``; ``"none"`` keeps each sample at
            its own length, which needs a collator that pads the micro-batch.
        text_only: ``"keep"`` emits a conversation without images as a sample without image fields;
            ``"placeholder"`` gives it one blank image, so every rank runs the vision tower.
        **transform_options: Reserved model-specific transform options.

    Returns:
        The configured :class:`VLMChatTransform`.

    Raises:
        ValueError: If ``processor`` is not provided, or a policy is outside :data:`PADDING_MODES`
            or :data:`TEXT_ONLY_MODES`.
    """
    del transform_options
    if processor is None:
        raise ValueError("processor is required for the VLM data transform")
    return VLMChatTransform(processor, max_seq_len=max_seq_len, padding=padding, text_only=text_only)
