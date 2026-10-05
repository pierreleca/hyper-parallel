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
"""A Qwen3-VL sample transform that keeps each sample at its own length.

``hyper_parallel.data.vlm.build_vlm_data_transform`` pads every sample to
``max_seq_len``, so every rank runs the same text shape whatever the sample
holds, and the pad positions are routed through the experts like real tokens.
That hides the data heterogeneity the experiments of this directory are about.
This transform truncates but does not pad: a sample keeps the length its
conversation has. It needs ``micro_batch_size: 1``, since the collator stacks
the text fields of one micro-batch.

A sample that would lose every image to the truncation is refused instead of
silently turned into a text-only one: with the vision tower sharded over all
ranks, a rank that skipped the tower would leave the others waiting for its
share of the all-gather.
"""

from __future__ import annotations

from typing import Any

from hyper_parallel.data.vlm.build_data_transform import VLMChatTransform

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")


class VariableLengthVLMChatTransform(VLMChatTransform):
    """Encode one conversation into one sample, truncated to ``max_seq_len`` and never padded."""

    def _truncate_and_pad(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Truncate (dropping cut images) to ``max_seq_len``; leave shorter samples as they are."""
        seq_len = int(sample["input_ids"].shape[0])
        if seq_len <= self.max_seq_len:
            return sample
        images_before = int(sample["image_grid_thw"].shape[0])
        sample = self._drop_truncated_images(sample, self.max_seq_len)
        if images_before and int(sample["image_grid_thw"].shape[0]) == 0:
            raise ValueError(
                f"a sample of {seq_len} tokens would lose all {images_before} of its images at "
                f"max_seq_len={self.max_seq_len}; raise max_seq_len or build shorter samples"
            )
        for field in _SEQ_FIELDS:
            sample[field] = sample[field][: self.max_seq_len]
        return sample


def build_variable_length_vlm_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 16384,
        **transform_options: Any,
) -> VariableLengthVLMChatTransform:
    """Build the transform; the signature of ``build_vlm_data_transform``.

    Args:
        processor: Qwen3-VL processor used to render and encode conversations.
        max_seq_len: Longest sample kept; longer ones are truncated.
        **transform_options: Reserved model-specific transform options.

    Returns:
        The configured :class:`VariableLengthVLMChatTransform`.

    Raises:
        ValueError: If ``processor`` is not provided.
    """
    del transform_options
    if processor is None:
        raise ValueError("processor is required for the VLM data transform")
    return VariableLengthVLMChatTransform(processor, max_seq_len=max_seq_len)


__all__ = ["VariableLengthVLMChatTransform", "build_variable_length_vlm_transform"]
