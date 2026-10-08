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
"""Build the VLM micro-batch collator."""

__all__ = ["VLMCollator", "build_vlm_collator"]

from typing import Any, Optional

import torch
from torch.utils.data import default_collate

from hyper_parallel.data.constants import IGNORE_INDEX

_TEXT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "loss_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
    "mm_token_type_ids",
}


class VLMCollator:
    """Collate text and modality fields into one VLM micro-batch.

    Text fields are padded on their last dimension to a common length and then default-collated;
    modality fields such as ``pixel_values`` and ``image_grid_thw`` are concatenated along dim 0 so
    variable-length images batch correctly. Padding here rather than in the sample transform is what
    lets a transform keep every sample at its own length: a micro-batch then costs its longest member
    instead of a fixed constant.

    Args:
        pad_token_id: Fill value for ``input_ids``.
        ignore_index: Fill value for ``labels``, excluded from the loss.
        pad_to_length: Pad to this fixed length instead of the longest sample of the micro-batch.
    """

    def __init__(
            self,
            *,
            pad_token_id: int = 0,
            ignore_index: int = IGNORE_INDEX,
            pad_to_length: Optional[int] = None,
    ) -> None:
        """Store the fill values and the padding target."""
        if pad_to_length is not None and pad_to_length < 1:
            raise ValueError(f"pad_to_length must be a positive integer or None, but got {pad_to_length}")
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.pad_to_length = pad_to_length

    def _fill_value(self, field: str) -> int:
        """Return the value that pads ``field``: a pad token, an ignored label, or an inactive mask."""
        if field == "labels":
            return self.ignore_index
        if field == "input_ids":
            return self.pad_token_id
        # attention_mask, loss_mask, mm_token_type_ids and the position ids all read 0 as "not a token".
        return 0

    def _target_length(self, samples: Any) -> int:
        """Return the length every text field is padded to.

        Raises:
            ValueError: If a sample carries no ``input_ids``, or is longer than ``pad_to_length``.
        """
        lengths = []
        for sample in samples:
            input_ids = sample.get("input_ids")
            if input_ids is None:
                raise ValueError("every VLM sample must carry input_ids")
            lengths.append(int(input_ids.shape[-1]))
        longest = max(lengths)
        if self.pad_to_length is None:
            return longest
        if longest > self.pad_to_length:
            raise ValueError(
                f"a sample of {longest} tokens does not fit pad_to_length={self.pad_to_length}; "
                "lower the transform's max_seq_len or raise pad_to_length"
            )
        return self.pad_to_length

    def _pad_text_fields(self, sample: Any, target: int) -> dict[str, Any]:
        """Return ``sample`` with every text tensor extended on its last dimension to ``target``."""
        padded = {}
        for field, value in sample.items():
            if field not in _TEXT_FIELDS or not isinstance(value, torch.Tensor) or value.shape[-1] >= target:
                padded[field] = value
                continue
            # The sequence is the last dimension for the 1-D fields and for the 3-D mrope position ids
            # alike, so one rule covers both.
            shape = list(value.shape)
            shape[-1] = target - value.shape[-1]
            padding = value.new_full(shape, self._fill_value(field))
            padded[field] = torch.cat([value, padding], dim=-1)
        return padded

    def __call__(self, samples: Any) -> dict[str, Any]:
        """Collate one micro-batch of VLM samples.

        Args:
            samples: The samples of one micro-batch, each a mapping of field name to tensor.

        Returns:
            One batch dictionary, text fields stacked and modality fields concatenated.

        Raises:
            ValueError: If ``samples`` is empty, or a sample carries no ``input_ids``.
        """
        if not samples:
            raise ValueError("samples must contain at least one VLM sample")
        target = self._target_length(samples)
        samples = [self._pad_text_fields(sample, target) for sample in samples]

        text_samples = [
            {field: value for field, value in sample.items() if field in _TEXT_FIELDS}
            for sample in samples
        ]
        modal_samples = [
            {field: value for field, value in sample.items() if field not in _TEXT_FIELDS}
            for sample in samples
        ]

        batch = default_collate(text_samples)
        if any(modal_samples):
            for field in {field for sample in modal_samples for field in sample}:
                values = [sample[field] for sample in modal_samples if field in sample]
                batch[field] = (
                    torch.cat(values, dim=0)
                    if isinstance(values[0], torch.Tensor)
                    else default_collate(values)
                )
        return batch


def build_vlm_collator(
        *,
        packing: bool = False,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
        pad_to_length: Optional[int] = None,
) -> VLMCollator:
    """Build the VLM micro-batch collator.

    Args:
        packing: Reserved switch for VeOmni-style text packing.
        pad_token_id: Padding value for text input IDs.
        ignore_index: Label value excluded from loss computation.
        pad_to_length: Pad to this fixed length instead of the micro-batch's longest sample.

    Returns:
        A collator producing one VLM micro-batch dictionary.

    Raises:
        NotImplementedError: If ``packing`` is requested.
    """
    if packing:
        raise NotImplementedError("The temporary VLM collator does not support packing")
    return VLMCollator(pad_token_id=pad_token_id, ignore_index=ignore_index, pad_to_length=pad_to_length)
