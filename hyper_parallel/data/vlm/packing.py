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
"""Pack several VLM samples into one row, and give the packed row its position ids.

Packing removes the padding a micro-batch of unequal samples would otherwise carry, but it is only
correct if nothing lets one document attend to another. Two things carry that information:

* ``cu_seq_lens``, the leading-zero cumulative document ends, which the attention kernel reads as its
  variable-length boundaries;
* ``position_ids``, four rows of ``[text, temporal, height, width]``. Row 0 restarts at zero on every
  document, and Transformers segments the causal mask wherever that row does not advance by one. Rows
  1 to 3 are the model's own multimodal rope, computed per document, and reach the rotary embedding.

Row 0 cannot be the temporal row: an image block holds the temporal position constant across all of
its tokens, so a mask derived from it would cut every image away from the text before it.
"""

__all__ = [
    "VLMPackingCollator",
    "build_packed_position_ids",
    "build_vlm_packing_collator",
    "enable_packed_position_ids",
]

from typing import Any, Optional

import torch

from hyper_parallel.data.constants import IGNORE_INDEX

_PACKED_FIELDS = ("input_ids", "labels", "mm_token_type_ids")
_MODAL_FIELDS = ("pixel_values", "image_grid_thw")
# [text, temporal, height, width]: the text model splits row 0 off and keeps the rest for the rope.
_POSITION_ROWS = 4


class VLMPackingCollator:
    """Concatenate the samples of one micro-batch into a single row of tokens.

    The output carries no ``attention_mask``: the row has no padding to mask, and a mask would make the
    flash-attention path rebuild its own boundaries from it and ignore ``cu_seq_lens``.

    Args:
        pad_to_length: Pad the packed row up to this length. The tail is labelled with the ignore index
            and registered as one final document, so attention metadata covers every physical token.
        pad_token_id: Fill value for the alignment tail of ``input_ids``.
        ignore_index: Label value excluded from loss computation.
    """

    def __init__(
            self,
            *,
            pad_to_length: Optional[int] = None,
            pad_token_id: int = 0,
            ignore_index: int = IGNORE_INDEX,
    ) -> None:
        """Store the packing target and the fill values."""
        if pad_to_length is not None and pad_to_length < 1:
            raise ValueError(f"pad_to_length must be a positive integer or None, but got {pad_to_length}")
        self.pad_to_length = pad_to_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def _lengths(self, samples: Any) -> list[int]:
        """Return the token count of every sample.

        Raises:
            ValueError: If a sample carries no ``input_ids``, or the row would exceed ``pad_to_length``.
        """
        lengths = []
        for sample in samples:
            input_ids = sample.get("input_ids")
            if input_ids is None:
                raise ValueError("every VLM sample must carry input_ids")
            lengths.append(int(input_ids.shape[-1]))
        total = sum(lengths)
        if self.pad_to_length is not None and total > self.pad_to_length:
            raise ValueError(
                f"{len(lengths)} samples pack to {total} tokens, over pad_to_length={self.pad_to_length}; "
                "pack fewer samples per micro-batch or raise pad_to_length"
            )
        return lengths

    def _fill_value(self, field: str) -> int:
        """Return the value the alignment tail of ``field`` carries."""
        if field == "labels":
            return self.ignore_index
        if field == "input_ids":
            return self.pad_token_id
        return 0

    def __call__(self, samples: Any) -> dict[str, Any]:
        """Pack one micro-batch of VLM samples into a single row.

        Args:
            samples: The samples of one micro-batch, each a mapping of field name to tensor.

        Returns:
            One batch holding ``input_ids``, ``labels`` and ``mm_token_type_ids`` of shape
            ``[1, packed_length]``, an int32 ``cu_seq_lens``, and the concatenated modality fields.

        Raises:
            ValueError: If ``samples`` is empty or a sample carries no ``input_ids``.
        """
        if not samples:
            raise ValueError("samples must contain at least one VLM sample")
        lengths = self._lengths(samples)
        tail = 0 if self.pad_to_length is None else self.pad_to_length - sum(lengths)

        batch: dict[str, Any] = {}
        for field in _PACKED_FIELDS:
            values = [sample[field] for sample in samples if field in sample]
            if not values:
                continue
            packed = torch.cat([value.reshape(-1) for value in values], dim=-1)
            if tail:
                packed = torch.cat([packed, packed.new_full((tail,), self._fill_value(field))], dim=-1)
            batch[field] = packed.unsqueeze(0)

        ends = list(torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist())
        if tail:
            # The tail is one more document, so every physical token sits inside a boundary; its labels
            # are the ignore index, so it adds nothing to the loss.
            ends.append(ends[-1] + tail)
        batch["cu_seq_lens"] = torch.tensor([0, *ends], dtype=torch.int32)

        # Image order must follow the order the placeholders appear in the packed row: the rope index
        # and the feature scatter both walk the grids in that order, and a mismatch is silent.
        for field in _MODAL_FIELDS:
            values = [sample[field] for sample in samples if field in sample]
            if values:
                batch[field] = torch.cat(values, dim=0)
        return batch


def build_packed_position_ids(
        model: Any,
        *,
        input_ids: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the ``[4, 1, packed_length]`` position ids of a packed row.

    Row 0 is a text ramp restarting at zero on every document, which is what the causal mask is
    segmented on. Rows 1 to 3 are the model's own ``get_rope_index`` output, computed one document at a
    time so each starts from zero, and concatenated.

    Args:
        model: The model whose ``get_rope_index`` describes multimodal positions.
        input_ids: The packed row, shaped ``[1, packed_length]``.
        cu_seq_lens: Leading-zero cumulative document ends.
        mm_token_type_ids: Modality of every packed token, shaped like ``input_ids``.
        image_grid_thw: Image grids, in the order their placeholders appear in the row.

    Returns:
        The position ids, shaped ``[4, 1, packed_length]``.

    Raises:
        ValueError: If ``cu_seq_lens`` does not start at zero and cover the row exactly.
    """
    boundaries = [int(value) for value in cu_seq_lens.tolist()]
    total = int(input_ids.shape[-1])
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError("cu_seq_lens must contain a leading zero and at least one document")
    if any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("cu_seq_lens must be strictly increasing")
    if boundaries[-1] != total:
        raise ValueError(f"cu_seq_lens must cover the packed row of {total} tokens, but ends at {boundaries[-1]}")

    flat_ids = input_ids.reshape(-1)
    flat_types = None if mm_token_type_ids is None else mm_token_type_ids.reshape(-1)
    grids_used = 0
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        length = end - start
        document_ids = flat_ids[start:end].unsqueeze(0)
        text = torch.arange(length, device=input_ids.device, dtype=input_ids.dtype).view(1, 1, -1)
        if flat_types is None or image_grid_thw is None:
            rows.append(text.expand(_POSITION_ROWS, 1, -1))
            continue
        document_types = flat_types[start:end].unsqueeze(0)
        # One grid per placeholder run; a run is contiguous, so counting runs counts the images.
        images = _count_runs(document_types.reshape(-1))
        document_grid = image_grid_thw[grids_used:grids_used + images] if images else None
        grids_used += images
        mrope, _ = model.get_rope_index(
            document_ids, document_types, image_grid_thw=document_grid, attention_mask=None,
        )
        rows.append(torch.cat([text, mrope.to(text.dtype)], dim=0))
    return torch.cat(rows, dim=-1)


def _count_runs(modality: torch.Tensor) -> int:
    """Return how many contiguous image runs ``modality`` holds, which is how many grids it consumes."""
    is_image = modality == 1
    if not bool(is_image.any()):
        return 0
    previous = torch.cat([is_image.new_zeros(1), is_image[:-1]])
    return int((is_image & ~previous).sum())


def build_vlm_packing_collator(
        *,
        pad_to_length: Optional[int] = None,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
) -> VLMPackingCollator:
    """Build the VLM packing collator.

    Args:
        pad_to_length: Pad the packed row up to this length, registering the tail as one document.
        pad_token_id: Padding value for the alignment tail of the text input IDs.
        ignore_index: Label value excluded from loss computation.

    Returns:
        A collator packing one micro-batch into a single row.
    """
    return VLMPackingCollator(
        pad_to_length=pad_to_length, pad_token_id=pad_token_id, ignore_index=ignore_index,
    )


def _rope_owner(model: Any) -> Any:
    """Return the module that carries ``get_rope_index``: the inner model, or the model itself."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "get_rope_index"):
        return inner
    if hasattr(model, "get_rope_index"):
        return model
    raise ValueError("packed position ids need a model exposing get_rope_index")


def enable_packed_position_ids(model: Any) -> Any:
    """Give every packed batch its per-document position ids on the way into ``model``.

    A packed batch carries ``cu_seq_lens`` and no ``position_ids``, and the model would otherwise build
    one continuous ramp over the whole row: every document would attend to the ones before it and read
    the wrong rope positions. The hook fills ``position_ids`` in, and leaves a batch that already has
    them, or that is not packed, exactly as it is.

    Args:
        model: The model to install the hook on.

    Returns:
        The handle of the registered hook, so a caller can remove it.
    """
    owner = _rope_owner(model)

    def _fill_position_ids(module: Any, args: Any, kwargs: Any) -> Any:
        del module
        if kwargs.get("cu_seq_lens") is None or kwargs.get("position_ids") is not None:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            return None
        kwargs["position_ids"] = build_packed_position_ids(
            owner,
            input_ids=input_ids,
            cu_seq_lens=kwargs["cu_seq_lens"],
            mm_token_type_ids=kwargs.get("mm_token_type_ids"),
            image_grid_thw=kwargs.get("image_grid_thw"),
        )
        return args, kwargs

    return model.register_forward_pre_hook(_fill_position_ids, with_kwargs=True)
