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
"""Unit tests for VLM sequence packing and the position ids of a packed row.

A packed row is only correct if nothing lets one document attend to another, which rests on two
things: the ``cu_seq_lens`` boundaries, and position ids whose text row restarts on every document.
These tests check both against a stubbed rope index and need no model.

``examples/qwen3_vl_30b_perf/check_packing.py`` proves the numbers on a small real model.
"""

import unittest
from typing import Any

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.vlm.packing import (
    build_packed_position_ids,
    build_vlm_packing_collator,
)
from tests.common.mark_utils import arg_mark


def _sample(length, images=0, image_at=1):
    """Build one sample of ``length`` tokens, optionally with one image run of two tokens."""
    sample = {
        "input_ids": torch.arange(1, length + 1, dtype=torch.long),
        "labels": torch.arange(1, length + 1, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(length, dtype=torch.long),
    }
    if images:
        sample["mm_token_type_ids"][image_at:image_at + 2] = 1
        sample["pixel_values"] = torch.ones(4 * images, 3)
        sample["image_grid_thw"] = torch.ones(images, 3, dtype=torch.long)
    return sample


class _StubRopeModel:
    """Model stand-in whose rope index is a plain ramp, so concatenation is easy to read."""

    def __init__(self) -> None:
        """Record the grids every call received, in order."""
        self.grids: list[Any] = []

    def get_rope_index(self, input_ids: Any, mm_token_type_ids: Any, image_grid_thw: Any = None,
                       attention_mask: Any = None) -> tuple[torch.Tensor, None]:
        """Return a three-row ramp over the document, and remember the grid it was given."""
        del mm_token_type_ids, attention_mask
        self.grids.append(None if image_grid_thw is None else image_grid_thw.tolist())
        length = int(input_ids.shape[-1])
        ramp = torch.arange(length, dtype=torch.long).view(1, 1, -1)
        return ramp.expand(3, 1, -1).contiguous(), None


class TestVlmPackingCollator(unittest.TestCase):
    """The collator concatenates a micro-batch into one row and records its boundaries."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_samples_pack_into_one_row_with_their_boundaries(self):
        """Verify the packed row is the concatenation and cu_seq_lens marks each document's end.

        Feature: VLM sequence packing.
        Description: Pack samples of six, four and five tokens.
        Expectation: One row of fifteen tokens and a leading-zero cumulative boundary list.
        """
        batch = build_vlm_packing_collator()([_sample(6), _sample(4), _sample(5)])

        self.assertEqual(tuple(batch["input_ids"].shape), (1, 15),
                         f"row shape mismatch: expected=(1, 15), got={tuple(batch['input_ids'].shape)}")
        got = batch["cu_seq_lens"].tolist()
        self.assertEqual(got, [0, 6, 10, 15], f"boundaries mismatch: expected=[0, 6, 10, 15], got={got}")
        self.assertEqual(batch["cu_seq_lens"].dtype, torch.int32,
                         f"boundaries dtype mismatch: expected=int32, got={batch['cu_seq_lens'].dtype}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_packed_row_carries_no_attention_mask(self):
        """Verify no mask is emitted, since one would override the boundaries downstream.

        Feature: VLM sequence packing.
        Description: Pack two samples and inspect the keys.
        Expectation: No attention_mask key.
        """
        batch = build_vlm_packing_collator()([_sample(6), _sample(4)])

        self.assertNotIn("attention_mask", batch, "a packed row carries an attention mask")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_modality_fields_keep_the_order_of_the_row(self):
        """Verify image grids are concatenated in the order their placeholders appear.

        Feature: VLM sequence packing.
        Description: Pack an image sample, a text-only sample and a two-image sample.
        Expectation: Three grid rows, from the first and third samples only.
        """
        batch = build_vlm_packing_collator()([_sample(6, images=1), _sample(4), _sample(5, images=2)])

        self.assertEqual(int(batch["image_grid_thw"].shape[0]), 3,
                         f"grid rows mismatch: expected=3, got={int(batch['image_grid_thw'].shape[0])}")
        self.assertEqual(int(batch["pixel_values"].shape[0]), 12,
                         f"pixel rows mismatch: expected=12, got={int(batch['pixel_values'].shape[0])}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_alignment_tail_is_one_more_document(self):
        """Verify padding to a fixed length registers the tail so every token sits in a boundary.

        Feature: VLM sequence packing.
        Description: Pack six and four tokens into a row of twelve.
        Expectation: A final boundary at twelve, with the tail labelled out of the loss.
        """
        batch = build_vlm_packing_collator(pad_to_length=12)([_sample(6), _sample(4)])

        got = batch["cu_seq_lens"].tolist()
        self.assertEqual(got, [0, 6, 10, 12], f"boundaries mismatch: expected=[0, 6, 10, 12], got={got}")
        tail = batch["labels"][0, 10:].tolist()
        self.assertEqual(tail, [IGNORE_INDEX] * 2,
                         f"tail is trained on: expected={[IGNORE_INDEX] * 2}, got={tail}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_row_that_does_not_fit_is_refused(self):
        """Verify packing more than the target length raises instead of truncating.

        Feature: VLM sequence packing.
        Description: Pack six and four tokens into a row of eight.
        Expectation: ValueError naming both the total and the target.
        """
        with self.assertRaises(ValueError) as caught:
            build_vlm_packing_collator(pad_to_length=8)([_sample(6), _sample(4)])

        message = str(caught.exception)
        self.assertIn("10", message, f"error does not name the total: {message}")
        self.assertIn("8", message, f"error does not name the target: {message}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_inputs_are_refused(self):
        """Verify an empty micro-batch and a sample without IDs are both refused.

        Feature: VLM sequence packing.
        Description: Call the collator with no samples and with a sample lacking input_ids.
        Expectation: ValueError in both cases.
        """
        collator = build_vlm_packing_collator()
        with self.assertRaises(ValueError):
            collator([])
        with self.assertRaises(ValueError):
            collator([{"labels": torch.arange(3)}])


class TestPackedPositionIds(unittest.TestCase):
    """Position ids of a packed row restart on every document."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_text_row_restarts_on_every_document(self):
        """Verify row zero is a ramp per document, which is what the causal mask is segmented on.

        Feature: Packed position ids.
        Description: Build the ids of a row packed from three and two tokens.
        Expectation: Four rows, and a text row that restarts at zero on the second document.
        """
        batch = build_vlm_packing_collator()([_sample(3), _sample(2)])

        position_ids = build_packed_position_ids(
            _StubRopeModel(), input_ids=batch["input_ids"], cu_seq_lens=batch["cu_seq_lens"],
            mm_token_type_ids=batch["mm_token_type_ids"], image_grid_thw=None,
        )

        self.assertEqual(tuple(position_ids.shape), (4, 1, 5),
                         f"shape mismatch: expected=(4, 1, 5), got={tuple(position_ids.shape)}")
        got = position_ids[0, 0].tolist()
        self.assertEqual(got, [0, 1, 2, 0, 1], f"text row mismatch: expected=[0, 1, 2, 0, 1], got={got}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_each_document_is_given_only_its_own_image_grids(self):
        """Verify the grids are handed to the rope index document by document, in order.

        Feature: Packed position ids.
        Description: Pack a one-image sample, a text-only sample and a two-image sample.
        Expectation: The rope index sees one grid, then none, then two.
        """
        batch = build_vlm_packing_collator()([_sample(6, images=1), _sample(4), _sample(5, images=2)])
        model = _StubRopeModel()

        build_packed_position_ids(
            model, input_ids=batch["input_ids"], cu_seq_lens=batch["cu_seq_lens"],
            mm_token_type_ids=batch["mm_token_type_ids"], image_grid_thw=batch["image_grid_thw"],
        )

        counts = [0 if grid is None else len(grid) for grid in model.grids]
        self.assertEqual(counts, [1, 0, 1], f"grid hand-out mismatch: expected=[1, 0, 1], got={counts}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_boundaries_that_do_not_describe_the_row_are_refused(self):
        """Verify a boundary list that is malformed or does not cover the row raises.

        Feature: Packed position ids.
        Description: Build ids with boundaries missing the leading zero, not increasing, and short.
        Expectation: ValueError in all three cases.
        """
        row = torch.arange(5, dtype=torch.long).view(1, -1)
        model = _StubRopeModel()
        for boundaries in ([1, 5], [0, 3, 3], [0, 3]):
            with self.assertRaises(ValueError):
                build_packed_position_ids(
                    model, input_ids=row,
                    cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32),
                )


if __name__ == "__main__":
    unittest.main()
