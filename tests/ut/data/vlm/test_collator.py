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
"""Unit tests for the VLM micro-batch collator.

The collator is what lets a sample transform keep every sample at its own length: it pads a
micro-batch to its longest member instead of to a fixed constant. These tests import the collator
module directly and need no processor, so they run on a plain CPU environment.
"""

import unittest

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.vlm.collator import VLMCollator, build_vlm_collator
from tests.common.mark_utils import arg_mark

_TEXT_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")


def _sample(length, images=1):
    """Build one VLM sample of ``length`` tokens carrying ``images`` images."""
    sample = {
        "input_ids": torch.arange(1, length + 1, dtype=torch.long),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "labels": torch.arange(1, length + 1, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(length, dtype=torch.long),
    }
    if images:
        sample["pixel_values"] = torch.ones(4 * images, 3)
        sample["image_grid_thw"] = torch.ones(images, 3, dtype=torch.long)
    return sample


class TestVlmCollatorPadding(unittest.TestCase):
    """The collator pads a micro-batch to a common length before stacking it."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mixed_lengths_pad_to_the_longest_sample(self):
        """Verify a micro-batch of unequal samples collates to the longest length.

        Feature: VLM micro-batch padding.
        Description: Collate a five-token and a three-token sample with the default collator.
        Expectation: Every text field is stacked at length five.
        """
        batch = build_vlm_collator()([_sample(5), _sample(3)])

        for field in _TEXT_FIELDS:
            self.assertEqual(tuple(batch[field].shape), (2, 5),
                             f"{field} shape mismatch: expected=(2, 5), got={tuple(batch[field].shape)}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_padding_uses_the_fill_value_of_each_field(self):
        """Verify each text field pads with the value that marks "not a token".

        Feature: VLM micro-batch padding.
        Description: Collate a three-token sample behind a five-token one and read the padded tail.
        Expectation: input_ids pad with the pad token, labels with the ignore index, masks with zero.
        """
        batch = build_vlm_collator(pad_token_id=7)([_sample(5), _sample(3)])

        expected = {"input_ids": [7, 7], "labels": [IGNORE_INDEX, IGNORE_INDEX],
                    "attention_mask": [0, 0], "mm_token_type_ids": [0, 0]}
        for field, tail in expected.items():
            got = batch[field][1, 3:].tolist()
            self.assertEqual(got, tail, f"{field} padded tail mismatch: expected={tail}, got={got}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_equal_lengths_are_left_untouched(self):
        """Verify a micro-batch that needs no padding is collated unchanged.

        Feature: VLM micro-batch padding.
        Description: Collate two samples of the same length.
        Expectation: The stacked input IDs are the two originals, with nothing appended.
        """
        batch = build_vlm_collator()([_sample(4), _sample(4)])

        expected = [[1, 2, 3, 4], [1, 2, 3, 4]]
        got = batch["input_ids"].tolist()
        self.assertEqual(got, expected, f"equal-length batch altered: expected={expected}, got={got}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pad_to_length_overrides_the_longest_sample(self):
        """Verify a fixed padding target is honoured above the micro-batch's own longest sample.

        Feature: VLM micro-batch padding.
        Description: Collate a five-token and a three-token sample with pad_to_length of eight.
        Expectation: Both rows come out at length eight.
        """
        batch = build_vlm_collator(pad_to_length=8)([_sample(5), _sample(3)])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8),
                         f"pad_to_length ignored: expected=(2, 8), got={tuple(batch['input_ids'].shape)}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sample_longer_than_pad_to_length_is_refused(self):
        """Verify a sample that does not fit the fixed target raises instead of being truncated.

        Feature: VLM micro-batch padding.
        Description: Collate a five-token sample with pad_to_length of four.
        Expectation: ValueError naming both lengths.
        """
        with self.assertRaises(ValueError) as caught:
            build_vlm_collator(pad_to_length=4)([_sample(5)])

        message = str(caught.exception)
        self.assertIn("5", message, f"error does not name the sample length: {message}")
        self.assertIn("4", message, f"error does not name the target length: {message}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_modality_fields_concatenate_across_the_micro_batch(self):
        """Verify image fields are concatenated on dim 0 rather than stacked.

        Feature: VLM micro-batch padding.
        Description: Collate two samples carrying one and two images.
        Expectation: image_grid_thw holds three rows and pixel_values twelve.
        """
        batch = build_vlm_collator()([_sample(5, images=1), _sample(3, images=2)])

        self.assertEqual(tuple(batch["image_grid_thw"].shape), (3, 3),
                         f"grid rows mismatch: expected=(3, 3), got={tuple(batch['image_grid_thw'].shape)}")
        self.assertEqual(tuple(batch["pixel_values"].shape), (12, 3),
                         f"pixel rows mismatch: expected=(12, 3), got={tuple(batch['pixel_values'].shape)}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_text_only_samples_collate_without_modality_fields(self):
        """Verify a micro-batch of image-free samples carries no image keys.

        Feature: VLM text-only samples.
        Description: Collate two samples built with no images.
        Expectation: The batch holds the text fields and neither pixel_values nor image_grid_thw.
        """
        batch = build_vlm_collator()([_sample(5, images=0), _sample(3, images=0)])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5),
                         f"text shape mismatch: expected=(2, 5), got={tuple(batch['input_ids'].shape)}")
        for field in ("pixel_values", "image_grid_thw"):
            self.assertNotIn(field, batch, f"image-free batch carries {field}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_multi_row_position_ids_pad_on_the_sequence_axis(self):
        """Verify a field whose sequence is its last axis pads on that axis.

        Feature: VLM micro-batch padding.
        Description: Collate samples carrying four-row position IDs of unequal sequence length.
        Expectation: The row count is preserved and only the sequence axis grows.
        """
        first, second = _sample(5, images=0), _sample(3, images=0)
        first["position_ids"] = torch.ones(4, 5, dtype=torch.long)
        second["position_ids"] = torch.ones(4, 3, dtype=torch.long)

        batch = build_vlm_collator()([first, second])

        self.assertEqual(tuple(batch["position_ids"].shape), (2, 4, 5),
                         f"position shape mismatch: expected=(2, 4, 5), got={tuple(batch['position_ids'].shape)}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_inputs_are_refused(self):
        """Verify the collator refuses an empty micro-batch, a sample with no IDs and a bad target.

        Feature: VLM micro-batch padding.
        Description: Call the collator with no samples, with a sample lacking input_ids, and build
            one with a non-positive pad_to_length.
        Expectation: ValueError in all three cases.
        """
        collator = build_vlm_collator()
        with self.assertRaises(ValueError):
            collator([])
        with self.assertRaises(ValueError):
            collator([{"labels": torch.arange(3)}])
        with self.assertRaises(ValueError):
            VLMCollator(pad_to_length=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_packing_switch_selects_the_packing_collator(self):
        """Verify the switch returns the collator that concatenates instead of padding.

        Feature: VLM micro-batch padding.
        Description: Build the collator with packing enabled and collate two samples.
        Expectation: One packed row carrying the document boundaries.
        """
        batch = build_vlm_collator(packing=True)([_sample(5, images=0), _sample(3, images=0)])

        self.assertEqual(tuple(batch["input_ids"].shape), (1, 8),
                         f"not packed: expected=(1, 8), got={tuple(batch['input_ids'].shape)}")
        got = batch["cu_seq_lens"].tolist()
        self.assertEqual(got, [0, 5, 8], f"boundaries mismatch: expected=[0, 5, 8], got={got}")


if __name__ == "__main__":
    unittest.main()
