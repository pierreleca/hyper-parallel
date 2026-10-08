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
"""Unit tests for the VLM sample transform's truncation and padding policy.

``padding="max_length"`` is the historical behaviour and stays the default; ``padding="none"`` keeps
each sample at its own length, which is what makes a rank's work depend on its sample. A stub
processor stands in for the real one, so these tests need no checkpoint and no network.
"""

import unittest
from typing import Any

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.vlm.build_data_transform import (
    PADDING_MODES,
    TEXT_ONLY_MODES,
    VLMChatTransform,
    build_vlm_data_transform,
)
from tests.common.mark_utils import arg_mark

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")


class _StubImageProcessor:
    """Patch geometry of the Qwen3-VL image processor, which the placeholder is sized from."""

    merge_size = 2
    patch_size = 16
    temporal_patch_size = 2


class _StubProcessor:
    """Processor stand-in: encodes four tokens per message, with or without modality outputs."""

    chat_template = "{{ messages }}"
    image_token_id = 151655

    def __init__(self, *, with_images: bool = True) -> None:
        """Record whether the encodings carry image outputs."""
        self.with_images = with_images
        self.image_processor = _StubImageProcessor()

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a deterministic encoding of ``messages``, ignoring every template option."""
        del kwargs
        length = 4 * len(messages)
        encoded = {
            "input_ids": [list(range(1, length + 1))],
            "attention_mask": [[1] * length],
            "mm_token_type_ids": [[0] * length],
        }
        if self.with_images:
            encoded["mm_token_type_ids"] = [[1, 1] + [0] * (length - 2)]
            encoded["pixel_values"] = torch.ones(4, 3)
            encoded["image_grid_thw"] = torch.tensor([[1, 2, 2]], dtype=torch.long)
        return encoded


def _sample(length, images=1):
    """Build one encoded sample of ``length`` tokens whose first two tokens belong to an image."""
    sample = {
        "input_ids": torch.arange(1, length + 1, dtype=torch.long),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "labels": torch.arange(1, length + 1, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(length, dtype=torch.long),
    }
    if images:
        sample["mm_token_type_ids"][:2] = 1
        sample["pixel_values"] = torch.ones(4 * images, 3)
        sample["image_grid_thw"] = torch.ones(images, 3, dtype=torch.long)
    return sample


class TestVlmTransformPadding(unittest.TestCase):
    """The transform truncates always and pads only when asked to."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_max_length_padding_is_the_default(self):
        """Verify the historical behaviour is unchanged when padding is not configured.

        Feature: VLM sample padding policy.
        Description: Run a six-token sample through a transform of max_seq_len ten.
        Expectation: Every sequence field comes out at ten, labels padded with the ignore index.
        """
        transform = VLMChatTransform(_StubProcessor(), max_seq_len=10)

        padded = transform._truncate_and_pad(_sample(6))

        for field in _SEQ_FIELDS:
            self.assertEqual(int(padded[field].shape[0]), 10,
                             f"{field} length mismatch: expected=10, got={int(padded[field].shape[0])}")
        tail = padded["labels"][6:].tolist()
        self.assertEqual(tail, [IGNORE_INDEX] * 4,
                         f"label padding mismatch: expected={[IGNORE_INDEX] * 4}, got={tail}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_padding_none_keeps_the_sample_at_its_own_length(self):
        """Verify a short sample is left alone when padding is disabled.

        Feature: VLM sample padding policy.
        Description: Run a six-token sample through a transform of max_seq_len ten, padding none.
        Expectation: Every sequence field stays at six.
        """
        transform = VLMChatTransform(_StubProcessor(), max_seq_len=10, padding="none")

        kept = transform._truncate_and_pad(_sample(6))

        for field in _SEQ_FIELDS:
            self.assertEqual(int(kept[field].shape[0]), 6,
                             f"{field} length mismatch: expected=6, got={int(kept[field].shape[0])}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_truncation_applies_under_both_policies(self):
        """Verify max_seq_len stays a hard ceiling whether or not padding is on.

        Feature: VLM sample padding policy.
        Description: Run a twelve-token sample through a transform of max_seq_len eight, both modes.
        Expectation: Both come out at exactly eight.
        """
        for padding in PADDING_MODES:
            transform = VLMChatTransform(_StubProcessor(), max_seq_len=8, padding=padding)

            cut = transform._truncate_and_pad(_sample(12))

            got = int(cut["input_ids"].shape[0])
            self.assertEqual(got, 8, f"padding={padding} ceiling mismatch: expected=8, got={got}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_an_unknown_padding_mode_is_refused(self):
        """Verify a misspelled padding policy fails at construction rather than silently.

        Feature: VLM sample padding policy.
        Description: Construct the transform and the builder with an unsupported padding value.
        Expectation: ValueError naming the offending value, from both entry points.
        """
        with self.assertRaises(ValueError) as caught:
            VLMChatTransform(_StubProcessor(), padding="longest")
        self.assertIn("longest", str(caught.exception),
                      f"error does not name the value: {caught.exception}")

        with self.assertRaises(ValueError):
            build_vlm_data_transform(processor=_StubProcessor(), padding="longest")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_builder_forwards_the_padding_policy(self):
        """Verify the configured policy reaches the transform the builder returns.

        Feature: VLM sample padding policy.
        Description: Build a transform with padding none.
        Expectation: The returned transform reports that policy.
        """
        transform = build_vlm_data_transform(processor=_StubProcessor(), max_seq_len=32, padding="none")

        self.assertEqual(transform.padding, "none",
                         f"padding not forwarded: expected=none, got={transform.padding}")


class TestVlmTransformTextOnly(unittest.TestCase):
    """A conversation with no image yields a sample with no modality fields."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_text_only_record_omits_the_modality_fields(self):
        """Verify an image-free conversation encodes without pixel_values.

        Feature: VLM text-only samples.
        Description: Transform a record whose processor returns no modality outputs.
        Expectation: The sample carries the four text fields and neither image field.
        """
        transform = VLMChatTransform(_StubProcessor(with_images=False), max_seq_len=16, padding="none")
        record = {"messages": [{"role": "user", "content": "hello"},
                               {"role": "assistant", "content": "hi"}]}

        sample = transform(record)

        for field in _SEQ_FIELDS:
            self.assertIn(field, sample, f"text-only sample lacks {field}")
        for field in ("pixel_values", "image_grid_thw"):
            self.assertNotIn(field, sample, f"text-only sample carries {field}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_an_image_record_still_carries_the_modality_fields(self):
        """Verify the image path is untouched by the text-only tolerance.

        Feature: VLM text-only samples.
        Description: Transform a record whose processor returns modality outputs.
        Expectation: Both image fields are present.
        """
        transform = VLMChatTransform(_StubProcessor(with_images=True), max_seq_len=16, padding="none")
        record = {"messages": [{"role": "user", "content": "describe"},
                               {"role": "assistant", "content": "a cat"}]}

        sample = transform(record)

        for field in ("pixel_values", "image_grid_thw"):
            self.assertIn(field, sample, f"image sample lacks {field}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_truncating_a_sample_without_images_does_not_fail(self):
        """Verify the image-dropping step tolerates a sample with no image grid.

        Feature: VLM text-only samples.
        Description: Truncate a twelve-token image-free sample to eight tokens.
        Expectation: The sample is cut to eight with no error raised.
        """
        transform = VLMChatTransform(_StubProcessor(with_images=False), max_seq_len=8, padding="none")

        cut = transform._truncate_and_pad(_sample(12, images=0))

        got = int(cut["input_ids"].shape[0])
        self.assertEqual(got, 8, f"image-free truncation mismatch: expected=8, got={got}")


class TestVlmTransformVisionPlaceholder(unittest.TestCase):
    """``text_only="placeholder"`` gives an image-free sample one blank image."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_keep_is_the_default_policy(self):
        """Verify a text-only sample stays image-free unless a placeholder is asked for.

        Feature: VLM text-only placeholder.
        Description: Transform an image-free record with the default policy.
        Expectation: No image fields, and the policy reads "keep".
        """
        transform = VLMChatTransform(_StubProcessor(with_images=False), max_seq_len=64)

        sample = transform({"messages": [{"role": "user", "content": "a"},
                                         {"role": "assistant", "content": "b"}]})

        self.assertEqual(transform.text_only, "keep",
                         f"default policy mismatch: expected=keep, got={transform.text_only}")
        self.assertNotIn("pixel_values", sample, "default policy added an image")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_placeholder_is_one_merged_token_of_blank_patches(self):
        """Verify the inserted image is the smallest the vision tower accepts.

        Feature: VLM text-only placeholder.
        Description: Transform an image-free record with the placeholder policy.
        Expectation: A one-block grid, merge-squared pixel rows, each of the patch volume.
        """
        transform = VLMChatTransform(_StubProcessor(with_images=False), max_seq_len=64,
                                     padding="none", text_only="placeholder")

        sample = transform({"messages": [{"role": "user", "content": "a"},
                                         {"role": "assistant", "content": "b"}]})

        grid = sample["image_grid_thw"].tolist()
        self.assertEqual(grid, [[1, 2, 2]], f"grid mismatch: expected=[[1, 2, 2]], got={grid}")
        # 3 channels x 2 temporal patches x 16 x 16 pixels is one row of the tower's input.
        self.assertEqual(tuple(sample["pixel_values"].shape), (4, 1536),
                         f"pixel shape mismatch: expected=(4, 1536), got={tuple(sample['pixel_values'].shape)}")
        self.assertEqual(float(sample["pixel_values"].abs().sum()), 0.0,
                         f"placeholder is not blank: sum={float(sample['pixel_values'].abs().sum())}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_placeholder_token_matches_the_feature_row_count(self):
        """Verify exactly one image token is inserted, as the model's consistency check requires.

        Feature: VLM text-only placeholder.
        Description: Transform an image-free record and count the image tokens and the merged tokens.
        Expectation: One image token, marked as an image, and excluded from the loss.
        """
        transform = VLMChatTransform(_StubProcessor(with_images=False), max_seq_len=64,
                                     padding="none", text_only="placeholder")

        sample = transform({"messages": [{"role": "user", "content": "a"},
                                         {"role": "assistant", "content": "b"}]})

        grid = sample["image_grid_thw"]
        merged_tokens = int(grid.prod(-1).sum()) // _StubImageProcessor.merge_size ** 2
        image_tokens = int((sample["input_ids"] == _StubProcessor.image_token_id).sum())
        self.assertEqual(image_tokens, merged_tokens,
                         f"token count mismatch: expected={merged_tokens}, got={image_tokens}")
        self.assertEqual(int(sample["mm_token_type_ids"][0]), 1,
                         f"placeholder not marked as an image: got={int(sample['mm_token_type_ids'][0])}")
        self.assertEqual(int(sample["labels"][0]), IGNORE_INDEX,
                         f"placeholder is trained on: label={int(sample['labels'][0])}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_record_with_images_gets_no_placeholder(self):
        """Verify the placeholder policy leaves image-bearing records alone.

        Feature: VLM text-only placeholder.
        Description: Transform a record whose processor returns one image.
        Expectation: The processor's own grid survives, with no extra blank image.
        """
        transform = VLMChatTransform(_StubProcessor(with_images=True), max_seq_len=64,
                                     padding="none", text_only="placeholder")

        sample = transform({"messages": [{"role": "user", "content": "a"},
                                         {"role": "assistant", "content": "b"}]})

        grid = sample["image_grid_thw"].tolist()
        self.assertEqual(grid, [[1, 2, 2]], f"grid mismatch: expected=[[1, 2, 2]], got={grid}")
        self.assertEqual(int(sample["image_grid_thw"].shape[0]), 1,
                         f"extra image added: rows={int(sample['image_grid_thw'].shape[0])}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_an_unknown_text_only_mode_is_refused(self):
        """Verify a misspelled text-only policy fails at construction.

        Feature: VLM text-only placeholder.
        Description: Construct the transform with an unsupported text_only value.
        Expectation: ValueError naming the value, and the supported set has two members.
        """
        with self.assertRaises(ValueError) as caught:
            VLMChatTransform(_StubProcessor(), text_only="dummy")

        self.assertIn("dummy", str(caught.exception), f"error does not name the value: {caught.exception}")
        self.assertEqual(len(TEXT_ONLY_MODES), 2,
                         f"text-only modes changed: got={TEXT_ONLY_MODES}")


if __name__ == "__main__":
    unittest.main()
