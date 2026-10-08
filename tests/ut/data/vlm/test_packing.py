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

from hyper_parallel.data.batching.build_dataloader import TextTokenBatcher
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


class TestPackingNeverSplitsASample(unittest.TestCase):
    """No document is ever cut between two rows: the invariant the whole feature rests on.

    A split sample's second half would begin a row with no prompt in front of it, and document masking
    puts the first half out of reach, so the model would be trained to answer a question it was never
    shown. Packing therefore refuses to split, and pays for it in fill rate instead.
    """

    @staticmethod
    def _runs(row: torch.Tensor) -> list[tuple[int, int]]:
        """Return the (token value, run length) pairs of a packed row, each sample having its own value."""
        values = row.reshape(-1).tolist()
        runs = []
        start = 0
        for index in range(1, len(values) + 1):
            if index == len(values) or values[index] != values[start]:
                runs.append((values[start], index - start))
                start = index
        return runs

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_collator_lays_every_sample_down_whole_and_contiguous(self):
        """Verify a packed row holds each sample's tokens unbroken and in order.

        Feature: VLM sequence packing.
        Description: Pack three samples whose tokens each carry a distinct value.
        Expectation: Three runs, of the three original lengths, in the order given.
        """
        lengths = [4, 7, 3]
        samples = [{"input_ids": torch.full((length,), 10 + index, dtype=torch.long),
                    "labels": torch.full((length,), 10 + index, dtype=torch.long)}
                   for index, length in enumerate(lengths)]

        batch = build_vlm_packing_collator()(samples)

        runs = self._runs(batch["input_ids"])
        expected = [(10, 4), (11, 7), (12, 3)]
        self.assertEqual(runs, expected, f"row is not the samples laid end to end: "
                                         f"expected={expected}, got={runs}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_boundaries_locate_every_sample_exactly(self):
        """Verify cu_seq_lens slices the row back into the samples it was built from.

        Feature: VLM sequence packing.
        Description: Pack three samples and slice the row at every boundary.
        Expectation: Each slice holds one sample's tokens and nothing else.
        """
        lengths = [4, 7, 3]
        samples = [{"input_ids": torch.full((length,), 10 + index, dtype=torch.long),
                    "labels": torch.full((length,), 10 + index, dtype=torch.long)}
                   for index, length in enumerate(lengths)]

        batch = build_vlm_packing_collator()(samples)

        row = batch["input_ids"].reshape(-1)
        bounds = batch["cu_seq_lens"].tolist()
        for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            document = row[start:end].tolist()
            expected = [10 + index] * lengths[index]
            self.assertEqual(document, expected,
                             f"document {index} mismatch: expected={expected}, got={document}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_token_budget_selection_keeps_every_sample_whole(self):
        """Verify the budget batcher defers a sample that does not fit instead of cutting it.

        Feature: VLM sequence packing.
        Description: Drive the token batcher with lengths that cannot fill the budget exactly, one of
            them larger than the whole budget, and pack whatever it hands over.
        Expectation: Every sample appears once, whole, and the over-budget one has a row to itself.
        """
        budget = 1000
        lengths = [300, 400, 450, 1700, 200, 150, 380, 900]
        batcher = TextTokenBatcher(token_budget=budget, min_buffered_samples=1)
        collate = build_vlm_packing_collator()

        rows = []
        for index, length in enumerate(lengths):
            batcher.put_item({"input_ids": torch.full((length,), 10 + index, dtype=torch.long),
                              "labels": torch.full((length,), 10 + index, dtype=torch.long)})
            while batcher.is_ready_for_micro_batch():
                rows.append(collate(batcher.get_micro_batch()))
        while not batcher.empty():
            rows.append(collate(batcher.get_micro_batch()))

        seen: dict[int, int] = {}
        for row in rows:
            for value, run in self._runs(row["input_ids"]):
                self.assertNotIn(value - 10, seen,
                                 f"sample {value - 10} appears in more than one row")
                seen[value - 10] = run
        for index, length in enumerate(lengths):
            self.assertEqual(seen.get(index), length,
                             f"sample {index} was cut or lost: expected={length}, got={seen.get(index)}")
        over_budget = [int(row["input_ids"].shape[-1]) for row in rows if int(row["input_ids"].shape[-1]) > budget]
        self.assertEqual(over_budget, [1700],
                         f"the over-budget sample did not get a row of its own: got={over_budget}")


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
