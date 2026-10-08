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
"""Check that packing changes nothing but the shape, on a small model, on the host.

Packing concatenates several samples into one row. It is correct only if no document attends to
another and each reads its own rope positions, so the test is the one that matters: run every
document on its own, then run them packed, and compare the logits token by token. A packed row
without position ids is run first, because it must come out *wrong* -- if it does not, the test is
not exercising what it claims.

The model here is a few hundred thousand parameters, so this runs on a laptop in seconds and needs
no NPU and no checkpoint. What it proves is the data contract: the boundaries, the image order and
the four-row position ids. What it cannot prove is the Ascend variable-length kernel, which only the
cluster exercises.

PYTHONPATH carries the repository root, because a script's own directory is what Python puts on the
path and the editable install may point at another checkout:

    PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packing.py
    PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packing.py --tolerance 1e-5
"""

import argparse
import sys
from typing import Any, Optional

import torch

from hyper_parallel.data.vlm.packing import build_vlm_packing_collator, enable_packed_position_ids

_IMAGE_TOKEN = 900
_VOCAB = 1000


def build_model(attn_implementation: str = "eager") -> tuple[Any, Any]:
    """Build a small Qwen3-VL-MoE on the host.

    Args:
        attn_implementation: Attention backend; ``eager`` is the one that runs without an NPU.

    Returns:
        The model and its configuration.
    """
    # Heavy optional imports, only needed by this check.
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (  # pylint: disable=C0415
        Qwen3VLMoeConfig, Qwen3VLMoeTextConfig, Qwen3VLMoeVisionConfig,
    )
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (  # pylint: disable=C0415
        Qwen3VLMoeForConditionalGeneration,
    )

    # The Transformers configurations are dataclasses whose generated __init__ pylint cannot resolve,
    # so it reports every field below as an unexpected argument. They are declared fields.
    # pylint: disable=unexpected-keyword-arg
    text = Qwen3VLMoeTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=32, vocab_size=_VOCAB,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    vision = Qwen3VLMoeVisionConfig(
        hidden_size=32, num_heads=2, depth=2, out_hidden_size=32, intermediate_size=64,
        deepstack_visual_indexes=[0], patch_size=16,
    )
    config = Qwen3VLMoeConfig(text_config=text, vision_config=vision, image_token_id=_IMAGE_TOKEN)
    for holder in (config, config.text_config, config.vision_config):
        holder._attn_implementation = attn_implementation  # pylint: disable=protected-access
    torch.manual_seed(0)
    return Qwen3VLMoeForConditionalGeneration(config).eval(), config


def make_document(config: Any, text_tokens: int, grid: Optional[list[int]], seed: int) -> dict[str, Any]:
    """Build one document, with an image's placeholder run spliced into its text or without one.

    Args:
        config: The model configuration, read for the patch geometry.
        text_tokens: Text tokens of the document.
        grid: The image's ``[t, h, w]`` grid of patches, or None for a text-only document.
        seed: Seed of the token draw.

    Returns:
        One sample, shaped as the VLM transform produces them.
    """
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 800, (text_tokens,), generator=generator)
    types = torch.zeros(text_tokens, dtype=torch.long)
    sample = {"input_ids": ids, "labels": ids.clone(), "mm_token_type_ids": types}
    if grid is None:
        return sample

    vision = config.vision_config
    merge = vision.spatial_merge_size
    patch_row = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
    grid_tensor = torch.tensor([grid], dtype=torch.long)
    patches = int(grid_tensor.prod(-1).sum())
    placeholders = patches // merge ** 2
    ids = torch.cat([ids[:2], torch.full((placeholders,), _IMAGE_TOKEN), ids[2:]])
    types = torch.zeros(ids.numel(), dtype=torch.long)
    types[2:2 + placeholders] = 1
    sample.update(
        input_ids=ids, labels=ids.clone(), mm_token_type_ids=types,
        image_grid_thw=grid_tensor, pixel_values=torch.zeros(patches, patch_row),
    )
    return sample


def logits_of(model: Any, batch: dict[str, Any]) -> torch.Tensor:
    """Return the logits of one batch, which may be a single document or a packed row."""
    inputs = {name: value for name, value in batch.items() if name != "labels"}
    for name in ("input_ids", "mm_token_type_ids"):
        if name in inputs:
            inputs[name] = inputs[name].reshape(1, -1)
    with torch.no_grad():
        return model(**inputs, use_cache=False).logits[0]


def main(argv: Optional[list[str]] = None) -> int:
    """Run the check and report whether packing changed the logits.

    Args:
        argv: Command line, or None to read ``sys.argv``.

    Returns:
        0 when packing reproduces the separate runs and unpacked boundaries do not.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="largest logit difference accepted between packed and separate runs")
    args = parser.parse_args(argv)

    model, config = build_model()
    # A mix worth exercising: four images of different grids and two text-only documents, so the
    # grids must be consumed in the order their placeholders appear and a document with none must be
    # skipped over rather than handed someone else's grid.
    specifications = [(6, [1, 2, 2]), (5, None), (4, [1, 4, 4]), (3, None), (7, [1, 2, 4]), (4, [1, 4, 2])]
    documents = [make_document(config, text_tokens, grid, seed=20 + index)
                 for index, (text_tokens, grid) in enumerate(specifications)]
    print(f"{len(documents)} documents of "
          f"{[int(document['input_ids'].numel()) for document in documents]} tokens")

    separate = torch.cat([logits_of(model, document) for document in documents], dim=0)
    packed = build_vlm_packing_collator()(documents)
    print(f"packed row {tuple(packed['input_ids'].shape)}, boundaries {packed['cu_seq_lens'].tolist()}, "
          f"{int(packed['image_grid_thw'].shape[0])} image grids")

    loose = float((separate - logits_of(model, packed)).abs().max())
    handle = enable_packed_position_ids(model)
    try:
        tight = float((separate - logits_of(model, packed)).abs().max())
    finally:
        handle.remove()

    print(f"packed without position ids: max|diff| = {loose:.3e}  (must be large)")
    print(f"packed with position ids:    max|diff| = {tight:.3e}  (must be <= {args.tolerance:.0e})")

    if loose <= args.tolerance:
        print("FAIL: an unbounded packed row matched the separate runs, so this check proves nothing")
        return 1
    if tight > args.tolerance:
        print("FAIL: packing changed the logits")
        return 1
    print("PASS: packing changed the shape and nothing else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
