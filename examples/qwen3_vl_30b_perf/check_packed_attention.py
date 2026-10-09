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
"""Check that the Ascend kernel keeps the documents of a packed row apart, on one die, with no model.

``check_packing.py`` proves the data contract, and it proves it through Transformers' mask: a packed
batch carries per-document position ids, Transformers reads the restarts in them and builds a
block-diagonal mask, and the eager kernel obeys it. On the device that mask does not exist. With
``attn_implementation: flash_attention_2`` Transformers hands the attention ``attention_mask=None``
and the kernel is expected to find the boundaries itself, from ``cu_seq_lens``. Isolation therefore
rests on a different mechanism on the device than the one the host check exercises, and on a run of
one document per row the two are indistinguishable: a single segment is a plain causal row either
way.

This check exercises the device mechanism alone, at the level of the operator and nothing above it.
It calls the production wrapper -- the one ``run_qwen3_moe_flash_attention`` routes a packed batch to
-- once per document without boundaries, then once on the concatenated row with them, and compares.
A document's output must not depend on what was packed beside it.

Three legs and one control:

- **separate:** each document alone, the wrapper's dense path (BNSD, compressed causal mask);
- **packed:** the documents concatenated into one row, the wrapper's variable-length path (TND,
  ``sparse_mode`` 3, ``actual_seq_qlen`` from the boundaries). It must match ``separate``;
- **reference:** the same attention in float32 with an explicit block-diagonal causal mask, in plain
  Torch. A kernel that is wrong in the same way on both paths would pass the comparison above and
  fail this one;
- **control:** the packed row with its boundaries collapsed to a single segment. It must come out
  *wrong*, which is what proves the boundaries were read rather than ignored.

The dense path is the baseline the campaign measured, so the comparison is exactly the one the
32-die arms cannot make: their steps hold one document per row, where both paths agree trivially.

Run it on a node, on one die, no checkpoint and no dataloader::

    PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packed_attention.py
    PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packed_attention.py --dtype bfloat16

float32 is the default because it compares tightest; the study's runs are bfloat16, and the second
form reruns the same comparison in the dtype they use, with the tolerance relaxed to match. Should
the kernel of a given CANN release refuse float32, that second form is the one to read.

On a host without ``torch_npu`` the kernel legs are skipped and the two references are compared with
each other, which checks this script's own notion of isolation and nothing about the device.
"""

import argparse
import sys
from typing import Any, Optional, Sequence

import torch

_DOCUMENTS = (1024, 512, 2048, 7, 333)
_HEADS = 4
_HEAD_DIM = 128


def make_inputs(lengths: Sequence[int], dtype: torch.dtype, device: str) -> tuple[torch.Tensor, ...]:
    """Draw query, key and value for one packed row.

    Args:
        lengths: Token count of each document in the row.
        dtype: Element type of the drawn tensors.
        device: Device to place them on.

    Returns:
        Query, key and value in BNSD layout with a batch of one and the row's total length.
    """
    generator = torch.Generator().manual_seed(0)
    total = sum(lengths)
    shape = (1, _HEADS, total, _HEAD_DIM)
    drawn = [torch.randn(shape, generator=generator, dtype=torch.float32) for _ in range(3)]
    return tuple(tensor.to(device=device, dtype=dtype) for tensor in drawn)


def reference_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        lengths: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Return causal attention in float32, block-diagonal over the documents when given their lengths.

    Args:
        query: Query in BNSD layout.
        key: Key in BNSD layout.
        value: Value in BNSD layout.
        lengths: Token count of each document; None treats the row as one document.

    Returns:
        The attention output in BSND layout, as the wrapper returns it.
    """
    query, key, value = (tensor.to(torch.float32) for tensor in (query, key, value))
    total = query.shape[2]
    scores = query @ key.transpose(-1, -2) * _HEAD_DIM ** -0.5
    allowed = torch.ones(total, total, dtype=torch.bool, device=query.device).tril()
    if lengths is not None:
        same_document = torch.zeros(total, total, dtype=torch.bool, device=query.device)
        start = 0
        for length in lengths:
            same_document[start:start + length, start:start + length] = True
            start += length
        allowed &= same_document
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (scores.softmax(dim=-1) @ value).transpose(1, 2)


def kernel_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     boundaries: Optional[Sequence[int]] = None) -> tuple[torch.Tensor, str, int]:
    """Run the production wrapper, dense without boundaries and variable-length with them.

    Args:
        query: Query in BNSD layout.
        key: Key in BNSD layout.
        value: Value in BNSD layout.
        boundaries: Leading-zero cumulative document ends, or None for the dense path.

    Returns:
        The output in BSND layout, the layout the kernel was called in, and its sparse mode.
    """
    # The module imports torch_npu at its own scope, so it is reached from here and not at module
    # scope. The private helper reports which branch the wrapper took, which turns "the numbers
    # agree" into "the variable-length branch ran and the numbers agree".
    from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
        _prepare_fusion_attention_context,
        npu_fusion_attention_forward,
    )

    module = torch.nn.Module()
    kwargs: dict[str, Any] = {}
    if boundaries is not None:
        kwargs["cu_seq_lens"] = torch.tensor(boundaries, dtype=torch.int32, device=query.device)
    context = _prepare_fusion_attention_context(module, query, key, value, None, dict(kwargs))
    output, _ = npu_fusion_attention_forward(module, query, key, value, None, **kwargs)
    return output, context.input_layout, context.sparse_mode


def largest_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the largest absolute difference between two outputs, compared in float32."""
    return float((left.to(torch.float32) - right.to(torch.float32)).abs().max())


def main(argv: Optional[list[str]] = None) -> int:
    """Run the check and report whether a document's attention depends on its neighbours.

    Args:
        argv: Command line, or None to read ``sys.argv``.

    Returns:
        0 when the packed row reproduces the separate documents and collapsed boundaries do not.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"),
                        help="element type; the study's runs are bfloat16, float32 compares tightest")
    parser.add_argument("--device", default="npu:0", help="device to run the kernel on")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="largest difference accepted; by default 1e-5 in float32, 2e-2 otherwise")
    args = parser.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    tolerance = args.tolerance if args.tolerance is not None else (1e-5 if dtype == torch.float32 else 2e-2)
    lengths = list(_DOCUMENTS)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)

    reachable = args.device != "cpu"
    if reachable:
        try:
            import torch_npu  # pylint: disable=C0415,unused-import
        except ImportError:
            reachable = False
    if not reachable:
        print("the kernel is out of reach here: comparing the two references, which checks this "
              "script and not the device")
        query, key, value = make_inputs(lengths, torch.float32, "cpu")
        separate = torch.cat([reference_attention(query[:, :, start:end], key[:, :, start:end],
                                                  value[:, :, start:end])
                              for start, end in zip(boundaries, boundaries[1:])], dim=1)
        blocked = reference_attention(query, key, value, lengths)
        difference = largest_difference(separate, blocked)
        print(f"per document against one block-diagonal row: max|diff| = {difference:.3e}")
        if difference > 1e-5:
            print("FAIL: the references disagree, so this script's notion of isolation is wrong")
            return 1
        print("PASS (references only): run this on a die to reach the kernel")
        return 0

    query, key, value = make_inputs(lengths, dtype, args.device)
    print(f"documents of {lengths} tokens, {_HEADS} heads x {_HEAD_DIM}, {args.dtype} on {args.device}")

    separate_parts, layouts = [], []
    for start, end in zip(boundaries, boundaries[1:]):
        part, layout, sparse = kernel_attention(query[:, :, start:end], key[:, :, start:end],
                                                value[:, :, start:end])
        separate_parts.append(part)
        layouts.append((layout, sparse))
    separate = torch.cat(separate_parts, dim=1)
    packed, packed_layout, packed_sparse = kernel_attention(query, key, value, boundaries)
    collapsed, _, _ = kernel_attention(query, key, value, [0, boundaries[-1]])
    reference = reference_attention(query, key, value, lengths)

    print(f"  separate: layout {layouts[0][0]}, sparse mode {layouts[0][1]}")
    print(f"  packed:   layout {packed_layout}, sparse mode {packed_sparse}")
    against_separate = largest_difference(packed, separate)
    against_reference = largest_difference(packed, reference)
    separate_against_reference = largest_difference(separate, reference)
    against_collapsed = largest_difference(packed, collapsed)
    print(f"  packed against separate documents:  max|diff| = {against_separate:.3e}  "
          f"(must be <= {tolerance:.0e})")
    print(f"  packed against the float32 reference: max|diff| = {against_reference:.3e}  "
          f"(must be <= {tolerance:.0e})")
    print(f"  separate against the reference:       max|diff| = {separate_against_reference:.3e}")
    print(f"  control, boundaries collapsed to one: max|diff| = {against_collapsed:.3e}  (must be large)")

    if packed_layout != "TND":
        print(f"FAIL: the boundaries did not reach the kernel, which ran in {packed_layout}")
        return 1
    if against_collapsed <= tolerance:
        print("FAIL: collapsing the boundaries changed nothing, so this check proves nothing")
        return 1
    if against_separate > tolerance:
        print("FAIL: a document's attention depends on what was packed beside it")
        return 1
    if against_reference > tolerance:
        print("FAIL: both kernel paths agree with each other and neither matches the reference")
        return 1
    print("PASS: the kernel reads the boundaries, and a document attends to itself alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
