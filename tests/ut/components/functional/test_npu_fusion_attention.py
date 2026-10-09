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
"""Resolving the document boundaries of a packed batch, which decide whether the kernel runs packed.

The resolution is the whole of the packed contract on this path: lengths found means TND with a
per-document causal mask, lengths missed means one causal mask over the whole row, and the second is
not a weaker mask but a wrong one -- every document attends to the ones packed before it, and the
loss that comes back looks ordinary. So the keyword a packed batch really carries is tested here.
"""

import importlib.machinery
import sys
from unittest import mock

import pytest
import torch


def _resolver():
    """Import the resolver, standing torch_npu in for the absent accelerator.

    The module imports ``torch_npu`` at its own scope, so a CPU-only checkout needs the stand-in.
    The resolver itself is plain Python and Torch, and touches nothing of the stub.
    """
    stub = mock.MagicMock()
    stub.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    sys.modules.setdefault("torch_npu", stub)
    from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
        resolve_packed_sequence_lengths,
    )
    return resolve_packed_sequence_lengths


def test_a_plain_cu_seq_lens_keyword_is_what_a_packed_batch_carries():
    """Three documents in one row of 12 tokens: both sides must come back with their ends.

    This is the keyword the VLM packing collator emits and Transformers forwards. It was once read
    only out of a ``packed_seq_params`` carrier, so a real packed batch resolved to no boundaries and
    the dense branch ran -- which the 32-die arms could not see, because a row of one document is a
    plain causal row either way.
    """
    resolve = _resolver()

    query_lengths, key_lengths = resolve({"cu_seq_lens": torch.tensor([0, 7, 10, 12])}, 12, 12)

    assert query_lengths == [7, 10, 12], f"query boundaries mismatch: got={query_lengths}"
    assert key_lengths == [7, 10, 12], f"key boundaries mismatch: got={key_lengths}"


def test_the_leading_zero_is_required_of_the_plain_keyword():
    """A list of ends without the leading zero is a different convention, and is refused."""
    resolve = _resolver()

    with pytest.raises(ValueError, match="must start with zero"):
        resolve({"cu_seq_lens": torch.tensor([7, 10, 12])}, 12, 12)


def test_the_plain_keyword_must_cover_the_row():
    """Boundaries that stop short of the tokens handed over are a mismatch, not a packed row."""
    resolve = _resolver()

    with pytest.raises(ValueError, match="final query sequence length"):
        resolve({"cu_seq_lens": torch.tensor([0, 7, 10])}, 12, 12)


def test_a_prefixed_name_disagreeing_with_the_plain_one_is_refused():
    """Two spellings of the same thing must agree; silently preferring one would hide a bug."""
    resolve = _resolver()

    with pytest.raises(ValueError, match="conflicting"):
        resolve({"cu_seq_lens": torch.tensor([0, 7, 12]), "cu_seq_lens_q": torch.tensor([0, 6, 12])}, 12, 12)


def test_agreeing_spellings_resolve():
    """The same boundaries under both spellings are one answer, not a conflict."""
    resolve = _resolver()

    query_lengths, _ = resolve(
        {"cu_seq_lens": torch.tensor([0, 7, 12]), "cu_seq_lens_q": torch.tensor([0, 7, 12])}, 12, 12,
    )

    assert query_lengths == [7, 12], f"agreeing spellings must resolve: got={query_lengths}"


def test_an_unpacked_batch_still_resolves_to_nothing():
    """No boundaries means the dense path, which is correct for a batch that is not packed."""
    resolve = _resolver()

    assert resolve({}, 12, 12) == (None, None), "an unpacked batch must not be taken for a packed one"


def test_the_parameter_carrier_still_works():
    """The Megatron-style carrier is the other way boundaries arrive, and it is untouched."""
    resolve = _resolver()

    query_lengths, key_lengths = resolve({"packed_seq_params": {"cu_seq_lens": [0, 7, 12]}}, 12, 12)

    assert query_lengths == [7, 12], f"carrier query boundaries mismatch: got={query_lengths}"
    assert key_lengths == [7, 12], f"carrier key boundaries mismatch: got={key_lengths}"
