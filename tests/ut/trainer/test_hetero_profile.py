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
"""The heterogeneity profiler: workload of a micro-batch, component boundaries, routing by modality."""

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.trainer.callbacks.hetero_profile_callback import HeteroProfileCallback
from hyper_parallel.trainer.config import HeteroProfileConfig
from hyper_parallel.trainer.runtime.hetero_profile import (
    BWD, ENTER, EXIT, FWD, RECOMPUTE, HeteroProfiler, batch_workload, normalize_module_path, step_digest,
)

HIDDEN, EXPERTS, TOP_K, TOKENS = 8, 4, 2, 6


class _Router(nn.Module):
    """Returns (logits, scores, indices) as the Hugging Face top-k router does."""

    def __init__(self) -> None:
        """Create the router weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(EXPERTS, HIDDEN))

    def forward(self, hidden: torch.Tensor) -> tuple:
        """Score the experts of every token and keep the top ones."""
        logits = hidden.reshape(-1, HIDDEN) @ self.weight.t()
        scores, indices = logits.softmax(-1).topk(TOP_K, dim=-1)
        return logits, scores, indices


class _Moe(nn.Module):
    """A MoE block that only consults its router."""

    def __init__(self) -> None:
        """Create the router and a projection."""
        super().__init__()
        self.gate = _Router()
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Route, then project."""
        self.gate(hidden)
        return self.proj(hidden)


class _Layer(nn.Module):
    """A decoder layer, optionally checkpointed (``recompute`` is the checkpoint's ``use_reentrant``)."""

    def __init__(self, recompute: Optional[bool] = None) -> None:
        """Create the attention stand-in and the MoE block."""
        super().__init__()
        self.self_attn = nn.Linear(HIDDEN, HIDDEN)
        self.mlp = _Moe()
        self.recompute = recompute

    def _body(self, hidden: torch.Tensor) -> torch.Tensor:
        """The layer's computation."""
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.mlp(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the body, under a checkpoint when asked."""
        if self.recompute is None:
            return self._body(hidden)
        return checkpoint(self._body, hidden, use_reentrant=self.recompute)


class _Vision(nn.Module):
    """Three blocks over the patches."""

    def __init__(self) -> None:
        """Create the patch embedding and the blocks."""
        super().__init__()
        self.patch_embed = nn.Linear(HIDDEN, HIDDEN)
        self.blocks = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(3))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed, then add each block's output."""
        hidden = self.patch_embed(patches)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return hidden


class _Text(nn.Module):
    """Two decoder layers."""

    def __init__(self, recompute: Optional[bool] = None) -> None:
        """Create the layers."""
        super().__init__()
        self.layers = nn.ModuleList(_Layer(recompute) for _ in range(2))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the layers in order."""
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _Model(nn.Module):
    """The module paths of Qwen3-VL-MoE: visual, language_model, lm_head."""

    def __init__(self, recompute: Optional[bool] = None) -> None:
        """Create the towers and the head."""
        super().__init__()
        self.visual = _Vision()
        self.language_model = _Text(recompute)
        self.lm_head = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, patches: torch.Tensor, mm_token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Vision, text, head; the media mask is read by the profiler's hook, not by the model."""
        del mm_token_type_ids
        return self.lm_head(self.language_model(self.visual(patches)))


def _profiler(tmp_path, **options):
    profiler = HeteroProfiler()
    profiler.configure(enabled=True, output_dir=str(tmp_path), **options)
    return profiler


def _run_step(model, profiler, step=1):
    """One forward and backward under the profiler; return the record."""
    media = torch.tensor([[1, 1, 1, 0, 0, 0]])
    batch = {"input_ids": torch.zeros(1, TOKENS, dtype=torch.long), "mm_token_type_ids": media}
    profiler.begin_step(step, [batch])
    out = model(torch.randn(TOKENS, HIDDEN), mm_token_type_ids=media)
    out.sum().backward()
    return profiler.end_step(loss=1.0)


def _marks_of(record, profiler, role, index=None):
    ids = {m["id"] for m in profiler.modules if m["role"] == role and (index is None or m["index"] == index)}
    return [mark for mark in record["marks"] if mark[0] in ids]


def test_batch_workload_counts_text_and_media():
    """Tokens, loss positions, patches, visual tokens and the vision attention cost."""
    batch = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.tensor([[1] * 8 + [0] * 2]),
        "labels": torch.tensor([[-100] * 5 + [7, 8, 9, -100, -100]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]]),
        "pixel_values": torch.zeros(20, 3),
    }
    work = batch_workload(batch)
    assert (work["tokens"], work["real_tokens"], work["label_tokens"], work["image_tokens"]) == (10, 8, 3, 4)
    assert (work["images"], work["patches"], work["visual_tokens"], work["pixel_rows"]) == (2, 20, 5, 20)
    assert work["vision_attn_pairs"] == 16 ** 2 + 4 ** 2
    assert work["grids"] == [[1, 4, 4], [1, 2, 2]]


def test_batch_workload_tolerates_missing_fields():
    """A text-only batch has no media and still reports its tokens."""
    work = batch_workload({"input_ids": torch.zeros(2, 5, dtype=torch.long)})
    assert (work["tokens"], work["batch_size"], work["images"], work["patches"]) == (10, 2, 0, 0)


def test_wrapper_segments_do_not_change_a_role():
    """A checkpoint wrapper in the path leaves the module's role alone."""
    path = "model.language_model.layers.3._checkpoint_wrapped_module.self_attn"
    assert normalize_module_path(path) == "model.language_model.layers.3.self_attn"
    assert HeteroProfiler().match_role(path) == ("text.attn", 3)
    assert HeteroProfiler().match_role("model.visual.blocks.12") == ("vision.block", 12)
    assert HeteroProfiler().match_role("model.language_model.layers.1.mlp.gate") == ("text.router", 1)
    assert HeteroProfiler().match_role("model.language_model.layers.1.mlp.experts") == (None, None)


def test_attach_hooks_each_role_once(tmp_path):
    """The model, both towers, the vision blocks, the layers with their sublayers, the routers and the head."""
    profiler = _profiler(tmp_path)
    counts = profiler.attach(_Model())
    assert counts == {
        "root": 1, "vision.root": 1, "vision.patch_embed": 1, "vision.block": 3, "text.root": 1, "text.layer": 2,
        "text.attn": 2, "text.moe": 2, "text.router": 2, "lm_head": 1,
    }


def test_detail_switches_drop_the_finer_levels(tmp_path):
    """Without vision blocks, sublayers or routing only the coarse modules are hooked."""
    profiler = _profiler(tmp_path, vision_blocks=False, sublayers=False, routing_by_modality=False)
    assert set(profiler.attach(_Model())) == {"root", "vision.root", "text.root", "text.layer", "lm_head"}


def test_forward_and_backward_boundaries_of_every_block(tmp_path):
    """Each block has four boundaries; the backward visits the blocks in reverse order."""
    profiler = _profiler(tmp_path)
    model = _Model()
    profiler.attach(model)
    record = _run_step(model, profiler)

    for index in range(3):
        marks = _marks_of(record, profiler, "vision.block", index)
        kinds = [(mark[1], mark[3]) for mark in marks]
        assert sorted(kinds) == sorted([(FWD, ENTER), (FWD, EXIT), (BWD, ENTER), (BWD, EXIT)]), kinds
        times = {(mark[1], mark[3]): mark[4] for mark in marks}
        assert times[(FWD, ENTER)] <= times[(FWD, EXIT)]
        assert times[(BWD, ENTER)] <= times[(BWD, EXIT)]
    backward_start = [_marks_of(record, profiler, "vision.block", i) for i in range(3)]
    starts = [next(m[4] for m in marks if m[1] == BWD and m[3] == ENTER) for marks in backward_start]
    assert starts[2] <= starts[1] <= starts[0]


def test_micro_batch_index_follows_the_call_count(tmp_path):
    """Two micro-batches in a step give occurrences 0 and 1 on every forward module."""
    profiler = _profiler(tmp_path)
    model = _Model()
    profiler.attach(model)
    media = torch.tensor([[1, 1, 0, 0, 0, 0]])
    profiler.begin_step(1, [{"input_ids": torch.zeros(1, TOKENS)}] * 2)
    for _ in range(2):
        model(torch.randn(TOKENS, HIDDEN), mm_token_type_ids=media).sum().backward()
    record = profiler.end_step()
    occurrences = {mark[2] for mark in _marks_of(record, profiler, "text.layer", 0) if mark[1] == FWD}
    assert occurrences == {0, 1}
    assert len(record["micro_batches"]) == 2


def test_routing_is_split_by_modality(tmp_path):
    """Per layer, the expert choices of the image tokens and of the text tokens add up to every choice."""
    profiler = _profiler(tmp_path)
    model = _Model()
    profiler.attach(model)
    record = _run_step(model, profiler)
    assert sorted(row["layer"] for row in record["routing"]) == [0, 1]
    for row in record["routing"]:
        assert sum(row["visual"]) == 3 * TOP_K
        assert sum(row["text"]) == 3 * TOP_K
        assert len(row["visual"]) == EXPERTS


@pytest.mark.parametrize("reentrant", [True, False])
def test_recompute_is_told_from_the_forward_pass(tmp_path, reentrant):
    """A checkpointed block's second run is its recompute; the backward boundaries still fire once."""
    profiler = _profiler(tmp_path)
    model = _Model(recompute=reentrant)
    profiler.attach(model)
    record = _run_step(model, profiler)
    attn = _marks_of(record, profiler, "text.attn", 0)
    passes = {mark[1] for mark in attn}
    assert {FWD, RECOMPUTE} <= passes
    layer = _marks_of(record, profiler, "text.layer", 0)
    assert sum(1 for mark in layer if mark[1] == BWD and mark[3] == ENTER) == 1
    assert sum(1 for mark in layer if mark[1] == BWD and mark[3] == EXIT) == 1


def test_record_is_written_with_a_header(tmp_path):
    """One file per rank: a header with the module table, then the step records."""
    profiler = _profiler(tmp_path)
    model = _Model()
    profiler.attach(model)
    _run_step(model, profiler, step=3)
    profiler.close()
    lines = [json.loads(line) for line in (tmp_path / "rank000.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "header" and lines[0]["time_source"] == "host"
    assert {m["role"] for m in lines[0]["modules"]} >= {"root", "vision.block", "text.moe", "lm_head"}
    assert lines[1]["kind"] == "step" and lines[1]["step"] == 3 and lines[1]["loss"] == 1.0
    assert lines[1]["inter_step_ms"] is None


def test_inactive_hooks_change_nothing(tmp_path):
    """Outside a recorded step, the model computes the same and nothing is recorded."""
    torch.manual_seed(0)
    model = _Model()
    patches = torch.randn(TOKENS, HIDDEN)
    expected = model(patches)
    profiler = _profiler(tmp_path)
    profiler.attach(model)
    assert torch.equal(model(patches), expected)
    assert profiler.end_step() is None


def test_step_digest_sums_the_roots(tmp_path):
    """The live digest reports the workload and the forward time of the three big components."""
    profiler = _profiler(tmp_path)
    model = _Model()
    profiler.attach(model)
    record = _run_step(model, profiler)
    digest = step_digest(record, profiler.modules)
    assert digest["micro_batches"] == 1.0
    assert digest["step_ms"] >= digest["bwd_ms"] >= 0.0
    assert digest["head_fwd_ms"] >= 0.0


def test_callback_window_follows_the_config(tmp_path):
    """Steps count from one; the window is [start_step, end_step) and 0 means to the end."""
    config = SimpleNamespace(hetero_profile=HeteroProfileConfig(enabled=True, output_dir=str(tmp_path),
                                                                start_step=2, end_step=4))
    trainer = SimpleNamespace(config=config, mesh=None, model=None, global_rank=0)
    callback = HeteroProfileCallback(trainer)
    try:
        assert [callback._records(SimpleNamespace(global_step=step)) for step in range(5)] \
            == [False, True, True, False, False]  # pylint: disable=protected-access
    finally:
        callback.profiler.configure(enabled=False, output_dir="")   # the callback configures the shared recorder
    with pytest.raises(ValueError):
        HeteroProfileConfig(start_step=0)
    with pytest.raises(ValueError):
        HeteroProfileConfig(start_step=3, end_step=3)


def test_records_feed_the_report(tmp_path):
    """What the profiler writes is what the analysis script reads: a rank file per rank, steps, workloads."""
    path = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf" / "analyze_hetero.py"
    spec = importlib.util.spec_from_file_location("analyze_hetero_for_test", path)
    report = importlib.util.module_from_spec(spec)
    sys.modules["analyze_hetero_for_test"] = report
    spec.loader.exec_module(report)

    profiler = _profiler(tmp_path / "source")
    model = _Model(recompute=True)
    profiler.attach(model)
    for step in range(1, 5):
        _run_step(model, profiler, step=step)
    profiler.close()
    lines = (tmp_path / "source" / "rank000.jsonl").read_text().splitlines()
    run_dir = tmp_path / "run" / "hetero"
    run_dir.mkdir(parents=True)
    for rank in range(3):                  # the same records stand in for three ranks
        header = json.loads(lines[0])
        header["rank"] = rank
        (run_dir / f"rank{rank:03d}.jsonl").write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")
    text, summary = report.analyse(str(run_dir), skip=1, ep_size=2, stages=[2], top=3, out_dir=None)
    joined = "\n".join(text)
    assert summary["overview"]["ranks"] == 3 and summary["overview"]["steps"] == 3
    assert summary["model"]["vision"]["total"] > 0 and summary["model"]["text_layer"]["total"] > 0
    assert "ROUTING by modality" in joined and summary["routing"]["layers"]
    assert summary["imbalance"]["busiest_over_mean"] == 1.0 or summary["imbalance"]["busiest_over_mean"] >= 1.0


def test_an_error_in_the_recorder_switches_it_off_and_not_the_training(tmp_path, monkeypatch):
    """A failure inside a hook must not reach the model: the run goes on, the recorder says why it stopped."""
    torch.manual_seed(0)
    model = _Model()
    patches = torch.randn(TOKENS, HIDDEN)
    expected = model(patches)
    profiler = _profiler(tmp_path)
    profiler.attach(model)
    media = torch.tensor([[1, 1, 1, 0, 0, 0]])
    profiler.begin_step(1, [{"input_ids": torch.zeros(1, TOKENS, dtype=torch.long)}])

    def broken() -> None:
        """Stand in for a clock that fails."""
        raise RuntimeError("the clock broke")

    monkeypatch.setattr(profiler._clock, "stamp", broken)  # pylint: disable=protected-access
    out = model(patches, mm_token_type_ids=media)
    out.sum().backward()
    assert torch.equal(out, expected)
    assert profiler.end_step() is None
    profiler.begin_step(2, [])
    assert profiler.end_step() is None          # stays off


def test_attach_to_nothing_is_harmless(tmp_path):
    """A recorder that is off, or given no model, hooks nothing."""
    assert not HeteroProfiler().attach(_Model())
    assert not _profiler(tmp_path).attach(None)
