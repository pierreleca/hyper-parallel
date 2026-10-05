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
"""The heterogeneity report: spans, cost-model fit, imbalance, what-ifs and routing, on records of known cost."""

import importlib.util
import json
import pathlib
import random
import sys
from types import ModuleType

import pytest

_DIR = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf"


def _load(name: str) -> ModuleType:
    """Import an example script by path."""
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = _load("analyze_hetero")

LAYERS, BLOCKS = 2, 2
# Known cost of each component, in ms: intercept, per 1k tokens (or patches), per 1M squared tokens (or pairs).
COST = {
    "vision_block": (0.4, 0.8, 0.5), "layer": (0.3, 1.2, 0.9), "attn": (0.1, 0.3, 0.6), "moe": (0.1, 0.7, 0.0),
    "head": (0.2, 0.5, 0.0), "loss": (0.2, 0.4, 0.0),
}


def _modules() -> list:
    """The module table of a tiny Qwen3-VL-like model."""
    table = [("root", None), ("vision.root", None)] + [("vision.block", i) for i in range(BLOCKS)] + [
        ("text.root", None)] + [("text.layer", i) for i in range(LAYERS)] + [
        ("text.attn", i) for i in range(LAYERS)] + [("text.moe", i) for i in range(LAYERS)] + [("lm_head", None)]
    return [{"id": i, "name": f"{role}.{index}", "role": role, "index": index} for i, (role, index) in enumerate(table)]


def _cost(kind: str, x1: float, x2: float) -> float:
    """The exact time of one component."""
    c0, c1, c2 = COST[kind]
    return c0 + c1 * x1 + c2 * x2


def _record(ids: dict, step: int, batch: dict, scale: float = 1.0) -> dict:
    """One step record whose spans follow COST exactly (times scaled by ``scale``, the rank's speed)."""
    tokens, patches = batch["real_tokens"] / 1e3, batch["patches"] / 1e3
    pairs = batch["vision_attn_pairs"] / 1e6
    marks, clock = [], [0.0]

    def span(module_id: int, phase: str, duration: float, start: float = None) -> tuple:
        """Append the two boundaries of a span and return its ends."""
        begin = clock[0] if start is None else start
        marks.append([module_id, phase, 0, "in", begin, 1000])
        marks.append([module_id, phase, 0, "out", begin + duration, 1000])
        return begin, begin + duration

    def tick(duration: float) -> None:
        """Advance the clock."""
        clock[0] += duration

    marks.append([ids["root"], "fwd", 0, "in", 0.0, 1000])
    start = clock[0]
    tick(0.05)
    for block in ids["blocks"]:
        duration = scale * _cost("vision_block", patches, pairs) * 0.4
        span(block, "fwd", duration)
        tick(duration + 0.02)
    marks.append([ids["vision.root"], "fwd", 0, "in", start, 1000])
    marks.append([ids["vision.root"], "fwd", 0, "out", clock[0], 1000])
    text_start = clock[0]
    for layer, attn, moe in zip(ids["layers"], ids["attn"], ids["moe"]):
        total = scale * _cost("layer", tokens, tokens * tokens) * 0.4
        begin, end = span(layer, "fwd", total)
        span(attn, "fwd", scale * _cost("attn", tokens, tokens * tokens) * 0.4, begin)
        span(moe, "fwd", scale * _cost("moe", tokens, 0.0) * 0.4, begin + 0.5)
        clock[0] = end + 0.03
    marks.append([ids["text.root"], "fwd", 0, "in", text_start, 1000])
    marks.append([ids["text.root"], "fwd", 0, "out", clock[0], 1000])
    head = scale * _cost("head", tokens, 0.0) * 0.4
    span(ids["head"], "fwd", head)
    tick(head)
    marks.append([ids["root"], "fwd", 0, "out", clock[0], 1000])
    loss = scale * _cost("loss", tokens, 0.0)
    tick(loss)
    marks.append([ids["head"], "bwd", 0, "in", clock[0], 1000])
    tick(head * 1.5)
    marks.append([ids["head"], "bwd", 0, "out", clock[0], 1000])
    for layer in reversed(ids["layers"]):
        duration = scale * _cost("layer", tokens, tokens * tokens) * 0.6
        span(layer, "bwd", duration)
        tick(duration + 0.04)
    for block in reversed(ids["blocks"]):
        duration = scale * _cost("vision_block", patches, pairs) * 0.6
        span(block, "bwd", duration)
        tick(duration + 0.02)
    return {"kind": "step", "step": step, "wall_ms": clock[0] + 5, "device_ms": clock[0] + 4.0, "inter_step_ms": 3.0,
            "micro_batches": [batch], "marks": marks, "routing": [], "peak_allocated": 2 ** 30 * (1 + tokens / 10),
            "peak_reserved": None}


def _write_run(directory: pathlib.Path, ranks: int = 4, steps: int = 8, speeds: dict = None, seed: int = 0,
               shared: bool = False) -> dict:
    """Write rank files for a run; return the module ids. With ``shared`` every rank of a step gets the same sample."""
    rng = random.Random(seed)
    modules = _modules()
    ids = {
        "root": 0, "vision.root": 1, "blocks": [2, 3], "text.root": 4, "layers": [5, 6], "attn": [7, 8], "moe": [9, 10],
        "head": 11,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for rank in range(ranks):
        lines = [{"kind": "header", "rank": rank, "world_size": ranks, "host": "h", "device_type": "npu",
                  "time_source": "device", "spatial_merge_size": 2, "modules": modules}]
        for step in range(1, steps + 1):
            if shared:
                rng = random.Random(seed * 1000 + step)
            tokens = rng.randint(1000, 9000)
            patches = rng.randint(400, 6000)
            batch = {"real_tokens": tokens, "tokens": tokens, "label_tokens": 50, "images": 2, "patches": patches,
                     "visual_tokens": patches // 4, "vision_attn_pairs": patches * patches // 2}
            lines.append(_record(ids, step, batch, (speeds or {}).get(rank, 1.0)))
        (directory / f"rank{rank:03d}.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return ids


def test_spans_pair_the_boundaries_of_a_module(tmp_path):
    """Entry and exit of the same module, pass and micro-batch form one span."""
    _write_run(tmp_path, ranks=1, steps=1)
    run = report.Run(str(tmp_path))
    spans = report.spans_of(run.steps[0][0])
    assert set(spans[(2, "fwd", 0)]) == {"in", "out"} and set(spans[(2, "bwd", 0)]) == {"in", "out"}
    assert spans[(2, "fwd", 0)]["out"] > spans[(2, "fwd", 0)]["in"]


def test_rows_carry_the_workload_and_the_component_times(tmp_path):
    """Each micro-batch row has the sample's features and forward and backward time per component."""
    _write_run(tmp_path, ranks=2, steps=3)
    rows = report.build_rows(report.Run(str(tmp_path)))
    assert len(rows) == 6
    row = rows[0]
    assert row["real_tokens"] > 0 and row["patches"] > 0
    assert row["t"]["vision"]["fwd"] > 0 and row["t"]["vision"]["bwd"] > 0
    assert row["t"]["text_layer"]["fwd"] > 0 and row["t"]["loss"]["bwd"] > 0
    assert row["gap"]["vision"]["fwd"] == pytest.approx(0.02, abs=1e-6)
    assert row["gap"]["text"]["bwd"] == pytest.approx(0.0, abs=1e-6) or row["gap"]["text"]["bwd"] > 0


def test_fit_recovers_the_known_cost(tmp_path):
    """The cost model finds the coefficients the records were built from."""
    _write_run(tmp_path, ranks=4, steps=12)
    rows = report.build_rows(report.Run(str(tmp_path)))
    models = report.fit_components(rows)
    c0, c1, c2 = models["text_layer"]["coefficients"]
    # forward 0.4 + backward 0.6 of the layer's cost, summed over the two layers
    assert (c0, c1, c2) == pytest.approx(tuple(LAYERS * v for v in COST["layer"]), rel=1e-3, abs=1e-4)
    assert models["text_layer"]["r2"] > 0.999
    vision = models["vision"]["coefficients"]
    assert vision == pytest.approx([BLOCKS * COST["vision_block"][i] for i in range(3)], rel=1e-3, abs=1e-4)
    assert models["loss"]["coefficients"][1] == pytest.approx(COST["loss"][1], rel=1e-3)


def test_fit_drops_a_feature_that_does_not_vary():
    """A constant column cannot be told from the intercept; the fit keeps the intercept and a zero coefficient."""
    model = report.fit([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], [2.0, 4.0, 6.0])
    assert model["coefficients"] == pytest.approx([0.0, 2.0, 0.0], abs=1e-6)
    assert model["r2"] == pytest.approx(1.0)


def test_slow_rank_is_the_busiest_and_explains_the_excess(tmp_path):
    """A rank 30% slower than the others is the busiest in every step, and the report says so."""
    _write_run(tmp_path, ranks=4, steps=10, speeds={2: 1.3}, shared=True)
    out: list = []
    run = report.Run(str(tmp_path))
    rows = report.build_rows(run)
    imbalance = report.report_imbalance(rows, report.fit_components(rows), out)
    assert imbalance["busiest_over_mean"] > 1.1
    assert "rank 2 x10" in "\n".join(out)
    assert sum(imbalance["excess_by_part"].values()) == pytest.approx(1.0)


def test_data_report_measures_the_spread(tmp_path):
    """The data section reports each feature's spread and the busiest-over-mean of a step."""
    _write_run(tmp_path, ranks=4, steps=10)
    rows = report.build_rows(report.Run(str(tmp_path)))
    summary = report.report_data(rows, [])
    assert summary["real_tokens"]["cv"] > 0.2
    assert summary["real_tokens"]["max_over_mean"] > 1.1


def test_bucketing_removes_most_of_the_modeled_excess(tmp_path):
    """Sorting the samples of a window into steps leaves little for the slowest rank to add."""
    _write_run(tmp_path, ranks=8, steps=16)
    run = report.Run(str(tmp_path))
    rows = report.build_rows(run)
    models = report.fit_components(rows)
    imbalance = report.report_imbalance(rows, models, [])
    outcome = report.report_whatif_balancing(rows, models, imbalance, [])
    assert outcome["modeled_bucketed"] < outcome["modeled_actual"]
    assert outcome["removable_share_of_excess"] > 0.5


def test_partition_cuts_a_sequence_evenly():
    """The cut minimizing the heaviest stage, and where it falls."""
    heaviest, cuts = report.partition([5, 1, 1, 1, 1, 1, 5], 2)
    assert heaviest == 8 and cuts == [4] or heaviest == 8 and cuts == [3]
    heaviest, cuts = report.partition([1.0] * 8, 4)
    assert heaviest == 2.0 and cuts == [2, 4, 6]


def test_pipeline_report_counts_the_towers(tmp_path):
    """The vision tower and the head are shares of the work; cuts are reported per stage count."""
    _write_run(tmp_path, ranks=2, steps=4)
    run = report.Run(str(tmp_path))
    rows = report.build_rows(run)
    out: list = []
    summary = report.report_pipeline(run, rows, [2], out)
    assert 0 < summary["vision_share"] < 1
    assert summary["2"]["best_over_mean"] >= 1.0
    assert summary["2"]["best_over_mean"] <= summary["2"]["even_layers_over_mean"] + 1e-9


def test_divergence_of_identical_and_disjoint_choices():
    """Jensen-Shannon: 0 for the same distribution, 1 bit for disjoint ones."""
    assert report.divergence([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0)
    assert report.divergence([1, 0], [0, 1]) == pytest.approx(1.0)


def test_routing_by_modality_finds_an_overloaded_ep_rank(tmp_path):
    """Image tokens that all pick the experts of one EP rank load it far above the mean; text tokens do not."""
    _write_run(tmp_path, ranks=4, steps=2)
    experts, ep_size = 8, 4
    for rank in range(4):
        path = tmp_path / f"rank{rank:03d}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for record in lines[1:]:
            record["routing"] = [{"layer": 0, "mb": 0, "visual": [10, 10, 0, 0, 0, 0, 0, 0],
                                  "text": [3] * experts}]
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    out: list = []
    summary = report.report_routing(report.Run(str(tmp_path)), ep_size, out, 5)
    layer = summary["layers"][0]
    assert layer["visual"] == pytest.approx(4.0)      # experts 0 and 1 both belong to EP rank 0
    assert layer["text"] == pytest.approx(1.0)
    assert layer["all"] == pytest.approx(104 / 44) and layer["js_bits"] > 0.1
    assert layer["hot_visual"][:2] == [0, 1]


def test_memory_report_fits_the_peak_on_the_workload(tmp_path):
    """Per-step peaks follow the tokens, and the report extrapolates to the longest sample."""
    _write_run(tmp_path, ranks=2, steps=10)
    rows = report.build_rows(report.Run(str(tmp_path)))
    out: list = []
    summary = report.report_memory(rows, out)
    assert summary["peak_gib_max"] >= summary["peak_gib_mean"] > 1.0
    assert any("per-step peak" in line for line in out)


def test_whole_report_runs_and_writes_its_files(tmp_path):
    """The command-line path: every section, the CSV and the JSON."""
    run_dir = tmp_path / "hetero"
    _write_run(run_dir, ranks=4, steps=6, speeds={1: 1.2})
    lines, summary = report.analyse(str(run_dir), skip=1, ep_size=4, stages=[2, 4], top=3,
                                    out_dir=str(tmp_path / "analysis"))
    text = "\n".join(lines)
    for heading in ("RUN ", "DATA:", "MODEL:", "COST MODEL:", "IMBALANCE:", "WHAT IF", "PIPELINE", "MEMORY:"):
        assert heading in text
    assert (tmp_path / "analysis" / "microbatches.csv").is_file()
    assert json.loads((tmp_path / "analysis" / "hetero_report.json").read_text())["run"] == summary["run"]
    assert summary["overview"]["steps"] == 5


def test_sweep_rows_line_up_runs(tmp_path):
    """One row per run, with the step time and the heterogeneity measures."""
    summaries = []
    for name, speeds in (("fixed", None), ("skewed", {0: 1.5})):
        _write_run(tmp_path / name / "hetero", ranks=4, steps=6, speeds=speeds)
        summaries.append(report.analyse(str(tmp_path / name / "hetero"), 0, 4, [2], 3, None)[1])
    lines: list = []
    report.report_sweep(summaries, lines)
    assert len(lines) == 4 and "fixed" in lines[2] and "skewed" in lines[3]
    assert report.sweep_row(summaries[1])[4] > report.sweep_row(summaries[0])[4]
