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
"""The report on simulated runs with the structure of the real one: exchange, ceilings, assignment, groups, regions."""

import importlib.util
import json
import pathlib
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


_load("hetero_sampling")
synthetic = _load("synthetic_hetero_records")
report = _load("analyze_hetero")
compare = _load("compare_runs")

SMALL = {"ranks": 8, "steps": 6, "accumulation": 1, "layers": 4, "blocks": 4, "ep_size": 4, "experts": 16, "seed": 2,
         "slow_rank": None}


def _make(directory: pathlib.Path, scenario: str = "both", **overrides) -> pathlib.Path:
    """Simulate a small run into ``directory/hetero``."""
    options = {**SMALL, **overrides}
    synthetic.write_run(str(directory / "hetero"), scenario=scenario, **options)
    return directory / "hetero"


def test_records_are_labelled_synthetic_and_the_report_says_so(tmp_path):
    """The header and a note file say it; the report starts with a banner."""
    records = _make(tmp_path)
    assert (records / "SYNTHETIC.txt").is_file()
    lines, summary = report.analyse(str(records), 1, 4, [2], 3, None)
    assert summary["synthetic"] and "SYNTHETIC RECORDS" in lines[1]


def test_every_span_is_ordered_and_a_rank_never_works_on_two_modules_at_once(tmp_path):
    """The simulated timeline is coherent: in before out, and the leaf modules of a rank do not overlap."""
    run = report.Run(str(_make(tmp_path)))
    leaves = {"vision.block", "text.attn", "text.experts", "lm_head"}
    for rank in run.ranks:
        roles = {m["id"]: m["role"] for m in run.headers[rank]["modules"]}
        for record in run.steps[rank]:
            intervals = []
            for (module_id, pass_name, _occ), span in report.spans_of(record).items():
                assert span["out"] >= span["in"]
                if roles[module_id] in leaves and pass_name != "bwd":
                    intervals.append((span["in"], span["out"]))
            intervals.sort()
            assert all(later[0] >= earlier[1] - 1e-6 for earlier, later in zip(intervals, intervals[1:]))


def test_the_moe_exchange_is_taken_out_of_the_layers_own_work(tmp_path):
    """The block's span minus the experts' is the exchange; the layer then holds attention, experts and norms only."""
    rows = report.build_rows(report.Run(str(_make(tmp_path)), 0))
    row = rows[0]
    assert row["has_experts"] and sum(row["t"]["ep_exchange"].values()) > 0
    layer = report.component_time(row, "text_layer")
    attention = report.component_time(row, "text_attn")
    experts = report.component_time(row, "text_experts")
    moe = report.component_time(row, "text_moe")
    exchange = report.component_time(row, "ep_exchange")
    assert exchange == pytest.approx(moe - experts, rel=1e-6)
    assert layer == pytest.approx(attention + experts, rel=0.2)       # plus the norms, which are small


def test_expert_time_follows_the_pairs_received_not_the_samples_own_tokens(tmp_path):
    """Experts work on what their group sends them: the fit on received pairs is exact, on the own tokens it is not."""
    run = report.Run(str(_make(tmp_path, steps=10)), 0)
    rows = report.build_rows(run)
    without = report.fit_components(rows)["text_experts"]["r2"]
    assert report.attach_received_pairs(run, rows, 4) == len(rows)
    with_pairs = report.fit_components(rows)["text_experts"]["r2"]
    assert with_pairs > 0.99 and without < 0.8


def test_balancing_every_module_has_a_higher_ceiling_than_balancing_the_rank_totals(tmp_path):
    """Each module is a barrier, so the slowest rank counts per module: the sum of those excesses is the larger."""
    records = _make(tmp_path, steps=8)
    _, summary = report.analyse(str(records), 1, 4, [2], 3, None)
    ceilings = summary["ceilings"]
    data = ceilings["balance the data: every rank carries the mean total work"]["ms"]
    modules = ceilings["balance every module's work across ranks (each layer is a barrier)"]["ms"]
    assert modules >= data > 0
    assert ceilings["needed_share"] == pytest.approx(1 - 1 / 1.2)
    for name, entry in ceilings.items():
        if isinstance(entry, dict):
            assert 0.0 <= entry["share"] <= 1.0, name


def test_a_data_balanced_run_has_a_lower_ceiling_than_a_random_one(tmp_path):
    """The balanced order of the same samples leaves less for any balancing change to gain."""
    random_run = _make(tmp_path / "random", steps=8)
    balanced = _make(tmp_path / "balanced", steps=8, order="balanced")
    first = report.analyse(str(random_run), 1, 4, [2], 3, None)[1]["imbalance"]["busiest_over_mean"]
    second = report.analyse(str(balanced), 1, 4, [2], 3, None)[1]["imbalance"]["busiest_over_mean"]
    assert second < first


def test_dealing_by_cost_needs_more_than_one_micro_batch_per_rank(tmp_path):
    """With one sample per rank no assignment helps; with two or more, dealing biggest first does."""
    _, summary = report.analyse(str(_make(tmp_path, ranks=8, steps=24)), 0, 4, [2], 3, None)
    one, two = summary["assignment"]["1"], summary["assignment"]["2"]
    assert one["by_cost"] == pytest.approx(one["arrival"])
    assert two["by_cost"] < two["arrival"] and two["saved"] > 0.05


def test_ranks_that_hold_other_modules_form_their_own_group(tmp_path):
    """A design that places the vision tower apart shows as two kinds of rank, each with its own utilisation."""
    records = _make(tmp_path)
    for rank in range(4, 8):                                    # these ranks keep only the vision tower
        path = records / f"rank{rank:03d}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        keep = {m["id"] for m in lines[0]["modules"] if not m["role"].startswith(("text.", "lm_head"))}
        lines[0]["modules"] = [m for m in lines[0]["modules"] if m["id"] in keep]
        for record in lines[1:]:
            record["marks"] = [mark for mark in record["marks"] if mark[0] in keep]
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    run = report.Run(str(records), 0)
    groups = report.report_groups(run, report.build_rows(run), [])
    assert len(groups) == 2
    assert all(0.0 < group["utilisation"] <= 1.0 for group in groups.values())


def test_a_custom_component_is_work_and_is_charged_to_the_rank_that_ran_it(tmp_path):
    """A region the design adds appears as 'custom', counts in the rank's work, and in the imbalance attribution."""
    records = _make(tmp_path)
    for rank in range(8):
        path = records / f"rank{rank:03d}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        module_id = len(lines[0]["modules"])
        lines[0]["modules"].append({"id": module_id, "name": "redistribute", "role": "custom", "index": None})
        for record in lines[1:]:
            duration = 5.0 + 20.0 * (rank == 3)                 # rank 3 does the redistribution's heavy part
            record["marks"] += [[module_id, "fwd", 0, "in", 1.0, 0], [module_id, "fwd", 0, "out", 1.0 + duration, 0]]
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    run = report.Run(str(records), 0)
    rows = report.build_rows(run)
    by_rank = {row["rank"]: row for row in rows}
    assert report.component_time(by_rank[3], "custom") == pytest.approx(25.0)
    assert report.component_time(by_rank[0], "custom") == pytest.approx(5.0)
    lines: list = []
    imbalance = report.report_imbalance(rows, {}, lines)
    assert imbalance["excess_by_part"]["custom"] > 0
    assert any("custom" in line for line in lines)


def test_records_without_module_hooks_still_analyse(tmp_path):
    """A light-mode run has no marks: the data sections work and the component sections are empty, not broken."""
    records = _make(tmp_path)
    for rank in range(8):
        path = records / f"rank{rank:03d}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        lines[0].update(modules=[], hooks=False, time_source="device")
        for record in lines[1:]:
            record.update(marks=[], routing=[])
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    text, summary = report.analyse(str(records), 1, 4, [2], 3, None)
    assert summary["overview"]["steps"] == 5 and summary["data"]["real_tokens"]["mean"] > 0
    assert "DATA:" in "\n".join(text)


def test_the_candidate_with_a_balanced_order_is_credited_on_work_not_on_time(tmp_path):
    """Balanced against random order, same samples: the comparison reports work per second and the imbalance drop."""
    random_run = _make(tmp_path / "random", steps=12, seed=3)
    balanced = _make(tmp_path / "balanced", steps=12, seed=3, order="balanced")
    lines, summary = compare.compare(compare.Arm([str(random_run)], 1), compare.Arm([str(balanced)], 1), skip=1,
                                     draws=200)
    assert summary["work"]["speedup"] > 0.1
    assert summary["components"]["candidate"]["busiest_over_mean"] < summary["components"]["baseline"][
        "busiest_over_mean"]
    assert any("where the time moved" in line for line in lines)
