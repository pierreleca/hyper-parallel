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
"""Host-link benchmark: the scenario plan and the summaries (the copies need a device)."""

import importlib.util
import pathlib

import pytest

_BENCH = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf" / "host_link_bench.py"


def _load_bench():
    """Import the example benchmark by path."""
    spec = importlib.util.spec_from_file_location("host_link_bench", _BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenarios_cover_solo_cards_counts_and_numa_groups():
    """Every die alone, card pairs, growing counts, and the NUMA groups the binding shows."""
    bench = _load_bench()
    numa = [0, 0, 0, 0, 1, 1, 1, 1]  # 8 dies, 4 per NUMA node
    scenarios = dict(bench.build_scenarios(8, numa))
    assert [scenarios[f"alone r{rank}"] for rank in range(8)] == [[rank] for rank in range(8)]
    assert scenarios["one card, both dies (r0, r1)"] == [0, 1]
    assert scenarios["two cards, one die each (r0, r2)"] == [0, 2]
    assert scenarios["first 8 dies"] == list(range(8)) and "first 16 dies" not in scenarios
    assert scenarios["one die per card"] == [0, 2, 4, 6]
    assert scenarios["all dies on NUMA 1"] == [4, 5, 6, 7]
    assert scenarios["one die per NUMA node"] == [0, 4]


def test_scenarios_skip_numa_groups_when_the_binding_is_unknown():
    """Without a CPU binding there is nothing to group by."""
    bench = _load_bench()
    names = [name for name, _ranks in bench.build_scenarios(4, [-1] * 4)]
    assert not any("NUMA" in name for name in names)


def test_bound_numa_names_a_node_only_when_the_mask_sits_on_it():
    """A mask mostly on one node names it; an unrestricted mask names none."""
    bench = _load_bench()
    mapping = {cpu: cpu // 24 for cpu in range(96)}
    assert bench.bound_numa({24, 25, 26, 27, 50}, mapping) == 1, "four of five cores on node 1"
    assert bench.bound_numa(set(range(96)), mapping) == -1, "free to run anywhere, not bound"
    assert bench.bound_numa({24, 25, 50, 51}, mapping) == -1, "split evenly over two nodes"
    assert bench.bound_numa({500}, mapping) == -1


def test_summarize_reports_spread_and_sum():
    """Per-die spread and what the dies moved together."""
    bench = _load_bench()
    summary = bench.summarize([10.0, 30.0, 20.0])
    assert summary == pytest.approx({"min": 10.0, "median": 20.0, "max": 30.0, "sum": 60.0})


def test_report_folds_the_dies_alone_into_one_line_per_direction():
    """The digest lists every die alone on one line, then each other scenario."""
    bench = _load_bench()
    results = []
    for direction, base in (("D2H", 30.0), ("H2D", 40.0)):
        results += [{"scenario": f"alone r{rank}", "direction": direction, "ranks": [rank],
                     **bench.summarize([base + rank])} for rank in range(2)]
        results.append({"scenario": "first 2 dies", "direction": direction, "ranks": [0, 1],
                        **bench.summarize([base / 2, base / 2])})
    lines = bench.report({"world": 2, "gib": 1.0, "repeat": 5, "numa_of_rank": [0, 0], "results": results})
    assert "r0:0 r1:0" in lines[1]
    assert "none" in bench.report({"world": 2, "gib": 1.0, "repeat": 5, "numa_of_rank": [-1, -1],
                                   "results": results})[1]
    assert lines[2].startswith("alone D2H: min 30.0 median 30.5 max 31.0") and lines[2].endswith("r0:30 r1:31")
    assert lines[3].startswith("alone H2D: min 40.0")
    assert [line.split()[0] for line in lines[5:]] == ["first", "first"] and len(lines) == 7
