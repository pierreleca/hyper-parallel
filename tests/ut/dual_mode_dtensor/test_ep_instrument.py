# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Expert-parallel imbalance instrumentation: recorder and its analysis.

The recorder is exercised single-process on CPU, where it falls back to the
host clock and reports no allocator bytes; the analysis is exercised on
synthetic records with a known imbalance, so the reported factors can be
checked against arithmetic.
"""

# The recorder is a module-level singleton and the tests read its private
# bookkeeping to assert what it captured.
# pylint: disable=protected-access

import importlib.util
import json
import pathlib
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
)
from hyper_parallel.distributed.expert_parallel.instrument import EP_INSTRUMENT
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS

_ANALYZER = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf" / "analyze_ep_instrument.py"


def _load_analyzer():
    """Import the example analysis script by path."""
    spec = importlib.util.spec_from_file_location("analyze_ep_instrument", _ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Experts(nn.Module):
    """Stacked SwiGLU experts in the fused gate_up layout."""

    def __init__(self, num_experts=8, hidden=16, inter=8):
        """Create expert weights small enough for a CPU test."""
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.02)


class _Moe(nn.Module):
    """Minimal MoE block with the interface ep_routed_forward expects."""

    def __init__(self, num_experts=8, hidden=16, inter=8, top_k=2):
        """Build the router and the stacked experts."""
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = _Experts(num_experts, hidden, inter)
        self.top_k = top_k


@pytest.fixture(name="world_one_group")
def fixture_world_one_group():
    """Provide a single-process gloo group for the all-to-all primitives."""
    created = False
    if not dist.is_initialized():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
        created = True
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


@pytest.fixture(name="recorder_off")
def fixture_recorder_off():
    """Leave the module-level recorder disabled after each test."""
    yield
    EP_INSTRUMENT.configure(enabled=False, output_dir="")
    EP_INSTRUMENT.close()
    EP_INSTRUMENT._header_written = False


def test_recorder_captures_every_pass(tmp_path, world_one_group, recorder_off):
    """A checkpointed step yields forward, recompute and backward records."""
    from hyper_parallel.core.activation_memory.checkpoint import checkpoint

    EP_INSTRUMENT.configure(
        enabled=True, output_dir=str(tmp_path), align_steps=False
    )
    model = nn.ModuleList([_Moe(), _Moe()])
    for block in model:
        bind_local_expert_forward(block, ep_size=1)
    EP_INSTRUMENT.register_modules(model)

    EP_INSTRUMENT.begin_step(1)
    hidden = torch.randn(1, 6, 16, requires_grad=True)
    out = hidden
    for block in model:
        out = checkpoint(
            lambda x, block=block: ep_routed_forward(
                block, x, router_fn=MOE_ROUTER_ADAPTERS["default"],
                ep_group=world_one_group,
            ),
            out,
        )
    out.sum().backward()
    record = EP_INSTRUMENT.end_step()
    EP_INSTRUMENT.close()

    passes = {mark[1] for mark in record["marks"]}
    assert passes == {"step", "fwd", "recompute", "bwd"}, "every pass is recorded"
    fwd_names = [mark[3] for mark in record["marks"] if mark[1] == "fwd" and mark[0] == 0]
    assert fwd_names == ["start", "routed", "dispatched", "experts", "combined", "end"]
    bwd_names = [mark[3] for mark in record["marks"] if mark[1] == "bwd" and mark[0] == 0]
    assert bwd_names == ["start", "aggregate", "combine", "experts", "dispatch", "end"]
    assert [mark[4] for mark in record["marks"]] == sorted(mark[4] for mark in record["marks"]), \
        "stamps are resolved in recording order"

    forward_calls = [call for call in record["calls"] if call["pass"] == "fwd"]
    assert len(forward_calls) == 2, "one forward call per MoE block"
    for call in forward_calls:
        assert sum(call["expert_counts"]) == call["tokens"] * call["top_k"]
        assert sum(call["send"]) == sum(call["recv"]) == call["tokens"] * call["top_k"]
    assert all("expert_counts" not in call for call in record["calls"]
               if call["pass"] == "recompute"), "counts are taken once per step"

    digest = EP_INSTRUMENT.step_summary(record)
    assert digest["pairs"] == sum(sum(call["recv"]) for call in forward_calls)
    assert digest["fwd_ms"] > 0 and digest["recompute_ms"] > 0 and digest["bwd_ms"] > 0

    lines = [
        json.loads(line)
        for line in (tmp_path / "rank000.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["kind"] == "header"
    assert lines[0]["layers"] == {"0": "0", "1": "1"}
    assert lines[0]["num_experts"] == 8 and lines[0]["intermediate"] == 8
    assert lines[1]["kind"] == "step" and lines[1]["step"] == 1
    assert all(len(mark) == 9 for mark in lines[1]["marks"]), "reserved bytes and their peak recorded"

    analyzer = _load_analyzer()
    headers, steps = analyzer.load_records(str(tmp_path), skip=0)
    collected = analyzer.collect(headers, steps)
    assert {row["layer"] for row in collected["rows"]} == {0, 1}, "the analysis reads real records"


def test_disabled_recorder_leaves_the_forward_untouched(world_one_group, recorder_off):
    """With the recorder off, the routed forward records nothing."""
    EP_INSTRUMENT.configure(enabled=False, output_dir="")
    block = _Moe()
    bind_local_expert_forward(block, ep_size=1)
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    out = ep_routed_forward(
        block, hidden, router_fn=MOE_ROUTER_ADAPTERS["default"], ep_group=world_one_group
    )
    out.sum().backward()
    assert EP_INSTRUMENT.open_call(block, world_one_group) is None
    assert EP_INSTRUMENT.end_step() is None


def test_fold_peaks_survives_segment_resets(recorder_off):
    """The recorder returns its own running peak once it resets the counters."""
    EP_INSTRUMENT.configure(enabled=True, output_dir="", segment_peaks=True)
    EP_INSTRUMENT._run_peaks = [4096, 8192]
    assert EP_INSTRUMENT.fold_peaks(10, 20) == (4096, 8192)
    assert EP_INSTRUMENT.fold_peaks(9000, 100) == (9000, 8192)
    EP_INSTRUMENT.configure(enabled=True, output_dir="", segment_peaks=False)
    assert EP_INSTRUMENT.fold_peaks(10, 20) == (10, 20), "untouched when peaks are not read"


def _synthetic_records(directory, loads, peaks):
    """Write one file per rank with a known load and memory profile.

    Rank ``r`` receives ``loads[r]`` pairs in every layer, its expert GEMM
    lasts one millisecond per ten pairs, and its step peak is ``peaks[r]``.
    """
    for rank, load in enumerate(loads):
        header = {
            "kind": "header", "rank": rank, "world_size": len(loads), "host": "test",
            "device_type": "npu", "time_source": "device", "segment_peaks": True,
            "layers": {"0": "layers.0", "1": "layers.1"},
            "num_experts": 8, "top_k": 2, "hidden": 16, "element_size": 2,
            "local_experts": 2, "intermediate": 8, "expert_element_size": 2,
        }
        lines = [json.dumps(header)]
        for step in (1, 2):
            marks, calls, time_ms = [[-1, "step", 0, "start", 0.0, 0, 0]], [], 1.0
            for layer in (0, 1):
                for name, span in (("start", 0.0), ("routed", 1.0), ("dispatched", 1.0),
                                   ("experts", load / 10.0), ("combined", 1.0), ("end", 0.5)):
                    time_ms += span
                    # The block keeps 4 KiB per routed pair once it ends.
                    alloc = 2 ** 30 + (load * 4096 if name == "end" else 0)
                    marks.append([layer, "fwd", 0, name, round(time_ms, 3), alloc, peaks[rank]])
                time_ms += 3.0  # the gap a transfer could hide in
                calls.append({
                    "layer": layer, "pass": "fwd", "occurrence": 0, "tokens": 100,
                    "top_k": 2, "hidden": 16, "send": [load], "recv": [load],
                    "expert_counts": [load // 8] * 8,
                })
            marks.append([-1, "step", 0, "end", round(time_ms, 3), 0, peaks[rank]])
            lines.append(json.dumps({
                "kind": "step", "step": step, "wall_s": 0.1,
                "memory": {"step_peak_allocated": peaks[rank], "step_peak_reserved": peaks[rank]},
                "marks": marks, "calls": calls,
            }))
        (directory / f"rank{rank:03d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analysis_reports_known_imbalance(tmp_path, capsys):
    """The analysis recovers the imbalance, the idle time and the headroom."""
    analyzer = _load_analyzer()
    loads = [150, 100, 80, 70]  # mean 100, so lambda = 1.5
    peaks = [8 * 1024 ** 3, 6 * 1024 ** 3, 6 * 1024 ** 3, 6 * 1024 ** 3]
    _synthetic_records(tmp_path, loads, peaks)

    headers, steps = analyzer.load_records(str(tmp_path), skip=0)
    collected = analyzer.collect(headers, steps)
    table = analyzer._by_step_layer(collected["rows"])
    out = []

    routing = analyzer.report_routing(table, collected["ranks"], out)
    assert routing["lambda_mean"] == pytest.approx(1.5), "max / mean of the loads"
    assert routing["lambda_max"] == pytest.approx(1.5)
    assert "r0 100%" in "\n".join(out), "rank 0 is always the busiest"

    persistence = analyzer.report_persistence(collected, out)
    assert persistence["lambda_step_mean"] == pytest.approx(1.5), "same rank in every layer"

    timing = analyzer.report_time(table, collected["ranks"], out)
    assert timing["expert_time_lambda_mean"] == pytest.approx(1.5), "time follows the load"
    # Rank 3 waits (150 - 70) / 10 ms per layer, over two layers.
    assert timing["idle_ms_per_step"]["3"] == pytest.approx(16.0)
    assert timing["idle_ms_per_step"]["0"] == pytest.approx(0.0)

    memory = analyzer.report_memory(collected, out)
    assert memory["peak_allocated_gib"]["0"] == pytest.approx(8.0)
    assert memory["peak_spread_gib"] == pytest.approx(2.0)

    offload = analyzer.report_offload(collected, headers, table, out)
    # (hidden 16 + 3 x intermediate 8) x 2 B = 80 B per pair, 50 pairs above the mean.
    assert offload["bytes_per_pair"] == 80
    assert offload["excess_mib_per_layer"] == pytest.approx(50 * 80 / 1024 ** 2)
    assert offload["median_gap_ms"] == pytest.approx(3.0)

    analyzer.write_outputs(collected, str(tmp_path / "analysis"))
    rows = (tmp_path / "analysis" / "layer_step.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(loads) * 2 * 2, "header, then one row per rank, layer and step"
    capsys.readouterr()


def test_analysis_subtracts_the_recompute_from_the_backward(tmp_path):
    """Recompute time inside a backward phase is not counted twice."""
    analyzer = _load_analyzer()
    marks = [
        [0, "bwd", 0, "start", 0.0, 0, 0],
        [0, "recompute", 0, "start", 1.0, 0, 0],
        [0, "recompute", 0, "end", 4.0, 0, 0],
        [0, "bwd", 0, "aggregate", 5.0, 0, 0],
        [0, "bwd", 0, "combine", 6.0, 0, 0],
    ]
    times = analyzer.phase_times({"marks": marks})
    assert times[(0, "bwd", "aggregate bwd")] == pytest.approx(2.0), "5 ms minus 3 ms of recompute"
    assert times[(0, "bwd", "combine a2a bwd")] == pytest.approx(1.0)
    assert tmp_path.exists()


def test_register_modules_reads_dtensor_expert_weights(make_mesh, recorder_off):
    """Expert weights are DTensors once sharded; reading their shape must not dispatch.

    ``element_size()`` has no DTensor layout rule and raised at train begin;
    attribute reads such as ``dtype`` and ``shape`` pass through.
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor
    from hyper_parallel.core.dtensor.placement_types import Replicate

    mesh = make_mesh((1,), ("ep",))
    block = _Moe(inter=8)
    bind_local_expert_forward(block, ep_size=1)
    local = block.experts.gate_up_proj.detach().to(torch.bfloat16)
    block.experts.gate_up_proj = nn.Parameter(DTensor.from_local(local, mesh, [Replicate()]))

    EP_INSTRUMENT.configure(enabled=True, output_dir="", align_steps=False)
    EP_INSTRUMENT._experts = {}
    EP_INSTRUMENT.register_modules(nn.ModuleList([block]))
    assert EP_INSTRUMENT._experts == {
        "local_experts": 8, "intermediate": 8, "expert_element_size": 2,
    }


def test_register_modules_reads_the_grouped_experts_layout(recorder_off):
    """GroupedExperts stores gate_up as [E, H, 2I]; the intermediate size is still I.

    Halving axis 1 gave H / 2 (1024 instead of 768 for Qwen3-VL-30B).
    """
    block = _Moe(hidden=16, inter=12)
    bind_local_expert_forward(block, ep_size=1)
    block.experts.gate_up_proj = nn.Parameter(torch.zeros(8, 16, 24))  # [E, H, 2I]
    block.experts.down_proj = nn.Parameter(torch.zeros(8, 12, 16))     # [E, I, H]

    EP_INSTRUMENT.configure(enabled=True, output_dir="", align_steps=False)
    EP_INSTRUMENT._experts = {}
    EP_INSTRUMENT.register_modules(nn.ModuleList([block]))
    assert EP_INSTRUMENT._experts["intermediate"] == 12


def test_analysis_reports_layer_memory_and_capacity(tmp_path, capsys):
    """Per-block memory follows the load, and the capacity sweep spills the excess."""
    analyzer = _load_analyzer()
    loads = [150, 100, 80, 70]
    _synthetic_records(tmp_path, loads, [2 ** 31] * 4)
    headers, steps = analyzer.load_records(str(tmp_path), skip=0)
    collected = analyzer.collect(headers, steps)
    table = analyzer._by_step_layer(collected["rows"])
    out = []

    layer_memory = analyzer.report_layer_memory(table, collected["ranks"], out)
    assert layer_memory["retained_bytes_per_pair"] == pytest.approx(4096)
    assert layer_memory["retained_lambda_per_layer_mean"] == pytest.approx(1.5)
    assert layer_memory["retained_lambda_summed"] == pytest.approx(1.5)
    assert layer_memory["bytes_per_received_pair"] == pytest.approx(4096)
    assert layer_memory["fixed_bytes"] == pytest.approx(0.0, abs=1e-6)

    rows = {row["capacity_factor"]: row for row in analyzer.report_capacity(
        table, collected["ranks"], 4096, "measured", out)}
    # Mean load 100: at C = 1 rank 0 is 50 pairs over, 50 / 400 of the tokens.
    assert rows[1.0]["tokens_over_share"] == pytest.approx(0.125)
    assert rows[1.0]["layers_over_share"] == pytest.approx(1.0)
    assert rows[1.0]["hot_spill_max_mib"] == pytest.approx(50 * 4096 / 1024 ** 2)
    assert rows[1.2]["tokens_over_share"] == pytest.approx(30 / 400)
    assert rows[1.5]["tokens_over_share"] == 0.0 and rows[1.5]["hot_spill_max_mib"] == 0.0
    assert rows[1.5]["reserved_mib"] == pytest.approx(1.5 * 100 * 4096 / 1024 ** 2)
    capsys.readouterr()


def test_fit_separates_the_receive_side_from_the_fixed_part():
    """Only the slope on received pairs follows the routing."""
    analyzer = _load_analyzer()
    # retained = 100 B per received pair + 5000 B the rank holds regardless.
    entries = [(100 * pairs + 5000, 0, pairs) for pairs in (70, 80, 100, 150)]
    fit = analyzer._fit_receive_side(entries)
    assert fit["bytes_per_received_pair"] == pytest.approx(100)
    assert fit["fixed_bytes"] == pytest.approx(5000)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_analysis_reports_reserved_memory_and_its_growth(tmp_path, capsys):
    """Cached-but-unused bytes that grow over the steps show up as a trend."""
    analyzer = _load_analyzer()
    _synthetic_records(tmp_path, [150, 100, 80, 70], [2 ** 31] * 4)
    for path in tmp_path.glob("rank*.jsonl"):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in lines[1:]:
            # Reserved runs 1 GiB above allocated in step 1, 3 GiB in step 2.
            extra = (2 * record["step"] - 1) * 2 ** 30
            record["marks"] = [mark + [(mark[5] or 0) + extra, None] for mark in record["marks"]]
            record["memory"]["step_peak_reserved"] = 2 ** 31 + extra
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    headers, steps = analyzer.load_records(str(tmp_path), skip=0)
    collected = analyzer.collect(headers, steps)
    table = analyzer._by_step_layer(collected["rows"])
    out = []
    memory = analyzer.report_memory(collected, out)
    assert memory["peak_reserved_gib"]["0"] == pytest.approx(4.0), "mean of 3 and 5 GiB"
    # Reserved peak minus allocated peak: 1 GiB in step 1, 3 GiB in step 2.
    assert memory["reserve_above_peak_gib"]["0"] == {"first_third": 1.0, "last_third": 3.0}

    routing = analyzer.report_routing(table, collected["ranks"], out)
    assert routing["lambda_first_third"] == pytest.approx(1.5)
    assert "drift: mean lambda" in "\n".join(out)
    capsys.readouterr()


def test_reserved_growth_is_located_and_tagged_with_the_swaps(tmp_path):
    """A reserve that rises inside one segment of one step is reported there, with that step's swap."""
    analyzer = _load_analyzer()
    records = tmp_path / "records"
    records.mkdir()
    _synthetic_records(records, [150, 100, 80, 70], [2 ** 31] * 4)
    base = 10 * 2 ** 30
    for path in records.glob("rank*.jsonl"):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in lines[1:]:
            grown = False
            for mark in record["marks"]:
                if record["step"] == 2 and path.name == "rank000.jsonl" and mark[:4] == [1, "fwd", 0, "experts"]:
                    grown = True
                    mark += [base, base + 2 ** 29]  # the segment's peak passes the reserve by 512 MiB
                else:
                    mark += [base + (2 ** 29 if grown else 0), None]
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    swap_dir = tmp_path / "swap"
    swap_dir.mkdir()
    (swap_dir / "host_swap_rank0.jsonl").write_text("\n".join([
        json.dumps({"header": True, "rank": 0}),
        json.dumps({"step": 2, "rank": 0, "layers": [{"index": 1}]}),
    ]) + "\n", encoding="utf-8")

    _headers, steps = analyzer.load_records(str(records), skip=0)
    out = []
    growth = analyzer.report_reserved_growth(steps, out, analyzer.load_swaps(str(swap_dir)))
    assert growth["0"]["growth_gib"] == pytest.approx(0.5)
    assert growth["0"]["events"] == [{"step": 2, "segment": "L1:fwd:dispatched -> L1:fwd:experts", "mib": 512.0}]
    assert growth["1"]["growth_gib"] == 0.0 and not growth["1"]["events"]
    assert "[swapped L1]" in "\n".join(out)


def test_compare_finds_the_first_diverging_step(tmp_path):
    """Two identical runs compare equal; a changed step is located."""
    analyzer = _load_analyzer()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _synthetic_records(first, [150, 100, 80, 70], [2 ** 31] * 4)
    _synthetic_records(second, [150, 100, 80, 70], [2 ** 31] * 4)

    out = []
    assert analyzer.compare_runs(str(first), str(second), out)["records_differing"] == 0
    assert "route identically" in "\n".join(out)

    path = second / "rank002.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[2]["calls"][1]["expert_counts"][0] += 3   # step 2, layer 1
    lines[2]["calls"][1]["expert_counts"][1] -= 3
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    out = []
    summary = analyzer.compare_runs(str(first), str(second), out)
    assert summary["first_differing_step"] == 2 and summary["records_differing"] == 1
    assert "step 2, rank 2, layer 1 (3 token-expert assignments moved)" in "\n".join(out)


def test_step_budget_compares_one_budget_with_per_layer_budgets(tmp_path, capsys):
    """Totals over the layers, their worst case and the eviction a budget implies."""
    analyzer = _load_analyzer()
    loads = [150, 100, 80, 70]
    _synthetic_records(tmp_path, loads, [2 ** 31] * 4)
    headers, steps = analyzer.load_records(str(tmp_path), skip=0)
    collected = analyzer.collect(headers, steps)
    table = analyzer._by_step_layer(collected["rows"])
    out = []
    budget = analyzer.report_step_budget(table, out)
    # Each rank keeps 4 KiB per received pair in each of 2 layers.
    mean_total = 2 * 100 * 4096
    assert budget["mean_total_gib"] * 2 ** 30 == pytest.approx(mean_total)
    assert budget["max_total_gib"] * 2 ** 30 == pytest.approx(1.5 * mean_total)
    assert budget["per_layer_worst_sum_gib"] == pytest.approx(budget["max_total_gib"])
    at_mean = budget["sweep"][0]
    assert at_mean["factor"] == 1.0 and at_mean["over_share"] == pytest.approx(0.25), "only rank 0"
    assert at_mean["worst_eviction_gib"] * 2 ** 30 == pytest.approx(0.5 * mean_total)
    capsys.readouterr()


def test_compare_reports_the_slowest_rank_step_time(tmp_path):
    """The step time is the slowest rank's, compared over the steps both runs kept."""
    analyzer = _load_analyzer()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _synthetic_records(first, [150, 100, 80, 70], [2 ** 31] * 4)
    _synthetic_records(second, [150, 100, 80, 70], [2 ** 31] * 4)
    path = first / "rank001.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[2]["wall_s"] = 0.3  # step 2 of rank 1 in the first run
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    summary = analyzer.compare_runs(str(first), str(second), [], skip=1)
    assert summary["step_s_mean"] == pytest.approx((0.1, 0.3)), "only step 2 is kept, rank 1 sets it"


def test_swap_activity_summarizes_the_swap_records(tmp_path):
    """Per rank: steps swapping, bytes and copy times, warm-up steps left out."""
    analyzer = _load_analyzer()
    base = {"swapped_layers": 1, "d2h_gib": 0.5, "d2h_gbps": 30.0, "h2d_gbps": 40.0,
            "h2d_hidden_ms": 10.0, "stall_ms": 2.0, "evictions": [{}]}
    lines = [{"header": True, "budget": "step"},
             {**base, "step": 1, "rank": 0, "d2h_gib": 9.0},  # warm-up
             {**base, "step": 2, "rank": 0},
             {**base, "step": 3, "rank": 0, "swapped_layers": 0, "d2h_gib": 0.0, "d2h_gbps": None,
              "h2d_gbps": None, "h2d_hidden_ms": 0.0, "stall_ms": 0.0, "evictions": []}]
    (tmp_path / "host_swap_rank0.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n",
                                                    encoding="utf-8")
    out = []
    summary = analyzer.report_swap_activity(str(tmp_path), 1, out)
    row = summary["ranks"][0]
    assert summary["budget"] == "step" and row["steps"] == 2 and row["steps_swapping"] == 1
    assert row["gib_per_step"] == pytest.approx(0.25) and row["d2h_gbps"] == pytest.approx(30.0)
    assert row["exposed_ms"] == pytest.approx(1.0) and row["evictions_per_step"] == pytest.approx(0.5)


def test_sweep_compares_runs_one_row_each(tmp_path, capsys):
    """Each run contributes a row; the deltas are taken against the first."""
    analyzer = _load_analyzer()
    for name, peak, moved in (("noswap", 8 * 1024 ** 3, None), ("f090", 7 * 1024 ** 3, 0.5)):
        run = tmp_path / name
        (run / "instrument").mkdir(parents=True)
        _synthetic_records(run / "instrument", [150, 100, 80, 70], [peak] * 4)
        if moved is None:
            continue
        (run / "ep_host_swap").mkdir()
        lines = [{"header": True, "budget": "step"}] + [
            {"step": step, "rank": 0, "swapped_layers": 1, "d2h_gib": moved, "d2h_gbps": 30.0,
             "h2d_gbps": 40.0, "h2d_hidden_ms": 9.0, "stall_ms": 4.0, "pinned_gib": 2.0, "evictions": [{}]}
            for step in (1, 2)]
        (run / "ep_host_swap" / "host_swap_rank0.jsonl").write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    out = []
    rows = analyzer.report_sweep([str(tmp_path / "noswap"), str(tmp_path / "f090")], 1, out)
    assert [row["run"] for row in rows] == ["noswap", "f090"]
    assert rows[0]["moved_gib"] == 0.0 and rows[0]["d2h_min"] is None, "no swap records, no swap columns"
    assert rows[1]["moved_gib"] == pytest.approx(0.5) and rows[1]["d2h_min"] == pytest.approx(30.0)
    assert rows[1]["exposed_max"] == pytest.approx(4.0) and rows[1]["pinned_max"] == pytest.approx(2.0)
    assert rows[0]["reserved_worst"] == pytest.approx(8.0) and rows[1]["reserved_worst"] == pytest.approx(7.0)
    assert "-1.00 GiB" in "\n".join(out), "the reserved delta against the baseline"
    capsys.readouterr()
