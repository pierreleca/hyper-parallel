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
"""Ascend trace reader: layout detection, step windows, gaps and the report."""

import importlib.util
import json
import pathlib
import sys

import pytest

_EXAMPLE = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf"


def _load(name):
    """Import one example module by path."""
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _meta(kind, pid, name, tid=None):
    """One process_name / thread_name metadata event."""
    event = {"ph": "M", "name": kind, "pid": pid, "args": {"name": name}}
    if tid is not None:
        event["tid"] = tid
    return event


def _x(name, ts, dur, pid, tid):
    """One complete event; the Ascend exporter writes ts as a string."""
    return {"ph": "X", "name": name, "ts": f"{ts:.3f}", "dur": dur, "pid": pid, "tid": tid}


def _events(a2a_start: float = 300.0) -> list[dict]:
    """Two profiled steps; each sits idle 300 us before its grouped matmul.

    Per step (1000 us): compute 360 us, stream wait 100 us, idle 310 us
    between tasks, 230 us before the first task and after the last one.
    """
    events = [
        _meta("process_name", 1, "Python"), _meta("thread_name", 1, "Thread 1", 1),
        _meta("process_name", 2, "Ascend Hardware"),
        _meta("thread_name", 2, "Stream 2", 10), _meta("thread_name", 2, "Stream 5", 11),
        _meta("process_name", 3, "Communication"), _meta("thread_name", 3, "Plane 0", 20),
    ]
    for step, base in ((6, 1000.0), (7, 2000.0)):
        events.append(_x(f"ProfilerStep#{step}", base, 1000, 1, 1))
        events += [
            _x("aclnnMatmul_MatMulV2_MatMulV2", base + 10, 100, 2, 10),
            _x("aclnnSort_Sort_Sort", base + 110, 10, 2, 10),
            _x("aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul", base + 420, 200, 2, 10),
            _x("aclnnAdd_Add_Add", base + 620, 50, 2, 10),
            _x("aclnnCast_Cast_Cast", base + 700, 5, 2, 11),
            _x("EVENT WAIT", base + 680, 100, 2, 10),
            # Ends with the step's only idle stretch: no compute hides it.
            _x(f"hcom_alltoallv__909_50{step}_1", base + a2a_start, 400 - a2a_start, 3, 20),
            # 70 us under compute, 10 us idle, 5 us inside the stream wait it releases.
            _x(f"hcom_allGather__12_{step}_1", base + 600, 85, 3, 20),
            _x("Notify_Wait", base + 690, 80, 3, 20),
        ]
    return events


def _write(run_dir: pathlib.Path, events: list[dict]) -> None:
    """Write one run directory as torch_npu lays it out."""
    output = run_dir / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    (output / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")


@pytest.fixture(name="trace_path")
def fixture_trace_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """One rank's trace."""
    _write(tmp_path / "run_ascend_pt", _events())
    return tmp_path


def test_layout_steps_and_gaps(trace_path):
    """The compute stream, the step windows and the host wait are found."""
    lib = _load("ascend_trace")
    trace = lib.Trace.load(str(trace_path))
    assert trace.compute_thread() == (2, 10), "the stream with the most aclnn kernels"
    assert trace.steps() == [(6, 1000.0, 2000.0), (7, 2000.0, 3000.0)]

    tasks = lib.window(trace.thread_events((2, 10)), 1000.0, 2000.0)
    kernels, sync = lib.split_sync(tasks)
    assert [event.name for event in sync] == ["EVENT WAIT"], "stream waits are not compute"
    assert lib.busy_time(kernels) == pytest.approx(360.0)
    assert [row["kernels"] for row in trace.streams()] == [8, 2]
    communications = lib.window(trace.communications(), 1000.0, 2000.0)
    assert [lib.comm_type(event.name) for event in communications] == ["alltoallv", "allGather"]
    assert lib.attribute_waits(sync, communications) == [communications[1]], "the collective ending in the wait"

    parts = lib.step_breakdown(tasks, 1000.0, 2000.0)
    assert {name: round(parts[name]) for name in ("compute", "wait", "idle", "edges")} == {
        "compute": 360, "wait": 100, "idle": 310, "edges": 230}, "the four parts add up to the step"
    assert [row["count"] for row in parts["idle_buckets"]] == [0, 1, 1, 0, 0]
    exposure = {row["name"]: row for row in lib.comm_exposure(
        communications, parts["compute_union"], parts["sync_union"])}
    fields = ("total_us", "hidden_us", "exposed_us", "in_wait_us")
    assert {name: round(exposure["allGather"][name]) for name in fields} == {
        "total_us": 85, "hidden_us": 70, "exposed_us": 15, "in_wait_us": 5}
    assert round(exposure["alltoallv"]["exposed_us"]) == 100
    assert round(exposure["all collectives"]["total_us"]) == 185
    assert lib.attribute_waits(sync, communications[:1]) == [None]
    widest, before, after = max(lib.gaps(kernels), key=lambda item: item[0])
    assert widest == pytest.approx(300.0)
    assert before.name.startswith("aclnnSort") and after.name.startswith("aclnnGroupedMatmul")

    top = lib.summarize(kernels)[0]
    assert top["name"] == "aclnnGroupedMatmulV4" and top["total_us"] == pytest.approx(200.0)
    assert lib.category(top["name"]) == "grouped matmul"
    assert lib.category("aclnnSort_Sort_Sort") == "sort / index"

    context = lib.around(kernels, "GroupedMatmul", before=2, after=1, limit=5)
    assert len(context) == 1
    offsets = [(offset, round(gap)) for offset, gap, _event in context[0]]
    assert offsets == [(-2, 0), (-1, 0), (0, 300), (1, 0)]
    assert lib.comm_name("hcom_allGather__123_4_1") == "hcom_allGather"

    later = lib.around(tasks, "aclnn", before=0, after=1, limit=5, skip=1)
    assert [context[0][2].name[:10] for context in later] == ["aclnnSort_", "aclnnAdd_A"], "no overlapping contexts"


def test_report_runs_end_to_end(trace_path, capsys, monkeypatch):
    """The report prints every section and writes its tables."""
    _load("ascend_trace")  # the report imports its library by name, as a script run would find it
    report = _load("analyze_npu_trace")
    monkeypatch.setattr(sys, "argv", ["analyze_npu_trace.py", str(trace_path), "--around", "GroupedMatmul"])
    assert report.main() == 0
    printed = capsys.readouterr().out
    for heading in ("compute stream: Ascend Hardware / Stream 2", "step 6:", "DETAIL: step 7",
                    "compute     0.4 ( 36%) + stream wait     0.1 ( 10%) + idle     0.3 ( 31%)",
                    "mean of 2:    1.0 ms = compute     0.4",
                    "1       0.10 100.0%  allGather", "allGather 85", "idle between tasks: 0.3 ms",
                    "1        0.09       0.07       0.01       0.01     82%  allGather", "around 'GroupedMatmul'"):
        assert heading in printed, heading
    analysis = next(trace_path.rglob("analysis"))
    assert (analysis / "kernels.csv").read_text(encoding="utf-8").count("\n") == 1 + 5


def test_ranks_split_the_alltoallv_into_skew_and_transfer(tmp_path, capsys, monkeypatch):
    """Rank 1 joins the alltoallv 50 us late: rank 0 waits 50 us, then both move data 50 us."""
    _write(tmp_path / "rank0_1_ascend_pt", _events(a2a_start=300.0))
    _write(tmp_path / "rank1_1_ascend_pt", _events(a2a_start=350.0))
    lib = _load("ascend_trace")
    assert list(lib.find_rank_traces(str(tmp_path))) == [0, 1]
    report = _load("analyze_npu_trace")
    monkeypatch.setattr(sys, "argv", ["analyze_npu_trace.py", str(tmp_path), "--ranks", "--list-matched"])
    assert report.main() == 0
    printed = capsys.readouterr().out
    assert "MATCHED alltoallv: 1 per rank" in printed
    summary = json.loads((tmp_path / "analysis_ranks" / "ranks.json").read_text(encoding="utf-8"))
    matched = summary["alltoallv"]
    assert round(matched["skew_ms"] * 1e3) == 50 and round(matched["transfer_ms"] * 1e3) == 50
    assert {rank: round(value * 1e3) for rank, value in matched["waited_ms"].items()} == {"0": 50, "1": 0}
    assert matched["last"] == {"0": 0, "1": 1}
