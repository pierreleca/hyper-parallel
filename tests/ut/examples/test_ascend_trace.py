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


@pytest.fixture(name="trace_path")
def fixture_trace_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two profiled steps; each waits 300 us for the host before its grouped matmul."""
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
            _x("hcom_alltoallv_", base + 300, 100, 3, 20),
        ]
    output = tmp_path / "run_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    (output / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
    return tmp_path


def test_layout_steps_and_gaps(trace_path):
    """The compute stream, the step windows and the host wait are found."""
    lib = _load("ascend_trace")
    trace = lib.Trace.load(str(trace_path))
    assert trace.compute_thread() == (2, 10), "the stream with the most aclnn kernels"
    assert trace.steps() == [(6, 1000.0, 2000.0), (7, 2000.0, 3000.0)]

    kernels = lib.window(trace.thread_events((2, 10)), 1000.0, 2000.0)
    assert lib.busy_time(kernels) == pytest.approx(360.0)
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

    collectives = lib.window(trace.collectives(), 1000.0, 2000.0)
    assert [event.name for event in collectives] == ["hcom_alltoallv_"]


def test_report_runs_end_to_end(trace_path, capsys, monkeypatch):
    """The report prints every section and writes its tables."""
    report = _load("analyze_npu_trace")
    monkeypatch.setattr(sys, "argv", ["analyze_npu_trace.py", str(trace_path), "--around", "GroupedMatmul"])
    assert report.main() == 0
    printed = capsys.readouterr().out
    for heading in ("compute stream: Ascend Hardware / Stream 2", "step 6:", "DETAIL: step 7",
                    "longest", "collectives", "around 'GroupedMatmul'"):
        assert heading in printed, heading
    analysis = next(trace_path.rglob("analysis"))
    assert (analysis / "kernels.csv").read_text(encoding="utf-8").count("\n") == 1 + 4
