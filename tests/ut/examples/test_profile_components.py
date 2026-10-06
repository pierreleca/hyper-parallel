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
"""The components read from the profiler's trace alone: classes, the stream's state, the collectives, the numbers."""

import importlib.util
import json
import pathlib
import sys
from types import ModuleType
from typing import Any

import pytest

_DIR = pathlib.Path(__file__).parents[3] / "examples" / "qwen3_vl_30b_perf"


def _load(name: str) -> ModuleType:
    """Import an example script by path."""
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("ascend_trace")
pc = _load("profile_components")


def _read(path: pathlib.Path) -> Any:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _meta(kind: str, pid: int, name: str, tid: int = None) -> dict:
    """One process_name / thread_name metadata event."""
    event = {"ph": "M", "name": kind, "pid": pid, "args": {"name": name}}
    if tid is not None:
        event["tid"] = tid
    return event


def _x(name: str, ts: float, dur: float, pid: int, tid: int) -> dict:
    """One complete event, microseconds."""
    return {"ph": "X", "name": name, "ts": ts, "dur": dur, "pid": pid, "tid": tid, "args": {}}


def _events() -> list[dict]:
    """A step of 1400 us on the compute stream whose accounting can be done by hand.

    0-100 attention, 110-160 dense matmul, 160-460 a wait released by an alltoallv, 460-660 grouped matmul, a gap of
    500 us, 1160-1180 norm, 1180-1280 a wait released by an allGather, a record, 1300-1310 a routing kernel.
    """
    compute = [
        _x("aclnnFlashAttentionScore_FlashAttentionScore", 0, 100, 2, 5),
        _x("aclnnMatmul_MatMulV2", 110, 50, 2, 5),
        _x("EVENT_WAIT", 160, 300, 2, 5),
        _x("aclnnGroupedMatmulV4_GroupedMatmul", 460, 200, 2, 5),
        _x("aclnnRmsNorm_RmsNorm", 1160, 20, 2, 5),
        _x("EVENT_WAIT", 1180, 100, 2, 5),
        _x("EVENT_RECORD", 1300, 0, 2, 5),
        _x("aclnnMoeTokenPermute_MoeTokenPermute", 1300, 10, 2, 5),
    ]
    comms = [
        _x("hcom_alltoallv__1_2_1", 170, 290, 3, 1),
        _x("hcom_alltoallv__3_2_1", 300, 100, 3, 2),             # overlaps the first: another row
        _x("hcom_allGather__2_2_1", 1100, 180, 3, 1),
    ]
    return [_meta("process_name", 1, "CANN"), _meta("thread_name", 1, "python", 1),
            _meta("process_name", 2, "Ascend Hardware"), _meta("thread_name", 2, "Stream 5", 5),
            _meta("process_name", 3, "Communication"), _meta("thread_name", 3, "Group_0 Communication", 1),
            _x("ProfilerStep#3", 0, 1400, 1, 1)] + compute + comms


def _signals(tmp_path: pathlib.Path, rules=()):
    """Write the hand-made trace and read it back."""
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps(_events()))
    capture = pc.load_capture(str(path))
    return capture, pc.read_signals(capture.trace, rules)


# -- intervals ---------------------------------------------------------------------------------------------

def test_interval_helpers():
    """Union, subtraction and the covered time of a set of intervals."""
    assert pc.merged([(5, 7), (1, 3), (2, 4), (9, 9)]) == [(1, 4), (5, 7)]
    assert pc.subtract((0, 10), [(2, 3), (5, 12)]) == [(0, 2), (3, 5)]
    assert pc.subtract((0, 10), []) == [(0, 10)]
    assert pc.total([(0, 2), (5, 6)]) == 3
    busy = pc.Busy([(0, 10), (20, 30), (5, 12)])
    assert busy.overlap(0, 100) == 22
    assert busy.overlap(8, 25) == 4 + 5
    assert busy.overlap(40, 50) == 0 and busy.overlap(3, 3) == 0


# -- the classes of kernels -------------------------------------------------------------------------------

def test_kernels_are_classed_by_name_and_a_rule_wins_over_the_defaults():
    """The defaults know the Ascend names, with or without separators; a CLASS=REGEX rule is tried first."""
    classify = pc.make_classifier()
    assert classify("aclnnFlashAttentionScoreGrad_FlashAttentionScoreGrad") == "attention"
    assert classify("aclnnFlashAttentionVarLenScore") == "attention"
    assert classify("aclnnGroupedMatmulV4_GroupedMatmul") == "experts"
    assert classify("npu_grouped_matmul_kernel") == "experts" and classify("Gmm_op") == "experts"
    assert classify("aclnnMoeInitRouting") == "routing" and classify("aclnnArgsort") == "routing"
    assert classify("aclnnMatmul_MatMulV2") == "dense" and classify("aclnnRmsNorm") == "norm"
    assert classify("aclnnCrossEntropyLoss") == "other" and classify("aclnnInplaceCopy") == "other"
    ruled = pc.make_classifier(pc.parse_rules(["experts=Matmul_Special", "other=^aclnnRmsNorm"]))
    assert ruled("aclnnMatmul_Special_x") == "experts" and ruled("aclnnRmsNorm_1") == "other"
    assert ruled("aclnnMatmul_MatMulV2") == "dense", "the other kernels keep their defaults"


def test_a_bad_rule_is_refused():
    """A rule is CLASS=REGEX with a known class."""
    for bad in ("attention", "nonsense=x", "attention="):
        with pytest.raises(ValueError):
            pc.parse_rules([bad])


# -- the numbers -------------------------------------------------------------------------------------------

def test_the_step_is_accounted_by_class_wait_and_idle(tmp_path):
    """Computing, waiting and idle add up to the step; busy time is split by class, waits by what ended them."""
    _, signals = _signals(tmp_path)
    (number, start, end), = signals.steps
    numbers = pc.step_numbers(signals, number, start, end)
    assert (numbers["span"], numbers["compute"], numbers["wait"], numbers["idle"], numbers["edges"]) == (
        1400, 380, 400, 530, 90)
    assert numbers["compute"] + numbers["wait"] + numbers["idle"] + numbers["edges"] == numbers["span"]
    assert numbers["busy_by_class"] == {"attention": 100, "dense": 50, "experts": 200, "norm": 20, "routing": 10}
    assert numbers["waits"] == {"alltoallv": 300, "allGather": 100}
    gaps = {bucket["bucket"]: (bucket["count"], bucket["total_us"]) for bucket in numbers["idle_buckets"]}
    assert gaps["0.1-1 ms"] == (1, 500) and gaps["10-100 us"] == (2, 30)
    text = "\n".join(pc.describe_step(numbers))
    assert "ProfilerStep 3" in text and "computing, by class of kernel: attention" in text
    assert "waiting, by what released it: alltoallv" in text and "allGather" in text
    assert "idle gaps: " in text and "collectives in flight / not hidden by compute (ms): " in text


def test_the_inventory_names_the_streams_the_classes_and_the_collectives(tmp_path):
    """What the trace holds, with the kernels that fell in each class, so a wrong class is easy to see."""
    capture, signals = _signals(tmp_path)
    text = "\n".join(pc.inventory(capture, signals))
    assert "Stream 5" in text and "(compute)" in text and "EVENT_WAIT, aclnnFlashAttentionScore" in text
    assert "aclnn kernels outside the compute stream: 0 of 5 (0.0% of them)" in text, "no compute is missed"
    assert "attention kernels (FlashAttention only): 1 kernels" in text and "aclnnFlashAttentionScore" in text
    assert "expert GEMM kernels (grouped matmul): 1 kernels" in text
    assert "collectives (hcom): alltoallv x2, allGather x1" in text
    assert "EVENT_WAIT x2" in text and "EVENT_RECORD x1" in text
    assert "event-record tasks: 1 on the compute stream, 1 on the device" in text
    assert "profiler steps: 3 (1 ms)" in text
    (tmp_path / "ruled").mkdir()
    capture, ruled = _signals(tmp_path / "ruled", pc.parse_rules(["other=FlashAttention"]))
    listed = "\n".join(pc.inventory(capture, ruled))
    assert "attention kernels (FlashAttention only): none" in listed, "a rule moves the kernel out of its class"


# -- the lanes ---------------------------------------------------------------------------------------------

def test_the_state_lane_partitions_the_compute_stream(tmp_path):
    """Computing, waiting for a kind of collective, idle: in order, without overlap; small gaps stay in the run."""
    _, signals = _signals(tmp_path)
    state = pc.state_pieces(signals)
    assert [(piece.name, piece.start, piece.end) for piece in state] == [
        ("computing", 0, 160), ("waiting: alltoallv", 160, 460), ("computing", 460, 660), ("idle", 660, 1160),
        ("computing", 1160, 1180), ("waiting: allGather", 1180, 1280), ("computing", 1300, 1310)]
    assert state[0].args["kernels"] == 2 and state[0].args["computing_ms"] == pytest.approx(0.15)


def test_the_class_lanes_merge_kernels_that_are_close_and_the_collectives_never_overlap_in_a_row(tmp_path):
    """One slice per run of same-class kernels; overlapping collectives of one kind go to another row."""
    _, signals = _signals(tmp_path)
    lanes = pc.profile_lanes(signals, merge_us=100.0, idle_us=20.0)
    assert [(p.start, p.end, p.name) for p in lanes["class:experts"]] == [(460, 660, "expert GEMM")]
    assert [(p.start, p.end) for p in lanes["class:attention"]] == [(0, 100)]
    assert {(p.start, p.end) for p in lanes["class:routing"]} == {(1300, 1310)}
    rows = {p.row for p in lanes["coll:alltoallv"]}
    assert rows == {0, 1}
    for row in rows:
        pieces = sorted((p.start, p.end) for p in lanes["coll:alltoallv"] if p.row == row)
        assert all(a[1] <= b[0] for a, b in zip(pieces, pieces[1:]))
    assert [p.name for p in lanes["coll:allgather"]] == ["allGather"] and not lanes["coll:reducescatter"]


# -- the events written into the trace --------------------------------------------------------------------

def test_process_events_name_the_lanes_scale_the_slices_and_hide_the_slivers():
    """Metadata for the process and each lane (and row), slices at offset + start * scale."""
    pieces = {"a": [pc.Piece(1.0, 3.0, "x", {"k": 1}), pc.Piece(2.0, 4.0, "y", row=1), pc.Piece(5.0, 5.0005, "tiny")]}
    events = pc.process_events(77, "name", -5, (("a", "lane A"), ("b", "lane B")), pieces, offset_us=100.0,
                               scale=1000.0)
    threads = {(e["tid"], e["args"]["name"]) for e in events if e["name"] == "thread_name"}
    assert threads == {(10, "lane A"), (11, "lane A (row 2)"), (20, "lane B")}
    assert {e["args"]["name"] for e in events if e["name"] == "process_name"} == {"name"}
    slices = [e for e in events if e["ph"] == "X"]
    assert [(e["name"], e["tid"], e["ts"], e["dur"]) for e in slices] == [
        ("x", 10, 1100.0, 2000.0), ("y", 11, 2100.0, 2000.0)], "the 0.5 us sliver is not drawn"
    assert all(e["pid"] == 77 for e in events)


def test_new_process_ids_avoid_the_ones_a_trace_uses():
    """Integer ids that no event of the trace has."""
    assert pc.free_pids([{"pid": 9_000_000}, {"pid": "9000001"}, {"pid": 4}], 2) == [9_000_002, 9_000_003]
    assert pc.free_pids([], 2, base=10) == [10, 11]


def test_the_integrated_trace_keeps_the_container_and_every_original_event(tmp_path):
    """A list stays a list, a dict keeps its other keys; the originals come first and unchanged."""
    capture, _ = _signals(tmp_path)
    extra = [{"ph": "X", "name": "new", "pid": 9, "tid": 1, "ts": 1.0, "dur": 2.0}]
    size = pc.write_integrated(capture, extra, str(tmp_path / "out" / "list.json"))
    text = (tmp_path / "out" / "list.json").read_text(encoding="utf-8")
    written = json.loads(text)
    assert size > 0 and isinstance(written, list) and written == _events() + extra
    original = (tmp_path / "trace_view.json").read_text(encoding="utf-8").rstrip()
    assert text.startswith(original[:-1]), "the original text is kept byte for byte, then the new events"
    empty = pc.write_integrated(capture, [], str(tmp_path / "out" / "empty.json"))
    assert empty > 0 and _read(tmp_path / "out" / "empty.json") == _events()
    wrapped = pc.Capture(capture.path, {"traceEvents": _events(), "schema": 1}, _events(), capture.trace)
    pc.write_integrated(wrapped, extra, str(tmp_path / "out" / "dict.json"))
    assert _read(tmp_path / "out" / "dict.json") == {
        "traceEvents": _events() + extra, "schema": 1}


def test_compute_kernels_on_another_stream_are_counted_as_missed(tmp_path):
    """A stream that holds model kernels is work the compute split does not count, and the inventory says how much."""
    events = _events() + [
        _meta("thread_name", 2, "Stream 9", 9),
        _x("aclnnMatmul_MatMulV2", 500, 40, 2, 9),
        _x("NOTIFY_WAIT", 600, 200, 2, 9),
    ]
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps(events))
    capture = pc.load_capture(str(path))
    text = "\n".join(pc.inventory(capture, pc.read_signals(capture.trace)))
    assert "aclnn kernels outside the compute stream: 1 of 6 (16.7% of them), 0 ms against" in text
    assert "of its busy time)" in text, "the share of the time, which is the one that matters"
    assert "Stream 9" in text and "NOTIFY_WAIT" in text, "the stream is named with what it holds"
