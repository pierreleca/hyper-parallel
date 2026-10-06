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
"""The component trace: lanes that partition a step, the clocks' alignment, the check against the kernels, and the
original trace that comes back with the components added."""

import glob
import importlib.util
import json
import pathlib
import random
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional

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
_load("ascend_trace")
pc = _load("profile_components")
ct = _load("component_trace")

OFFSET_MS = 4321.987
RANK_SKEW_US = 37.0            # the synthetic trace of rank r starts RANK_SKEW_US * r later
SMALL = {"scenario": "both", "ranks": 8, "steps": 5, "layers": 4, "blocks": 4, "ep_size": 4, "experts": 16,
         "seed": 2, "slow_rank": None}


def _read(path: pathlib.Path) -> Any:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _make(path: pathlib.Path, *, checkpoint: str = "nonreentrant", accumulation: int = 1, shift: int = 1,
          ranks=(0, 3), layers: int = SMALL["layers"]) -> pathlib.Path:
    """Simulate a run of records with Ascend-like traces for some ranks; return the run directory."""
    synthetic.write_run(str(path / "hetero"), accumulation=accumulation, checkpoint=checkpoint,
                        ascend_dir=str(path / "profile"), ascend_ranks=list(ranks), ascend_offset_ms=OFFSET_MS,
                        ascend_step_shift=shift, **{**SMALL, "layers": layers})
    return path


def _truth_us(rank: int) -> float:
    """Where the step of a rank starts on its synthetic trace."""
    return OFFSET_MS * 1000.0 + RANK_SKEW_US * rank


def _run(path: pathlib.Path, *extra: str) -> list[dict]:
    """Run the command on a run directory; return the summary it wrote."""
    assert ct.main([str(path), *extra]) == 0
    with open(path / "profile" / "components" / "components_summary.json", encoding="utf-8") as stream:
        return json.load(stream)


def _trace_files(path: pathlib.Path) -> list[str]:
    """The synthetic Ascend traces of a run."""
    return sorted(glob.glob(str(path / "profile" / "rank*" / "ASCEND_PROFILER_OUTPUT" / "trace_view.json")))


def _edit_traces(path: pathlib.Path, edit: Callable[[dict], Optional[dict]]) -> None:
    """Rewrite every trace of a run with ``edit(event) -> event or None`` (None drops the event)."""
    for name in _trace_files(path):
        with open(name, encoding="utf-8") as stream:
            data = json.load(stream)
        data["traceEvents"] = [new for new in (edit(event) for event in data["traceEvents"]) if new is not None]
        with open(name, "w", encoding="utf-8") as stream:
            json.dump(data, stream)


def _edit_headers(path: pathlib.Path, edit: Callable[[dict], None]) -> None:
    """Rewrite the header line of every record file with ``edit(header)``."""
    for name in glob.glob(str(path / "hetero" / "rank*.jsonl")):
        with open(name, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        header = json.loads(lines[0])
        edit(header)
        lines[0] = json.dumps(header)
        with open(name, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


# -- the lanes ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("checkpoint", ["reentrant", "nonreentrant"])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_lanes_partition_the_step_and_carry_the_reports_milliseconds(tmp_path, checkpoint, accumulation):
    """No two slices of the partition lanes overlap, they cover the whole window, and they add up as the report does."""
    run = _make(tmp_path, checkpoint=checkpoint, accumulation=accumulation, ranks=())
    for rank in ct.ranks_with_records(str(run / "hetero")):
        header, records = ct.load_rank(str(run / "hetero"), rank)
        for record in records:
            spans = ct.spans_of(header, record)
            lanes = ct.build_lanes(spans, record["device_ms"], record["step"])
            pieces = sorted((p.start, p.end) for lane in ct.PARTITION for p in lanes[lane])
            for (_, end), (start, _) in zip(pieces, pieces[1:]):
                assert start >= end - 1e-6, "two slices of the partition lanes overlap"
            window = (min(s.start for s in spans), max(s.end for s in spans))
            assert ct.total(pieces) == pytest.approx(window[1] - window[0], abs=1e-3)
            for lane, (mine, theirs) in ct.reconcile(header, record, lanes).items():
                assert mine == pytest.approx(theirs, abs=ct.RECONCILE_TOLERANCE_MS), lane


def _slice_names(directory: pathlib.Path, checkpoint: str) -> set[str]:
    """The names of the partition slices of the last step of rank 0 in a simulated run."""
    run = _make(directory, checkpoint=checkpoint, ranks=())
    header, records = ct.load_rank(str(run / "hetero"), 0)
    lanes = ct.build_lanes(ct.spans_of(header, records[-1]), records[-1]["device_ms"], records[-1]["step"])
    return {piece.name for lane in ct.PARTITION for piece in lanes[lane]}


def test_the_lazy_recompute_is_carved_out_of_the_exchange_and_the_eager_one_is_not(tmp_path):
    """Non-reentrant: the recomputed attention and experts sit in the MoE block's backward span, with slices of their
    own; its recompute never closes, so it has no exchange slice. Reentrant: the block's recompute closes."""
    lazy = _slice_names(tmp_path / "lazy", "nonreentrant")
    eager = _slice_names(tmp_path / "eager", "reentrant")
    assert {"attention recompute", "experts recompute", "exchange bwd"} <= lazy
    assert "exchange recompute" not in lazy
    assert {"attention recompute", "experts recompute", "exchange recompute", "exchange bwd"} <= eager


def test_slices_carry_the_component_pass_layer_and_a_reason(tmp_path):
    """A slice is named after its component and pass, and says where it is and why it is there."""
    run = _make(tmp_path, ranks=())
    header, records = ct.load_rank(str(run / "hetero"), 0)
    lanes = ct.build_lanes(ct.spans_of(header, records[-1]), records[-1]["device_ms"], records[-1]["step"])
    attention = next(p for p in lanes["attention"] if p.args["pass"] == "fwd" and p.args["layer"] == 2)
    assert attention.name == "attention fwd" and attention.args["module"] == "text.attn.2"
    exchange = [p for p in lanes["exchange"] if p.args["pass"] == "fwd" and p.args["layer"] == 2]
    assert [p.args["where"].split(":")[0] for p in exchange] == ["before the experts", "after the experts"]
    assert all("before" in p.args for p in lanes["between"]) and lanes["between"]
    assert {p.name.split()[0] for p in lanes["layers"]} == {"layer", "vision"}
    assert any(p.name.startswith("backward") for p in lanes["steps"])


# -- the alignment and the check --------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [1, 0])
def test_the_clocks_are_aligned_on_the_recorders_event_records(tmp_path, shift):
    """The step's start is found to the microsecond, with the right pairing of record and profiler steps."""
    run = _make(tmp_path, shift=shift)
    findings = _run(run, "--rank", "0", "3")
    assert [f["rank"] for f in findings] == [0, 3]
    for finding in findings:
        assert finding["offset_us"] == pytest.approx(_truth_us(finding["rank"]), abs=2.0)
        assert finding["step"] == 7 and finding["profiler_step"] == 7 - shift
        assert finding["anchored"] > 0.99 and "alignment" not in finding
        assert finding["verdict"].startswith("AGREE")
        assert all(check["share"] > 0.99 for check in finding["checks"])


def test_the_default_ranks_are_the_idlest_and_the_busiest_and_each_gets_its_own_trace_file(tmp_path):
    """The small directory holds the report and summary; each rank's trace (original plus components) is a file."""
    run = _make(tmp_path)
    findings = _run(run)
    assert sorted(f["rank"] for f in findings) == [0, 3]
    small, full = run / "profile" / "components", run / "profile" / "components_full"
    assert {p.name for p in small.iterdir()} == {"components_summary.json", "components.txt"}
    assert {p.name for p in full.iterdir()} == {"rank0_trace_with_components.json", "rank3_trace_with_components.json"}
    for rank in (0, 3):
        with open(full / f"rank{rank}_trace_with_components.json", encoding="utf-8") as stream:
            events = json.load(stream)["traceEvents"]
        names = {e["pid"]: e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "process_name"}
        ours = [name for pid, name in names.items() if pid >= ct.PROCESS_BASE]
        assert len(ours) == 2 and all(f"rank {rank} |" in name for name in ours), "only this rank's components"
        assert any("from the hooks" in name and "placed on the profiler's timestamps" in name for name in ours)
        assert any("from the profile alone" in name for name in ours)
        threads = {e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "thread_name"}
        assert "6 MoE exchange: router, all-to-all, wait" in threads
        assert "A attention kernels (FlashAttention only)" in threads
        assert "Stream 7" in threads, "the original threads are still there"
        assert all(e["dur"] >= 0 for e in events if e["ph"] == "X")


def _foreign_events() -> list[dict]:
    """Events of kinds the analysis ignores, with string timestamps, as a profiler writes them."""
    return [
        {"ph": "s", "id": 5, "pid": 1, "tid": 7, "ts": "10.5", "name": "HostToDevice", "cat": "async_npu"},
        {"ph": "f", "id": 5, "pid": 2, "tid": 7, "ts": "20.5", "name": "HostToDevice", "cat": "async_npu", "bp": "e"},
        {"ph": "C", "name": "memory", "pid": 3, "ts": 12.0, "args": {"MB": 3}},
        {"ph": "i", "name": "marker", "pid": 3, "tid": 1, "ts": 30.0, "s": "g"},
        {"ph": "X", "name": "aclnnWeird", "pid": 2, "tid": 7, "ts": "5000000.25", "dur": "3.5", "args": {"a": 1}},
    ]


def test_the_whole_original_trace_comes_back_untouched_with_only_this_ranks_components_added(tmp_path):
    """Every original event, of whatever kind, is kept as written and first; what follows is this rank's components."""
    run = _make(tmp_path)
    for name in _trace_files(run):
        with open(name, encoding="utf-8") as stream:
            data = json.load(stream)
        data["traceEvents"] += _foreign_events()
        data["displayTimeUnit"] = "ns"
        with open(name, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
    _run(run, "--rank", "0")
    full = run / "profile" / "components_full"
    assert [p.name for p in full.iterdir()] == ["rank0_trace_with_components.json"], "only the rank asked for"
    original = _read(_trace_files(run)[0])                   # rank 0
    combined = _read(full / "rank0_trace_with_components.json")
    assert combined["displayTimeUnit"] == "ns", "the container keeps its other keys"
    count = len(original["traceEvents"])
    assert combined["traceEvents"][:count] == original["traceEvents"]
    new = combined["traceEvents"][count:]
    used = {event.get("pid") for event in original["traceEvents"]}
    assert new and all(event["pid"] not in used for event in new), "new processes only"
    kinds = {event["ph"] for event in original["traceEvents"]}
    assert {"s", "f", "C", "i", "M", "X"} <= kinds, "the original holds more than slices and names"


def test_a_trace_written_as_a_list_comes_back_as_a_list(tmp_path):
    """The Ascend exporter may write a bare list of events; so does the combined file."""
    run = _make(tmp_path)
    name = _trace_files(run)[0]
    with open(name, encoding="utf-8") as stream:
        original = json.load(stream)["traceEvents"]
    with open(name, "w", encoding="utf-8") as stream:
        json.dump(original, stream)
    _run(run, "--rank", "0")
    path = run / "profile" / "components_full" / "rank0_trace_with_components.json"
    combined = _read(path)
    assert isinstance(combined, list) and combined[:len(original)] == original and len(combined) > len(original)


def test_the_profile_alone_is_enough_for_any_trace_file(tmp_path, capsys):
    """Given one trace_view.json and no records: the profile process only, numbers from the trace, no hook part."""
    run = _make(tmp_path)
    shutil.rmtree(run / "hetero")
    assert ct.main([_trace_files(run)[0]]) == 0
    printed = capsys.readouterr().out
    assert "THE PROFILE ALONE" in printed and "THE HOOKS" not in printed
    assert "no hetero records: the profile alone for rank 0" in printed
    finding = _read(run / "profile" / "components" / "components_summary.json")[0]
    assert finding["rank"] == 0 and finding["profile"] and "step" not in finding and "checks" not in finding
    numbers = finding["profile"][0]
    assert numbers["busy_by_class"]["experts"] > 0 and numbers["waits"]["alltoallv"] > 0
    events = _read(run / "profile" / "components_full" / "rank0_trace_with_components.json")["traceEvents"]
    names = [e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "process_name"]
    assert sum("components from" in name for name in names) == 1 and any("profile alone" in name for name in names)


def test_a_trace_next_to_its_records_gets_the_hooks_too(tmp_path):
    """Given the trace file of a rank under <run>/profile, the records are found in <run>/hetero."""
    run = _make(tmp_path)
    assert ct.main([_trace_files(run)[0]]) == 0
    finding = _read(run / "profile" / "components" / "components_summary.json")[0]
    assert finding["rank"] == 0 and finding["verdict"].startswith("AGREE") and finding["profile"]


def test_a_class_rule_finds_the_kernels_the_defaults_miss(tmp_path):
    """A kernel named otherwise than the defaults expect is found with --class, and the check can then run."""
    run = _make(tmp_path)
    _edit_traces(run, lambda event: {**event, "name": "aclnnMyExpertKernel"}
                 if event.get("name", "").startswith("aclnnGroupedMatmul") else event)
    without = _run(run, "--rank", "0")[0]
    assert without["checks"][0]["share"] is None, "no kernel of the class: nothing to check"
    assert without["profile"][0]["busy_by_class"].get("experts", 0.0) == 0.0
    shared = _run(run, "--rank", "0", "--class", "experts=MyExpertKernel")[0]
    assert shared["checks"][0]["share"] > 0.99 and shared["profile"][0]["busy_by_class"]["experts"] > 0.0
    assert ct.main([str(run), "--class", "nonsense=x"]) == 2


def test_each_stamp_is_moved_onto_its_own_record_task():
    """Stamps take the timestamp of the record task next to them; those without one take their neighbours' shift."""
    offset = 1000.0
    shifts = [2.0, 2.0, 3.0, None, 5.0, 5.0, 6.0, 7.0, None, None, 9.0, 10.0]
    marks = [[1, "fwd", 0, "in", float(t), 0] for t in range(len(shifts))]
    tasks = sorted([offset + t * 1000.0 + shift for t, shift in enumerate(shifts) if shift is not None]
                   + [offset + 5500.0, offset + 40000.0])            # records of other events, far from any stamp
    signals = SimpleNamespace(records=tasks, compute_records=tasks)
    placement = ct.place_on_profile({"marks": marks, "step": 1}, signals, offset)
    assert (placement.matched, placement.stamps) == (9, 12)
    assert placement.median_us == pytest.approx(5.0) and placement.worst_us == pytest.approx(10.0)
    assert 700.0 < placement.drift_us_per_s < 900.0, "the shift grows by ~0.8 us per ms"
    moved = [mark[4] for mark in placement.record["marks"]]
    assert moved[0] == pytest.approx(0.002) and moved[11] == pytest.approx(11.010)
    assert moved[3] == pytest.approx(3.004), "halfway between the shifts 3 and 5"
    assert moved[8] == pytest.approx(8.0 + 0.007667, abs=1e-5) and moved[9] == pytest.approx(9.0 + 0.008333, abs=1e-5)
    assert [mark[:4] for mark in placement.record["marks"]] == [mark[:4] for mark in marks]
    nothing = ct.place_on_profile({"marks": marks}, SimpleNamespace(records=[], compute_records=[]), offset)
    assert nothing.matched == 0 and nothing.record["marks"] == marks


def test_the_report_says_what_the_trace_holds_and_how_the_clocks_agree(tmp_path, capsys):
    """The inventory, the profile numbers and the stamp placement are in the report."""
    run = _make(tmp_path)
    assert ct.main([str(run), "--rank", "3"]) == 0
    printed = capsys.readouterr().out
    for expected in ("the trace: ", "streams of the device (", "aclnn kernels outside the compute stream: ",
                     "attention kernels (FlashAttention only): ",
                     "expert GEMM kernels (grouped matmul)", "collectives (hcom): alltoallv",
                     "event-record tasks: ", "profiler steps: 6",
                     "computing, by class of kernel: attention", "waiting, by what released it: ",
                     "each stamp moved onto its own record task: ", "median shift +0.0 us", "wrote ",
                     "how tightly the spans bracket their kernels"):
        assert expected in printed, expected


def test_kernels_alone_align_when_the_trace_lists_no_record_task(tmp_path):
    """Without anchors the kernels place the step to within the idle tail of the slices (a few milliseconds at worst
    here), which is enough for the classes to agree."""
    run = _make(tmp_path)
    _edit_traces(run, lambda event: None if event.get("name") == "EVENT_RECORD" else event)
    findings = _run(run, "--rank", "0", "3")
    for finding in findings:
        assert "anchored" not in finding and finding["alignment"] > 0.5
        assert abs(finding["offset_us"] - _truth_us(finding["rank"])) < 2000.0
        assert not finding["verdict"].startswith("DISAGREE")
        assert finding["checks"][0]["share"] > 0.95 and finding["checks"][1]["share"] > 0.95


def _swap_attention_and_experts(header: dict) -> None:
    """Give the attention modules the experts' role and the experts the attention's."""
    swapped = {"text.attn": "text.experts", "text.experts": "text.attn"}
    for module in header["modules"]:
        module["role"] = swapped.get(module["role"], module["role"])


def test_labels_on_the_wrong_modules_are_reported_as_a_disagreement(tmp_path):
    """Hooks attached to the wrong modules (attention and experts swapped) put the kernels outside their slices."""
    run = _make(tmp_path)
    _edit_headers(run, _swap_attention_and_experts)
    findings = _run(run, "--rank", "0", "3")
    for finding in findings:
        assert finding["anchored"] > 0.99                       # the anchors do not depend on the labels
        assert finding["verdict"].startswith("DISAGREE")
        assert finding["checks"][0]["share"] < 0.2


def _attention_becomes_a_wait(event: dict) -> dict:
    """Turn every attention kernel of a trace into a stream wait."""
    if event.get("name", "").startswith("aclnnFlashAttention"):
        return {**event, "name": "EVENT_WAIT", "args": {}}
    return event


def test_stream_waits_inside_compute_slices_are_flagged_as_a_hook_order_problem(tmp_path):
    """If the weights' wait fell inside the attention slices, the verdict says the hooks may sit before the unshard."""
    run = _make(tmp_path)
    _edit_traces(run, _attention_becomes_a_wait)
    finding = _run(run, "--rank", "0")[0]
    assert finding["compute_wait_share"] > ct.HOOK_ORDER_WAIT_SHARE
    assert "hooks may sit" in finding["verdict"]


def test_an_offset_can_be_given_and_a_wrong_one_is_caught(tmp_path):
    """--offset-ms skips the search; at a place where nothing lines up the classes disagree."""
    run = _make(tmp_path)
    right = _run(run, "--rank", "0", "--offset-ms", str(_truth_us(0) / 1000.0))[0]
    assert right["verdict"].startswith("AGREE") and "anchored" not in right
    wrong = _run(run, "--rank", "0", "--offset-ms", str(_truth_us(0) / 1000.0 + 37.0))[0]
    assert wrong["verdict"].startswith("DISAGREE")


def test_searching_the_whole_trace_finds_the_same_place(tmp_path):
    """--search-all ignores the profiler step's range."""
    run = _make(tmp_path)
    finding = _run(run, "--rank", "3", "--search-all")[0]
    assert finding["offset_us"] == pytest.approx(_truth_us(3), abs=2.0)


def test_lanes_alone_without_a_trace_or_on_request(tmp_path):
    """No profile directory, or --no-trace: the lanes on the recorder's clock, with the reconciliation but no check."""
    run = _make(tmp_path)
    findings = _run(run, "--rank", "0", "--no-trace")
    assert "composition" not in findings[0] and "offset_us" not in findings[0]
    assert all(abs(v["lane_ms"] - v["report_ms"]) < 0.01 for v in findings[0]["reconcile"].values())
    bare = _make(tmp_path / "bare", ranks=())
    findings = _run(bare, "--rank", "1")
    assert findings[0]["rank"] == 1 and "checks" not in findings[0]
    assert not (bare / "profile" / "components_full").exists()


def test_a_run_recorded_without_module_hooks_has_nothing_to_draw(tmp_path, capsys):
    """The light recorder has no boundaries: the command says so and exits non-zero."""
    run = _make(tmp_path, ranks=())
    for name in glob.glob(str(run / "hetero" / "rank*.jsonl")):
        with open(name, encoding="utf-8") as stream:
            lines = [json.loads(line) for line in stream]
        for line in lines[1:]:
            line["marks"] = []
        with open(name, "w", encoding="utf-8") as stream:
            stream.write("\n".join(json.dumps(line) for line in lines) + "\n")
    assert ct.main([str(run), "--rank", "0"]) == 1
    assert "no step with module boundaries" in capsys.readouterr().out


def test_the_verdict_distinguishes_agreement_disagreement_and_unsure_clocks():
    """Agreement, disagreement with the clocks aligned, a failing class with unsure clocks, nothing to compare."""
    agreeing = [{"what": "x", "share": 0.99, "floor": 0.9}, {"what": "y", "share": None, "floor": 0.9}]
    failing = [{"what": "x", "share": 0.4, "floor": 0.9}]
    assert ct.verdict(agreeing, True, 0.0).startswith("AGREE")
    assert "not unique" in ct.verdict(agreeing, False, 0.0)
    assert ct.verdict(failing, True, 0.0).startswith("DISAGREE")
    assert ct.verdict(failing, False, 0.0).startswith("ALIGNMENT UNCERTAIN")
    assert ct.verdict([{"what": "x", "share": None, "floor": 0.9}], True, None).startswith("NOTHING TO COMPARE")
    assert "hooks may sit" in ct.verdict(agreeing, True, 0.3)


def test_a_directory_without_records_is_reported(tmp_path, capsys):
    """Pointing the command at the wrong directory says what it expects."""
    assert ct.main([str(tmp_path)]) == 1
    assert "no rank*.jsonl" in capsys.readouterr().out


def test_slivers_are_counted_in_the_lanes_but_not_drawn():
    """Pieces shorter than a microsecond (adjacent hooks) count, so the lanes add up exactly, but are not emitted."""
    spans = [
        ct.Span("text.layer", 0, "text.layer.0", "fwd", 0, 0.0, 10.0),
        ct.Span("text.attn", 0, "text.attn.0", "fwd", 0, 0.0004, 5.0),
        ct.Span("text.moe", 0, "text.moe.0", "fwd", 0, 5.0003, 9.9997),
        ct.Span("text.experts", 0, "text.experts.0", "fwd", 0, 6.0, 8.0),
    ]
    lanes = ct.build_lanes(spans, 10.0, 1)
    assert ct.lane_ms(lanes)["inside"] == pytest.approx(0.0004 + 0.0003 + 0.0003, abs=1e-9)
    drawn = [e for e in pc.process_events(1, "x", 0, ct.LANES, lanes, 0.0, 1000.0)
             if e["ph"] == "X" and e["cat"] == "inside"]
    assert drawn == []
    assert ct.reconcile_tolerance(1.0, 1.0) == ct.RECONCILE_TOLERANCE_MS
    assert ct.reconcile_tolerance(8507.0, 8507.0) == pytest.approx(ct.RECONCILE_RELATIVE * 8507.0)


def test_the_two_ranks_drawn_by_default_are_named_idlest_and_busiest(tmp_path, capsys):
    """The report says which of the two ranks is which, with the work each holds."""
    run = _make(tmp_path)
    assert ct.main([str(run), "--no-original"]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("ranks drawn, among 2: the idlest, rank ") and "and the busiest, rank " in first


def test_microsecond_jitter_in_the_stamps_does_not_make_the_two_bookings_differ(tmp_path):
    """Boundaries of adjacent hooks that come out of order by a few microseconds move the lanes a hair, not a span."""
    run = _make(tmp_path, layers=24)
    noise = random.Random(5)
    for name in glob.glob(str(run / "hetero" / "rank*.jsonl")):
        with open(name, encoding="utf-8") as stream:
            lines = [json.loads(line) for line in stream]
        for record in lines[1:]:
            for mark in record["marks"]:
                mark[4] = round(mark[4] + noise.uniform(-0.002, 0.002), 4)
        with open(name, "w", encoding="utf-8") as stream:
            stream.write("\n".join(json.dumps(line) for line in lines) + "\n")
    for finding in _run(run, "--rank", "0", "3", "--no-trace"):
        for lane, entry in finding["reconcile"].items():
            assert abs(entry["lane_ms"] - entry["report_ms"]) <= ct.reconcile_tolerance(
                entry["lane_ms"], entry["report_ms"]), lane


def test_the_ep_wait_is_the_reports_and_is_written_on_the_exchange_slices(tmp_path):
    """Floor and waiting come from the whole EP group's records, equal the report's cells, and sit on the slices."""
    run = _make(tmp_path, ranks=(), layers=6)
    findings = _run(run, "--rank", "0", "3", "--no-trace", "--ep-size", "4")
    last_step = max(record["step"] for record in report.Run(str(run / "hetero")).steps[0])
    cells = {key: by_rank for key, by_rank in report.exchange_cells(report.Run(str(run / "hetero")), 4).items()
             if key[0] == last_step}
    for finding in findings:
        mine = [by_rank for by_rank in cells.values() if finding["rank"] in by_rank]
        assert finding["ep_wait"]["wait_ms"] == pytest.approx(
            sum(by_rank[finding["rank"]] - min(by_rank.values()) for by_rank in mine), abs=1e-6)
        assert finding["ep_wait"]["floor_ms"] == pytest.approx(sum(min(by_rank.values()) for by_rank in mine), abs=1e-6)
        assert finding["ep_wait"]["exchange_ms"] == pytest.approx(
            finding["ep_wait"]["floor_ms"] + finding["ep_wait"]["wait_ms"], abs=1e-6)
    with open(run / "profile" / "components" / "rank0_lanes.json", encoding="utf-8") as stream:
        events = [e for e in json.load(stream)["traceEvents"]
                  if e["ph"] == "X" and e["cat"] == "exchange" and e["pid"] == ct.PROCESS_BASE]
    per_layer = {(e["args"]["micro_batch"], e["args"]["layer"]): e["args"]["layer_wait_ms"] for e in events}
    assert len(per_layer) == 6
    assert sum(per_layer.values()) == pytest.approx(findings[0]["ep_wait"]["wait_ms"], abs=0.001 * len(per_layer))
    assert all(0 <= e["args"]["last_to_arrive_rank"] <= 3 for e in events)


def test_an_ep_group_that_is_not_whole_is_reported_not_guessed(tmp_path, capsys):
    """With 8 ranks and the default group of 16, no EP wait is computed and the report says why."""
    run = _make(tmp_path, ranks=())
    finding = _run(run, "--rank", "0", "--no-trace")[0]
    assert "ep_wait" not in finding
    assert "EP wait not computed" in capsys.readouterr().out


def test_a_window_writes_a_small_file_that_still_holds_both_processes_and_the_real_events(tmp_path):
    """--window-ms keeps the stretch asked for: the originals that start in it, the lanes clipped to it."""
    run = _make(tmp_path)
    whole = _run(run, "--rank", "0")[0]
    whole_events = _read(whole["trace_file"])["traceEvents"]
    findings = _run(run, "--rank", "0", "--window-ms", "300", "120")
    path = findings[0]["trace_file"]
    assert path.endswith("rank0_trace_with_components_at300ms_for120ms.json"), "the window is named in the file"
    events = _read(path)["traceEvents"]
    assert len(events) < len(whole_events) / 3, "a window of the step is a fraction of the file"
    slices = [e for e in events if e["ph"] == "X"]
    start = whole["offset_us"] + 300_000.0
    window = (start, start + 120_000.0)
    assert slices, "the window is not empty"
    for event in slices:
        assert event["ts"] >= window[0] - 1e-6 and event["ts"] <= window[1] + 1e-6, "nothing starts outside it"
        if event["pid"] >= ct.PROCESS_BASE:
            assert event["ts"] + event["dur"] <= window[1] + 1e-6, "an added slice is clipped to the window"
    pids = {e["pid"] for e in slices}
    assert len([p for p in pids if p >= ct.PROCESS_BASE]) == 2, "both component processes are still there"
    assert any(p < ct.PROCESS_BASE for p in pids), "and the rank's own streams"
    names = {e["name"] for e in slices if e["pid"] < ct.PROCESS_BASE}
    assert any(name.startswith("aclnn") for name in names), "real kernels, not only the lanes"
    originals = {(e["pid"], e["tid"], e["ts"], e["name"]) for e in whole_events if e["ph"] == "X"
                 and e["pid"] < ct.PROCESS_BASE and window[0] <= e["ts"] <= window[1]}
    kept = {(e["pid"], e["tid"], e["ts"], e["name"]) for e in slices if e["pid"] < ct.PROCESS_BASE}
    assert kept == originals, "every original event of the window, and only those, kept as written"


def test_the_offsets_searched_never_collapse_to_nothing():
    """A recorder step longer than the profiler's range of it still leaves a window the anchors can be found in."""
    low, high = ct.search_window((1_000_000.0, 1_020_000.0), 25.0)          # a 20 ms range, a 25 ms step
    assert high - low >= ct.MIN_SEARCH_US and low < 1_000_000.0 < high, "the step's own start is searched"
    low, high = ct.search_window((1_000_000.0, 9_000_000.0), 2.0)           # a wide range, a short step
    assert high > 8_000_000.0, "a wide profiler step is searched to its end"


def _lane(events: list, cat: str, span: tuple = None) -> list:
    """The slices of one lane, optionally inside a window."""
    return [e for e in events if e.get("cat") == cat and e["ph"] == "X"
            and (span is None or span[0] <= e["ts"] <= span[1])]


def test_a_window_draws_one_slice_per_kernel_and_a_whole_trace_merges_them(tmp_path):
    """With --window-ms the profile lanes are kernel-exact; over a whole trace they are runs, to stay drawable."""
    run = _make(tmp_path)
    _run(run, "--rank", "0", "--window-ms", "300", "120")
    windowed = _read(run / "profile" / "components_full" /
                     "rank0_trace_with_components_at300ms_for120ms.json")["traceEvents"]
    _run(run, "--rank", "0")
    whole = _read(run / "profile" / "components_full" / "rank0_trace_with_components.json")["traceEvents"]

    start = _run(run, "--rank", "0", "--no-original")[0]["offset_us"] + 300_000.0
    span = (start, start + 120_000.0)
    exact = _lane(windowed, "class:experts")
    merged_ = _lane(whole, "class:experts", span)
    assert exact and len(exact) > len(merged_), "one slice per kernel, not one per run of them"
    kernels = [e for e in windowed if e["ph"] == "X" and e["pid"] < ct.PROCESS_BASE
               and "GroupedMatmul" in str(e["name"]) and e["ts"] + e["dur"] <= span[1]]
    assert kernels, "the window holds expert kernels"
    for kernel in kernels:
        assert any(abs(slice_["ts"] - kernel["ts"]) < 1e-6 and abs(slice_["dur"] - kernel["dur"]) < 1e-6
                   for slice_ in exact), "a slice of the class lane is exactly one kernel"


def test_the_bracketing_says_how_tightly_a_span_holds_its_kernels():
    """Lead-in and lead-out of each span, the share that end mid-kernel, and the shift that would centre them."""
    lanes = {"attention": [ct.Piece(0.0, 1.0, "attention fwd"), ct.Piece(2.0, 3.0, "attention fwd")]}
    # Kernels at 100-900 us and 2100-2900 us: the span leads by 100 us at each end, nothing straddles.
    kernels = SimpleNamespace(compute=[
        SimpleNamespace(ts=100.0, end=900.0, dur=800.0, name="aclnnFlashAttentionScore"),
        SimpleNamespace(ts=2100.0, end=2900.0, dur=800.0, name="aclnnFlashAttentionScore"),
    ])
    tight = ct.bracketing(lanes, kernels, 0.0, names=("attention",))
    assert tight == {"spans": 2, "lead_in_us": 100.0, "lead_out_us": 100.0, "straddled": 0.0, "shift_us": 0.0}
    # The same kernels with the lanes drawn 150 us late: the spans start late and end while a kernel still runs.
    late = ct.bracketing(lanes, kernels, 150.0, names=("attention",))
    assert late["lead_in_us"] == -50.0 and late["lead_out_us"] == 250.0
    assert late["straddled"] == 0.0 and late["shift_us"] == pytest.approx(150.0), "the shift that would centre them"
    early = ct.bracketing(lanes, kernels, -150.0, names=("attention",))
    assert early["straddled"] == 1.0, "drawn early, every span ends while its kernel is still running"
    assert early["lead_out_us"] == -50.0 and early["shift_us"] == pytest.approx(-150.0)
    assert ct.bracketing({"attention": []}, kernels, 0.0, names=("attention",)) == {"spans": 0}


def test_a_kernel_placed_step_says_how_near_its_stamps_fall_to_the_record_tasks(tmp_path, capsys):
    """Without anchors the report measures why: the share of stamps near a record task, and the median distance.

    A uniform shift of the tasks would simply move the offset the anchors find, so the tasks are jittered: no offset
    then puts the stamps on them, which is the case the line has to explain.
    """
    run = _make(tmp_path)
    jitter = random.Random(3)
    _edit_traces(run, lambda event: {**event, "ts": float(event["ts"]) + jitter.uniform(-300.0, 300.0)}
                 if event.get("name") == "EVENT_RECORD" else event)
    finding = _run(run, "--rank", "0")[0]
    assert "alignment" in finding and "anchored" not in finding, "the kernels had to place it"
    near = finding["nearest_record"]["the compute stream"]
    assert near["stamps"] > 20 and near["share"] < 0.5, "the stamps no longer are the record tasks"
    assert near["median_us"] is not None
    assert "why no anchor:" in capsys.readouterr().out


def test_the_nearest_record_of_stamps_that_are_the_tasks():
    """Stamps that are the record tasks: every one within the tolerance, a median distance of zero."""
    marks = [[1, "fwd", 0, "in", float(t), 0] for t in range(40)]
    tasks = [1000.0 + t * 1000.0 for t in range(40)]
    exact = ct.nearest_record({"marks": marks}, tasks, 1000.0)
    assert exact == {"stamps": 40, "share": 1.0, "median_us": 0.0}
    assert ct.nearest_record({"marks": marks}, [], 1000.0)["median_us"] is None
    assert ct.nearest_record({"marks": []}, tasks, 0.0) == {"stamps": 0, "share": 0.0, "median_us": None}


def test_the_anchor_tasks_are_the_compute_streams_when_it_has_any():
    """The recorder records on the compute stream; the device's other streams record far more, of other events."""
    signals = SimpleNamespace(compute_records=[1.0, 2.0], records=[1.0, 2.0, 3.0, 4.0])
    assert ct.anchor_tasks(signals) == [1.0, 2.0]
    assert ct.anchor_tasks(SimpleNamespace(compute_records=[], records=[9.0])) == [9.0], "else whatever there is"
