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
"""The component trace: lanes that partition a step, the clocks' alignment, and the check against the kernels."""

import glob
import importlib.util
import json
import pathlib
import sys
from types import ModuleType
from typing import Callable, Optional

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
_load("analyze_hetero")
_load("ascend_trace")
ct = _load("component_trace")

OFFSET_MS = 4321.987
RANK_SKEW_US = 37.0            # the synthetic trace of rank r starts RANK_SKEW_US * r later
SMALL = {"scenario": "both", "ranks": 8, "steps": 5, "layers": 4, "blocks": 4, "ep_size": 4, "experts": 16,
         "seed": 2, "slow_rank": None}


def _make(path: pathlib.Path, *, checkpoint: str = "nonreentrant", accumulation: int = 1, shift: int = 1,
          ranks=(0, 3)) -> pathlib.Path:
    """Simulate a run of records with Ascend-like traces for some ranks; return the run directory."""
    synthetic.write_run(str(path / "hetero"), accumulation=accumulation, checkpoint=checkpoint,
                        ascend_dir=str(path / "profile"), ascend_ranks=list(ranks), ascend_offset_ms=OFFSET_MS,
                        ascend_step_shift=shift, **SMALL)
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


# -- intervals ---------------------------------------------------------------------------------------------

def test_interval_helpers():
    """Union, subtraction and the covered time of a set of intervals."""
    assert ct.merged([(5, 7), (1, 3), (2, 4), (9, 9)]) == [(1, 4), (5, 7)]
    assert ct.subtract((0, 10), [(2, 3), (5, 12)]) == [(0, 2), (3, 5)]
    assert ct.subtract((0, 10), []) == [(0, 10)]
    busy = ct.Busy([(0, 10), (20, 30), (5, 12)])
    assert busy.overlap(0, 100) == 22
    assert busy.overlap(8, 25) == 4 + 5
    assert busy.overlap(40, 50) == 0 and busy.overlap(3, 3) == 0


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


def test_the_default_ranks_are_the_idlest_and_the_busiest_and_the_files_are_written(tmp_path):
    """The small files hold the lanes, the summary and the report; the large ones add the real events."""
    run = _make(tmp_path)
    findings = _run(run)
    assert sorted(f["rank"] for f in findings) == [0, 3]
    small = run / "profile" / "components"
    assert {p.name for p in small.iterdir()} == {"components.json", "components_summary.json", "components.txt"}
    with open(small / "components.json", encoding="utf-8") as stream:
        lanes_only = json.load(stream)["traceEvents"]
    processes = [e["args"]["name"] for e in lanes_only if e["ph"] == "M" and e["name"] == "process_name"]
    assert len(processes) == 2 and all("SYNTHETIC rank" in name and "components" in name for name in processes)
    threads = {e["args"]["name"] for e in lanes_only if e["ph"] == "M" and e["name"] == "thread_name"}
    assert len(threads) == len(ct.LANES) and "6 MoE exchange: router, all-to-all, wait" in threads
    assert all(e["dur"] >= 0 for e in lanes_only if e["ph"] == "X")
    with open(run / "profile" / "components_full" / "components_rank0_with_trace.json", encoding="utf-8") as stream:
        full = json.load(stream)["traceEvents"]
    names = {e["args"]["name"] for e in full if e["ph"] == "M" and e["name"] == "process_name"}
    assert {"rank 0 | Ascend Hardware", "rank 0 | Communication"} <= names
    pids = {e["pid"] for e in full if e["ph"] == "X"}
    assert len(pids) == 3 and len(full) > len(lanes_only)
    real = [e for e in full if e["ph"] == "X" and e["pid"] != min(pids)]
    assert any(e["name"].startswith("hcom_alltoallv") for e in real) and all(e["dur"] >= 5.0 for e in real)


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
