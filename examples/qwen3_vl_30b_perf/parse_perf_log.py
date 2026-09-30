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
"""Summarize step time and peak memory from a Qwen3-VL performance run log.

Reads the per-step metric lines the Trainer's logging callback writes on rank 0
(``step=… performance/step_time=… memory/device_max_reserved_gb=…``), tolerating
any prefix, so cluster-kit's ``[nodeN]`` streams and plain torchrun logs both
parse. Warm-up steps are skipped by default because the first steps include
allocator growth, the first all-gathers and dataloader worker start-up.

    python examples/qwen3_vl_30b_perf/parse_perf_log.py run.log
    python examples/qwen3_vl_30b_perf/parse_perf_log.py --skip 3 node*.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, Sequence

_STEP_RE = re.compile(r"\bstep=(\d+)\b")
_FIELD_RE = re.compile(r"\b([A-Za-z_]+/[A-Za-z_]+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_STEP_TIME = "performance/step_time"
_TOKENS_PER_SECOND = "performance/tokens_per_second"
_RESERVED = "memory/device_max_reserved_gb"
_ALLOCATED = "memory/device_max_allocated_gb"


def parse_steps(lines: Iterable[str]) -> dict[int, dict[str, float]]:
    """Return ``{step: {metric: value}}`` for every metric line found.

    Args:
        lines: Log lines in any order, optionally prefixed (``[node0] …``).

    Returns:
        One entry per optimizer step; later duplicates of a step win, which
        keeps the last epoch's value when a log concatenates several runs.
    """
    steps: dict[int, dict[str, float]] = {}
    for line in lines:
        step_match = _STEP_RE.search(line)
        if step_match is None:
            continue
        fields = {name: float(value) for name, value in _FIELD_RE.findall(line)}
        if _STEP_TIME not in fields:
            continue
        steps.setdefault(int(step_match.group(1)), {}).update(fields)
    return steps


def summarize(steps: dict[int, dict[str, float]], skip: int) -> dict[str, float]:
    """Reduce parsed steps to the reported statistics.

    Args:
        steps: Output of :func:`parse_steps`.
        skip: Number of leading steps excluded from the timing statistics.

    Returns:
        Timing statistics over the kept steps, plus the peak memory over all
        steps (the warm-up steps can hold the peak, so they always count).

    Raises:
        ValueError: If no step survives the skip.
    """
    ordered = [steps[key] for key in sorted(steps)]
    kept = ordered[skip:]
    if not kept:
        raise ValueError(f"no steps left after skipping {skip} of {len(ordered)}")
    times = [entry[_STEP_TIME] for entry in kept]
    tokens = [entry[_TOKENS_PER_SECOND] for entry in kept if _TOKENS_PER_SECOND in entry]
    summary = {
        "steps_parsed": float(len(ordered)),
        "steps_used": float(len(times)),
        "step_time_median_s": statistics.median(times),
        "step_time_mean_s": statistics.fmean(times),
        "step_time_min_s": min(times),
        "step_time_max_s": max(times),
    }
    if tokens:
        summary["tokens_per_second_median"] = statistics.median(tokens)
    for name, key in ((_RESERVED, "peak_reserved_gb"), (_ALLOCATED, "peak_allocated_gb")):
        values = [entry[name] for entry in ordered if name in entry]
        if values:
            summary[key] = max(values)
    return summary


def _read_lines(paths: Sequence[Path]) -> list[str]:
    """Read every input file, or standard input when no path is given."""
    if not paths:
        return sys.stdin.read().splitlines()
    lines: list[str] = []
    for path in paths:
        lines.extend(path.read_text(errors="replace").splitlines())
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Print the summary for the logs named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="*", type=Path)
    parser.add_argument("--skip", type=int, default=5,
                        help="leading steps excluded from timing statistics (default: 5)")
    args = parser.parse_args(argv)

    steps = parse_steps(_read_lines(args.logs))
    if not steps:
        print("no metric lines found: expected lines like "
              "'step=7 ... performance/step_time=23.6 ...'", file=sys.stderr)
        return 1
    summary = summarize(steps, max(0, args.skip))
    width = max(len(name) for name in summary)
    for name, value in summary.items():
        print(f"{name:<{width}}  {value:,.3f}" if name.endswith(("_s", "_gb", "_median"))
              else f"{name:<{width}}  {value:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
