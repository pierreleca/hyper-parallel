#!/usr/bin/env bash
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
# Run a campaign of EP host-swap experiments on A3 with cluster-kit, from the
# control node, and analyse every run.
#
#   examples/qwen3_vl_30b_perf/a3_campaign.sh <plan.sh>
#
# Before it: select the nodes (cluster select), deploy the code to them and to the
# control node, and build the dataset CONFIG reads on every node (A3_RUNS.md); run
# it from the control node's checkout of that same code,
# which provides the analysis scripts. Every run goes to the selection active when
# it starts, so leave the selection alone until the campaign ends.
#
# The plan (see plans/*.sh) is a bash file that sets:
#   CONFIG     training YAML, repository-relative
#   BASELINE   (optional) the run the others are compared with: routing, sweep
#   RUNS       one entry per run: "<name> [--override=value ...]", run in order
#
# What it does, stopping a run (not the campaign) at its first failure:
#   1. per run: launches it once every device is free (torchrun -w --run-id),
#      waits for its end (status -w), kills what is left if it failed;
#   2. gathers the small records (not the traces), merges the ranks of every node
#      into one directory, and runs the reports: the EP instrument report (with
#      the swap activity), the routing comparison with BASELINE, the rule replay
#      on no-swap runs, the trace report on the nodes for profiled runs;
#   3. writes SUMMARY.txt with each run's state and, over all runs, the sweep;
#   4. distils the campaign into results.json (export_results.py), the one file to send.
#
# Everything lands in $OUT_BASE/<campaign>/ on the control node; paste SUMMARY.txt
# and the reports it points to. Environment knobs (defaults in brackets):
#   OUT_BASE [/home/pl/a3_runs]   RUNS_DIR [/home/pl/runs/qwen3_vl_30b_perf]
#   INTERVAL [30]   RUN_TIMEOUT [7200]   CLUSTER [cluster], e.g. "cluster -c other.env"
#   LOG_WAIT [45]: seconds of `cluster logs` captured per run into <run>/log.txt (the kit
#   follows the logs from their first line and never stops on its own); the summary
#   quotes their first error and, with debug.check_nan_inf, the first non-finite gradient.
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <plan.sh>" >&2; exit 2; }
PLAN="$(realpath "$1")"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this checkout's analysis scripts
# shellcheck source=/dev/null
source "$PLAN"
: "${CONFIG:?plan sets CONFIG}"
[[ ${#RUNS[@]} -ge 1 ]] || { echo "the plan sets no RUNS" >&2; exit 2; }
BASELINE="${BASELINE:-}"

OUT_BASE="${OUT_BASE:-/home/pl/a3_runs}"
RUNS_DIR="${RUNS_DIR:-/home/pl/runs/qwen3_vl_30b_perf}"
INTERVAL="${INTERVAL:-30}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"
LOG_WAIT="${LOG_WAIT:-45}"

# Every node's log of a run, captured once into <dir>/log.txt without the colours.
capture_log() {  # capture_log <run> <dir>
  timeout "$LOG_WAIT" "${CL[@]}" logs "$1" 2>/dev/null | sed -u 's/\x1b\[[0-9;]*m//g' > "$2/log.txt" || true
}
# The first line of a captured log that matches a pattern, cut short.
first_line() {  # first_line <dir> <extended regex>
  grep -h -m1 -E "$2" "$1/log.txt" 2>/dev/null | grep -v ERR99999 | cut -c1-220
}

CAMPAIGN="$(basename "$PLAN" .sh)_$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_BASE/$CAMPAIGN"
mkdir -p "$OUT/raw"
exec > >(tee -a "$OUT/campaign.log") 2>&1
read -r -a CL <<< "${CLUSTER:-cluster}"
echo "campaign $CAMPAIGN: ${#RUNS[@]} run(s), $CONFIG; output in $OUT"
echo "code: $(git -C "$TOOLS" rev-parse --short HEAD 2>/dev/null || echo "not a git checkout") at $TOOLS"
cp "$PLAN" "$OUT/plan.sh"

# 1 and 2. The runs, each analysed before the next starts.
states=()
for entry in "${RUNS[@]}"; do
  read -r -a words <<< "$entry"
  name="${words[0]}"
  overrides=("${words[@]:1}")
  run="${CAMPAIGN}_${name}"
  remote="$RUNS_DIR/$run"
  local_dir="$OUT/$name"
  mkdir -p "$local_dir"
  echo "=== $name: ${overrides[*]:-(no overrides)}"
  if ! "${CL[@]}" torchrun -w -i "$INTERVAL" --run-id "$run" scripts/train_vl.py "$CONFIG" \
      --ep_instrument.output_dir="$remote/instrument" --ep_host_swap.output_dir="$remote/ep_host_swap" \
      --profiling.trace_dir="$remote/profile" "${overrides[@]}"; then
    states+=("$name: launch failed"); continue
  fi
  if ! "${CL[@]}" status -w -i "$INTERVAL" -t "$RUN_TIMEOUT" "$run"; then
    "${CL[@]}" status "$run" > "$local_dir/status.txt" 2>&1 || true
    "${CL[@]}" kill "$run" > /dev/null 2>&1 || true
    capture_log "$run" "$local_dir"
    error="$(first_line "$local_dir" 'OutOfMemoryError|[A-Za-z]+(Error|Exception): ' || true)"
    states+=("$name: FAILED: ${error:-no error line in $local_dir/log.txt}")
    nonfinite="$(first_line "$local_dir" 'non-finite at step' || true)"
    [[ -n "$nonfinite" ]] && states+=("    first non-finite${nonfinite#*non-finite}")
    continue
  fi
  states+=("$name: finished")
  capture_log "$run" "$local_dir"
  nonfinite="$(first_line "$local_dir" 'non-finite at step' || true)"
  [[ -n "$nonfinite" ]] && states+=("    first non-finite${nonfinite#*non-finite}")

  # Records of every node into one directory per kind.
  "${CL[@]}" gather "$remote/instrument" "$remote/ep_host_swap" "$OUT/raw/$name" > /dev/null 2>&1 || true
  for kind in instrument ep_host_swap; do
    if compgen -G "$OUT/raw/$name/node*/$kind/*.jsonl" > /dev/null; then
      mkdir -p "$local_dir/$kind"
      cp "$OUT/raw/$name"/node*/"$kind"/*.jsonl "$local_dir/$kind/"
    fi
  done
  if [[ -d "$local_dir/instrument" ]]; then
    swap_args=()
    [[ -d "$local_dir/ep_host_swap" ]] && swap_args=(--swap-dir "$local_dir/ep_host_swap")
    python3 "$TOOLS/analyze_ep_instrument.py" "$local_dir/instrument" "${swap_args[@]}" \
      --out-dir "$local_dir/analysis" > "$local_dir/report.txt" 2>&1 || states+=("$name: report failed")
    if [[ ! -d "$local_dir/ep_host_swap" ]]; then
      python3 "$TOOLS/replay_ep_host_swap.py" "$local_dir/instrument" > "$local_dir/replay.txt" 2>&1 \
        || states+=("$name: replay failed")
    fi
    if [[ -n "$BASELINE" && "$name" != "$BASELINE" && -d "$OUT/$BASELINE/instrument" ]]; then
      python3 "$TOOLS/analyze_ep_instrument.py" "$local_dir/instrument" \
        --compare "$OUT/$BASELINE/instrument" > "$local_dir/compare.txt" 2>&1 || states+=("$name: compare failed")
    fi
  fi
  if [[ " ${overrides[*]} " == *" --profiling.enabled=true "* ]]; then
    "${CL[@]}" exec -p "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $remote/profile --ranks" \
      > "$local_dir/trace.txt" 2>&1 || states+=("$name: trace report failed")
    # Each node's per-rank summary, small enough to keep: <run>/profile/node<N>.json.
    "${CL[@]}" gather "$remote/profile/analysis_ranks" "$OUT/raw/$name" > /dev/null 2>&1 || true
    mkdir -p "$local_dir/profile"
    for summary in "$OUT/raw/$name"/node*/analysis_ranks/ranks.json; do
      [[ -f "$summary" ]] || continue
      node="${summary#"$OUT/raw/$name/"}"
      cp "$summary" "$local_dir/profile/${node%%/*}.json"
    done
  fi
done

# 3. The summary.
{
  echo "campaign $CAMPAIGN, config $CONFIG"
  printf '  %s\n' "${states[@]}"
  sweep=()
  [[ -n "$BASELINE" && -d "$OUT/$BASELINE/instrument" ]] && sweep+=("$OUT/$BASELINE")
  for entry in "${RUNS[@]}"; do
    name="${entry%% *}"
    [[ "$name" != "$BASELINE" && -d "$OUT/$name/instrument" ]] && sweep+=("$OUT/$name")
  done
  if [[ ${#sweep[@]} -ge 1 ]]; then
    echo
    python3 "$TOOLS/analyze_ep_instrument.py" --sweep "${sweep[@]}" 2>&1 || echo "sweep failed"
  fi
  echo
  echo "reports: $OUT/<run>/{report,compare,replay,trace}.txt"
} | tee "$OUT/SUMMARY.txt"
# 4. The figures' and the results' data, small enough to send: results.json.
python3 "$TOOLS/export_results.py" "$OUT" 2>&1 | tee -a "$OUT/SUMMARY.txt" || true
