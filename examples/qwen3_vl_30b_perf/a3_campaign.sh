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
# Run a campaign of EP host-swap experiments on A3 with cluster-kit, from the
# control node, and analyse every run.
#
#   examples/qwen3_vl_30b_perf/a3_campaign.sh <plan.sh>
#   examples/qwen3_vl_30b_perf/a3_campaign.sh --resume <campaign dir> [--rerun <run> ...]
#   examples/qwen3_vl_30b_perf/a3_campaign.sh --summarize <campaign dir>
#
# --resume carries on a campaign that was interrupted: a run with a recorded state
# (<run>/state: finished or FAILED) is kept; one still running on the cluster is
# waited for, one that finished there is analysed, one that was killed or never
# launched is launched (again, under a new run id). --rerun discards the named runs'
# results and launches them again. --summarize only redoes steps 3 and 4 below.
# campaign_status.sh <campaign dir> shows what --resume would do.
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
# Everything lands in $OUT_BASE/<campaign>/ on the control node; <run>/run_id names
# the run on the cluster (cluster status/logs/kill). Environment knobs (defaults in
# brackets):
#   OUT_BASE [/home/pl/a3_runs]   RUNS_DIR [/home/pl/runs/qwen3_vl_30b_perf]
#   INTERVAL [30]   RUN_TIMEOUT [7200]   CLUSTER [cluster], e.g. "cluster -c other.env"
#   LOG_WAIT [45]: seconds of `cluster logs` captured per run into <run>/log.txt (the kit
#   follows the logs from their first line and never stops on its own); the summary
#   quotes their first error and, with debug.check_nan_inf, the first non-finite gradient.
set -euo pipefail

MODE=run DIR="" RERUN=()
case "${1:-}" in
  --resume|--summarize)
    MODE="${1#--}"
    [[ $# -ge 2 ]] || { echo "usage: $0 $1 <campaign dir>" >&2; exit 2; }
    DIR="$(realpath "$2")"; shift 2
    if [[ "${1:-}" == "--rerun" && "$MODE" == resume ]]; then shift; RERUN=("$@"); set --; fi
    [[ $# -eq 0 ]] || { echo "unexpected arguments: $*" >&2; exit 2; }
    set -- "$DIR/plan.sh" ;;
esac
[[ $# -eq 1 ]] || { echo "usage: $0 <plan.sh> | --resume <campaign dir> [--rerun <run> ...] | --summarize <campaign dir>" >&2; exit 2; }
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
ERROR_LINE='OutOfMemoryError|[A-Za-z]+(Error|Exception): '

# Every node's log of a run, captured once into <dir>/log.txt without the colours.
capture_log() {  # capture_log <run id> <dir>
  timeout "$LOG_WAIT" "${CL[@]}" logs "$1" 2>/dev/null | sed -u 's/\x1b\[[0-9;]*m//g' > "$2/log.txt" || true
}
# The first line of a captured log that matches a pattern, cut short.
first_line() {  # first_line <dir> <extended regex>
  grep -h -m1 -E "$2" "$1/log.txt" 2>/dev/null | grep -v ERR99999 | cut -c1-220
}
# The state a run left: finished, FAILED: <first error>, or nothing yet. Campaigns from
# before state files count a run with records as finished, one with a log as failed.
run_state() {  # run_state <name>
  local dir="$OUT/$1"
  if [[ -f "$dir/state" ]]; then cat "$dir/state"
  elif [[ -d "$dir/instrument" || -f "$dir/trace.txt" ]]; then echo finished
  elif [[ -f "$dir/log.txt" ]]; then
    echo "FAILED: $(first_line "$dir" "$ERROR_LINE" || true)"
  fi
}
# A run id the cluster has never seen: <campaign>_<name>, then _r2, _r3, ...
fresh_run_id() {  # fresh_run_id <name>
  local id="${CAMPAIGN}_$1" attempt=1
  while "${CL[@]}" status "$id" > /dev/null 2>&1 || [[ $? -ne 2 ]]; do
    attempt=$((attempt + 1)); id="${CAMPAIGN}_$1_r$attempt"
  done
  echo "$id"
}

if [[ "$MODE" == run ]]; then
  CAMPAIGN="$(basename "$PLAN" .sh)_$(date +%Y%m%d_%H%M%S)"
  OUT="$OUT_BASE/$CAMPAIGN"
else
  OUT="$DIR"
  CAMPAIGN="$(basename "$OUT")"
fi
mkdir -p "$OUT/raw"
exec > >(tee -a "$OUT/campaign.log") 2>&1
read -r -a CL <<< "${CLUSTER:-cluster}"

# One run: launch it unless the cluster already has it, wait, record its state, analyse it.
process_run() {  # process_run <name> <override>...
  local name="$1"; shift
  local overrides=("$@") local_dir="$OUT/$name" run launch=1 state
  mkdir -p "$local_dir"
  run="$(cat "$local_dir/run_id" 2>/dev/null || echo "${CAMPAIGN}_${name}")"
  if [[ "$MODE" == resume ]]; then
    local code=0
    "${CL[@]}" status "$run" > "$local_dir/status.txt" 2>&1 || code=$?
    case "$code" in
      0|3) launch=0 ;;                                   # finished, or still running: no new launch
      2) ;;                                              # never launched under this id
      *) capture_log "$run" "$local_dir"                 # failed: keep a real error, relaunch the rest
         if [[ -n "$(first_line "$local_dir" "$ERROR_LINE" || true)" ]]; then launch=0
         else run="$(fresh_run_id "$name")"; rm -f "$local_dir/log.txt"; fi ;;
    esac
  fi
  local remote="$RUNS_DIR/$run"
  echo "=== $name ($run): ${overrides[*]:-(no overrides)}"
  if [[ "$launch" -eq 1 ]]; then
    echo "$run" > "$local_dir/run_id"
    if ! "${CL[@]}" torchrun -w -i "$INTERVAL" --run-id "$run" scripts/train_vl.py "$CONFIG" \
        --ep_instrument.output_dir="$remote/instrument" --ep_host_swap.output_dir="$remote/ep_host_swap" \
        --profiling.trace_dir="$remote/profile" "${overrides[@]}"; then
      echo "launch failed" > "$local_dir/state"; return
    fi
  fi
  if ! "${CL[@]}" status -w -i "$INTERVAL" -t "$RUN_TIMEOUT" "$run"; then
    "${CL[@]}" status "$run" > "$local_dir/status.txt" 2>&1 || true
    "${CL[@]}" kill "$run" > /dev/null 2>&1 || true
    capture_log "$run" "$local_dir"
    state="FAILED: $(first_line "$local_dir" "$ERROR_LINE" || true)"
    echo "${state%: }" > "$local_dir/state"
    return
  fi
  capture_log "$run" "$local_dir"
  # Records of every node into one directory per kind.
  "${CL[@]}" gather "$remote/instrument" "$remote/ep_host_swap" "$OUT/raw/$name" > /dev/null 2>&1 || true
  for kind in instrument ep_host_swap; do
    if compgen -G "$OUT/raw/$name/node*/$kind/*.jsonl" > /dev/null; then
      mkdir -p "$local_dir/$kind"
      cp "$OUT/raw/$name"/node*/"$kind"/*.jsonl "$local_dir/$kind/"
    fi
  done
  if [[ -d "$local_dir/instrument" ]]; then
    local swap_args=()
    [[ -d "$local_dir/ep_host_swap" ]] && swap_args=(--swap-dir "$local_dir/ep_host_swap")
    python3 "$TOOLS/analyze_ep_instrument.py" "$local_dir/instrument" "${swap_args[@]}" \
      --out-dir "$local_dir/analysis" > "$local_dir/report.txt" 2>&1 || echo "$name: report failed"
    if [[ ! -d "$local_dir/ep_host_swap" ]]; then
      python3 "$TOOLS/replay_ep_host_swap.py" "$local_dir/instrument" > "$local_dir/replay.txt" 2>&1 \
        || echo "$name: replay failed"
    fi
    if [[ -n "$BASELINE" && "$name" != "$BASELINE" && -d "$OUT/$BASELINE/instrument" ]]; then
      python3 "$TOOLS/analyze_ep_instrument.py" "$local_dir/instrument" \
        --compare "$OUT/$BASELINE/instrument" > "$local_dir/compare.txt" 2>&1 || echo "$name: compare failed"
    fi
  fi
  if [[ " ${overrides[*]} " == *" --profiling.enabled=true "* ]]; then
    "${CL[@]}" exec -p "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $remote/profile --ranks" \
      > "$local_dir/trace.txt" 2>&1 || echo "$name: trace report failed"
    # Each node's per-rank summary, small enough to keep: <run>/profile/node<N>.json.
    "${CL[@]}" gather "$remote/profile/analysis_ranks" "$OUT/raw/$name" > /dev/null 2>&1 || true
    mkdir -p "$local_dir/profile"
    local summary node
    for summary in "$OUT/raw/$name"/node*/analysis_ranks/ranks.json; do
      [[ -f "$summary" ]] || continue
      node="${summary#"$OUT/raw/$name/"}"
      cp "$summary" "$local_dir/profile/${node%%/*}.json"
    done
  fi
  echo finished > "$local_dir/state"
}

# 1 and 2. The runs, each analysed before the next starts.
if [[ "$MODE" != summarize ]]; then
  echo "campaign $CAMPAIGN ($MODE): ${#RUNS[@]} run(s), $CONFIG; output in $OUT"
  echo "code: $(git -C "$TOOLS" rev-parse --short HEAD 2>/dev/null || echo "not a git checkout") at $TOOLS"
  [[ "$MODE" == run ]] && cp "$PLAN" "$OUT/plan.sh"
  for name in "${RERUN[@]}"; do
    printf '%s\n' "${RUNS[@]%% *}" | grep -qx "$name" || { echo "--rerun: no run $name in the plan" >&2; exit 2; }
    old="$(cat "$OUT/$name/run_id" 2>/dev/null || echo "${CAMPAIGN}_${name}")"
    rm -rf "${OUT:?}/$name" "${OUT:?}/raw/$name"
    mkdir -p "$OUT/$name"
    fresh_run_id "$name" > "$OUT/$name/run_id"
    echo "--rerun $name: results removed; $old stays on the nodes under $RUNS_DIR/$old"
  done
  for entry in "${RUNS[@]}"; do
    read -r -a words <<< "$entry"
    if [[ "$MODE" == resume && -n "$(run_state "${words[0]}")" ]]; then
      echo "=== ${words[0]}: kept ($(run_state "${words[0]}" | cut -c1-80))"; continue
    fi
    process_run "${words[@]}"
  done
fi

# 3. The summary.
{
  echo "campaign $CAMPAIGN, config $CONFIG"
  for entry in "${RUNS[@]}"; do
    name="${entry%% *}"
    state="$(run_state "$name")"
    echo "  $name: ${state:-not run}"
    nonfinite="$(first_line "$OUT/$name" 'non-finite at step' || true)"
    [[ -n "$nonfinite" ]] && echo "      first non-finite${nonfinite#*non-finite}"
  done
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
