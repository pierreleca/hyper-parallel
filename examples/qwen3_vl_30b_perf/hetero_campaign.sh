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
# Run a campaign of heterogeneity experiments on A3 with cluster-kit, from the
# control node, and analyse every run.
#
#   examples/qwen3_vl_30b_perf/hetero_campaign.sh <plan.sh>
#   examples/qwen3_vl_30b_perf/hetero_campaign.sh --resume <campaign dir> [--rerun <run> ...]
#   examples/qwen3_vl_30b_perf/hetero_campaign.sh --summarize <campaign dir>
#
# The runs go first, back to back, and are analysed afterwards: the devices are busy
# only while a run trains, so the analysis (slow with profiler traces) never keeps
# them from the next run or from somebody else. TRAIN_ONLY=1 stops after the last run;
# --resume then does the analysis whenever it suits. ANALYSE_EACH=1 does the opposite,
# analysing every run before the next one trains: the devices idle meanwhile, but only
# one run's traces sit on the nodes at a time, which a profiled campaign may need.
#
# --resume carries on a campaign that was interrupted: a run with a recorded state
# (<run>/state: trained, finished or FAILED) is kept (a trained one is analysed);
# one still running on the cluster is waited for, one that finished there is
# analysed, one that was killed or never launched is launched (again, under a new run
# id). --rerun discards the named runs' results and launches them again.
# --summarize only redoes steps 3 and 4 below.
# hetero_campaign_status.sh <campaign dir> shows what --resume would do.
#
# Before it: select the nodes (cluster select -a 2), deploy the code to them and to the
# control node, and build the datasets the plan reads on every node (A3_HETERO_RUNS.md);
# run it from the control node's checkout of that same code, which provides the analysis
# scripts. Every run goes to the selection active when it starts, so leave the selection
# alone until the campaign ends.
#
# The plan (see plans/hetero_*.sh) is a bash file that sets:
#   CONFIG     training YAML, repository-relative
#   SKIP       (optional) recorded steps the reports drop as warm-up [0]
#   SKIP_LOG   (optional) leading steps of the trainer's log that perf.txt leaves out [2]
#   EP_SIZE    (optional) ranks per expert-parallel group, for the routing report [16]
#   RUNS       one entry per run: "<name> [--override=value ...]", run in order
#   COMPARE    (optional) A/B comparisons after the runs: "<candidate>:<baseline>" entries; either side may list
#              several runs separated by commas, which are pooled (repeats). BASELINE=<run> compares every other
#              run with that one.
#
# What it does, stopping a run (not the campaign) at its first failure:
#   1. per run, in order: launches it once every device is free (torchrun -w --run-id),
#      waits for its end (status -w), kills what is left if it failed, captures its log;
#   2. then, per run that trained: gathers the small records (not the traces), merges the ranks of every node
#      into one directory, and runs the reports: the heterogeneity report
#      (analyze_hetero.py), the EP phase report, the trace report on the nodes for
#      profiled runs, the component trace for profiled runs (component_trace.py, on the
#      nodes: for the busiest and the idlest rank of each node, the profiler's own trace with
#      the detected components added as new processes, and a report that checks the hooks
#      against the kernels), and perf.txt, the step times the trainer logged itself;
#   3. runs the comparisons of COMPARE (compare_runs.py: speedup and interval, pairing by sample, loss
#      equivalence, per-component change) into compare_<candidate>_vs_<baseline>.txt;
#   4. writes SUMMARY.txt with each run's state, one sweep table over all runs, and each comparison's verdict.
#
# Everything lands in $OUT_BASE/<campaign>/ on the control node; <run>/run_id names
# the run on the cluster (cluster status/logs/kill). Environment knobs (defaults in
# brackets):
#   OUT_BASE [/home/pl/a3_runs]   RUNS_DIR [/home/pl/runs/qwen3_vl_30b_perf]
#   INTERVAL [30]   RUN_TIMEOUT [7200]   CLUSTER [cluster], e.g. "cluster -c other.env"
#   TRAIN_ONLY [0]: 1 stops after the last run has trained; analyse later with --resume
#   ANALYSE_EACH [0]: 1 analyses each run before the next one trains (one run's traces at a time)
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
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this checkout's analysis scripts
# A plan is named from the working directory, which is this repository's root because the plans set
# CONFIG from there. A name that does not resolve there is looked up beside this script as well, so
# `plans/hetero_packing_32dev.sh` and the bare `hetero_packing_32dev` both find it.
PLAN="$1"
for candidate in "$PLAN" "$TOOLS/$PLAN" "$TOOLS/plans/$PLAN" "$TOOLS/plans/${PLAN}.sh"; do
  if [[ -f "$candidate" ]]; then PLAN="$(realpath "$candidate")"; break; fi
done
[[ -f "$PLAN" ]] || { echo "no such plan: $1 (looked in . and $TOOLS/plans)" >&2; exit 2; }
COMPARE=() BASELINE=""
# shellcheck source=/dev/null
source "$PLAN"
: "${CONFIG:?plan sets CONFIG}"
[[ ${#RUNS[@]} -ge 1 ]] || { echo "the plan sets no RUNS" >&2; exit 2; }
SKIP="${SKIP:-0}"
SKIP_LOG="${SKIP_LOG:-2}"
EP_SIZE="${EP_SIZE:-16}"
if [[ -n "$BASELINE" && ${#COMPARE[@]} -eq 0 ]]; then
  for entry in "${RUNS[@]}"; do
    [[ "${entry%% *}" != "$BASELINE" ]] && COMPARE+=("${entry%% *}:$BASELINE")
  done
fi

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
# The first line of a captured log that matches a pattern, cut short. The SIGTERM a
# failed run's other ranks get is quoted only when nothing else matches.
first_line() {  # first_line <dir> <extended regex>
  local lines
  lines="$(grep -h -E "$2" "$1/log.txt" 2>/dev/null | grep -v ERR99999 || true)"
  { grep -v -m1 SignalException <<< "$lines" || head -n1 <<< "$lines"; } | cut -c1-220
}
# The state a run left: finished, FAILED: <first error>, or nothing yet. Campaigns from
# before state files count a run with records as finished, one with a log as failed.
run_state() {  # run_state <name>
  local dir="$OUT/$1"
  if [[ -f "$dir/state" ]]; then cat "$dir/state"
  elif [[ -d "$dir/hetero" || -d "$dir/instrument" || -f "$dir/trace.txt" ]]; then echo finished
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

# One run: launch it unless the cluster already has it, wait for it, capture its log, record its state
# (trained, FAILED, launch failed). Nothing here needs more than the devices.
train_run() {  # train_run <name> <override>...
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
  printf '%s\n' "${overrides[*]:-}" > "$local_dir/cmdline.txt"
  if [[ "$launch" -eq 1 ]]; then
    echo "$run" > "$local_dir/run_id"
    if ! "${CL[@]}" torchrun -w -i "$INTERVAL" --run-id "$run" scripts/train_vl.py "$CONFIG" \
        --hetero_profile.output_dir="$remote/hetero" --ep_instrument.output_dir="$remote/instrument" \
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
  echo trained > "$local_dir/state"
}

# The analysis of a run that trained: reports from the records, and for a profiled run the reports on the traces,
# which stay on the nodes (so the selection must still be the one the run used). Records the state finished.
analyse_run() {  # analyse_run <name>
  local name="$1"
  local local_dir="$OUT/$name" run remote
  run="$(cat "$local_dir/run_id" 2>/dev/null || echo "${CAMPAIGN}_${name}")"
  remote="$RUNS_DIR/$run"
  echo "=== analysing $name ($run)"
  # The step times the trainer logged itself (no recorder involved): the reference for an overhead or a clean timing.
  if [[ -s "$local_dir/log.txt" ]]; then
    python3 "$TOOLS/parse_perf_log.py" --skip "$SKIP_LOG" "$local_dir/log.txt" > "$local_dir/perf.txt" 2>&1 \
      || echo "$name: no metric lines in the log"
  fi
  # Records of every node into one directory per kind.
  "${CL[@]}" gather "$remote/hetero" "$remote/instrument" "$OUT/raw/$name" > /dev/null 2>&1 || true
  for kind in hetero instrument; do
    if compgen -G "$OUT/raw/$name/node*/$kind/*.jsonl" > /dev/null; then
      mkdir -p "$local_dir/$kind"
      cp "$OUT/raw/$name"/node*/"$kind"/*.jsonl "$local_dir/$kind/"
    fi
  done
  if [[ -d "$local_dir/hetero" ]]; then
    python3 "$TOOLS/analyze_hetero.py" "$local_dir/hetero" --skip "$SKIP" --ep-size "$EP_SIZE" \
      --out-dir "$local_dir/analysis_hetero" > "$local_dir/report.txt" 2>&1 || echo "$name: hetero report failed"
  fi
  if [[ -d "$local_dir/instrument" ]]; then
    python3 "$TOOLS/analyze_ep_instrument.py" "$local_dir/instrument" --skip "$SKIP" --ep-size "$EP_SIZE" \
      --out-dir "$local_dir/analysis_ep" > "$local_dir/ep_report.txt" 2>&1 || echo "$name: EP report failed"
  fi
  if grep -q -- "--profiling.enabled=true" "$local_dir/cmdline.txt" 2>/dev/null; then
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
    # The components added to the profiler's trace (component_trace.py). The report and the summary come back to
    # <run>/components/<node>_*; each rank's trace (the original plus the components, as large as the original) stays on
    # the node in profile/components_full: gather the one to look at.
    "${CL[@]}" exec -p "python examples/qwen3_vl_30b_perf/component_trace.py $remote --ep-size $EP_SIZE" \
      > "$local_dir/components.txt" 2>&1 || echo "$name: component trace failed"
    "${CL[@]}" gather "$remote/profile/components" "$OUT/raw/$name" > /dev/null 2>&1 || true
    mkdir -p "$local_dir/components"
    local file
    for file in "$OUT/raw/$name"/node*/components/*; do
      [[ -f "$file" ]] || continue
      node="${file#"$OUT/raw/$name/"}"
      cp "$file" "$local_dir/components/${node%%/*}_$(basename "$file")"
    done
  fi
  echo finished > "$local_dir/state"
}

# 1. The runs, one after the other. 2. Their analysis.
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
    train_run "${words[@]}"
    # The traces of a profiled run are gigabytes per rank: analysing it now leaves only one run's traces on
    # the nodes, at the price of the devices idling until the next run starts.
    if [[ "${ANALYSE_EACH:-0}" == 1 && "$(run_state "${words[0]}")" == trained ]]; then
      analyse_run "${words[0]}"
    fi
  done
  if [[ "${TRAIN_ONLY:-0}" == 1 ]]; then
    echo "TRAIN_ONLY: the devices are free; analyse with: $0 --resume $OUT"
    exit 0
  fi
  for entry in "${RUNS[@]}"; do
    if [[ "$(run_state "${entry%% *}")" == trained ]]; then analyse_run "${entry%% *}"; fi
  done
fi

# 3. The comparisons: <candidate>:<baseline>, each side one run or several pooled with commas.
sides() {  # sides <a,b,...>: the run directories, space separated
  local name out=()
  for name in ${1//,/ }; do out+=("$OUT/$name"); done
  printf '%s\n' "${out[@]}"
}
for pair in "${COMPARE[@]}"; do
  candidate="${pair%%:*}" baseline="${pair#*:}"
  file="$OUT/compare_${candidate//,/+}_vs_${baseline//,/+}.txt"
  mapfile -t base_dirs < <(sides "$baseline")
  mapfile -t cand_dirs < <(sides "$candidate")
  python3 "$TOOLS/compare_runs.py" --baseline "${base_dirs[@]}" --candidate "${cand_dirs[@]}" --skip 1 \
    > "$file" 2>&1 || echo "compare $pair failed (see $file)"
done

# 4. The summary.
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
  for entry in "${RUNS[@]}"; do
    name="${entry%% *}"
    [[ -d "$OUT/$name/hetero" ]] && sweep+=("$OUT/$name/hetero")
  done
  if [[ ${#sweep[@]} -ge 1 ]]; then
    echo
    python3 "$TOOLS/analyze_hetero.py" --sweep --skip "$SKIP" --ep-size "$EP_SIZE" "${sweep[@]}" 2>&1 || echo "sweep failed"
  fi
  for file in "$OUT"/compare_*.txt; do
    [[ -f "$file" ]] || continue
    echo
    grep -E '^A/B:|end to end|work per second|paired steps|verdict|noise:|numerics, |memory:' "$file" \
      || echo "$(basename "$file"): see the file"
  done
  keep='ranks drawn|rank [0-9]+:|ProfilerStep|by class of kernel|waiting, by what|EP wait|last to arrive|starts at'
  keep+='|each stamp moved|lanes against|AGREE|UNCERTAIN|NOTHING TO COMPARE|no Ascend trace|no step|nothing to draw|failed'
  for entry in "${RUNS[@]}"; do
    file="$OUT/${entry%% *}/components.txt"
    [[ -s "$file" ]] || continue
    echo
    echo "components against the kernels, ${entry%% *}:"
    { grep -E "$keep" "$file" || true; } | cut -c1-210
  done
  echo
  echo "reports: $OUT/<run>/{report,ep_report,trace,components,perf}.txt, $OUT/compare_*.txt,"
  echo "         $OUT/<run>/analysis_hetero/{hetero_report.json,microbatches.csv}, $OUT/<run>/components/ (report, summary)"
} | tee "$OUT/SUMMARY.txt"
