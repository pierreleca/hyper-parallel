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
# control node, and analyse every run: one call, one set of nodes throughout.
#
#   a3_campaign.sh <code.zip> <plan.sh>
#
# The plan (see plans/*.sh) is a bash file that sets:
#   NODES      nodes to run on (16 dies each)
#   CONFIG     training YAML, repository-relative
#   DATASET    prepare_cauldron_data.py arguments for the dataset CONFIG reads
#   BASELINE   (optional) the run the others are compared with: routing, sweep
#   RUNS       one entry per run: "<name> [--override=value ...]", run in order
#
# What it does, stopping a run (not the campaign) at its first failure:
#   1. waits for NODES free nodes and saves the selection, so every run of the
#      campaign lands on the same nodes (cluster select -aw N --census -o);
#   2. installs the zip on them (cluster deploy) and builds the dataset on each
#      node where it is missing;
#   3. per run: launches it once every device is free (torchrun -w --run-id),
#      waits for its end (status -w), kills what is left if it failed;
#   4. gathers the small records (not the traces), merges the ranks of every node
#      into one directory, and runs the reports: the EP instrument report (with
#      the swap activity), the routing comparison with BASELINE, the rule replay
#      on no-swap runs, the trace report on the nodes for profiled runs;
#   5. writes SUMMARY.txt with each run's state and, over all runs, the sweep.
#
# Everything lands in $OUT_BASE/<campaign>/ on the control node; paste SUMMARY.txt
# and the reports it points to. Environment knobs (defaults in brackets):
#   OUT_BASE  [/home/pl/a3_runs]   REMOTE_REPO [/mnt/data/pl/hyper-parallel]
#   RUNS_DIR  [/home/pl/runs/qwen3_vl_30b_perf]  INTERVAL [30]  RUN_TIMEOUT [7200]
#   SKIP_DEPLOY=1 to reuse the code already installed on the nodes it gets.
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <code.zip> <plan.sh>" >&2; exit 2; }
ZIP="$(realpath "$1")"
PLAN="$(realpath "$2")"
# shellcheck source=/dev/null
source "$PLAN"
: "${NODES:?plan sets NODES}" "${CONFIG:?plan sets CONFIG}" "${DATASET:?plan sets DATASET}"
[[ ${#RUNS[@]} -ge 1 ]] || { echo "the plan sets no RUNS" >&2; exit 2; }
BASELINE="${BASELINE:-}"

OUT_BASE="${OUT_BASE:-/home/pl/a3_runs}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/data/pl/hyper-parallel}"
RUNS_DIR="${RUNS_DIR:-/home/pl/runs/qwen3_vl_30b_perf}"
INTERVAL="${INTERVAL:-30}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

CAMPAIGN="$(basename "$PLAN" .sh)_$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_BASE/$CAMPAIGN"
mkdir -p "$OUT/tools" "$OUT/raw"
exec > >(tee -a "$OUT/campaign.log") 2>&1
echo "campaign $CAMPAIGN: ${#RUNS[@]} run(s) on $NODES node(s), $CONFIG; output in $OUT"
cp "$PLAN" "$OUT/plan.sh"

# The analysis scripts, from the same zip as the code the nodes run (standard library only).
python3 - "$ZIP" "$OUT/tools" <<'EOF'
import sys, zipfile
archive, target = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
for name in archive.namelist():
    if name.endswith(("examples/qwen3_vl_30b_perf/analyze_ep_instrument.py",
                      "examples/qwen3_vl_30b_perf/replay_ep_host_swap.py")):
        with open(f"{target}/{name.rsplit('/', 1)[1]}", "wb") as out:
            out.write(archive.read(name))
EOF

# 1. The nodes, kept for the whole campaign.
cluster select -aw "$NODES" --census -i "$INTERVAL" -o "$OUT/cluster.env"
CL=(cluster -c "$OUT/cluster.env")

# 2. The code and the data.
if [[ "${SKIP_DEPLOY:-0}" != 1 ]]; then
  "${CL[@]}" deploy --dry-run "$ZIP" "$REMOTE_REPO" hyper_parallel
  "${CL[@]}" deploy "$ZIP" "$REMOTE_REPO" hyper_parallel
fi
"${CL[@]}" exec -p "python examples/qwen3_vl_30b_perf/prepare_cauldron_data.py $DATASET"

# 3 and 4. The runs, each analysed before the next starts.
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
    states+=("$name: FAILED (cluster logs $run; last lines in $local_dir/status.txt)"); continue
  fi
  states+=("$name: finished")

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
    python3 "$OUT/tools/analyze_ep_instrument.py" "$local_dir/instrument" "${swap_args[@]}" \
      --out-dir "$local_dir/analysis" > "$local_dir/report.txt" 2>&1 || states+=("$name: report failed")
    if [[ ! -d "$local_dir/ep_host_swap" ]]; then
      python3 "$OUT/tools/replay_ep_host_swap.py" "$local_dir/instrument" > "$local_dir/replay.txt" 2>&1 \
        || states+=("$name: replay failed")
    fi
    if [[ -n "$BASELINE" && "$name" != "$BASELINE" && -d "$OUT/$BASELINE/instrument" ]]; then
      python3 "$OUT/tools/analyze_ep_instrument.py" "$local_dir/instrument" \
        --compare "$OUT/$BASELINE/instrument" > "$local_dir/compare.txt" 2>&1 || states+=("$name: compare failed")
    fi
  fi
  if [[ " ${overrides[*]} " == *" --profiling.enabled=true "* ]]; then
    "${CL[@]}" exec -p "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $remote/profile --ranks" \
      > "$local_dir/trace.txt" 2>&1 || states+=("$name: trace report failed")
  fi
done

# 5. The summary.
{
  echo "campaign $CAMPAIGN on $(grep -m1 -o 'NODES=.*' "$OUT/cluster.env" || echo "$NODES nodes")"
  echo "config $CONFIG, zip $(basename "$ZIP")"
  printf '  %s\n' "${states[@]}"
  sweep=()
  [[ -n "$BASELINE" && -d "$OUT/$BASELINE/instrument" ]] && sweep+=("$OUT/$BASELINE")
  for entry in "${RUNS[@]}"; do
    name="${entry%% *}"
    [[ "$name" != "$BASELINE" && -d "$OUT/$name/instrument" ]] && sweep+=("$OUT/$name")
  done
  if [[ ${#sweep[@]} -ge 1 ]]; then
    echo
    python3 "$OUT/tools/analyze_ep_instrument.py" --sweep "${sweep[@]}" 2>&1 || echo "sweep failed"
  fi
  echo
  echo "reports: $OUT/<run>/{report,compare,replay,trace}.txt"
} | tee "$OUT/SUMMARY.txt"
