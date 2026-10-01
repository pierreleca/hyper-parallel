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
# Where a campaign of a3_campaign.sh stands, one line per run of its plan, and what
# `a3_campaign.sh --resume` would do with it.
#
#   examples/qwen3_vl_30b_perf/campaign_status.sh <campaign dir>
#
# A run with a recorded state (finished, FAILED) is kept by --resume; for the others
# it asks the cluster (cluster status) about the run's id.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <campaign dir>" >&2; exit 2; }
OUT="$(realpath "$1")"
CAMPAIGN="$(basename "$OUT")"
read -r -a CL <<< "${CLUSTER:-cluster}"
# shellcheck source=/dev/null
source "$OUT/plan.sh"

counts=()
for entry in "${RUNS[@]}"; do
  name="${entry%% *}"
  dir="$OUT/$name"
  run="$(cat "$dir/run_id" 2>/dev/null || echo "${CAMPAIGN}_${name}")"
  if [[ -f "$dir/state" ]]; then
    what="$(cut -c1-100 "$dir/state")"
  elif [[ -d "$dir/instrument" || -f "$dir/trace.txt" ]]; then
    what="finished"
  else
    code=0
    "${CL[@]}" status "$run" > /dev/null 2>&1 || code=$?
    case "$code" in
      0) what="finished on the cluster, not analysed (--resume analyses it)" ;;
      3) what="RUNNING (--resume waits for it)" ;;
      2) what="not launched (--resume launches it)" ;;
      *) what="failed or killed on the cluster, not recorded (--resume keeps it if its log shows an error, else relaunches it)" ;;
    esac
  fi
  counts+=("${what%%[ :(]*}")
  printf '  %-16s %-40s %s\n' "$name" "$run" "$what"
done
echo "$(printf '%s\n' "${counts[@]}" | sort | uniq -c | awk '{printf "%s%s %s", sep, $1, $2; sep=", "}') of ${#RUNS[@]} runs"
