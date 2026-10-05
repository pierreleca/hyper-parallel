# Data heterogeneity at equal mean work. Five datasets with the same mean tokens (8192) and visual
# tokens (2048) per sample and different spread: fixed (none), text (tokens vary), vision (visual
# tokens vary, tokens do not), both (independent), longtail (both, heavy-tailed, correlated); and
# natural (real conversations of one subset per sample, at the size the data has, so its mean differs).
# 14 steps each, the same model and parallelism.
# The sweep table of SUMMARY.txt puts them side by side: step time, the spread of the ranks' work,
# the share of the step the busiest rank costs the others, and what bucketing would save.
# Each directory is built once per node (A3_HETERO_RUNS.md, "Datasets"). Runs in order of heterogeneity.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATA=/home/pl/data/qwen3_vl_30b_perf
RUNS=()
for scenario in fixed text vision both longtail natural; do
  RUNS+=("$scenario --dataset.data_path=$DATA/hetero_${scenario}_n640/vlm_conversations.json")
done
