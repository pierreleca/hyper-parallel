# What the imbalance between ranks costs, measured: the same samples in two orders. "random" is the draw
# order (every step mixes cheap and costly samples); "balanced" sorts the samples by estimated cost and
# cuts them into steps, so the 32 ranks of a step get alike samples while the steps differ. Same total
# work, so the step-time difference is the cost of the imbalance that the order can remove. Then
# the same with 2 micro-batches per rank (global batch 64), where each rank's work is a sum of two
# samples and the imbalance should shrink on its own.
# All 640 samples are consumed (20 steps of 32; 10 steps of 64), so both orders of a pair do the same total
# work; the steps differ, which is why compare_runs.py credits work per second, not time per step.
# Needs hetero_both_n640 and hetero_longtail_n640 on every node, built with --arrange random balanced
# (the default). Pairs run back to back, on whatever nodes the selection holds.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATA=/home/pl/data/qwen3_vl_30b_perf
ALL="--training.train_iters=20"
RUNS=()
COMPARE=()
for scenario in both longtail; do
  RUNS+=(
    "${scenario}_random $ALL --dataset.data_path=$DATA/hetero_${scenario}_n640/vlm_conversations.json"
    "${scenario}_balanced $ALL --dataset.data_path=$DATA/hetero_${scenario}_n640/vlm_conversations.balanced.json"
  )
  COMPARE+=("${scenario}_balanced:${scenario}_random")
done
RUNS+=(
  "both_random_accum2 --training.global_batch_size=64 --training.train_iters=10 \
--dataset.data_path=$DATA/hetero_both_n640/vlm_conversations.json"
  "both_balanced_accum2 --training.global_batch_size=64 --training.train_iters=10 \
--dataset.data_path=$DATA/hetero_both_n640/vlm_conversations.balanced.json"
)
COMPARE+=("both_balanced_accum2:both_random_accum2")
