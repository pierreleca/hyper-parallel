# The Ascend profiler on every rank for two steps, on the reference ("fixed") and on a heterogeneous
# ("both") dataset: what the spans of hetero_profile cannot show, the kernels, the streams' waits and
# the collectives. The trace report runs on the nodes (--ranks: per rank the compute, the waits and
# the idle time of the compute stream, the exposed all-to-alls and all-gathers, and the k-th
# all-to-all matched across the ranks of an EP group: the wait for the last rank to arrive against the
# transfer). Traces stay on the nodes, 16 rank directories each; delete them when read:
#   cluster exec 'rm -rf /home/pl/runs/qwen3_vl_30b_perf/<campaign>_*/profile'
# The EP instrument is off: its device events would sit in the traces. hetero_profile stays on, so each
# rank's workload is recorded next to its trace (profiler step N is optimizer step N + 1).
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATA=/home/pl/data/qwen3_vl_30b_perf
PROFILE="--training.train_iters=8 --ep_instrument.enabled=false --hetero_profile.start_step=2 \
--profiling.enabled=true --profiling.rank=-1 --profiling.start_step=6 --profiling.end_step=8"
RUNS=(
  "profile_fixed $PROFILE --dataset.data_path=$DATA/hetero_fixed_n640/vlm_conversations.json"
  "profile_both $PROFILE --dataset.data_path=$DATA/hetero_both_n640/vlm_conversations.json"
)
