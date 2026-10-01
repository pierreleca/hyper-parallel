# 64 dies, 8 text layers, 8192 tokens, no activation recompute: the profiled pair of
# the sweep (plans/sweep_64dev.sh), no swap and 6.4 of 8 mean layers, to see where
# the swap's time goes. Every rank traces steps 8 to 10 only: 64 traces fill a disk
# fast, so check the nodes' free space first. Reads the 8192-token dataset.
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
BASELINE=profile_noswap
LAYERS="--model.num_hidden_layers=8 \
--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/cauldron_seq8192_n1280/vlm_conversations.json \
--dataset.data_transform.max_seq_len=8192"
PROFILE="--training.train_iters=10 --ep_instrument.enabled=false --profiling.enabled=true --profiling.rank=-1 \
--profiling.start_step=8 --profiling.end_step=10"
RUNS=(
  "profile_noswap $LAYERS --ep_host_swap.enabled=false $PROFILE"
  "profile_k64 $LAYERS --ep_host_swap.budget_layers=6.4 $PROFILE"
)
