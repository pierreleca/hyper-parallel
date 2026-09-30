# 64 dies, no activation recompute: what fits? 8-step no-swap runs at three shapes,
# and at the two 8-layer shapes a run with half the MoE activations swapped
# (4 mean layers of 8), which may fit where the no-swap run does not. A run that
# runs out of memory is recorded as failed and the probe moves on. Read each run's
# peak memory in its report; the sweep then uses the largest shape that fits.
# The 8192-token runs read a dataset built once per node (A3_RUNS.md).
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
SHORT="--training.train_iters=8"
S8K="--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/cauldron_seq8192_n1280/vlm_conversations.json \
--dataset.data_transform.max_seq_len=8192"
RUNS=(
  "s16k_l8 --model.num_hidden_layers=8 --ep_host_swap.enabled=false $SHORT"
  "s16k_l8_k40 --model.num_hidden_layers=8 --ep_host_swap.budget_layers=4.0 $SHORT"
  "s16k_l4 --model.num_hidden_layers=4 --ep_host_swap.enabled=false $SHORT"
  "s8k_l8 --model.num_hidden_layers=8 --ep_host_swap.enabled=false $S8K $SHORT"
  "s8k_l8_k40 --model.num_hidden_layers=8 --ep_host_swap.budget_layers=4.0 $S8K $SHORT"
)
