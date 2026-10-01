# 64 dies, 8 text layers, 8192 tokens, no activation recompute (what the probe found
# to fit; 16384 tokens runs out of memory without recompute): a no-swap baseline, the
# budget from 1.1 down to 0.2 mean layers per layer (8.8 down to 1.6 of 8), and a
# profiled pair. Every run routes like the baseline (full determinism), so the
# reports compare step by step. Reads the 8192-token dataset built once per node.
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
BASELINE=noswap
LAYERS="--model.num_hidden_layers=8 \
--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/cauldron_seq8192_n1280/vlm_conversations.json \
--dataset.data_transform.max_seq_len=8192"
PROFILE="--training.train_iters=10 --ep_instrument.enabled=false --profiling.enabled=true --profiling.rank=-1 \
--profiling.start_step=6 --profiling.end_step=10"
RUNS=(
  "noswap $LAYERS --ep_host_swap.enabled=false"
  "k88 $LAYERS --ep_host_swap.budget_layers=8.8"
  "k80 $LAYERS --ep_host_swap.budget_layers=8.0"
  "k72 $LAYERS --ep_host_swap.budget_layers=7.2"
  "k64 $LAYERS --ep_host_swap.budget_layers=6.4"
  "k48 $LAYERS --ep_host_swap.budget_layers=4.8"
  "k32 $LAYERS --ep_host_swap.budget_layers=3.2"
  "k24 $LAYERS --ep_host_swap.budget_layers=2.4"
  "k16 $LAYERS --ep_host_swap.budget_layers=1.6"
  "profile_noswap $LAYERS --ep_host_swap.enabled=false $PROFILE"
  "profile_k64 $LAYERS --ep_host_swap.budget_layers=6.4 $PROFILE"
)
