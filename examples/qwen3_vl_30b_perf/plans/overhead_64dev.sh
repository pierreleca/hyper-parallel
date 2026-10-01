# Optional. 64 dies, 8 text layers, 8192 tokens, no recompute: what the swap costs at
# a budget above the mean (8.8 of 8), where the sweep measured +7 ms per step. Two
# no-swap runs measure the run-to-run noise; the untimed swap runs as the shipped
# code does (no per-step sync, no timing events), the timed one as the sweep did.
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
BASELINE=noswap
LAYERS="--model.num_hidden_layers=8 \
--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/cauldron_seq8192_n1280/vlm_conversations.json \
--dataset.data_transform.max_seq_len=8192"
RUNS=(
  "noswap $LAYERS --ep_host_swap.enabled=false"
  "k88_untimed $LAYERS --ep_host_swap.budget_layers=8.8 --ep_host_swap.timed=false"
  "noswap2 $LAYERS --ep_host_swap.enabled=false"
  "k88 $LAYERS --ep_host_swap.budget_layers=8.8"
)
