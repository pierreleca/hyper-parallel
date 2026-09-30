# One node (16 dies): does the router collapse without recompute? The same short
# no-swap run with activation recompute on and off, 2 text layers, 8192 tokens.
# SUMMARY.txt's routing max/mean is about 1.5-2 when healthy and about 16 when every
# token goes to one rank per EP group. Select ONE of the nodes that hold the
# 8192-token dataset.
CONFIG=examples/qwen3_vl_30b_perf/train_16dev_a3_ep_host_swap.yaml
COMMON="--model.num_hidden_layers=2 --ep_host_swap.enabled=false --training.train_iters=5 --ep_instrument.start_step=1 \
--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/cauldron_seq8192_n1280/vlm_conversations.json \
--dataset.data_transform.max_seq_len=8192"
RUNS=(
  "recompute_full $COMMON --activation_checkpoint.mode=full"
  "recompute_off $COMMON"
)
