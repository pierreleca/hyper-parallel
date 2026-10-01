# 64 dies, 8192 tokens, no activation recompute: how deep the model can go with and
# without the swap. Per depth a no-swap run, the swap as a safety net (budget = the
# mean, 1.0 per layer) and as a memory reduction (0.2 per layer). Out of memory is a
# result here: the summary quotes it. Runs from the shallowest up; stop the campaign
# once every run of a depth fails. Reads the 8192-token dataset built once per node.
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
BASELINE=
DATA="--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/cauldron_seq8192_n1280/vlm_conversations.json \
--dataset.data_transform.max_seq_len=8192"
RUNS=()
for layers in 10 12 14 16; do
  net="$layers.0"
  low="$(awk "BEGIN { printf \"%.1f\", $layers * 0.2 }")"
  RUNS+=(
    "l${layers}_noswap --model.num_hidden_layers=$layers $DATA --ep_host_swap.enabled=false"
    "l${layers}_k${layers}0 --model.num_hidden_layers=$layers $DATA --ep_host_swap.budget_layers=$net"
    "l${layers}_k${low//./} --model.num_hidden_layers=$layers $DATA --ep_host_swap.budget_layers=$low"
  )
done
