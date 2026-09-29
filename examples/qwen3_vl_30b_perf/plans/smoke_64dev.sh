# First 64-die run: does FSDP 64 x EP 16 (experts FSDP-sharded over the 4 nodes)
# build, train and swap? 6 text layers, the budget measured on one node (5.4 mean
# layers of 6), EP instrument on, 10 steps. Read in its report: every rank's peak
# memory, the swap activity, and whether the EP all-to-alls stayed inside a node.
# Reads the 1280-sample dataset, built once per node (A3_RUNS.md, "64 dies, four nodes").
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
RUNS=(
  "swap54 --training.train_iters=10"
)
