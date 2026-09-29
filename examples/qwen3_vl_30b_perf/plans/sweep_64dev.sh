# 64 dies, after the smoke run: a no-swap baseline, the budget from 1.1 down to
# 0.3 mean layers per layer at 6 text layers, and a profiled pair. Every run
# routes like the baseline (full determinism), so the reports compare step by step.
CONFIG=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
DATASET="--output-dir /home/pl/data/qwen3_vl_30b_perf/cauldron_seq16384_n1280 --seq-len 16384 --num-samples 1280 \
--processor-path /home/e00642590/Qwen3-VL-30B-A3B-Instruct --download-dir /home/pl/data/the_cauldron --offline"
BASELINE=noswap
PROFILE="--training.train_iters=10 --ep_instrument.enabled=false --profiling.enabled=true --profiling.rank=-1 \
--profiling.start_step=6 --profiling.end_step=10"
RUNS=(
  "noswap --ep_host_swap.enabled=false"
  "k66 --ep_host_swap.budget_layers=6.6"
  "k60 --ep_host_swap.budget_layers=6.0"
  "k54 --ep_host_swap.budget_layers=5.4"
  "k48 --ep_host_swap.budget_layers=4.8"
  "k36 --ep_host_swap.budget_layers=3.6"
  "k24 --ep_host_swap.budget_layers=2.4"
  "k18 --ep_host_swap.budget_layers=1.8"
  "profile_noswap --ep_host_swap.enabled=false $PROFILE"
  "profile_k48 --ep_host_swap.budget_layers=4.8 $PROFILE"
)
