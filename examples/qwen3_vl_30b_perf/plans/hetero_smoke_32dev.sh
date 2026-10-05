# First 32-die run: does FSDP 32 x EP 16 (experts FSDP-sharded over the 2 nodes) build and train the
# whole 48-layer model on samples of 1k to 16k tokens, and does every record come back? One 8-step run
# on the "both" dataset (tokens and visual tokens vary independently), full recompute, the profiler's
# hooks and the EP instrument on, the first non-finite gradient named if there is one.
# Read in its report.txt: that every section prints, the peak memory per rank (MEMORY), and the step
# time. Needs the dataset hetero_both_n640 on every node (A3_HETERO_RUNS.md, "Datasets").
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
RUNS=(
  "smoke --training.train_iters=8 --hetero_profile.start_step=2 --ep_instrument.start_step=2 --debug.check_nan_inf=true"
)
