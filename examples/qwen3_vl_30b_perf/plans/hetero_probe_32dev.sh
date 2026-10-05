# What fits on 32 dies? The longest sample (16384 tokens) sets the peak memory, so one 8-step run per
# recompute mode and depth, on the "both" dataset. A run that runs out of memory is recorded as failed
# and the probe moves on; read each run's peak in its report (MEMORY) and take the shallowest mode
# that fits at the depth you want to study.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
SHORT="--training.train_iters=8 --hetero_profile.start_step=2 --ep_instrument.start_step=2 --debug.check_nan_inf=true"
RUNS=(
  "l48_full $SHORT"
  "l48_selective $SHORT --activation_checkpoint.mode=selective"
  "l24_off $SHORT --activation_checkpoint.mode=off --model.num_hidden_layers=24"
  "l12_off $SHORT --activation_checkpoint.mode=off --model.num_hidden_layers=12"
)
