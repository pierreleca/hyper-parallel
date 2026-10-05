# The baseline of a speedup claim: how steady it is, and what the recorders cost. Every run is 20 steps on
# the same 640 samples in the same order, so two runs of one configuration see the same samples in each step.
#   fixed_a, fixed_b   the reference dataset (no data heterogeneity) twice: the noise floor of the model side
#   both_a, both_b     the heterogeneous dataset twice: the noise floor of the data side
#   both_hooks         both, module hooks ON (the sweep and ladder runs use them): their overhead
#   both_off           both, the heterogeneity recorder OFF: only the trainer's own log (perf.txt); the cost of
#                      the light recorder that every A/B run keeps on
# The light recorder (hooks=false) writes the step record only: two device events and one synchronization per step.
# Read the noise floor in the first two comparisons ("speedup" of a run against its twin is what no change can
# claim), the recorders' cost in the last two (a negative speedup is the slowdown). A change of less than the
# noise is not a result. Runs alternate between the datasets to spread any drift of the cluster over both.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATA=/home/pl/data/qwen3_vl_30b_perf
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false --training.train_iters=20"
FIXED="--dataset.data_path=$DATA/hetero_fixed_n640/vlm_conversations.json"
BOTH="--dataset.data_path=$DATA/hetero_both_n640/vlm_conversations.json"
RUNS=(
  "both_a $LIGHT $BOTH"
  "fixed_a $LIGHT $FIXED"
  "both_b $LIGHT $BOTH"
  "fixed_b $LIGHT $FIXED"
  "both_hooks --ep_instrument.enabled=false --training.train_iters=20 $BOTH"
  "both_off --hetero_profile.enabled=false --ep_instrument.enabled=false --training.train_iters=20 $BOTH"
)
COMPARE=("fixed_b:fixed_a" "both_b:both_a" "both_hooks:both_a" "both_off:both_a")
