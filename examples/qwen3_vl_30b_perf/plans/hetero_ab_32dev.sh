# A baseline against a candidate: the protocol for certifying a speedup. Nothing here knows the candidate: it
# is whatever the flags in IDEA turn on (a configuration override of the trainer, e.g. --my_idea.enabled=true).
#
#   IDEA="--my_idea.enabled=true" DATASET=both  examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_ab_32dev.sh
#   IDEA="--my_model_idea=true"   DATASET=fixed examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_ab_32dev.sh
#
# DATASET is the directory suffix of hetero_<DATASET>_n640: the data-heterogeneity idea is judged on both (or
# longtail, natural, vision), the model-heterogeneity idea on fixed, where no sample differs from another and the
# only heterogeneity left is the model's. BASE_FLAGS are extra flags of the baseline arm only (e.g. the baseline's
# own setting of a knob the candidate changes); ORDER=balanced reads vlm_conversations.balanced.json instead.
#
# Four timing runs alternate base, candidate, base, candidate (the cluster drifts; alternation spreads it over both
# arms), recorder light (hooks off: the step record only). Two more, with the module hooks on, say where the
# time moved. The comparison pools the repeats, pairs the steps by the samples they ran, and prints the speedup
# with its interval, the verdict against +20%, the loss and gradient-norm check, the memory and the per-component
# change. Run hetero_baseline_32dev.sh first: it tells how large a difference is noise.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATASET="${DATASET:-both}"
ORDER="${ORDER:-}"
IDEA="${IDEA:?set IDEA to the flags that turn the candidate on}"
BASE_FLAGS="${BASE_FLAGS:-}"
FILE="vlm_conversations${ORDER:+.$ORDER}.json"
COMMON="--dataset.data_path=/home/pl/data/qwen3_vl_30b_perf/hetero_${DATASET}_n640/$FILE --training.train_iters=20"
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false"
HOOKED="--ep_instrument.enabled=false"
RUNS=(
  "base_1 $COMMON $LIGHT $BASE_FLAGS"
  "cand_1 $COMMON $LIGHT $IDEA"
  "base_2 $COMMON $LIGHT $BASE_FLAGS"
  "cand_2 $COMMON $LIGHT $IDEA"
  "base_hooks $COMMON $HOOKED $BASE_FLAGS"
  "cand_hooks $COMMON $HOOKED $IDEA"
)
COMPARE=("cand_1,cand_2:base_1,base_2" "cand_hooks:base_hooks")
