# Is the packed path correct on Ascend? One document per row, so it must match the dense path exactly.
#
# check_packing.py proves the data contract on the host to 1e-7, but it cannot reach the Ascend
# variable-length kernel. This arm can: packing on, micro_batch_size 1, the study's dataloader and
# sample order, so each row is a single document and the packed path has to compute what the dense
# path computes. Paired against the baseline's both_a and both_b.
#
#   # the baseline, if it is not already run
#   DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_baseline_32dev.sh
#
#   # this arm
#   DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_single_32dev.sh
#
#   # then pair them: same samples, same order, one document per row
#   python examples/qwen3_vl_30b_perf/compare_runs.py \
#     --baseline  /home/pl/a3_runs/<baseline campaign>/both_a/hetero \
#                 /home/pl/a3_runs/<baseline campaign>/both_b/hetero \
#     --candidate /home/pl/a3_runs/<this campaign>/single_1/hetero \
#                 /home/pl/a3_runs/<this campaign>/single_2/hetero
#
# What to read. The loss, step for step, against the baseline's. It should track closely -- not
# bit-identical, because the four-row position ids and the variable-length kernel reassociate the
# same arithmetic, but far inside the run-to-run interval. The step time should land on the
# baseline's: one document per row recovers no padding and no imbalance, so a large change either
# way is itself a finding.
#
# The loss is a mean over few tokens, because the sample transform supervises only the final answer
# of each conversation -- about 25 tokens a sample, 800 a step across the 32 ranks. So compare the
# curve over the whole run rather than one step, and use compare_runs.py's pairing, which is what
# makes this arm sharper than eyeballing two curves.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_packing_single.yaml
DATASET="${DATASET:-both}"
DATA="/home/pl/data/qwen3_vl_30b_perf/hetero_${DATASET}_n640/vlm_conversations.json"
COMMON="--dataset.data_path=$DATA --training.train_iters=20"
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false"
RUNS=(
  "single_1 $COMMON $LIGHT"
  "single_2 $COMMON $LIGHT"
)
COMPARE=("single_2:single_1")
