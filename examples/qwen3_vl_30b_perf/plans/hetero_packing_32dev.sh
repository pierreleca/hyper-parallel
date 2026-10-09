# What packing is worth. Each rank takes samples until it holds a token budget, instead of taking one
# sample whatever its length, so a rank's unit of work stops being a sample and becomes a budget.
#
# This plan runs the PACKED arm. Its baseline is the unpacked ladder on the same dataset, which a
# campaign of its own produces, because a campaign carries one CONFIG and the two arms need two:
# _target_ cannot be changed from the command line, so packing lives in its own configuration file.
#
#   # the baseline, with the ordinary configuration
#   DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_baseline_32dev.sh
#
#   # this arm
#   DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_32dev.sh
#
#   # then join them: the baseline plan's both_a and both_b are the unpacked repeats on this dataset
#   python examples/qwen3_vl_30b_perf/compare_runs.py \
#     --baseline  /home/pl/a3_runs/<baseline campaign>/both_a/hetero \
#                 /home/pl/a3_runs/<baseline campaign>/both_b/hetero \
#     --candidate /home/pl/a3_runs/<this campaign>/packed_1/hetero \
#                 /home/pl/a3_runs/<this campaign>/packed_2/hetero
#
# NOT paired. A packed row is one sequence with one fingerprint where the unpacked arm has one per
# sample, so compare_runs.py cannot match the steps and reports work per second with the wide
# interval. Run plans/hetero_baseline_32dev.sh first so that interval is known.
#
# What to read in the report, beyond the step time. The recorder now writes, per row, how many
# documents it held, the shortest and longest of them, and the pairs attention really scored. The
# number to watch is the drop in busiest/mean: unpacked, a step costs its longest sample, so on a
# corpus of this spread the busiest die does about twice the mean die's work. Packed, every die runs
# about one budget, and what is left is the packing remainder.
#
# Correctness is not what this measures. examples/qwen3_vl_30b_perf/check_packing.py proves on the
# host that a packed row gives the logits of its documents run alone. What the cluster adds is
# whether the Ascend variable-length kernel agrees, and that shows up as a loss that tracks the
# baseline's or does not.
#
# Sizing, for when the knobs are turned. The token budget is micro_batch_size x max_seq_len, and
# max_seq_len is also the truncation ceiling, so they move together. Peak memory is about 6.7 GiB
# plus 1.89 per thousand tokens a rank holds: a 16384 budget is about 38 GiB of the 61 a die has, and
# a 32768 one would not fit. Raising micro_batch_size also forces global_batch_size up, because the
# sampler insists global is divisible by micro x 32.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_packing.yaml
DATASET="${DATASET:-both}"
DATA="/home/pl/data/qwen3_vl_30b_perf/hetero_${DATASET}_n640/vlm_conversations.json"
# Enough steps to reach the end: a token budget turns an equal number of samples into an unequal
# number of rows, so the ranks finish several steps apart. The epoch now runs until the last of them
# is done, with the ranks that finished early joining the collectives on padded work, so the whole
# epoch is consumed. On this spread that is about 16 steps for 20 samples a rank.
COMMON="--dataset.data_path=$DATA --training.train_iters=20"
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false"
HOOKED="--ep_instrument.enabled=false"
# The budget counts input_ids, which does not price the vision encoder: an image's placeholder run
# takes one position each like a text token, and also drives the tower. 0.31 is what the measured run
# implies -- the tower is 6.1% of a step on `natural` against a visual share of about a fifth of the
# tokens -- so a visual token is charged 1.31 text tokens. Rows then hold fewer real tokens than the
# budget, never more, so the memory ceiling is unchanged. The arm answers whether the vision residual
# the plain budget leaves is worth closing; 0.0 is the default and is what packed_1 and packed_2 run.
COSTAWARE="--dataloader.visual_token_weight=0.31"
RUNS=(
  "packed_1 $COMMON $LIGHT"
  "packed_2 $COMMON $LIGHT"
  "packed_costaware $COMMON $LIGHT $COSTAWARE"
  "packed_hooks $COMMON $HOOKED"
)
# packed_2:packed_1 is the repeat interval. The cost-aware arm is NOT paired with it -- weighting
# changes which samples share a row, so the fingerprints do not match and the wide interval applies.
COMPARE=("packed_2:packed_1" "packed_costaware:packed_1,packed_2")
