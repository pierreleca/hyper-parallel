# What padding costs. The baseline arm is the library's default, which pads every sample to max_seq_len;
# the candidate keeps each sample at its own length and lets the collator pad the micro-batch.
#
#   DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_padding_32dev.sh
#   DATASET=natural CEILING=16384 examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_padding_32dev.sh
#
# This is a paired comparison: padding changes the shape of a step, never which samples are in it, and the
# recorder's fingerprint weights token ids by position with the pad id at zero, so a padded sample and an
# unpadded one share it. compare_runs.py therefore pairs the steps and reports the tight interval.
#
# Both arms use the same ceiling, 16384 by default: the longest sample the builder composes, and so the
# cheapest ceiling a padded run can legally use, since a lower one would truncate real samples. The study's
# own 20000 would make padding cost more (about 2.4x the tokens rather than 2.0x) and take the peak near
# 45 GiB of the 61 available. Raise CEILING to 20000 to price the configuration the other runs use.
#
# Expect the candidate to win: on hetero_both_n640 the mean sample is about 8200 real tokens, so the padded
# arm pushes roughly twice the tokens through the decoder and the experts, and the router scores a pad
# position like any other token. The padded arm also has no data heterogeneity left at all -- every rank
# runs the same shape -- so its busiest/mean should fall to the control run's figure.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATASET="${DATASET:-both}"
CEILING="${CEILING:-16384}"
DATA="/home/pl/data/qwen3_vl_30b_perf/hetero_${DATASET}_n640/vlm_conversations.json"
COMMON="--dataset.data_path=$DATA --training.train_iters=20 --dataset.data_transform.max_seq_len=$CEILING"
PADDED="--dataset.data_transform.padding=max_length"
VARLEN="--dataset.data_transform.padding=none"
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false"
HOOKED="--ep_instrument.enabled=false"
RUNS=(
  "padded_1 $COMMON $LIGHT $PADDED"
  "varlen_1 $COMMON $LIGHT $VARLEN"
  "padded_2 $COMMON $LIGHT $PADDED"
  "varlen_2 $COMMON $LIGHT $VARLEN"
  "padded_hooks $COMMON $HOOKED $PADDED"
  "varlen_hooks $COMMON $HOOKED $VARLEN"
)
COMPARE=("varlen_1,varlen_2:padded_1,padded_2" "varlen_hooks:padded_hooks")
