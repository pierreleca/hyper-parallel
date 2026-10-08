# Can a corpus that holds text-only documents train at all, and what does one cost?
#
#   examples/qwen3_vl_30b_perf/prepare_hetero_data.py --scenario both --num-samples 640 \
#     --text-only-share 0.3 --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_textonly_n640 \
#     --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct
#   DATASET=textonly examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_textonly_32dev.sh
#
# Run the first arm alone first (--only keep_1): it is the one that may hang rather than fail. With the
# vision tower sharded over all 32 dies, a rank whose sample carries no image does not join the all-gather
# of the tower's weights, and the other 31 wait for it. text_only=keep is that configuration, kept here so
# the failure is observed rather than assumed; if it hangs, the run times out and the campaign moves on.
#
# text_only=placeholder is the fix: the smallest image the tower accepts, one merge block of blank patches,
# with the single image token the model's feature-count check requires and a masked label. Every rank then
# runs the tower on every step. This arm is the one expected to train.
#
# The comparison is paired only within an arm, not across: the placeholder adds one token to a third of the
# samples, which changes their fingerprint. Read the two placeholder repeats against each other for the
# noise, and the throughput against the hetero_both_n640 ladder for the cost of the text-only share.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATASET="${DATASET:-textonly}"
DATA="/home/pl/data/qwen3_vl_30b_perf/hetero_${DATASET}_n640/vlm_conversations.json"
COMMON="--dataset.data_path=$DATA --training.train_iters=20"
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false"
HOOKED="--ep_instrument.enabled=false"
KEEP="--dataset.data_transform.text_only=keep"
PLACEHOLDER="--dataset.data_transform.text_only=placeholder"
RUNS=(
  "keep_1 $COMMON $LIGHT $KEEP"
  "placeholder_1 $COMMON $LIGHT $PLACEHOLDER"
  "placeholder_2 $COMMON $LIGHT $PLACEHOLDER"
  "placeholder_hooks $COMMON $HOOKED $PLACEHOLDER"
)
COMPARE=("placeholder_2:placeholder_1")
