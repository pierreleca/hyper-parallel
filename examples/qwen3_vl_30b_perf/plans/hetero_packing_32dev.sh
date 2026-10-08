# What packing is worth. Both arms put two samples on every rank. The baseline pads them into two rows
# of the longer one's length; the candidate concatenates them into one row and tells attention where the
# boundary is. Same samples, same step, fewer tokens.
#
#   examples/qwen3_vl_30b_perf/prepare_hetero_data.py --scenario both --num-samples 640 \
#     --mean-len 4096 --max-len 8192 --mean-visual 1024 \
#     --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_short_n640 \
#     --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct
#   DATASET=short examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_32dev.sh
#
# Why a shorter corpus, and why only two samples a rank. Peak memory is linear in the tokens one rank
# holds, about 6.7 GiB plus 1.89 per thousand tokens, and a rank has 61.3. Packing does not reduce the
# token count, it removes the padding, so both arms must fit:
#
#   two packed samples of mean 4096      ~8200 tokens -> ~22 GiB
#   two padded samples, worst case 8192 ~16400 tokens -> ~38 GiB
#   four padded samples, worst case     ~32800 tokens -> ~69 GiB, over the die
#
# So four samples a rank cannot be padded at this length, and the full-length corpus cannot be packed two
# at a time either. Raise the count only with the length lowered to match.
#
# This comparison is not paired. The packed row is one sequence with one fingerprint where the padded arm
# has two, so compare_runs.py falls back to the rate over the whole run and the interval is the wide one.
# Run hetero_baseline_32dev.sh on the same dataset first to know what that interval is.
#
# Correctness is not what this measures: examples/qwen3_vl_30b_perf/check_packing.py proves on the host
# that a packed row gives the same logits as the documents run alone. What the cluster adds is whether the
# Ascend variable-length kernel agrees, which shows up as a loss that tracks the baseline's or does not.
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATASET="${DATASET:-short}"
SAMPLES_PER_RANK="${SAMPLES_PER_RANK:-2}"
DATA="/home/pl/data/qwen3_vl_30b_perf/hetero_${DATASET}_n640/vlm_conversations.json"
BATCH="--training.micro_batch_size=$SAMPLES_PER_RANK --training.global_batch_size=$((SAMPLES_PER_RANK * 32))"
COMMON="--dataset.data_path=$DATA --training.train_iters=20 $BATCH"
PADDED="--dataloader.collate_fn.packing=false"
PACKED="--dataloader.collate_fn.packing=true --model.packed_position_ids=true"
LIGHT="--hetero_profile.hooks=false --ep_instrument.enabled=false"
HOOKED="--ep_instrument.enabled=false"
RUNS=(
  "padded_1 $COMMON $LIGHT $PADDED"
  "packed_1 $COMMON $LIGHT $PACKED"
  "padded_2 $COMMON $LIGHT $PADDED"
  "packed_2 $COMMON $LIGHT $PACKED"
  "padded_hooks $COMMON $HOOKED $PADDED"
  "packed_hooks $COMMON $HOOKED $PACKED"
)
COMPARE=("packed_1,packed_2:padded_1,padded_2" "packed_hooks:padded_hooks")
