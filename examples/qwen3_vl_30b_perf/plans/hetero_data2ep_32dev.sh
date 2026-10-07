# Data heterogeneity at equal work, measured over a whole epoch, after a whole epoch of warm-up.
#
# Same six datasets as hetero_data_32dev.sh, two epochs each, and the reports cover the second one
# entirely. Two things the 14-step version leaves open:
#
# - equal work. A dataset holds 640 samples, 20 steps of 32; 14 steps consume 448 of them, so the
#   scenarios did not consume the same work and their step times had to be divided by what each one
#   actually carried (measured: 7954 to 8150 tokens, 1.5%). A whole epoch consumes all 640, and every
#   scenario was built to the same mean, so the work is equal by construction, nothing needs
#   normalising, and natural -- which nothing rescales -- becomes comparable with the rest.
# - warm-up. The allocator grows, the HCCL links are touched for the first time and the caches fill
#   during the first steps. start_step 3 drops the worst of it, but nothing says 3 is enough.
#
# train_iters 40 is exactly two epochs: len(dataloader) is 640 / (micro_batch_size x dp_size) = 20, so
# train_steps is 20 and train_epochs is 2. The "single" sampler is sequential with no index mapping and
# set_epoch only rewinds consumed_samples, so epoch 2 replays epoch 1 sample for sample: step 20 + k
# carries the same 32 samples as step k. Comparing the two epochs therefore measures the warm-up on
# identical data, which no choice of start_step can do.
#
# start_step 1 on both recorders is what keeps the two concerns from cancelling each other. A recorded
# step is numbered global_step + 1, so the 40 steps are 1 to 40, epoch 1 is 1 to 20 and epoch 2 is 21
# to 40; the configuration's own start_step 3 would leave epoch 1 incomplete in the records and force
# SKIP to eat into epoch 2, dropping the equal work it was for. From 1 everything is recorded, and
# SKIP 20 drops epoch 1 whole and keeps epoch 2 whole -- all 20 steps, all 640 samples, nothing
# dropped inside the epoch that is measured. (start_step 0 is rejected: the window could never open.)
#
# Step 21 opens the second epoch and is kept. It is sound to keep it: a new epoch calls
# iter(train_dataloader), which respawns the 4 data-worker processes per rank, but that cost falls
# between steps and the report times a step by its own device span -- the host gap is a separate
# column. Watch it anyway: "step <mean> ms (steps <min> to <max>)" and step_ms_cv would show step 21
# as an outlier, and the gap column would show the respawn.
#
# Epoch over epoch, from the records already gathered, at no device cost (A3_HETERO_RUNS.md,
# "Two epochs"):
#   analyze_hetero.py --sweep --steps  1:20 <campaign>/*/hetero   # epoch 1, warm-up included
#   analyze_hetero.py --sweep --steps 21:40 <campaign>/*/hetero   # epoch 2, what the plan reports
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATA=/home/pl/data/qwen3_vl_30b_perf
# Drop epoch 1 (steps 1-20) whole, keep epoch 2 (steps 21-40) whole. Needs start_step 1 below.
SKIP=20
# perf.txt reads the trainer's own log, which logs every step from the first: keep the same range.
SKIP_LOG=20
RUNS=()
for scenario in fixed text vision both longtail natural; do
  RUNS+=("$scenario --training.train_iters=40 --hetero_profile.start_step=1 --ep_instrument.start_step=1 --dataset.data_path=$DATA/hetero_${scenario}_n640/vlm_conversations.json")
done
