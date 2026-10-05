# Model and data heterogeneity of Qwen3-VL-30B-A3B on 32 A3 dies: runbook

What costs step time when the model is a dense vision tower in front of a MoE decoder, and the samples
differ in length, images and image size? Each run records, per rank and step, the workload of the
sample and the device time of every component, so the answer comes from the run itself rather than from
a model of it.

| Question | Run | Where to read it |
| --- | --- | --- |
| Does the full model train on 32 dies, and at what peak memory? | `hetero_smoke`, `hetero_probe` | `MEMORY`, step time |
| How much does each kind of data heterogeneity cost, at equal mean work? | `hetero_data` | the sweep table; `IMBALANCE` |
| Which component carries the imbalance (vision, attention, MoE, head)? | any | `IMBALANCE`: where the excess comes from |
| Does the sample explain who is slow? | any | `COST MODEL`, `IMBALANCE`: rank correlation |
| What would sorting samples into steps save, measured? | `hetero_balance` | step time of `*_random` against `*_balanced` |
| Does accumulating micro-batches average the imbalance away? | `hetero_balance` | the `*_accum2` pair |
| Where does the model's own heterogeneity bite (per layer, vision, head)? | any | `MODEL`, `DECODER LAYERS`, `VISION BLOCKS`, `PIPELINE` |
| Do image and text tokens load the experts differently? | any | `ROUTING by modality` |
| What do the spans not show (kernels, stream waits, collectives)? | `hetero_profile` | `trace.txt` |
| How noisy is the baseline, and what do the recorders cost? | `hetero_baseline` | `compare_*.txt` of twin runs |
| How much could an idea gain at most? | any run with module hooks | `CEILINGS`, `ASSIGNMENT` |
| Did the idea deliver 20%, and did the loss hold? | `hetero_ab` | `compare_*.txt`: verdict, numerics, components |

## Quick start: the commands, in order

From the control node, in the root of its checkout of this branch (`hetero-profile-32dev`); two nodes of 16 dies.
Each campaign blocks until its runs end (run it in `tmux`) and writes to `/home/pl/a3_runs/<plan>_<stamp>/`.

```bash
# 0. nodes, devices, raw data, environment
cluster select -a 2                                  # leave the selection alone until the campaigns end
cluster npu                                          # every die free?
cluster exec -E 'ls /home/pl/data/the_cauldron | wc -l'    # 4 parquet files on each node
cluster exec 'env | grep -E "PYTORCH_NPU_ALLOC_CONF|TASK_QUEUE_ENABLE|CPU_AFFINITY_CONF|HCCL_CONNECT_TIMEOUT"'

# 1. the code on both nodes and on the control node (the campaign runs its analysis scripts from the control node)
git archive --format=zip --prefix=hyper-parallel/ -o ~/hetero-profile-32dev.zip hetero-profile-32dev   # dev machine
cluster deploy --dry-run ~/hetero-profile-32dev.zip /mnt/data/pl/hyper-parallel hyper_parallel
cluster deploy ~/hetero-profile-32dev.zip /mnt/data/pl/hyper-parallel hyper_parallel

# 2. datasets on every node (offline, deterministic, skipped if present); fixed and both first
build() {
  cluster exec "python examples/qwen3_vl_30b_perf/prepare_hetero_data.py \
    --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_$1_n640 --scenario $1 --num-samples 640 \
    --processor-path /home/e00642590/Qwen3-VL-30B-A3B-Instruct \
    --download-dir /home/pl/data/the_cauldron --offline"
}
for S in both fixed; do build $S; done
for S in text vision longtail natural; do build $S; done
cluster verify /home/pl/data/qwen3_vl_30b_perf/hetero_both_n640/samples.json     # same bytes on every node
cluster exec 'du -sh /home/pl/data/qwen3_vl_30b_perf/hetero_*; df -h /home/pl | tail -1'

# 3. smoke: does the whole model train, and do the recorders see what they should?
examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_smoke_32dev.sh
C=$(ls -dt /home/pl/a3_runs/hetero_smoke_32dev_* | head -1)
cat $C/SUMMARY.txt
grep -E "^RUN|text_experts|ep_exchange|note:|^ROUTING|^MEMORY" $C/smoke/report.txt
# expect: a RUN line for 32 ranks, a text_experts row, no "note: no text.experts spans", ROUTING and MEMORY present

# 4. baseline and noise floor (twin runs; what the recorders cost)
examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_baseline_32dev.sh
C=$(ls -dt /home/pl/a3_runs/hetero_baseline_32dev_* | head -1); sed -n '/^A\/B/,$p' $C/SUMMARY.txt

# 5. data ladder: size the imbalance, read the ceilings
examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_data_32dev.sh
C=$(ls -dt /home/pl/a3_runs/hetero_data_32dev_* | head -1)
sed -n '/^SWEEP/,/^$/p' $C/SUMMARY.txt
sed -n '/^CEILINGS/,/data loading/p' $C/both/report.txt $C/fixed/report.txt

# 6. optional: balanced order measured; kernel-level check (then delete the traces)
examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_balance_32dev.sh
examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_profile_32dev.sh
cluster exec 'rm -rf /home/pl/runs/qwen3_vl_30b_perf/hetero_profile_32dev_*/profile'

# 7. certify an idea: the flags that switch it on, and the dataset where it should win
IDEA="<flags of the data idea>"  DATASET=both  examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_ab_32dev.sh
IDEA="<flags of the model idea>" DATASET=fixed examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_ab_32dev.sh
C=$(ls -dt /home/pl/a3_runs/hetero_ab_32dev_* | head -1)
grep -E '^A/B|end to end|work per second|paired steps|verdict|numerics|memory' $C/compare_*.txt

# interrupted, failed, or one run to redo
examples/qwen3_vl_30b_perf/hetero_campaign_status.sh $C
examples/qwen3_vl_30b_perf/hetero_campaign.sh --resume $C [--rerun <run>]
cluster status; cluster logs; cluster kill
```

### Following a campaign

Runs per plan: `hetero_smoke_32dev.sh` 1 (8 steps); `hetero_baseline_32dev.sh` 6 (20 steps each, in the order
`both_a`, `fixed_a`, `both_b`, `fixed_b`, `both_hooks`, `both_off`) and 4 comparisons made afterwards on the control
node; `hetero_data_32dev.sh` 6 (14 steps); `hetero_balance_32dev.sh` 6 (20 and 10 steps); `hetero_profile_32dev.sh` 2;
`hetero_probe_32dev.sh` 4; `hetero_ab_32dev.sh` 6 (20 steps). They run one after the other.

The campaign's own terminal prints `=== <run> (<run id>): <overrides>` when a run starts, then nothing until it ends
(it waits for the devices to be free, then polls the run every 30 s); the same text goes to `$C/campaign.log`. From a
second terminal:

```bash
C=$(ls -dt /home/pl/a3_runs/hetero_baseline_32dev_* | head -1)           # the campaign's directory
watch -n 30 examples/qwen3_vl_30b_perf/hetero_campaign_status.sh $C       # one line per run: finished / RUNNING / not launched / FAILED
R=$(ls -t $C/*/run_id | head -1)                                          # the run that started last
cluster logs "$(cat $R)" | grep --line-buffered -E "performance/step_time|Hetero profile: |Error|OutOfMemory|Traceback"
cluster status "$(cat $R)"                                                # per node: RUNNING / FINISHED / DEAD (exit N)
cat $C/*/state                                                            # finished, or FAILED: <first error>
```

`Ctrl-C` on `cluster logs` stops the tail only. A run goes through: waiting for free devices, a silent start-up (the
checkpoint is read, the model sharded, the data workers started), step lines (`Training: 5/20 ... performance/step_time=`),
then the campaign gathers the records and writes `$C/<run>/{report,perf}.txt`. `SUMMARY.txt` and the comparisons are
written when the last run has ended; the noise floor of `both` can be read earlier, once `both_b` has finished:
`python3 examples/qwen3_vl_30b_perf/compare_runs.py --baseline $C/both_a --candidate $C/both_b --skip 1`.

A run that fails (out of memory, a crash) is killed, recorded as `FAILED: <first error>` and the campaign goes on with
the next one. `Ctrl-C` on the campaign does not stop the cluster's job: `cluster kill "$(cat $R)"`, then
`hetero_campaign.sh --resume $C`. A run longer than `RUN_TIMEOUT` (7200 s) is killed; raise it with the environment
variable. To estimate a campaign from the smoke run:

```bash
C=$(ls -dt /home/pl/a3_runs/hetero_smoke_32dev_* | head -1)
stat -c '%y  %n' $C/smoke/run_id $C/smoke/perf.txt     # launch and end of the run, reports included
grep step_time $C/smoke/perf.txt                       # step_time_median_s
# start-up = (end - launch) - 8 x step time;  baseline ~ 6 x (start-up + 20 x step time), a little less with the light recorder
```

If the smoke run runs out of memory, `plans/hetero_probe_32dev.sh` tries the recompute modes and depths; crop with
`--model.num_hidden_layers=N` in the plan's `RUNS`.

## Parallelism and memory

Two nodes of 16 dies: FSDP 32 over the dense weights, EP 16 inside each node (8 of the 128 experts per
rank, all-to-all inside the node), each expert group FSDP-sharded over the two nodes (`edp_shard 2`).
One sample per rank and step (`global_batch_size 32`, `micro_batch_size 1`). The full 48-layer model
takes about 8 GB per die for weights, gradients and BF16 AdamW moments; the rest is activations, set by the
longest sample (16384 tokens) and the recompute mode. `hetero_probe` finds what fits; crop with
`--model.num_hidden_layers=N`.

The VLM batch path of the Trainer runs TP = CP = PP = 1, so pipeline stages are not run here. The
`PIPELINE` section prices them from the measured costs instead.

## Setup (once)

The A3 nodes have no internet and no shared disk, the code reaches them as a zip, and nothing but code lives
in the repository: data and run outputs sit at fixed paths, the same on every node.

| What | Path |
| --- | --- |
| checkpoint | `/home/e00642590/Qwen3-VL-30B-A3B-Instruct` (A2: `/home/pl/Qwen3-VL-30B-A3B-Instruct`) |
| raw cauldron Parquet files (0.8 GB, 4 files) | `/home/pl/data/the_cauldron`, copied once and mirrored by `SYNC_DIRS` |
| prepared datasets | `/home/pl/data/qwen3_vl_30b_perf/hetero_<scenario>_n640`, built on every node, offline |
| run outputs | `/home/pl/runs/qwen3_vl_30b_perf/<run>`, written on each node, gathered by the campaign |

`REMOTE_ENV_SETUP` in `cluster.env` exports, next to the toolkit and the Python environment:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1 HCCL_CONNECT_TIMEOUT=1800
```

`expandable_segments` is required (the fp32 logits gradient is one 9 GiB block). Keep `TASK_QUEUE_ENABLE=1`:
level 2 gives operator workspaces their own allocator and ran a memory-bound run out of memory.
`NPROC_PER_NODE=16`; select two nodes with `cluster select -a 2` and deploy the code to them and to the
control node.

## Datasets

Six scenarios with the same mean tokens (8192) and visual tokens (2048) per sample, and a different spread
(`prepare_hetero_data.py`, which explains each in its docstring). Build them on every node:

```bash
for S in fixed text vision both longtail natural; do
  cluster exec -p "python examples/qwen3_vl_30b_perf/prepare_hetero_data.py \
    --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_${S}_n640 --scenario $S --num-samples 640 \
    --processor-path /home/e00642590/Qwen3-VL-30B-A3B-Instruct \
    --download-dir /home/pl/data/the_cauldron --offline"
done
cluster exec 'du -sh /home/pl/data/qwen3_vl_30b_perf/hetero_*; cat /home/pl/data/qwen3_vl_30b_perf/hetero_both_n640/meta.json | head -40'
```

A directory that exists is skipped, so a repeated call is free. The build is deterministic in `--seed`, so
every node gets the same files; check it with `cluster verify /home/pl/data/qwen3_vl_30b_perf/hetero_both_n640/samples.json`.
Each directory holds the images, `vlm_conversations.json` (the draw order), `vlm_conversations.balanced.json`
(the same samples sorted by estimated cost into steps), `samples.json` (the builder's statistics per sample)
and `meta.json`. The builder prints, per order, how far the slowest rank of a step is above the mean *by its
cost estimate*, before any run: a balanced order should be near 1.0, a random one well above.

Every sample has at least one image on purpose. The vision tower is sharded over all ranks, so a rank that
skipped it (a text-only sample) would not join the all-gather of its weights and the others would wait for ever.
`variable_length_transform.py` refuses a sample that would lose all its images to truncation for the same reason.
The stock transform pads every sample to `max_seq_len`; this one does not, which is what lets the text shape
differ between ranks.

## Runs

A campaign runs the entries of a plan one after the other on the selected nodes, stopping a failed run (not
the campaign) at its first error, then gathers the records, merges the ranks of both nodes and writes the
reports and a `SUMMARY.txt`:

```bash
cluster select -a 2
examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_smoke_32dev.sh
# results in /home/pl/a3_runs/hetero_smoke_32dev_<stamp>/
```

| Plan | Runs | Needs |
| --- | --- | --- |
| `hetero_smoke_32dev.sh` | one 8-step run, the whole model, the `both` dataset | `hetero_both_n640` |
| `hetero_probe_32dev.sh` | recompute `full` / `selective` / `off` at 48, 24, 12 layers | `hetero_both_n640` |
| `hetero_data_32dev.sh` | the six datasets, 14 steps each | all six |
| `hetero_balance_32dev.sh` | `both` and `longtail` in random and balanced order, and with two micro-batches per rank | `both`, `longtail` |
| `hetero_profile_32dev.sh` | the Ascend profiler on every rank for two steps, on `fixed` and `both` | `fixed`, `both` |
| `hetero_baseline_32dev.sh` | twin runs of `fixed` and `both` (the noise floor), with module hooks on, and with the recorder off | `fixed`, `both` |
| `hetero_ab_32dev.sh` | base, candidate, base, candidate with the light recorder, then one hooked pair; `IDEA=<flags>` | the dataset you pass |

`hetero_campaign.sh --resume <campaign dir>` carries on an interrupted campaign, `--rerun <run>` redoes a run,
`--summarize` redoes the summary, and `hetero_campaign_status.sh <campaign dir>` says what `--resume` would do.
Environment knobs are listed in the script's header. Start with smoke, then probe, then the baseline (it
decides how large a difference counts), then the data ladder.

The same run by hand, for one configuration (the campaign passes the three output directories for you):

```bash
R=/home/pl/runs/qwen3_vl_30b_perf/mine
cluster torchrun scripts/train_vl.py examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml \
  --hetero_profile.output_dir=$R/hetero --ep_instrument.output_dir=$R/instrument --training.train_iters=14
cluster gather $R/hetero $R/instrument ./a3_runs
mkdir -p ./a3_runs/mine/hetero && cp ./a3_runs/node*/hetero/rank*.jsonl ./a3_runs/mine/hetero/
python examples/qwen3_vl_30b_perf/analyze_hetero.py ./a3_runs/mine/hetero
python examples/qwen3_vl_30b_perf/analyze_hetero.py --sweep ./a3_runs/*/hetero      # one row per run
```

`analyze_hetero.py` needs nothing beyond the standard library and writes `microbatches.csv` (one line per
rank, step and micro-batch: the sample's workload and the time of each component) next to its JSON, for plots.

## Reading the report

`analyze_hetero.py` prints, in this order:

- **RUN**: ranks, steps, step time, the time after the last backward boundary (gradient clip, optimizer), and
  the host gap between steps (data loading and callbacks; a large one is a data-loader bottleneck).
- **DATA**: per feature (tokens, visual tokens, images, patches, vision attention pairs), the spread over all
  micro-batches and `max/mean over ranks`, how far the biggest sample of a step is above the mean one.
- **MODEL**: the time of a micro-batch by component and phase. `recompute` is the second forward of a
  checkpointed block; `loss` runs from the end of the model's forward to the gradient reaching the logits.
  The *idle between modules* line is the wait for weights (exposed all-gather) and launch gaps.
- **DECODER LAYERS / VISION BLOCKS**: the slowest indices, and whether the first and last stand out.
- **COST MODEL**: each component's time fitted on the sample's features (tokens, tokens squared for the
  decoder; patches and the vision attention cost for the vision tower), with R squared. A high R squared
  means the data decides that component's time.
- **IMBALANCE**: a rank's *work* is the sum of its modules' own time. The busiest rank over the mean, its
  excess as a share of the step, which component the excess comes from, and the rank correlation between
  the cost model's order and the measured order inside a step.
- **WHAT IF**: the same samples sorted into steps by modeled cost; the share of the excess that removes.
- **PIPELINE**: the model's pieces cut into 2, 4, 8 contiguous stages (best cut, and the even-layers cut).
- **MEMORY**: the per-step peak against tokens and patches; the longest sample sets the die's limit.
- **ROUTING by modality**: per MoE layer, the busiest EP rank over the mean for all tokens, image tokens and
  text tokens alone, and the Jensen-Shannon divergence between the two kinds' expert choices.

| What the report shows | Reading |
| --- | --- |
| `busiest/mean` near 1.0 on `fixed`, well above on `text`, `vision`, `both` | the imbalance is the data's; compare the three to see which axis costs |
| rank correlation near +1, and `hetero_balance` balanced steps much faster | the samples explain it, and grouping or packing samples by cost recovers it |
| `busiest/mean` high but rank correlation low | the imbalance comes from elsewhere (routing, a slow die, the host); see `ROUTING` and `trace.txt` |
| the excess is mostly `vision` although vision is a small share of the work | the vision tower is quadratic in the image size: bucket by image tokens, or place the tower on its own ranks |
| the excess is mostly `text_layer`, `text_moe` share large | tokens and routing: look at `ROUTING` for the layers whose busiest EP rank is far above the mean |
| `ROUTING`: image tokens alone are far more imbalanced than text, large divergence | image tokens pile onto a few experts; the imbalance follows the images of the step |
| large *idle between modules* in the vision blocks | the 27 small FSDP units are latency-bound: fuse their all-gathers, or prefetch deeper |
| a large `loss` or `head` share, and a peak that follows the tokens | the logits dominate memory and time at long samples: chunk the loss |
| `PIPELINE`: the vision tower plus the first layers outweigh a fair stage | the stage holding the tower needs fewer layers, and its cost varies with the images |
| host gap between steps of tens of ms or more | the data loader (image decode and resize) is a bottleneck: raise `dataloader.num_workers` |

## Measuring the two 20% targets

The project claims two speedups: one from an idea that targets the **data's** heterogeneity, one from an idea that
targets the **model's**. The runs above diagnose and size; this section is how they become a measurement of a claim.

**1. Fix the baseline and its noise (`hetero_baseline_32dev.sh`).** Twin runs of `fixed` and of `both`, 20 steps
each, the same 640 samples in the same order. The comparison of a run with its twin is the noise floor: a gain
smaller than it is not a result. The same plan prices the recorders: module hooks on, and the recorder off (the
trainer's own log, `perf.txt`). Every A/B run below uses the *light* recorder (`hetero_profile.hooks=false`): two
device events and one synchronization per step, nothing else.

**2. Size the idea before building it (`CEILINGS`, `ASSIGNMENT`).** A 20% speedup means the step shrinks by 16.7%,
so the idea has to remove at least that share of the step. The `CEILINGS` table lists what each family of change
could remove if it cost nothing:

| Idea targets | Row of `CEILINGS` that bounds it | Also read |
| --- | --- | --- |
| balancing samples over ranks (cost-aware assignment, bucketing, packing) | *balance the data*; *balance every module's work* | `ASSIGNMENT`: with one micro-batch per rank no assignment helps, the costliest sample is the floor; at 2, 4, 8 dealing by cost cuts the busiest rank's work by the percentage listed. `WHAT IF`: regrouping the steps |
| the vision tower's load (image-level balancing, a separate encoder group, overlap with the decoder) | *only the vision tower*; *vision tower free* | `DATA` (patches, attention pairs), `RANK GROUPS` once the design places the tower apart |
| the MoE (expert placement or replication, all-to-all overlap, router syncs) | *only the MoE experts*; *MoE router, all-to-alls and syncs free* | `ROUTING by modality`, the exchange row of `MODEL`, `ep_report.txt` |
| the weight gathers (fusing the small vision FSDP units, deeper prefetch, a different sharding of the tower) | *weight gathers hidden* (vision, decoder) | `idle between modules` in `MODEL`, `trace.txt` |
| recompute and memory (selective policies per component) | *recompute free* | `MEMORY`, per-component recompute in `MODEL` |
| the host (data loading, scheduling) | *data loading hidden* | `host gap` in `RUN` |

A row whose ceiling is below +20% cannot deliver the target alone, whatever its implementation; two rows that
overlap cannot be added. Read the ceilings on the dataset where the idea is supposed to win: `both`, `longtail` or
`natural` for the data idea, `fixed` for the model idea (no sample differs, so what is left is the model's own).
In the unpacked regime (`text`, `both`, `longtail`) the sample lengths differ; `vision` is the *packed* regime, where
every sequence has the same length and only the images differ. Judge the data idea against the baseline practice it
would replace: against packing, the `vision` ceiling is the realistic one.

**3. Record what the new code costs (`HETERO_PROFILE.region`).** A new design adds work: an all-to-all that moves
images between ranks, a scheduling decision, a gather of features. Wrap each in a region, and it appears in the
report as a `custom` component, counted in the rank's work and in the imbalance attribution:

```python
from hyper_parallel.trainer.runtime.hetero_profile import HETERO_PROFILE

with HETERO_PROFILE.region("vit_redistribute"):      # a stretch of code; "bwd" inside the autograd engine
    features = redistribute(images)
```

Modules of a new design are hooked by name with `hetero_profile.extra_roles=["my_role=regex"]` (the regex is matched
against the module path, its first group is the block index). A region outside a recorded step costs a method call.

**4. Certify the speedup (`hetero_ab_32dev.sh`, `compare_runs.py`).**

```bash
IDEA="--my_idea.enabled=true" DATASET=both  examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_ab_32dev.sh
IDEA="--my_model_idea=true"   DATASET=fixed examples/qwen3_vl_30b_perf/hetero_campaign.sh examples/qwen3_vl_30b_perf/plans/hetero_ab_32dev.sh
```

The plan runs base, candidate, base, candidate (alternation spreads the cluster's drift over both arms), then one pair
with module hooks. `compare_runs.py` pools the repeats and prints

- the speedup of the step time, end to end (with the host's gap before the step), and in samples, tokens and work
  per second, each with a 95% bootstrap interval over steps. Work is `tokens + 0.5 * visual tokens`, and the line
  below the table shows how much the verdict moves with that weight;
- the *paired* speedup when both arms ran the same samples in the same steps (matched by each micro-batch's
  fingerprint): the variance the data itself brings drops out, and this interval decides the verdict against +20%:
  **MET** if its lower end clears the target, **NOT MET** if its upper end falls short, **INCONCLUSIVE** otherwise;
- the loss and gradient norm of the paired steps (a speedup that moved them is not a speedup), the memory peak, and
  with the hooked pair the change of every component per 1k tokens, the custom components the idea added, and the
  busiest-over-mean of the ranks before and after.

An idea that regroups samples (a different order) changes which samples share a step: the arms are then unpaired, the
comparison rests on work per second, and the loss is compared as a curve over consumed samples, not step by step.

## What the code cannot measure, yet

- **Pipeline, tensor and context parallelism for the vision-language model.** The Trainer's VLM batch path requires
  TP = CP = PP = 1 on this branch, so a baseline of a design that needs them (pipeline stages that hold the vision
  tower, context-parallel long samples) cannot be run here. `PIPELINE` and `ASSIGNMENT` price them from the measured
  costs; the pipeline work lives on the `mpipe` line, where the recorder would have to be ported.
- **Text-only samples and videos** (see above): not generated, not accounted.
- **Kernel truth.** The spans come from device events at module boundaries. Their assumption (the entry stamp follows
  the weights' unshard) is checked against the Ascend trace of `hetero_profile_32dev.sh`, not automatically.
- **Convergence.** The numerics check covers the first steps on the same samples. Whether a regrouped or rebalanced
  order changes convergence needs a longer run and its loss curve.
- **Scales other than 32 dies.** Nothing is tied to 32; the ceilings and the sampling assume the sampler deals
  consecutive samples to the ranks of a step.

## Trying the report before a run

`synthetic_hetero_records.py` simulates a run with the structure of the real one (weight gathers and EP all-to-alls
that wait for the slowest rank) from rough FLOP-count costs. **Nothing in it was measured**; the report prints a
banner, and the header says `time_source: synthetic`. It exists to learn what each section shows and to test the
analysis at 32 ranks and 48 layers:

```bash
python examples/qwen3_vl_30b_perf/synthetic_hetero_records.py --out /tmp/demo/both/hetero --scenario both
python examples/qwen3_vl_30b_perf/synthetic_hetero_records.py --out /tmp/demo/both_balanced/hetero --scenario both --order balanced
python examples/qwen3_vl_30b_perf/analyze_hetero.py /tmp/demo/both/hetero --skip 1
python examples/qwen3_vl_30b_perf/compare_runs.py --baseline /tmp/demo/both --candidate /tmp/demo/both_balanced
```

## What the numbers are, and are not

- A module's span runs from its entry, taken after the sharding hooks have unsharded its weights, to its exit,
  taken before they reshard. It is the module's own work and any collective inside it (the expert all-to-alls);
  the wait for weights is the gap before it. If the first run shows the gaps holding most of a layer's time,
  the hooks do not sit where this assumes: check the stream waits in `trace.txt` (`hetero_profile` plan).
- Inside a layer the MoE span holds the all-to-alls and the wait for the slowest rank of the EP group, so on its own
  it would make the ranks of a group look alike. The `text.experts` span (the grouped GEMMs, no collective inside)
  is what separates the two: the block's span minus the experts' is `ep_exchange`, and `text_layer` keeps the
  rest. That relies on the EP path calling `module.experts(...)` as a module, as `_run_ep_local_experts` does; if
  the smoke run shows no `text_experts` row, the expert imbalance is hidden in `text_layer` (the report says so).
  `analyze_ep_instrument.py` (`ep_report.txt`) splits the block further, into router, dispatch, combine and
  aggregate.
- The report prices the critical path two ways. The rank totals (*balance the data*) say what equal total work on
  every rank would give. The per-module sum says what equal work in every module would give; every layer starts with
  a collective that all ranks join, so the step is paced by the slowest rank of each module, and this sum is the
  larger, and the one that the experts' routing and the attention's quadratic cost feed.
- The patch embedding's backward is not measured (its input has no gradient), nor is a tower root's. The
  vision tower's backward is the sum of its blocks, mergers and deepstack mergers.
- Peaks are per step (`hetero_profile.step_peaks`). With `ep_instrument.segment_peaks` on they would be reset in
  the middle of the step; the configuration turns it off.
- Videos and text-only samples are not covered: the first needs frame accounting in the recorder, the
  second would hang the all-gather of the vision tower (above).
- The cost model fits a straight line in tokens, tokens squared, patches and the vision attention cost. It
  explains how far the data predicts the time, not why; the kernel view is `hetero_profile_32dev.sh`.

## Files

| File | Role |
| --- | --- |
| `train_32dev_a3_hetero.yaml` | the configuration: 32 dies, the whole model, the recorders on |
| `prepare_hetero_data.py`, `variable_length_transform.py` | the datasets, and the transform that keeps each sample's length |
| `hetero_campaign.sh`, `hetero_campaign_status.sh`, `plans/hetero_*.sh` | the campaign runner (runs, reports, comparisons) and its plans |
| `analyze_hetero.py` | the heterogeneity report: data, components, cost model, imbalance, ceilings, what-ifs |
| `compare_runs.py` | baseline against candidate: speedup and interval, pairing, numerics, per-component change |
| `synthetic_hetero_records.py`, `hetero_sampling.py` | simulated records to learn the report; the datasets' arithmetic |
| `analyze_ep_instrument.py` | the MoE phase and routing report (records from `ep_instrument`) |
| `analyze_npu_trace.py`, `ascend_trace.py` | the Ascend profiler report (records from `profiling`) |
| `cropped_qwen3_vl.py`, `prepare_cauldron_data.py`, `parse_perf_log.py` | the model builder, the cauldron helpers, the log parser |

The recorders live in `hyper_parallel/trainer/runtime/hetero_profile.py` (configuration section
`hetero_profile`) and `hyper_parallel/distributed/expert_parallel/instrument.py` (section `ep_instrument`).
The host swap experiments that these files were taken from are on the paused branch `ep-host-swap`.
