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

`hetero_campaign.sh --resume <campaign dir>` carries on an interrupted campaign, `--rerun <run>` redoes a run,
`--summarize` redoes the summary, and `hetero_campaign_status.sh <campaign dir>` says what `--resume` would do.
Environment knobs are listed in the script's header. Start with smoke, then probe, then the data ladder.

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

## What the numbers are, and are not

- A module's span runs from its entry, taken after the sharding hooks have unsharded its weights, to its exit,
  taken before they reshard. It is the module's own work and any collective inside it (the expert all-to-alls);
  the wait for weights is the gap before it. If the first run shows the gaps holding most of a layer's time,
  the hooks do not sit where this assumes: check the stream waits in `trace.txt` (`hetero_profile` plan).
- Inside a layer the MoE span holds the all-to-alls, so the ranks of an EP group look more alike there than
  their routing is. `analyze_ep_instrument.py` (`ep_report.txt`) splits the block into router, dispatch,
  experts, combine and aggregate and shows the wait for the last rank.
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
| `hetero_campaign.sh`, `hetero_campaign_status.sh`, `plans/hetero_*.sh` | the campaign runner and its plans |
| `analyze_hetero.py` | the heterogeneity report |
| `analyze_ep_instrument.py` | the MoE phase and routing report (records from `ep_instrument`) |
| `analyze_npu_trace.py`, `ascend_trace.py` | the Ascend profiler report (records from `profiling`) |
| `cropped_qwen3_vl.py`, `prepare_cauldron_data.py`, `parse_perf_log.py` | the model builder, the cauldron helpers, the log parser |

The recorders live in `hyper_parallel/trainer/runtime/hetero_profile.py` (configuration section
`hetero_profile`) and `hyper_parallel/distributed/expert_parallel/instrument.py` (section `ep_instrument`).
The host swap experiments that these files were taken from are on the paused branch `ep-host-swap`.
