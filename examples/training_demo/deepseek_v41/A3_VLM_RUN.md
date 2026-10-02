# DeepSeek-V4.1 multimodal smoke on a 16-device A3 node

Runbook for `train_deepseek_v41_vlm_online.yaml` on one bare-metal A3 node
(16 × Ascend 910C dies, 64 GiB HBM per die, no internet).

The recipe trains a **depth-preserving crop** of DeepSeek-V4.1-Flash: all 40
decoder layers and all 32 vision blocks at their native indices and layer
roles, with widths, expert count (384 → 16), Engram tables and per-image token
budget scaled down. Weights are randomly initialised — **no checkpoint is
needed or loaded**. It exercises the vision tower, 3×3 aligner, mHC, Engram,
shared compressed DSA attention, image/text MoE routing and the Online VLM data
path end to end.

## 1. Carry onto the node

The node has no internet, so two things travel with the deploy:

| What | Where it is now | Notes |
| --- | --- | --- |
| This branch | `ds41-vlm` (off `origin/trainer_dev`) | `master` has no `deepseek_v41`; do not rebase first |
| Model assets, 6.4 MB | `/home/pl/dev/ds41-assets/` | config + tokenizer + 2 example images + prebuilt Engram assets |

Transformers needs nothing installed: the `hp` env on the A3 node already has
**5.17.0**, which satisfies §2.

`ds41-assets/` contains exactly what the run reads — no weights. **The tree
matters**: there are two different `config.json` files, and a non-recursive copy
that flattens `inference/` fails with
`DeepSeek-V4.1 inference config is missing: …/inference/config.json`.

```text
DeepSeek-V4.1-Flash/config.json                 # nested official config, read by the crop builder
DeepSeek-V4.1-Flash/tokenizer.json              # 6.4 MB
DeepSeek-V4.1-Flash/tokenizer_config.json
DeepSeek-V4.1-Flash/chat_template.jinja
DeepSeek-V4.1-Flash/inference/config.json       # vision params, read by build_deepseek_v41_processor
DeepSeek-V4.1-Flash/inference/examples/images/{carrots,corn}.jpeg
engram_depth_preserving_d4.json                 # divisor 4 — matches the committed VLM yaml
engram_depth_preserving_d8.json                 # divisor 8 — for the OOM fallback in §5
```

Both Engram asset files are already generated, so nothing has to be prepared
from the tokenizer on the node. **The asset file must match the divisor**: `d4`
with `text_parameter_divisor=4`, `d8` with `8`.

Keep the assets *outside* the repo directory — `cluster deploy` wipes the remote
repo dir. `/home/pl/ds41-assets` is assumed below. Fan them out with the kit
rather than by hand: `cluster sync <abs-dir>` does `mkdir -p` then `rsync -a` to
the *same absolute path* on every node, so the nesting cannot be lost.

```bash
cluster sync /home/pl/ds41-assets
cluster verify /home/pl/ds41-assets/DeepSeek-V4.1-Flash/inference/config.json
cluster exec -E 'find /home/pl/ds41-assets -maxdepth 3 | sort'   # if something still disagrees
```

Adding `/home/pl/ds41-assets` to `SYNC_DIRS` in `cluster.env` makes it ride
along with every later `cluster sync` and `cluster torchrun -s`; rsync is
incremental, so after the first push it costs nothing.

## 2. Node environment

Required: torch 2.9 + torch-npu, CANN, and **Transformers 5.x**. The adapter
imports these nine symbols from `transformers.models.deepseek_v4`:
`DeepseekV4Attention`, `DeepseekV4DecoderLayer`, `DeepseekV4ForCausalLM`,
`DeepseekV4PreTrainedModel`, `DeepseekV4RMSNorm`, `DeepseekV4RotaryEmbedding`,
`DeepseekV4Experts`, `DeepseekV4MLP`, `apply_rotary_pos_emb`. All nine exist in
5.13.0 and later; 4.x has no `deepseek_v4` package at all.

The A3 `hp` env (Python 3.11.16, aarch64) ships **Transformers 5.17.0**, so
nothing needs installing. 5.17.0 was diffed against the 5.13.0 the original
report pinned: all nine symbols are present with identical method sets, and the
only signature change in them is `DeepseekV4RotaryEmbedding.__init__` gaining an
optional `device` — the adapter constructs it as
`DeepseekV4RotaryEmbedding(config)` and never calls the reworked
`compute_default_rope_parameters`. `transformers.conversion_mapping`'s
`get_checkpoint_conversion_mapping` / `register_checkpoint_conversion_mapping`
also still exist, and those call sites are `try/except ImportError` guarded and
unused on a random-init run.

These two runtime variables belong in `REMOTE_ENV_SETUP` in `cluster.env`,
beside the conda activate and the CANN `set_env.sh`, so every node and every
rank inherits them:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
```

Verify the environment on every node before anything else — `cluster exec`
sources `REMOTE_ENV_SETUP` first, so this checks the real training shell:

```bash
cluster exec 'python -c "
import transformers, hyper_parallel, torch, torch_npu
from transformers.models.deepseek_v4 import DeepseekV4Config
print(transformers.__version__, hyper_parallel.__file__, torch.npu.device_count())
"'
```

`hyper_parallel.__file__` must point into the deployed checkout on each node. If
it does not, `cluster deploy` that node again — `PYTHONPATH` alone is not enough.

## 3. Build the dataset

There is no shared filesystem, so the dataset is built **on every node**, with
identical contents at an identical absolute path. Any image set works; the two
images shipped in the asset directory are enough, since they are cycled to reach
the sample count:

```bash
cluster exec 'python -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_vlm_data \
    --output-dir /home/pl/data/deepseek_v41_vlm \
    --images /home/pl/ds41-assets/DeepSeek-V4.1-Flash/inference/examples/images \
    --num-train-samples 512 --num-valid-samples 128'
```

`cluster exec` runs inside `REPO_DIR` on all nodes in parallel. Confirm the
result is byte-identical everywhere before training, since every rank derives
its sample indices from its own copy:

```bash
cluster verify /home/pl/data/deepseek_v41_vlm/train.jsonl
```

This writes `train.jsonl`, `valid.jsonl` and `images/` in the OpenAI-`messages`
form described in `docs/guide/data/deepseek_v41_vlm_online_data_guide.md`. Image
paths stay relative to the JSONL, which is how the native Online Mapping source
resolves them — so the whole directory can be moved or re-synced as a unit.

To train on real data instead, point `--images` at an image directory and edit
the question/answer templates, or hand-write JSONL to the same contract: one
`image_url` content block per sample, last message from `assistant` (only the
final assistant text is supervised; every vision token is masked).

### Heterogeneous data from the_cauldron

The generator above is deliberately uniform — two cycled images and fixed short
answers, so every sample encodes to the same cost. That makes it a correctness
smoke and a **useless performance baseline**: a packing or balancing change has
no imbalance to recover, so a before/after comparison measures scheduler noise.

For that work, convert the_cauldron Parquet subsets instead. The converter keeps
image bytes unmodified and every conversation turn, so sample cost varies as the
corpus does:

```bash
cluster exec 'python -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_cauldron_data \
    --parquet-dir /home/pl/data/the_cauldron \
    --output-dir /home/pl/data/deepseek_v41_cauldron \
    --model-dir /home/pl/ds41-assets/DeepSeek-V4.1-Flash \
    --max-seq-len 4096 --num-valid-samples 128 --trim-images --verify 3'
```

It also writes `cost_manifest.csv` (per-sample text tokens, image tokens, total
tokens, ViT patches) and `cost_summary.json`. The summary's
`imbalance_ceiling` is `E[max over W] / E[mean] - 1` for each world size — the
fraction a perfect balancer could remove at one sample per rank per step. Read
it **before** running: it bounds any speedup the data can show, under two cost
models, because `FirstFitPackingSelector.get_sample_cost()` prices samples by
token length while vision cost scales with ViT patches. `--verify N` re-encodes
N samples through the real transform and fails if a predicted length disagrees,
so the manifest can be trusted as a cost model.

Two caveats for experiment design. `vision_min_pixels` (295936) upscales small
images, so per-image tokens span roughly 170→1024 — there are no cheap tiny
images, and heterogeneity comes from large images, image count and text length.
And the committed YAML's `token_budget: 128` with `min_buffered_samples: 1`
hands the packing selector a one-element candidate list every step: raise both
(budget toward `max_seq_len`, buffer into the tens) before expecting packing to
do anything.

## 4. Launch

`cluster torchrun` builds the `torchrun` line itself: the world size is
`NPROC_PER_NODE` (from `cluster.env`) × the selected nodes, and `node_rank` /
`master_addr` / `master_port` are computed per node. There is no per-call
process-count flag, so **`NPROC_PER_NODE` in `cluster.env` is what sets the
topology** — on a full A3 node that is 16.

The declared topology in the YAML (`ep8 / dp_shard8`) is an 8-rank recipe, so it
must be overridden to match the real world size. Keep these three in step with
`world = nodes × NPROC_PER_NODE`:

| Override | Value | Why |
| --- | --- | --- |
| `--accelerator.ep_size` | `world` | one routed expert per rank; must divide `num_routed_experts` (16) |
| `--fsdp_config.dp_shard_size` | `world` | pure FSDP, no replicate group (it sets `dp_replicate_size`) |
| `--training.global_batch_size` | `world` | one micro-batch per rank at `micro_batch_size: 1` |

One node, 16 dies:

```bash
cluster select -a 1
cluster torchrun -w --run-id ds41vlm-ep16 \
    scripts/train_vl.py \
    examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml \
    --model.config_path=/home/pl/ds41-assets/DeepSeek-V4.1-Flash \
    --model.engram_assets_path=/home/pl/ds41-assets/engram_depth_preserving_d4.json \
    --dataset.model_assets.config_path=/home/pl/ds41-assets/DeepSeek-V4.1-Flash \
    --dataset.data_path=/home/pl/data/deepseek_v41_vlm/train.jsonl \
    --accelerator.cp_size=1 \
    --accelerator.ep_size=16 \
    --fsdp_config.dp_shard_size=16 \
    --training.global_batch_size=16
```

`-w` holds the launch until every die on the selection is free; `--run-id` names
the run for `logs` / `status` / `kill`. The job is detached on each node, so it
survives the control shell. Add `-s` to rsync `SYNC_DIRS` first when the change
is code-only and the package is already installed (`cluster deploy` remains the
route for a fresh or moved checkout — note it `rm -rf`s `remote_dir`, so keep
the assets and the dataset outside it).

Then watch it:

```bash
cluster status ds41vlm-ep16        # per-node RUNNING / FINISHED / DEAD (exit N) + last log line
cluster status -w ds41vlm-ep16     # block until it ends anywhere
timeout 120 cluster logs ds41vlm-ep16   # follows forever; wrap it
cluster kill ds41vlm-ep16          # stop it everywhere
```

Scaling to two nodes needs only the three counts raised to 32 — and, because
`ep_size` must divide the expert count, a wider expert layer:

```bash
cluster select -a 2
cluster torchrun -w --run-id ds41vlm-2n \
    scripts/train_vl.py \
    examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml \
    ... \
    --model.num_routed_experts=32 \
    --accelerator.ep_size=32 \
    --fsdp_config.dp_shard_size=32 \
    --training.global_batch_size=32
```

Raising `num_routed_experts` changes the crop, so treat the two-node run as its
own case rather than a scaling measurement of the 16-rank one. Keeping
`ep_size=16` with `dp_shard_size=32` is the alternative that preserves the model
(EP16 × EDP2), at the cost of a second expert-parallel domain to reason about.

Collect the evidence from node 0 and summarise it on the control node:

```bash
cluster gather output/training_demo/deepseek_v41 /home/pl/a3_runs/ds41vlm-ep16
python examples/training_demo/deepseek_v41/summarize_deepseek_v41_vlm_run.py \
    /home/pl/a3_runs/ds41vlm-ep16/node0/.../run_vlm.log \
    --csv /home/pl/a3_runs/ds41vlm-ep16/metrics.csv \
    --svg /home/pl/a3_runs/ds41vlm-ep16/loss.svg \
    --expected-steps 10
```

If the exact 8-rank recipe as committed is ever wanted, write a standalone
config with `cluster select -o ds41-8proc.env -a 1`, set `NPROC_PER_NODE=8` in
it, and run `cluster -c ds41-8proc.env torchrun …` with no topology overrides.
It shards dense parameters over 8 instead of 16, so it needs *more* memory per
die than the command above — the 16-rank form is the better first run.

Leaving `dp_shard_size=8` at 16 ranks is also legal — the trainer derives
`dp_replicate_size = dp_size * cp_size / dp_shard_size`, giving HSDP 2×8 — but
that adds a second parallelism to debug, so prefer the explicit sizes above.

`run_deepseek_v41_vlm_online.sh` wraps the same call; note its
`NPROC_PER_NODE` default is 16 while the YAML it passes is the 8-rank recipe.

## 5. If it fails

**OOM.** The committed crop uses divisor 4 for both text and vision; the
published text-only report used divisor 8 and peaked at 38.8 GiB per die, so
divisor 4 has materially less headroom. In order:

```bash
    --activation_checkpoint.mode=full                     # recompute first, it is free accuracy
    --model.text_parameter_divisor=8 \
    --model.vision_parameter_divisor=8 \
    --model.engram_assets_path=/home/pl/ds41-assets/engram_depth_preserving_d8.json
```

Reducing the crop changes the case: record it as a changed case rather than
treating the smaller run as evidence for the declared one.

**Custom operators are not required.** `use_optimized_sparse_attention`
defaults to `False`, so sparse attention runs the dense reference path and
nothing imports `omni_training_custom_ops`; mHC falls back to
`sinkhorn_knopps` when `torch.ops.custom.npu_sinkhorn` is absent. If an Omni
import error appears, something has turned the optimized path on.

**Data-path errors** (`must end with an assistant message`, `Found N image
tokens but got M images`, missing image) are contract errors in the JSONL —
§8 of the data guide lists them. The YAML already sets
`debug.check_dataset: debug`, which raises the `hyper_parallel.data.*` logger
only; model-side loggers stay at INFO.

**`TASK_QUEUE_ENABLE=2` has caused OOM on these nodes** (a separate
`NPUWorkspaceAllocator` reservation); keep it at 1 for memory-bound runs.

**A node died but the others look alive.** `cluster status` reports the state
per node, so a `DEAD (exit N)` beside `RUNNING` means the surviving ranks are
blocked in a collective. `cluster kill -f <run>` clears the group and any stray
`torchrun`, and `cluster npu` confirms the dies came back free before relaunch.

## 6. Already verified off-node

On a CPU-only host with Transformers 5.14.1, against the staged assets and a
dataset from §3 — so these are not open questions at launch time:

- `build_deepseek_v41_processor()` loads from `inference/config.json` + the
  tokenizer and returns a `DeepseekV41Processor`.
- `DeepseekV41OmniTransform.encode_sample()` encodes a generated record to
  462 tokens: 444 vision tokens, `pixel_values` `[3774, 3, 14, 14]`, one image
  grid entry, and exactly 6 supervised tokens decoding to
  `A photograph of produce.<｜end▁of▁sentence｜>` — answer plus EOS, with the
  prompt and every vision position masked.
- Both Engram asset files generate from the official tokenizer in ~10 s
  (206,592 padded rows total, ≈0.2 GiB bf16 at divisor 4 — not a memory risk).

The model graph itself has **not** been built off-node; that first happens on
the A3 node.
