# Data-heterogeneity features: what was built, what is proven, what to run

Two branches, both off `hetero-profile-32dev`, both pushed to `pierre`. The second contains the
first, so running it is running both. Cost-balanced batching was left out, as asked.

| branch | what it adds | state |
| --- | --- | --- |
| `data-hetero-varlen` | variable-length samples, text-only documents | proven on the host, unit tested |
| `data-hetero-packing` | packing by a token budget, per-document masking | numerics proven on the host; the Ascend variable-length kernel is untested |

Every default is unchanged — `padding="max_length"`, `text_only="keep"`, `packing=False`,
`packed_position_ids=false` — so a run that sets nothing behaves exactly as before.

---

## Run these, in this order

**1. Prove packing on the host. No cluster, no checkpoint, seconds.**

```bash
python examples/qwen3_vl_30b_perf/check_packing.py
```

Six documents — four carrying images of different grid shapes, two text-only — run alone and then
packed into one row. Last run: `2.811e-01` without position ids, **`1.937e-07`** with them. It fails
if either number is the wrong way round, so it cannot pass by doing nothing.

**2. Price padding.** Paired comparison, the tight interval.

```bash
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_padding_32dev.sh
```

Padding changes the shape of a step, never which samples are in it, and the recorder's fingerprint
weights ids by position with the pad id at zero — verified: a 137-token sample padded to 512 or to
16384 gives the identical number. So `compare_runs.py` pairs the steps.

**3. Text-only documents.** Needs a dataset first.

```bash
python examples/qwen3_vl_30b_perf/prepare_hetero_data.py --scenario both --num-samples 640 \
  --text-only-share 0.3 --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_textonly_n640 \
  --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct
DATASET=textonly examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_textonly_32dev.sh
```

The first arm, `keep_1`, is the one that may stall rather than fail, and it is first and alone on
purpose. Let it time out; the campaign moves on.

**4. Packing.** Two campaigns, because a campaign carries one configuration and the arms need two.

```bash
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_baseline_32dev.sh
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_32dev.sh
```

then join them with the `compare_runs.py` command in the packing plan's header. **Not paired** — a
packed row is one sequence with one fingerprint where the unpacked arm has one per sample, so the
wide interval applies.

---

## Version A against version B, and which is in

The collator is shared; what differs is which dataloader decides what list it gets.

| | decides the list | row length | the spread |
| --- | --- | --- | --- |
| A `FixedBatchDataLoader`, `micro_batch_size: 2` | the sampler: exactly two samples | `L₁ + L₂`, varies ~4× | **survives** |
| B `DynamicBatchDataLoader`, token budget | the lengths: fill to 16384 | ≈ the budget | **dies** |

**B is what is wired now**, in `train_32dev_a3_packing.yaml`. A was the first attempt and measured the
weaker thing: it recovers the padding two unequal samples would have needed and leaves the imbalance
between ranks exactly where it was.

Modelled on the study's spread, 32 dies, budget 16384, counting tokens:

| | steps | busiest/mean | idle | throughput |
| --- | --- | --- | --- | --- |
| unpacked, one sample a die | 40 | **2.00** | 50% | — |
| packed, buffer 2 | 26 | **1.32** | 24% | +53% |
| packed, buffer 8 | 22 | 1.12 | 11% | +80% |
| packed, buffer 200 (the library default) | 21 | 1.05 | 5% | +89% |

Read that as a direction, not a number: it assumes a step costs the slowest die's token count, and
the measured ladder already shows a step is not purely linear in tokens. The point is the mechanism —
packing changes a die's unit of work from *one sample*, which varies four-fold, to *one token
budget*, which is fixed.

**The configuration uses buffer 2, not a deeper one, on purpose.** Depth buys fill and costs
reordering, and 2 is the last depth at which nothing is reordered at all:

| `min_buffered_samples` | token cv | fill | sample drift mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| 1 or 2 | 0.26 | 76% | **0.0** | **0** | **0** |
| 8 | 0.11 | 89% | 1.3 | 4 | 6 |
| 200 (default) | 0.06 | 95% | 25.9 | 121 | 197 |

A sample that does not fit is deferred, so a deep buffer effectively picks by length — a
small-window version of the sorting that was ruled out. At buffer 2 the batcher holds only the two-odd
samples that reach the budget, takes them in order, and a deferred one simply leads the next row.
The sequence the sampler chose survives intact; only the row and step boundaries move.

**No sample is ever split.** The batcher takes a sample whole or defers it whole, and a sample larger
than the whole budget forms a row by itself rather than being cut — which is why rows come out at 76%
full rather than 100%. That is the price of not splitting, and it is the right price to pay: a split
document's second half would begin a row with no prompt in front of it and document masking puts the
first half out of reach, so the model would be trained to answer a question it was never shown.
Three unit tests guard it.

---

## Three things that would have broken, found and fixed

**Ranks would have hung, and the obvious fix would have dropped a third of the epoch.** A token
budget turns an equal number of samples into an unequal number of rows: on this study's spread, 32
ranks produce between **9 and 16** rows from the same 20 samples. The step loop caught
`StopIteration` per rank and broke, so the first rank to run out would leave while the others were
still in that step's collectives, and there is no number of steps every rank can reach.

Stopping them together fixes the hang but consumes only **69% of the epoch**, because the ranks that
still held data are cut off. So the epoch instead runs until the *last* rank is done, and a rank that
finished early replays its last micro-batch with the labels masked: it joins every collective and
moves the weights by nothing, since a micro-batch's loss is weighted by its supervised tokens and
this one keeps a single token out of the step's hundreds of thousands. One token and not none —
every label masked makes the model's cross-entropy a mean over nothing, and the weighting then
multiplies NaN by zero. The whole epoch is consumed, at the cost of 17% of rank-steps being padded
work.

**The heterogeneity report would have said there is no heterogeneity.** A packed row is one sequence
to every shape-based measure: `batch_size` reads 1 and the spread between its documents disappears, so
`real_tokens` cv would have read ~0. The recorder now reads `cu_seq_lens` and reports the document
count, the shortest and longest of them, and the pairs attention really scored.

**The cost model would have been 2.5× wrong.** Attention is quadratic in a sequence, and a packed row
is several sequences that cannot see each other, so it scores the sum of the squared lengths, not the
square of the sum. For documents of 2000, 8000 and 6000 tokens that is 1.04e8 pairs against the 2.56e8
the row's length suggests. The model takes the measured figure where the recorder supplies it.

---

## What is not validated, and the one thing I would not trust

**The Ascend variable-length kernel.** `run_qwen3_moe_flash_attention` opened with `del kwargs` and
hard-coded `input_layout="BNSD"`, so the study's text attention discarded the document boundaries and
asked the kernel for one causal mask over the whole row — any packed batch would have been silently
wrong. It now hands such a batch to `components/functional/npu_fusion_attention.py`, which already
implements the TND contract, and is untouched otherwise. **The host cannot exercise this.** It shows up
on the cluster as a loss that tracks the baseline's or does not.

**A rank that genuinely skips the vision tower still hangs.** `text_only = placeholder` side-steps it
with the smallest image the tower accepts — one blank merge block, one image token, label masked,
checked against the tower's own feature-count rule. The real fix is forcing the tower to run on every
rank, and it has a trap I could not settle here: the tower must run at the same point in the forward
on every rank or the collective order diverges, and with activation recompute the real tower runs
twice, so a hook firing once would desynchronise.

**Smaller, worth knowing:**

- `cu_seq_lens` reaches the model through its generic keyword arguments and so passes the vision
  tower, which forwards its own `cu_seq_lens_q`/`_k` explicitly. An extra unprefixed key should be
  ignored there, but that is reasoning, not a measurement.
- The loss weighting is exact at one micro-batch per step, which is this configuration. Above that,
  `step_token_counts = current × num_micro_steps` weights each micro-batch 1/M instead of by its token
  share — already true today with `padding: none`, and worth fixing before raising
  `global_batch_size`.
- The token budget balances `input_ids` length, which does not balance the vision tower: its cost
  follows `pixel_values`, and on `natural` a sample carries eleven images. This caps the gain but
  does not threaten the target: the tower is 6.1% of `natural`'s step, so even at four times
  imbalanced it adds 18 points of imbalance against the 131 the run carries today, and the modelled
  gain stays between +55% and +120%. Weighting a sample by `tokens + 0.31 x visual` in the budget
  would close most of it, and costs no reordering.
- `global_batch_size: 32` becomes decorative under a token budget — a step consumes a variable number
  of samples. Compare loss against consumed samples, never against step number, and never set
  `train_samples`.

---

## Upstreaming

The library commits are separable from the example ones:

```
fix(data): defer the VLM processor's transformers import
feat(data): keep VLM samples at their own length, pad the micro-batch
feat(data): let a text-only VLM sample carry a blank image placeholder
feat(data): pack VLM samples into one row, per document
fix(trainer): stop every rank on the same step when the data runs out
```

The first is a prerequisite for the others' tests: `build_processor` imported `transformers` at module
scope, which pulled `torchvision` and made the whole `data.vlm` package unimportable on the unit-test
executors. Deferring it is what let the collator, transform and packing tests exist at all. The last
is a genuine bug fix independent of packing — it also covers a ragged final epoch with the fixed
dataloader.
