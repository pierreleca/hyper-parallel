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

Every command below runs **from the repository root** on the control node, which is where the plans'
`CONFIG=examples/...` paths resolve. The campaign looks a plan up from there and beside itself, so
`plans/hetero_packing_32dev.sh` and the bare `hetero_packing_32dev` both work; a checkout without
that lookup needs the full `examples/qwen3_vl_30b_perf/plans/...` path.

**1. Prove it on the host. No cluster, no checkpoint, seconds.**

```bash
PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packing.py
PYTHONPATH=.:examples/qwen3_vl_30b_perf python examples/qwen3_vl_30b_perf/check_padded_work.py
```

The first: six documents — four carrying images of different grid shapes, two text-only — run alone
and then packed into one row. Last run: `2.811e-01` without position ids, **`1.937e-07`** with them.
It fails if either number is the wrong way round, so it cannot pass by doing nothing.

Read what it covers precisely, because the device does it differently. Here isolation arrives
through Transformers' mask: the packed batch carries per-document position ids, Transformers reads
the restarts in them and builds a block-diagonal causal mask — printed and confirmed, queries of the
second document see its own tokens and none of the first's — and the eager kernel obeys it. Under
`attn_implementation: flash_attention_2` that mask is `None` and the kernel is expected to find the
boundaries itself, from `cu_seq_lens`, which does reach the attention's keyword arguments and routes
to `npu_fusion_attention_forward` (TND, `sparse_mode` 3, `actual_seq_qlen` from the boundaries). Two
mechanisms, one checked here and the other on a die, in step 4.

The second: a rank's padded work trains on nothing. Every parameter gradient bit-exact zero, with
the router's load-balancing term off and on, no supervised token counted, the forward's NaN
contained — and it fails if the real batch it is compared against moves no weight.

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

**4. Is the packed path correct on Ascend?** One document per row, paired against the baseline.

```bash
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_single_32dev.sh
```

`train_32dev_a3_packing_single.yaml` is the study configuration with exactly two keys flipped,
`collate_fn.packing` and `model.packed_position_ids`, so `micro_batch_size` stays 1 and every packed
row holds a single document: the same samples in the same order as `both_a`, one segment per row, and
the packed path must compute what the dense path computes. That makes the steps **pairable**, which a
real packed arm's are not. Run this before reading any packed step time — a loss that fails to track
here localises the fault in the kernel or the four-row position ids, with nothing else moved. It
recovers no padding and no imbalance, so expect the baseline's step time.

**Ran 2026-10-09**, `single_1`+`single_2` against `both_a`+`both_b`, 17 paired steps: end to end
**−0.1%** [−1.0%, +0.6%], loss mean **0.46%** worst **0.91%**, peak 38.5 GiB against 38.5. The twin
packed runs put the floor at 0.30% and 0.69%, so the packed path lands on the dense path at the
resolution two runs of one configuration leave. The gradient norm differs by 40% on average and 498%
at worst, against 15% and 43% between the twins: the forward's rounding moves router decisions, so
the set of experts carrying gradient differs, which the loss averages away and a global norm does
not. It certifies nothing either way, which is why the loss is the check.

**What one document per row cannot see.** With a single segment the variable-length path and the
dense path are the same computation, so the arm cannot tell whether the kernel read the boundaries at
all. That needs two documents in a row, and no 32-die arm can supply the pairing for it. This does,
at the level of the operator, on one die, with no model and no checkpoint:

```bash
PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packed_attention.py
PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packed_attention.py --dtype bfloat16
```

Five documents, run one at a time through the wrapper's dense path and then concatenated into one
row through its variable-length path, against each other and against a float32 reference with an
explicit block-diagonal mask. The control collapses the boundaries to a single segment and must come
out wrong. On a host without `torch_npu`, or with `--device cpu`, it compares its own two references
instead and says so.

**5. Packing, for speed.** Two campaigns, because a campaign carries one configuration and the arms
need two.

```bash
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_baseline_32dev.sh
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_32dev.sh
```

then join them with the `compare_runs.py` command in the packing plan's header. **Not paired** — a
packed row is one sequence with one fingerprint where the unpacked arm has one per sample, so the
wide interval applies.

The packing campaign runs four arms: `packed_1` and `packed_2` are the repeat pair,
`packed_costaware` adds `--dataloader.visual_token_weight=0.31`, and `packed_hooks` carries the
per-component hooks. The campaign compares `packed_costaware` against the pooled repeats itself.

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
finished early replays its last micro-batch with **every** label masked: it joins every collective
and moves the weights by nothing. The whole epoch is consumed, at the cost of 17% of rank-steps being
padded work.

Nothing is trained twice, and the claim is exact rather than small. A replayed sample is a real
forward over real tokens, so the question is whether any of it reaches the weights: it does not,
because the cross-entropy writes a gradient only at the positions it supervises and this batch
supervises none. On the real model, with the router's load-balancing term both off and on, **every
parameter gradient is bit-exact zero** — `check_padded_work.py`, which fails if the real batch does
not move the weights, so it cannot pass by doing nothing. The batch also counts no supervised token,
so it does not dilute the step's denominator and the reported loss stays the token-weighted mean over
the ranks that had data.

The forward does produce NaN, a mean over no supervised token, and it never leaves the forward:
`ModelOutputLoss` already replaces the loss of a batch with no valid label by zero. The gradient was
never NaN to begin with.

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

**The loss is a mean over few tokens.** The sample transform encodes the prompt as every message but
the last, so it supervises **only the final answer** of each conversation — about 25 tokens a sample,
so roughly 800 a step across the 32 ranks unpacked and 1600 packed. Two consequences. The trainer's
own `data/step_tokens` and `tokens/s` are therefore tiny and are not comparable with the recorder's
input-token figures, which are some 300 times larger. And "does the packed loss track the baseline's"
is a weak test on one step: compare the curve over the whole run, which is what step 4 above is for.
`data/consumed_samples` counts **rows** rather than samples under a token budget, because the meter
reads `input_ids.shape[0]`; the recorder's `documents` field is the one to trust.

**The Ascend variable-length kernel.** `run_qwen3_moe_flash_attention` opened with `del kwargs` and
hard-coded `input_layout="BNSD"`, so the study's text attention discarded the document boundaries and
asked the kernel for one causal mask over the whole row — any packed batch would have been silently
wrong. It now hands such a batch to `components/functional/npu_fusion_attention.py`, which already
implements the TND contract, and is untouched otherwise. **The host cannot exercise this**, and the
single-segment arm of step 4 cannot either: it showed the variable-length path landing on the dense
path to within the run-to-run floor, which is worth having and is not the same statement, because one
document a row is a plain causal row whichever path runs. `check_packed_attention.py` is the one that
settles it, and until it has run on a die the isolation of two documents in one row rests on reading
the code: TND, `sparse_mode` 3 and `actual_seq_qlen` from the boundaries, with `is_causal` defaulting
to true through both `GQAAttention` and `_attention_options`.

**A rank that genuinely skips the vision tower still hangs.** `text_only = placeholder` side-steps it
with the smallest image the tower accepts — one blank merge block, one image token, label masked,
checked against the tower's own feature-count rule. The real fix is forcing the tower to run on every
rank, and it has a trap I could not settle here: the tower must run at the same point in the forward
on every rank or the collective order diverges, and with activation recompute the real tower runs
twice, so a hook firing once would desynchronise.

**Smaller, worth knowing:**

- `cu_seq_lens` reaches the model through its generic keyword arguments and so passes the vision
  tower, which forwards its own `cu_seq_lens_q`/`_k` explicitly. Measured on the host: the key does
  arrive at the vision attention. It is ignored there, which is now read rather than assumed — the
  replacement matches `*.language_model.layers.*.self_attn` only, so the tower keeps Transformers'
  own implementation, and that one names `cu_seq_lens_q` and `cu_seq_lens_k` as parameters and never
  reads the unprefixed name.
- The loss weighting is exact at one micro-batch per step, which is this configuration. Above that,
  `step_token_counts = current × num_micro_steps` weights each micro-batch 1/M instead of by its token
  share — already true today with `padding: none`, and worth fixing before raising
  `global_batch_size`.
- The token budget balances `input_ids` length, which does not balance the vision tower: its cost
  follows `pixel_values`, and on `natural` a sample carries eleven images. This caps the gain but
  does not threaten the target: the tower is 6.1% of `natural`'s step, so even at four times
  imbalanced it adds 18 points of imbalance against the 131 the run carries today, and the modelled
  gain stays between +55% and +120%. The `packed_costaware` arm now measures whether closing it is
  worth it: `dataloader.visual_token_weight` charges a multimodal token that many extra text tokens
  of the budget, and the arm runs at 0.31 — 6.1% of a step over a visual share of about a fifth of
  the tokens. It reorders nothing, and a weighted row holds fewer real tokens than the budget, never
  more, so memory only improves. Default 0.0 is exactly the unweighted behaviour.
  If the arm needs more than 20 steps to drain, read its throughput over the 20 it ran rather than
  raising `train_iters`: the comparison is work per second, not epochs.
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
