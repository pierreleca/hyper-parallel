# Data-heterogeneity features: what was built, what is proven, what to run

Two branches, both off `hetero-profile-32dev`, both pushed to `pierre`.

| branch | what it adds | state |
| --- | --- | --- |
| `data-hetero-varlen` | variable-length samples, text-only documents | proven on the host, unit tested |
| `data-hetero-packing` | packing and per-document masking, on top of the above | numerics proven on the host; the Ascend variable-length kernel is untested |

`data-hetero-packing` contains `data-hetero-varlen`, so running the second is running both. Cost-balanced
batching was left out, as asked.

---

## Run these, in this order

**1. Prove packing on the host. No cluster, no checkpoint, seconds.**

```bash
PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_packing.py
```

Three documents, two of them carrying images, run alone and then packed into one row. It prints the
largest logit difference both without position ids and with them, and fails if either is wrong way
round. Last run: `3.635e-01` without, `1.639e-07` with.

**2. Price padding.** The claim on the ideas slide is arithmetic; this measures it.

```bash
DATASET=both examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_padding_32dev.sh
```

Six runs: three repeats a side, light recorder, plus one hooked pair. **Paired comparison**, so the
interval is the tight one — padding changes the shape of a step, never which samples are in it, and the
recorder's fingerprint weights token ids by position with the pad id at zero, so a padded sample and an
unpadded one share it. Verified: a 137-token sample padded to 512 and to 16384 gives the same number.

Expect the unpadded arm to win by roughly the ratio of the ceiling to the mean sample. The padded arm
should also report `busiest/mean` near the control run's 1.09, because padding removes the data
heterogeneity entirely.

**3. Text-only documents.** Needs a dataset first.

```bash
python examples/qwen3_vl_30b_perf/prepare_hetero_data.py --scenario both --num-samples 640 \
  --text-only-share 0.3 --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_textonly_n640 \
  --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct
DATASET=textonly examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_textonly_32dev.sh
```

The first arm, `keep_1`, is the one that may hang rather than fail, and it is first and alone on purpose.
Let it time out; the campaign moves on.

**4. Packing.** Needs a shorter dataset, for the reason in the plan's header.

```bash
python examples/qwen3_vl_30b_perf/prepare_hetero_data.py --scenario both --num-samples 640 \
  --mean-len 4096 --max-len 8192 --mean-visual 1024 \
  --output-dir /home/pl/data/qwen3_vl_30b_perf/hetero_short_n640 \
  --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct
DATASET=short examples/qwen3_vl_30b_perf/hetero_campaign.sh plans/hetero_packing_32dev.sh
```

**Not paired** — a packed row is one sequence with one fingerprint where the padded arm has two, so the
wide interval applies. Run `hetero_baseline_32dev.sh` on the same dataset first to know what it is.

---

## The three things that decided the design

**Padding is not free in the experts.** The router scores a pad position like any real token; only
attention masks it. So padding to a constant costs the ratio of that constant to the mean sample, in the
decoder *and* in the MoE block.

**A packed row needs four rows of position ids, and row 0 must be the text ramp.** `get_rope_index`
returns three rows — temporal, height, width — and takes no document-boundary argument, so on a packed
row it emits one continuous ramp. Transformers segments the causal mask wherever row 0 of a four-row
`[text, T, H, W]` tensor fails to advance by one. The temporal row cannot serve: an image block holds it
*constant* across all of its tokens, so a mask read from it would cut every image away from the text
before it. Measured on a small model: `row1 = [0, 1, 2, 0, 1, 2, 2, 2, 2, 4, 5, 6]` for two documents, the
plateau being the second one's image.

**The study's text attention discarded the boundaries.** `run_qwen3_moe_flash_attention` began with
`del kwargs` and asked the kernel for one causal mask over the whole row with `input_layout="BNSD"`. Any
packed batch would have been silently wrong. It now hands such a batch to
`components/functional/npu_fusion_attention.py`, which already implements the TND variable-length
contract, and is untouched otherwise. **This is the part the host cannot test.**

---

## What is not done, and the one thing I would not trust

**A rank that genuinely skips the vision tower still hangs.** The tower is sharded over all 32 dies, so a
rank whose sample holds no image does not join the all-gather of the tower's weights. `text_only =
placeholder` sidesteps this by giving such a sample the smallest image the tower accepts — one merge block
of blank patches, one image token, label masked — so every rank stays on the same path. That is a
side-step, not a fix.

The fix is to make every rank run the tower whether or not it has an image. I did not build it, and the
reason is worth recording: the tower must run at the *same point in the forward* on every rank or the
collective order diverges and the job deadlocks on something unrelated. The real tower runs between the
embedding and the decoder, and with activation recompute it runs twice, so a hook that fires once would
desynchronise against ranks that fire twice. That needs a cluster to settle, and guessing it in code that
looks finished would be worse than leaving it out.

**Also untested on hardware:** the variable-length kernel path above, and whether `cu_seq_lens` flowing
through the model's generic keyword arguments reaches the vision tower harmlessly. The vision attention
passes its own `cu_seq_lens_q`/`cu_seq_lens_k` explicitly, so an extra unprefixed key should be ignored,
but that is reasoning, not a measurement.

---

## Environment note worth fixing

`pip show hyper_parallel` in the `hp-cpu` environment points at `/home/pl/dev/hyper-parallel-sapp`, not
this checkout. Imports resolve to this tree only because the current directory precedes the editable
finder, which holds for `python -c` and `pytest` but **not** for running a script — a script puts its own
directory on the path, not the working directory. That is why `check_packing.py` is documented with
`PYTHONPATH=.`. `AGENTS.md` says to re-run `pip install -e .` from the repository root; I did not, because
it would repoint the shared environment away from whatever `hyper-parallel-sapp` is being used for.

---

## Upstreaming

The library commits are separable from the example ones, so an MR can take just these:

```
9450581a fix(data): defer the VLM processor's transformers import
581cb270 feat(data): keep VLM samples at their own length, pad the micro-batch
804da712 feat(data): let a text-only VLM sample carry a blank image placeholder
7408a4a1 feat(data): pack VLM samples into one row, per document
```

The first is a prerequisite for the others' tests: `build_processor` imported `transformers` at module
scope, which pulled `torchvision` and made the whole `data.vlm` package unimportable on the unit-test
executors. Deferring it is what let the collator, the transform and the packing tests exist at all.

Every default is unchanged: `padding="max_length"`, `text_only="keep"`, `packing=False`. A run that sets
nothing behaves exactly as before.
