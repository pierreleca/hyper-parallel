# MoE host swap on 16 A3 dies: runs and raw numbers

The measurements behind [`docs/guide/ep_host_swap.md`](../../docs/guide/ep_host_swap.md):
what was run, on what, and the per-rank tables. The commands are in
[`A3_RUNS.md`](A3_RUNS.md).

## What was run

One A3 node of 16 dies, `train_16dev_a3_ep_host_swap.yaml`: Qwen3-VL-30B-A3B-Instruct
cropped to 6 text layers (8 for the runs named `8l`), FSDP 16 and EP 16, activation
recompute `full_except_moe`, 20 steps, 320 the_cauldron samples at sequence length
16384. Memory and step time cover steps 5–20; the swap records cover steps 3–20.

| Run | What it is |
| --- | --- |
| `a3_16dev_noswap` | baseline, `--ep_host_swap.enabled=false` |
| `a3_16dev_step090` | `--ep_host_swap.capacity_factor=0.9` |
| `a3_16dev_8l_noswap` | 8 text layers, baseline (`--model.num_hidden_layers=8`) |
| `a3_16dev_8l_step100` … `_step010` | 8 text layers, factor 1.0 down to 0.1 in steps of 0.1 |
| `a3_16dev_8l_profile_noswap` / `_step080` | torch_npu profile of the 8-layer pair, all ranks, instrument off |
| `host_link_bench` | `host_link_bench.py`: dies copying to host alone and together |

Every swap run routes identically to its no-swap baseline step by step
(`analyze_ep_instrument.py --compare`: 1728 routing records at 6 layers, 2304 at 8,
0 differ), so the differences are the swap's.

```bash
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $R/a3_16dev_step090/instrument --swap-dir $R/a3_16dev_step090/ep_host_swap
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $R/a3_16dev_step090/instrument --compare $R/a3_16dev_noswap/instrument
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py --sweep $R/a3_16dev_8l_noswap $R/a3_16dev_8l_step100 ...
python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $R/a3_16dev_8l_profile_step080 --ranks
python examples/qwen3_vl_30b_perf/host_link_bench.py --report host_link_bench.json
```

## Environment

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1 HCCL_CONNECT_TIMEOUT=1800
```

- **Expandable segments are required**: without them the allocator held 10 GiB it
  could not hand out as one 9.28 GiB block (the fp32 logits gradient).
- **`TASK_QUEUE_ENABLE=1`, not 2**: level 2 gives operator workspaces their own
  allocator outside PyTorch's pool, and a 5.28 GiB workspace then found no room.
- **Depth**: 10 text layers ran out of memory on 16 dies; 8 fit without swap
  (59.4 GiB worst reserved) once the default group's communicator is created at train
  begin (its ~420 MB of HCCL buffers otherwise landed mid-step).
- **Topology**: 8 cards of 2 dies; the host side has 8 NUMA nodes of 24 cores.

## Imbalance, no swap (6 layers)

- **Per layer**, the busiest rank received 1.61 times the mean on average (median
  1.65, p90 1.81, max 1.88). **Per rank over the six layers**: 1.18 on average,
  1.20 at worst. No rank is busiest in more than 25% of (step, layer) pairs; eight
  of the sixteen never are.
- **Memory rule** (fit over every rank and layer, R² = 1.00): retained = 8712 B per
  received pair + 1645 MiB per layer. MoE memory per rank and step: mean 16.02 GiB,
  worst 17.32 GiB.
- **Allocator**: every rise of the reserve falls between the end of the last layer's
  forward and the start of backward; reserved minus allocated is 6–8 GiB per rank and
  grows over the steps. The swap neither causes nor fixes that growth.
- **Time**: waiting for the busiest rank's experts costs 102–205 ms per rank and step
  (3–5% of a 3.9 s step). The swap does not change it.

**Budget sizing from the no-swap routing.** MoE memory summed over the six layers,
against a budget of a multiple of its mean (of MoE memory, including the part every
rank holds alike; `capacity_factor` multiplies only the routed part):

| Budget (× mean MoE memory) | GiB | (step, rank) over | worst eviction, GiB |
| --- | --- | --- | --- |
| 1.00 | 16.02 | 57% | 1.30 |
| 1.02 | 16.34 | 49% | 0.98 |
| 1.05 | 16.82 | 20% | 0.50 |
| 1.10 | 17.62 | 0% | 0.00 |

Size budgets from a no-swap run: in a swap run each block's recorded memory is net of
the evictions it triggered.

## Factor 0.9, per rank (6 layers)

Run `a3_16dev_step090`. Every rank swapped in all 18 steps; no rank ever waited for a
copy back.

| Rank | reserved, no swap | reserved, 0.9 | allocated, 0.9 | to host, GiB/step | layers/step | D2H GB/s | H2D GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r0 | 52.56 | 51.58 | 44.82 | 0.93 | 1.0 | 20.8 | 26.5 |
| r1 | 52.17 | 51.60 | 44.69 | 0.79 | 1.0 | 18.4 | 25.3 |
| r2 | 51.96 | 51.33 | 43.76 | 0.62 | 1.0 | 20.3 | 31.2 |
| r3 | 53.45 | 51.91 | 45.21 | 1.59 | 2.0 | 26.5 | 36.3 |
| r4 | 51.59 | 51.14 | 44.25 | 0.39 | 1.0 | 17.5 | 21.1 |
| r5 | 52.46 | 51.39 | 45.08 | 1.04 | 1.0 | 20.7 | 26.4 |
| r6 | 53.95 | 52.94 | 45.69 | 1.96 | 3.0 | 33.5 | 35.8 |
| r7 | 53.46 | 52.50 | 45.24 | 1.53 | 2.0 | 24.4 | 31.9 |
| r8 | 52.75 | 51.63 | 44.63 | 1.06 | 1.0 | 31.4 | 45.2 |
| r9 | 53.31 | 52.18 | 45.53 | 1.30 | 1.4 | 25.0 | 31.0 |
| r10 | 54.27 | 52.64 | 45.56 | 1.82 | 2.0 | 28.7 | 38.2 |
| r11 | 53.77 | 52.46 | 45.37 | 1.72 | 2.0 | 23.1 | 33.0 |
| r12 | 53.34 | 52.69 | 45.51 | 1.43 | 2.0 | 26.1 | 43.3 |
| r13 | 53.10 | 52.05 | 45.06 | 1.17 | 1.3 | 22.7 | 27.7 |
| r14 | 54.49 | 52.62 | 44.83 | 1.95 | 2.0 | 25.7 | 35.1 |
| r15 | 52.92 | 51.84 | 44.59 | 1.08 | 1.0 | 24.8 | 26.7 |

Worst reserved 54.49 → 52.94 GiB; spread of the allocated peak across ranks
3.22 → 1.94 GiB; MoE memory per rank and step 14.74 GiB mean, 15.38 worst. Step time
of the slowest rank: mean 3.889 → 3.885 s, median 3.709 → 3.682 s.

## Factor 0.9, per rank (8 layers)

| Rank | reserved, no swap | reserved, 0.9 | allocated, no swap | allocated, 0.9 | to host, GiB/step |
| --- | --- | --- | --- | --- | --- |
| r0 | 58.14 | 57.13 | 51.18 | 50.08 | 1.10 |
| r1 | 56.74 | 55.98 | 50.17 | 49.17 | 0.97 |
| r2 | 58.56 | 57.74 | 50.88 | 50.10 | 0.78 |
| r3 | 59.31 | 57.42 | 52.26 | 50.42 | 1.85 |
| r4 | 57.94 | 57.43 | 51.48 | 50.38 | 1.08 |
| r5 | 56.86 | 55.33 | 50.27 | 48.79 | 1.40 |
| r6 | 58.78 | 56.65 | 51.94 | 49.81 | 2.16 |
| r7 | 58.48 | 57.17 | 51.20 | 49.53 | 1.67 |
| r8 | 57.79 | 56.51 | 51.16 | 49.96 | 1.21 |
| r9 | 58.26 | 56.83 | 51.24 | 49.75 | 1.46 |
| r10 | 59.20 | 57.20 | 52.48 | 50.54 | 1.96 |
| r11 | 59.16 | 57.37 | 52.39 | 50.24 | 2.09 |
| r12 | 58.99 | 57.41 | 52.60 | 50.04 | 2.51 |
| r13 | 57.37 | 56.04 | 50.61 | 49.05 | 1.52 |
| r14 | 59.44 | 56.84 | 51.58 | 49.25 | 2.35 |
| r15 | 57.29 | 56.00 | 50.27 | 49.01 | 1.28 |

Step time of the slowest rank: mean 4.391 → 4.429 s, median 4.186 → 4.273 s. Copies ran
at 15–30 GB/s to host and 23–46 GB/s back, none waited for. Routing is harder at 8
layers: per-layer max/mean 1.75 on average and 2.57 at worst (L6), step level 1.17 at
worst.

## Factor sweep (8 layers)

| Factor | to host, GiB/rank/step | alloc mean | alloc worst | reserved mean | reserved worst | step s, mean / median | D2H GB/s | copy back waited, max ms | pinned GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no swap | – | 51.36 | 52.60 | 58.27 | 59.44 | 4.391 / 4.186 | – | – | – |
| 1.0 | 0.73 | 50.62 | 51.31 | 57.65 | 58.56 | 4.434 / 4.234 | 23–35 | 0.0 | 2.88 |
| 0.9 | 1.59 | 49.76 | 50.54 | 56.81 | 57.74 | 4.429 / 4.273 | 15–30 | 0.0 | 4.12 |
| 0.8 | 2.40 | 48.95 | 49.75 | 56.04 | 56.95 | 4.440 / 4.322 | 11–33 | 0.0 | 5.38 |
| 0.7 | 3.18 | 48.18 | 48.79 | 55.25 | 56.62 | 4.427 / 4.335 | 11–33 | 0.0 | 7.31 |
| 0.6 | 3.96 | 47.40 | 47.98 | 54.51 | 56.12 | 4.479 / 4.314 | 15–27 | 0.0 | 7.38 |
| 0.5 | 4.77 | 46.58 | 47.13 | 53.77 | 55.41 | 4.472 / 4.343 | 13–29 | 0.0 | 9.56 |
| 0.4 | 5.64 | 45.71 | 46.24 | 52.98 | 54.63 | 4.515 / 4.342 | 16–31 | 0.0 | 10.62 |
| 0.3 | 6.44 | 44.91 | 45.38 | 52.32 | 54.06 | 4.476 / 4.320 | 16–31 | 0.0 | 13.31 |
| 0.2 | 7.10 | 44.25 | 44.59 | 51.96 | 53.07 | 4.443 / 4.333 | 12–31 | 21.9 | 13.00 |
| 0.1 | 7.78 | 43.58 | 43.80 | 51.64 | 53.46 | 4.555 / 4.432 | 14–26 | 99.1 | 13.00 |

- Mean reserved bought per 0.1 of factor: 0.83, 0.78, 0.79, 0.74, 0.74, 0.79, 0.66,
  0.36, 0.32 GiB (1.0 down to 0.1).
- The allocator gap decides the worst rank: from 0.8 to 0.7 the mean fell 0.79 GiB
  but the worst only 0.33.
- At 0.1 the worst rank gives some back (53.07 → 53.46 GiB): more sits in flight at once.
- The slow copiers change from one factor to the next, so the spread in D2H bandwidth
  is contention on shared links, not a fixed placement.

## Profile, 8 layers: no swap against factor 0.8

One profiled step on every rank, production kernels, instrument off; means over the
16 ranks, ms.

| | No swap | Factor 0.8 |
| --- | --- | --- |
| Step span | 4360.2 | 4381.4 (+21, +0.5%) |
| Compute | 2769.4 | 2771.4 |
| alltoallv per rank | 451.8 | 552.8 |
| — waiting for the last rank | 241.8 | 338.5 |
| — transfer after the last arrival | 210.0 | 214.3 |
| allGather in flight | 497.7 | 447.6 |
| Swap copies in flight | – | 199.3 |
| — while compute was idle | – | 121.5 |

Every rank's compute is within 3 ms of its no-swap value. Over the 40 matched alltoallv
calls, skew across ranks grows 369.0 → 472.1 ms while transfer moves 224.6 → 229.7 ms.
The per-rank compute spread (2438 to 3357 ms) is the vision and attention path on
samples of different sizes, not MoE (grouped matmul spans 30 ms of it).

## Host-link benchmark

1 GiB per die, 5 copies per measurement, GB/s. Ranks 2c and 2c+1 are the two dies of
card c.

| Scenario | dies | to host: min / median / sum | back: min / median / sum |
| --- | --- | --- | --- |
| each die alone | 1 | 39.8 / 41.2 / – | 58.4 / 59.6 / – |
| one card, both dies | 2 | 39.8 / 40.4 / 80.9 | 58.0 / 58.9 / 117.7 |
| two cards, one die each | 2 | 39.8 / 40.1 / 80.3 | 58.0 / 58.5 / 116.9 |
| first 4 dies | 4 | 36.4 / 38.3 / 154.6 | 55.0 / 56.8 / 228.5 |
| first 8 dies (4 cards) | 8 | 14.4 / 14.6 / 187.3 | 22.9 / 26.5 / 294.5 |
| one die per card | 8 | 23.3 / 25.9 / 226.5 | 37.7 / 38.9 / 342.7 |
| all 16 dies | 16 | 9.6 / 15.9 / 277.3 | 14.4 / 27.2 / 427.9 |

Alone, every die lands within 39.8–42.5 GB/s to host. Near saturation the two dies of a
card contend (8 dies packed into 4 cards get 21% less than 8 spread one per card), and
arbitration is unfair: at 16 dies one die keeps 41.7 GB/s while the slowest falls to 9.6.
