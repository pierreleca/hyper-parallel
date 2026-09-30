# MoE host swap

Under expert parallelism the router decides, every step, how many tokens each rank
receives, and so how much activation memory each rank's MoE layers keep for
backward. `ep_host_swap` bounds that memory by a number the routing cannot change:
each rank ends every forward pass holding at most `budget_layers` **mean layers** of
these activations, and swaps the rest to pinned host memory. A mean layer is what
one MoE layer keeps when a rank receives exactly the tokens it sends. No token is
dropped and the gradients are unchanged.

On 64 Ascend A3 dies (Qwen3-VL-30B-A3B, FSDP 64, EP 16, no recompute) it cut the
worst rank's peak reserved memory by 0.1 to 3.1 GiB for 0.5 to 3.6% of median step
time, depending on the budget.

## The problem

![MoE memory per layer and per rank](../images/ep_host_swap_problem.png)

Each MoE layer keeps 8.7 KB per token-expert pair a rank received. At the step where
one rank received the most, the busiest rank of a layer held up to 1.7 times the
mean, and the lightest well below it (left); the busiest rank changes from layer to
layer and from step to step. Every rank peaks at the end of forward, holding all its
layers at once, so what decides an out-of-memory is each rank's total over the pass
(right). Without a bound, memory has to be provisioned for the
worst routing ever seen.

## How it works

![The swap through one step](../images/ep_host_swap_mechanism.png)

- **Budget.** `budget_layers` mean layers per rank. A rank's mean layer follows from
  its own tokens times top-k, known before routing, so the budget needs no collective
  and no routing can move it.
- **Trigger.** After each MoE layer the rank projects its end-of-forward total: what it
  holds plus the remaining layers at one mean layer each. While the projection is
  over the budget, it evicts. On the figure this is a threshold on what the rank
  holds (red), `budget − (L − i) × mean layer`, rising by one mean layer per layer
  to the budget at the last one; a circle is what the rank held before an eviction.
  Keeping room for the remaining layers settles most evictions early in the pass,
  and leaves little to evict after the last layer, when copies would still run at
  the step's peak. A rank that is heavy early but light later moves nothing.
- **What moves.** Whole tensors the experts saved for backward, earliest layers
  first: their backward comes last, so their copies have the most time on both ends.
  Copies run on a side stream during the next attention and are never waited for.
- **Coming back.** A swapped layer is copied back while the layer above it runs
  backward, so the reload lands after backward has freed most activations.
- **Under one mean layer per layer**, every rank evicts every step (right panel):
  the budget becomes a memory setting rather than a safety net. Over it, the budget
  is headroom for routing imbalance (left panel).

The figure draws the swap's own decisions on the rank and step that received the
most in the no-swap run (every run routes identically); the x axis counts MoE
layers, not time.

## Usage

```yaml
ep_host_swap:
  enabled: true
  budget_layers: 5.4      # mean layers a rank may keep; here 0.9 per MoE layer of 6
```

Each step a rank swapped something, it logs how many MoE layers it swapped and the
bytes it sent to host.

Fields can be overridden on the command line (`--ep_host_swap.budget_layers=3`).
On 64 A3 dies the copies stayed hidden down to 0.6 mean layers per MoE layer; below,
the compute stream starts waiting for copies back.
The swap applies to the trainer's EP path
(`hyper_parallel/distributed/expert_parallel/experts.py`) and needs the MoE blocks to
keep their activations: it requires `activation_checkpoint.mode: off`, and is refused
with compile or pipeline parallelism. On Ascend, set
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, or fragmentation can undo the bound.

## Results

![Budget sweep](../images/ep_host_swap_sweep.png)

64 A3 dies, 8 text layers, 8192 tokens, no recompute, 16 measured steps per run; every
swap run routes identically to the no-swap run, step by step.

| `budget_layers` (of 8) | Moved to host, GiB per rank and step | Worst rank's peak reserved, GiB | Median step time | Worst wait for a copy back |
| --- | --- | --- | --- | --- |
| no swap | – | 52.27 | 1.972 s | – |
| 8.8 | 0.09 | 52.15 (−0.12) | +0.5% | 0 ms |
| 8.0 | 0.38 | 51.95 (−0.32) | +0.7% | 0 ms |
| 7.2 | 0.81 | 51.50 (−0.77) | +0.8% | 0 ms |
| 6.4 | 1.22 | 51.01 (−1.25) | +1.5% | 0.2 ms |
| 4.8 | 2.00 | 50.39 (−1.88) | +2.0% | 0 ms |
| 3.2 | 2.83 | 49.74 (−2.52) | +2.1% | 8 ms |
| 2.4 | 3.24 | 49.41 (−2.86) | +3.3% | 16 ms |
| 1.6 | 3.57 | 49.16 (−3.11) | +3.6% | 19 ms |

- **Memory falls almost one-for-one with the bytes moved**: from 8.0 of 8 down, the
  ranks' mean peak falls by 0.8 to 0.94 GiB per GiB moved.
- **Above the mean the budget still acts**: at 8.8 of 8, about 20 of the 64 ranks swap
  something every step, up to 0.7 GiB each. The budget governs the routed activations
  alone, whose spread across ranks is wider than that of a rank's whole memory.
- **The step time grows with the bytes moved**, up to 3.6% at 1.6 of 8. A profile on
  16 dies (with recompute, budget 6.4 of 8) showed compute unchanged and transfers
  flat; the cost was ranks that evict arriving later at the all-to-all.
- **Below 4.8 of 8 the copies stop hiding**: the compute stream waits up to 19 ms per
  step for copies back.
- **Pinned host memory** follows the bytes moved: 0.9 to 6.8 GiB per rank, up to about
  108 GiB per 16-die node.

Deeper models, same setup:

| Text layers | Run | Worst rank's peak reserved, GiB | Median step time |
| --- | --- | --- | --- |
| 10 | 10 mean layers | 55.49 | 2.283 s |
| 10 | 2.0 mean layers | 51.53 (−3.96) | +0.7% |
| 12 | no swap | 58.40 | 2.490 s |
| 12 | 12 mean layers | 59.18 (+0.78) | +1.0% |
| 12 | 2.4 mean layers | 54.55 (−3.85) | +2.4% |
| 14 | no swap | out of memory | – |

At 0.2 mean layers per layer the worst rank's peak sits 3.1 to 4.0 GiB below the run
without swap (8 and 12 layers) or with the budget at the mean (10 layers): about 2.5
text layers' worth, as each adds about 1.5 GiB here. Whether that lets 14 layers fit
was not measured.

## Limitations

- **What the budget guarantees.** Whatever the routing, a rank ends each forward
  pass holding at most `budget_layers` mean layers of the activations the MoE layers
  keep for backward, plus evictions whose copy has not finished. This covers only the
  routed part (8.7 KB per pair); the ~1.6 GiB per layer every rank holds regardless,
  and the allocator's fragmentation (about 9 GiB reserved but not allocated per rank
  here), are outside it.
- **What it does not guarantee.** A MoE layer's working memory while it computes
  follows what the rank received in that layer. The EP path drops no tokens and has no capacity limit, so nothing bounds a
  single layer's receive: if every token chose experts on one rank, that rank would
  receive up to EP times the mean. Keeping it bounded rests on the router's load
  balancing, that is, on probability.
- **Fragmentation can eat a small saving.** A swapped layer comes back into fresh
  device memory, and evicted blocks are freed only once their copy is done. At 12
  layers with the budget at the mean, the worst rank's allocated peak fell by 1.1 GiB
  but its reserved peak rose by 0.8 GiB.
- **Evictions in flight** count until their copy finishes. The caching allocator
  should stall rather than fail when copies are late; this holds for the CUDA
  allocator and is still to be confirmed for torch_npu's.
- **The first pass** learns the number of layers, and **interleaved pipeline
  schedules** are not supported.
- **Host link.** The copies of all 16 dies of a node share the host path: with every
  rank copying, a die's copies ran at 9 to 37 GB/s.
- **Not covered:** MegaMoE (`core/multicore`), which replaces the EP expert path.
  Enabling HyperOffload at the same time is untested.
- Measured on one model and one sequence length, on 16 and 64 dies.

## Related work

Megatron-Core's MoE paged stash also caps routed-expert activations at a factor of
the balanced load and can spill to host; it fills a device pool first, spills what
comes last and reruns a step on overflow. Megatron-Core's fine-grained activation
offloading moves chosen modules' activations statically. MemFine
([arXiv 2511.21431](https://arxiv.org/abs/2511.21431)) bounds the same spikes with
chunked recompute instead of offload.

## Code

| What | Where |
| --- | --- |
| The swap | `hyper_parallel/distributed/expert_parallel/host_swap.py` |
| Config and callback | `EPHostSwapConfig` in `hyper_parallel/trainer/config/training.py`, `trainer/callbacks/ep_host_swap_callback.py` |
| Tests (CPU) | `tests/ut/dual_mode_dtensor/test_ep_host_swap.py` |
