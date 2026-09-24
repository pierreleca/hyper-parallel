# EP imbalance and MoE host swap on A3: runbook

Two runs per configuration, identical except for the swap, then the reports:

| Config | Nodes | Parallelism | Samples |
| --- | --- | --- | --- |
| `train_16dev_a3_ep_host_swap.yaml` | 1 | FSDP 16, EP 16 | 320 |
| `train_64dev_a3_ep_host_swap.yaml` | 4 | FSDP 64, EP 16, experts FSDP-sharded over the nodes | 1280 |

Everything runs from the control node with cluster-kit (`NPROC_PER_NODE=16`).
Nodes have no shared disk: data is built on every node, and outputs are gathered
before the reports, which need only the Python standard library.

## Once: environment

`REMOTE_ENV_SETUP` in `cluster.env` must export, next to the toolkit and the
Python environment:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1 HCCL_CONNECT_TIMEOUT=1800
```

The checkpoint path in the YAMLs (`model.pretrained_model_name_or_path`) must
exist on every node. `outputs/` must be in `SYNC_EXCLUDES`, so runs stay
node-local.

## 16 dies, one node

```bash
cluster select -a 1                      # or pick the node by hand
cluster sync
cluster exec -p 'python examples/qwen3_vl_30b_perf/prepare_cauldron_data.py \
  --output-dir outputs/qwen3_vl_30b_perf/data/cauldron_seq16384_n320 \
  --seq-len 16384 --num-samples 320 --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct'

C=examples/qwen3_vl_30b_perf/train_16dev_a3_ep_host_swap.yaml
R=outputs/qwen3_vl_30b_perf
cluster torchrun scripts/train_vl.py $C \
  --ep_host_swap.enabled=false --ep_instrument.output_dir=$R/a3_16dev_noswap/instrument
cluster logs                             # Ctrl-C detaches; cluster status says when it is done
cluster torchrun scripts/train_vl.py $C  # swap on
# profile of the swap run: every rank, steps 6-9, instrument off
cluster torchrun scripts/train_vl.py $C \
  --training.train_iters=10 --ep_instrument.enabled=false \
  --profiling.enabled=true --profiling.rank=-1 --profiling.start_step=6 --profiling.end_step=10 \
  --ep_host_swap.output_dir=$R/a3_16dev_profile/ep_host_swap
```

Reports. The profile traces stay on the node, where the trace report runs; the
small instrument and swap records come back to the control node:

```bash
cluster exec "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $R/a3_16dev_profile --ranks"
cluster exec "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $R/a3_16dev_profile --rank 0 --waits 5 --top 10"
cluster gather $R/a3_16dev_noswap $R/a3_16dev_swap ./a3_runs
A=./a3_runs/node0
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_16dev_swap/instrument \
  --compare $A/a3_16dev_noswap/instrument
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_16dev_noswap/instrument
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_16dev_swap/instrument \
  --swap-dir $A/a3_16dev_swap/ep_host_swap
```

## 64 dies, four nodes

```bash
cluster select -a 4
cluster sync
cluster exec -p 'python examples/qwen3_vl_30b_perf/prepare_cauldron_data.py \
  --output-dir outputs/qwen3_vl_30b_perf/data/cauldron_seq16384_n1280 \
  --seq-len 16384 --num-samples 1280 --processor-path /home/pl/Qwen3-VL-30B-A3B-Instruct'
```

If the four cauldron subsets run out of conversations, build what they hold on
every node and lower `--training.train_iters` to samples / 64, the same in both runs.

```bash
C=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
R=outputs/qwen3_vl_30b_perf
cluster torchrun scripts/train_vl.py $C \
  --ep_host_swap.enabled=false --ep_instrument.output_dir=$R/a3_64dev_noswap/instrument
cluster torchrun scripts/train_vl.py $C
cluster torchrun scripts/train_vl.py $C \
  --training.train_iters=10 --ep_instrument.enabled=false \
  --profiling.enabled=true --profiling.rank=-1 --profiling.start_step=6 --profiling.end_step=10 \
  --ep_host_swap.output_dir=$R/a3_64dev_profile/ep_host_swap
```

Reports. Each node holds the traces of its 16 ranks, which form one EP group, so
`--ranks` on each node matches every all-to-all of that group; the collectives
table shows the cross-node all-gathers and reduce-scatters next to the swap:

```bash
cluster exec -p "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $R/a3_64dev_profile --ranks"
# one rank per node in detail: node N holds ranks 16N..16N+15
cluster exec 'python examples/qwen3_vl_30b_perf/analyze_npu_trace.py outputs/qwen3_vl_30b_perf/a3_64dev_profile \
  --rank $(ls -d outputs/qwen3_vl_30b_perf/a3_64dev_profile/rank*_ascend_pt | head -1 | sed "s/.*rank\([0-9]*\)_.*/\1/") --waits 5 --top 10'
```

The instrument needs every rank's records in one directory:

```bash
cluster gather $R/a3_64dev_noswap $R/a3_64dev_swap ./a3_runs
for run in a3_64dev_noswap a3_64dev_swap; do
  mkdir -p ./a3_runs/$run/instrument && cp ./a3_runs/node*/$run/instrument/rank*.jsonl ./a3_runs/$run/instrument/
done
mkdir -p ./a3_runs/a3_64dev_swap/ep_host_swap
cp ./a3_runs/node*/a3_64dev_swap/ep_host_swap/host_swap_rank*.jsonl ./a3_runs/a3_64dev_swap/ep_host_swap/
A=./a3_runs
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_64dev_swap/instrument \
  --compare $A/a3_64dev_noswap/instrument
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_64dev_noswap/instrument
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_64dev_swap/instrument \
  --swap-dir $A/a3_64dev_swap/ep_host_swap
```

The node topology is not fixed (the nodes are picked among 12 by availability),
so read the cross-node speed off the trace rather than assuming it: the 64-die
report's all-gather and reduce-scatter times against the 16-die report's.
