# EP imbalance and MoE host swap on A3: runbook

Two runs per configuration, identical except for the swap, then the reports:

| Config | Nodes | Parallelism | Samples |
| --- | --- | --- | --- |
| `train_16dev_a3_ep_host_swap.yaml` | 1 | FSDP 16, EP 16 | 320 |
| `train_64dev_a3_ep_host_swap.yaml` | 4 | FSDP 64, EP 16, experts FSDP-sharded over the nodes | 1280 |

The A3 nodes have no internet and no shared disk, and the code reaches them as a
zip of the A2 checkout. So nothing but code lives in the repository: data and
run outputs sit at fixed paths under `/home/pl`, the same on A2 and on every A3
node; only the checkpoint differs.

## Campaigns: `a3_campaign.sh`

Select the nodes (`cluster select`), deploy the zip to them and to the control
node, and build the plan's dataset on each node once (below). Then, from the
control node's checkout, one call runs a whole plan on that selection: it runs every entry of the plan
in order (a failed run is killed and recorded, the next one starts), then gathers
the records and writes every report next to a `SUMMARY.txt`:

```bash
examples/qwen3_vl_30b_perf/a3_campaign.sh examples/qwen3_vl_30b_perf/plans/smoke_64dev.sh
# results in /home/pl/a3_runs/smoke_64dev_<stamp>/
```

Plans live in `plans/`: `smoke_64dev.sh` (one 10-step run, does 64 dies work?) and
`sweep_64dev.sh` (baseline, budgets, a profiled pair). The sections below are the
same steps by hand.

## Where things live

| What | Path | A2 | A3 |
| --- | --- | --- | --- |
| checkpoint | A2 `/home/pl/Qwen3-VL-30B-A3B-Instruct`, A3 `/home/e00642590/Qwen3-VL-30B-A3B-Instruct` | there | on every node |
| raw cauldron Parquet files (0.8 GB, 4 files) | `/home/pl/data/the_cauldron` | downloaded once | copied once, then synced to every node |
| prepared datasets (JSON + images) | `/home/pl/data/qwen3_vl_30b_perf/<name>` | built there | built on every node from the raw files, offline |
| run outputs (records, profiles) | `/home/pl/runs/qwen3_vl_30b_perf/<run>` | written there | written on each node, gathered for the reports |

`prepare_cauldron_data.py` reads `--download-dir` (default
`/home/pl/data/the_cauldron`) and downloads only the files it does not find;
with `--offline`, as on A3, it never downloads and a missing file stops it with
the names it expects, which must sit directly in that directory:
`vsr__train-00000-of-00001-b56e9224d46b0ed3.parquet`,
`infographic_vqa__train-00000-of-00001-9187ab6377a43fd2.parquet`,
`scienceqa__train-00000-of-00001-c411546b9bc4df22.parquet`,
`finqa__train-00000-of-00001-4eb0e3dd12354fba.parquet`. A prepared dataset refers to
its images relative to its JSON file, so its directory can move as a whole.

## Once on A2: move data and runs out of the repository

```bash
O=/home/pl/hyper-parallel/outputs/qwen3_vl_30b_perf
du -sh $O/* $O/data/*                    # see what is there first
mkdir -p /home/pl/data/qwen3_vl_30b_perf /home/pl/runs/qwen3_vl_30b_perf
mv $O/data/cauldron_parquet /home/pl/data/the_cauldron
mv $O/data/* /home/pl/data/qwen3_vl_30b_perf/          # prepared datasets
rmdir $O/data
rm -rf $O/profile_moe_cauldron $O/profile_moe_cauldron_ranks $O/train_4dev_a2_profile   # traces, results already read
mv $O/* /home/pl/runs/qwen3_vl_30b_perf/               # the remaining run records (small)
```

The A2 configs now read and write these paths, so the A2 runs work as before.
Zip without outputs, in case a run wrote there:

```bash
cd /home/pl && zip -qr hyper-parallel.zip hyper-parallel -x 'hyper-parallel/outputs/*'
```

## Once on A3: place the raw data on every node

Copy the raw files to the control node, the way the zip travels:

```bash
scp -r /home/pl/data/the_cauldron <a3 control node>:/home/pl/data/     # from A2
```

Then add `/home/pl/data/the_cauldron` to `SYNC_DIRS` in the A3 `cluster.env`:
`cluster sync` then mirrors it to the nodes, a no-op once they have it. Sync the
whole pool once, so a later node selection finds it everywhere:

```bash
cluster -c <path to the full-pool cluster.env> sync
cluster -c <path to the full-pool cluster.env> exec 'ls /home/pl/data/the_cauldron'   # the four files, on every node
```

If the files sit one directory deeper (`scp -r` or `mv` into an existing
`the_cauldron` nests them), move them up a level before syncing.

## Environment

`REMOTE_ENV_SETUP` in `cluster.env` must export, next to the toolkit and the
Python environment:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1 HCCL_CONNECT_TIMEOUT=1800
```

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is required: without it, the
  first A3 run failed with 10 GiB cached that the allocator could not hand out
  as one 9.28 GiB block (the fp32 logits gradient).
- `TASK_QUEUE_ENABLE=1` (the torch_npu default) launches operators from a second
  thread. Level 2 moves more of the launch work there but gives operator
  workspaces their own allocator, outside PyTorch's pool: the 16-die baseline
  then failed to find 5.28 GiB for one workspace with the pool holding the rest
  of the die. Keep level 1 for these memory-bound runs.
- `CPU_AFFINITY_CONF=1` pins each process's threads to cores near its NPU; it
  only speeds up the host side. The A2 runs did not set it, so host-bound phases
  (router and counts) do not compare one to one.
- `HCCL_CONNECT_TIMEOUT=1800` gives slow multi-node starts time to connect.

Full determinism sets `HCCL_DETERMINISTIC` itself.

## 16 dies, one node

```bash
cluster select -a 1                      # or pick the node by hand
cluster sync
cluster exec -p 'python examples/qwen3_vl_30b_perf/prepare_cauldron_data.py \
  --output-dir /home/pl/data/qwen3_vl_30b_perf/cauldron_seq16384_n320 \
  --seq-len 16384 --num-samples 320 --processor-path /home/e00642590/Qwen3-VL-30B-A3B-Instruct \
  --download-dir /home/pl/data/the_cauldron --offline'

C=examples/qwen3_vl_30b_perf/train_16dev_a3_ep_host_swap.yaml
R=/home/pl/runs/qwen3_vl_30b_perf
cluster torchrun scripts/train_vl.py $C \
  --ep_host_swap.enabled=false --ep_instrument.output_dir=$R/a3_16dev_noswap/instrument
cluster logs                             # Ctrl-C detaches; wait until `cluster status` says it finished
cluster torchrun scripts/train_vl.py $C  # swap on
# profile of the swap run: every rank, steps 6-9, instrument off
cluster torchrun scripts/train_vl.py $C \
  --training.train_iters=10 --ep_instrument.enabled=false \
  --profiling.enabled=true --profiling.rank=-1 --profiling.start_step=6 --profiling.end_step=10 \
  --ep_host_swap.output_dir=$R/a3_16dev_profile/ep_host_swap
```

`cluster torchrun` returns at once: start each run after the previous one ends.
The dataset build is skipped where the dataset already exists.

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

## Budget sweep, one node

Every configuration against its own no-swap baseline, at 6 and 8 text layers. The
runs are sequential; `wait_run` polls until the current one ends and stops the
loop if it failed.

```bash
wait_run() {
  sleep 60
  while cluster status | grep -q RUNNING; do sleep 30; done
  ! cluster status | grep -q -e DEAD -e KILLED
}
C=examples/qwen3_vl_30b_perf/train_16dev_a3_ep_host_swap.yaml
R=/home/pl/runs/qwen3_vl_30b_perf
run() {  # run <name> <overrides...>
  local name=$1; shift
  cluster torchrun scripts/train_vl.py $C --ep_instrument.output_dir=$R/$name/instrument \
    --ep_host_swap.output_dir=$R/$name/ep_host_swap "$@" && wait_run
}
run a3_16dev_noswap --ep_host_swap.enabled=false &&
run a3_16dev_k54 --ep_host_swap.budget_layers=5.4 &&
run a3_16dev_8l_noswap --model.num_hidden_layers=8 --ep_host_swap.enabled=false &&
for K in 8.8 8.0 7.2 6.4 5.6 4.8 4.0 3.2 2.4 1.6 0.8; do
  run a3_16dev_8l_k${K/./} --model.num_hidden_layers=8 --ep_host_swap.budget_layers=$K || break
done
# profiles of one pair: every rank, steps 6-9, instrument off
for K in noswap 6.4; do
  if [ $K = noswap ]; then swap=--ep_host_swap.enabled=false; else swap=--ep_host_swap.budget_layers=$K; fi
  cluster torchrun scripts/train_vl.py $C --model.num_hidden_layers=8 $swap \
    --training.train_iters=10 --ep_instrument.enabled=false \
    --profiling.enabled=true --profiling.rank=-1 --profiling.start_step=6 --profiling.end_step=10 \
    --profiling.trace_dir=$R/a3_16dev_8l_profile_${K/./} \
    --ep_host_swap.output_dir=$R/a3_16dev_8l_profile_${K/./}/ep_host_swap && wait_run || break
done
```

Reports, from the gathered records (`8.8` is over one mean layer per layer, the
others at or under it):

```bash
runs="a3_16dev_noswap a3_16dev_k54 a3_16dev_8l_noswap"
for K in 88 80 72 64 56 48 40 32 24 16 08; do runs="$runs a3_16dev_8l_k$K"; done
cluster gather $(for run in $runs; do echo $R/$run; done) ./a3_runs   # records only, not the traces
A=./a3_runs/node0
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py --sweep $A/a3_16dev_8l_noswap $A/a3_16dev_8l_k*
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_16dev_k54/instrument --compare $A/a3_16dev_noswap/instrument
python examples/qwen3_vl_30b_perf/analyze_ep_instrument.py $A/a3_16dev_k54/instrument --swap-dir $A/a3_16dev_k54/ep_host_swap
python examples/qwen3_vl_30b_perf/replay_ep_host_swap.py $A/a3_16dev_8l_noswap/instrument
cluster exec "python examples/qwen3_vl_30b_perf/analyze_npu_trace.py $R/a3_16dev_8l_profile_64 --ranks"
```

## 64 dies, four nodes

```bash
cluster select -a 4
cluster sync
cluster exec -p 'python examples/qwen3_vl_30b_perf/prepare_cauldron_data.py \
  --output-dir /home/pl/data/qwen3_vl_30b_perf/cauldron_seq16384_n1280 \
  --seq-len 16384 --num-samples 1280 --processor-path /home/e00642590/Qwen3-VL-30B-A3B-Instruct \
  --download-dir /home/pl/data/the_cauldron --offline --passes 4'
```

The four cauldron subsets (about 14,500 conversations) fill fewer than 1280
samples of 16384 tokens in one pass; `--passes` lets the builder go through them
again under another shuffle, so a conversation can recur, packed with other
neighbours. `meta.json` records how many passes it took.

```bash
C=examples/qwen3_vl_30b_perf/train_64dev_a3_ep_host_swap.yaml
R=/home/pl/runs/qwen3_vl_30b_perf
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
cluster exec 'python examples/qwen3_vl_30b_perf/analyze_npu_trace.py /home/pl/runs/qwen3_vl_30b_perf/a3_64dev_profile \
  --rank $(ls -d /home/pl/runs/qwen3_vl_30b_perf/a3_64dev_profile/rank*_ascend_pt | head -1 | sed "s/.*rank\([0-9]*\)_.*/\1/") --waits 5 --top 10'
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
