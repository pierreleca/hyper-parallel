# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Measure how the host links of one node share bandwidth between dies.

One process per die, launched like training (``cluster torchrun``, so the same
REMOTE_ENV_SETUP and CPU affinity apply). Every die holds a device buffer and a
pinned host buffer of ``--gib`` GiB. For each scenario the chosen dies copy at
the same moment, device to host and then host to device, ``--repeat`` times in a
row, while the others wait; each die times its own copies with device events.

Scenarios: every die alone; the two dies of one card; two dies of different
cards; the first 2, 4, 8 and 16 dies; one die per card; and, from the NUMA node
each process is bound to, all dies of one NUMA node and one die per NUMA node.
Comparing a die's bandwidth alone with its bandwidth in company shows which
dies share a link and how much a shared link gives each of them.

    cluster torchrun examples/qwen3_vl_30b_perf/host_link_bench.py --out /home/pl/runs/host_link_bench.json

Rank 0 prints one line per scenario and direction (per-die GB/s: min, median,
max; and the sum over the dies that copied) and writes every measurement to
``--out``. GB/s is decimal (1e9 bytes per second), as in the host-swap records.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from typing import Any

import torch
import torch.distributed as dist

GIB = 1024 ** 3


def device_module() -> Any:
    """Return the accelerator module (``torch.npu`` with torch_npu installed, else ``torch.cuda``)."""
    try:
        import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        return torch.npu  # pylint: disable=no-member
    except ImportError:
        return torch.cuda


def device_type(device: Any) -> str:
    """Return the torch device type string of the accelerator module."""
    return "npu" if device is getattr(torch, "npu", None) else "cuda"


def cpu_numa_nodes() -> dict[int, int]:
    """Return {cpu: NUMA node} from sysfs; empty where sysfs has no NUMA nodes."""
    mapping: dict[int, int] = {}
    for path in glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"):
        node = int(re.search(r"node(\d+)", path).group(1))
        with open(path, encoding="utf-8") as stream:
            for part in stream.read().strip().split(","):
                if not part:
                    continue
                first, _, last = part.partition("-")
                for cpu in range(int(first), int(last or first) + 1):
                    mapping[cpu] = node
    return mapping


def bound_numa(cpus: set[int], mapping: dict[int, int]) -> int:
    """Return the NUMA node holding most of ``cpus``, or -1 when unknown."""
    nodes = [mapping[cpu] for cpu in cpus if cpu in mapping]
    return max(set(nodes), key=nodes.count) if nodes else -1


def build_scenarios(world: int, numa_of_rank: list[int]) -> list[tuple[str, list[int]]]:
    """Return (name, ranks that copy) for every scenario, in the order they run.

    Ranks 2c and 2c + 1 are the two dies of card c, as npu-smi's topology
    shows them (SIO between them).
    """
    scenarios = [(f"alone r{rank}", [rank]) for rank in range(world)]
    if world >= 2:
        scenarios.append(("one card, both dies (r0, r1)", [0, 1]))
    if world >= 3:
        scenarios.append(("two cards, one die each (r0, r2)", [0, 2]))
    for count in (2, 4, 8, 16):
        if count <= world:
            scenarios.append((f"first {count} dies", list(range(count))))
    if world >= 4:
        scenarios.append(("one die per card", list(range(0, world, 2))))
    nodes = sorted({node for node in numa_of_rank if node >= 0})
    if len(nodes) > 1:
        for node in nodes:
            members = [rank for rank, value in enumerate(numa_of_rank) if value == node]
            if len(members) > 1:
                scenarios.append((f"all dies on NUMA {node}", members))
        scenarios.append(("one die per NUMA node",
                          [min(rank for rank, value in enumerate(numa_of_rank) if value == node)
                           for node in nodes]))
    return scenarios


def summarize(rates: list[float]) -> dict[str, float]:
    """Per-die min / median / max and the sum over the dies, GB/s."""
    return {"min": min(rates), "median": statistics.median(rates), "max": max(rates), "sum": sum(rates)}


def timed_copies(device: Any, target: torch.Tensor, source: torch.Tensor, repeat: int) -> float:
    """Copy ``source`` into ``target`` ``repeat`` times; return GB/s from device events."""
    start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        target.copy_(source, non_blocking=True)
    end.record()
    end.synchronize()
    return target.numel() * target.element_size() * repeat / (start.elapsed_time(end) * 1e6)


def main() -> int:
    """Run every scenario and have rank 0 report them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gib", type=float, default=1.0, help="buffer size per die, GiB")
    parser.add_argument("--repeat", type=int, default=5, help="copies per measurement")
    parser.add_argument("--out", default="host_link_bench.json", help="JSON file rank 0 writes")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = device_module()
    device.set_device(int(os.environ.get("LOCAL_RANK", rank)))

    numa = bound_numa(os.sched_getaffinity(0), cpu_numa_nodes())
    numa_of_rank: list[int] = [0] * world
    dist.all_gather_object(numa_of_rank, numa)

    count = int(args.gib * GIB) // 2
    on_device = torch.ones(count, dtype=torch.bfloat16, device=f"{device_type(device)}:{device.current_device()}")
    on_host = torch.empty(count, dtype=torch.bfloat16, pin_memory=True)
    # Warm both directions once, so first-touch and driver setup stay out of the numbers.
    timed_copies(device, on_host, on_device, 1)
    timed_copies(device, on_device, on_host, 1)

    results = []
    for name, members in build_scenarios(world, numa_of_rank):
        for direction, (target, source) in (("D2H", (on_host, on_device)), ("H2D", (on_device, on_host))):
            dist.barrier()
            rate = timed_copies(device, target, source, args.repeat) if rank in members else None
            rates: list[Any] = [None] * world
            dist.all_gather_object(rates, rate)
            measured = {member: rates[member] for member in members}
            results.append({"scenario": name, "direction": direction, "ranks": members, "gbps": measured,
                            **summarize(list(measured.values()))})

    if rank == 0:
        print(f"host link bench: {world} dies, {args.gib} GiB x {args.repeat} copies per measurement")
        print("NUMA node of each rank's CPU binding: "
              + ", ".join(f"r{member} {node}" for member, node in enumerate(numa_of_rank)))
        print(f"{'scenario':38} dir   per-die GB/s: min  median    max    sum")
        for row in results:
            print(f"{row['scenario']:38} {row['direction']}  {row['min']:17.1f}{row['median']:8.1f}"
                  f"{row['max']:7.1f}{row['sum']:7.1f}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump({"world": world, "gib": args.gib, "repeat": args.repeat, "numa_of_rank": numa_of_rank,
                       "results": results}, stream, indent=1)
        print(f"wrote {args.out}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
