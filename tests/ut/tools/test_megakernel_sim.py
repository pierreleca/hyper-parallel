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
"""CPU-only tests for the host-side MegaKernel schedule simulator."""

import struct
import unittest
from collections import Counter
from dataclasses import replace

from tools.megakernel_sim.blob import parse_runtime_image
from tools.megakernel_sim.build import MoeShape, build_images
from tools.megakernel_sim.check import audit_event_balance
from tools.megakernel_sim.engine import SimConfig, Simulator
from tools.megakernel_sim.hardware import list_targets, resolve_target
from tools.megakernel_sim.semantics import Semantics
from tools.megakernel_sim.trace import build_trace
from tools.megakernel_sim.workload import balanced_route, sampled_route

_SHAPE = MoeShape(
    local_num_tokens=256,
    hidden_size=512,
    intermediate_size=128,
    num_experts=8,
    top_k=4,
    ep_size=4,
    num_cube_cores=24,
)


def _fixture(direction="forward"):
    """Build images, a balanced route and a resolver for the test shape."""
    raw_images, stage_names, _graph = build_images(_SHAPE, direction=direction)
    images = [parse_runtime_image(data, f"rank{rank}") for rank, data in enumerate(raw_images)]
    route = balanced_route(
        ep_size=_SHAPE.ep_size,
        num_experts=_SHAPE.num_experts,
        local_experts=_SHAPE.local_experts,
        local_num_tokens=_SHAPE.local_num_tokens,
        top_k=_SHAPE.top_k,
    )
    semantics = Semantics(
        images[0], stage_names, route,
        num_cube_cores=_SHAPE.num_cube_cores,
        hidden_size=_SHAPE.hidden_size,
        dtype_bytes=_SHAPE.dtype_bytes,
    )
    return images, route, semantics, raw_images


class RuntimeImageTest(unittest.TestCase):
    """The wire reader must agree with the scheduler's serializer."""

    def test_decodes_the_real_schedule(self):
        """Decoded counts match the graph the scheduler built."""
        images, _route, _semantics, _raw = _fixture()
        image = images[0]
        self.assertEqual(len(images), _SHAPE.ep_size)
        self.assertEqual(image.local_experts, _SHAPE.local_experts)
        self.assertEqual(image.num_workers, 2 * _SHAPE.num_cube_cores)
        self.assertGreater(image.ready_event, 0)
        kinds = Counter(task.type_name for task in image.tasks[: image.task_num])
        self.assertEqual(kinds["GroupedMatmul"], 2 * _SHAPE.num_cube_cores * _SHAPE.local_experts)
        self.assertEqual(kinds["ShmemPutMemSignal"] % 2, 0)

    def test_rejects_a_truncated_image(self):
        """A short image is reported rather than silently misparsed."""
        _images, _route, _semantics, raw_images = _fixture()
        with self.assertRaises(ValueError):
            parse_runtime_image(raw_images[0][:-4], "truncated")


class EventBalanceAuditTest(unittest.TestCase):
    """The audit must pass real schedules and fail broken ones."""

    def test_real_schedules_are_balanced(self):
        """Forward and backward schedules produce exactly the triggers they need."""
        for direction in ("forward", "backward"):
            with self.subTest(direction=direction):
                images, _route, semantics, _raw = _fixture(direction)
                findings = audit_event_balance(images, semantics)
                errors = [item for item in findings if item.severity == "error"]
                self.assertEqual(errors, [], f"{direction}: {[item.message for item in errors]}")

    def test_detects_an_under_triggered_event(self):
        """Raising a trigger count past its supply is reported as an error."""
        images, _route, semantics, raw_images = _fixture()
        image = images[0]
        first_gmm = next(task for task in image.tasks[: image.task_num]
                         if task.type_name == "GroupedMatmul")
        event = first_gmm.dependent_event
        corrupted = bytearray(raw_images[0])
        struct.pack_into("<i", corrupted, 64 + 4 * event, image.required(event) + 1)
        images[0] = parse_runtime_image(bytes(corrupted), "corrupted")

        findings = audit_event_balance(images, semantics)
        errors = [item for item in findings if item.severity == "error"]
        self.assertTrue(errors, "an under-triggered event must be reported")
        self.assertTrue(any(item.event == event for item in errors))
        self.assertIn("wait forever", errors[0].message)


class SimulatorTest(unittest.TestCase):
    """The replay must execute the schedule exactly once and terminate."""

    def test_every_scheduled_task_runs_once(self):
        """No task is dropped, repeated, or left blocked."""
        images, _route, semantics, _raw = _fixture()
        result = Simulator(images, semantics, resolve_target("a2"), SimConfig(),
                           num_cube_cores=_SHAPE.num_cube_cores).run()
        self.assertEqual(result.deadlocked, [])
        scheduled = set(images[0].cube_queue) | set(images[0].vector_queue)
        seen = Counter((record.rank, record.task_id) for record in result.records)
        self.assertEqual(len(result.records), len(scheduled) * _SHAPE.ep_size)
        self.assertEqual(max(seen.values()), 1)
        self.assertGreater(result.makespan_us, 0.0)

    def test_row_accounting_matches_the_route(self):
        """Simulated dispatch and combine move exactly the routed rows."""
        images, route, semantics, _raw = _fixture()
        result = Simulator(images, semantics, resolve_target("a2"), SimConfig(),
                           num_cube_cores=_SHAPE.num_cube_cores).run()
        expected = route.local_num_tokens * route.top_k * route.ep_size
        for kind in ("dispatch", "combine"):
            moved = sum(record.rows for record in result.records if record.kind == kind)
            self.assertEqual(moved, expected, kind)

    def test_removing_the_poll_backoff_cannot_slow_the_run(self):
        """An instant wake-up is never worse than the modelled re-read interval."""
        images, _route, semantics, _raw = _fixture()
        target = resolve_target("a2")
        baseline = Simulator(images, semantics, target, SimConfig(),
                             num_cube_cores=_SHAPE.num_cube_cores).run()
        ideal = Simulator(images, semantics, target, SimConfig(poll_backoff=False),
                          num_cube_cores=_SHAPE.num_cube_cores).run()
        self.assertLessEqual(ideal.makespan_us, baseline.makespan_us + 1e-6)

    def test_folding_empty_signals_preserves_trigger_totals(self):
        """Folding removes remote atomics without changing what waiters observe."""
        images, _route, semantics, _raw = _fixture()
        target = resolve_target("a2")
        baseline = Simulator(images, semantics, target, SimConfig(),
                             num_cube_cores=_SHAPE.num_cube_cores).run()
        folded = Simulator(images, semantics, target, SimConfig(fold_empty_signals=True),
                           num_cube_cores=_SHAPE.num_cube_cores).run()
        self.assertEqual(folded.deadlocked, [])
        self.assertEqual(folded.signals_sent, baseline.signals_sent)
        self.assertGreater(folded.signals_skipped, 0)


class RouteTest(unittest.TestCase):
    """Routes must respect the bound the static schedule is sized for."""

    def test_sampled_routes_never_exceed_one_row_per_token(self):
        """A token selects an expert at most once, so counts stay within bounds."""
        for skew in (0.0, 1.5):
            with self.subTest(skew=skew):
                route = sampled_route(
                    ep_size=4, num_experts=8, local_experts=2,
                    local_num_tokens=128, top_k=4, skew=skew, seed=7,
                )
                for row in route.counts:
                    self.assertEqual(sum(row), 128 * 4)
                    self.assertLessEqual(max(row), 128)

    def test_skew_raises_imbalance(self):
        """A skewed Router produces a less even receive load than a uniform one."""
        common = {"ep_size": 4, "num_experts": 8, "local_experts": 2,
                  "local_num_tokens": 128, "top_k": 4, "seed": 3}
        uniform = sampled_route(**common, skew=0.0)
        skewed = sampled_route(**common, skew=1.5)
        self.assertGreater(skewed.imbalance(), uniform.imbalance())


class HardwareTest(unittest.TestCase):
    """Targets must be resolvable and self-consistent."""

    def test_aliases_resolve(self):
        """Family labels and SoC aliases reach the same target."""
        self.assertEqual(resolve_target("a2").name, "ascend910b")
        self.assertEqual(resolve_target("A3").name, "ascend910_93")
        self.assertEqual(resolve_target("950").name, "ascend950")
        with self.assertRaises(KeyError):
            resolve_target("ascend1")

    def test_poll_interval_follows_the_system_counter(self):
        """The kernel's backoff thresholds convert through the declared counter."""
        target = resolve_target("a2")
        self.assertAlmostEqual(target.poll_interval_us(50), 50.0)
        self.assertAlmostEqual(target.poll_interval_us(150), 150.0)
        faster = replace(target, system_counter_mhz=1800.0)
        self.assertLess(faster.poll_interval_us(50), 2.0)

    def test_every_target_costs_a_transfer(self):
        """All registered targets produce a positive, finite transfer time."""
        for target in list_targets():
            with self.subTest(target=target.name):
                cost = target.transfer_us(1 << 20, same_node=True, concurrent=4)
                self.assertGreater(cost, 0.0)
                self.assertLess(cost, 1e6)


class TraceTest(unittest.TestCase):
    """The exported timeline must be viewer-safe and labelled as simulated."""

    def test_spans_do_not_overlap_on_a_thread(self):
        """Chrome Trace requires strictly nested or disjoint spans per thread."""
        images, route, semantics, _raw = _fixture()
        result = Simulator(images, semantics, resolve_target("a2"), SimConfig(),
                           num_cube_cores=_SHAPE.num_cube_cores).run()
        document = build_trace(result, route=route, direction="forward")
        self.assertFalse(document["simulation"]["measured"])
        spans = {}
        for event in document["traceEvents"]:
            if event["ph"] != "X":
                continue
            spans.setdefault((event["pid"], event["tid"]), []).append(event)
        for events in spans.values():
            events.sort(key=lambda item: item["ts"])
            for earlier, later in zip(events, events[1:]):
                self.assertLessEqual(earlier["ts"] + earlier["dur"], later["ts"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
