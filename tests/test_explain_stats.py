"""Explain lists potential switch points; stats report runtime facts only."""

from __future__ import annotations

import unittest

from join_exec import ControlPoint
from tests.util import fixture_path, make_engine
from tests.test_walkthrough_a import three_way_tree


class ExplainAndStatsTest(unittest.TestCase):
    def test_explain_potential_points_vs_runtime_facts(self):
        tree = three_way_tree()
        engine, _budget, spill, _ctrl = make_engine(
            1024 * 1024, skew_threshold=5)
        try:
            plan = engine.build(tree)
            explain = engine.explain(tree)
            # static explain never claims a switch happened
            points = {p["join_id"]: p for p in
                      explain["potential_switch_points"]}
            self.assertEqual(set(points), {"j1", "j2"})
            for p in points.values():
                self.assertIn("control_after_outputs", p["triggers"])
                self.assertIn("build_phase_budget_overflow", p["triggers"])

            # before running: no switch events, mode defaults to memory
            pre = engine.stats()["joins"]
            self.assertTrue(all(j["switch_events"] == [] for j in pre))

            ctrl = ControlPoint()
            ctrl.request_switch_after_outputs(2, "j1")
            engine2, _b2, spill2, _ctrl2 = make_engine(
                1024 * 1024, control=ctrl)
            rows = list(engine2.run(engine2.build(tree)))
            stats = engine2.stats()
            j1 = next(j for j in stats["joins"] if j["join_id"] == "j1")
            ev = j1["switch_events"][0]
            # runtime facts: timing, ranges, temp space all present and real
            self.assertEqual(ev["outputs_before_switch"], 2)
            self.assertIsInstance(ev["build_rid_range"], list)
            self.assertIsInstance(ev["probe_rid_range"], list)
            self.assertGreaterEqual(ev["spill_bytes_written_so_far"], 0)
            self.assertIn("partitions", ev)
            self.assertEqual(j1["mode_final"], "disk")
            self.assertGreater(stats["spill"]["bytes_written"], 0)
            self.assertEqual(len(rows), 12)
        finally:
            spill.cleanup()
            try:
                spill2.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
