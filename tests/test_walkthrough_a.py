"""Main walkthrough A: hand-computed results across every switch boundary.

Relations (see fixtures):
  A: rids 1,2 have k=10; rid 3 k=99 unmatched; rid 4 k=NULL.
  B: rids 10,11,12 have k=10; rid 13 k=NULL.
  C: rids 100,101 k=10; rid 102 k=99.

Hand expectations
-----------------
INNER  A x B on k: (1,2) x (10,11,12) = 6 distinct lineage pairs;
                   NULL rows never connect (A4, B13 excluded).
LEFT   same keys : 6 matched pairs + 1 pad for A3 + 1 pad for A4 = 8 rows,
                   each unmatched left padded exactly once.
SEMI   same keys : only A1, A2 -> 2 rows.
3-way  (A INNER B) INNER C on k: 6 AB pairs x 2 C rows = 12 pairs.
Aggregate over the 3-way result:
  count = 12; sum of a.v = each of (100,200) repeated 3*2=6 times
  -> 6*100 + 6*200 = 1800.
"""

from __future__ import annotations

import unittest

from join_exec import ControlPoint
from tests.util import (
    fixture_path,
    lineage_set,
    make_engine,
    reference_bag,
    run_tree,
)


def ab_tree(join_type: str) -> dict:
    return {
        "node": "join", "id": "j1", "type": join_type,
        "left": {"node": "scan", "relation": "A", "alias": "a",
                 "file": fixture_path("A.jsonl")},
        "right": {"node": "scan", "relation": "B", "alias": "b",
                  "file": fixture_path("B.jsonl")},
        "on": [["a.k", "b.k"]],
    }


def three_way_tree() -> dict:
    return {
        "node": "join", "id": "j2", "type": "inner",
        "left": {
            "node": "join", "id": "j1", "type": "inner",
            "left": {"node": "scan", "relation": "A", "alias": "a",
                     "file": fixture_path("A.jsonl")},
            "right": {"node": "scan", "relation": "B", "alias": "b",
                      "file": fixture_path("B.jsonl")},
            "on": [["a.k", "b.k"]],
        },
        "right": {"node": "scan", "relation": "C", "alias": "c",
                  "file": fixture_path("C.jsonl")},
        "on": [["a.k", "c.k"]],
    }


def aggregate_tree() -> dict:
    return {
        "node": "aggregate",
        "keys": [],
        "sum": "a.v",
        "input": three_way_tree(),
    }


EXPECTED_INNER_PAIRS = {
    tuple(sorted([("a", 1), ("b", 10)])),
    tuple(sorted([("a", 1), ("b", 11)])),
    tuple(sorted([("a", 1), ("b", 12)])),
    tuple(sorted([("a", 2), ("b", 10)])),
    tuple(sorted([("a", 2), ("b", 11)])),
    tuple(sorted([("a", 2), ("b", 12)])),
}


class WalkthroughATest(unittest.TestCase):

    def test_inner_always_memory(self):
        rows, stats, _ = run_tree(ab_tree("inner"), mode="force_memory")
        self.assertEqual(len(rows), 6)
        self.assertEqual(lineage_set(rows), EXPECTED_INNER_PAIRS)
        self.assertEqual(stats["joins"][0]["mode_final"], "memory")
        self.assertEqual(stats["joins"][0]["switch_events"], [])

    def test_inner_always_disk(self):
        rows, stats, _ = run_tree(ab_tree("inner"), mode="force_disk")
        self.assertEqual(len(rows), 6)
        self.assertEqual(lineage_set(rows), EXPECTED_INNER_PAIRS)
        self.assertEqual(stats["joins"][0]["mode_final"], "disk")
        self.assertEqual(len(stats["joins"][0]["switch_events"]), 1)

    def test_inner_switch_after_two_outputs(self):
        # The headline scenario: switch after exactly the 2nd emitted pair;
        # already delivered pairs are not replayed and the total stays 6.
        ctrl = ControlPoint()
        ctrl.request_switch_after_outputs(2, join_id="j1")
        rows, stats, _ = run_tree(ab_tree("inner"), control=ctrl)
        self.assertEqual(len(rows), 6)
        self.assertEqual(lineage_set(rows), EXPECTED_INNER_PAIRS)
        ev = stats["joins"][0]["switch_events"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["outputs_before_switch"], 2)
        self.assertEqual(ev[0]["reason"], "after_output_control")
        self.assertGreaterEqual(ev[0]["partitions"], 1)
        # residual carries the in-flight left row and its 2 delivered rids
        self.assertIsNotNone(ev[0]["residual"])
        self.assertEqual(
            sorted(ev[0]["residual"]["suppressed_right_rids"]), [10, 11])

    def test_inner_switch_boundaries_against_reference(self):
        for n in range(0, 7):
            with self.subTest(after_outputs=n):
                ctrl = ControlPoint()
                if n:
                    ctrl.request_switch_after_outputs(n, join_id="j1")
                else:
                    ctrl.request_switch_after_input_batches(1, join_id="j1")
                rows, _, _ = run_tree(
                    ab_tree("inner"), control=ctrl,
                    join_overrides={"j1": {"batch_size": 2}})
                ref = reference_bag(ab_tree("inner"))
                self.assertEqual(len(rows), len(ref))
                self.assertEqual(lineage_set(rows),
                                 {tuple(sorted(l)) for l, _ in ref})

    def test_batch_and_output_triggers_share_switch(self):
        # Both control channels must end in the same disk mode with identical
        # bags, exercising one shared _switch implementation.
        ref = {tuple(sorted(l)) for l, _ in reference_bag(ab_tree("inner"))}
        ctrl_b = ControlPoint()
        ctrl_b.request_switch_after_input_batches(1, "j1")
        rows_b, stats_b, _ = run_tree(
            ab_tree("inner"), control=ctrl_b,
            join_overrides={"j1": {"batch_size": 2}})
        ctrl_o = ControlPoint()
        ctrl_o.request_switch_after_outputs(1, "j1")
        rows_o, stats_o, _ = run_tree(ab_tree("inner"), control=ctrl_o)
        self.assertEqual(stats_b["joins"][0]["mode_final"], "disk")
        self.assertEqual(stats_o["joins"][0]["mode_final"], "disk")
        self.assertEqual(lineage_set(rows_b), ref)
        self.assertEqual(lineage_set(rows_o), ref)

    def test_left_join_padding_once(self):
        # 6 matches + A3 pad + A4(null) pad = 8 rows, two pads total.
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(2, join_id="j1")
                rows, stats, _ = run_tree(ab_tree("left"), mode=mode,
                                          control=ctrl)
                self.assertEqual(len(rows), 8)
                pads = [r for r in rows
                        if all(v is None for v in
                               r.values[len(r.values) - 1:])]
                # two padded rows (A3 and A4); B has one column
                pad_rows = [r for r in rows if r.values[-1] is None]
                self.assertEqual(len(pad_rows), 2)
                padded_left = sorted(r.rid_of("a") for r in pad_rows)
                self.assertEqual(padded_left, [3, 4])
                self.assertEqual(stats["joins"][0]["left_padded"], 2)
                ref = reference_bag(ab_tree("left"))
                self.assertEqual(len(rows), len(ref))

    def test_semi_join_at_most_once(self):
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(1, join_id="j1")
                rows, stats, _ = run_tree(ab_tree("semi"), mode=mode,
                                          control=ctrl)
                self.assertEqual(len(rows), 2)
                self.assertEqual(sorted(r.rid_of("a") for r in rows), [1, 2])
                ref = reference_bag(ab_tree("semi"))
                self.assertEqual(len(rows), len(ref))

    def test_null_keys_never_connect(self):
        rows, _, _ = run_tree(ab_tree("inner"), mode="force_memory")
        for r in rows:
            self.assertNotIn(("a", 4), r.rids)
            self.assertNotIn(("b", 13), r.rids)

    def test_three_way_twelve_pairs(self):
        # 6 AB pairs * 2 matching C rows = 12, every lineage distinct.
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(4, join_id="j1")
                    ctrl.request_switch_after_outputs(3, join_id="j2")
                rows, stats, _ = run_tree(three_way_tree(), mode=mode,
                                          control=ctrl)
                self.assertEqual(len(rows), 12)
                lineages = lineage_set(rows)
                self.assertEqual(len(lineages), 12)  # no duplicate pairs
                ref = reference_bag(three_way_tree())
                self.assertEqual(lineages,
                                 {tuple(sorted(l)) for l, _ in ref})

    def test_aggregate_hand_values(self):
        rows, stats, _ = run_tree(aggregate_tree())
        self.assertEqual(len(rows), 1)
        # header: agg.count, agg.sum_a_v (no group keys)
        values = rows[0].values
        self.assertEqual(values[0], 12)              # count
        self.assertEqual(values[1], 1800)            # hand-computed sum

    def test_post_join_filter_tree(self):
        tree = {
            "node": "filter",
            "expr": {"op": "gt", "left": "b.w", "right": {"literal": 1}},
            "input": ab_tree("inner"),
        }
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(1, join_id="j1")
                rows, _, _ = run_tree(tree, mode=mode, control=ctrl)
                ref = reference_bag(tree)
                self.assertEqual(len(rows), len(ref))  # w in {2,3} -> 4 rows
                self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
