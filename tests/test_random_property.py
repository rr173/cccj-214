"""Randomised property tests vs the independent nested-loop oracle.

Each trial builds random relations (including NULLs, duplicate business values
and skew), a random 2-4 leaf tree of inner/left/semi joins with optional
filter/aggregate, and compares every execution mode against the reference
interpreter as full lineage bags.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest

from join_exec import (
    Budget,
    ControlPoint,
    MemoryLimitExceeded,
    QueryEngine,
    ReferenceInterpreter,
    SchemaCatalog,
    SpillManager,
)
from join_exec.errors import JoinExecError
from join_exec.types import schema_from_json
from tests.util import lineage_set


class RandomPropertyTest(unittest.TestCase):
    def test_many_random_trees_match_oracle(self):
        master = random.Random(20261121)
        tmp = tempfile.mkdtemp()
        trials = 40
        for trial in range(trials):
            # Independent stream per trial: engine-side random choices must
            # never perturb the data/tree generation RNG (and vice versa).
            rng = random.Random(master.randrange(1 << 30))
            run_rng = random.Random(master.randrange(1 << 30))
            n_rel = rng.randint(2, 4)
            rels = [chr(ord("R") + i) for i in range(n_rel)]
            specs = []
            paths = {}
            for i, rel in enumerate(rels):
                alias = f"r{i}"
                specs.append((rel, [
                    {"name": "k", "type": "int64"},
                    {"name": "v", "type": "int64"},
                ]))
                n_rows = rng.randint(0, 60)
                skew_key = rng.choice([1, 2, 3])
                path = os.path.join(tmp, f"{rel}_{trial}.jsonl")
                with open(path, "w", encoding="utf-8") as fh:
                    for rid in range(n_rows):
                        if rng.random() < 0.15:
                            key = None
                        elif rng.random() < 0.5:
                            key = skew_key          # duplicated business value
                        else:
                            key = rng.randint(1, 8)
                        v = None if rng.random() < 0.2 else rng.randint(0, 50)
                        fh.write(json.dumps(
                            {"row_id": rid, "k": key, "v": v}) + "\n")
                paths[alias] = (rel, path)

            def scan(alias):
                rel, path = paths[alias]
                return {"node": "scan", "relation": rel, "alias": alias,
                        "file": path}

            # left-deep random join tree
            node = scan("r0")
            for i in range(1, n_rel):
                jtype = rng.choice(["inner", "inner", "left", "semi"])
                node = {
                    "node": "join", "id": f"j{i}", "type": jtype,
                    "left": node, "right": scan(f"r{i}"),
                    "on": [["r0.k", f"r{i}.k"]],
                }
            if rng.random() < 0.5:
                node = {"node": "filter",
                        "expr": {"op": rng.choice(["gt", "lt", "eq"]),
                                 "left": "r0.v",
                                 "right": {"literal": rng.randint(5, 40)}},
                        "input": node}
            if rng.random() < 0.5:
                node = {"node": "aggregate", "keys": ["r0.k"],
                        "sum": "r0.v", "input": node}

            tree = node
            schema_hint = {rel: [c["name"] for c in cols]
                           for rel, cols in specs}
            ref = ReferenceInterpreter(tree, schemas=schema_hint).run()
            ref_bag = {tuple(sorted(l)) for l, _ in ref}
            cat = SchemaCatalog(
                [schema_from_json(r, c) for r, c in specs])

            for mode in ("force_memory", "force_disk", "adaptive"):
                # Each mode gets an independent derived stream so a
                # force_memory early-exit cannot perturb the others.
                mrng = random.Random(run_rng.randrange(1 << 30))
                # Small enough to force join spilling, large enough that a
                # terminal aggregate's group table can still be reserved.
                budget = Budget(mrng.choice([16 * 1024, 64 * 1024,
                                             256 * 1024]))
                control = ControlPoint()
                spill = SpillManager(budget, base_dir=tmp)
                overrides = {}
                if mode == "adaptive":
                    # force a mid-run switch on at least one join
                    jid = mrng.choice([f"j{i}" for i in range(1, n_rel)])
                    if mrng.random() < 0.5:
                        control.request_switch_after_outputs(
                            mrng.randint(1, 5), jid)
                    else:
                        control.request_switch_after_input_batches(1, jid)
                    overrides[jid] = {"batch_size": mrng.randint(1, 8)}
                engine = QueryEngine(
                    cat, budget, spill, control, mode=mode,
                    skew_threshold=mrng.choice([None, 2, 8, 20]),
                    join_overrides=overrides)
                try:
                    try:
                        rows = list(engine.run(engine.build(tree)))
                    except (MemoryLimitExceeded, JoinExecError):
                        # force_memory may legitimately fail when the build
                        # side cannot fit; adaptive/force_disk must never fail.
                        if mode == "force_memory":
                            continue
                        raise
                    # Aggregates have empty lineage; compare group maps.
                    if tree["node"] == "aggregate":
                        n_keys = len(tree.get("keys", []))
                        has_sum = tree.get("sum") is not None
                        got_map = {
                            tuple(r.values[:n_keys]):
                            (r.values[n_keys],
                             r.values[n_keys + 1] if has_sum else None)
                            for r in rows
                        }
                        self.assertEqual(
                            len(got_map), len(ref),
                            msg=f"trial {trial} {mode}: agg group count")
                        for _, v in ref:
                            key = tuple(v[f"agg.key_{i}"]
                                        for i in range(n_keys))
                            cnt = v["agg.count"]
                            sum_keys = [k for k in v
                                        if k.startswith("agg.sum_")]
                            total = v[sum_keys[0]] if sum_keys else None
                            self.assertEqual(
                                got_map.get(key), (cnt, total),
                                msg=f"trial {trial} {mode}: key {key}")
                    else:
                        self.assertEqual(
                            len(rows), len(ref),
                            msg=f"trial {trial} {mode}: count")
                        self.assertEqual(
                            lineage_set(rows), ref_bag,
                            msg=f"trial {trial} {mode}: lineage bag")
                finally:
                    spill.cleanup()


if __name__ == "__main__":
    unittest.main()
