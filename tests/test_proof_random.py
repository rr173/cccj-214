"""Randomised provenance property tests vs the independent proof oracle.

Each trial builds random relations (NULLs, duplicate business values, skew),
a random join tree with optional filter/project/aggregate, and compares the
durable proof documents from memory, disk and adaptive executions against an
oracle that enumerates witnesses directly from the raw JSONL inputs without
touching product provenance, spill or switch machinery.
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
    ProofReader,
    ProofStore,
    ProvenanceSession,
    QueryEngine,
    SchemaCatalog,
    SpillManager,
)
from join_exec.engine import _assign_node_ids
from join_exec.types import schema_from_json
from tests.proof_oracle import ProofOracle


def _doc_tuple(doc):
    if doc["kind"] in ("count", "sum"):
        contrib = tuple(sorted(
            tuple(sorted(tuple(s) for s in c["sources"]))
            for c in doc["contributions"]))
        return (doc["kind"], doc.get("metric_value"), contrib,
                doc.get("excluded_null_contributors"))
    if doc["kind"] == "semi":
        # The product also carries the left-row witness; compare the
        # full set of source combinations including it.
        return (doc["kind"], tuple(sorted(
            tuple(sorted(tuple(s) for s in w["sources"]))
            for w in doc["witnesses"])))
    sources = tuple(sorted(
        tuple(s) for w in doc["witnesses"] for s in w["sources"]))
    return (doc["kind"], sources)


class RandomProvenancePropertyTest(unittest.TestCase):
    def test_random_trees_proofs_match_oracle_all_modes(self):
        master = random.Random(777011)
        tmp = tempfile.mkdtemp()
        for trial in range(12):
            rng = random.Random(master.randrange(1 << 30))
            n_rel = rng.randint(2, 3)
            rels = [chr(ord("R") + i) for i in range(n_rel)]
            specs, paths = [], {}
            for i, rel in enumerate(rels):
                alias = f"r{i}"
                specs.append((rel, [
                    {"name": "k", "type": "int64"},
                    {"name": "v", "type": "int64"},
                ]))
                n_rows = rng.randint(0, 25)
                path = os.path.join(tmp, f"{rel}_{trial}.jsonl")
                with open(path, "w", encoding="utf-8") as fh:
                    for rid in range(n_rows):
                        key = None if rng.random() < 0.15 \
                            else rng.choice([1, 2, 3])
                        v = None if rng.random() < 0.25 \
                            else rng.randint(0, 9)
                        fh.write(json.dumps(
                            {"row_id": rid, "k": key, "v": v}) + "\n")
                paths[alias] = (rel, path)

            def scan(alias):
                rel, path = paths[alias]
                return {"node": "scan", "relation": rel, "alias": alias,
                        "file": path}

            node = scan("r0")
            for i in range(1, n_rel):
                node = {
                    "node": "join", "id": f"j{i}",
                    "type": rng.choice(["inner", "inner", "left", "semi"]),
                    "left": node, "right": scan(f"r{i}"),
                    "on": [["r0.k", f"r{i}.k"]],
                }
            if rng.random() < 0.5:
                node = {"node": "filter",
                        "expr": {"op": rng.choice(["gt", "eq"]),
                                 "left": "r0.v",
                                 "right": {"literal": rng.randint(2, 6)}},
                        "input": node}
            if rng.random() < 0.5:
                node = {"node": "project",
                        "outputs": [["r0", "r0.k"], ["r0", "r0.v"]],
                        "input": node}
            if rng.random() < 0.6:
                node = {"node": "aggregate",
                        "keys": ["r0.k"] if rng.random() < 0.5 else [],
                        "sum": "r0.v" if rng.random() < 0.7 else None,
                        "input": node}
            tree = node
            catalog = SchemaCatalog(
                [schema_from_json(r, c) for r, c in specs])
            oracle = ProofOracle(tree)

            per_mode = {}
            for mode in ("force_memory", "force_disk", "adaptive"):
                work = tempfile.mkdtemp(dir=tmp)
                budget = Budget(rng.choice([16 * 1024, 128 * 1024]))
                control = ControlPoint()
                if mode == "adaptive":
                    jid = rng.choice([f"j{i}" for i in range(1, n_rel)])
                    control.request_switch_after_outputs(
                        rng.randint(1, 4), jid)
                spill = SpillManager(budget, base_dir=os.path.join(
                    work, "spill"))
                engine = QueryEngine(catalog, budget, spill, control,
                                     mode=mode)
                store = ProofStore(os.path.join(work, "proofs"),
                                   budget, shards=4)
                session = ProvenanceSession(
                    store, tree, _assign_node_ids(tree))
                engine.provenance = session
                try:
                    plan = engine.build(tree)
                    rows = list(engine.run(plan))
                    reader = ProofReader.open(
                        os.path.join(work, "proofs"), session.run_id)
                    got = {d["proof_ref"]: _doc_tuple(d)
                           for d in reader.iter_proofs()}
                finally:
                    spill.cleanup()
                per_mode[mode] = (got, [list(r.values) for r in rows])

            expected = {ref: _doc_tuple_expected(exp)
                        for ref, exp in oracle.by_ref.items()}
            for mode, (got, values) in per_mode.items():
                self.assertEqual(
                    set(got), set(expected),
                    msg=f"trial {trial} {mode}: proof ref set")
                for ref, tup in expected.items():
                    self.assertEqual(
                        got.get(ref), tup,
                        msg=f"trial {trial} {mode}: proof content {ref}")
            # Business outputs identical across modes as multisets (result
            # order is not promised).
            def bag(rows):
                return sorted(tuple(v) for v in rows)

            self.assertEqual(bag(per_mode["force_memory"][1]),
                             bag(per_mode["force_disk"][1]))
            self.assertEqual(bag(per_mode["force_memory"][1]),
                             bag(per_mode["adaptive"][1]))


def _doc_tuple_expected(exp):
    if exp.kind in ("count", "sum"):
        contrib = tuple(sorted(exp.contributors))
        return (exp.kind, exp.value, contrib,
                exp.null_contributors if exp.kind == "sum" else None)
    if exp.kind == "semi":
        # Oracle match tuples plus the left-only source combination,
        # matching the product's semi_left + semi_match witnesses.
        all_combos = set(exp.match_sources)
        all_combos.add(exp.sources)
        return (exp.kind, tuple(sorted(all_combos)))
    return (exp.kind, exp.sources)


if __name__ == "__main__":
    unittest.main()
