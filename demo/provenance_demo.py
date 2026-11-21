#!/usr/bin/env python3
"""Reproducible provenance demonstration and independent verification.

Run::

    python3 -I demo/provenance_demo.py [--workdir DIR]

The script generates its (small, fixed-shape) input relations at runtime,
runs one three-way query through filter -> project(collision) -> inner joins
-> count/sum with provenance enabled, and exercises every required scenario:

  1. three-way filter/project/join with projection collision,
  2. LEFT join pad (no fabricated right row_id),
  3. SEMI join with multiple real matches but one output,
  4. partial-NULL composite keys (never witnessed),
  5. all-NULL sum (excluded null contributors),
  6. empty-input count=0 / sum=NULL with no sources,
  7. a mid-output memory->disk switch whose proofs merge into one set,
  8. a single high-fanout count proof larger than the memory budget,
  9. proof write failure and consumer cancellation (no partial index).

Every business result is checked against hand expectations, and every proof
is checked by an independent oracle (tests/proof_oracle.py) that enumerates
witnesses row by row from the raw JSONL inputs without using any product
provenance, spill or switch machinery.  Memory- and disk-execution proof
sets are compared for identity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from join_exec import (  # noqa: E402
    Budget,
    ControlPoint,
    ProofError,
    ProofReader,
    ProofStore,
    ProvenanceSession,
    QueryEngine,
    RunCancelled,
    SchemaCatalog,
    SpillManager,
)
from join_exec.engine import _assign_node_ids  # noqa: E402
from join_exec.proof_store import aggregate_proof_ref  # noqa: E402
from join_exec.types import schema_from_json  # noqa: E402
from tests.proof_oracle import ProofOracle, expected_run_id  # noqa: E402

PASS = "ok"
FAIL = "FAILED"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def make_relation(tmp, name, cols, rows):
    path = os.path.join(tmp, f"{name}.jsonl")
    write_jsonl(path, rows)
    return path


def build_engine(tmp, tree, catalog, mem, mode="adaptive",
                 control=None, shards=8, proof_subdir="proofs",
                 write_fault=None):
    spill_dir = os.path.join(tmp, "spill")
    proof_dir = os.path.join(tmp, proof_subdir)
    os.makedirs(spill_dir, exist_ok=True)
    budget = Budget(mem)
    control = control or ControlPoint()
    spill = SpillManager(budget, base_dir=spill_dir)
    engine = QueryEngine(catalog, budget, spill, control, mode=mode)
    store = ProofStore(proof_dir, budget, shards=shards)
    if write_fault is not None:
        store.write_fault = write_fault
    session = ProvenanceSession(store, tree, _assign_node_ids(tree))
    engine.provenance = session
    return engine, spill, store, session, proof_dir


def check(label, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failures += 1


check.failures = 0


def doc_sources(doc):
    key = "contributions" if doc["kind"] in ("count", "sum") else "witnesses"
    out = set()
    for item in doc[key]:
        for s in item["sources"]:
            out.add(tuple(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    root = args.workdir or tempfile.mkdtemp(prefix="proof-demo-")
    os.makedirs(root, exist_ok=True)
    print(f"workdir: {root}")

    # --------------------------------------------------------------- data
    data = os.path.join(root, "data")
    os.makedirs(data, exist_ok=True)
    cols3 = [("T1", [{"name": "k", "type": "int64"},
                     {"name": "v", "type": "int64"}]),
             ("T2", [{"name": "k", "type": "int64"},
                     {"name": "w", "type": "int64"}]),
             ("T3", [{"name": "k", "type": "int64"},
                     {"name": "z", "type": "int64"}])]
    catalog = SchemaCatalog(
        [schema_from_json(rel, c) for rel, c in cols3])
    t1 = make_relation(data, "t1", cols3[0][1], [
        {"row_id": 1, "k": 7, "v": 100},
        {"row_id": 2, "k": 7, "v": 100},   # projection collision with rid 1
        {"row_id": 3, "k": 7, "v": None},  # filter UNKNOWN -> dropped
        {"row_id": 4, "k": 0, "v": 9},     # filter FALSE -> dropped
    ])
    t2 = make_relation(data, "t2", cols3[1][1], [
        {"row_id": 10, "k": 7, "w": 1},
        {"row_id": 11, "k": 7, "w": 2},
    ])
    t3 = make_relation(data, "t3", cols3[2][1], [
        {"row_id": 100, "k": 7, "z": 5},
        {"row_id": 101, "k": 7, "z": 6},
    ])

    tree = {
        "node": "aggregate", "keys": [], "sum": "t1.v",
        "input": {
            "node": "join", "id": "j2", "type": "inner",
            "left": {
                "node": "join", "id": "j1", "type": "inner",
                "left": {
                    "node": "project",
                    "outputs": [["t1", "t1.k"], ["t1", "t1.v"]],
                    "input": {
                        "node": "filter",
                        "expr": {"op": "gt", "left": "t1.v",
                                 "right": {"literal": 50}},
                        "input": {"node": "scan", "relation": "T1",
                                  "alias": "t1", "file": t1},
                    },
                },
                "right": {"node": "scan", "relation": "T2",
                          "alias": "t2", "file": t2},
                "on": [["t1.k", "t2.k"]],
            },
            "right": {"node": "scan", "relation": "T3",
                      "alias": "t3", "file": t3},
            "on": [["t1.k", "t3.k"]],
        },
    }

    # ------------------------------------------- 1+4: three-way + oracle
    print("scenario 1: three-way filter/project(collision)/join + count/sum")
    run_dir = os.path.join(root, "run1")
    engine, spill, store, session, proof_dir = build_engine(
        run_dir, tree, catalog, mem=256 * 1024, mode="force_memory")
    try:
        plan = engine.build(tree)
        rows = list(engine.run(plan))
    finally:
        spill.cleanup()
    check("business count=8 sum=800",
          [list(r.values) for r in rows] == [[8, 800]],
          str([list(r.values) for r in rows]))
    reader = ProofReader.open(proof_dir, session.run_id)
    check("run id is content-addressed",
          reader.stats()["run_id"] == expected_run_id(tree))
    oracle = ProofOracle(tree)
    docs = {d["proof_ref"]: d for d in reader.iter_proofs()}
    check("proof ref set equals independent oracle",
          set(docs) == set(oracle.by_ref),
          f"{len(docs)} vs {len(oracle.by_ref)}")
    sum_doc = reader.get_proof(
        aggregate_proof_ref(oracle.root_id, "sum", ()))
    check("sum proof has exactly 8 non-null contributions",
          len(sum_doc["contributions"]) == 8
          and all(c["value"] == 100 for c in sum_doc["contributions"]))
    check("filter rejects (t1,3),(t1,4) absent above filter",
          all(("t1", 3) not in doc_sources(d) and
              ("t1", 4) not in doc_sources(d)
              for d in docs.values() if d["node"] not in
              {n for n, ex in oracle.expected.items()
               if any(e.kind == "scan" for e in ex)}))
    proj_node = next(n for n, ex in oracle.expected.items()
                     if any(e.kind == "project" for e in ex))
    check("projection collision keeps two distinct proofs",
          len({e.proof_ref for e in oracle.expected[proj_node]}) == 2)

    # --------------------- 2: memory vs disk proof-set identity + switch
    print("scenario 7: mid-output switch merges into one complete proof set")
    ref_sets = {}
    for mode, ctrl in (
            ("force_memory", ControlPoint()),
            ("force_disk", ControlPoint()),
            ("adaptive", _switch_control("j1", 2, "j2", 3))):
        d = os.path.join(root, f"run-{mode}")
        engine, spill, store, sess, pdir = build_engine(
            d, tree, catalog, mem=64 * 1024, mode=mode, control=ctrl)
        try:
            plan = engine.build(tree)
            rws = list(engine.run(plan))
        finally:
            spill.cleanup()
        rd = ProofReader.open(pdir, sess.run_id)
        ref_sets[mode] = {p["p"]: (p["w"], p["s"])
                          for shard in range(rd.shards)
                          for p in _index_entries(rd, shard)}
        check(f"{mode} business identical",
              [list(r.values) for r in rws] == [[8, 800]])
        if mode == "adaptive":
            ev = engine.stats()["joins"][0]["switch_events"]
            check("adaptive really switched after partial outputs",
                  len(ev) == 1 and ev[0]["outputs_before_switch"] == 2)
    check("memory/disk/adaptive index identity",
          ref_sets["force_memory"] == ref_sets["force_disk"]
          == ref_sets["adaptive"])

    # ------------------------------------------------- 2: LEFT pad + SEMI
    print("scenario 2/3: LEFT pad and SEMI multi-match")
    lr_cols = [
        ("L", [{"name": "k", "type": "int64"},
               {"name": "v", "type": "int64"}]),
        ("R", [{"name": "k", "type": "int64"},
               {"name": "w", "type": "int64"}])]
    lcat = SchemaCatalog([schema_from_json(r, c) for r, c in lr_cols])
    lp = make_relation(os.path.join(root, "data"), "L", lr_cols[0][1], [
        {"row_id": 1, "k": 5, "v": 1},
        {"row_id": 2, "k": 6, "v": 2},
        {"row_id": 3, "k": None, "v": 3}])
    rp = make_relation(os.path.join(root, "data"), "R", lr_cols[1][1], [
        {"row_id": 10, "k": 5, "w": 10},
        {"row_id": 11, "k": 5, "w": 11},
        {"row_id": 12, "k": 5, "w": 12},
        {"row_id": 13, "k": None, "w": 13}])

    def lr_tree(jtype):
        return {"node": "join", "id": "j1", "type": jtype,
                "left": {"node": "scan", "relation": "L", "alias": "l",
                         "file": lp},
                "right": {"node": "scan", "relation": "R", "alias": "r",
                          "file": rp},
                "on": [["l.k", "r.k"]]}

    d = os.path.join(root, "run-left")
    engine, spill, _s, sess, pdir = build_engine(
        d, lr_tree("left"), lcat, mem=64 * 1024, mode="force_disk")
    try:
        plan = engine.build(lr_tree("left"))
        lrows = list(engine.run(plan))
    finally:
        spill.cleanup()
    rd = ProofReader.open(pdir, sess.run_id)
    pads = [x for x in rd.iter_proofs() if x["kind"] == "pad"]
    check("LEFT: 3 matches + 2 pads", len(lrows) == 5 and len(pads) == 2)
    check("pad proofs name only left sources, no right row_id",
          all(all(a == "l" for a, _ in doc_sources(p)) for p in pads)
          and all(p["no_right_match"] for p in pads))

    d = os.path.join(root, "run-semi")
    engine, spill, _s, sess, pdir = build_engine(
        d, lr_tree("semi"), lcat, mem=64 * 1024)
    try:
        plan = engine.build(lr_tree("semi"))
        srows = list(engine.run(plan))
    finally:
        spill.cleanup()
    rd = ProofReader.open(pdir, sess.run_id)
    semi = [x for x in rd.iter_proofs() if x["kind"] == "semi"]
    check("SEMI emits once but proof lists 3 real matches",
          len(srows) == 1 and len(semi) == 1
          and len(semi[0]["witnesses"]) == 4
          and {tuple(sorted(tuple(s) for s in w["sources"]))
               for w in semi[0]["witnesses"]
               if w["role"] == "semi_match"} == {
              (("l", 1), ("r", 10)),
              (("l", 1), ("r", 11)),
              (("l", 1), ("r", 12))})

    # --------------------------------------- 3: partial NULL composite key
    print("scenario 4: partial-NULL composite keys never match")
    cn_cols = [
        ("A", [{"name": "k1", "type": "int64"},
               {"name": "k2", "type": "text"},
               {"name": "v", "type": "int64"}]),
        ("B", [{"name": "k1", "type": "int64"},
               {"name": "k2", "type": "text"}])]
    ccat = SchemaCatalog([schema_from_json(r, c) for r, c in cn_cols])
    ap_ = make_relation(os.path.join(root, "data"), "A2", cn_cols[0][1], [
        {"row_id": 1, "k1": 1, "k2": "x", "v": 7},
        {"row_id": 2, "k1": 1, "k2": None, "v": 8},
        {"row_id": 3, "k1": None, "k2": "x", "v": 9}])
    bp = make_relation(os.path.join(root, "data"), "B2", cn_cols[1][1], [
        {"row_id": 10, "k1": 1, "k2": "x"},
        {"row_id": 11, "k1": 1, "k2": None}])
    ctree = {"node": "join", "id": "j1", "type": "inner",
             "left": {"node": "scan", "relation": "A", "alias": "a",
                      "file": ap_},
             "right": {"node": "scan", "relation": "B", "alias": "b",
                       "file": bp},
             "on": [["a.k1", "b.k1"], ["a.k2", "b.k2"]]}
    engine, spill, _s, sess, pdir = build_engine(
        os.path.join(root, "run-null"), ctree, ccat, mem=64 * 1024)
    try:
        plan = engine.build(ctree)
        nrows = list(engine.run(plan))
    finally:
        spill.cleanup()
    rd = ProofReader.open(pdir, sess.run_id)
    match_docs = [x for x in rd.iter_proofs() if x["kind"] == "match"]
    check("exactly one (a1,b10) match",
          len(nrows) == 1 and len(match_docs) == 1
          and doc_sources(match_docs[0]) == {("a", 1), ("b", 10)})

    # ------------------------------- 5/6: all-NULL sum and empty aggregates
    print("scenario 5/6: all-NULL sum and empty count/sum")
    e_cols = [
        ("A", [{"name": "k", "type": "int64"},
               {"name": "v", "type": "int64"}]),
        ("B", [{"name": "k", "type": "int64"}])]
    ecat = SchemaCatalog([schema_from_json(r, c) for r, c in e_cols])
    an = make_relation(os.path.join(root, "data"), "An", e_cols[0][1], [
        {"row_id": 1, "k": 1, "v": None},
        {"row_id": 2, "k": 1, "v": None}])
    bn = make_relation(os.path.join(root, "data"), "Bn", e_cols[1][1], [
        {"row_id": 10, "k": 1}])
    atree = {"node": "aggregate", "keys": [], "sum": "a.v",
             "input": {"node": "join", "id": "j1", "type": "inner",
                        "left": {"node": "scan", "relation": "A",
                                 "alias": "a", "file": an},
                        "right": {"node": "scan", "relation": "B",
                                  "alias": "b", "file": bn},
                        "on": [["a.k", "b.k"]]}}
    engine, spill, _s, sess, pdir = build_engine(
        os.path.join(root, "run-nullsum"), atree, ecat, mem=64 * 1024)
    try:
        plan = engine.build(atree)
        erows = list(engine.run(plan))
    finally:
        spill.cleanup()
    rd = ProofReader.open(pdir, sess.run_id)
    o = ProofOracle(atree)
    sd = rd.get_proof(aggregate_proof_ref(o.root_id, "sum", ()))
    cd = rd.get_proof(aggregate_proof_ref(o.root_id, "count", ()))
    check("all-NULL sum -> NULL, 0 sum contributors, 2 excluded",
          [list(r.values) for r in erows] == [[2, None]]
          and sd["contributions"] == []
          and sd["excluded_null_contributors"] == 2
          and len(cd["contributions"]) == 2)

    empty = make_relation(os.path.join(root, "data"), "empty",
                          e_cols[0][1], [])
    etree = {"node": "aggregate", "keys": [], "sum": "a.v",
             "input": {"node": "join", "id": "j1", "type": "inner",
                        "left": {"node": "scan", "relation": "A",
                                 "alias": "a", "file": empty},
                        "right": {"node": "scan", "relation": "B",
                                  "alias": "b", "file": bn},
                        "on": [["a.k", "b.k"]]}}
    engine, spill, _s, sess, pdir = build_engine(
        os.path.join(root, "run-empty"), etree, ecat, mem=64 * 1024)
    try:
        plan = engine.build(etree)
        erows2 = list(engine.run(plan))
    finally:
        spill.cleanup()
    rd = ProofReader.open(pdir, sess.run_id)
    o = ProofOracle(etree)
    cd0 = rd.get_proof(aggregate_proof_ref(o.root_id, "count", ()))
    sd0 = rd.get_proof(aggregate_proof_ref(o.root_id, "sum", ()))
    check("empty count=0 and sum=NULL with no sources",
          [list(r.values) for r in erows2] == [[0, None]]
          and cd0["contributions"] == [] and sd0["contributions"] == []
          and doc_sources(cd0) == set())

    # --------------------------------------------- 8: high-fanout over budget
    print("scenario 8: single high-fanout count proof over memory budget")
    n_left, n_right = 400, 50
    hf_a = make_relation(os.path.join(root, "data"), "hfA", e_cols[0][1], [
        {"row_id": i, "k": 1, "v": i % 7} for i in range(n_left)])
    hf_b = make_relation(os.path.join(root, "data"), "hfB", e_cols[1][1], [
        {"row_id": 10000 + i, "k": 1} for i in range(n_right)])
    htree = {"node": "aggregate", "keys": [], "sum": "a.v",
             "input": {"node": "join", "id": "j1", "type": "inner",
                        "left": {"node": "scan", "relation": "A",
                                 "alias": "a", "file": hf_a},
                        "right": {"node": "scan", "relation": "B",
                                  "alias": "b", "file": hf_b},
                        "on": [["a.k", "b.k"]]}}
    d = os.path.join(root, "run-fanout")
    engine, spill, _s, sess, pdir = build_engine(
        d, htree, ecat, mem=12 * 1024, shards=4)
    try:
        plan = engine.build(htree)
        hrows = list(engine.run(plan))
        peak = engine.budget.peak_bytes
    finally:
        spill.cleanup()
    rd = ProofReader.open(pdir, sess.run_id)
    o = ProofOracle(htree)
    hc = rd.get_proof(aggregate_proof_ref(o.root_id, "count", ()))
    combos = {tuple(sorted(tuple(s) for s in c["sources"]))
              for c in hc["contributions"]}
    check("count proof streams 20000 exact combos under 12KiB budget",
          hrows[0].values[0] == n_left * n_right
          and len(hc["contributions"]) == n_left * n_right
          and len(combos) == n_left * n_right
          and combos == set(o.by_ref[hc["proof_ref"]].contributors)
          and peak <= 12 * 1024,
          f"peak={peak}")
    # Streaming frame reader keeps constant memory for the same proof.
    n_frames = sum(1 for _ in rd.iter_proof_frames(hc["proof_ref"]))
    check("streaming frame API exposes all frames",
          n_frames == n_left * n_right + 2)  # header + trailer

    # --------------------------------------------- 9: failure and cancellation
    print("scenario 9: write failure / cancellation leave no partial index")
    f_tree = {"node": "join", "id": "j1", "type": "inner",
              "left": {"node": "scan", "relation": "A", "alias": "a",
                       "file": hf_a},
              "right": {"node": "scan", "relation": "B", "alias": "b",
                        "file": hf_b},
              "on": [["a.k", "b.k"]]}
    d = os.path.join(root, "run-fault")

    def disk_full(_path):
        raise OSError("simulated disk full")

    engine, spill, store, sess, pdir = build_engine(
        d, f_tree, ecat, mem=16 * 1024, mode="force_disk",
        write_fault=disk_full)
    try:
        plan = engine.build(f_tree)
        try:
            list(engine.run(plan))
            raised = None
        except ProofError as exc:
            raised = exc
    finally:
        spill.cleanup()
    check("proof write failure raises ProofError", raised is not None)
    check("failed run leaves no directory/manifest",
          not os.path.exists(os.path.join(pdir, sess.run_id)))

    d = os.path.join(root, "run-cancel")
    ctrl = ControlPoint()
    engine, spill, _s, sess, pdir = build_engine(
        d, f_tree, ecat, mem=16 * 1024, mode="force_disk", control=ctrl)
    gen = iter(engine.run(engine.build(f_tree)))
    next(gen)
    ctrl.cancel("demo cancel")
    try:
        list(gen)
        cancelled = None
    except RunCancelled as exc:
        cancelled = exc
    spill.cleanup()
    check("cancellation raises RunCancelled and removes proofs",
          cancelled is not None
          and not os.path.exists(os.path.join(pdir, sess.run_id)))

    # ------------------------------------------------------------- paging
    print("api: paged index/mappings and direct proof fetch")
    entries, token, all_entries = [], "x", []
    token = None
    while True:
        page, token = rd.iter_index(token=token, page_size=37)
        all_entries.extend(page)
        if token is None:
            break
    check("index paging covers every proof",
          {e["p"] for e in all_entries}
          == {d["proof_ref"] for d in rd.iter_proofs()})

    print("\nstats (last run):", json.dumps(rd.stats(), indent=2))
    print(f"\n{check.failures} failure(s)")
    if args.workdir is None:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if check.failures else 0


def _switch_control(j1, n1, j2, n2):
    c = ControlPoint()
    c.request_switch_after_outputs(n1, j1)
    c.request_switch_after_outputs(n2, j2)
    return c


def _index_entries(reader, shard):
    path = os.path.join(reader.run_dir, "index",
                        f"shard-{shard:02x}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


if __name__ == "__main__":
    raise SystemExit(main())
