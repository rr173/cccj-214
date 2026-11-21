# Runtime-Adaptive Relational Join Executor

A from-scratch Python executor whose join operator can switch **from in-memory
hash processing to disk-based processing in the middle of a running query**,
without losing or duplicating any result.  It is an observable execution engine
(`HybridJoin` owns its mode and switch state), not an explain-text-only
optimizer and not a wrapper around an external database.

## What is delivered

| Requirement | Where |
| --- | --- |
| int64 / text / null fields, unique `row_id`, bag multiplicity | `join_exec/scan.py`, `join_exec/types.py` |
| scan, 3-valued filter (`= != < <= > >=`, and/or/not), project, inner/left/semi join, per-key count/sum | `join_exec/ops.py`, `join_exec/join.py` |
| multi-column keys; NULL in any component never matches (NULL ≠ NULL) | `make_join_key` in `join_exec/types.py` |
| LEFT pad distinct from real right NULL rows; SEMI ≤ 1 per left row | `join_exec/join.py` |
| sum with no non-null input → NULL; checked int64 overflow | `join_exec/ops.py`, `checked_add_int64` |
| 3–6 leaf join trees; INNER reorderable region; LEFT/SEMI barriers never crossed | `join_exec/engine.py` |
| shared memory budget, row counts, hot-key histogram, spill-byte accounting | `join_exec/budget.py`, `JoinStats`, `engine.stats()` |
| real runtime memory→disk switch (build pressure, probe hot group, two control points) | `join_exec/join.py` |
| already-emitted results not replayed; remainder not lost | residual suppression records |
| independent nested-loop oracle sharing no product machinery | `join_exec/reference.py` |
| million-row skew run under a hard 128 MiB process cap | `bench/bench_1m.py` |

## The memory→disk transition

One operator, `join_exec.join.HybridJoin`, runs both phases.  It starts in hash
mode and can switch at any of these points, all of which call the **same**
`_switch_*` implementation:

1. **Build-phase budget overflow** — the next build row cannot be reserved.
2. **Build/probe hot-key threshold** — a key group reaches `skew_threshold`.
3. **Control point after N input batches** —
   `ControlPoint.request_switch_after_input_batches`.
4. **Control point after N emitted outputs** —
   `ControlPoint.request_switch_after_outputs` (may fire *inside* an expanding
   match group).

`force_memory` and `force_disk` exist for comparison; `adaptive` is the real
behaviour.  A switch is never chosen as two static plans before the first row.

### Why no duplicates and no gaps

When a switch happens after some pairs were already emitted, the in-flight
left row is written to disk as a **residual record** carrying the right
`row_id`s already delivered:

```json
{"kind": "residual", "values": [...], "rids": ["a", 1],
 "suppress": [["b", 10], ["b", 11]]}
```

The disk phase skips exactly those suppressed right rows, so the cross product
is completed rather than replayed.  Suppression follows the left row into its
single partition; matched state is therefore partition-local.  A LEFT row is
padded only after the whole matching right group is consumed, and SEMI emits a
left row at most once.  No business-value deduplication and no
"emit-then-cancel" trick is used.

### Disk phase

* Build and left rows are hash-partitioned to JSONL spill files (NULL keys go
  to separate files and can never match).
* A partition whose build side fits the remaining budget uses a partition hash
  join, streaming its left side from disk (never whole-partition materialised).
* An oversized partition (e.g. one dominant hot key) uses an **external
  sort-merge join**: both sides are sorted with bounded chunks + `heapq.merge`
  (`SpillManager.external_sort`) and merged group by group with full key
  alignment.  Neither side is ever held whole, so a hot group larger than the
  budget streams its result in batches.

### Two subtle correctness rules the engine enforces

* **Scans are re-openable.** Disk joins re-run whole input subtrees and re-read
  spill sources, so duplicate-`row_id` detection is local to one scan pass; a
  second pass over the same file never raises a false duplicate error.
* **SEMI preserves bag multiplicity after an upstream join.** Once a join has
  run, one base `row_id` occurs in many distinct input tuples (different joined
  rids).  SEMI's "at most once per left row" therefore dedups on the **full
  lineage tuple** (`Row.rids`), never on a single base rid, so duplicate
  business values with different `row_id`s are all retained.

## Memory and accounting semantics

* `Budget` is a single logical byte budget shared by every live operator
  (hash entries, buffered rows, aggregate groups, spill handles).  Data-row
  reservations are hard; spill OS handles are accounted best-effort so an
  extremely small budget cannot block spilling itself.
* Spill files live on disk; only small fixed buffers are charged.  Every byte
  written/read and every file is counted in `SpillStats`, and the temp
  directory is removed on every exit path (success, error, cancellation).
* The benchmark additionally records real process RSS via `resource` and
  enforces the 128 MiB cap with `RLIMIT_AS`, so logical accounting is
  cross-checked against the OS.

## Query tree

```json
{
  "node": "join", "id": "j1", "type": "inner",
  "left":  {"node": "scan", "relation": "A", "alias": "a", "file": "..."},
  "right": {"node": "scan", "relation": "B", "alias": "b", "file": "..."},
  "on": [["a.k1", "b.k1"], ["a.k2", "b.k2"]]
}
```

Nodes: `scan`, `filter` (3VL), `project`, `join` (`inner|left|semi`),
`aggregate` (`keys`, optional `sum`).  Field refs are always qualified
`alias.column`.  Ints are validated to int64 on read; count and sum raise
`OverflowError` rather than wrapping.

## Run it

```bash
# full test suite (41 tests, including the independent oracle and
# 300-trial randomized property checks)
python3 -m unittest discover -s tests -v

# headline scenario: switch after the 2nd of 6 pairs, total stays 6
python3 -m join_exec.cli run \
  --schema tests/fixtures/schema.json --query tests/fixtures/q_ab_inner.json \
  --memory-bytes 1048576 --switch-after-outputs j1:2 \
  --output out.jsonl --stats stats.json

# bag-equality vs the nested-loop oracle for all three modes
python3 -m join_exec.cli verify --schema tests/fixtures/schema.json \
  --query tests/fixtures/q_ab_inner.json --memory-bytes 1048576 \
  --mode force_disk --stats stats.json

# million-row skew benchmark under a 128 MiB process cap (data generated at
# runtime; only the small JSON result is kept)
python3 -m bench.bench_1m --mode adaptive

# everything in one shot
bash bench/run_all.sh
```

## Walkthrough A (hand-computed, all asserted in `tests/test_walkthrough_a.py`)

With A rows `rids 1,2` on key 10 (+ an unmatched row and a NULL-key row) and B
rows `rids 10,11,12` on key 10 (+ a NULL-key row):

* INNER → exactly the 6 pairs `(1|2) × (10|11|12)`; switching after pair 2
  still yields exactly 6.
* LEFT → those 6 plus one pad for the unmatched A row and one pad for the
  NULL-key A row (8 rows, two pads, each padded at most once).
* SEMI → only the two matching A rows.
* Joining a third relation C (two key-10 rows) gives `6 × 2 = 12` triples;
  the terminal aggregate yields `count = 12` and
  `sum(a.v) = 6·100 + 6·200 = 1800`.

Every switch boundary is compared against both always-memory and always-disk
runs and against the independent oracle, asserting full bag equality
(lineage sets, not just counts).

## Failure semantics

Bad JSONL / out-of-range ints → `BadRowError`; spill IO failure → `SpillError`;
consumer close / `ControlPoint.cancel()` → deterministic termination with
files and sub-iterators closed (`RunCancelled`).  No cross-process recovery is
attempted in this version, per the stated scope.

## Layout

```
join_exec/   product executor (engine, join operator, spill, budget, ops, oracle)
tests/       unittest suite + small JSONL/JSON fixtures (no large data)
bench/       runtime million-row generator, 128 MiB-capped runner, results JSON
```
