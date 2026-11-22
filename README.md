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
| durable, content-addressed result provenance (row + count/sum proofs) | `join_exec/provenance.py`, `join_exec/proof_store.py` |
| durable, incrementally refreshed SQLite summary view with local-access evidence | `join_exec/materialized.py`, `join_exec/incremental_oracle.py` |
| maximal pre-delivery quarantine snapshot cleanup, key-local SQLite propagation with locality evidence | `join_exec/snapshot_cleanup.py`, `tests/snapshot_oracle.py` |
| independent proof oracle enumerating witnesses from raw inputs only | `tests/proof_oracle.py` |
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

## Result provenance (auditable source proofs)

A query can optionally deliver, **separately from its business results**, a
durable proof for every output row and every terminal count/sum value.
Proofs never change row values, filter predicates, join matching or
aggregate results: they are produced by an observer side channel
(`Row.proof_ref`, `HybridJoin.proof_sink`) attached only for the run.

### What a proof says

* **scan / filter / project** — one proof per passing tuple naming every
  contributing base `(alias, row_id)`.  Filter rows that evaluate to FALSE
  or UNKNOWN have no proof above the filter (their scan record exists as an
  input only).  Projection collisions — different input combinations
  producing identical business values — keep distinct proofs.
* **inner / left join match** — a proof naming the full contributing input
  combination.  **LEFT pads** carry only the left sources plus
  `"no_right_match": true`; a right `row_id` is never fabricated.
* **semi join** — one output per left row, but its proof contains the left
  witness plus **every real matching right row**; suppressed/duplicate
  matches are not counted.
* **count** — one contribution per input row actually counted.
* **sum** — contributions only from non-NULL inputs, with
  `excluded_null_contributors` recording how many NULL inputs were skipped;
  all-NULL input yields `sum = NULL` with zero contributions, and empty
  input yields `count = 0` / `sum = NULL` proofs with no sources.

### Stable references and durable layout

Proof references are SHA-256 content ids over the node id, proof kind and
contributing sources (`rp_…` for rows, `ap_…` for aggregates).  They do not
depend on temp directory names, iteration order or process addresses, so the
same logical source yields the same reference in memory and disk
executions and across re-runs.  Re-running against identical inputs recreates
the same `run_id` (a digest of the canonical query tree).

```
<proof-dir>/<run-id>/
  manifest.json            # present ONLY after a successful commit
  data/shard-XX.jsonl      # framed proof documents (h/w/c/t JSONL frames)
  index/shard-XX.jsonl     # proof_ref -> offset/length/witness/distinct counts
  index/shard-XX.dir.jsonl # sparse pid/offset directory for paged lookup
  mapping/shard-XX.jsonl   # output row seq -> proof_ref + value hash
```

`ProofReader` opens a committed run after the executor and its spill
directory are gone: `get_proof(ref)` assembles one document,
`iter_proof_frames(ref)` streams a high-fanout proof with constant reader
memory, and `iter_index`/`iter_output_mappings` page via opaque tokens.
Join output order is not promised; output mapping rows carry `seq`.

### Memory, fan-out and the mid-switch merge

Proof events stream into fixed JSONL staging shards — nothing is
accumulated per proof in Python, so a single count/sum proof can hold more
combinations than the whole budget (a 20 000-combination proof runs under a
12 KiB budget).  Commit uses a bounded external sort (chunk floor plus
round-wise merges of at most 8 runs) and streams each proof document while
tracking exact per-proof and global distinct source counts through a
budget-bounded set that spills to an external-sort file on overflow.  The
store reports actual bytes written/read, file count, proof/witness counts,
distinct-source counts and the output-row ↔ proof mapping.

When a join switches from memory to disk after emitting partial matches,
the residual-suppression machinery guarantees the proof stream sees each
real combination exactly once: already delivered pairs are not replayed and
suppressed duplicates never become sources.  Memory, forced-disk and
adaptive (switched) executions produce identical proof reference and content
sets — asserted by the randomized property tests.

### Failure semantics

A proof write failure, disk exhaustion, corrupt input, consumer early close
or cancellation raises (`ProofError`, `BadRowError`, `RunCancelled`) and the
run directory is removed entirely.  The manifest is published last via
atomic rename, so no partial proof index can ever look complete, and neither
business rows nor proofs are reported as successfully delivered.

### API and CLI

```python
from join_exec import ProofStore, ProvenanceSession, ProofReader
from join_exec.engine import _assign_node_ids

store = ProofStore(proof_dir, budget, shards=8)
session = ProvenanceSession(store, tree, _assign_node_ids(tree))
engine.provenance = session
for row in engine.run(engine.build(tree)):
    print(row.proof_ref)          # stable reference on every output row
reader = ProofReader.open(proof_dir, session.run_id)
reader.get_proof(row.proof_ref)   # assembled proof document
```

```bash
python3 -m join_exec.cli run --schema s.json --query q.json \
    --memory-bytes 1048576 --proof-dir ./proofs --stats stats.json
python3 -m join_exec.cli proof stats    --proof-dir ./proofs --run-id run-xxxx
python3 -m join_exec.cli proof get      --proof-dir ./proofs --run-id run-xxxx --proof-ref rp_...
python3 -m join_exec.cli proof stream   --proof-dir ./proofs --run-id run-xxxx --proof-ref ap_...
python3 -m join_exec.cli proof index    --proof-dir ./proofs --run-id run-xxxx --page-size 128
python3 -m join_exec.cli proof mappings --proof-dir ./proofs --run-id run-xxxx
```

The end-to-end scenario catalogue (three-way collision query, LEFT pad,
SEMI multi-match, partial-NULL composite key, all-NULL/empty aggregates,
mid-switch merge, over-budget single high-fanout proof, write failure and
cancellation — each verified against the independent oracle) is:

```bash
python3 -I demo/provenance_demo.py --workdir /tmp/proof-demo
```

## Durable, incrementally refreshed summary view

Offline fact tables change a few rows at a time, but the maintained aggregate
used to require re-running the whole query. `join_exec.materialized` adds a
**continuously refreshed summary view** persisted in a local SQLite state
database. It is initialized once from the full JSONL inputs; afterwards
callers submit only the row corrections of the batch and readers get the
refreshed full summary from the state database — no original files, no
full-query re-execution. The one-shot query and provenance features above
are untouched.

### Maintained query shape (the only shape accepted)

```
AGGREGATE keys=[<cat-alias>.<group-column>]  sum=<fact-alias>.<int64-column>
  SEMI JOIN  on  fact.<elig-key> = elig.<key>
    LEFT JOIN  on  fact.<cat-key> = cat.<key>
      scan fact        scan category
    scan eligibility
```

All semantics match the one-shot engine: duplicate category rows each
contribute (including equal group values), an unmatched/NULL-key LEFT fact
contributes exactly one NULL-padded row, SEMI only keeps/drops the fact
without multiplying, NULL join keys never match, NULL category values are a
legal group, all-NULL-amount groups publish `sum = NULL`, empty groups
disappear, and sums are exact int64-checked so an out-of-range published
value rejects the batch regardless of correction ordering.

### API

```python
from join_exec import MaterializedSummary

view = MaterializedSummary.initialize("state.db", tree, catalog)
# later, possibly in another process:
view = MaterializedSummary.open("state.db")
evidence = view.apply_corrections([
    {"relation": "F", "row_id": 7, "op": "upsert",
     "row": {"row_id": 7, "fk": 10, "ek": 100, "amt": 5}},
    {"relation": "C", "row_id": 9, "op": "delete"},
])
view.version()        # one entry point for the current version
view.read_summary()   # one entry point for the whole committed summary
view.close()
```

Every valid non-empty batch publishes the next version even when the visible
summary is identical. A batch runs in one SQLite transaction: bad rows,
deleting a missing `row_id`, int64 overflow or an injected write/publish
fault all roll back, so source rows, maintenance state, summary and version
advance together or not at all; reopening resumes from the last committed
version.

### Refresh locality and its testable evidence

A refresh reads only the directly corrected rows, the facts associated with
the changed category/eligibility keys, and the category/eligibility match
state of those facts — via indexed point lookups; unrelated facts are never
scanned and only affected groups are updated. The evidence is inspectable,
not just a row count:

* the durable `access_trace` table (surfaced through `access_trace()`,
  `last_access_trace()`, `trace_summary()` and the `apply_corrections`
  return value) records every localized access with the table, `row_id` or
  join key and the publishing version; failed batches roll their traces back;
* `explain_local_access()` returns `EXPLAIN QUERY PLAN` output for each
  lookup statement, asserting index `SEARCH` plans rather than table scans;
* `apply_corrections` returns the exact set of affected fact `row_id`s and
  the groups that changed.

The independent ground truth is `join_exec.incremental_oracle`, a row-by-row
enumerator over plain row maps that shares no incremental machinery; the
tests also cross-check against the original `ReferenceInterpreter` running
the full query.

```bash
# 25 materialized-view tests (oracle + original-engine cross-checks,
# locality evidence, order independence, overflow, fault/reopen recovery)
python3 -m unittest tests.test_materialized_summary -v

# end-to-end reproducible scenario catalogue (small generated fixtures only)
python3 -I demo/materialized_summary_demo.py --workdir /tmp/ms-demo
```

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
# full test suite (the independent business oracle, the independent proof
# oracle, randomized property checks, all provenance failure/lifecycle
# scenarios and the snapshot-cleanup oracle cross-checks)
python3 -m unittest discover -s tests -v

# snapshot cleanup only (27 tests incl. naive-fixpoint oracle cross-checks,
# hot-key fan-out, long elimination chain and failure-atomicity evidence)
python3 -m unittest discover -s tests -p test_snapshot_cleanup.py -v

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
proof-file IO failure / disk exhaustion → `ProofError` (the partial proof
directory is removed and no manifest is published); consumer close /
`ControlPoint.cancel()` → deterministic termination with files and
sub-iterators closed (`RunCancelled`).  No cross-process recovery is
attempted in this version, per the stated scope; committed proof
directories are independent of spill temp directories and remain readable
after the executor exits and all temp resources are cleaned.

## Pre-delivery quarantine snapshot cleanup

An upstream relation snapshot may name a few confirmed-untrusted rows that
must be isolated before delivery.  Simply deleting those rows is not enough:
other rows can lose the target they required, and their removal can ripple
through further tables.  `join_exec.snapshot_cleanup` adds one pre-delivery
entry point that produces, in a fresh directory, the **maximal** row subset
in which no retained row has a dangling required reference.

### Contract

A *required reference* rule names a source and a target relation and pairs
their columns by position (single-column or composite keys; paired columns
must share a type).  A retained source row must find, for every rule that
applies to it, at least one retained target row with an equal key.  Equality
follows the engine's join semantics — any NULL key component can never match
(NULL ≠ NULL).  Target keys are not unique: removing one target only
invalidates referrers when no other retained target carries the same key, and
equal rows are never merged.  Rules may be self-referential or form cycles.

The result is the greatest fixpoint of that contract, not just any
orphan-free set:

* two accounts that require each other, with no other missing support, are
  both retained;
* if one additionally requires a quarantined subject with no substitute, both
  are removed;
* a row with no outgoing required reference is never removed just because
  other rows point at it.

The answer does not depend on input row order, file load order or rule
order.  Each relation still ships one JSONL file on its original schema
(empty tables ship a file too); retained `row_id`s and field values are
unchanged.

### SQLite staging, locality and failure semantics

Nothing is materialised whole in Python: per relation, one row table (the
original columns plus a status) carries one partial index per outgoing rule;
per rule, one compact **support table** holds `(distinct target key,
alive_count)` — one row per distinct key, never a referrer×candidate pair, so
a many-to-many fan-out of M referrers and N same-key candidates costs O(M+N)
staging rows rather than O(M·N).  A durable work queue drives key-local
propagation: a dead row decrements exactly the support counts of the keys it
carried, and only when a count reaches zero are the referrers on that key
removed through index point searches.  Unrelated keys and rows are never
rescanned — there is no per-round full-relation sweep.  A durable,
count-aggregated `access_trace` plus `explain_local_lookups()` make the
locality inspectable (index `SEARCH` on row tables; only the compact support
tables may be scanned).

All staging lives in a temporary directory next to the (not-yet-existing)
delivery directory; the delivery directory itself is created last by a single
atomic rename, so any failure removes the working directory and leaves no
observable partial delivery, and the read-only input files are never touched.

### API and CLI

```python
from join_exec import SchemaCatalog, clean_snapshot, rules_from_json

catalog = SchemaCatalog.from_json(schema_spec)
rules = rules_from_json(rule_specs, catalog)
summary = clean_snapshot(
    catalog,
    input_paths={"S": "src/S.jsonl", "F": "src/F.jsonl", "L": "src/L.jsonl"},
    rules=rules,
    quarantine=[("S", 1), ("S", 2)],   # may be empty
    delivery_dir="delivery/")
summary["output_paths"]   # one JSONL path per relation, handed to executors
```

```bash
python3 -m join_exec.cli clean-snapshot \
  --schema schema.json --inputs inputs.json --rules rules.json \
  --quarantine quarantine.json --delivery-dir ./delivery --stats stats.json
```

Bad rows, duplicate `row_id`s, unknown quarantine ids and invalid rules raise
distinct errors (`BadRowError`, `QuarantineError`, `RuleError`,
`DeliveryError`).  The independent ground truth is
`tests/snapshot_oracle.py`, a deliberately naive row-by-row fixpoint that
shares no cleanup machinery; tests compare exact per-relation retained
`row_id` sets, not just counts.  The reproducible scenario catalogue (two
same-key deliveries, complete vs broken cycles, composite NULL keys, multiple
rules per row, hot-key fan-out, a long elimination chain and injected write
faults) is:

```bash
python3 -I demo/snapshot_cleanup_demo.py --workdir /tmp/snap-demo
```

## Layout

```
join_exec/   product executor (engine, join operator, spill, budget, ops,
             oracle, durable provenance session + proof store, durable
             incremental materialized summary view, pre-delivery quarantine
             snapshot cleanup)
tests/       unittest suite + small JSONL/JSON fixtures (no large data);
             proof_oracle.py is an independent witness enumerator;
             incremental_oracle.py independently enumerates maintained rows;
             snapshot_oracle.py independently enumerates the retained set
demo/        reproducible end-to-end provenance, materialized-view and
             snapshot-cleanup demos
bench/       runtime million-row generator, 128 MiB-capped runner, results JSON
```
