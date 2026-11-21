"""Result-provenance session: proofs for every output and aggregate value.

A :class:`ProvenanceSession` is created per query run and handed to the
engine.  It receives the same row stream the operators produce and turns it
into durable proof events in a :class:`~join_exec.proof_store.ProofStore`.

Design properties
-----------------
* Proofs are a separate deliverable: the session never changes row values,
  filter predicates, join matching or aggregate results.  It only observes
  rows and attaches a stable ``proof_ref`` side channel.
* Proof references are content-addressed from the node id, the proof kind
  and the contributing ``(alias, row_id)`` source tuples, so they are
  independent of temp paths, iteration order and memory-vs-disk execution.
  The same logical source yields the same reference in every run.
* Nothing is accumulated per proof in Python: join/aggregate witnesses are
  streamed straight to proof shard files, so a single high-fanout count
  proof stays within the shared memory budget.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .errors import ProofError
from .proof_store import (
    ProofStore,
    aggregate_proof_ref,
    canonical_json,
    content_id,
    row_proof_ref,
)
from .types import Row

#: Witness role vocabulary (stable on disk; tests/oracles rely on it).
ROLE_SCAN = "scan"
ROLE_PASS = "filter_pass"
ROLE_PROJECT = "project"
ROLE_MATCH = "join_match"
ROLE_PAD = "left_pad"
ROLE_SEMI_LEFT = "semi_left"
ROLE_SEMI_MATCH = "semi_match"
ROLE_COUNT = "count_input"
ROLE_SUM = "sum_input"


def query_run_id(tree: dict) -> str:
    """Deterministic run id from the canonical query tree.

    Input *contents* are not part of the tree, so callers mixing in data
    digests do so through the tree-embedded scan file paths; the id is still
    stable across re-runs against the same inputs.
    """
    digest = hashlib.sha256(
        canonical_json(tree).encode("utf-8")).hexdigest()[:16]
    return f"run-{digest}"


class _JoinSink:
    """Per-join event sink attached to a :class:`HybridJoin` for one run."""

    def __init__(self, session: "ProvenanceSession", node_id: str) -> None:
        self._session = session
        self._node_id = node_id

    def note_match(self, join_id: str, row: Row) -> None:
        self._session.record_row_proof(
            self._node_id, "join", row, kind="match",
            witnesses=[{"role": ROLE_MATCH, "sources": [list(s)
                        for s in row.sources()]}])

    def note_pad(self, join_id: str, row: Row) -> None:
        self._session.record_row_proof(
            self._node_id, "join", row, kind="pad",
            witnesses=[{"role": ROLE_PAD, "sources": [list(s)
                        for s in row.sources()]}])

    def note_semi_match(self, join_id: str, lrow: Row, rrow: Row) -> None:
        # One proof per left row, accumulating every real right match.
        self._session.record_semi_match(self._node_id, lrow, rrow)


class ProvenanceSession:
    def __init__(self, store: ProofStore, tree: dict,
                 node_ids: Dict[Any, str]) -> None:
        self.store = store
        self.tree = tree
        self.node_ids = node_ids
        self.run_id = query_run_id(tree)
        self.store.begin_run(self.run_id)
        self._seq = 0
        self._semi_seen: Dict[Tuple[str, Tuple[Tuple[str, int], ...]],
                              List[dict]] = {}
        # aggregate node id -> {group_key: {"count": ref, "sum": ref}}
        self._agg_refs: Dict[str, Dict[tuple, dict]] = {}
        # aggregate node id -> group_key -> NULL sum inputs excluded
        self._agg_nulls: Dict[str, Dict[tuple, int]] = {}
        self._agg_descriptors: Dict[str, dict] = {}

    # ------------------------------------------------------------- join sink
    def join_sink(self, node_id: str) -> _JoinSink:
        return _JoinSink(self, node_id)

    # ---------------------------------------------------------- row proofs
    def record_row_proof(self, node_id: str, node_kind: str, row: Row,
                         kind: str, witnesses: List[dict]) -> str:
        sources = row.sources()
        pid = row_proof_ref(node_id, kind, sources)
        self.store.write_event(pid, {
            "t": "r", "node": node_id, "node_kind": node_kind,
            "kind": kind,
        })
        for w in witnesses:
            self.store.write_event(pid, {"t": "w", "w": w})
        row.proof_ref = pid
        return pid

    def record_scan(self, node_id: str, row: Row) -> None:
        self.record_row_proof(
            node_id, "scan", row, kind="scan",
            witnesses=[{"role": ROLE_SCAN,
                        "sources": [list(s) for s in row.sources()]}])

    def record_passthrough(self, node_id: str, node_kind: str,
                           row: Row, role: str) -> None:
        kind = "filter" if node_kind == "filter" else "project"
        self.record_row_proof(
            node_id, node_kind, row, kind=kind,
            witnesses=[{"role": role,
                        "sources": [list(s) for s in row.sources()]}])

    def record_semi_match(self, node_id: str, lrow: Row,
                          rrow: Row) -> None:
        left_sources = lrow.sources()
        key = (node_id, left_sources)
        pair_sources = tuple(sorted(left_sources + rrow.sources()))
        bucket = self._semi_seen.get(key)
        if bucket is None:
            bucket = [{
                "role": ROLE_SEMI_LEFT,
                "sources": [list(s) for s in left_sources],
            }]
            self._semi_seen[key] = bucket
            pid = row_proof_ref(node_id, "semi", left_sources)
            # Registration once; every later event extends the same proof.
            self.store.write_event(pid, {
                "t": "r", "node": node_id, "node_kind": "join",
                "kind": "semi",
            })
            self.store.write_event(
                pid, {"t": "w", "w": bucket[0]})
        else:
            pid = row_proof_ref(node_id, "semi", left_sources)
        witness = {"role": ROLE_SEMI_MATCH,
                   "sources": [list(s) for s in pair_sources]}
        # Duplicate matches are possible across the memory/disk boundary;
        # the commit sort drops identical event lines.
        self.store.write_event(pid, {"t": "w", "w": witness})
        lrow.proof_ref = pid

    # ------------------------------------------------------ aggregate proofs
    def prepare_aggregate(self, node_id: str, sum_ref: Optional[str],
                          key_refs: List[str]) -> None:
        self._agg_refs[node_id] = {}
        self._agg_nulls[node_id] = {}
        self._agg_descriptors[node_id] = {
            "node": node_id, "node_kind": "aggregate",
            "key_refs": key_refs, "sum_ref": sum_ref,
        }

    def make_agg_observer(self, node_id: str, sum_idx: Optional[int]
                          ) -> Callable[[Row, tuple], None]:
        def observe(row: Row, group_key: tuple) -> None:
            refs = self._agg_refs[node_id].get(group_key)
            if refs is None:
                count_ref = aggregate_proof_ref(node_id, "count", group_key)
                sum_ref_id = aggregate_proof_ref(node_id, "sum", group_key)
                refs = {"count": count_ref, "sum": sum_ref_id}
                self._agg_refs[node_id][group_key] = refs
            sources = [list(s) for s in row.sources()]
            self.store.write_event(refs["count"], {
                "t": "c",
                "c": {"role": ROLE_COUNT, "sources": sources},
            })
            if sum_idx is not None and row.values[sum_idx] is not None:
                self.store.write_event(refs["sum"], {
                    "t": "c",
                    "c": {"role": ROLE_SUM, "sources": sources,
                          "value": row.values[sum_idx]},
                })
            elif sum_idx is not None:
                nulls = self._agg_nulls[node_id]
                nulls[group_key] = nulls.get(group_key, 0) + 1
        return observe

    def annotate_aggregate_rows(self, node_id: str, rows: List[Row],
                                key_width: int, has_sum: bool
                                ) -> List[Row]:
        refs = self._agg_refs[node_id]
        for row in rows:
            group_key = tuple(row.values[:key_width])
            count_value = row.values[key_width]
            sum_value = row.values[key_width + 1] if has_sum else None
            group_refs = refs.get(group_key)
            if group_refs is None:
                # Empty-input scalar group: count=0 / sum=NULL proofs with
                # no contributors.
                count_ref = aggregate_proof_ref(node_id, "count", group_key)
                sum_ref_id = aggregate_proof_ref(node_id, "sum", group_key)
                group_refs = {"count": count_ref, "sum": sum_ref_id}
                refs[group_key] = group_refs
            # Exactly one registration per proof, written only now that the
            # final metric values are known; events are grouped by the
            # content-addressed pid at commit time regardless of write order.
            self.store.write_event(group_refs["count"], {
                "t": "r", "node": node_id, "node_kind": "aggregate",
                "kind": "count", "group_key": list(group_key),
                "value": count_value,
            })
            if has_sum:
                self.store.write_event(group_refs["sum"], {
                    "t": "r", "node": node_id, "node_kind": "aggregate",
                    "kind": "sum", "group_key": list(group_key),
                    "value": sum_value,
                    "null_contributors": self._agg_nulls[node_id]
                        .get(group_key, 0),
                })
            row.proof_ref = group_refs["count"]
        return rows

    # ----------------------------------------------------------- mappings
    def record_output(self, row: Row, output_node_id: str) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "node": output_node_id,
            "row_proof_ref": row.proof_ref,
            "value_hash": hashlib.sha256(
                canonical_json(list(row.values)).encode("utf-8")
            ).hexdigest()[:24],
        }
        self.store.write_output_mapping(record)

    # ------------------------------------------------------------- commit
    def commit(self) -> dict:
        digest = hashlib.sha256(
            canonical_json(self.tree).encode("utf-8")).hexdigest()
        return self.store.commit(self._agg_descriptors, digest)

    def abort(self) -> None:
        self.store.abort()
