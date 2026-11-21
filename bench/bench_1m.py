"""Million-row skew benchmark under a 128 MiB process constraint.

The data is generated at runtime (nothing large is committed):

* A: 1,000,000 rows.  90% share one hot key, the rest spread over cold keys;
* B: 100,000 rows with the same hot key plus matching cold keys;
* the hot pair alone would demand far more than 128 MiB as an in-memory
  cross product, forcing real spilling and block-by-block result streaming.

The process is wrapped with ``resource.setrlimit`` so the address space is
capped at 128 MiB; peak RSS, spill bytes and output counts are recorded to
``bench/results/``.  Output correctness is checked against an independent
streaming verifier (per-key expected multiplicities), not against the
product's own state.

Run:  python3 -m bench.bench_1m
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from typing import Dict, Tuple

from join_exec import (
    Budget,
    ControlPoint,
    QueryEngine,
    SchemaCatalog,
    SpillManager,
)
from join_exec.types import schema_from_json

HOT_KEY = 1
N_A = 1_000_000
N_B = 25_000
HOT_FRACTION_A = 0.90
# Only a few build rows carry the hot key: the *input* is severely skewed
# (90% of a million rows share one key) and forces spilling, while the hot
# matching set stays bounded enough to complete and verify.
HOT_B_MULTIPLICITY = 3
COLD_KEYS = 2_000
MEM_LIMIT = 128 * 1024 * 1024
LOGICAL_BUDGET = 16 * 1024 * 1024  # product budget inside the 128 MiB process


def generate(tmpdir: str) -> Tuple[str, str, Dict[int, int], Dict[int, int]]:
    a_path = os.path.join(tmpdir, "A_1m.jsonl")
    b_path = os.path.join(tmpdir, "B_25k.jsonl")
    a_hist: Dict[int, int] = {}
    b_hist: Dict[int, int] = {}
    hot_a = int(N_A * HOT_FRACTION_A)
    with open(a_path, "w", encoding="utf-8") as fh:
        for i in range(N_A):
            key = HOT_KEY if i < hot_a else (i % COLD_KEYS) + 100
            fh.write(json.dumps({"row_id": i, "k": key, "v": i & 0xFFFF})
                     + "\n")
            a_hist[key] = a_hist.get(key, 0) + 1
    with open(b_path, "w", encoding="utf-8") as fh:
        # Hot build rows first (the skew detector trips during the build).
        for i in range(HOT_B_MULTIPLICITY):
            fh.write(json.dumps({"row_id": i, "k": HOT_KEY,
                                 "w": i & 0xFFFF}) + "\n")
            b_hist[HOT_KEY] = b_hist.get(HOT_KEY, 0) + 1
        rid = HOT_B_MULTIPLICITY
        for i in range(N_B - HOT_B_MULTIPLICITY):
            key = (i % COLD_KEYS) + 100
            fh.write(json.dumps({"row_id": rid, "k": key,
                                 "w": rid & 0xFFFF}) + "\n")
            b_hist[key] = b_hist.get(key, 0) + 1
            rid += 1
    return a_path, b_path, a_hist, b_hist


def build_tree(a_path: str, b_path: str) -> dict:
    return {
        "node": "aggregate",
        "keys": ["a.k"],
        "sum": "a.v",
        "input": {
            "node": "join", "id": "j1", "type": "inner",
            "left": {"node": "scan", "relation": "A", "alias": "a",
                     "file": a_path},
            "right": {"node": "scan", "relation": "B", "alias": "b",
                      "file": b_path},
            "on": [["a.k", "b.k"]],
        },
    }


def expected_groups(a_hist: Dict[int, int], b_hist: Dict[int, int],
                    a_path: str) -> Dict[int, Tuple[int, int]]:
    """Independent expected (count, sum of a.v) per key.

    Sum is computed by re-streaming raw A rows, never touching product state.
    """
    per_key_sum: Dict[int, int] = {}
    with open(a_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            per_key_sum[obj["k"]] = per_key_sum.get(obj["k"], 0) + obj["v"]
    out: Dict[int, Tuple[int, int]] = {}
    for key, na in a_hist.items():
        nb = b_hist.get(key, 0)
        if nb:
            # each A row repeats nb times in the inner join
            out[key] = (na * nb, per_key_sum[key] * nb)
    return out


def peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: ru_maxrss is kilobytes
    return ru.ru_maxrss * 1024


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="adaptive",
                   choices=["adaptive", "force_disk", "force_memory"])
    p.add_argument("--skew-threshold", type=int, default=2)
    p.add_argument("--results", default=os.path.join(
        os.path.dirname(__file__), "results", "bench_1m.json"))
    args = p.parse_args()

    # Hard 128 MiB address-space cap for the whole process (Linux).
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(MEM_LIMIT, hard if hard != resource.RLIM_INFINITY else MEM_LIMIT)
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))

    tmp = tempfile.mkdtemp(prefix="bench1m-")
    catalog = SchemaCatalog([
        schema_from_json("A", [{"name": "k", "type": "int64"},
                               {"name": "v", "type": "int64"}]),
        schema_from_json("B", [{"name": "k", "type": "int64"},
                               {"name": "w", "type": "int64"}]),
    ])
    t0 = time.time()
    a_path, b_path, a_hist, b_hist = generate(tmp)
    gen_seconds = time.time() - t0

    tree = build_tree(a_path, b_path)
    budget = Budget(LOGICAL_BUDGET)
    control = ControlPoint()
    spill = SpillManager(budget, base_dir=tmp)
    engine = QueryEngine(
        catalog, budget, spill, control,
        mode=args.mode,
        skew_threshold=args.skew_threshold,
        join_overrides={"j1": {"expected_build_rows": N_B}})
    t1 = time.time()
    output_rows = 0
    output_by_key: Dict[int, Tuple[int, int]] = {}
    error = None
    try:
        plan = engine.build(tree)
        for row in engine.run(plan):
            # aggregate header: agg.key_0, agg.count, agg.sum_a_v
            key, cnt, total = row.values
            output_by_key[int(key)] = (int(cnt),
                                       None if total is None else int(total))
            output_rows += 1
    except MemoryError as exc:
        error = f"MemoryError under 128MiB cap: {exc}"
    except Exception as exc:  # noqa: BLE001 - recorded as benchmark result
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t1
    stats = engine.stats()
    expected = expected_groups(a_hist, b_hist, a_path)

    verified = (not error) and output_by_key == expected
    result = {
        "mode": args.mode,
        "process_cap_bytes": MEM_LIMIT,
        "logical_budget_bytes": LOGICAL_BUDGET,
        "rows_A": N_A,
        "rows_B": N_B,
        "generate_seconds": round(gen_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_mib": round(peak_rss_bytes() / 1024 / 1024, 2),
        "logical_peak_bytes": stats["budget"]["peak_bytes"],
        "spill": stats["spill"],
        "joins": stats["joins"],
        "output_groups": output_rows,
        "expected_groups": len(expected),
        "expected_total_join_rows": sum(c for c, _ in expected.values()),
        "verified_bag_aggregate": verified,
        "error": error,
    }
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    spill.cleanup()
    print(json.dumps({k: result[k] for k in (
        "mode", "peak_rss_mib", "output_groups", "expected_groups",
        "verified_bag_aggregate", "error")}, indent=2))
    print(f"full result: {args.results}")
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
