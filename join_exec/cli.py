"""Command line entry point.

Example
-------
python -m join_exec.cli run \
    --schema fixtures/schema.json --query fixtures/q.json \
    --memory-bytes 1048576 --skew-threshold 50 \
    --switch-after-outputs 'j1:2' --output rows.jsonl

Switch control syntax: ``join_id:N`` (or ``:N`` to target every join).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .budget import Budget
from .control import ControlPoint
from .engine import QueryEngine
from .errors import JoinExecError
from .reference import ReferenceInterpreter
from .schema_catalog import SchemaCatalog
from .spill import SpillManager


def _parse_switch(spec: Optional[str], method):
    if not spec:
        return
    target, _, n = spec.partition(":")
    n = int(n)
    method(n, join_id=target or None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="join_exec")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--schema", required=True)
        sp.add_argument("--query", required=True)
        sp.add_argument("--memory-bytes", type=int, required=True)
        sp.add_argument("--mode", choices=["adaptive", "force_memory",
                                            "force_disk"], default="adaptive")
        sp.add_argument("--skew-threshold", type=int, default=None)
        sp.add_argument("--temp-dir", default=None)
        sp.add_argument("--switch-after-outputs", default=None)
        sp.add_argument("--switch-after-batches", default=None)
        sp.add_argument("--stats", default=None)
        sp.add_argument("--explain", action="store_true")
        sp.add_argument("--limit", type=int, default=None,
                        help="consumer closes after N rows (tests early close)")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--output", default="-")

    e = sub.add_parser("explain")
    common(e)

    v = sub.add_parser("verify")
    common(v)
    v.add_argument("--output", default="-")

    args = p.parse_args(argv)
    with open(args.schema, "r", encoding="utf-8") as fh:
        catalog = SchemaCatalog.from_json(json.load(fh))
    with open(args.query, "r", encoding="utf-8") as fh:
        tree = json.load(fh)

    budget = Budget(args.memory_bytes)
    control = ControlPoint()
    _parse_switch(args.switch_after_outputs,
                  control.request_switch_after_outputs)
    _parse_switch(args.switch_after_batches,
                  control.request_switch_after_input_batches)
    spill = SpillManager(budget, base_dir=args.temp_dir)
    engine = QueryEngine(catalog, budget, spill, control,
                         mode=args.mode,
                         skew_threshold=args.skew_threshold)
    try:
        plan = engine.build(tree)
        if args.cmd == "explain" or args.explain:
            json.dump(engine.explain(tree), sys.stdout, indent=2,
                      ensure_ascii=False)
            sys.stdout.write("\n")
            return 0
        out_fh = sys.stdout if args.output == "-" else \
            open(args.output, "w", encoding="utf-8")
        count = 0
        try:
            for row in engine.run(plan):
                rec = {"values": list(row.values),
                       "rids": [list(row.rids[i:i + 2])
                                for i in range(0, len(row.rids), 2)]}
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                if args.limit is not None and count >= args.limit:
                    # Consumer goes away early: close downstream explicitly.
                    break
        finally:
            if out_fh is not sys.stdout:
                out_fh.close()
        stats = engine.stats()
        stats["rows_delivered"] = count
        if args.cmd == "verify":
            expected = ReferenceInterpreter(tree).run()
            stats["reference_rows"] = len(expected)
            stats["bag_equal_count"] = len(expected) == count
        if args.stats:
            with open(args.stats, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2, ensure_ascii=False)
        else:
            json.dump(stats, sys.stderr, indent=2, ensure_ascii=False)
            sys.stderr.write("\n")
        return 0
    except JoinExecError as exc:
        sys.stderr.write(f"run ended: {type(exc).__name__}: {exc}\n")
        return 2
    finally:
        spill.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
