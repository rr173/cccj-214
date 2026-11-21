#!/usr/bin/env bash
# Reproducible entry points for the runtime-adaptive join executor.
# All commands are self-contained and write only small fixtures/results.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
QUERY=tests/fixtures/q_ab_inner.json

echo "== 1. Full unit/oracle test suite =="
$PY -m unittest discover -s tests -v

echo
echo "== 2. Main walkthrough A: switch after the 2nd output pair =="
$PY -m join_exec.cli run \
  --schema tests/fixtures/schema.json \
  --query "$QUERY" \
  --memory-bytes 1048576 \
  --switch-after-outputs j1:2 \
  --output /tmp/joinexec_walkthrough.jsonl \
  --stats  /tmp/joinexec_walkthrough.stats.json \
  --temp-dir /tmp/joinexec-spill
echo "expect 6 distinct lineage pairs; got $(wc -l < /tmp/joinexec_walkthrough.jsonl)"

echo
echo "== 3. Bag-equality across always-memory / adaptive / always-disk =="
for MODE in force_memory adaptive force_disk; do
  $PY -m join_exec.cli verify \
    --schema tests/fixtures/schema.json \
    --query "$QUERY" \
    --memory-bytes 1048576 --mode "$MODE" \
    --output "/tmp/joinexec_${MODE}.jsonl" \
    --stats  "/tmp/joinexec_${MODE}.stats.json" >/dev/null
  $PY -c "import json;d=json.load(open('/tmp/joinexec_${MODE}.stats.json'));print('$MODE'.ljust(13),'bag_equal=',d['bag_equal_count'],'rows=',d['rows_delivered'])"
done

echo
echo "== 4. Million-row skew run under a 128 MiB process cap =="
$PY -m bench.bench_1m --mode adaptive
echo "full metrics: bench/results/bench_1m.json"
