"""External sort utility and budget/spill accounting tests."""

from __future__ import annotations

import tempfile
import unittest

from join_exec import Budget, SpillManager
from join_exec.types import Row


class ExternalSortTest(unittest.TestCase):
    def test_external_sort_order_and_content(self):
        tmp = tempfile.mkdtemp()
        budget = Budget(64 * 1024)
        spill = SpillManager(budget, base_dir=tmp)
        try:
            # Deliberately unsorted, including repeated business values with
            # distinct rids (bag multiplicity must survive the sort).
            rows = [Row([i % 5, 1000 - i], ("a", i)) for i in range(2000, 0, -1)]
            sf = spill.external_sort(
                rows, key_fn=lambda r: tuple(r.values),
                owner="t", mem_bytes=8 * 1024)
            back = [r for r, _ in sf.read()]
            keys = [tuple(r.values) for r in back]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(len(back), 2000)
            # every rid retained exactly once (no dedup, no loss)
            rids = sorted(r.rid_of("a") for r in back)
            self.assertEqual(rids, list(range(1, 2001)))
            self.assertGreater(spill.stats.bytes_written, 0)
        finally:
            spill.cleanup()
        self.assertEqual(budget.used_bytes, 0)

    def test_spill_cleanup_removes_temp_dir(self):
        tmp = tempfile.mkdtemp()
        budget = Budget(1024 * 1024)
        spill = SpillManager(budget, base_dir=tmp)
        sf = spill.new_file(tag="x")
        sf.write_row(Row([1], ("a", 1)))
        sf.finalize_write()
        import os
        self.assertTrue(os.path.isdir(spill.temp_dir))
        spill.cleanup()
        self.assertFalse(os.path.exists(spill.temp_dir))
        self.assertEqual(budget.used_bytes, 0)


if __name__ == "__main__":
    unittest.main()
