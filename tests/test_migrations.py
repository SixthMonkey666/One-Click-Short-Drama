from __future__ import annotations

import sqlite3
import unittest

from app.evaluation.migrations import add_missing_columns, column_names
from tests import _test_env  # noqa: F401


class TestEvaluationMigrations(unittest.TestCase):
    def test_add_missing_columns_is_idempotent(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE evaluation_items (id TEXT PRIMARY KEY)")

        columns = {
            "auto_total_score": "REAL",
            "auto_judge_provider": "TEXT",
        }
        add_missing_columns(conn, "evaluation_items", columns)
        add_missing_columns(conn, "evaluation_items", columns)

        names = column_names(conn, "evaluation_items")
        self.assertIn("auto_total_score", names)
        self.assertIn("auto_judge_provider", names)

    def test_real_migration_errors_are_not_swallowed(self):
        conn = sqlite3.connect(":memory:")
        with self.assertRaises(sqlite3.OperationalError):
            add_missing_columns(conn, "missing_table", {"new_column": "TEXT"})


if __name__ == "__main__":
    unittest.main()
