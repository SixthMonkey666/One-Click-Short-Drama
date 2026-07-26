from __future__ import annotations

import sqlite3


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def add_missing_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = column_names(conn, table)
    for name, definition in columns.items():
        if name in existing:
            continue
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}')
