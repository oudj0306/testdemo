"""Copy an offline SQLite snapshot into a fresh local MySQL database."""

import argparse
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from grading_agent.storage.mysql import MySQLConnection, create_schema


def migrate(source: Path) -> dict[str, int]:
    source_uri = source.resolve(strict=True).as_uri() + "?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as snapshot,
        closing(MySQLConnection()) as target,
    ):
        snapshot.execute("BEGIN")
        schema = Path(__file__).with_name("mysql_schema.sql").read_text(encoding="utf-8")
        tables = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", schema)
        source_tables = {
            row[0] for row in snapshot.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        unknown = source_tables.difference(tables, {"sqlite_sequence"})
        if unknown:
            raise ValueError(f"Unrecognized SQLite tables: {', '.join(sorted(unknown))}")
        create_schema(target)
        target.execute("BEGIN IMMEDIATE")
        counts = {}
        try:
            for table in tables:
                if target.execute(f"SELECT 1 FROM `{table}` LIMIT 1").fetchone():
                    raise ValueError(f"MySQL destination must be empty: {table}")
                if table not in source_tables or table in {"stable_grades", "grade_leases"}:
                    continue
                cursor = snapshot.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
                columns = [column[0] for column in cursor.description]
                target_columns = {
                    row["Field"] for row in target.execute(f"SHOW COLUMNS FROM `{table}`")
                }
                if not set(columns) <= target_columns:
                    raise ValueError(f"Unrecognized SQLite columns in {table}")
                names = ", ".join(f"`{name}`" for name in columns)
                placeholders = ", ".join("?" for _ in columns)
                count = 0
                # Queue payloads can be large. Copy one upload per insert.
                size = 1 if table == "batch_jobs" else 100
                while rows := cursor.fetchmany(size):
                    target.executemany(
                        f"INSERT INTO `{table}` ({names}) VALUES ({placeholders})", rows
                    )
                    count += len(rows)
                if target.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()[0] != count:
                    raise RuntimeError(f"Row count mismatch: {table}")
                counts[table] = count
            target.commit()
        except Exception:
            target.rollback()
            raise
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    for table, count in migrate(args.source).items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
