"""MySQL connection and the small SQL dialect surface used by repositories."""

import hashlib
import os
import re
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import pymysql
from pymysql.constants import CLIENT
from pymysql.cursors import DictCursor


def mysql_enabled() -> bool:
    backend = os.getenv("GRADING_DB_BACKEND", "mysql").lower()
    if backend not in {"sqlite", "mysql"}:
        raise ValueError("GRADING_DB_BACKEND must be mysql or sqlite")
    return backend == "mysql"


class Row(dict):
    def __getitem__(self, key):
        return tuple(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


@dataclass
class Result:
    rows: list[Row]
    rowcount: int

    def __iter__(self):
        return iter(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


def mysql_statement(sql: str) -> str:
    # Statements are application constants. All user values remain parameters.
    ignored_insert = re.match(r"INSERT OR IGNORE INTO (\w+)\((\w+)", sql)
    if ignored_insert:
        table, column = ignored_insert.groups()
        sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO", 1)
        sql += f" ON DUPLICATE KEY UPDATE {column}={table}.{column}"
    sql = re.sub(r"CAST\(([\w.]+) AS INTEGER\)", r"CAST(\1 AS SIGNED)", sql)
    sql = sql.replace("INSERT OR REPLACE INTO", "REPLACE INTO")
    sql = re.sub(r"ON CONFLICT\([^)]+\) DO UPDATE SET", "ON DUPLICATE KEY UPDATE", sql)
    return re.sub(r"excluded\.(\w+)", r"VALUES(\1)", sql)


class MySQLConnection:
    def __init__(self):
        from grading_agent.config import get_settings

        config = ConfigParser(interpolation=None)
        config.read(get_settings().data_dir / "mysql.local.ini", encoding="utf-8-sig")
        local = config["mysql"] if config.has_section("mysql") else {}
        user = os.getenv("GRADING_MYSQL_USER", local.get("user"))
        password = os.getenv("GRADING_MYSQL_PASSWORD", local.get("password"))
        if not user or password is None:
            raise ValueError(
                "Configure data/mysql.local.ini or GRADING_MYSQL_USER and GRADING_MYSQL_PASSWORD"
            )
        database = os.getenv("GRADING_MYSQL_DATABASE", local.get("database", "grading_agent"))
        self.lock_name = "grading:" + hashlib.sha256(database.encode()).hexdigest()[:40]
        self.connection = pymysql.connect(
            host="127.0.0.1",
            port=int(os.getenv("GRADING_MYSQL_PORT", local.get("port", "3306"))),
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            binary_prefix=True,
            autocommit=False,
            cursorclass=DictCursor,
            client_flag=CLIENT.FOUND_ROWS,
            connect_timeout=5,
        )

    def execute(self, sql, parameters=None):
        if sql.strip() == "BEGIN IMMEDIATE":
            # Preserve serialized claim/check/write sequences across connections.
            result = self.execute("SELECT GET_LOCK(?, 30) AS acquired", (self.lock_name,))
            if result.fetchone()["acquired"] != 1:
                raise pymysql.OperationalError("Timed out acquiring grading write lock")
            self.connection.begin()
            return result
        sql = mysql_statement(sql)
        with self.connection.cursor() as cursor:
            if parameters is not None:
                sql = sql.replace("%", "%%").replace("?", "%s")
            cursor.execute(sql, parameters)
            rows = [Row(row) for row in cursor.fetchall()] if cursor.description else []
            return Result(rows, cursor.rowcount)

    def executemany(self, sql, parameters):
        with self.connection.cursor() as cursor:
            cursor.executemany(
                mysql_statement(sql).replace("%", "%%").replace("?", "%s"), parameters
            )

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


def create_schema(connection: MySQLConnection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    schema = Path(__file__).with_name("mysql_schema.sql").read_text(encoding="utf-8")
    for statement in schema.split(";"):
        if statement.strip():
            connection.execute(statement)
    # Upgrade existing installations without replacing their cache tables.
    for table, index, column in (
        ("stable_grades", "idx_stable_grades_created_at", "created_at"),
        ("grade_leases", "idx_grade_leases_expires_at", "expires_at"),
    ):
        if not connection.execute(
            "SELECT 1 FROM information_schema.statistics "
            "WHERE table_schema=DATABASE() AND table_name=? AND index_name=?",
            (table, index),
        ).fetchone():
            connection.execute(f"CREATE INDEX {index} ON {table} ({column})")
