"""MySQL connection and the small SQL dialect surface used by repositories."""

import hashlib
import os
import re
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
        if not os.getenv("GRADING_MYSQL_USER") or "GRADING_MYSQL_PASSWORD" not in os.environ:
            raise ValueError("Set GRADING_MYSQL_USER and GRADING_MYSQL_PASSWORD for local MySQL")
        database = os.getenv("GRADING_MYSQL_DATABASE", "grading_agent")
        self.lock_name = "grading:" + hashlib.sha256(database.encode()).hexdigest()[:40]
        self.connection = pymysql.connect(
            host="127.0.0.1",
            port=int(os.getenv("GRADING_MYSQL_PORT", "3306")),
            user=os.environ["GRADING_MYSQL_USER"],
            password=os.environ["GRADING_MYSQL_PASSWORD"],
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
    schema = Path(__file__).with_name("mysql_schema.sql").read_text(encoding="utf-8")
    for statement in schema.split(";"):
        if statement.strip():
            connection.execute(statement)
