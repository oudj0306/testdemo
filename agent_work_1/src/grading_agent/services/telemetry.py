"""Request-level usage and wall times. Never store prompts or credentials here."""

import json
import logging
import sqlite3
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from threading import Lock
from time import perf_counter
from uuid import uuid4

from grading_agent.db import utc_now
from grading_agent.services.model_io.llm_client import response_usage

_scope = ContextVar("performance_scope", default={})
LOG = logging.getLogger(__name__)


@contextmanager
def scope(settings, **identifiers):
    parent = _scope.get()
    root = not parent
    counter = parent.get("_database") or {"elapsed_ms": 0.0, "count": 0, "lock": Lock()}
    token = _scope.set(
        {**parent, "path": settings.database_path, "_database": counter, **identifiers}
    )
    try:
        yield
    finally:
        if root and counter["count"]:
            event("database", counter["elapsed_ms"], details={"operations": counter["count"]})
        _scope.reset(token)


def event(stage, elapsed_ms=0, success=True, usage=None, details=None, event_id=None):
    context = _scope.get()
    if not context.get("path"):
        return event_id
    usage = usage or {}
    event_id = event_id or uuid4().hex
    try:
        with sqlite3.connect(context["path"], timeout=30) as db:
            db.execute(
                """INSERT INTO performance_events
                (id,batch_id,assessment_id,submission_id,stage,elapsed_ms,success,
                 input_tokens,output_tokens,cached_tokens,details,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    context.get("batch_id"),
                    context.get("assessment_id"),
                    context.get("submission_id"),
                    stage,
                    elapsed_ms,
                    int(success),
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cached_tokens", 0),
                    json.dumps(details or {}),
                    utc_now(),
                ),
            )
    except sqlite3.Error:
        LOG.warning("Performance event could not be recorded", exc_info=True)
    return event_id


def database_elapsed(elapsed_ms):
    counter = _scope.get().get("_database")
    if counter:
        # Never open a second writer from a nested database transaction.
        with counter["lock"]:
            counter["elapsed_ms"] += elapsed_ms
            counter["count"] += 1


@contextmanager
def timed(stage, **details):
    start = perf_counter()
    success = False
    try:
        yield
        success = True
    finally:
        event(stage, (perf_counter() - start) * 1000, success, details=details)


def request_json(request, settings, stage, timeout, **details):
    """A response that fails later validation still consumes this request's usage."""
    start, request_id = perf_counter(), uuid4().hex
    data, success = {}, False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            data = {}
            raise ValueError("模型响应必须是 JSON 对象")
        success = True
        return data, request_id
    finally:
        try:
            usage = response_usage(data)
        except (ValueError, TypeError, AttributeError):
            usage = {}
            details["usage_unavailable"] = True
        with scope(settings):
            event(
                stage + "_request",
                (perf_counter() - start) * 1000,
                success,
                usage,
                {"retry": bool(_scope.get().get("retry")), **details},
                request_id,
            )


def summary(batch_id):
    from grading_agent.db import decode_json, transaction

    with transaction() as db:
        batch = db.execute("SELECT * FROM grading_batches WHERE id=?", (batch_id,)).fetchone()
        if not batch:
            raise LookupError("批次不存在")
        rows = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM performance_events WHERE batch_id=? ORDER BY created_at,id",
                (batch_id,),
            )
        ]
    stages = {}
    for row in rows:
        row["details"] = decode_json(row["details"], {})
        group = stages.setdefault(row["stage"], {"count": 0, "elapsed_ms": 0, "failures": 0})
        group["count"] += 1
        group["elapsed_ms"] += row["elapsed_ms"]
        group["failures"] += not row["success"]
    requests = [r for r in rows if r["stage"].endswith("_request")]
    end = batch["completed_at"] or utc_now()
    wall_ms = (
        datetime.fromisoformat(end) - datetime.fromisoformat(batch["created_at"])
    ).total_seconds() * 1000
    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "wall_ms": wall_ms,
        "stages": stages,
        "requests": len(requests),
        "failed_requests": sum(not r["success"] for r in requests),
        "retries": sum(bool(r["details"].get("retry")) for r in requests),
        "validation_failures": sum(r["stage"] == "validation_failure" for r in rows),
        "tokens": {
            k: sum(r[k] for r in requests)
            for k in ("input_tokens", "output_tokens", "cached_tokens")
        },
        "cost": None,
        "cost_note": "未配置价格，不估算费用",
        "events": rows,
    }
