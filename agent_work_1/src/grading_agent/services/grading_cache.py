"""Exact-answer reuse of completed, evidence-backed grading consensus only.

This disposable cache does not replace the grading/audit database. Its SQLite
lease prevents duplicate concurrent requests, including across server workers.
Any cache failure leaves the normal grading path available.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator
import uuid

from ..config import Settings
from ..schemas import ScoreState
from .math_grader import GradeDecision


LOGGER = logging.getLogger(__name__)
CACHE_VERSION = "stable-grade-v1"
ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def review_revision(settings: Settings, question_id: str) -> str | None:
    """Append-only review events invalidate previous automated consensus.

    Read-only mode never creates/changes the main database. An unavailable main
    database disables reuse rather than trusting an unverified old entry.
    """
    if not settings.database_path.exists():
        return "no-database"
    try:
        with closing(sqlite3.connect(settings.database_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            row = db.execute(
                """SELECT COUNT(*), MAX(r.created_at), MAX(r.id)
                FROM review_events r JOIN grade_items g ON g.id=r.grade_item_id
                WHERE g.question_id=?""", (question_id,),
            ).fetchone()
        return json.dumps(row, separators=(",", ":"))
    except (sqlite3.Error, OSError):
        LOGGER.warning("Cannot verify grading review revision; bypassing grade cache")
        return None


def cache_key(settings: Settings, question: dict[str, Any], answer: str,
              prompt_version: str, revision: str | None) -> str | None:
    # Incomplete ad-hoc questions and provisional standards must not share work.
    if revision is None or settings.review_mode != "selective" or any(
        field not in question for field in ("assessment_id", "id", "prompt", "answer_key")
    ) or not question.get("id") or not question.get("assessment_id"):
        return None
    if question.get("answer_key_state") != "verified" or question.get("rule_config", {}).get("force_review"):
        return None
    try:
        rubric = question.get("rubric", [])
        if not rubric or not all(str(r.get("criterion", "")).strip() and math.isfinite(float(r["score"]))
                                 and float(r["score"]) >= 0 for r in rubric) or sum(float(r["score"]) for r in rubric) <= 0:
            return None
        value = {
            "version": CACHE_VERSION, "prompt_version": prompt_version,
            "model": settings.openai_model, "base_url": settings.openai_base_url,
            "enable_thinking": settings.enable_thinking,
            "passes": settings.llm_grading_passes, "review_mode": settings.review_mode,
            "question": question, "review_revision": revision,
            # Do not trim, normalize, or semantically match student work.
            "raw_text": answer,
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, KeyError):
        return None


class StableGradeCache:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "grading_cache.sqlite3"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=2)
        try:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS stable_grades (
                    cache_key TEXT PRIMARY KEY, decision TEXT NOT NULL,
                    independent_passes INTEGER NOT NULL, created_at TEXT NOT NULL
                )""")
                db.execute("""CREATE TABLE IF NOT EXISTS grade_leases (
                    cache_key TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at REAL NOT NULL
                )""")
                yield db
        finally:
            db.close()

    @contextmanager
    def single_flight(self, key: str) -> Iterator[bool]:
        owner = uuid.uuid4().hex
        acquired = False
        # One grading has at most 3 x 45s model calls. The lease also recovers
        # automatically after a crashed process without holding a DB transaction.
        deadline = time.monotonic() + 155
        try:
            while not acquired:
                try:
                    with self._connect() as db:
                        now = time.time()
                        cursor = db.execute(
                            """INSERT INTO grade_leases(cache_key,owner,expires_at) VALUES (?,?,?)
                            ON CONFLICT(cache_key) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at
                            WHERE grade_leases.expires_at < ?""", (key, owner, now + 150, now),
                        )
                        acquired = bool(cursor.rowcount)
                except (sqlite3.Error, OSError):
                    LOGGER.warning("Grade cache unavailable; continuing normal grading")
                    break
                if not acquired:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                try:
                    with self._connect() as db:
                        db.execute("DELETE FROM grade_leases WHERE cache_key=? AND owner=?", (key, owner))
                except (sqlite3.Error, OSError):
                    LOGGER.warning("Could not release grade-cache lease; it will expire")

    def get(self, key: str, max_score: float) -> GradeDecision | None:
        try:
            with self._connect() as db:
                row = db.execute("SELECT decision,independent_passes,created_at FROM stable_grades WHERE cache_key=?", (key,)).fetchone()
            if row is None or row[1] < 2:
                return None
            data = json.loads(row[0])
            if data["needs_review"] is not False or data["score_state"] != ScoreState.CONSENSUS_CANDIDATE:
                return None
            if not isinstance(data["evidence"], list) or not data["evidence"]:
                return None
            if not math.isfinite(float(data["score"])) or not 0 <= float(data["score"]) <= max_score or float(data["max_score"]) != max_score:
                return None
            data["score_state"] = ScoreState(data["score_state"])
            data["usage"] = dict(ZERO_USAGE)
            # Preserve original evidence, while distinguishing its historical
            # API usage from this request (which consumed no model tokens).
            data["evidence"] = [("cache_source_" + e if e.startswith("api_input_tokens=") else e) for e in data["evidence"]]
            data["evidence"].append(f"stable_grade_cache_hit={key};source_created_at={row[2]};independent_passes={row[1]}")
            return GradeDecision(**data)
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, AttributeError):
            LOGGER.warning("Invalid or unavailable grade cache entry; grading normally")
            return None

    def put(self, key: str, decision: GradeDecision, pass_count: int) -> None:
        if pass_count < 2 or decision.needs_review or decision.score_state != ScoreState.CONSENSUS_CANDIDATE or not decision.evidence:
            return
        try:
            encoded = json.dumps(asdict(decision), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            with self._connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO stable_grades(cache_key,decision,independent_passes,created_at) VALUES (?,?,?,?)",
                    (key, encoded, pass_count, datetime.now(UTC).isoformat()),
                )
        except (sqlite3.Error, OSError, TypeError, ValueError):
            LOGGER.warning("Could not save stable grade cache; original grade remains valid")
