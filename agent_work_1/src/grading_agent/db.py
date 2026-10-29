from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from grading_agent.config import get_settings

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS subjects (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS assessments (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'draft',
    subject_codes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    published_at TEXT
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL REFERENCES assessments(id),
    question_no TEXT NOT NULL,
    prompt TEXT NOT NULL,
    question_type TEXT NOT NULL,
    capture_mode TEXT NOT NULL,
    answer_form TEXT NOT NULL,
    scoring_mode TEXT NOT NULL,
    max_score REAL NOT NULL,
    answer_key TEXT NOT NULL,
    answer_key_state TEXT NOT NULL DEFAULT 'verified',
    rule_config TEXT NOT NULL,
    rubric TEXT NOT NULL,
    review_policy TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(assessment_id, question_no)
);

CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL REFERENCES assessments(id),
    student_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    normalization_report TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finalized_at TEXT
);

CREATE TABLE IF NOT EXISTS submission_pages (
    id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    page_no INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    quality_state TEXT NOT NULL,
    quality_report TEXT NOT NULL,
    UNIQUE(submission_id, page_no)
);

CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    question_id TEXT NOT NULL REFERENCES questions(id),
    raw_text TEXT NOT NULL DEFAULT '',
    crop_path TEXT,
    ocr_confidence REAL,
    blank_confirmed INTEGER NOT NULL DEFAULT 0,
    extraction_source TEXT NOT NULL DEFAULT 'manual',
    updated_at TEXT NOT NULL,
    UNIQUE(submission_id, question_id)
);

CREATE TABLE IF NOT EXISTS grade_items (
    id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    question_id TEXT NOT NULL REFERENCES questions(id),
    response_id TEXT REFERENCES responses(id),
    score REAL,
    max_score REAL NOT NULL,
    score_state TEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    reason_codes TEXT NOT NULL,
    evidence TEXT NOT NULL,
    grader_kind TEXT NOT NULL,
    grader_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    superseded_at TEXT
);

CREATE TABLE IF NOT EXISTS review_events (
    id TEXT PRIMARY KEY,
    grade_item_id TEXT NOT NULL REFERENCES grade_items(id),
    reviewer_ref TEXT NOT NULL,
    final_score REAL NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_assets (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id),
    response_id TEXT REFERENCES responses(id),
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    redacted_text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    approved_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assessment_documents (
    id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL REFERENCES assessments(id),
    kind TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    extracted_text TEXT NOT NULL DEFAULT '',
    extraction_state TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(assessment_id, kind, source_hash)
);

CREATE TABLE IF NOT EXISTS grading_batches (
    id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL REFERENCES assessments(id),
    status TEXT NOT NULL,
    total_count INTEGER NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS grading_batch_items (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES grading_batches(id),
    submission_id TEXT REFERENCES submissions(id),
    filename TEXT NOT NULL,
    student_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    id TEXT PRIMARY KEY,
    assessment_id TEXT REFERENCES assessments(id),
    submission_id TEXT REFERENCES submissions(id),
    stage TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_usage_assessment ON token_usage(assessment_id);
CREATE INDEX IF NOT EXISTS idx_token_usage_submission ON token_usage(submission_id);

CREATE INDEX IF NOT EXISTS idx_questions_assessment ON questions(assessment_id);
CREATE INDEX IF NOT EXISTS idx_submissions_assessment ON submissions(assessment_id);
CREATE INDEX IF NOT EXISTS idx_submissions_hash ON submissions(assessment_id,source_hash);
CREATE INDEX IF NOT EXISTS idx_reviews_grade ON review_events(grade_item_id,created_at);
CREATE INDEX IF NOT EXISTS idx_batch_items_submission ON grading_batch_items(submission_id,status);
CREATE INDEX IF NOT EXISTS idx_grade_items_submission ON grade_items(submission_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_assets(status);
CREATE INDEX IF NOT EXISTS idx_documents_assessment ON assessment_documents(assessment_id);
CREATE INDEX IF NOT EXISTS idx_batch_items_batch ON grading_batch_items(batch_id);

CREATE TABLE IF NOT EXISTS batch_jobs (
    item_id TEXT PRIMARY KEY REFERENCES grading_batch_items(id),
    content BLOB NOT NULL,
    media_type TEXT,
    auto_grade INTEGER NOT NULL,
    answer_layout TEXT NOT NULL DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS performance_events (
    id TEXT PRIMARY KEY, batch_id TEXT, assessment_id TEXT, submission_id TEXT,
    stage TEXT NOT NULL, elapsed_ms REAL NOT NULL, success INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0, details TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_performance_batch ON performance_events(batch_id,stage);
CREATE TABLE IF NOT EXISTS assessment_layouts (
    assessment_id TEXT PRIMARY KEY REFERENCES assessments(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES assessment_documents(id) ON DELETE CASCADE,
    source_hash TEXT NOT NULL, question_fingerprint TEXT NOT NULL, version TEXT NOT NULL,
    payload TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_approvals (
    id TEXT PRIMARY KEY, grade_item_id TEXT NOT NULL REFERENCES grade_items(id) ON DELETE CASCADE,
    review_id TEXT REFERENCES review_events(id) ON DELETE CASCADE, redacted_text TEXT NOT NULL,
    approved_by TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def connect(path: Path | None = None) -> sqlite3.Connection:
    database_path = path or get_settings().database_path
    connection = sqlite3.connect(database_path, check_same_thread=False, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def transaction(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    from time import perf_counter

    started = perf_counter()
    connection = connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        from grading_agent.services.telemetry import database_elapsed

        database_elapsed((perf_counter() - started) * 1000)


def initialize_database(path: Path | None = None) -> None:
    with transaction(path) as connection:
        connection.executescript(SCHEMA)
        question_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(questions)").fetchall()
        }
        if "answer_key_state" not in question_columns:
            connection.execute(
                "ALTER TABLE questions ADD COLUMN answer_key_state TEXT NOT NULL DEFAULT 'verified'"
            )
        grade_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(grade_items)")
        }
        if "snapshot" not in grade_columns:
            connection.execute("ALTER TABLE grade_items ADD COLUMN snapshot TEXT")
        batch_columns = {row["name"] for row in connection.execute("PRAGMA table_info(batch_jobs)")}
        if "answer_layout" not in batch_columns:
            connection.execute(
                "ALTER TABLE batch_jobs ADD COLUMN answer_layout TEXT NOT NULL DEFAULT 'auto'"
            )
        # Review used to block finalization even when every question already had
        # a numeric score. Migrate those historical rows to the new optional-
        # review workflow without touching genuinely scoreless papers.
        connection.execute(
            """UPDATE submissions AS s SET status='finalized', finalized_at=COALESCE(finalized_at, ?)
            WHERE status='review_required'
            AND EXISTS(SELECT 1 FROM questions q WHERE q.assessment_id=s.assessment_id)
            AND NOT EXISTS(
                SELECT 1 FROM questions q
                LEFT JOIN grade_items g ON g.question_id=q.id AND g.submission_id=s.id AND g.superseded_at IS NULL
                WHERE q.assessment_id=s.assessment_id AND (
                    g.id IS NULL OR (g.score IS NULL AND NOT EXISTS(
                        SELECT 1 FROM review_events r WHERE r.grade_item_id=g.id
                    ))
                )
            )""",
            (utc_now(),),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO subjects(code, name, enabled) VALUES (?, ?, ?)",
            [
                ("math", "数学", 1),
                ("physics", "物理", 0),
                ("chinese", "语文", 0),
                ("english", "英语", 0),
            ],
        )
