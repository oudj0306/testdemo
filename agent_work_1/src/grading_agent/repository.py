from __future__ import annotations

import math
import sqlite3
import uuid
from typing import Any

from grading_agent.db import decode_json, encode_json, transaction, utc_now

JSON_COLUMNS = {
    "snapshot",
    "subject_codes",
    "answer_key",
    "rule_config",
    "rubric",
    "normalization_report",
    "quality_report",
    "reason_codes",
    "evidence",
    "metadata",
}
BOOL_COLUMNS = {"enabled", "blank_confirmed", "needs_review"}


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    value = dict(row)
    for key in JSON_COLUMNS.intersection(value):
        value[key] = decode_json(
            value[key], [] if key in {"subject_codes", "rubric", "reason_codes", "evidence"} else {}
        )
    for key in BOOL_COLUMNS.intersection(value):
        value[key] = bool(value[key])
    return value


def rows_dict(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_dict(row) for row in rows]  # type: ignore[list-item]


def new_id() -> str:
    return uuid.uuid4().hex


def list_subjects() -> list[dict[str, Any]]:
    with transaction() as connection:
        return rows_dict(connection.execute("SELECT * FROM subjects ORDER BY code").fetchall())


def create_assessment(title: str, subject_codes: list[str]) -> dict[str, Any]:
    assessment_id = new_id()
    with transaction() as connection:
        known = {row["code"] for row in connection.execute("SELECT code FROM subjects").fetchall()}
        invalid = set(subject_codes) - known
        if invalid:
            raise ValueError(f"未知学科: {', '.join(sorted(invalid))}")
        connection.execute(
            "INSERT INTO assessments(id,title,subject_codes,created_at) VALUES (?,?,?,?)",
            (assessment_id, title, encode_json(subject_codes), utc_now()),
        )
    return get_assessment(assessment_id)  # type: ignore[return-value]


def list_assessments() -> list[dict[str, Any]]:
    with transaction() as connection:
        return rows_dict(
            connection.execute("SELECT * FROM assessments ORDER BY created_at DESC").fetchall()
        )


def get_assessment(assessment_id: str) -> dict[str, Any] | None:
    with transaction() as connection:
        assessment = row_dict(
            connection.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
        )
        if assessment:
            assessment["questions"] = rows_dict(
                connection.execute(
                    "SELECT * FROM questions WHERE assessment_id=? ORDER BY CAST(question_no AS INTEGER), question_no",
                    (assessment_id,),
                ).fetchall()
            )
        return assessment


def create_question(assessment_id: str, data: dict[str, Any]) -> dict[str, Any]:
    question_id = new_id()
    with transaction() as connection:
        exists = connection.execute(
            "SELECT 1 FROM assessments WHERE id=?", (assessment_id,)
        ).fetchone()
        if not exists:
            raise LookupError("试卷不存在")
        connection.execute(
            """INSERT INTO questions(
                id,assessment_id,question_no,prompt,question_type,capture_mode,answer_form,
                scoring_mode,max_score,answer_key,answer_key_state,rule_config,rubric,review_policy,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                question_id,
                assessment_id,
                data["question_no"],
                data["prompt"],
                data["question_type"],
                data["capture_mode"],
                data["answer_form"],
                data["scoring_mode"],
                data["max_score"],
                encode_json(data["answer_key"]),
                data.get("answer_key_state", "verified"),
                encode_json(data.get("rule_config", {})),
                encode_json(data.get("rubric", [])),
                data.get("review_policy", "auto_if_verified"),
                utc_now(),
            ),
        )
        return row_dict(
            connection.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        )  # type: ignore[return-value]


def create_questions_if_empty(assessment_id: str, values: list[dict[str, Any]]) -> None:
    """Commit a complete generated question set; never leave a partial paper."""
    from grading_agent.schemas import QuestionCreate

    checked = [QuestionCreate.model_validate(v).model_dump(mode="json") for v in values]
    if not checked or len({q["question_no"] for q in checked}) != len(checked):
        raise ValueError("模型未识别到题目或返回重复题号，未写入任何题目")
    with transaction() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM questions WHERE assessment_id=?", (assessment_id,)).fetchone():
            raise ValueError("该试卷已有题目，未覆盖已有配置；请刷新后继续生成位置模板")
        db.executemany(
            """INSERT INTO questions(id,assessment_id,question_no,prompt,question_type,capture_mode,
            answer_form,scoring_mode,max_score,answer_key,answer_key_state,rule_config,rubric,review_policy,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    new_id(),
                    assessment_id,
                    q["question_no"],
                    q["prompt"],
                    q["question_type"],
                    q["capture_mode"],
                    q["answer_form"],
                    q["scoring_mode"],
                    q["max_score"],
                    encode_json(q["answer_key"]),
                    q["answer_key_state"],
                    encode_json(q["rule_config"]),
                    encode_json(q["rubric"]),
                    q["review_policy"],
                    utc_now(),
                )
                for q in checked
            ],
        )


def update_question(question_id: str, data: dict[str, Any]) -> dict[str, Any]:
    with transaction() as connection:
        previous = connection.execute(
            "SELECT * FROM questions WHERE id=?", (question_id,)
        ).fetchone()
        cursor = connection.execute(
            """UPDATE questions SET question_no=?,prompt=?,question_type=?,capture_mode=?,answer_form=?,
            scoring_mode=?,max_score=?,answer_key=?,answer_key_state=?,rule_config=?,rubric=?,review_policy=?
            WHERE id=?""",
            (
                data["question_no"],
                data["prompt"],
                data["question_type"],
                data["capture_mode"],
                data["answer_form"],
                data["scoring_mode"],
                data["max_score"],
                encode_json(data["answer_key"]),
                data.get("answer_key_state", "verified"),
                encode_json(data.get("rule_config", {})),
                encode_json(data.get("rubric", [])),
                data.get("review_policy", "auto_if_verified"),
                question_id,
            ),
        )
        if not cursor.rowcount:
            raise LookupError("题目不存在")
        old_region = decode_json(previous["rule_config"], {}).get("region") if previous else None
        new_region = data.get("rule_config", {}).get("region")
        layout = (
            connection.execute(
                "SELECT payload,document_id FROM assessment_layouts WHERE assessment_id=?",
                (previous["assessment_id"],),
            ).fetchone()
            if previous
            else None
        )
        explicit_remove = "region" in data.get("rule_config", {}) and new_region is None
        if layout and (new_region != old_region or explicit_remove):
            from grading_agent.schemas import RegionInput

            content = decode_json(layout["payload"], {})
            original_item = next(
                (i for i in content["items"] if i["question_id"] == question_id), {}
            )
            content["items"] = [i for i in content["items"] if i["question_id"] != question_id]
            if new_region:
                source_id = data.get("rule_config", {}).get("region_document_id")
                if source_id and source_id != layout["document_id"]:
                    raise ValueError("纠错资料与自动模板来源不同，请选择模板对应原卷或空白答题卡")
                b = RegionInput.model_validate(new_region).model_dump(exclude_none=True)
                if b["page"] not in {p["page"] for p in content["pages"]}:
                    raise ValueError("纠错区域页码不在自动模板中")
                content["items"].append(
                    {
                        "question_id": question_id,
                        "section": original_item.get("section", "人工校正"),
                        "number": original_item.get("number", data["question_no"]),
                        "method": "text",
                        "confidence": 1.0,
                        "regions": [{k: b[k] for k in ("page", "x", "y", "width", "height")}],
                        "options": [],
                    }
                )
            import hashlib

            content.setdefault("region_reviews", {})[question_id] = {
                "source": "manual",
                "confirmed": [],
            }
            content["template_id"] = hashlib.sha256(
                encode_json(content["items"]).encode()
            ).hexdigest()
            connection.execute(
                "UPDATE assessment_layouts SET payload=? WHERE assessment_id=?",
                (encode_json(content), previous["assessment_id"]),
            )
        return row_dict(
            connection.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        )  # type: ignore[return-value]


def update_question_key_rubric(
    question_id: str, answer_key: Any, rubric: list, answer_key_state: str | None = None
) -> None:
    with transaction() as connection:
        if answer_key_state is not None:
            cursor = connection.execute(
                "UPDATE questions SET answer_key=?, rubric=?, answer_key_state=? WHERE id=?",
                (encode_json(answer_key), encode_json(rubric), answer_key_state, question_id),
            )
        else:
            cursor = connection.execute(
                "UPDATE questions SET answer_key=?, rubric=? WHERE id=?",
                (encode_json(answer_key), encode_json(rubric), question_id),
            )
        if not cursor.rowcount:
            raise LookupError("题目不存在")


def publish_assessment(assessment_id: str) -> dict[str, Any]:
    with transaction() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM questions WHERE assessment_id=?", (assessment_id,)
        ).fetchone()[0]
        if not count:
            raise ValueError("至少录入一道题后才能发布")
        if connection.execute(
            "SELECT 1 FROM questions WHERE assessment_id=? AND answer_key_state='missing'",
            (assessment_id,),
        ).fetchone():
            raise ValueError("题目结构已就绪，但参考答案尚未配置；请独立生成或录入答案后发布")
        cursor = connection.execute(
            "UPDATE assessments SET status='published', published_at=? WHERE id=?",
            (utc_now(), assessment_id),
        )
        if not cursor.rowcount:
            raise LookupError("试卷不存在")
    return get_assessment(assessment_id)  # type: ignore[return-value]


def confirm_assessment_answer_keys(assessment_id: str) -> dict[str, Any]:
    with transaction() as connection:
        if connection.execute(
            "SELECT 1 FROM questions WHERE assessment_id=? AND answer_key_state='missing'",
            (assessment_id,),
        ).fetchone():
            raise ValueError("仍有题目没有参考答案，不能将题目提取结果确认为答案")
        cursor = connection.execute(
            "UPDATE questions SET answer_key_state='verified' WHERE assessment_id=?",
            (assessment_id,),
        )
        if not cursor.rowcount:
            raise LookupError("试卷不存在或没有题目")
    return get_assessment(assessment_id)  # type: ignore[return-value]


def create_submission_record(
    assessment_id: str,
    student_ref: str,
    source_hash: str,
    source_path: str,
    media_type: str,
    normalization_report: dict[str, Any],
    pages: list[dict[str, Any]],
    status: str = "extracted",
) -> dict[str, Any]:
    submission_id = new_id()
    with transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT id FROM submissions WHERE assessment_id=? AND source_hash=? ORDER BY created_at,id LIMIT 1",
            (assessment_id, source_hash),
        ).fetchone()
        if existing:
            return {**get_submission(existing["id"]), "duplicate": True}
        assessment = connection.execute(
            "SELECT status FROM assessments WHERE id=?", (assessment_id,)
        ).fetchone()
        if not assessment:
            raise LookupError("试卷不存在")
        if assessment["status"] != "published":
            raise ValueError("只能向已发布试卷提交答卷")
        connection.execute(
            """INSERT INTO submissions(
                id,assessment_id,student_ref,status,source_hash,source_path,media_type,normalization_report,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                submission_id,
                assessment_id,
                student_ref,
                status,
                source_hash,
                source_path,
                media_type,
                encode_json(normalization_report),
                utc_now(),
            ),
        )
        for page in pages:
            connection.execute(
                """INSERT INTO submission_pages(
                    id,submission_id,page_no,image_path,width,height,quality_state,quality_report
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    submission_id,
                    page["page_no"],
                    page["image_path"],
                    page["width"],
                    page["height"],
                    page["quality_state"],
                    encode_json(page["quality_report"]),
                ),
            )
    return get_submission(submission_id)  # type: ignore[return-value]


def list_submissions() -> list[dict[str, Any]]:
    with transaction() as connection:
        return rows_dict(
            connection.execute(
                """SELECT s.*, a.title AS assessment_title,
                (SELECT COUNT(*) FROM review_events r JOIN grade_items g ON g.id=r.grade_item_id WHERE g.submission_id=s.id) AS review_count,
                (SELECT COUNT(*) FROM grade_items g WHERE g.submission_id=s.id AND g.superseded_at IS NULL
                    AND g.needs_review=1 AND g.score IS NOT NULL
                    AND NOT EXISTS(SELECT 1 FROM review_events r WHERE r.grade_item_id=g.id)) AS review_recommended_count
                FROM submissions s
                JOIN assessments a ON a.id=s.assessment_id ORDER BY s.created_at DESC"""
            ).fetchall()
        )


def delete_submission(submission_id: str) -> None:
    with transaction() as connection:
        # Use the same write lock as queue claiming/enqueueing, so a submission
        # cannot disappear while a queued or in-flight job still references it.
        connection.execute("BEGIN IMMEDIATE")
        exists = connection.execute(
            "SELECT id FROM submissions WHERE id=?", (submission_id,)
        ).fetchone()
        if not exists:
            raise LookupError("答卷不存在")
        active = connection.execute(
            "SELECT 1 FROM grading_batch_items WHERE submission_id=? AND status IN ('queued','extracting','grading') LIMIT 1",
            (submission_id,),
        ).fetchone()
        if active:
            raise ValueError("答卷正在排队或处理中，请等待本次任务结束后再删除")
        connection.execute(
            "DELETE FROM review_events WHERE grade_item_id IN (SELECT id FROM grade_items WHERE submission_id=?)",
            (submission_id,),
        )
        connection.execute("DELETE FROM grade_items WHERE submission_id=?", (submission_id,))
        connection.execute(
            "UPDATE knowledge_assets SET response_id=NULL WHERE response_id IN (SELECT id FROM responses WHERE submission_id=?)",
            (submission_id,),
        )
        connection.execute("DELETE FROM responses WHERE submission_id=?", (submission_id,))
        connection.execute("DELETE FROM submission_pages WHERE submission_id=?", (submission_id,))
        connection.execute(
            "UPDATE grading_batch_items SET submission_id=NULL WHERE submission_id=?",
            (submission_id,),
        )
        # Usage is a historical accounting event, not disposable submission data.
        connection.execute(
            "UPDATE token_usage SET submission_id=NULL WHERE submission_id=?", (submission_id,)
        )
        connection.execute("DELETE FROM submissions WHERE id=?", (submission_id,))


def upsert_response(
    submission_id: str,
    question_id: str,
    raw_text: str,
    ocr_confidence: float | None,
    blank_confirmed: bool,
    extraction_source: str,
    crop_path: str | None = None,
) -> dict[str, Any]:
    response_id = new_id()
    with transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT * FROM responses WHERE submission_id=? AND question_id=?",
            (submission_id, question_id),
        ).fetchone()
        if existing and existing["extraction_source"] == "manual" and extraction_source != "manual":
            return row_dict(existing)
        connection.execute(
            "UPDATE grade_items SET superseded_at=? WHERE submission_id=? AND question_id=? AND superseded_at IS NULL",
            (utc_now(), submission_id, question_id),
        )
        connection.execute(
            "UPDATE submissions SET status='extracted',finalized_at=NULL WHERE id=?",
            (submission_id,),
        )
        connection.execute(
            """INSERT INTO responses(
                id,submission_id,question_id,raw_text,crop_path,ocr_confidence,blank_confirmed,extraction_source,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(submission_id,question_id) DO UPDATE SET
                raw_text=excluded.raw_text,
                crop_path=COALESCE(excluded.crop_path,responses.crop_path),
                ocr_confidence=excluded.ocr_confidence,
                blank_confirmed=excluded.blank_confirmed,
                extraction_source=excluded.extraction_source,
                updated_at=excluded.updated_at""",
            (
                response_id,
                submission_id,
                question_id,
                raw_text,
                crop_path,
                ocr_confidence,
                int(blank_confirmed),
                extraction_source,
                utc_now(),
            ),
        )
        return row_dict(
            connection.execute(
                "SELECT * FROM responses WHERE submission_id=? AND question_id=?",
                (submission_id, question_id),
            ).fetchone()
        )  # type: ignore[return-value]


def get_submission(submission_id: str) -> dict[str, Any] | None:
    with transaction() as connection:
        submission = row_dict(
            connection.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
        )
        if not submission:
            return None
        submission["pages"] = rows_dict(
            connection.execute(
                "SELECT * FROM submission_pages WHERE submission_id=? ORDER BY page_no",
                (submission_id,),
            ).fetchall()
        )
        submission["responses"] = rows_dict(
            connection.execute(
                """SELECT r.*,q.question_no,q.prompt,q.max_score FROM responses r
                JOIN questions q ON q.id=r.question_id WHERE r.submission_id=?
                ORDER BY CAST(q.question_no AS INTEGER), q.question_no""",
                (submission_id,),
            ).fetchall()
        )
        grades = rows_dict(
            connection.execute(
                """SELECT g.*,q.question_no,q.prompt,
                (SELECT re.final_score FROM review_events re WHERE re.grade_item_id=g.id ORDER BY re.created_at DESC,re.rowid DESC LIMIT 1) AS final_score,
                (SELECT re.id FROM review_events re WHERE re.grade_item_id=g.id ORDER BY re.created_at DESC,re.rowid DESC LIMIT 1) AS latest_review_id,
                (SELECT COUNT(*) FROM review_events re WHERE re.grade_item_id=g.id) AS review_count,
                (SELECT re.reason FROM review_events re WHERE re.grade_item_id=g.id ORDER BY re.created_at DESC,re.rowid DESC LIMIT 1) AS review_reason
                FROM grade_items g JOIN questions q ON q.id=g.question_id
                WHERE g.submission_id=? AND g.superseded_at IS NULL
                ORDER BY CAST(q.question_no AS INTEGER),q.question_no""",
                (submission_id,),
            ).fetchall()
        )
        for grade in grades:
            grade["effective_score"] = (
                grade["final_score"] if grade["final_score"] is not None else grade["score"]
            )
            grade["effective_state"] = (
                "HUMAN_FINAL" if grade["final_score"] is not None else grade["score_state"]
            )
            grade["bulk_review_eligible"] = bulk_review_eligible(grade)
        submission["grade_items"] = grades
        # A numeric machine score is the current result immediately after grading.
        # Human review is an optional correction layer; only a genuinely missing
        # score keeps the paper unresolved.
        scores = [grade["effective_score"] for grade in grades]
        expected = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(max_score),0) FROM questions WHERE assessment_id=?",
            (submission["assessment_id"],),
        ).fetchone()
        submission["score_summary"] = {
            "confirmed_score": sum(score for score in scores if score is not None),
            "current_score": sum(score for score in scores if score is not None),
            "max_score": expected[1],
            "unresolved_count": sum(score is None for score in scores)
            + max(0, expected[0] - len(grades)),
            "review_recommended_count": sum(
                grade["score"] is not None
                and grade["needs_review"]
                and grade["final_score"] is None
                for grade in grades
            ),
        }
        return submission


def questions_for_submission(
    submission_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    with transaction() as connection:
        submission = row_dict(
            connection.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
        )
        if not submission:
            raise LookupError("答卷不存在")
        questions = rows_dict(
            connection.execute(
                "SELECT * FROM questions WHERE assessment_id=? ORDER BY CAST(question_no AS INTEGER),question_no",
                (submission["assessment_id"],),
            ).fetchall()
        )
        responses = rows_dict(
            connection.execute(
                "SELECT * FROM responses WHERE submission_id=?", (submission_id,)
            ).fetchall()
        )
        return submission, questions, {item["question_id"]: item for item in responses}


def replace_active_grades(
    submission_id: str,
    grades: list[dict[str, Any]],
    partial: bool = False,
    expected_inputs: dict | None = None,
    started_at: str | None = None,
) -> None:
    now = utc_now()
    with transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        for question_id, (expected_question, expected_response) in (expected_inputs or {}).items():
            question = row_dict(
                connection.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
            )
            response = row_dict(
                connection.execute(
                    "SELECT * FROM responses WHERE submission_id=? AND question_id=?",
                    (submission_id, question_id),
                ).fetchone()
            )
            if question != expected_question or response != expected_response:
                raise ValueError("批改期间作答或评分标准已修改；旧结果未保存，请重新批改待处理题")
        if (
            started_at
            and connection.execute(
                """SELECT 1 FROM review_events r JOIN grade_items g ON g.id=r.grade_item_id
                WHERE g.submission_id=? AND r.created_at>=? LIMIT 1""",
                (submission_id, started_at),
            ).fetchone()
        ):
            raise ValueError("批改期间已有人工复核；旧结果未保存，避免覆盖人工成绩")
        if partial:
            connection.executemany(
                "UPDATE grade_items SET superseded_at=? WHERE submission_id=? AND question_id=? AND superseded_at IS NULL",
                [(now, submission_id, g["question_id"]) for g in grades],
            )
        else:
            connection.execute(
                "UPDATE grade_items SET superseded_at=? WHERE submission_id=? AND superseded_at IS NULL",
                (now, submission_id),
            )
        for grade in grades:
            question = row_dict(
                connection.execute(
                    "SELECT * FROM questions WHERE id=?", (grade["question_id"],)
                ).fetchone()
            )
            response = row_dict(
                connection.execute(
                    "SELECT * FROM responses WHERE id=?", (grade.get("response_id"),)
                ).fetchone()
            )
            connection.execute(
                """INSERT INTO grade_items(
                    id,submission_id,question_id,response_id,score,max_score,score_state,needs_review,
                    reason_codes,evidence,grader_kind,grader_version,created_at,snapshot
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    submission_id,
                    grade["question_id"],
                    grade.get("response_id"),
                    grade["score"],
                    grade["max_score"],
                    grade["score_state"],
                    int(grade["needs_review"]),
                    encode_json(grade["reason_codes"]),
                    encode_json(grade["evidence"]),
                    grade["grader_kind"],
                    grade["grader_version"],
                    now,
                    encode_json({"question": question, "response": response}),
                ),
            )
        pending = connection.execute(
            """SELECT COUNT(*) FROM questions q
            JOIN submissions s ON s.assessment_id=q.assessment_id
            LEFT JOIN grade_items g ON g.question_id=q.id AND g.submission_id=s.id AND g.superseded_at IS NULL
            WHERE s.id=? AND (g.id IS NULL OR (g.score IS NULL AND NOT EXISTS
                (SELECT 1 FROM review_events r WHERE r.grade_item_id=g.id)))""",
            (submission_id,),
        ).fetchone()[0]
        status = "review_required" if pending else "finalized"
        finalized_at = None if status == "review_required" else now
        connection.execute(
            "UPDATE submissions SET status=?,finalized_at=? WHERE id=?",
            (status, finalized_at, submission_id),
        )


def bulk_review_eligible(grade: dict) -> bool:
    anomaly_reasons = {
        "score_disagreement",
        "model_requested_review",
        "extraction_uncertain",
        "ocr_confidence_too_low",
        "provisional_answer_key",
        "rubric_incomplete",
        "insufficient_independent_passes",
        "grading_evidence_missing",
        "answer_missing",
        "llm_request_failed",
        "llm_not_configured",
        "manual_scoring_required",
    }
    return (
        grade.get("final_score") is None
        and grade.get("score") is not None
        and math.isfinite(float(grade["score"]))
        and grade.get("needs_review", False)
        and grade.get("score_state") not in {"BLOCKED", "UNSTABLE"}
        and not (set(grade.get("reason_codes", [])) & anomaly_reasons)
        and not grade.get("snapshot", {})
        .get("question", {})
        .get("rule_config", {})
        .get("force_review")
        and grade.get("snapshot", {}).get("question", {}).get("answer_key_state") == "verified"
    )


def create_review(
    grade_item_id: str,
    reviewer_ref: str,
    final_score: float,
    reason: str,
    expected_review_id: str | None = None,
    check_revision: bool = False,
    _connection=None,
) -> dict[str, Any]:
    from contextlib import nullcontext

    reviewer_ref = (reviewer_ref or "").strip()
    reason = (reason or "").strip()
    review_id = new_id()
    with nullcontext(_connection) if _connection is not None else transaction() as connection:
        if _connection is None:
            connection.execute("BEGIN IMMEDIATE")
        grade = connection.execute(
            "SELECT * FROM grade_items WHERE id=?", (grade_item_id,)
        ).fetchone()
        if not grade:
            raise LookupError("批改项不存在")
        if grade["superseded_at"]:
            raise ValueError("此批改记录已被重批取代，请复核最新记录")
        latest = connection.execute(
            "SELECT id FROM review_events WHERE grade_item_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
            (grade_item_id,),
        ).fetchone()
        if check_revision and (latest["id"] if latest else None) != expected_review_id:
            raise ValueError("该题已有新的复核，请刷新后再提交，避免覆盖他人的决定")
        if not math.isfinite(final_score) or not 0 <= final_score <= float(grade["max_score"]):
            raise ValueError("人工定分必须在零分和满分之间")
        connection.execute(
            "INSERT INTO review_events(id,grade_item_id,reviewer_ref,final_score,reason,created_at) VALUES (?,?,?,?,?,?)",
            (review_id, grade_item_id, reviewer_ref, final_score, reason, utc_now()),
        )
        submission_id = grade["submission_id"]
        unresolved = connection.execute(
            """SELECT COUNT(*) FROM questions q JOIN submissions s ON s.assessment_id=q.assessment_id
            LEFT JOIN grade_items g ON g.question_id=q.id AND g.submission_id=s.id AND g.superseded_at IS NULL
            WHERE s.id=? AND (g.id IS NULL OR (g.score IS NULL
            AND NOT EXISTS(SELECT 1 FROM review_events r WHERE r.grade_item_id=g.id)))""",
            (submission_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE submissions SET status=?,finalized_at=? WHERE id=?",
            (
                "review_required" if unresolved else "finalized",
                None if unresolved else utc_now(),
                submission_id,
            ),
        )
        return row_dict(
            connection.execute("SELECT * FROM review_events WHERE id=?", (review_id,)).fetchone()
        )  # type: ignore[return-value]


def bulk_review(ids: list[str], reviewer_ref: str, reason: str) -> list[dict]:
    with transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        results = []
        for grade_id in dict.fromkeys(ids):
            grade = row_dict(
                connection.execute("SELECT * FROM grade_items WHERE id=?", (grade_id,)).fetchone()
            )
            if not grade or grade["superseded_at"] or not bulk_review_eligible(grade):
                raise ValueError("所选题目存在不适合批量确认的记录，请逐题复核；本次未保存任何修改")
            results.append(
                create_review(
                    grade_id, reviewer_ref, grade["score"], reason, None, True, connection
                )
            )
        return results


def eligible_review_ids(assessment_id: str) -> list[str]:
    with transaction() as connection:
        rows = connection.execute(
            """SELECT g.id, g.score, g.score_state, g.needs_review, g.reason_codes, g.snapshot
               FROM grade_items g JOIN submissions s ON s.id=g.submission_id
               JOIN questions q ON q.id=g.question_id
               WHERE s.assessment_id=? AND g.superseded_at IS NULL
                 AND g.score IS NOT NULL AND g.needs_review=1
                 AND g.score_state NOT IN ('BLOCKED','UNSTABLE')
                 AND q.answer_key_state='verified'
                 AND NOT EXISTS (SELECT 1 FROM review_events r WHERE r.grade_item_id=g.id)""",
            (assessment_id,),
        ).fetchall()
    eligible: list[str] = []
    for row in rows:
        grade = {
            "final_score": None,
            "score": row["score"],
            "score_state": row["score_state"],
            "needs_review": bool(row["needs_review"]),
            "reason_codes": decode_json(row["reason_codes"], []),
            "snapshot": decode_json(row["snapshot"], {}),
        }
        if bulk_review_eligible(grade):
            eligible.append(row["id"])
    return eligible


def record_token_usage(
    assessment_id: str | None,
    submission_id: str | None,
    stage: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> None:
    with transaction() as connection:
        connection.execute(
            "INSERT INTO token_usage(id,assessment_id,submission_id,stage,input_tokens,output_tokens,cached_tokens,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                new_id(),
                assessment_id,
                submission_id,
                stage,
                int(input_tokens or 0),
                int(output_tokens or 0),
                int(cached_tokens or 0),
                utc_now(),
            ),
        )


def token_usage_for_assessment(assessment_id: str) -> dict[str, Any]:
    with transaction() as connection:
        rows = rows_dict(
            connection.execute(
                """SELECT t.*, s.student_ref FROM token_usage t
               LEFT JOIN submissions s ON s.id=t.submission_id
               WHERE t.assessment_id=? ORDER BY t.created_at""",
                (assessment_id,),
            ).fetchall()
        )
    stages: dict[str, dict[str, int]] = {}
    submissions: dict[str, dict[str, Any]] = {}
    for r in rows:
        stage = r["stage"]
        stages.setdefault(stage, {"input": 0, "output": 0, "cached": 0})
        stages[stage]["input"] += r["input_tokens"]
        stages[stage]["output"] += r["output_tokens"]
        stages[stage]["cached"] += r["cached_tokens"]
        if r["submission_id"]:
            s = submissions.setdefault(
                r["submission_id"],
                {
                    "submission_id": r["submission_id"],
                    "student": r["student_ref"] or "",
                    "stages": {},
                },
            )
            st = s["stages"].setdefault(stage, {"input": 0, "output": 0, "cached": 0})
            st["input"] += r["input_tokens"]
            st["output"] += r["output_tokens"]
            st["cached"] += r["cached_tokens"]
    for s in submissions.values():
        s["total"] = {
            "input": sum(x["input"] for x in s["stages"].values()),
            "output": sum(x["output"] for x in s["stages"].values()),
            "cached": sum(x["cached"] for x in s["stages"].values()),
        }
    total = {
        "input": sum(x["input"] for x in stages.values()),
        "output": sum(x["output"] for x in stages.values()),
        "cached": sum(x["cached"] for x in stages.values()),
    }
    return {"stages": stages, "total": total, "submissions": list(submissions.values())}


def create_knowledge_asset(
    response_id: str, kind: str, redacted_text: str, approved_by: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    asset_id = new_id()
    with transaction() as connection:
        response = connection.execute(
            "SELECT question_id FROM responses WHERE id=?", (response_id,)
        ).fetchone()
        if not response:
            raise LookupError("答案不存在")
        connection.execute(
            """INSERT INTO knowledge_assets(
                id,question_id,response_id,kind,status,redacted_text,metadata,approved_by,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                asset_id,
                response["question_id"],
                response_id,
                kind,
                "approved",
                redacted_text,
                encode_json(metadata),
                approved_by,
                utc_now(),
            ),
        )
        return row_dict(
            connection.execute("SELECT * FROM knowledge_assets WHERE id=?", (asset_id,)).fetchone()
        )  # type: ignore[return-value]


def search_knowledge(query: str, limit: int = 20) -> list[dict[str, Any]]:
    with transaction() as connection:
        pattern = f"%{query}%"
        approved_answers = rows_dict(
            connection.execute(
                """SELECT k.*,q.question_no,q.prompt,a.title AS assessment_title
                FROM knowledge_assets k JOIN questions q ON q.id=k.question_id
                JOIN assessments a ON a.id=q.assessment_id
                WHERE k.status='approved' AND (k.redacted_text LIKE ? OR q.prompt LIKE ?)
                ORDER BY k.created_at DESC LIMIT ?""",
                (pattern, pattern, max(1, min(limit, 100))),
            ).fetchall()
        )
        documents = rows_dict(
            connection.execute(
                """SELECT d.*,a.title AS assessment_title
                FROM assessment_documents d JOIN assessments a ON a.id=d.assessment_id
                WHERE d.extracted_text LIKE ? OR a.title LIKE ?
                ORDER BY d.created_at DESC LIMIT ?""",
                (pattern, pattern, max(1, min(limit, 100))),
            ).fetchall()
        )
        for item in approved_answers:
            item["result_type"] = "approved_answer"
        for item in documents:
            item["result_type"] = "assessment_document"
        return sorted(
            approved_answers + documents, key=lambda item: item["created_at"], reverse=True
        )[:limit]


def upsert_assessment_document(
    assessment_id: str,
    kind: str,
    source_hash: str,
    source_path: str,
    media_type: str,
    extracted_text: str,
    extraction_state: str,
    page_count: int,
) -> dict[str, Any]:
    document_id = new_id()
    with transaction() as connection:
        if not connection.execute(
            "SELECT 1 FROM assessments WHERE id=?", (assessment_id,)
        ).fetchone():
            raise LookupError("试卷不存在")
        connection.execute(
            """INSERT INTO assessment_documents(
                id,assessment_id,kind,source_hash,source_path,media_type,extracted_text,
                extraction_state,page_count,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(assessment_id,kind,source_hash) DO UPDATE SET
                extracted_text=CASE WHEN excluded.extracted_text<>'' THEN excluded.extracted_text ELSE assessment_documents.extracted_text END,
                extraction_state=excluded.extraction_state,
                page_count=excluded.page_count""",
            (
                document_id,
                assessment_id,
                kind,
                source_hash,
                source_path,
                media_type,
                extracted_text,
                extraction_state,
                page_count,
                utc_now(),
            ),
        )
        return row_dict(
            connection.execute(
                "SELECT * FROM assessment_documents WHERE assessment_id=? AND kind=? AND source_hash=?",
                (assessment_id, kind, source_hash),
            ).fetchone()
        )  # type: ignore[return-value]


def list_assessment_documents(assessment_id: str) -> list[dict[str, Any]]:
    with transaction() as connection:
        return rows_dict(
            connection.execute(
                "SELECT * FROM assessment_documents WHERE assessment_id=? ORDER BY created_at DESC",
                (assessment_id,),
            ).fetchall()
        )


def update_assessment_document_text(document_id: str, extracted_text: str) -> dict[str, Any]:
    with transaction() as connection:
        cursor = connection.execute(
            "UPDATE assessment_documents SET extracted_text=?,extraction_state='verified' WHERE id=?",
            (extracted_text, document_id),
        )
        if not cursor.rowcount:
            raise LookupError("知识文档不存在")
        return row_dict(
            connection.execute(
                "SELECT * FROM assessment_documents WHERE id=?", (document_id,)
            ).fetchone()
        )  # type: ignore[return-value]


def create_batch(assessment_id: str, total_count: int) -> dict[str, Any]:
    batch_id = new_id()
    with transaction() as connection:
        if not connection.execute(
            "SELECT 1 FROM assessments WHERE id=?", (assessment_id,)
        ).fetchone():
            raise LookupError("试卷不存在")
        connection.execute(
            """INSERT INTO grading_batches(id,assessment_id,status,total_count,created_at)
            VALUES (?,?,?,?,?)""",
            (batch_id, assessment_id, "processing", total_count, utc_now()),
        )
    return get_batch(batch_id)  # type: ignore[return-value]


def add_batch_item(
    batch_id: str,
    filename: str,
    student_ref: str,
    status: str,
    submission_id: str | None = None,
    error: str | None = None,
) -> None:
    with transaction() as connection:
        connection.execute(
            """INSERT INTO grading_batch_items(id,batch_id,submission_id,filename,student_ref,status,error,created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (new_id(), batch_id, submission_id, filename, student_ref, status, error, utc_now()),
        )


def complete_batch(batch_id: str) -> dict[str, Any]:
    with transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        counts = connection.execute(
            """SELECT SUM(CASE WHEN status IN ('completed','duplicate') THEN 1 ELSE 0 END) AS succeeded,
            SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
            FROM grading_batch_items WHERE batch_id=?""",
            (batch_id,),
        ).fetchone()
        succeeded = int(counts["succeeded"] or 0)
        failed = int(counts["failed"] or 0)
        batch = connection.execute(
            "SELECT total_count,status FROM grading_batches WHERE id=?", (batch_id,)
        ).fetchone()
        total = batch["total_count"]
        finished = succeeded + failed == total
        status = (
            ("completed" if failed == 0 else ("failed" if succeeded == 0 else "partial"))
            if finished
            else "processing"
        )
        if not finished and batch["status"] in {"paused", "stopped"}:
            status = batch["status"]
        connection.execute(
            "UPDATE grading_batches SET status=?,success_count=?,failed_count=?,completed_at=? WHERE id=?",
            (status, succeeded, failed, utc_now() if finished else None, batch_id),
        )
    return get_batch(batch_id)  # type: ignore[return-value]


def get_batch(batch_id: str) -> dict[str, Any] | None:
    with transaction() as connection:
        batch = row_dict(
            connection.execute("SELECT * FROM grading_batches WHERE id=?", (batch_id,)).fetchone()
        )
        if batch:
            batch["items"] = rows_dict(
                connection.execute(
                    "SELECT * FROM grading_batch_items WHERE batch_id=? ORDER BY created_at",
                    (batch_id,),
                ).fetchall()
            )
            batch["duplicate_count"] = sum(i["status"] == "duplicate" for i in batch["items"])
            batch["active_count"] = sum(
                i["status"] in {"extracting", "grading"} for i in batch["items"]
            )
            batch["settling"] = (
                batch["status"] in {"paused", "stopped"} and batch["active_count"] > 0
            )
            batch["question_count"] = connection.execute(
                "SELECT COUNT(*) FROM questions WHERE assessment_id=?", (batch["assessment_id"],)
            ).fetchone()[0] * sum(i["status"] != "duplicate" for i in batch["items"])
            batch["saved_question_count"] = connection.execute(
                """SELECT COUNT(*) FROM grade_items g
                WHERE g.superseded_at IS NULL AND (g.score IS NOT NULL OR EXISTS(SELECT 1 FROM review_events r WHERE r.grade_item_id=g.id))
                AND g.submission_id IN (SELECT submission_id FROM grading_batch_items WHERE batch_id=?)""",
                (batch_id,),
            ).fetchone()[0]
        return batch
