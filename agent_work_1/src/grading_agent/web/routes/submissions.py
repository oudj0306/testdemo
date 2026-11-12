from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from grading_agent.config import get_settings
from grading_agent.schemas import (
    AnswersUpsert,
)
from grading_agent.services.scoring.submission_grader import grade_submission
from grading_agent.services.workflows.submission_processing import (
    persist_submission,
    prepare_submission,
)
from grading_agent.storage.repository import (
    delete_submission,
    get_assessment,
    get_submission,
    list_submissions,
    upsert_response,
)
from grading_agent.web.http_errors import fail
from grading_agent.web.paths import MAX_UPLOAD_BYTES

router = APIRouter()


@router.post("/api/submissions")
async def submission_create(
    assessment_id: str = Form(...),
    student_ref: str = Form(...),
    answer_layout: str = Form(default="auto"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件不能超过 25 MB")
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    try:
        return await run_in_threadpool(
            persist_submission,
            assessment,
            student_ref,
            content,
            file.content_type,
            None,
            answer_layout,
        )
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/submissions")
def submissions_list() -> list[dict[str, Any]]:
    return list_submissions()


@router.get("/api/submissions/{submission_id}")
def submission_get(submission_id: str) -> dict[str, Any]:
    result = get_submission(submission_id)
    if not result:
        raise HTTPException(404, "答卷不存在")
    return result


@router.delete("/api/submissions/{submission_id}")
def submission_delete(submission_id: str) -> dict[str, Any]:
    try:
        delete_submission(submission_id)
    except Exception as exc:
        raise fail(exc) from exc
    return {"deleted": submission_id}


@router.post("/api/submissions/{submission_id}/answers")
def answers_upsert(submission_id: str, payload: AnswersUpsert) -> dict[str, Any]:
    submission = get_submission(submission_id)
    if not submission:
        raise HTTPException(404, "答卷不存在")
    assessment = get_assessment(submission["assessment_id"])
    questions = assessment["questions"] if assessment else []
    by_number = {question["question_no"]: question for question in questions}
    by_id = {question["id"]: question for question in questions}
    try:
        for answer in payload.answers:
            question = (
                by_id.get(answer.question_id)
                if answer.question_id
                else by_number.get(str(answer.question_no))
            )
            if not question:
                raise ValueError(
                    f"题号或题目 ID 不存在: {answer.question_no or answer.question_id}"
                )
            upsert_response(
                submission_id,
                question["id"],
                answer.raw_text,
                answer.ocr_confidence,
                answer.blank_confirmed,
                "manual",
            )
        return get_submission(submission_id)  # type: ignore[return-value]
    except Exception as exc:
        raise fail(exc) from exc


@router.post("/api/submissions/{submission_id}/grade")
def submission_grade(submission_id: str, pending_only: bool = False) -> dict[str, Any]:
    from grading_agent.infrastructure.work_locks import SubmissionBusyError, submission_work

    try:
        settings = get_settings()
        with submission_work(settings.database_path, submission_id, blocking=False):
            saved = get_submission(submission_id)
            from grading_agent.services.recognition.hybrid_recognition import needs_recognition

            if saved and needs_recognition(
                saved, get_assessment(saved["assessment_id"])["questions"], settings
            ):
                prepared = prepare_submission(submission_id)
                if prepared.get("extraction_warning") and prepared.get(
                    "normalization_report", {}
                ).get("local_recognition", {}).get("mode") not in {"local_only", "hybrid"}:
                    raise ValueError(prepared["extraction_warning"])
            return grade_submission(submission_id, settings, pending_only=pending_only)
    except SubmissionBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise fail(exc) from exc
