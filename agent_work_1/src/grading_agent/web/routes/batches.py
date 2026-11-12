from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from grading_agent.infrastructure.telemetry import summary
from grading_agent.schemas import (
    GradeBatchRequest,
)
from grading_agent.services.recognition.answer_extractor import (
    validate_answer_layout,
)
from grading_agent.services.workflows.batch_queue import enqueue, enqueue_saved
from grading_agent.services.workflows.submission_processing import (
    store_submission,
)
from grading_agent.storage.db import transaction
from grading_agent.storage.repository import (
    add_batch_item,
    complete_batch,
    create_batch,
    get_assessment,
    get_batch,
)
from grading_agent.web.http_errors import fail
from grading_agent.web.paths import MAX_UPLOAD_BYTES

router = APIRouter()


@router.post("/api/batches", status_code=202)
async def batch_create(
    assessment_id: str = Form(...),
    files: list[UploadFile] = File(...),
    auto_grade: bool = Form(default=False),
    answer_layout: str = Form(default="auto"),
    folder: bool = Form(default=False),
) -> dict[str, Any]:
    if not files or (not folder and len(files) > 100):
        raise HTTPException(400, "每批可上传 1 到 100 份答卷")
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    if assessment["status"] != "published":
        raise HTTPException(400, "请先发布试卷，再上传答卷")
    try:
        answer_layout = validate_answer_layout(answer_layout)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        batch = create_batch(assessment_id, len(files))
        for index, file in enumerate(files, start=1):
            filename = file.filename or f"student-{index}.bin"
            student_ref = Path(filename).stem[:100] or f"student-{index}"
            try:
                content = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise ValueError("文件超过 25 MB")
                if auto_grade:
                    await run_in_threadpool(
                        enqueue,
                        batch["id"],
                        filename,
                        student_ref,
                        content,
                        file.content_type,
                        True,
                        answer_layout,
                    )
                else:
                    saved = await run_in_threadpool(
                        store_submission,
                        assessment,
                        student_ref,
                        content,
                        file.content_type,
                        answer_layout,
                    )
                    add_batch_item(
                        batch["id"],
                        filename,
                        student_ref,
                        "duplicate" if saved.get("duplicate") else "completed",
                        saved["id"],
                    )
            except Exception as exc:
                add_batch_item(batch["id"], filename, student_ref, "failed", error=str(exc)[:500])
        return complete_batch(batch["id"])
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/batches")
def batches_list(assessment_id: str) -> list[dict[str, Any]]:
    with transaction() as connection:
        ids = [
            row[0]
            for row in connection.execute(
                """SELECT id FROM grading_batches WHERE assessment_id=? AND
            (status IN ('processing','paused','stopped') OR id IN
             (SELECT recent.id FROM (
                SELECT id FROM grading_batches WHERE assessment_id=? ORDER BY created_at DESC LIMIT 20
             ) AS recent))
            ORDER BY created_at DESC""",
                (assessment_id, assessment_id),
            )
        ]
    return [get_batch(batch_id) for batch_id in ids]


@router.post("/api/assessments/{assessment_id}/grade-batch", status_code=202)
def grade_saved_batch(assessment_id: str, payload: GradeBatchRequest):
    try:
        return enqueue_saved(assessment_id, payload.submission_ids)
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/batches/{batch_id}")
def batch_get(batch_id: str) -> dict[str, Any]:
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    return batch


@router.post("/api/batches/{batch_id}/{action}")
def batch_control_action(batch_id: str, action: str):
    from grading_agent.services.workflows.batch_control import control_batch

    try:
        return control_batch(batch_id, action)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/batches/{batch_id}/performance")
def performance(batch_id: str):
    try:
        return summary(batch_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
