from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from grading_agent.config import get_settings
from grading_agent.schemas import (
    AnswerConfirmation,
    BulkReviewRequest,
    ReviewCreate,
)
from grading_agent.services.documents.ingestion import serialized_document_io
from grading_agent.storage.db import encode_json, transaction
from grading_agent.storage.repository import (
    bulk_review,
    create_review,
    eligible_review_ids,
    get_assessment,
    token_usage_for_assessment,
)
from grading_agent.web.http_errors import fail
from grading_agent.web.image_response import png_response

router = APIRouter()


@router.post("/api/reviews/bulk")
def review_bulk(payload: BulkReviewRequest):
    try:
        return bulk_review(payload.grade_item_ids, payload.reviewer_ref, payload.reason)
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/assessments/{assessment_id}/review-eligible")
def review_eligible(assessment_id: str) -> dict[str, Any]:
    ids = eligible_review_ids(assessment_id)
    return {"count": len(ids), "grade_item_ids": ids}


@router.get("/api/assessments/{assessment_id}/token-usage")
def token_usage(assessment_id: str) -> dict[str, Any]:
    return token_usage_for_assessment(assessment_id)


@router.post("/api/grade-items/{grade_item_id}/review")
def grade_review(grade_item_id: str, payload: ReviewCreate) -> dict[str, Any]:
    try:
        return create_review(
            grade_item_id,
            payload.reviewer_ref,
            payload.final_score,
            payload.reason,
            payload.expected_review_id,
            payload.check_revision,
        )
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/submissions/{submission_id}/review-visual")
@serialized_document_io
def review_visual(submission_id: str, question_id: str):
    from grading_agent.services.review.review_images import visual_manifest

    try:
        return visual_manifest(submission_id, question_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/submissions/{submission_id}/review-page")
@serialized_document_io
def review_student_page(submission_id: str, page: int = Query(1, ge=1)):
    from grading_agent.services.review.review_images import student_page
    from grading_agent.storage.repository import get_submission

    submission = get_submission(submission_id)
    if not submission:
        raise HTTPException(404, "答卷不存在")
    try:
        with student_page(submission, page) as image:
            return png_response(image)
    except (ValueError, OSError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/submissions/{submission_id}/review-region")
@serialized_document_io
def review_student_region(submission_id: str, question_id: str, region: int = Query(0, ge=0)):
    from grading_agent.services.review.review_images import region_image

    try:
        with region_image(submission_id, question_id, region) as image:
            return png_response(image)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/submissions/{submission_id}/questions/{question_id}/confirm-answer")
def confirm_answer(submission_id: str, question_id: str, payload: AnswerConfirmation):
    from grading_agent.infrastructure.work_locks import SubmissionBusyError, submission_work
    from grading_agent.services.scoring.submission_grader import grade_submission
    from grading_agent.storage.repository import get_submission, upsert_response

    settings = get_settings()
    try:
        with submission_work(settings.database_path, submission_id, blocking=False):
            submission = get_submission(submission_id)
            if not submission:
                raise HTTPException(404, "答卷不存在")
            assessment = get_assessment(submission["assessment_id"])
            if question_id not in {q["id"] for q in (assessment or {}).get("questions", [])}:
                raise HTTPException(404, "题目不属于该答卷")
            upsert_response(
                submission_id,
                question_id,
                payload.raw_text,
                None,
                payload.blank_confirmed,
                "manual",
            )
            current = get_submission(submission_id)
            report = current.get("normalization_report", {})
            if "local_recognition" in report:
                from grading_agent.services.recognition.answer_extractor import reusable_answer_ids

                reliable = reusable_answer_ids(
                    assessment["questions"], current["responses"], settings
                )
                local = report["local_recognition"]
                local["missing"] = [
                    q["question_no"] for q in assessment["questions"] if q["id"] not in reliable
                ]
                number = next(
                    q["question_no"] for q in assessment["questions"] if q["id"] == question_id
                )
                local["warnings"] = [
                    w for w in local.get("warnings", []) if not w.startswith(number + "：")
                ]
                with transaction() as db:
                    db.execute(
                        "UPDATE submissions SET normalization_report=? WHERE id=?",
                        (encode_json(report), submission_id),
                    )
            return grade_submission(submission_id, settings, question_ids={question_id})
    except SubmissionBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
