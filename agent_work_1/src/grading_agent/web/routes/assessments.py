from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from grading_agent.config import get_settings
from grading_agent.schemas import (
    AssessmentCreate,
    DocumentTextUpdate,
    QuestionCreate,
)
from grading_agent.services.documents.assessment_compiler import (
    compile_question_paper,
    fill_missing_parts,
)
from grading_agent.services.documents.ingestion import (
    OCRAdapter,
    extract_reference_text,
    ingest_document,
)
from grading_agent.services.scoring.question_classifier import infer_question_route
from grading_agent.storage.repository import (
    confirm_assessment_answer_keys,
    create_assessment,
    create_question,
    get_assessment,
    list_assessment_documents,
    list_assessments,
    list_subjects,
    publish_assessment,
    record_token_usage,
    update_assessment_document_text,
    update_question,
    update_question_key_rubric,
    upsert_assessment_document,
)
from grading_agent.web.http_errors import fail
from grading_agent.web.paths import MAX_UPLOAD_BYTES

router = APIRouter()


@router.get("/api/subjects")
def subjects() -> list[dict[str, Any]]:
    return list_subjects()


@router.post("/api/assessments")
def assessments_create(payload: AssessmentCreate) -> dict[str, Any]:
    try:
        return create_assessment(payload.title, payload.subject_codes)
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/assessments")
def assessments_list() -> list[dict[str, Any]]:
    return list_assessments()


@router.get("/api/assessments/{assessment_id}")
def assessment_get(assessment_id: str) -> dict[str, Any]:
    result = get_assessment(assessment_id)
    if not result:
        raise HTTPException(404, "试卷不存在")
    return result


@router.post("/api/assessments/{assessment_id}/questions")
def question_create(assessment_id: str, payload: QuestionCreate) -> dict[str, Any]:
    data = payload.model_dump(mode="json")
    if data["question_type"] == "auto" or data["scoring_mode"] == "auto":
        inferred_type, inferred_mode = infer_question_route(
            data["prompt"], data["answer_key"], data["rubric"]
        )
        if data["question_type"] == "auto":
            data["question_type"] = inferred_type
        if data["scoring_mode"] == "auto":
            data["scoring_mode"] = inferred_mode
    try:
        return create_question(assessment_id, data)
    except Exception as exc:
        raise fail(exc) from exc


@router.put("/api/questions/{question_id}")
def question_update(question_id: str, payload: QuestionCreate) -> dict[str, Any]:
    data = payload.model_dump(mode="json")
    if data["question_type"] == "auto" or data["scoring_mode"] == "auto":
        inferred_type, inferred_mode = infer_question_route(
            data["prompt"], data["answer_key"], data["rubric"]
        )
        if data["question_type"] == "auto":
            data["question_type"] = inferred_type
        if data["scoring_mode"] == "auto":
            data["scoring_mode"] = inferred_mode
    try:
        return update_question(question_id, data)
    except Exception as exc:
        raise fail(exc) from exc


@router.post("/api/assessments/{assessment_id}/publish")
def assessment_publish(assessment_id: str) -> dict[str, Any]:
    try:
        return publish_assessment(assessment_id)
    except Exception as exc:
        raise fail(exc) from exc


@router.post("/api/assessments/{assessment_id}/documents")
async def assessment_documents_upload(
    assessment_id: str,
    kind: str = Form(...),
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    if kind not in {"question_paper", "answer_sheet", "answer_key", "rubric", "reference"}:
        raise HTTPException(
            400, "kind 必须是 question_paper、answer_sheet、answer_key、rubric 或 reference"
        )
    if not files or len(files) > 20:
        raise HTTPException(400, "每次可上传 1 到 20 个知识文档")
    if not get_assessment(assessment_id):
        raise HTTPException(404, "试卷不存在")
    settings = get_settings()
    ocr = OCRAdapter(settings.ocr_provider)
    results: list[dict[str, Any]] = []
    try:
        for file in files:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise ValueError(f"{file.filename or '文件'} 超过 25 MB")
            document = await run_in_threadpool(
                ingest_document, content, file.content_type, settings
            )
            extracted_text, extraction_state = await run_in_threadpool(
                extract_reference_text, document, ocr
            )
            results.append(
                upsert_assessment_document(
                    assessment_id,
                    kind,
                    document.source_hash,
                    str(document.source_path),
                    document.media_type,
                    extracted_text,
                    extraction_state,
                    len(document.pages),
                )
            )
        return results
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/assessments/{assessment_id}/documents")
def assessment_documents_list(assessment_id: str) -> list[dict[str, Any]]:
    return list_assessment_documents(assessment_id)


@router.post("/api/assessments/{assessment_id}/generate-candidate-key")
def assessment_generate_candidate_key(assessment_id: str) -> dict[str, Any]:
    return _prepare_questions(assessment_id, structure_only=False)


@router.post("/api/assessments/{assessment_id}/extract-questions")
def assessment_extract_questions(assessment_id: str) -> dict[str, Any]:
    return _prepare_questions(assessment_id, structure_only=True)


def _prepare_questions(assessment_id: str, *, structure_only: bool) -> dict[str, Any]:
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    if assessment["questions"]:
        raise HTTPException(400, "该试卷已有逐题配置；为避免覆盖，请新建试卷版本后再生成")
    try:
        documents = list_assessment_documents(assessment_id)
        compiled = (
            compile_question_paper(documents, get_settings(), structure_only=True)
            if structure_only
            else compile_question_paper(documents, get_settings())
        )
        usage = compiled.get("usage", {})
        record_token_usage(
            assessment_id,
            None,
            "structure" if structure_only else "compile",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            0,
        )
        items = []
        for item in compiled["questions"]:
            import math

            if not math.isfinite(float(item["max_score"])) or not math.isfinite(
                float(item["confidence"])
            ):
                raise ValueError("题目分值或置信度无效，未写入题目")
            max_score = max(0.5, min(float(item["max_score"]), 1000.0))
            items.append(
                {
                    "question_no": item["question_no"],
                    "prompt": item["prompt"],
                    "question_type": item["question_type"],
                    "capture_mode": "long_handwriting"
                    if item["scoring_mode"] == "rubric_assisted"
                    else "short_handwriting",
                    "answer_form": "text",
                    "scoring_mode": item["scoring_mode"],
                    "max_score": max_score,
                    "answer_key": "" if structure_only else item["candidate_answer"],
                    "answer_key_state": "missing" if structure_only else "candidate",
                    "rule_config": {
                        "generated_confidence": max(0.0, min(float(item["confidence"]), 1.0))
                    },
                    "rubric": [] if structure_only else item["rubric"],
                    "review_policy": "always_review",
                },
            )
        from grading_agent.storage.repository import create_questions_if_empty

        create_questions_if_empty(assessment_id, items)
        return {"assessment": get_assessment(assessment_id), "usage": usage}
    except Exception as exc:
        raise fail(exc) from exc


@router.post("/api/assessments/{assessment_id}/confirm-candidate-key")
def assessment_confirm_candidate_key(assessment_id: str) -> dict[str, Any]:
    try:
        return confirm_assessment_answer_keys(assessment_id)
    except Exception as exc:
        raise fail(exc) from exc


@router.post("/api/assessments/{assessment_id}/fill-missing")
def assessment_fill_missing(assessment_id: str) -> dict[str, Any]:
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    try:
        result = fill_missing_parts(
            assessment["questions"], get_settings(), list_assessment_documents(assessment_id)
        )
        usage = result.get("usage", {})
        if any(usage.get(k, 0) for k in ("input_tokens", "output_tokens", "cached_tokens")):
            record_token_usage(
                assessment_id,
                None,
                "answer_key",
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cached_tokens", 0),
            )
        for item in result["updates"]:
            update_question_key_rubric(
                item["id"],
                item["answer_key"],
                item["rubric"],
                "candidate" if item.get("generated_answer") else None,
            )
        return {
            "filled": len(result["updates"]),
            "usage": result["usage"],
            "assessment": get_assessment(assessment_id),
        }
    except Exception as exc:
        raise fail(exc) from exc


@router.put("/api/assessment-documents/{document_id}/text")
def assessment_document_text_update(
    document_id: str, payload: DocumentTextUpdate
) -> dict[str, Any]:
    try:
        return update_assessment_document_text(document_id, payload.extracted_text)
    except Exception as exc:
        raise fail(exc) from exc
