from __future__ import annotations

import sqlite3
import hashlib
import json
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import get_settings, update_llm_settings
from .model_catalog import MODEL_PROVIDERS, apply_provider_options, provider_for, validate_model_connection
from .db import initialize_database
from .db import transaction
from .services.batch_queue import BatchQueue, enqueue, enqueue_saved
from .db import encode_json
from .repository import bulk_review, eligible_review_ids, new_id
from .schemas import GradeBatchRequest, BulkReviewRequest
from .repository import (
    add_batch_item,
    complete_batch,
    confirm_assessment_answer_keys,
    create_batch,
    create_assessment,
    create_knowledge_asset,
    create_question,
    create_review,
    create_submission_record,
    delete_submission,
    get_assessment,
    get_batch,
    get_submission,
    list_assessments,
    list_assessment_documents,
    list_subjects,
    list_submissions,
    publish_assessment,
    record_token_usage,
    search_knowledge,
    token_usage_for_assessment,
    update_assessment_document_text,
    update_question,
    update_question_key_rubric,
    upsert_assessment_document,
    upsert_response,
)
from .schemas import AssessmentCreate, AnswersUpsert, DocumentTextUpdate, KnowledgeApprove, LLMSettingsUpdate, QuestionCreate, ReviewCreate
from .services.grading import grade_submission
from .services.ingestion import OCRAdapter, crop_question_regions, extract_reference_text, ingest_document, sniff_media_type, ALLOWED_MEDIA
from .services.question_classifier import infer_question_route
from .services.assessment_compiler import compile_question_paper, fill_missing_parts
from .services.answer_extractor import extract_answers_with_model, reusable_answer_ids, validate_answer_layout, is_choice_question
from .services.llm_client import parse_json_object, response_output_text
from .services.model_transport import model_request


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    queue = BatchQueue(persist_submission, prepare_submission)
    queue.start()
    try:
        yield
    finally:
        await run_in_threadpool(queue.close)


app = FastAPI(title="作业/试卷批改 Agent", version="0.1.0", lifespan=lifespan)
from .workspace_views import router as workspace_router
app.include_router(workspace_router)
from .optimization_api import router as optimization_router
app.include_router(optimization_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def fail(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc))
    if isinstance(exc, (ValueError, sqlite3.IntegrityError)):
        return HTTPException(400, str(exc))
    return HTTPException(500, str(exc))


def persist_submission(
    assessment: dict[str, Any], student_ref: str, content: bytes, content_type: str | None,
    existing_id: str | None = None,
    answer_layout: str = 'auto',
) -> dict[str, Any]:
    settings = get_settings()
    saved_submission = get_submission(existing_id) if existing_id else None
    if saved_submission:
        answer_layout = saved_submission.get('normalization_report', {}).get('answer_layout', answer_layout)
    answer_layout = validate_answer_layout(answer_layout)
    from .services.layout_templates import load_layout
    local_layout = load_layout(assessment)
    has_regions = bool(local_layout) or any(q.get("rule_config", {}).get("region") for q in assessment["questions"])
    # Text-only route is opt-in; formulas/diagrams keep their visual input.
    text_types = {'single_choice', 'multiple_choice', 'true_false', 'short_answer', 'essay', 'reading'}
    local_first = (answer_layout == 'answer_list' and settings.local_ocr_first
                   and settings.ocr_provider == 'paddleocr'
                   and all(q['question_type'] in text_types and not is_choice_question(q) for q in assessment['questions']))
    document = ingest_document(content, content_type, settings, render_pages=has_regions or local_first,
                               render_word_pages=has_regions)
    pages = [
        {
            "page_no": page.page_no,
            "image_path": str(page.path),
            "width": page.width,
            "height": page.height,
            "quality_state": page.quality_state,
            "quality_report": page.quality_report,
        }
        for page in document.pages
    ]
    report = {"page_count": document.source_page_count if document.source_page_count is not None else len(pages),
              "states": [page["quality_state"] for page in pages], "answer_layout": answer_layout}
    submission = saved_submission if saved_submission else create_submission_record(
        assessment["id"], student_ref, document.source_hash, str(document.source_path), document.media_type, report, pages
    )
    if submission.get('duplicate'):
        return submission
    from .services.batch_control import bind_submission,checkpoint,BatchSuspended
    bind_submission(submission['id'])
    checkpoint()
    if existing_id:
        with transaction() as db:
            db.execute('UPDATE submissions SET normalization_report=? WHERE id=?', (encode_json(report),existing_id))
            for page in pages:
                db.execute('INSERT OR IGNORE INTO submission_pages(id,submission_id,page_no,image_path,width,height,quality_state,quality_report) VALUES (?,?,?,?,?,?,?,?)', (new_id(),existing_id,page['page_no'],page['image_path'],page['width'],page['height'],page['quality_state'],encode_json(page['quality_report'])))
    retained_ids = reusable_answer_ids(assessment['questions'], submission['responses'], settings)
    pending_questions = [q for q in assessment['questions'] if q['id'] not in retained_ids]
    if local_layout:
        # Reliable local answers stay local; uncertain ones may use the configured API.
        from .services.local_answers import recognize_answers
        def save_local_answer(answer):
            upsert_response(submission['id'], answer['question_id'], answer['raw_text'],
                            answer['confidence'], answer['blank_confirmed'], answer['source'])
        try:
            answers, warnings, diagnostics = recognize_answers(local_layout, document.pages, pending_questions, settings,on_answer=save_local_answer)
        except BatchSuspended as exc:
            _,warnings,diagnostics=exc.local_result
            reliable=reusable_answer_ids(assessment['questions'],get_submission(submission['id'])['responses'],settings)
            report['local_recognition']={**diagnostics,'warnings':warnings,'mode':'hybrid' if settings.hybrid_fallback_enabled else 'local_only',
                                         'missing':[q['question_no'] for q in assessment['questions'] if q['id'] not in reliable]}
            with transaction() as db:db.execute('UPDATE submissions SET normalization_report=? WHERE id=?',(encode_json(report),submission['id']))
            raise
        from .services.hybrid_recognition import supplement_local_answers
        diagnostics['warnings']=warnings
        try:
            supplement_local_answers(submission,assessment['questions'],document,settings,diagnostics,local_layout,extract_answers_with_model)
        except BatchSuspended:
            reliable=reusable_answer_ids(assessment['questions'],get_submission(submission['id'])['responses'],settings)
            report['local_recognition']={**diagnostics,'mode':'hybrid','missing':[q['question_no'] for q in assessment['questions'] if q['id'] not in reliable]}
            with transaction() as db:db.execute('UPDATE submissions SET normalization_report=? WHERE id=?',(encode_json(report),submission['id']))
            raise
        saved = get_submission(submission['id'])
        reliable = reusable_answer_ids(assessment['questions'], saved['responses'], settings)
        missing = [q['question_no'] for q in assessment['questions'] if q['id'] not in reliable]
        report['local_recognition'] = {**diagnostics, 'warnings': warnings, 'missing': missing,
                                     'mode': 'hybrid' if settings.hybrid_fallback_enabled else 'local_only',
                                     'remote_vision_requests':diagnostics.get('remote_vision_requests',0)}
        with transaction() as db:
            db.execute('UPDATE submissions SET status=?,normalization_report=? WHERE id=?',
                       ('extraction_failed' if missing else 'extracted', encode_json(report), submission['id']))
        result = get_submission(submission['id'])
        if missing:
            result['extraction_warning'] = '仍未可靠识别：' + '、'.join(missing) + '。可人工填写或确认未作答。' + '；'.join(warnings)
        return result
    # Choice marks are localized by the visual extractor, never by plain region OCR.
    crops = crop_question_regions(document.pages, [q for q in pending_questions if not is_choice_question(q)], settings)
    ocr = OCRAdapter(settings.ocr_provider)
    for question in assessment["questions"]:
        crop = crops.get(question["id"])
        if crop and question['id'] not in retained_ids:
            from .services.telemetry import timed, event
            with timed('local_ocr'):
                text, confidence, source = ocr.extract(crop)
            event('local_ocr_result', success=bool(text.strip() and confidence is not None and confidence >= 0.85))
            upsert_response(
                submission["id"], question["id"], text, confidence, False, source, str(crop),
            )
    warning: str | None = None
    local_text = None
    local_confidence = None
    if pending_questions and local_first and not crops and document.pages:
        recognized = [ocr.extract(page.path) for page in document.pages]
        if all(text.strip() and confidence is not None and confidence >= 0.98 for text, confidence, _ in recognized):
            local_text = '\n\n'.join(text for text, _, _ in recognized)
            local_confidence = min(confidence for _, confidence, _ in recognized)
    # Missing/low-confidence region recognition falls back to the visual model.
    saved = get_submission(submission['id'])
    by_id = {q['id']: q for q in assessment['questions']}
    missing_ids = set(by_id) - reusable_answer_ids(assessment['questions'], saved['responses'], settings)
    if missing_ids and settings.llm_extract_answers and settings.openai_api_key and assessment["questions"]:
        try:
            question_by_no = {item["question_no"]: item for item in assessment["questions"]}
            extraction_options = {'local_text': local_text, 'answer_layout': answer_layout}
            if len(missing_ids) < len(by_id):
                extraction_options['target_question_nos'] = [q['question_no'] for q in assessment['questions'] if q['id'] in missing_ids]
            answers, usage = extract_answers_with_model(document, assessment["questions"], settings, **extraction_options)
            record_token_usage(assessment["id"], submission["id"], "extract", usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"])
            # A teacher may have corrected an answer while the model was running.
            current = get_submission(submission['id'])
            missing_ids -= reusable_answer_ids(assessment['questions'], current['responses'], settings)
            for answer in answers:
                question = question_by_no.get(str(answer["question_no"]))
                if not question or question['id'] not in missing_ids:
                    continue
                upsert_response(
                    submission["id"], question["id"], str(answer["raw_text"]),
                    min(float(answer["confidence"]), local_confidence) if local_confidence is not None else float(answer["confidence"]), bool(answer["blank_confirmed"]), "local_ocr_model_mapping" if local_text is not None else "openai_vision",
                )
        except Exception as exc:
            warning = str(exc)[:500]
    with transaction() as db:
        db.execute('UPDATE submissions SET status=? WHERE id=?', ('extraction_failed' if warning else 'extracted',submission['id']))
    result = get_submission(submission["id"])  # type: ignore[assignment]
    if warning and result:
        result["extraction_warning"] = warning
    return result  # type: ignore[return-value]


def store_submission(assessment, student_ref, content, content_type, answer_layout='auto'):
    settings = get_settings()
    answer_layout = validate_answer_layout(answer_layout)
    media = sniff_media_type(content, content_type)
    digest = hashlib.sha256(content).hexdigest()
    path = settings.upload_dir / (digest + ALLOWED_MEDIA[media])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent identical uploads from truncating a file.
    try:
        with path.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        pass
    return create_submission_record(assessment['id'],student_ref,digest,str(path),media,
                                    {'answer_layout': answer_layout},[],status='uploaded')


def prepare_submission(submission_id):
    from .services.work_locks import submission_work
    with submission_work(get_settings().database_path, submission_id):
        return _prepare_submission_locked(submission_id)


def _prepare_submission_locked(submission_id):
    submission = get_submission(submission_id)
    if not submission:
        raise LookupError('答卷不存在')
    path = Path(submission['source_path']).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()):
        raise ValueError('答卷路径不在数据目录内')
    assessment = get_assessment(submission['assessment_id'])
    questions = assessment['questions']
    if questions and len(reusable_answer_ids(questions, submission['responses'], get_settings())) == len(questions):
        with transaction() as db:
            db.execute("UPDATE submissions SET status='extracted' WHERE id=? AND status IN ('uploaded','extraction_failed')", (submission_id,))
        return get_submission(submission_id)
    from .services.hybrid_recognition import supplement_saved_submission
    supplemented=supplement_saved_submission(submission,assessment,get_settings(),extract_answers_with_model)
    if supplemented is not None:return supplemented
    return persist_submission(assessment,submission['student_ref'],path.read_bytes(),submission['media_type'],submission_id)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={'Cache-Control': 'no-store'})


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    from .services.ocr_runtime import runtime_status
    return {
        "status": "ok",
        "app_build": "20261028-routing-rules-1",
        "capabilities": ["fixed_answers_v1", "direct_vision_routing", "batch_control"],
        "ocr_provider": settings.ocr_provider,
        "llm_configured": bool(settings.openai_api_key),
        "llm_passes": settings.llm_grading_passes,
        "llm_extract_answers": settings.llm_extract_answers,
        "ocr_runtime": runtime_status(),
        "ocr_device": settings.ocr_device,
        "review_mode": settings.review_mode,
        "grading_profile": settings.grading_profile,
    }


@app.post('/api/runtime/ocr-check')
def ocr_check() -> dict[str, Any]:
    settings = get_settings()
    if settings.ocr_provider != 'paddleocr':
        raise HTTPException(400, '请先在“可选 · 识别加速”选择 PaddleOCR 并保存；仅远程模型方式无需自检')
    from .services.ocr_diagnostics import check_ocr
    return check_ocr(settings.ocr_device, PACKAGE_DIR.parents[1] / '.cache' / 'ocr-check')


def _mask_api_key(key: str) -> str:
    if len(key) <= 8:
        return "..." + key[-2:] if key else ""
    return key[:4] + "..." + key[-4:]


@app.get("/api/settings")
def settings_get() -> dict[str, Any]:
    settings = get_settings()
    key = settings.openai_api_key or ""
    return {
        "api_key_set": bool(key),
        "api_key_masked": _mask_api_key(key),
        "model": settings.openai_model,
        "base_url": settings.openai_base_url,
        "provider": provider_for(settings.openai_base_url, settings.openai_model),
        "llm_grading_passes": settings.llm_grading_passes,
        "llm_extract_answers": settings.llm_extract_answers,
        "enable_thinking": settings.enable_thinking,
        "ocr_provider": settings.ocr_provider,
        "ocr_device": settings.ocr_device,
        "local_ocr_first": settings.local_ocr_first,
        "hybrid_fallback_enabled": settings.hybrid_fallback_enabled,
        "grading_profile": settings.grading_profile,
        "review_mode": settings.review_mode,
        "review_sample_rate": settings.review_sample_rate,
        "review_confidence_threshold": settings.review_confidence_threshold,
    }


@app.get("/api/model-catalog")
def model_catalog_get() -> dict[str, Any]:
    return {"providers": MODEL_PROVIDERS}


@app.put("/api/settings")
def settings_update(payload: LLMSettingsUpdate) -> dict[str, Any]:
    current = get_settings()
    provider = payload.provider or provider_for(payload.base_url, payload.model)
    try:
        model, base_url = validate_model_connection(provider, payload.model, payload.base_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if current.openai_api_key and current.openai_base_url != base_url and not (payload.api_key or '').strip():
        raise HTTPException(status_code=422, detail="切换 API 地址时请填写对应服务的密钥，避免旧密钥发送到新地址")
    update_llm_settings(
        api_key=payload.api_key,
        model=model,
        base_url=base_url,
        llm_grading_passes=payload.llm_grading_passes,
        llm_extract_answers=payload.llm_extract_answers,
        enable_thinking=payload.enable_thinking,
        ocr_provider=payload.ocr_provider,
        ocr_device=payload.ocr_device,
        local_ocr_first=payload.local_ocr_first,
        review_mode=payload.review_mode,
        review_sample_rate=payload.review_sample_rate,
        review_confidence_threshold=payload.review_confidence_threshold,
        grading_profile=payload.grading_profile,
        hybrid_fallback_enabled=payload.hybrid_fallback_enabled,
    )
    return settings_get()


@app.post("/api/settings/test-connection")
def settings_test_connection() -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="请先保存 API 密钥")
    payload = {
        "model": settings.openai_model, "store": False, "max_output_tokens": 128,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "连接测试。请返回 {\"ok\":true}。"}]}],
        "text": {"format": {"type": "json_schema", "name": "connection_test", "strict": True,
                            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                       "required": ["ok"], "additionalProperties": False}}},
    }
    apply_provider_options(payload, settings.openai_base_url, settings.enable_thinking)
    request = model_request(payload, settings)
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            result = json.loads(response.read())
        if parse_json_object(response_output_text(result)).get("ok") is not True:
            raise ValueError("模型未按预期返回结构化结果")
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"模型接口返回 HTTP {exc.code}；请核对密钥、模型权限及地域") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="连接模型接口失败；请检查网络和 API 地址") from exc
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="模型已响应，但结构化输出不符合批改所需格式") from exc
    return {"ok": True, "message": "基础连接与结构化输出测试通过；照片识别能力仍需用实际答卷验证。"}


@app.get("/api/subjects")
def subjects() -> list[dict[str, Any]]:
    return list_subjects()


@app.post("/api/assessments")
def assessments_create(payload: AssessmentCreate) -> dict[str, Any]:
    try:
        return create_assessment(payload.title, payload.subject_codes)
    except Exception as exc:
        raise fail(exc) from exc


@app.get("/api/assessments")
def assessments_list() -> list[dict[str, Any]]:
    return list_assessments()


@app.get("/api/assessments/{assessment_id}")
def assessment_get(assessment_id: str) -> dict[str, Any]:
    result = get_assessment(assessment_id)
    if not result:
        raise HTTPException(404, "试卷不存在")
    return result


@app.post("/api/assessments/{assessment_id}/questions")
def question_create(assessment_id: str, payload: QuestionCreate) -> dict[str, Any]:
    data = payload.model_dump(mode="json")
    if data["question_type"] == "auto" or data["scoring_mode"] == "auto":
        inferred_type, inferred_mode = infer_question_route(data["prompt"], data["answer_key"], data["rubric"])
        if data["question_type"] == "auto":
            data["question_type"] = inferred_type
        if data["scoring_mode"] == "auto":
            data["scoring_mode"] = inferred_mode
    try:
        return create_question(assessment_id, data)
    except Exception as exc:
        raise fail(exc) from exc


@app.put("/api/questions/{question_id}")
def question_update(question_id: str, payload: QuestionCreate) -> dict[str, Any]:
    data = payload.model_dump(mode="json")
    if data["question_type"] == "auto" or data["scoring_mode"] == "auto":
        inferred_type, inferred_mode = infer_question_route(data["prompt"], data["answer_key"], data["rubric"])
        if data["question_type"] == "auto":
            data["question_type"] = inferred_type
        if data["scoring_mode"] == "auto":
            data["scoring_mode"] = inferred_mode
    try:
        return update_question(question_id, data)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/assessments/{assessment_id}/publish")
def assessment_publish(assessment_id: str) -> dict[str, Any]:
    try:
        return publish_assessment(assessment_id)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/assessments/{assessment_id}/documents")
async def assessment_documents_upload(
    assessment_id: str,
    kind: str = Form(...),
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    if kind not in {"question_paper", "answer_sheet", "answer_key", "rubric", "reference"}:
        raise HTTPException(400, "kind 必须是 question_paper、answer_sheet、answer_key、rubric 或 reference")
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
            document = await run_in_threadpool(ingest_document, content, file.content_type, settings)
            extracted_text, extraction_state = await run_in_threadpool(extract_reference_text, document, ocr)
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


@app.get("/api/assessments/{assessment_id}/documents")
def assessment_documents_list(assessment_id: str) -> list[dict[str, Any]]:
    return list_assessment_documents(assessment_id)


@app.post("/api/assessments/{assessment_id}/generate-candidate-key")
def assessment_generate_candidate_key(assessment_id: str) -> dict[str, Any]:
    return _prepare_questions(assessment_id, structure_only=False)


@app.post("/api/assessments/{assessment_id}/extract-questions")
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
        compiled = (compile_question_paper(documents, get_settings(), structure_only=True) if structure_only
                    else compile_question_paper(documents, get_settings()))
        usage = compiled.get("usage", {})
        record_token_usage(assessment_id, None, "structure" if structure_only else "compile", usage.get("input_tokens", 0), usage.get("output_tokens", 0), 0)
        items = []
        for item in compiled["questions"]:
            import math
            if not math.isfinite(float(item["max_score"])) or not math.isfinite(float(item["confidence"])):
                raise ValueError("题目分值或置信度无效，未写入题目")
            max_score = max(0.5, min(float(item["max_score"]), 1000.0))
            items.append(
                {
                    "question_no": item["question_no"],
                    "prompt": item["prompt"],
                    "question_type": item["question_type"],
                    "capture_mode": "long_handwriting" if item["scoring_mode"] == "rubric_assisted" else "short_handwriting",
                    "answer_form": "text",
                    "scoring_mode": item["scoring_mode"],
                    "max_score": max_score,
                    "answer_key": "" if structure_only else item["candidate_answer"],
                    "answer_key_state": "missing" if structure_only else "candidate",
                    "rule_config": {"generated_confidence": max(0.0, min(float(item["confidence"]), 1.0))},
                    "rubric": [] if structure_only else item["rubric"],
                    "review_policy": "always_review",
                },
            )
        from .repository import create_questions_if_empty
        create_questions_if_empty(assessment_id,items)
        return {"assessment": get_assessment(assessment_id), "usage": usage}
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/assessments/{assessment_id}/confirm-candidate-key")
def assessment_confirm_candidate_key(assessment_id: str) -> dict[str, Any]:
    try:
        return confirm_assessment_answer_keys(assessment_id)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/assessments/{assessment_id}/fill-missing")
def assessment_fill_missing(assessment_id: str) -> dict[str, Any]:
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    try:
        result = fill_missing_parts(assessment["questions"], get_settings(),
                                    list_assessment_documents(assessment_id))
        usage = result.get("usage", {})
        if any(usage.get(k, 0) for k in ("input_tokens", "output_tokens", "cached_tokens")):
            record_token_usage(assessment_id, None, "answer_key", usage.get("input_tokens", 0),
                               usage.get("output_tokens", 0), usage.get("cached_tokens", 0))
        for item in result["updates"]:
            update_question_key_rubric(
                item["id"],
                item["answer_key"],
                item["rubric"],
                "candidate" if item.get("generated_answer") else None,
            )
        return {"filled": len(result["updates"]), "usage": result["usage"], "assessment": get_assessment(assessment_id)}
    except Exception as exc:
        raise fail(exc) from exc


@app.put("/api/assessment-documents/{document_id}/text")
def assessment_document_text_update(document_id: str, payload: DocumentTextUpdate) -> dict[str, Any]:
    try:
        return update_assessment_document_text(document_id, payload.extracted_text)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/demo/bootstrap")
def demo_bootstrap() -> dict[str, Any]:
    assessment = create_assessment("数学演示卷", ["math"])
    examples = [
        {
            "question_no": "1", "prompt": "选择正确答案", "question_type": "single_choice",
            "capture_mode": "short_handwriting", "answer_form": "text", "scoring_mode": "exact_text",
            "max_score": 5, "answer_key": "B", "rule_config": {}, "rubric": [], "review_policy": "auto_if_verified",
        },
        {
            "question_no": "2", "prompt": "计算 1/4 + 1/4", "question_type": "numeric",
            "capture_mode": "short_handwriting", "answer_form": "text", "scoring_mode": "numeric",
            "max_score": 5, "answer_key": "1/2", "rule_config": {"tolerance": "0"}, "rubric": [],
            "review_policy": "auto_if_verified",
        },
        {
            "question_no": "3", "prompt": "化简 2x+2", "question_type": "symbolic",
            "capture_mode": "short_handwriting", "answer_form": "text", "scoring_mode": "symbolic",
            "max_score": 5, "answer_key": "2*(x+1)", "rule_config": {}, "rubric": [],
            "review_policy": "auto_if_verified",
        },
        {
            "question_no": "4", "prompt": "说明解题过程", "question_type": "calculation",
            "capture_mode": "long_handwriting", "answer_form": "text", "scoring_mode": "rubric_assisted",
            "max_score": 10, "answer_key": "", "rule_config": {},
            "rubric": [{"criterion": "列式正确", "score": 4}, {"criterion": "推导正确", "score": 4}, {"criterion": "结论正确", "score": 2}],
            "review_policy": "always_review",
        },
    ]
    for example in examples:
        create_question(assessment["id"], example)
    return publish_assessment(assessment["id"])


@app.post("/api/submissions")
async def submission_create(
    assessment_id: str = Form(...),
    student_ref: str = Form(...),
    answer_layout: str = Form(default='auto'),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件不能超过 25 MB")
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    try:
        return await run_in_threadpool(persist_submission, assessment, student_ref, content, file.content_type,
                                       None, answer_layout)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/batches", status_code=202)
async def batch_create(
    assessment_id: str = Form(...),
    files: list[UploadFile] = File(...),
    auto_grade: bool = Form(default=False),
    answer_layout: str = Form(default='auto'),
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
                    await run_in_threadpool(enqueue, batch["id"], filename, student_ref, content,
                                             file.content_type, True, answer_layout)
                else:
                    saved = await run_in_threadpool(store_submission, assessment, student_ref, content,
                                                    file.content_type, answer_layout)
                    add_batch_item(batch['id'],filename,student_ref,'duplicate' if saved.get('duplicate') else 'completed',saved['id'])
            except Exception as exc:
                add_batch_item(batch["id"], filename, student_ref, "failed", error=str(exc)[:500])
        return complete_batch(batch["id"])
    except Exception as exc:
        raise fail(exc) from exc


@app.get("/api/batches")
def batches_list(assessment_id: str) -> list[dict[str, Any]]:
    with transaction() as connection:
        ids = [row[0] for row in connection.execute(
            """SELECT id FROM grading_batches WHERE assessment_id=? AND
            (status IN ('processing','paused','stopped') OR id IN
             (SELECT id FROM grading_batches WHERE assessment_id=? ORDER BY created_at DESC LIMIT 20))
            ORDER BY created_at DESC""", (assessment_id,assessment_id)
        )]
    return [get_batch(batch_id) for batch_id in ids]


@app.post('/api/assessments/{assessment_id}/grade-batch', status_code=202)
def grade_saved_batch(assessment_id: str, payload: GradeBatchRequest):
    try:
        return enqueue_saved(assessment_id, payload.submission_ids)
    except Exception as exc:
        raise fail(exc) from exc


@app.post('/api/reviews/bulk')
def review_bulk(payload: BulkReviewRequest):
    try:
        return bulk_review(payload.grade_item_ids, payload.reviewer_ref, payload.reason)
    except Exception as exc:
        raise fail(exc) from exc


@app.get('/api/assessments/{assessment_id}/review-eligible')
def review_eligible(assessment_id: str) -> dict[str, Any]:
    ids = eligible_review_ids(assessment_id)
    return {'count': len(ids), 'grade_item_ids': ids}


@app.get('/api/assessments/{assessment_id}/token-usage')
def token_usage(assessment_id: str) -> dict[str, Any]:
    return token_usage_for_assessment(assessment_id)


@app.get("/api/batches/{batch_id}")
def batch_get(batch_id: str) -> dict[str, Any]:
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    return batch


@app.get("/api/submissions")
def submissions_list() -> list[dict[str, Any]]:
    return list_submissions()


@app.get("/api/submissions/{submission_id}")
def submission_get(submission_id: str) -> dict[str, Any]:
    result = get_submission(submission_id)
    if not result:
        raise HTTPException(404, "答卷不存在")
    return result


@app.delete("/api/submissions/{submission_id}")
def submission_delete(submission_id: str) -> dict[str, Any]:
    try:
        delete_submission(submission_id)
    except Exception as exc:
        raise fail(exc) from exc
    return {"deleted": submission_id}


@app.post("/api/submissions/{submission_id}/answers")
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
            question = by_id.get(answer.question_id) if answer.question_id else by_number.get(str(answer.question_no))
            if not question:
                raise ValueError(f"题号或题目 ID 不存在: {answer.question_no or answer.question_id}")
            upsert_response(
                submission_id, question["id"], answer.raw_text, answer.ocr_confidence,
                answer.blank_confirmed, "manual",
            )
        return get_submission(submission_id)  # type: ignore[return-value]
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/submissions/{submission_id}/grade")
def submission_grade(submission_id: str, pending_only: bool = False) -> dict[str, Any]:
    from .services.work_locks import SubmissionBusyError, submission_work
    try:
        settings = get_settings()
        with submission_work(settings.database_path, submission_id, blocking=False):
            saved = get_submission(submission_id)
            from .services.hybrid_recognition import needs_recognition
            if saved and needs_recognition(saved,get_assessment(saved['assessment_id'])['questions'],settings):
                prepared = prepare_submission(submission_id)
                if prepared.get('extraction_warning') and prepared.get('normalization_report', {}).get('local_recognition', {}).get('mode') not in {'local_only','hybrid'}:
                    raise ValueError(prepared['extraction_warning'])
            return grade_submission(submission_id, settings, pending_only=pending_only)
    except SubmissionBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/grade-items/{grade_item_id}/review")
def grade_review(grade_item_id: str, payload: ReviewCreate) -> dict[str, Any]:
    try:
        return create_review(grade_item_id, payload.reviewer_ref, payload.final_score, payload.reason, payload.expected_review_id, payload.check_revision)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/api/knowledge/approve")
def knowledge_approve(payload: KnowledgeApprove) -> dict[str, Any]:
    try:
        return create_knowledge_asset(
            payload.response_id, payload.kind, payload.redacted_text, payload.approved_by, payload.metadata
        )
    except Exception as exc:
        raise fail(exc) from exc


@app.get("/api/knowledge/search")
def knowledge_search(q: str = Query(default="", max_length=200), limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    return search_knowledge(q, limit)
