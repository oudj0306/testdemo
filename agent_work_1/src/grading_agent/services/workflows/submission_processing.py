from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from grading_agent.config import get_settings
from grading_agent.services.documents.ingestion import (
    ALLOWED_MEDIA,
    OCRAdapter,
    crop_question_regions,
    ingest_document,
    sniff_media_type,
)
from grading_agent.services.recognition.answer_extractor import (
    extract_answers_with_model,
    is_choice_question,
    reusable_answer_ids,
    validate_answer_layout,
)
from grading_agent.storage.db import encode_json, transaction
from grading_agent.storage.repository import (
    create_submission_record,
    get_assessment,
    get_submission,
    new_id,
    record_token_usage,
    upsert_response,
)


def persist_submission(
    assessment: dict[str, Any],
    student_ref: str,
    content: bytes,
    content_type: str | None,
    existing_id: str | None = None,
    answer_layout: str = "auto",
) -> dict[str, Any]:
    settings = get_settings()
    saved_submission = get_submission(existing_id) if existing_id else None
    if saved_submission:
        answer_layout = saved_submission.get("normalization_report", {}).get(
            "answer_layout", answer_layout
        )
    answer_layout = validate_answer_layout(answer_layout)
    from grading_agent.services.recognition.layout.templates import load_layout

    local_layout = load_layout(assessment)
    has_regions = bool(local_layout) or any(
        q.get("rule_config", {}).get("region") for q in assessment["questions"]
    )
    # Text-only route is opt-in; formulas/diagrams keep their visual input.
    text_types = {
        "single_choice",
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "reading",
    }
    local_first = (
        answer_layout == "answer_list"
        and settings.local_ocr_first
        and settings.ocr_provider == "paddleocr"
        and all(
            q["question_type"] in text_types and not is_choice_question(q)
            for q in assessment["questions"]
        )
    )
    document = ingest_document(
        content,
        content_type,
        settings,
        render_pages=has_regions or local_first,
        render_word_pages=has_regions,
    )
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
    report = {
        "page_count": document.source_page_count
        if document.source_page_count is not None
        else len(pages),
        "states": [page["quality_state"] for page in pages],
        "answer_layout": answer_layout,
    }
    submission = (
        saved_submission
        if saved_submission
        else create_submission_record(
            assessment["id"],
            student_ref,
            document.source_hash,
            str(document.source_path),
            document.media_type,
            report,
            pages,
        )
    )
    if submission.get("duplicate"):
        return submission
    from grading_agent.services.workflows.batch_control import (
        BatchSuspended,
        bind_submission,
        checkpoint,
    )

    bind_submission(submission["id"])
    checkpoint()
    if existing_id:
        with transaction() as db:
            db.execute(
                "UPDATE submissions SET normalization_report=? WHERE id=?",
                (encode_json(report), existing_id),
            )
            for page in pages:
                db.execute(
                    "INSERT OR IGNORE INTO submission_pages(id,submission_id,page_no,image_path,width,height,quality_state,quality_report) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        new_id(),
                        existing_id,
                        page["page_no"],
                        page["image_path"],
                        page["width"],
                        page["height"],
                        page["quality_state"],
                        encode_json(page["quality_report"]),
                    ),
                )
    retained_ids = reusable_answer_ids(assessment["questions"], submission["responses"], settings)
    pending_questions = [q for q in assessment["questions"] if q["id"] not in retained_ids]
    if local_layout:
        # Reliable local answers stay local; uncertain ones may use the configured API.
        from grading_agent.services.recognition.local_answers import recognize_answers

        def save_local_answer(answer):
            upsert_response(
                submission["id"],
                answer["question_id"],
                answer["raw_text"],
                answer["confidence"],
                answer["blank_confirmed"],
                answer["source"],
            )

        try:
            answers, warnings, diagnostics = recognize_answers(
                local_layout,
                document.pages,
                pending_questions,
                settings,
                on_answer=save_local_answer,
            )
        except BatchSuspended as exc:
            _, warnings, diagnostics = exc.local_result
            reliable = reusable_answer_ids(
                assessment["questions"], get_submission(submission["id"])["responses"], settings
            )
            report["local_recognition"] = {
                **diagnostics,
                "warnings": warnings,
                "mode": "hybrid" if settings.hybrid_fallback_enabled else "local_only",
                "missing": [
                    q["question_no"] for q in assessment["questions"] if q["id"] not in reliable
                ],
            }
            with transaction() as db:
                db.execute(
                    "UPDATE submissions SET normalization_report=? WHERE id=?",
                    (encode_json(report), submission["id"]),
                )
            raise
        from grading_agent.services.recognition.hybrid_recognition import supplement_local_answers

        diagnostics["warnings"] = warnings
        try:
            supplement_local_answers(
                submission,
                assessment["questions"],
                document,
                settings,
                diagnostics,
                local_layout,
                extract_answers_with_model,
            )
        except BatchSuspended:
            reliable = reusable_answer_ids(
                assessment["questions"], get_submission(submission["id"])["responses"], settings
            )
            report["local_recognition"] = {
                **diagnostics,
                "mode": "hybrid",
                "missing": [
                    q["question_no"] for q in assessment["questions"] if q["id"] not in reliable
                ],
            }
            with transaction() as db:
                db.execute(
                    "UPDATE submissions SET normalization_report=? WHERE id=?",
                    (encode_json(report), submission["id"]),
                )
            raise
        saved = get_submission(submission["id"])
        reliable = reusable_answer_ids(assessment["questions"], saved["responses"], settings)
        missing = [q["question_no"] for q in assessment["questions"] if q["id"] not in reliable]
        report["local_recognition"] = {
            **diagnostics,
            "warnings": warnings,
            "missing": missing,
            "mode": "hybrid" if settings.hybrid_fallback_enabled else "local_only",
            "remote_vision_requests": diagnostics.get("remote_vision_requests", 0),
        }
        with transaction() as db:
            db.execute(
                "UPDATE submissions SET status=?,normalization_report=? WHERE id=?",
                (
                    "extraction_failed" if missing else "extracted",
                    encode_json(report),
                    submission["id"],
                ),
            )
        result = get_submission(submission["id"])
        if missing:
            result["extraction_warning"] = (
                "仍未可靠识别："
                + "、".join(missing)
                + "。可人工填写或确认未作答。"
                + "；".join(warnings)
            )
        return result
    # Choice marks are localized by the visual extractor, never by plain region OCR.
    crops = crop_question_regions(
        document.pages, [q for q in pending_questions if not is_choice_question(q)], settings
    )
    ocr = OCRAdapter(settings.ocr_provider)
    for question in assessment["questions"]:
        crop = crops.get(question["id"])
        if crop and question["id"] not in retained_ids:
            from grading_agent.infrastructure.telemetry import event, timed

            with timed("local_ocr"):
                text, confidence, source = ocr.extract(crop)
            event(
                "local_ocr_result",
                success=bool(text.strip() and confidence is not None and confidence >= 0.85),
            )
            upsert_response(
                submission["id"],
                question["id"],
                text,
                confidence,
                False,
                source,
                str(crop),
            )
    warning: str | None = None
    local_text = None
    local_confidence = None
    if pending_questions and local_first and not crops and document.pages:
        recognized = [ocr.extract(page.path) for page in document.pages]
        if all(
            text.strip() and confidence is not None and confidence >= 0.98
            for text, confidence, _ in recognized
        ):
            local_text = "\n\n".join(text for text, _, _ in recognized)
            local_confidence = min(confidence for _, confidence, _ in recognized)
    # Missing/low-confidence region recognition falls back to the visual model.
    saved = get_submission(submission["id"])
    by_id = {q["id"]: q for q in assessment["questions"]}
    missing_ids = set(by_id) - reusable_answer_ids(
        assessment["questions"], saved["responses"], settings
    )
    if (
        missing_ids
        and settings.llm_extract_answers
        and settings.openai_api_key
        and assessment["questions"]
    ):
        try:
            question_by_no = {item["question_no"]: item for item in assessment["questions"]}
            extraction_options = {"local_text": local_text, "answer_layout": answer_layout}
            if len(missing_ids) < len(by_id):
                extraction_options["target_question_nos"] = [
                    q["question_no"] for q in assessment["questions"] if q["id"] in missing_ids
                ]
            answers, usage = extract_answers_with_model(
                document, assessment["questions"], settings, **extraction_options
            )
            record_token_usage(
                assessment["id"],
                submission["id"],
                "extract",
                usage["input_tokens"],
                usage["output_tokens"],
                usage["cached_tokens"],
            )
            # A teacher may have corrected an answer while the model was running.
            current = get_submission(submission["id"])
            missing_ids -= reusable_answer_ids(
                assessment["questions"], current["responses"], settings
            )
            for answer in answers:
                question = question_by_no.get(str(answer["question_no"]))
                if not question or question["id"] not in missing_ids:
                    continue
                upsert_response(
                    submission["id"],
                    question["id"],
                    str(answer["raw_text"]),
                    min(float(answer["confidence"]), local_confidence)
                    if local_confidence is not None
                    else float(answer["confidence"]),
                    bool(answer["blank_confirmed"]),
                    "local_ocr_model_mapping" if local_text is not None else "openai_vision",
                )
        except Exception as exc:
            warning = str(exc)[:500]
    with transaction() as db:
        db.execute(
            "UPDATE submissions SET status=? WHERE id=?",
            ("extraction_failed" if warning else "extracted", submission["id"]),
        )
    result = get_submission(submission["id"])  # type: ignore[assignment]
    if warning and result:
        result["extraction_warning"] = warning
    return result  # type: ignore[return-value]


def store_submission(assessment, student_ref, content, content_type, answer_layout="auto"):
    settings = get_settings()
    answer_layout = validate_answer_layout(answer_layout)
    media = sniff_media_type(content, content_type)
    digest = hashlib.sha256(content).hexdigest()
    path = settings.upload_dir / (digest + ALLOWED_MEDIA[media])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent identical uploads from truncating a file.
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        pass
    return create_submission_record(
        assessment["id"],
        student_ref,
        digest,
        str(path),
        media,
        {"answer_layout": answer_layout},
        [],
        status="uploaded",
    )


def prepare_submission(submission_id):
    from grading_agent.infrastructure.work_locks import submission_work

    with submission_work(get_settings().database_path, submission_id):
        return _prepare_submission_locked(submission_id)


def _prepare_submission_locked(submission_id):
    submission = get_submission(submission_id)
    if not submission:
        raise LookupError("答卷不存在")
    path = Path(submission["source_path"]).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()):
        raise ValueError("答卷路径不在数据目录内")
    assessment = get_assessment(submission["assessment_id"])
    questions = assessment["questions"]
    if questions and len(
        reusable_answer_ids(questions, submission["responses"], get_settings())
    ) == len(questions):
        with transaction() as db:
            db.execute(
                "UPDATE submissions SET status='extracted' WHERE id=? AND status IN ('uploaded','extraction_failed')",
                (submission_id,),
            )
        return get_submission(submission_id)
    from grading_agent.services.recognition.hybrid_recognition import supplement_saved_submission

    supplemented = supplement_saved_submission(
        submission, assessment, get_settings(), extract_answers_with_model
    )
    if supplemented is not None:
        return supplemented
    return persist_submission(
        assessment,
        submission["student_ref"],
        path.read_bytes(),
        submission["media_type"],
        submission_id,
    )
