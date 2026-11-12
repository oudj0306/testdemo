from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from grading_agent.config import get_settings
from grading_agent.storage.repository import get_assessment

router = APIRouter()


@router.post("/api/submissions/{submission_id}/questions/{question_id}/api-recognition-preview")
def api_recognition_preview(submission_id: str, question_id: str):

    from grading_agent.services.documents.ingestion import IngestedDocument, NormalizedPage
    from grading_agent.services.recognition.answer_extractor import extract_answers_with_model
    from grading_agent.services.recognition.hybrid_recognition import extract_uncertain_question
    from grading_agent.services.recognition.layout.templates import load_layout
    from grading_agent.storage.repository import get_submission, record_token_usage

    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(409, "请先配置视觉API模型和密钥")
    submission = get_submission(submission_id)
    if not submission:
        raise HTTPException(404, "答卷不存在")
    assessment = get_assessment(submission["assessment_id"])
    question = next(
        (q for q in (assessment or {}).get("questions", []) if q["id"] == question_id), None
    )
    if not question:
        raise HTTPException(404, "题目不属于该答卷")
    source = Path(submission["source_path"]).resolve()
    if not source.is_relative_to(settings.data_dir.resolve()) or not source.is_file():
        raise HTTPException(404, "原件不可访问")
    pages = []
    for page in submission["pages"]:
        path = Path(page["image_path"]).resolve()
        if not path.is_relative_to(settings.data_dir.resolve()) or not path.is_file():
            raise HTTPException(404, "页面不可访问")
        pages.append(
            NormalizedPage(
                page["page_no"],
                path,
                page["width"],
                page["height"],
                page["quality_state"],
                page["quality_report"],
            )
        )
    document = IngestedDocument(submission["source_hash"], source, submission["media_type"], pages)
    diagnostics = submission.get("normalization_report", {}).get("local_recognition", {})
    try:
        answer, usage, input_source = extract_uncertain_question(
            document,
            question,
            settings,
            diagnostics,
            load_layout(assessment),
            extract_answers_with_model,
        )
        record_token_usage(
            submission["assessment_id"],
            submission_id,
            "extract_preview",
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cached_tokens"],
        )
        return {
            "preview_only": True,
            "writes_grades": False,
            "candidate": answer,
            "usage": usage,
            "input_source": input_source,
        }
    except Exception as exc:
        usage = getattr(exc, "usage", None)
        if usage:
            record_token_usage(
                submission["assessment_id"],
                submission_id,
                "extract_preview",
                usage["input_tokens"],
                usage["output_tokens"],
                usage["cached_tokens"],
            )
        raise HTTPException(422, str(exc)[:500]) from exc


@router.get("/api/submissions/{submission_id}/local-recognition-crop")
def local_recognition_crop(
    submission_id: str, question_id: str, part: int = Query(0, ge=0), region: int = Query(0, ge=0)
):
    from fastapi.responses import FileResponse

    from grading_agent.storage.repository import get_submission

    submission = get_submission(submission_id)
    if not submission:
        raise HTTPException(404, "答卷不存在")
    parts = (
        submission.get("normalization_report", {})
        .get("local_recognition", {})
        .get("recognitions", {})
        .get(question_id, {})
        .get("parts", [])
    )
    if part >= len(parts):
        raise HTTPException(404, "裁切不存在")
    crops = parts[part].get("crops") or [parts[part].get("crop", "")]
    if region >= len(crops):
        raise HTTPException(404, "区域不存在")
    path = (get_settings().crop_dir / crops[region]).resolve()
    if not path.is_relative_to(get_settings().crop_dir.resolve()) or not path.is_file():
        raise HTTPException(404, "裁切不可访问")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/api/submissions/{submission_id}/local-recognition-preview")
def local_recognition_preview(submission_id: str, question_id: str, part: int = Query(0, ge=0)):
    """Re-transcribe a saved crop without changing responses, grades or human labels."""
    from grading_agent.services.recognition.vision.qwen import kind_for, transcribe
    from grading_agent.storage.repository import get_submission

    settings = get_settings()
    if not settings.local_vision_enabled:
        raise HTTPException(409, "请先启用本地视觉识别")
    submission = get_submission(submission_id)
    if not submission:
        raise HTTPException(404, "答卷不存在")
    assessment = get_assessment(submission["assessment_id"])
    question = next(
        (q for q in (assessment or {}).get("questions", []) if q["id"] == question_id), None
    )
    if not question:
        raise HTTPException(404, "题目不属于该答卷")
    parts = (
        submission.get("normalization_report", {})
        .get("local_recognition", {})
        .get("recognitions", {})
        .get(question_id, {})
        .get("parts", [])
    )
    if part >= len(parts):
        raise HTTPException(404, "裁切不存在")
    names = parts[part].get("crops") or [parts[part].get("crop", "")]
    if not names or len(names) > 3:
        raise HTTPException(422, "单次预览最多3个连续区域")
    paths = [(settings.crop_dir / name).resolve() for name in names]
    if any(not p.is_relative_to(settings.crop_dir.resolve()) or not p.is_file() for p in paths):
        raise HTTPException(404, "裁切不可访问")
    try:
        if len(paths) > 1 and kind_for(question) == "text":
            # Historical reports grouped fill-in blanks as code; re-read each blank separately.
            values = [transcribe(path, question, settings) for path in paths]
            result = {
                **values[0],
                "text": "\n".join(value["text"] for value in values),
                "consistent": all(value["consistent"] for value in values),
                "blank_candidate": all(value.get("blank_candidate", False) for value in values),
                "issues": list(
                    dict.fromkeys(issue for value in values for issue in value["issues"])
                ),
                "requests": [request for value in values for request in value.get("requests", [])],
                "region_results": values,
            }
        else:
            result = transcribe(paths, question, settings)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "preview_only": True,
        "writes_grades": False,
        "question_id": question_id,
        "part": part,
        **result,
    }
