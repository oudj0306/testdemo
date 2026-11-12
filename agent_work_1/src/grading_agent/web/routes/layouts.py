import hashlib
from pathlib import Path

import pypdfium2 as pdfium
from fastapi import APIRouter, HTTPException, Query
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from grading_agent.config import get_settings
from grading_agent.schemas import RegionInput
from grading_agent.services.documents.document_renderer import (
    WORD_MEDIA,
    word_to_pdf,
)
from grading_agent.services.documents.ingestion import serialized_document_io
from grading_agent.storage.db import decode_json, encode_json, transaction
from grading_agent.storage.repository import get_assessment
from grading_agent.web.image_response import png_response

router = APIRouter()


class LayoutGenerate(BaseModel):
    document_id: str = Field(min_length=1)
    force: bool = False


class LayoutSwitch(BaseModel):
    enabled: bool


class RegionEdit(BaseModel):
    expected_template_id: str
    document_id: str
    regions: list[RegionInput] = Field(max_length=20)
    confirmed_indices: list[int] = Field(default_factory=list, max_length=20)


@router.post("/api/assessments/{assessment_id}/layout-template/calibrate")
def calibrate_layout(assessment_id: str):
    from grading_agent.services.recognition.layout.geometry import calibrate_regions
    from grading_agent.services.recognition.layout.templates import layout_status, validate_layout

    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    state = layout_status(assessment, _include_pages=True)
    if not state["exists"] or state["stale"]:
        raise HTTPException(409, "请先匹配有效的位置模板")
    protected = {
        qid for qid, r in state.get("region_reviews", {}).items() if r.get("source") == "manual"
    }
    protected.update(
        q["id"] for q in assessment["questions"] if q.get("rule_config", {}).get("region")
    )
    try:
        items, adjusted = calibrate_regions(state["items"], state["pages"], protected)
        validate_layout(
            {"items": items, "source_review": state.get("source_review")},
            assessment["questions"],
            len(state["pages"]),
        )
    except ImportError as exc:
        raise HTTPException(422, "本地边线校准需要 OpenCV，请使用 GPU 环境启动") from exc
    with transaction() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT payload FROM assessment_layouts WHERE assessment_id=?", (assessment_id,)
        ).fetchone()
        content = decode_json(row["payload"], {}) if row else {}
        if (
            content.get("template_id") != state["template_id"]
            or layout_status(get_assessment(assessment_id))["stale"]
        ):
            raise HTTPException(409, "校准期间模板已变化，请刷新后重试")
        content["items"] = items
        for qid in adjusted:
            content.setdefault("region_reviews", {})[qid] = {
                "source": "printed_lines",
                "confirmed": [],
            }
        content["template_id"] = hashlib.sha256(encode_json(content).encode()).hexdigest()
        db.execute(
            "UPDATE assessment_layouts SET payload=? WHERE assessment_id=?",
            (encode_json(content), assessment_id),
        )
    return {**layout_status(assessment), "adjusted_items": len(adjusted)}


@router.put("/api/assessments/{assessment_id}/layout-template/regions/{question_id}")
def edit_layout_regions(assessment_id: str, question_id: str, payload: RegionEdit):
    """Change geometry only; never overwrite question text, keys or rubrics."""
    from grading_agent.services.recognition.layout.geometry import region_key
    from grading_agent.services.recognition.layout.templates import layout_status, validate_layout

    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    q = next((q for q in assessment["questions"] if q["id"] == question_id), None)
    if not q or q["scoring_mode"] == "choice_set":
        raise HTTPException(422, "请选择非选择题")
    with transaction() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM assessment_layouts WHERE assessment_id=?", (assessment_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "尚无模板，请先匹配答题卡")
        content = decode_json(row["payload"], {})
        if (
            row["document_id"] != payload.document_id
            or content.get("template_id") != payload.expected_template_id
        ):
            raise HTTPException(409, "模板或来源已变化，请重新打开区域后保存")
        if any(i < 0 or i >= len(payload.regions) for i in payload.confirmed_indices):
            raise HTTPException(422, "确认区域编号不存在")
        regions = [
            {k: b.model_dump()[k] for k in ("page", "x", "y", "width", "height")}
            for b in payload.regions
        ]
        old = next((i for i in content["items"] if i["question_id"] == question_id), {})
        items = [i for i in content["items"] if i["question_id"] != question_id]
        if regions:
            items.append(
                dict(
                    question_id=question_id,
                    section=old.get("section", "人工校正"),
                    number=old.get("number", q["question_no"]),
                    method="text",
                    confidence=1.0,
                    regions=regions,
                    options=[],
                )
            )
        try:
            validate_layout(
                {"items": items, "source_review": content.get("source_review")},
                assessment["questions"],
                len(content["pages"]),
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        reviews = content.setdefault("region_reviews", {})
        previous = reviews.get(question_id, {}).get("confirmed", [])
        keys = [region_key(b) for b in regions]
        reviews[question_id] = {
            "source": "manual",
            "confirmed": list(
                dict.fromkeys(
                    [k for k in previous if k in keys]
                    + [keys[i] for i in payload.confirmed_indices]
                )
            ),
        }
        content["items"] = items
        content["template_id"] = hashlib.sha256(encode_json(content).encode()).hexdigest()
        db.execute(
            "UPDATE assessment_layouts SET payload=? WHERE assessment_id=?",
            (encode_json(content), assessment_id),
        )
    return layout_status(assessment)


@router.get("/api/assessments/{assessment_id}/layout-template")
def get_layout_template(assessment_id: str):
    from grading_agent.services.recognition.layout.templates import layout_status
    from grading_agent.services.recognition.local_answers import dependencies

    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    return {**layout_status(assessment), "local_dependencies": dependencies()}


@router.post("/api/assessments/{assessment_id}/layout-template")
def create_layout_template(assessment_id: str, payload: LayoutGenerate):
    from grading_agent.services.recognition.layout.templates import generate_layout

    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    with transaction() as db:
        row = db.execute(
            "SELECT * FROM assessment_documents WHERE assessment_id=? AND id=? AND kind='answer_sheet'",
            (assessment_id, payload.document_id),
        ).fetchone()
    if not row:
        raise HTTPException(
            404, "请上传并选择资料类型为“空白答题卡模板”的文件；原试卷不作为答题卡定位来源"
        )
    try:
        return generate_layout(assessment, dict(row), get_settings(), payload.force)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("Template generation failed")
        raise HTTPException(
            502, "模板模型调用失败，请检查连接；未自动重试，原模板保持可用"
        ) from exc


@router.put("/api/assessments/{assessment_id}/layout-template")
def switch_layout_template(assessment_id: str, payload: LayoutSwitch):
    from grading_agent.services.recognition.layout.templates import layout_status

    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    state = layout_status(assessment)
    if not state["exists"]:
        raise HTTPException(404, "尚无模板")
    if payload.enabled and state["stale"]:
        raise HTTPException(409, "题目或来源已变更，请重新生成模板")
    with transaction() as db:
        db.execute(
            "UPDATE assessment_layouts SET enabled=? WHERE assessment_id=?",
            (int(payload.enabled), assessment_id),
        )
    return layout_status(assessment)


@router.get("/api/assessments/{assessment_id}/layout-template/page")
def layout_overlay(assessment_id: str, page: int = Query(1, ge=1)):
    from PIL import ImageDraw, ImageFont

    from grading_agent.services.recognition.layout.templates import layout_status

    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, "试卷不存在")
    state = layout_status(assessment)
    if not state["exists"]:
        raise HTTPException(404, "尚无模板")
    with transaction() as db:
        row = db.execute(
            "SELECT payload FROM assessment_layouts WHERE assessment_id=?", (assessment_id,)
        ).fetchone()
    payload = decode_json(row["payload"], {})
    spec = next((p for p in payload["pages"] if p["page"] == page), None)
    if not spec:
        raise HTTPException(404, "模板无此页")
    path = Path(spec["path"]).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()) or not path.is_file():
        raise HTTPException(404, "模板页面不可访问")
    numbers = {q["id"]: q["question_no"] for q in assessment["questions"]}
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for font_path in (
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(font_path).is_file():
            font = ImageFont.truetype(font_path, 18)
            break
    for item in payload["items"]:
        for b in item["regions"] + [o["box"] for o in item["options"]]:
            if b["page"] == page:
                xy = (
                    int(b["x"] * image.width),
                    int(b["y"] * image.height),
                    int((b["x"] + b["width"]) * image.width),
                    int((b["y"] + b["height"]) * image.height),
                )
                draw.rectangle(
                    xy, outline="#20a879" if item["method"] == "omr" else "#416ce5", width=3
                )
                draw.text(
                    (xy[0] + 3, xy[1] + 3),
                    numbers.get(item["question_id"], "?"),
                    fill="#173d80",
                    font=font,
                )
    return png_response(image)


@serialized_document_io
def page_image(assessment_id, document_id, page):
    with transaction() as db:
        row = db.execute(
            "SELECT * FROM assessment_documents WHERE id=? AND assessment_id=?",
            (document_id, assessment_id),
        ).fetchone()
    if not row:
        raise HTTPException(404, "资料不存在")
    path = Path(row["source_path"]).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()) or not path.is_file():
        raise HTTPException(404, "资料不可访问")
    if row["media_type"] in WORD_MEDIA:
        try:
            path = word_to_pdf(path, get_settings().data_dir)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if row["media_type"] == "application/pdf" or row["media_type"] in WORD_MEDIA:
        with pdfium.PdfDocument(path) as pdf:
            if not 1 <= page <= len(pdf):
                raise HTTPException(400, "页码不存在")
            p = pdf[page - 1]
            bitmap = p.render(scale=1.5)
            try:
                return bitmap.to_pil().convert("RGB").copy()
            finally:
                bitmap.close()
                p.close()
    if row["media_type"].startswith("image/") and page == 1:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    raise HTTPException(400, "模板预览支持图片和 PDF；请检查页码")


@router.get("/api/assessments/{assessment_id}/template-page")
def template_page(assessment_id: str, document_id: str, page: int = Query(1, ge=1)):
    with page_image(assessment_id, document_id, page) as image:
        return png_response(image)


@router.post("/api/assessments/{assessment_id}/template-preview")
def template_crop(assessment_id: str, region: RegionInput, document_id: str):
    with page_image(assessment_id, document_id, region.page) as image:
        box = (
            int(region.x * image.width),
            int(region.y * image.height),
            int((region.x + region.width) * image.width),
            int((region.y + region.height) * image.height),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise HTTPException(400, "区域过小")
        return png_response(image.crop(box))
