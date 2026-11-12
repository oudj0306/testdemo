from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from grading_agent.config import get_settings, update_llm_settings
from grading_agent.schemas import (
    LLMSettingsUpdate,
)
from grading_agent.services.documents.document_renderer import (
    renderer_status,
)
from grading_agent.services.model_io.catalog import (
    MODEL_PROVIDERS,
    apply_provider_options,
    provider_for,
    validate_model_connection,
)
from grading_agent.services.model_io.llm_client import parse_json_object, response_output_text
from grading_agent.services.model_io.model_transport import model_request
from grading_agent.web.paths import PACKAGE_DIR
from grading_agent.web.static_assets import render_index

router = APIRouter()


@router.get("/", include_in_schema=False)
def home() -> HTMLResponse:
    return HTMLResponse(render_index(), headers={"Cache-Control": "no-store"})


@router.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    from grading_agent.services.recognition.ocr.runtime import runtime_status

    return {
        "status": "ok",
        "app_build": "20261028-grade-recovery-1",
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


@router.post("/api/runtime/ocr-check")
def ocr_check() -> dict[str, Any]:
    settings = get_settings()
    if settings.ocr_provider != "paddleocr":
        raise HTTPException(
            400, "请先在“可选 · 识别加速”选择 PaddleOCR 并保存；仅远程模型方式无需自检"
        )
    from grading_agent.services.recognition.ocr.diagnostics import check_ocr

    return check_ocr(settings.ocr_device, PACKAGE_DIR.parents[1] / ".cache" / "ocr-check")


def _mask_api_key(key: str) -> str:
    if len(key) <= 8:
        return "..." + key[-2:] if key else ""
    return key[:4] + "..." + key[-4:]


@router.get("/api/settings")
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


@router.get("/api/model-catalog")
def model_catalog_get() -> dict[str, Any]:
    return {"providers": MODEL_PROVIDERS}


@router.put("/api/settings")
def settings_update(payload: LLMSettingsUpdate) -> dict[str, Any]:
    current = get_settings()
    provider = payload.provider or provider_for(payload.base_url, payload.model)
    try:
        model, base_url = validate_model_connection(provider, payload.model, payload.base_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if (
        current.openai_api_key
        and current.openai_base_url != base_url
        and not (payload.api_key or "").strip()
    ):
        raise HTTPException(
            status_code=422, detail="切换 API 地址时请填写对应服务的密钥，避免旧密钥发送到新地址"
        )
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


@router.post("/api/settings/test-connection")
def settings_test_connection() -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="请先保存 API 密钥")
    payload = {
        "model": settings.openai_model,
        "store": False,
        "max_output_tokens": 128,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": '连接测试。请返回 {"ok":true}。'}],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "connection_test",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            }
        },
    }
    apply_provider_options(payload, settings.openai_base_url, settings.enable_thinking)
    request = model_request(payload, settings)
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            result = json.loads(response.read())
        if parse_json_object(response_output_text(result)).get("ok") is not True:
            raise ValueError("模型未按预期返回结构化结果")
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"模型接口返回 HTTP {exc.code}；请核对密钥、模型权限及地域"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502, detail="连接模型接口失败；请检查网络和 API 地址"
        ) from exc
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail="模型已响应，但结构化输出不符合批改所需格式"
        ) from exc
    return {"ok": True, "message": "基础连接与结构化输出测试通过；照片识别能力仍需用实际答卷验证。"}


class LocalVisionSettings(BaseModel):
    enabled: bool
    model: str = Field(
        default="qwen3-vl:4b-instruct", min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:/-]+$"
    )
    verify: bool = True


@router.get("/api/runtime/local-vision")
def local_vision_status():
    from grading_agent.services.recognition.vision.qwen import runtime

    settings = get_settings()
    result = {
        "enabled": settings.local_vision_enabled,
        "model": settings.local_vision_model,
        "verify": settings.local_vision_verify,
        "url": settings.local_vision_url,
        "hybrid_fallback_enabled": settings.hybrid_fallback_enabled,
    }
    try:
        return {**result, **runtime(settings)}
    except ValueError as exc:
        return {**result, "ready": False, "error": str(exc)}


@router.put("/api/runtime/local-vision")
def local_vision_settings(payload: LocalVisionSettings):
    from grading_agent.config import update_local_vision

    if "cloud" in payload.model.lower():
        raise HTTPException(422, "本地识别禁止云端模型标签")
    if payload.enabled and not payload.verify:
        raise HTTPException(422, "质量优先模式必须启用两次转写核验")
    update_local_vision(payload.enabled, payload.model, payload.verify)
    return local_vision_status()


@router.get("/api/runtime/document-renderer")
def document_renderer_status():
    status = renderer_status()
    return {key: value for key, value in status.items() if key != "executable"}
