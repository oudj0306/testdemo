from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    upload_dir: Path
    page_dir: Path
    crop_dir: Path
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str
    llm_grading_passes: int
    llm_extract_answers: bool
    enable_thinking: bool
    ocr_provider: str
    ocr_device: str = "auto"
    local_ocr_first: bool = False
    review_mode: str = "selective"
    review_sample_rate: float = 0.05
    review_confidence_threshold: float = 0.85
    grading_profile: str = "legacy"
    local_vision_enabled: bool = False
    local_vision_url: str = 'http://127.0.0.1:11435'
    local_vision_model: str = 'qwen3-vl:4b-instruct'
    local_vision_verify: bool = True


def _data_dir() -> Path:
    raw_data_dir = os.getenv("GRADING_DATA_DIR", str(PROJECT_ROOT / "data"))
    return Path(raw_data_dir).expanduser().resolve()


def _settings_file() -> Path:
    return _data_dir() / "settings.json"


def _read_overrides() -> dict[str, Any]:
    path = _settings_file()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _as_bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _clamp_passes(value: Any) -> int:
    try:
        return max(1, min(3, int(value)))
    except (TypeError, ValueError):
        return 1


_settings_cache: Settings | None = None


def get_settings() -> Settings:
    global _settings_cache
    if _settings_cache is not None and _settings_cache.data_dir == _data_dir():
        return _settings_cache
    data_dir = _data_dir()
    overrides = _read_overrides()
    from .services.runtime_paths import inherited_ocr_defaults
    inherited = inherited_ocr_defaults(PROJECT_ROOT) if data_dir == (PROJECT_ROOT / 'data').resolve() else {}
    if "openai_api_key" in overrides:
        openai_api_key = overrides["openai_api_key"] or None
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY") or None
    settings = Settings(
        data_dir=data_dir,
        database_path=data_dir / "grading.db",
        upload_dir=data_dir / "uploads",
        page_dir=data_dir / "pages",
        crop_dir=data_dir / "crops",
        openai_api_key=openai_api_key,
        openai_model=str(overrides.get("openai_model") or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL),
        openai_base_url=str(overrides.get("openai_base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/"),
        llm_grading_passes=_clamp_passes(overrides.get("llm_grading_passes", os.getenv("LLM_GRADING_PASSES", "1"))),
        llm_extract_answers=_as_bool(overrides.get("llm_extract_answers", os.getenv("LLM_EXTRACT_ANSWERS", "true"))),
        enable_thinking=_as_bool(overrides.get("enable_thinking", os.getenv("LLM_ENABLE_THINKING", "false"))),
        ocr_provider=str(overrides.get("ocr_provider", os.getenv("OCR_PROVIDER", inherited.get('ocr_provider', 'none')))).lower(),
        ocr_device=str(overrides.get("ocr_device", os.getenv("OCR_DEVICE", inherited.get('ocr_device', 'auto')))),
        local_ocr_first=_as_bool(overrides.get("local_ocr_first", inherited.get('local_ocr_first', False))),
        review_mode=str(overrides.get("review_mode", "selective")),
        review_sample_rate=max(0.0, min(1.0, float(overrides.get("review_sample_rate", os.getenv('REVIEW_SAMPLE_RATE', '0.05'))))),
        review_confidence_threshold=float(overrides.get('review_confidence_threshold', 0.85)),
        grading_profile=str(overrides.get('grading_profile', 'legacy')),
        local_vision_enabled=_as_bool(overrides.get('local_vision_enabled', False)),
        local_vision_url=str(overrides.get('local_vision_url', 'http://127.0.0.1:11435')),
        local_vision_model=str(overrides.get('local_vision_model', 'qwen3-vl:4b-instruct')),
        local_vision_verify=_as_bool(overrides.get('local_vision_verify', True)),
    )
    for directory in (settings.data_dir, settings.upload_dir, settings.page_dir, settings.crop_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _settings_cache = settings
    return settings


def update_local_vision(enabled, model, verify=True):
    global _settings_cache
    overrides=_read_overrides()
    overrides.update(local_vision_enabled=enabled,local_vision_model=model,local_vision_verify=verify)
    _settings_file().write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf-8')
    _settings_cache=None
    return get_settings()


def update_llm_settings(
    api_key: str | None,
    model: str,
    base_url: str,
    llm_grading_passes: int,
    llm_extract_answers: bool,
    enable_thinking: bool,
    ocr_provider: str | None = None,
    ocr_device: str | None = None,
    local_ocr_first: bool | None = None,
    review_mode: str | None = None,
    review_sample_rate: float | None = None,
    review_confidence_threshold: float | None = None,
    grading_profile: str | None = None,
) -> Settings:
    global _settings_cache
    data_dir = _data_dir()
    overrides = _read_overrides()
    if api_key is not None:
        overrides["openai_api_key"] = api_key
    overrides["openai_model"] = model.strip()
    overrides["openai_base_url"] = base_url.rstrip("/")
    overrides["llm_grading_passes"] = str(_clamp_passes(llm_grading_passes))
    overrides["llm_extract_answers"] = "true" if llm_extract_answers else "false"
    overrides["enable_thinking"] = "true" if enable_thinking else "false"
    for name, value in (("ocr_provider", ocr_provider), ("ocr_device", ocr_device),
                        ("local_ocr_first", local_ocr_first), ("review_mode", review_mode),
                        ("review_sample_rate", review_sample_rate), ('review_confidence_threshold', review_confidence_threshold),
                        ('grading_profile', grading_profile)):
        if value is not None:
            overrides[name] = value
    data_dir.mkdir(parents=True, exist_ok=True)
    _settings_file().write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
    _settings_cache = None
    return get_settings()
