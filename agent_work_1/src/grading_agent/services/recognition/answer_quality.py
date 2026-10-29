"""识别结果的可信度边界；不依赖评分流程，也不进行网络请求。"""

import math
from collections.abc import Mapping
from typing import Any

ADOPTED_TEXT_SOURCES = frozenset({"manual", "local_vlm_consistent", "api_vision_fallback"})


def parse_confidence(value: object) -> float | None:
    """返回 [0, 1] 内的有限置信度；历史脏数据按未知处理，不能当高置信。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return confidence if math.isfinite(confidence) and 0 <= confidence <= 1 else None


def has_reliable_text(response: Mapping[str, Any] | None) -> bool:
    """已采纳的人工/API/双转写结果优先；普通 OCR 至少达到路由阈值。"""
    if not response or not isinstance(response.get("raw_text"), str):
        return False
    if not response["raw_text"].strip():
        return False
    if response.get("extraction_source") in ADOPTED_TEXT_SOURCES:
        return True
    confidence = parse_confidence(response.get("ocr_confidence"))
    return confidence is not None and confidence >= 0.5


def confidence_threshold(question: Mapping[str, Any], settings: Any) -> float:
    """复核阈值来自题目或设置；无效历史配置使用最保守的有效阈值。"""
    configured = question.get("rule_config", {}).get(
        "minimum_ocr_confidence", settings.review_confidence_threshold
    )
    threshold = parse_confidence(configured)
    return threshold if threshold is not None else 1.0
