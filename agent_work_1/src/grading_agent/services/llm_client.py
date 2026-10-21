from __future__ import annotations

import json
import re
from typing import Any


def response_output_text(data: dict[str, Any]) -> str:
    """从 Responses API 返回中提取最终文本，跳过 reasoning 等非 message 项。"""
    if data.get("type") == "message" and isinstance(data.get("content"), list):
        joined = "".join(part.get("text", "") for part in data["content"] if part.get("type") == "text")
        if joined.strip():
            return joined
    choices = data.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            joined = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if joined.strip():
                return joined
    if data.get("output_text"):
        return str(data["output_text"])
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    err = data.get("error")
    if err:
        raise RuntimeError(f"模型接口返回错误: {json.dumps(err, ensure_ascii=False)[:500]}")
    raise RuntimeError(
        f"模型响应缺少 output_text；status={data.get('status')!r}，"
        f"incomplete_details={data.get('incomplete_details')!r}；"
        f"原始响应: {json.dumps(data, ensure_ascii=False)[:800]}"
    )


def response_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage", {}) or {}
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens", usage.get("cache_read_input_tokens", 0))
    return {
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
        "cached_tokens": int(cached_tokens or 0),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    """把模型输出解析为 JSON 对象，兼容纯 JSON、代码块包裹、前后带说明文字。"""
    if not text or not text.strip():
        raise ValueError("模型没有返回内容")
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("模型输出无法解析为 JSON 对象")
