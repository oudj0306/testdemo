"""Convert our existing Responses-shaped requests for verified Chat API presets."""
from __future__ import annotations

import json
import urllib.request
from typing import Any
from urllib.parse import urlparse

from ..config import Settings
from ..model_catalog import MODEL_PROVIDERS, provider_for


def prepare_wire_payload(payload: dict[str, Any], settings: Settings) -> tuple[str, dict[str, Any]]:
    provider_id = provider_for(settings.openai_base_url, settings.openai_model)
    provider = next((item for item in MODEL_PROVIDERS if item["id"] == provider_id), None)
    if provider and provider.get("api_style") == "anthropic":
        return _anthropic_payload(payload)
    if not provider or provider.get("api_style") != "chat":
        return "/responses", payload

    messages = []
    for message in payload["input"]:
        content = []
        for part in message["content"]:
            if part["type"] == "input_text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "input_image":
                content.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
            else:
                raise ValueError(f"{provider['name']} 不支持此文件输入类型：{part['type']}")
        role = "system" if message["role"] == "developer" else message["role"]
        # Text-only system messages have the broadest Chat API compatibility.
        if role == "system":
            messages.append({"role": role, "content": "\n".join(part["text"] for part in content)})
        else:
            messages.append({"role": role, "content": content})

    result: dict[str, Any] = {"model": payload["model"], "messages": messages}
    token_limit = payload.get("max_output_tokens")
    if token_limit:
        result["max_completion_tokens" if provider_id == "kimi" else "max_tokens"] = token_limit
    if provider_id == "kimi" and settings.openai_model == "kimi-k3":
        result["reasoning_effort"] = "high" if settings.enable_thinking else "low"
    output_format = provider.get("chat_format")
    if output_format == "json_schema":
        schema = payload["text"]["format"]
        result["response_format"] = {"type": "json_schema", "json_schema": {
            "name": schema["name"], "strict": schema["strict"], "schema": schema["schema"],
        }}
    elif output_format == "json_object":
        result["response_format"] = {"type": "json_object"}
    return "/chat/completions", result


def _anthropic_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = []
    system = []
    for message in payload["input"]:
        content = []
        for part in message["content"]:
            if part["type"] == "input_text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "input_image":
                url = part["image_url"]
                if url.startswith("data:"):
                    header, marker, encoded = url.partition(",")
                    media_type = header[5:].split(";")[0]
                    if not marker or ";base64" not in header or media_type not in {
                        "image/png", "image/jpeg", "image/gif", "image/webp",
                    }:
                        raise ValueError("Claude 仅支持 PNG、JPEG、GIF 和 WebP 图片")
                    source = {"type": "base64", "media_type": media_type, "data": encoded}
                elif urlparse(url).scheme == "https":
                    source = {"type": "url", "url": url}
                else:
                    raise ValueError("Claude 图片地址必须是 HTTPS 或 Base64 数据")
                content.append({"type": "image", "source": source})
            else:
                raise ValueError(f"Claude 不支持此文件输入类型：{part['type']}")
        if message["role"] in {"developer", "system"}:
            system.extend(part["text"] for part in content)
        else:
            messages.append({"role": message["role"], "content": content})
    schema = payload["text"]["format"]["schema"]
    return "/messages", {
        "model": payload["model"], "max_tokens": payload.get("max_output_tokens", 4096),
        "system": "\n".join(system), "messages": messages,
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }


def model_request(payload: dict[str, Any], settings: Settings) -> urllib.request.Request:
    path, wire_payload = prepare_wire_payload(payload, settings)
    headers = {"Content-Type": "application/json"}
    if path == "/messages":
        headers.update({"x-api-key": settings.openai_api_key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    return urllib.request.Request(
        f"{settings.openai_base_url.rstrip('/')}{path}",
        data=json.dumps(wire_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
