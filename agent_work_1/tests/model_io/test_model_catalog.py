import json
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

from grading_agent.app import app
from grading_agent.config import get_settings
from grading_agent.services.model_io.catalog import (
    MODEL_PROVIDERS,
    apply_provider_options,
    provider_for,
)
from grading_agent.services.model_io.llm_client import response_output_text, response_usage
from grading_agent.services.model_io.model_transport import model_request, prepare_wire_payload
from grading_agent.services.model_io.vision_input import add_visual_document


def test_catalog_presets_have_verified_responses_urls_and_vision_models():
    ids = {item["id"] for item in MODEL_PROVIDERS}
    assert {
        "qwen",
        "deepseek",
        "doubao",
        "zhipu",
        "openai",
        "gemini",
        "kimi",
        "hunyuan",
        "baidu",
        "anthropic",
    } <= ids
    for item in MODEL_PROVIDERS:
        assert item["base_url"].startswith("https://")
        assert item["docs_url"].startswith("https://")
        assert provider_for(item["base_url"], item["models"][0]) == item["id"]
    assert provider_for("https://custom.example/v1", "my-vision-model") == "custom"


def test_provider_specific_thinking_parameter():
    cases = [
        ("qwen", "enable_thinking"),
        ("deepseek", "reasoning"),
        ("doubao", "thinking"),
        ("zhipu", "reasoning"),
    ]
    for provider_id, expected in cases:
        provider = next(p for p in MODEL_PROVIDERS if p["id"] == provider_id)
        payload = {}
        apply_provider_options(payload, provider["base_url"], True)
        assert expected in payload
        assert "enable_thinking" not in payload or provider_id == "qwen"
    custom = {}
    apply_provider_options(custom, "https://maas.qianwenaiapi.com/compatible-mode/v1", False)
    assert custom == {"enable_thinking": False}


def test_existing_custom_pdf_path_stays_unchanged_but_qwen_pdf_is_visible(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-test")
    settings = get_settings()
    content = []
    add_visual_document(
        content,
        path,
        "application/pdf",
        replace(
            settings, openai_model="custom-vision", openai_base_url="https://gateway.example/v1"
        ),
    )
    assert content[0]["type"] == "input_file"
    with patch(
        "grading_agent.services.model_io.vision_input.pdf_visual_pages", return_value=[(1, b"png")]
    ):
        content = []
        add_visual_document(
            content,
            path,
            "application/pdf",
            replace(
                settings,
                openai_model="qwen3.8-max",
                openai_base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
            ),
        )
    assert [item["type"] for item in content] == ["input_text", "input_image"]


def test_provider_switch_requires_new_key_and_preserves_old_configuration():
    qwen = next(p for p in MODEL_PROVIDERS if p["id"] == "qwen")
    deepseek = next(p for p in MODEL_PROVIDERS if p["id"] == "deepseek")
    with TestClient(app) as client:
        first = client.put(
            "/api/settings",
            json={
                "provider": "qwen",
                "model": qwen["models"][0],
                "base_url": qwen["base_url"],
                "api_key": "qwen-secret",
            },
        )
        assert first.status_code == 200
        rejected = client.put(
            "/api/settings",
            json={
                "provider": "deepseek",
                "model": deepseek["models"][0],
                "base_url": deepseek["base_url"],
            },
        )
        assert rejected.status_code == 422
        assert client.get("/api/settings").json()["provider"] == "qwen"
        accepted = client.put(
            "/api/settings",
            json={
                "provider": "deepseek",
                "model": deepseek["models"][0],
                "base_url": deepseek["base_url"],
                "api_key": "deepseek-secret",
            },
        )
        assert accepted.status_code == 200
        assert accepted.json()["provider"] == "deepseek"
        assert "deepseek-secret" not in accepted.text


def test_existing_custom_qwen_gateway_can_be_saved_without_changing_key():
    base_url = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    with TestClient(app) as client:
        original = client.put(
            "/api/settings",
            json={
                "provider": "custom",
                "model": "qwen3.8-max",
                "base_url": base_url,
                "api_key": "existing-private-key",
                "enable_thinking": False,
            },
        )
        assert original.status_code == 200
        saved = client.put(
            "/api/settings",
            json={
                "provider": "custom",
                "model": "qwen3.8-max",
                "base_url": base_url,
                "enable_thinking": False,
            },
        )
        assert saved.status_code == 200
        assert saved.json()["provider"] == "custom"
        assert saved.json()["base_url"] == base_url
        assert saved.json()["api_key_set"] is True
        assert "existing-private-key" not in saved.text


def test_connection_probe_uses_saved_endpoint_and_structured_output():
    provider = next(p for p in MODEL_PROVIDERS if p["id"] == "deepseek")
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"output_text": '{"ok":true}'}).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return FakeResponse()

    with TestClient(app) as client:
        client.put(
            "/api/settings",
            json={
                "provider": "deepseek",
                "model": provider["models"][0],
                "base_url": provider["base_url"],
                "api_key": "fake-key",
            },
        )
        with patch("grading_agent.app.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.post("/api/settings/test-connection")
    assert result.status_code == 200
    assert seen["url"] == provider["base_url"] + "/responses"
    assert seen["body"]["text"]["format"]["type"] == "json_schema"
    assert "fake-key" not in result.text


def test_chat_adapter_preserves_image_and_schema_without_responses_only_fields():
    settings = get_settings()
    payload = {
        "model": "gemini-3.8-flash",
        "store": False,
        "prompt_cache_key": "private-cache",
        "max_output_tokens": 200,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Return JSON"}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Read this"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,cG5n",
                        "detail": "high",
                    },
                ],
            },
        ],
        "text": {"format": {"name": "result", "strict": True, "schema": {"type": "object"}}},
    }
    gemini = next(p for p in MODEL_PROVIDERS if p["id"] == "gemini")
    path, body = prepare_wire_payload(
        payload,
        replace(settings, openai_model=gemini["models"][0], openai_base_url=gemini["base_url"]),
    )
    assert path == "/chat/completions"
    assert body["messages"][0] == {"role": "system", "content": "Return JSON"}
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert body["response_format"]["json_schema"]["name"] == "result"
    assert "store" not in body and "prompt_cache_key" not in body and "text" not in body
    kimi = next(p for p in MODEL_PROVIDERS if p["id"] == "kimi")
    _, kimi_body = prepare_wire_payload(
        payload | {"model": kimi["models"][0]},
        replace(settings, openai_model=kimi["models"][0], openai_base_url=kimi["base_url"]),
    )
    assert kimi_body["response_format"] == {"type": "json_object"}
    assert kimi_body["max_completion_tokens"] == 200
    for provider_id in ("hunyuan", "baidu"):
        provider = next(p for p in MODEL_PROVIDERS if p["id"] == provider_id)
        _, wire = prepare_wire_payload(
            payload | {"model": provider["models"][0]},
            replace(
                settings, openai_model=provider["models"][0], openai_base_url=provider["base_url"]
            ),
        )
        assert "response_format" not in wire


def test_chat_result_text_and_usage():
    data = {
        "choices": [{"message": {"content": '{"ok":true}'}}],
        "usage": {
            "prompt_tokens": 22,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 7},
        },
    }
    assert response_output_text(data) == '{"ok":true}'
    assert response_usage(data) == {"input_tokens": 22, "output_tokens": 3, "cached_tokens": 7}


def test_anthropic_native_image_schema_headers_and_result():
    provider = next(p for p in MODEL_PROVIDERS if p["id"] == "anthropic")
    settings = replace(
        get_settings(),
        openai_model=provider["models"][0],
        openai_base_url=provider["base_url"],
        openai_api_key="fake-claude-key",
    )
    payload = {
        "model": settings.openai_model,
        "max_output_tokens": 256,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Return JSON"}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
                    {"type": "input_text", "text": "Read this"},
                ],
            },
        ],
        "text": {"format": {"schema": {"type": "object"}}},
    }
    path, body = prepare_wire_payload(payload, settings)
    request = model_request(payload, settings)
    assert path == "/messages"
    assert body["system"] == "Return JSON"
    assert body["messages"][0]["content"][0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "cG5n",
    }
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert request.full_url == provider["base_url"] + "/messages"
    assert request.get_header("X-api-key") == "fake-claude-key"
    assert request.get_header("Anthropic-version") == "2023-06-01"
    assert not request.has_header("Authorization")
    result = {
        "type": "message",
        "content": [{"type": "text", "text": '{"ok":true}'}],
        "usage": {"input_tokens": 33, "output_tokens": 4, "cache_read_input_tokens": 10},
    }
    assert response_output_text(result) == '{"ok":true}'
    assert response_usage(result) == {"input_tokens": 33, "output_tokens": 4, "cached_tokens": 10}


def test_chat_connection_probe_targets_chat_endpoint():
    provider = next(p for p in MODEL_PROVIDERS if p["id"] == "gemini")
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]}).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return FakeResponse()

    with TestClient(app) as client:
        client.put(
            "/api/settings",
            json={
                "provider": provider["id"],
                "model": provider["models"][0],
                "base_url": provider["base_url"],
                "api_key": "fake-key",
            },
        )
        with patch("grading_agent.app.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.post("/api/settings/test-connection")
    assert result.status_code == 200
    assert seen["url"] == provider["base_url"] + "/chat/completions"
    assert seen["body"]["response_format"]["type"] == "json_schema"
