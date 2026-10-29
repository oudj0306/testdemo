import io
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from grading_agent.services.recognition.ocr import runtime as ocr_runtime


@pytest.fixture(autouse=True)
def isolate_ocr_cache(monkeypatch):
    for name, value in (
        ("_engines", {}),
        ("_errors", {}),
        ("_initialization_errors", set()),
        ("_active_device", None),
        ("_compatibility_warning", None),
    ):
        monkeypatch.setattr(ocr_runtime, name, value)


def test_gpu_engine_reused_and_minimum_confidence(monkeypatch):
    monkeypatch.setattr(ocr_runtime, "_engines", {})
    monkeypatch.setattr(ocr_runtime, "_errors", {})
    monkeypatch.setattr(ocr_runtime, "_active_device", None)
    devices = []

    class Engine:
        def __init__(self, **kwargs):
            devices.append(kwargs["device"])

        def predict(self, path):
            return [{"res": {"rec_texts": ["清楚", "模糊"], "rec_scores": [0.99, 0.4]}}]

    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
        ),
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Engine))
    for _ in range(2):
        text, confidence, source = ocr_runtime.recognize("test.png", "gpu:0")
        assert confidence == 0.4
        assert source == "paddleocr_gpu:0"
    assert devices == ["gpu:0"]


def test_forced_gpu_does_not_silently_use_cpu(monkeypatch):
    monkeypatch.setattr(ocr_runtime, "_engines", {})
    monkeypatch.setattr(ocr_runtime, "_errors", {})
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(is_compiled_with_cuda=lambda: False))
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=None))
    assert ocr_runtime.recognize("test.png", "gpu:0")[2] == "paddleocr_failed"
    assert "gpu:0" in ocr_runtime._errors


def test_cpu_uses_compatible_backend_instead_of_broken_onednn_path(monkeypatch):
    class Engine:
        def __init__(self, **kwargs):
            if kwargs.get("enable_mkldnn", True):
                raise RuntimeError("PIR oneDNN failure")

        def predict(self, path):
            return [{"res": {"rec_texts": ["存储器"], "rec_scores": [0.96]}}]

    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(is_compiled_with_cuda=lambda: False))
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Engine))
    assert ocr_runtime.recognize("answer.png", "cpu") == ("存储器", 0.96, "paddleocr_cpu")


def test_bad_image_does_not_disable_following_images(monkeypatch):
    class Engine:
        def predict(self, path):
            if path == "bad.png":
                raise ValueError("bad image")
            return [{"res": {"rec_texts": ["12345"], "rec_scores": [0.99]}}]

    ocr_runtime._engines["gpu:0"] = (Engine(), "gpu:0")
    assert ocr_runtime.recognize("bad.png", "gpu:0")[2] == "paddleocr_failed"
    assert ocr_runtime.recognize("good.png", "gpu:0")[0] == "12345"
    assert not ocr_runtime._errors


def test_auto_and_gpu_share_model(monkeypatch):
    devices = []

    class Engine:
        def __init__(self, **kwargs):
            devices.append(kwargs["device"])

        def predict(self, path):
            return []

    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
        ),
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Engine))
    ocr_runtime.recognize("sample.png", "auto")
    ocr_runtime.recognize("sample.png", "gpu:0")
    assert devices == ["gpu:0"]


def test_diagnostic_can_retry_initialization(monkeypatch):
    ocr_runtime._initialization_errors.add("gpu:0")
    ocr_runtime._errors["gpu:0"] = "old failure"

    class Engine:
        def __init__(self, **kwargs):
            pass

        def predict(self, path):
            return []

    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
        ),
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Engine))
    assert ocr_runtime.recognize("sample.png", "gpu:0")[2] == "paddleocr_failed"
    assert (
        ocr_runtime.recognize("sample.png", "gpu:0", retry_initialization=True)[2]
        == "paddleocr_gpu:0"
    )
    assert not ocr_runtime._initialization_errors
    assert not ocr_runtime._errors


@pytest.mark.parametrize("success", [True, False])
def test_local_diagnostic_cleanup(monkeypatch, tmp_path, success):
    from grading_agent.services.recognition.ocr import diagnostics as ocr_diagnostics

    monkeypatch.setattr(
        ocr_diagnostics,
        "recognize",
        lambda *args, **kwargs: ("12345" if success else "", 0.99, "paddleocr_gpu:0"),
    )
    monkeypatch.setattr(ocr_diagnostics, "runtime_status", lambda: {"active_device": "gpu:0"})
    result = ocr_diagnostics.check_ocr("gpu:0", tmp_path)
    assert result["ok"] == success
    assert len(result["durations_seconds"]) == (2 if success else 1)
    assert not list(tmp_path.iterdir())


def test_diagnostic_endpoint_requires_saved_ocr_setting(monkeypatch):
    from fastapi.testclient import TestClient

    from grading_agent import config
    from grading_agent.app import app

    monkeypatch.setattr(
        config, "_settings_cache", replace(config.get_settings(), ocr_provider="none")
    )
    with TestClient(app) as client:
        assert client.post("/api/runtime/ocr-check").status_code == 400


def test_diagnostic_endpoint_uses_configured_device(monkeypatch):
    from fastapi.testclient import TestClient

    from grading_agent import config
    from grading_agent.app import app
    from grading_agent.services.recognition.ocr import diagnostics as ocr_diagnostics

    monkeypatch.setattr(
        config,
        "_settings_cache",
        replace(config.get_settings(), ocr_provider="paddleocr", ocr_device="gpu:0"),
    )
    calls = []

    def check(device, directory):
        calls.append(device)
        return {"ok": True, "device": device}

    monkeypatch.setattr(ocr_diagnostics, "check_ocr", check)
    with TestClient(app) as client:
        assert client.post("/api/runtime/ocr-check").json() == {"ok": True, "device": "gpu:0"}
    assert calls == ["gpu:0"]


@pytest.mark.parametrize(
    "confidence,kind,layout,expected_local",
    [
        (0.99, "short_answer", "answer_list", True),
        (0.5, "short_answer", "answer_list", False),
        (0.99, "calculation", "answer_list", False),
        (0.99, "short_answer", "on_paper", False),
    ],
)
def test_text_route_and_visual_fallback(monkeypatch, confidence, kind, layout, expected_local):
    from fastapi.testclient import TestClient
    from PIL import Image

    from grading_agent import app as module
    from grading_agent import config

    settings = replace(
        config.get_settings(), openai_api_key="test", ocr_provider="paddleocr", local_ocr_first=True
    )
    monkeypatch.setattr(config, "_settings_cache", settings)
    monkeypatch.setattr(
        module.OCRAdapter, "extract", lambda *args: ("1. 学生作答", confidence, "paddleocr_gpu:0")
    )
    calls = []

    def extract(document, questions, settings, local_text=None, answer_layout="auto"):
        calls.append(local_text)
        return [
            {
                "question_no": "1",
                "raw_text": "学生作答",
                "confidence": 1.0,
                "blank_confirmed": False,
            }
        ], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    monkeypatch.setattr(module, "extract_answers_with_model", extract)
    with TestClient(module.app) as client:
        a = client.post("/api/assessments", json={"title": "路由测试"}).json()
        client.post(
            f"/api/assessments/{a['id']}/questions",
            json={
                "question_no": "1",
                "prompt": "解释",
                "question_type": kind,
                "scoring_mode": "rubric_assisted",
                "max_score": 5,
                "answer_key": "答案",
            },
        )
        client.post(f"/api/assessments/{a['id']}/publish")
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100), "white").save(buffer, "PNG")
        s = client.post(
            "/api/submissions",
            data={"assessment_id": a["id"], "student_ref": "s", "answer_layout": layout},
            files={"file": ("s.png", buffer.getvalue(), "image/png")},
        )
        assert s.status_code == 200, s.text
        assert (calls[0] is not None) == expected_local
        response = s.json()["responses"][0]
        assert response["extraction_source"] == (
            "local_ocr_model_mapping" if expected_local else "openai_vision"
        )
        assert response["ocr_confidence"] == (confidence if expected_local else 1.0)
