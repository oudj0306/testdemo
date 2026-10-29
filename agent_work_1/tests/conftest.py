import pytest


@pytest.fixture(autouse=True)
def isolate_runtime_settings(monkeypatch, tmp_path):
    from grading_agent import config

    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEW_SAMPLE_RATE", "0")
    monkeypatch.setenv("HYBRID_FALLBACK_ENABLED", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config, "_settings_cache", None)
