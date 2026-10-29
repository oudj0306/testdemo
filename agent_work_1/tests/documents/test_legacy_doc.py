import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_user_doc_upload_preview(monkeypatch, tmp_path):
    source = os.getenv("TEST_LEGACY_DOC")
    if not source:
        pytest.skip("Set TEST_LEGACY_DOC to test a local Word binary file")
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from grading_agent.app import app

    with TestClient(app) as client:
        a = client.post("/api/assessments", json={"title": "综合试卷"}).json()
        assert a["subject_codes"] == []
        r = client.post(
            f"/api/assessments/{a['id']}/documents",
            data={"kind": "question_paper"},
            files={"files": ("paper.doc", Path(source).read_bytes(), "application/msword")},
        )
        assert r.status_code == 200, r.text
        d = r.json()[0]
        assert "count_distinct_primes" in d["extracted_text"]
        assert "OSI" in d["extracted_text"]
        preview = client.get(f"/api/preview/document/{d['id']}")
        assert preview.status_code == 200
        assert "count_distinct_primes" in preview.text
