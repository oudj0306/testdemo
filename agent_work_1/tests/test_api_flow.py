from __future__ import annotations

import io
import time

from docx import Document
from fastapi.testclient import TestClient
from PIL import Image


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1000, 1400), "white").save(buffer, "PNG")
    return buffer.getvalue()


def docx_bytes() -> bytes:
    buffer = io.BytesIO()
    document = Document()
    document.add_heading("数学试卷", level=1)
    document.add_paragraph("1. 计算 1/4 + 1/4。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "题号"
    table.cell(0, 1).text = "分值"
    document.save(buffer)
    return buffer.getvalue()


def test_complete_rule_grading_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    from grading_agent.app import app

    with TestClient(app) as client:
        assessment = client.post(
            "/api/assessments", json={"title": "单元测试", "subject_codes": ["math"]}
        ).json()
        question = client.post(
            f"/api/assessments/{assessment['id']}/questions",
            json={
                "question_no": "1",
                "prompt": "1/4 + 1/4 = ?",
                "question_type": "auto",
                "scoring_mode": "auto",
                "max_score": 5,
                "answer_key": "1/2",
            },
        )
        assert question.status_code == 200
        assert question.json()["scoring_mode"] == "numeric"
        assert client.post(f"/api/assessments/{assessment['id']}/publish").status_code == 200

        upload = client.post(
            "/api/submissions",
            data={"assessment_id": assessment["id"], "student_ref": "s-001"},
            files={"file": ("answer.png", png_bytes(), "image/png")},
        )
        assert upload.status_code == 200, upload.text
        submission_id = upload.json()["id"]
        answers = client.post(
            f"/api/submissions/{submission_id}/answers",
            json={"answers": [{"question_no": "1", "raw_text": "0.5"}]},
        )
        assert answers.status_code == 200, answers.text
        graded = client.post(f"/api/submissions/{submission_id}/grade")
        assert graded.status_code == 200, graded.text
        assert graded.json()["score_summary"]["confirmed_score"] == 5
        assert graded.json()["status"] == "finalized"


def test_manual_review_and_knowledge_approval(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    from grading_agent.app import app

    with TestClient(app) as client:
        assessment = client.post("/api/demo/bootstrap").json()
        upload = client.post(
            "/api/submissions",
            data={"assessment_id": assessment["id"], "student_ref": "s-002"},
            files={"file": ("answer.png", png_bytes(), "image/png")},
        ).json()
        answer = client.post(
            f"/api/submissions/{upload['id']}/answers",
            json={"answers": [{"question_no": "4", "raw_text": "列式正确，计算完成"}]},
        ).json()
        response = next(item for item in answer["responses"] if item["question_no"] == "4")
        graded = client.post(f"/api/submissions/{upload['id']}/grade").json()
        item = next(item for item in graded["grade_items"] if item["question_no"] == "4")
        reviewed = client.post(
            f"/api/grade-items/{item['id']}/review",
            json={"reviewer_ref": "teacher", "final_score": 6, "reason": "按评分细则人工定分"},
        )
        assert reviewed.status_code == 200
        asset = client.post(
            "/api/knowledge/approve",
            json={
                "response_id": response["id"],
                "redacted_text": "列式正确，计算完成",
                "approved_by": "teacher",
                "metadata": {"quality": "reviewed"},
            },
        )
        assert asset.status_code == 200
        assert client.get("/api/knowledge/search", params={"q": "列式"}).json()


def test_batch_upload_and_reference_document_dedup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    from grading_agent.app import app

    with TestClient(app) as client:
        assessment = client.post("/api/demo/bootstrap").json()
        reference = png_bytes()
        first = client.post(
            f"/api/assessments/{assessment['id']}/documents",
            data={"kind": "answer_key"},
            files=[("files", ("answers.png", reference, "image/png"))],
        )
        second = client.post(
            f"/api/assessments/{assessment['id']}/documents",
            data={"kind": "answer_key"},
            files=[("files", ("answers-copy.png", reference, "image/png"))],
        )
        assert first.status_code == 200, first.text
        assert first.json()[0]["id"] == second.json()[0]["id"]

        batch = client.post(
            "/api/batches",
            data={"assessment_id": assessment["id"], "auto_grade": "true"},
            files=[
                ("files", ("student-a.png", png_bytes(), "image/png")),
                ("files", ("student-b.png", png_bytes(), "image/png")),
            ],
        )
        assert batch.status_code == 202, batch.text
        result = batch.json()
        deadline = time.monotonic() + 10
        while result["status"] == "processing" and time.monotonic() < deadline:
            time.sleep(0.05)
            result = client.get("/api/batches/" + result["id"]).json()
        assert result["success_count"] == 2, result
        assert len(result["items"]) == 2
        assert (
            client.get("/api/batches", params={"assessment_id": assessment["id"]}).json()[0]["id"]
            == result["id"]
        )


def test_docx_question_paper_is_extracted(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    from grading_agent.app import app

    with TestClient(app) as client:
        assessment = client.post(
            "/api/assessments", json={"title": "Word试卷", "subject_codes": ["math"]}
        ).json()
        uploaded = client.post(
            f"/api/assessments/{assessment['id']}/documents",
            data={"kind": "question_paper"},
            files=[
                (
                    "files",
                    (
                        "paper.docx",
                        docx_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
        )
        assert uploaded.status_code == 200, uploaded.text
        assert "1/4 + 1/4" in uploaded.json()[0]["extracted_text"]
        assert uploaded.json()[0]["extraction_state"] == "docx_text"


def test_candidate_answer_never_auto_finalizes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    from grading_agent.app import app

    with TestClient(app) as client:
        assessment = client.post(
            "/api/assessments", json={"title": "候选答案试卷", "subject_codes": ["math"]}
        ).json()
        question = client.post(
            f"/api/assessments/{assessment['id']}/questions",
            json={
                "question_no": "1",
                "prompt": "1+1=?",
                "question_type": "numeric",
                "scoring_mode": "numeric",
                "max_score": 5,
                "answer_key": "2",
                "answer_key_state": "candidate",
            },
        )
        assert question.status_code == 200, question.text
        client.post(f"/api/assessments/{assessment['id']}/publish")
        submission = client.post(
            "/api/submissions",
            data={"assessment_id": assessment["id"], "student_ref": "s-003"},
            files={"file": ("answer.png", png_bytes(), "image/png")},
        ).json()
        client.post(
            f"/api/submissions/{submission['id']}/answers",
            json={"answers": [{"question_no": "1", "raw_text": "2"}]},
        )
        graded = client.post(f"/api/submissions/{submission['id']}/grade").json()
        assert graded["status"] == "finalized"
        assert graded["score_summary"]["current_score"] == 5
        assert graded["score_summary"]["review_recommended_count"] == 1
        assert graded["grade_items"][0]["score_state"] == "CONSENSUS_CANDIDATE"
        assert "provisional_answer_key" in graded["grade_items"][0]["reason_codes"]
