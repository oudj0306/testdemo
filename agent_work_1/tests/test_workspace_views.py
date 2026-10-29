import io

from fastapi.testclient import TestClient
from PIL import Image


def test_preview_history_and_report(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADING_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from grading_agent.app import app

    with TestClient(app) as c:
        a = c.post("/api/assessments", json={"title": "<script>标题</script>"}).json()
        c.post(
            f"/api/assessments/{a['id']}/questions",
            json={
                "question_no": "1",
                "prompt": "1+1",
                "question_type": "numeric",
                "scoring_mode": "numeric",
                "max_score": 5,
                "answer_key": "2",
                "answer_key_state": "candidate",
            },
        ).json()
        c.post(f"/api/assessments/{a['id']}/publish")
        image = io.BytesIO()
        Image.new("RGB", (1000, 1000), "white").save(image, "PNG")
        s = c.post(
            "/api/submissions",
            data={"assessment_id": a["id"], "student_ref": "学生A"},
            files={"file": ("a.png", image.getvalue(), "image/png")},
        ).json()
        sid = s["id"]
        assert (
            c.get(f"/api/preview/submission/{sid}").headers["content-type"].startswith("image/png")
        )
        assert c.get("/api/preview/submission/missing").status_code == 404
        assert c.get("/api/preview/invalid/missing").status_code == 404
        c.post(
            f"/api/submissions/{sid}/answers",
            json={"answers": [{"question_no": "1", "raw_text": "2"}]},
        )
        g = c.post(f"/api/submissions/{sid}/grade").json()
        assert g["score_summary"]["confirmed_score"] == 5
        assert g["score_summary"]["unresolved_count"] == 0
        assert g["score_summary"]["review_recommended_count"] == 1
        report = c.get(f"/api/assessments/{a['id']}/report").json()
        assert report["average"] == 5
        gid = g["grade_items"][0]["id"]
        c.post(
            f"/api/grade-items/{gid}/review",
            json={"reviewer_ref": "teacher", "final_score": 4, "reason": "过程少一步"},
        )
        assert c.get(f"/api/assessments/{a['id']}/report").json()["average"] == 4
        c.post(
            f"/api/submissions/{sid}/answers",
            json={"answers": [{"question_no": "1", "raw_text": "3"}]},
        )
        assert c.get(f"/api/assessments/{a['id']}/report").json()["average"] is None
        c.post(f"/api/submissions/{sid}/grade")
        history = c.get(f"/api/submissions/{sid}/history").json()
        assert len(history) == 2
        old = next(g for g in history if g["id"] == gid)
        assert old["snapshot"]["response"]["raw_text"] == "2"
        assert old["reviews"][0]["final_score"] == 4
        assert old["superseded_at"]
        html = c.get(f"/api/assessments/{a['id']}/report.html")
        assert "&lt;script&gt;标题&lt;/script&gt;" in html.text
        assert "<script>标题</script>" not in html.text
