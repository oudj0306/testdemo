import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grading_agent.app import app, store_submission
from grading_agent.services.workflows.batch_queue import BatchQueue, enqueue_saved
from grading_agent.storage.db import initialize_database, transaction
from grading_agent.storage.repository import (
    bulk_review,
    create_assessment,
    create_question,
    create_review,
    get_submission,
    publish_assessment,
    replace_active_grades,
    upsert_response,
)
from grading_agent.web.workspace_views import history, report


def paper():
    initialize_database()
    a = create_assessment("工作流测试", [])
    create_question(
        a["id"],
        dict(
            question_no="1",
            prompt="1+1",
            question_type="numeric",
            capture_mode="short_handwriting",
            answer_form="text",
            scoring_mode="numeric",
            max_score=5,
            answer_key="2",
            rule_config={},
            rubric=[],
        ),
    )
    return publish_assessment(a["id"])


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(stream, "PNG")
    return stream.getvalue()


def saved(a):
    return store_submission(a, "student", image_bytes(), "image/png")


def reviewed_fixture():
    a = paper()
    s = saved(a)
    q = a["questions"][0]
    r = upsert_response(s["id"], q["id"], "2", None, False, "manual")
    replace_active_grades(
        s["id"],
        [
            dict(
                question_id=q["id"],
                response_id=r["id"],
                score=5,
                max_score=5,
                score_state="REPRODUCIBLE_EXACT",
                needs_review=True,
                reason_codes=["numeric_match", "quality_spot_check"],
                evidence=["2=2"],
                grader_kind="rules",
                grader_version="test",
            )
        ],
    )
    return a, s, get_submission(s["id"])["grade_items"][0]


def test_upload_is_save_only_and_rename_deduplicated(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("upload must not extract or grade")

    monkeypatch.setattr("grading_agent.app.persist_submission", forbidden)
    monkeypatch.setattr("grading_agent.app.ingest_document", forbidden)
    with TestClient(app) as client:
        a = paper()
        result = client.post(
            "/api/batches",
            data={"assessment_id": a["id"]},
            files=[
                ("files", ("a.png", image_bytes(), "image/png")),
                ("files", ("renamed.png", image_bytes(), "image/png")),
            ],
        )
        assert result.status_code == 202, result.text
        assert result.json()["duplicate_count"] == 1
        assert result.json()["status"] == "completed"
        rows = client.get("/api/submissions").json()
        assert len(rows) == 1 and rows[0]["status"] == "uploaded"
        s = get_submission(rows[0]["id"])
        assert not s["pages"] and not s["responses"] and not s["grade_items"]
        with transaction() as db:
            assert db.execute("SELECT COUNT(*) FROM batch_jobs").fetchone()[0] == 0


def test_parallel_dedup_and_assessment_scope():
    a = paper()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: saved(a), range(8)))
    assert len({s["id"] for s in results}) == 1
    assert sum(s.get("duplicate", False) for s in results) == 7
    assert saved(paper())["id"] != results[0]["id"]


def test_saved_queue_avoids_duplicate_jobs_and_prepares_once(monkeypatch):
    a = paper()
    s = saved(a)
    enqueue_saved(a["id"], [])
    with pytest.raises(ValueError):
        enqueue_saved(a["id"], [])
    calls = []

    def prepare(sid):
        calls.append(sid)
        upsert_response(sid, a["questions"][0]["id"], "2", None, False, "manual")
        return get_submission(sid)

    queue = BatchQueue(lambda *args: pytest.fail("must use saved original"), prepare)
    queue.process(queue.claim())
    assert calls == [s["id"]]
    assert get_submission(s["id"])["status"] == "finalized"
    with pytest.raises(ValueError):
        enqueue_saved(a["id"], [])


def test_repeat_review_latest_score_and_revision_conflict():
    a, s, g = reviewed_fixture()
    first = create_review(g["id"], "teacher", 4, "first", None, True)
    second = create_review(g["id"], "teacher", 3, "correction", first["id"], True)
    with pytest.raises(ValueError, match="刷新"):
        create_review(g["id"], "other", 5, "stale", first["id"], True)
    create_review(g["id"], "teacher", 2, "third", second["id"], True)
    assert report(a["id"])["students"][0]["final_score"] == 2
    assert len(history(s["id"])[0]["reviews"]) == 3
    assert get_submission(s["id"])["grade_items"][0]["review_count"] == 3


def test_bulk_review_atomic_reject_and_confirm():
    a, s, g = reviewed_fixture()
    assert g["bulk_review_eligible"]
    with pytest.raises(ValueError):
        bulk_review([g["id"], "missing"], "teacher", "checked")
    assert get_submission(s["id"])["grade_items"][0]["review_count"] == 0
    bulk_review([g["id"]], "teacher", "checked")
    assert report(a["id"])["students"][0]["final_score"] == 5
    with pytest.raises(ValueError):
        bulk_review([g["id"]], "teacher", "stale click")
    assert get_submission(s["id"])["grade_items"][0]["review_count"] == 1


def test_missing_grade_does_not_finalize_after_review():
    a, s, g = reviewed_fixture()
    create_question(
        a["id"],
        dict(
            question_no="2",
            prompt="2+2",
            question_type="numeric",
            capture_mode="short_handwriting",
            answer_form="text",
            scoring_mode="numeric",
            max_score=5,
            answer_key="4",
        ),
    )
    create_review(g["id"], "teacher", 5, "checked")
    assert get_submission(s["id"])["status"] != "finalized"
    assert report(a["id"])["students"][0]["final_score"] is None


@pytest.mark.parametrize(
    "reason",
    [
        "score_disagreement",
        "model_requested_review",
        "extraction_uncertain",
        "provisional_answer_key",
        "grading_evidence_missing",
    ],
)
def test_bulk_rejects_anomalies(reason):
    from grading_agent.storage.db import encode_json

    a, s, g = reviewed_fixture()
    with transaction() as db:
        db.execute(
            "UPDATE grade_items SET reason_codes=? WHERE id=?", (encode_json([reason]), g["id"])
        )
    with pytest.raises(ValueError):
        bulk_review([g["id"]], "teacher", "checked")
    assert get_submission(s["id"])["grade_items"][0]["review_count"] == 0


def test_settings_threshold_and_optional_runtime_preserve_strategy():
    with TestClient(app) as client:
        payload = {
            "model": "test-model",
            "base_url": "https://example.invalid/v1",
            "review_confidence_threshold": 0.8,
            "review_sample_rate": 0.1,
            "ocr_provider": "none",
        }
        response = client.put("/api/settings", json=payload)
        assert response.status_code == 200
        updated = client.put(
            "/api/settings",
            json={
                "model": "test-model",
                "base_url": "https://example.invalid/v1",
                "ocr_provider": "none",
            },
        ).json()
        assert updated["review_confidence_threshold"] == 0.8
        assert updated["review_sample_rate"] == 0.1
