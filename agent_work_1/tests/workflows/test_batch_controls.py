from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from grading_agent import config
from grading_agent.schemas import QuestionCreate, ScoreState
from grading_agent.services.scoring.decisions import GradeDecision
from grading_agent.services.workflows.batch_control import control_batch
from grading_agent.services.workflows.batch_queue import BatchQueue, enqueue_saved
from grading_agent.storage.db import initialize_database, transaction
from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    create_submission_record,
    get_batch,
    get_submission,
    publish_assessment,
    upsert_response,
)


def fixture(monkeypatch, profile="legacy", students=1):
    initialize_database()
    monkeypatch.setattr(
        config, "_settings_cache", replace(config.get_settings(), grading_profile=profile)
    )
    a = create_assessment("Controlled batch", [])
    qs = []
    submissions = []
    for n in range(2):
        qs.append(
            create_question(
                a["id"],
                QuestionCreate(
                    question_no=str(n + 1),
                    prompt="Explain",
                    question_type="short_answer",
                    scoring_mode="rubric_assisted",
                    max_score=5,
                    answer_key="expected",
                    rubric=[{"criterion": "correct", "score": 5}],
                ).model_dump(mode="json"),
            )
        )
    publish_assessment(a["id"])
    for n in range(students):
        s = create_submission_record(
            a["id"], "synthetic", str(n), "unused.png", "image/png", {}, []
        )
        for q in qs:
            upsert_response(s["id"], q["id"], "answer " + str(n), None, False, "manual")
        submissions.append(s)
    batch = enqueue_saved(a["id"], [s["id"] for s in submissions])
    return a, qs, submissions, batch


def test_pause_stop_resume_endpoints_and_claim_gate(monkeypatch):
    from grading_agent.app import app

    monkeypatch.setattr(BatchQueue, "start", lambda self: None)
    a, qs, subs, b = fixture(monkeypatch)
    queue = BatchQueue(None)
    with TestClient(app) as client:
        assert client.post(f"/api/batches/{b['id']}/pause").json()["status"] == "paused"
        assert queue.claim() is None
        assert client.post(f"/api/batches/{b['id']}/stop").json()["status"] == "stopped"
        assert client.post(f"/api/batches/{b['id']}/resume").json()["status"] == "processing"
        job = queue.claim()
        assert job["batch_id"] == b["id"]
        state = client.post(f"/api/batches/{b['id']}/stop").json()
        assert state["settling"] and state["active_count"] == 1
        assert client.post(f"/api/batches/{b['id']}/resume").status_code == 409
        queue.finish(job, requeue=True)
        assert client.post(f"/api/batches/{b['id']}/resume").status_code == 200
        assert client.post(f"/api/batches/{b['id']}/unknown").status_code == 409
        assert client.post("/api/batches/missing/pause").status_code == 404


@pytest.mark.parametrize("action", ["pause", "stop"])
def test_inflight_question_is_saved_then_resume_skips_it(monkeypatch, action):
    a, qs, subs, b = fixture(monkeypatch)
    sid = subs[0]["id"]
    calls = []

    def grade(self, q, answer):
        calls.append(q["id"])
        if len(calls) == 1:
            control_batch(b["id"], action)
        return GradeDecision(
            5,
            5,
            ScoreState.AUTO_SINGLE,
            False,
            ["ok"],
            ["answer"],
            grader_kind="llm",
            usage={"input_tokens": 10, "output_tokens": 2, "cached_tokens": 0},
        )

    monkeypatch.setattr(
        "grading_agent.services.scoring.submission_grader.OpenAIGrader.grade", grade
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(
            "grading_agent.services.scoring.submission_grader.question_executor", lambda: pool
        )
        queue = BatchQueue(None)
        queue.process(queue.claim())
        saved = get_submission(sid)
        assert len(calls) == 1 and len(saved["grade_items"]) == 1
        first = saved["grade_items"][0]
        assert get_batch(b["id"])["status"] == ("paused" if action == "pause" else "stopped")
        assert get_batch(b["id"])["saved_question_count"] == 1
        assert get_batch(b["id"])["items"][0]["status"] == "queued"
        assert queue.claim() is None
        control_batch(b["id"], "resume")
        queue.process(queue.claim())
    final = get_submission(sid)
    assert len(calls) == 2 and len(set(calls)) == 2 and len(final["grade_items"]) == 2
    assert (
        next(g for g in final["grade_items"] if g["question_id"] == first["question_id"])["id"]
        == first["id"]
    )
    assert get_batch(b["id"])["status"] == "completed"
    with transaction() as db:
        assert (
            db.execute(
                "SELECT SUM(input_tokens) FROM token_usage WHERE submission_id=? AND stage='grade'",
                (sid,),
            ).fetchone()[0]
            == 20
        )


def test_daily_bulk_saves_current_question_for_all_students_before_pause(monkeypatch):
    a, qs, subs, b = fixture(monkeypatch, profile="daily", students=2)
    calls = []

    def bulk(q, items, settings):
        calls.append(q["id"])
        if len(calls) == 1:
            control_batch(b["id"], "pause")
        return {
            i["answer_id"]: GradeDecision(
                5, 5, ScoreState.AUTO_SINGLE, False, ["ok"], ["answer"], grader_kind="llm"
            )
            for i in items
        }

    monkeypatch.setattr(
        "grading_agent.services.scoring.submission_grader.requires_model", lambda *args: True
    )
    monkeypatch.setattr("grading_agent.services.scoring.bulk_grader.grade_daily_items", bulk)
    queue = BatchQueue(None)
    queue.process_daily(queue.claim()["daily_group"])
    assert len(calls) == 1
    first_ids = {s["id"]: get_submission(s["id"])["grade_items"][0]["id"] for s in subs}
    assert all(len(get_submission(s["id"])["grade_items"]) == 1 for s in subs)
    assert get_batch(b["id"])["saved_question_count"] == 2
    control_batch(b["id"], "resume")
    queue.process_daily(queue.claim()["daily_group"])
    assert len(calls) == 2 and len(set(calls)) == 2
    assert get_batch(b["id"])["status"] == "completed"
    assert all(
        first_ids[s["id"]] in {g["id"] for g in get_submission(s["id"])["grade_items"]}
        for s in subs
    )


def test_stopped_batch_survives_restart_and_other_batch_is_claimable(monkeypatch):
    a, qs, subs, b = fixture(monkeypatch)
    control_batch(b["id"], "stop")
    queue = BatchQueue(None)
    monkeypatch.setattr(
        "grading_agent.services.workflows.batch_queue.Thread.start", lambda self: None
    )
    queue.start()
    assert get_batch(b["id"])["status"] == "stopped" and queue.claim() is None
    a2, qs2, subs2, b2 = fixture(monkeypatch)
    assert queue.claim()["batch_id"] == b2["id"]
    assert get_batch(b["id"])["items"][0]["status"] == "queued"
