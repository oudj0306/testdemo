"""Regression coverage for unsafe workflow edges found during the UI audit."""

from dataclasses import replace

import pytest

from grading_agent.config import get_settings
from grading_agent.db import initialize_database, transaction
from grading_agent.repository import (
    create_assessment,
    create_question,
    create_submission_record,
    delete_submission,
    get_submission,
    publish_assessment,
    record_token_usage,
    token_usage_for_assessment,
    upsert_response,
)
from grading_agent.schemas import ScoreState
from grading_agent.services.scoring.decisions import GradeDecision
from grading_agent.services.scoring.llm_grader import OpenAIGrader
from grading_agent.services.scoring.review_policy import apply_review_policy
from grading_agent.services.scoring.rule_grader import grade_with_rules
from grading_agent.services.scoring.submission_grader import grade_submission
from grading_agent.services.workflows.batch_queue import enqueue_saved
from grading_agent.workspace_views import report


def saved_paper(mode="exact_text", key_state="verified"):
    initialize_database()
    assessment = create_assessment("安全回归测试", [])
    question = create_question(
        assessment["id"],
        {
            "question_no": "1",
            "prompt": "请作答",
            "question_type": "short_answer",
            "capture_mode": "short_handwriting",
            "answer_form": "text",
            "scoring_mode": mode,
            "max_score": 5,
            "answer_key": "B",
            "answer_key_state": key_state,
            "rubric": [{"criterion": "答案完整", "score": 5}],
            "rule_config": {},
        },
    )
    publish_assessment(assessment["id"])
    submission = create_submission_record(
        assessment["id"],
        "student",
        "audit-hash",
        "unused-original.png",
        "image/png",
        {},
        [],
        status="uploaded",
    )
    return assessment, question, submission


@pytest.mark.parametrize("mode", ["exact_text", "rubric_assisted"])
@pytest.mark.parametrize("kind", ["missing", "unconfirmed_blank"])
def test_missing_extraction_never_becomes_final_zero(monkeypatch, mode, kind):
    assessment, question, submission = saved_paper(mode)
    if kind == "unconfirmed_blank":
        upsert_response(submission["id"], question["id"], "", 0.1, False, "openai_vision")
    monkeypatch.setattr(
        OpenAIGrader, "grade", lambda *args: pytest.fail("missing answers must not call the model")
    )
    result = grade_submission(submission["id"], replace(get_settings(), review_sample_rate=0))
    grade = result["grade_items"][0]
    assert grade["score"] is None
    assert grade["score_state"] == "BLOCKED"
    assert grade["needs_review"]
    assert not grade["bulk_review_eligible"]
    assert result["status"] == "review_required"
    assert result["score_summary"]["unresolved_count"] == 1
    assert report(assessment["id"])["students"][0]["final_score"] is None


@pytest.mark.parametrize("mode", ["exact_text", "rubric_assisted"])
def test_explicit_confirmed_blank_can_score_zero_without_model(monkeypatch, mode):
    assessment, question, submission = saved_paper(mode)
    upsert_response(submission["id"], question["id"], "", None, True, "manual")
    monkeypatch.setattr(
        OpenAIGrader, "grade", lambda *args: pytest.fail("confirmed blanks must not call the model")
    )
    result = grade_submission(submission["id"], replace(get_settings(), review_sample_rate=0))
    grade = result["grade_items"][0]
    assert grade["score"] == 0
    assert not grade["needs_review"]
    assert "confirmed_blank" in grade["reason_codes"]
    assert result["status"] == "finalized"
    assert report(assessment["id"])["students"][0]["final_score"] == 0


@pytest.mark.parametrize(
    "gate, reason",
    [
        ("strict", "teacher_review_policy"),
        ("candidate", "provisional_answer_key"),
        ("force_review", "teacher_review_policy"),
        ("low_confidence", "extraction_uncertain"),
    ],
)
def test_confirmed_blank_does_not_bypass_review_policy(gate, reason):
    question = {
        "id": "q",
        "scoring_mode": "exact_text",
        "answer_key": "B",
        "max_score": 5,
        "answer_key_state": "verified",
        "rule_config": {},
    }
    response = {"raw_text": "", "blank_confirmed": True, "extraction_source": "manual"}
    settings = replace(get_settings(), review_sample_rate=0)
    if gate == "strict":
        settings = replace(settings, review_mode="strict")
    elif gate == "candidate":
        question["answer_key_state"] = "candidate"
    elif gate == "force_review":
        question["rule_config"]["force_review"] = True
    else:
        response.update(extraction_source="openai_vision", ocr_confidence=0.1)
    decision = apply_review_policy(
        grade_with_rules(question, response), question, response, settings, "student"
    )
    assert decision.score == 0
    assert decision.needs_review
    assert reason in decision.reason_codes


def test_policy_defensively_rejects_legacy_missing_zero():
    question = {"id": "q", "answer_key_state": "verified", "rule_config": {}}
    decision = GradeDecision(0, 5, ScoreState.REPRODUCIBLE_EXACT, False, ["answer_missing"])
    apply_review_policy(
        decision, question, None, replace(get_settings(), review_sample_rate=0), "student"
    )
    assert decision.score is None
    assert decision.score_state == ScoreState.BLOCKED
    assert decision.needs_review


def test_delete_submission_preserves_usage_totals_without_foreign_key_failure():
    assessment, _, submission = saved_paper()
    record_token_usage(assessment["id"], submission["id"], "extract", 100, 12, 30)
    record_token_usage(assessment["id"], submission["id"], "grade", 50, 9, 10)
    before = token_usage_for_assessment(assessment["id"])
    delete_submission(submission["id"])
    assert get_submission(submission["id"]) is None
    after = token_usage_for_assessment(assessment["id"])
    assert after["total"] == before["total"] == {"input": 150, "output": 21, "cached": 40}
    assert after["stages"] == before["stages"]
    assert after["submissions"] == []
    with transaction() as db:
        rows = db.execute("SELECT submission_id FROM token_usage").fetchall()
        assert len(rows) == 2
        assert all(row["submission_id"] is None for row in rows)


@pytest.mark.parametrize("status", ["queued", "extracting", "grading"])
def test_cannot_delete_active_submission_or_orphan_its_job(status):
    assessment, _, submission = saved_paper()
    batch = enqueue_saved(assessment["id"], [submission["id"]])
    record_token_usage(assessment["id"], submission["id"], "extract", 100, 12)
    with transaction() as db:
        db.execute(
            "UPDATE grading_batch_items SET status=? WHERE batch_id=?", (status, batch["id"])
        )
    with pytest.raises(ValueError, match="排队或处理中"):
        delete_submission(submission["id"])
    assert get_submission(submission["id"]) is not None
    with transaction() as db:
        row = db.execute(
            "SELECT submission_id,status FROM grading_batch_items WHERE batch_id=?", (batch["id"],)
        ).fetchone()
        assert row["submission_id"] == submission["id"]
        assert row["status"] == status
        assert db.execute("SELECT COUNT(*) FROM batch_jobs").fetchone()[0] == 1
        assert db.execute("SELECT submission_id FROM token_usage").fetchone()[0] == submission["id"]


def test_finished_batch_does_not_block_deletion():
    assessment, _, submission = saved_paper()
    batch = enqueue_saved(assessment["id"], [submission["id"]])
    with transaction() as db:
        db.execute(
            "UPDATE grading_batch_items SET status='failed' WHERE batch_id=?", (batch["id"],)
        )
    delete_submission(submission["id"])
    assert get_submission(submission["id"]) is None
    with transaction() as db:
        row = db.execute(
            "SELECT submission_id,status FROM grading_batch_items WHERE batch_id=?", (batch["id"],)
        ).fetchone()
        assert row["submission_id"] is None
        assert row["status"] == "failed"
