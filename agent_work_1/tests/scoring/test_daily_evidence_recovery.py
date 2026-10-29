from dataclasses import replace

import pytest

from grading_agent.config import get_settings
from grading_agent.services.scoring.bulk_grader import grade_daily_items
from grading_agent.services.scoring.llm_grader import LLMResult, OpenAIGrader, daily_decision


def settings():
    return replace(get_settings(), openai_api_key="test", grading_profile="daily")


def question():
    return {"id": "q", "max_score": 5, "prompt": "编程题", "answer_key": "程序", "rubric": []}


def test_single_valid_score_survives_reformatted_code_evidence_without_retry(monkeypatch):
    from grading_agent.storage.db import initialize_database

    initialize_database()
    calls = []

    def single(*args):
        calls.append(1)
        return LLMResult(3, ["partial"], ["print ( x )"], False, 100, 0, 20, "request-one")

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    result = grade_daily_items(
        question(), [{"answer_id": "one", "answer": "print(x)"}], settings()
    )["one"]
    assert len(calls) == 1 and result.score == 3 and result.needs_review
    assert "grading_evidence_unverified" in result.reason_codes
    assert "print ( x )" not in result.evidence
    assert "request_id=request-one" in result.evidence


def test_bulk_cross_student_evidence_still_rejected():
    with pytest.raises(ValueError, match="Evidence"):
        daily_decision(
            question(), "student one", 3, ["partial"], ["student two"], False, settings()
        )


def test_out_of_range_score_still_blocked_and_actual_reason_saved(monkeypatch):
    calls = []

    def single(*args):
        calls.append(1)
        return LLMResult(99, ["correct"], ["answer"], False, 100, 0, 20)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    result = OpenAIGrader(settings()).grade_daily(question(), "answer")
    assert len(calls) == 2 and result.score is None
    assert any("Invalid score" in e for e in result.evidence)
    assert result.usage["input_tokens"] == 200


def test_response_failure_no_longer_loses_reason(monkeypatch):
    monkeypatch.setattr(
        OpenAIGrader,
        "_single_pass",
        lambda *a: (_ for _ in ()).throw(RuntimeError("missing output_text")),
    )
    result = OpenAIGrader(settings()).grade_daily(question(), "answer")
    assert result.score is None and len(result.evidence) == 2
    assert all("missing output_text" in e for e in result.evidence)
