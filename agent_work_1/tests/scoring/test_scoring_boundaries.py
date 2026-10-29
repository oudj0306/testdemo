"""异常输入不能触发付费请求、伪造零分或中断整份评分。"""

from dataclasses import replace

import pytest

from grading_agent.config import get_settings
from grading_agent.services.recognition.answer_quality import has_reliable_text, parse_confidence
from grading_agent.services.scoring import rule_grader
from grading_agent.services.scoring.bulk_grader import grade_daily_items
from grading_agent.services.scoring.llm_grader import OpenAIGrader
from grading_agent.services.scoring.rule_grader import grade_with_rules
from grading_agent.services.scoring.submission_grader import _grade_question


@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf"), -0.1, 1.1, True, []])
def test_invalid_confidence_is_unknown(value):
    assert parse_confidence(value) is None
    assert not has_reliable_text({"raw_text": "存储器", "ocr_confidence": value})


@pytest.mark.parametrize("source", ["manual", "api_vision_fallback", "local_vlm_consistent"])
def test_adopted_text_does_not_require_model_self_confidence(source):
    assert has_reliable_text({"raw_text": "存储器", "extraction_source": source})
    assert not has_reliable_text({"raw_text": "", "extraction_source": source})


def numeric_question(**overrides):
    return {
        "id": "q",
        "scoring_mode": "numeric",
        "answer_key": "2",
        "max_score": 5,
        "rule_config": {},
        "answer_key_state": "verified",
        **overrides,
    }


@pytest.mark.parametrize("answer", ["NaN", "Infinity", "-Infinity", "1/Infinity", "NaN%"])
def test_non_finite_answer_remains_unscored(answer):
    result = grade_with_rules(numeric_question(), {"raw_text": answer})
    assert result.score is None
    assert "answer_parse_failed" in result.reason_codes


@pytest.mark.parametrize("tolerance", ["Infinity", "NaN", "-1"])
def test_invalid_tolerance_cannot_accept_a_wrong_answer(tolerance):
    result = grade_with_rules(
        numeric_question(rule_config={"tolerance": tolerance}), {"raw_text": "999"}
    )
    assert result.score is None


def test_corrupt_ocr_confidence_blocks_one_question_without_crashing():
    settings = replace(get_settings(), grading_profile="daily", review_sample_rate=0)
    result = _grade_question(
        numeric_question(), {"raw_text": "2", "ocr_confidence": "invalid"}, settings, "s"
    )
    assert result.score is None
    assert result.needs_review


@pytest.mark.parametrize(
    "items",
    [
        None,
        [],
        [{"answer_id": []}],
        [{"answer_id": "", "answer": "x"}],
        [{"answer_id": "a", "answer": 2}],
        ["x"],
        [{"answer_id": "a", "answer": "x"}, {"answer_id": "a", "answer": "y"}],
    ],
)
def test_invalid_batch_is_rejected_before_request(monkeypatch, items):
    monkeypatch.setattr(
        "grading_agent.services.scoring.bulk_grader._request",
        lambda *args: pytest.fail("invalid input must not call API"),
    )
    with pytest.raises(ValueError):
        grade_daily_items(numeric_question(), items, get_settings())


def test_symbolic_difference_is_simplified_once(monkeypatch):
    calls = []
    original = rule_grader.simplify

    def counted(expression):
        calls.append(expression)
        return original(expression)

    monkeypatch.setattr(rule_grader, "simplify", counted)
    result = grade_with_rules(
        numeric_question(scoring_mode="symbolic", answer_key="2*(x+1)"), {"raw_text": "2x+2"}
    )
    assert result.score == 5
    assert len(calls) == 1


@pytest.mark.parametrize("attempts", [0, 3, -1, True, 1.5])
def test_daily_request_budget_is_bounded(monkeypatch, attempts):
    monkeypatch.setattr(
        OpenAIGrader, "_single_pass", lambda *args: pytest.fail("invalid retry budget")
    )
    with pytest.raises(ValueError):
        OpenAIGrader(get_settings()).grade_daily(numeric_question(), "answer", attempts=attempts)
