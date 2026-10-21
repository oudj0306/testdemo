from grading_agent.schemas import ScoreState
from grading_agent.services.math_grader import grade_with_rules
from grading_agent.services.question_classifier import infer_question_route


def question(mode: str, key: object, **rules: object) -> dict[str, object]:
    return {"scoring_mode": mode, "answer_key": key, "max_score": 5, "rule_config": rules}


def test_numeric_fraction_matches_decimal() -> None:
    decision = grade_with_rules(question("numeric", "0.5"), {"raw_text": "1/2"})
    assert decision.score == 5
    assert decision.score_state == ScoreState.REPRODUCIBLE_EXACT


def test_symbolic_equivalence() -> None:
    decision = grade_with_rules(question("symbolic", "2*(x+1)"), {"raw_text": "2x+2"})
    assert decision.score == 5


def test_low_ocr_confidence_is_not_scored() -> None:
    decision = grade_with_rules(
        question("exact_text", "B", minimum_ocr_confidence=0.9),
        {"raw_text": "B", "ocr_confidence": 0.5},
    )
    assert decision.score is None
    assert decision.needs_review


def test_unconfirmed_blank_requires_review() -> None:
    decision = grade_with_rules(question("exact_text", "B"), {"raw_text": "", "blank_confirmed": False})
    assert decision.score is None
    assert decision.needs_review
    assert "blank_not_confirmed" in decision.reason_codes


def test_auto_route_uses_rubric_for_process_question() -> None:
    route = infer_question_route("写出计算过程", "42", [{"criterion": "列式", "score": 2}])
    assert route == ("calculation", "rubric_assisted")
