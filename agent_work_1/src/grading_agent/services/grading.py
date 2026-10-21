from __future__ import annotations

from typing import Any
import math

from ..config import Settings
from ..db import utc_now
from ..repository import get_submission, questions_for_submission, record_token_usage, replace_active_grades
from ..schemas import ScoreState
from .llm_grader import OpenAIGrader
from .math_grader import GradeDecision, grade_with_rules
from .review_policy import apply_review_policy, confidence_threshold
from .grading_executor import question_executor
from .work_locks import submission_work


RULE_MODES = {"exact_text", "choice_set", "numeric", "symbolic"}


def grade_submission(submission_id: str, settings: Settings, pending_only: bool = False) -> dict[str, Any]:
    with submission_work(settings.database_path, submission_id, blocking=False):
        return _grade_submission(submission_id, settings, pending_only)


def _grade_question(question, response, settings, submission_id):
    # Independent questions may run together; a question's stability passes
    # remain sequential and use the original stopping/review rules.
    if question["scoring_mode"] in RULE_MODES:
        routed_question = {**question, 'rule_config': {**question.get('rule_config', {}), 'minimum_ocr_confidence': confidence_threshold(question, settings)}}
        decision = grade_with_rules(routed_question, response)
    elif question["scoring_mode"] == "manual":
        decision = GradeDecision(
            None, float(question["max_score"]), ScoreState.BLOCKED, True, ["manual_scoring_required"]
        )
    elif not response or not str(response.get("raw_text", "")).strip():
        if response and response.get('blank_confirmed'):
            decision = GradeDecision(
                0.0, float(question["max_score"]), ScoreState.REPRODUCIBLE_EXACT, False, ["confirmed_blank"]
            )
        else:
            decision = GradeDecision(
                None, float(question["max_score"]), ScoreState.BLOCKED, True,
                ["blank_not_confirmed" if response else "answer_missing"],
            )
    elif response.get('extraction_source') != 'manual' and (
        response.get('ocr_confidence') is None or not math.isfinite(float(response['ocr_confidence'])) or
        float(response['ocr_confidence']) < 0.5
    ):
        decision = GradeDecision(None, float(question['max_score']), ScoreState.BLOCKED, True, ['extraction_uncertain'])
    else:
        decision = OpenAIGrader(settings).grade(question, str(response["raw_text"]))
    return apply_review_policy(decision, question, response, settings, submission_id)


def _grade_submission(submission_id: str, settings: Settings, pending_only: bool) -> dict[str, Any]:
    started_at = utc_now()
    submission, questions, response_map = questions_for_submission(submission_id)
    output: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    existing = get_submission(submission_id) if pending_only else None
    # A review recommendation does not make an existing numeric score pending.
    # Retry only missing/blocked items; teachers can still request a full
    # regrade explicitly with pending_only=False.
    passed = {g['question_id'] for g in existing['grade_items']
              if g.get('score') is not None or g.get('final_score') is not None} if existing else set()
    pending = [q for q in questions if q['id'] not in passed]
    if not pending and existing:
        return existing
    # The pool is shared process-wide, so two batch workers do not multiply
    # the concurrency limit. No database transaction is held while waiting.
    futures = [question_executor().submit(
        _grade_question, q, response_map.get(q['id']), settings, submission_id,
    ) for q in pending]
    failures = []
    for question, future in zip(pending, futures):
        response = response_map.get(question["id"])
        try:
            decision = future.result()
        except Exception as exc:
            failures.append(exc)
            continue
        if decision.usage:
            usage_total["input_tokens"] += decision.usage["input_tokens"]
            usage_total["output_tokens"] += decision.usage["output_tokens"]
            usage_total["cached_tokens"] += decision.usage["cached_tokens"]
        output.append(
            {
                "question_id": question["id"],
                "response_id": response["id"] if response else None,
                "score": decision.score,
                "max_score": decision.max_score,
                "score_state": str(decision.score_state),
                "needs_review": decision.needs_review,
                "reason_codes": decision.reason_codes,
                "evidence": decision.evidence,
                "grader_kind": decision.grader_kind,
                "grader_version": decision.grader_version,
            }
        )
    expected = {q['id']: (q, response_map.get(q['id'])) for q in questions if q['id'] not in passed}
    if usage_total["input_tokens"] or usage_total["output_tokens"]:
        record_token_usage(submission["assessment_id"], submission_id, "grade", usage_total["input_tokens"], usage_total["output_tokens"], usage_total["cached_tokens"])
    if failures:
        # Drain every future and record known usage before reporting failure.
        # Preserve the original all-or-nothing grade commit.
        raise failures[0]
    replace_active_grades(submission_id, output, partial=pending_only, expected_inputs=expected, started_at=started_at)
    return get_submission(submission_id)  # type: ignore[return-value]
