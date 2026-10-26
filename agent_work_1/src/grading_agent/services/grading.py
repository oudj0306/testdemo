from __future__ import annotations

from typing import Any
import math
import re
import json
from difflib import SequenceMatcher
from contextvars import copy_context

from ..config import Settings
from ..db import utc_now
from ..repository import get_submission, questions_for_submission, record_token_usage, replace_active_grades
from ..schemas import ScoreState
from .llm_grader import OpenAIGrader
from .math_grader import GradeDecision, grade_with_rules
from .review_policy import apply_review_policy, confidence_threshold
from .grading_executor import question_executor
from .work_locks import submission_work
from .telemetry import scope, timed, event


RULE_MODES = {"exact_text", "choice_set", "numeric", "symbolic"}


def grading_question(question):
    if question['scoring_mode'] not in RULE_MODES:return question
    constraints={k:v for k,v in question.get('rule_config',{}).items() if k in {'tolerance','required_form','strict_exact_match','allow_semantic_grading'}}
    if not constraints:return question
    return {**question,'prompt':question['prompt']+'\n判分约束（容差及表达形式要求须遵守）：'+json.dumps(constraints,ensure_ascii=False)}


def grade_submission(submission_id: str, settings: Settings, pending_only: bool = False,
                     prepared=None, expected=None, started_at=None,question_ids=None) -> dict[str, Any]:
    with submission_work(settings.database_path, submission_id, blocking=False):
        with scope(settings, submission_id=submission_id), timed('grading'):
            return _grade_submission(submission_id, settings, pending_only, prepared, expected, started_at,question_ids)


def reliable_text(response):
    return bool(response
            and str(response.get('raw_text', '')).strip()
            and (response.get('extraction_source') in {'manual','local_vlm_consistent'} or (
                response.get('ocr_confidence') is not None
                and math.isfinite(float(response['ocr_confidence']))
                and float(response['ocr_confidence']) >= 0.5)))


def rule_needs_model(question,response,decision):
    if not reliable_text(response):return False
    if decision.score is None:
        return bool(set(decision.reason_codes)&{'answer_parse_failed','equivalent_but_required_form_unverified','non_deterministic_route'})
    if question['scoring_mode']=='choice_set':
        return bool(re.search(r'[^A-Ha-h\s,，、;；]',response['raw_text']))
    if question['scoring_mode']!='exact_text' or decision.score!=0 or question.get('rule_config',{}).get('strict_exact_match'):
        return False
    if question.get('rubric') or question.get('rule_config',{}).get('allow_semantic_grading'):return True
    text=response['raw_text'].strip()
    expected=question.get('answer_key',[])
    expected=expected if isinstance(expected,list) else [expected]
    return any(min(len(text),len(str(answer)))>=2 and any(c.isalpha() for c in text)
               and SequenceMatcher(None,text,str(answer)).ratio()>=.6 for answer in expected)


def requires_model(question,response,settings=None):
    if not reliable_text(response) or question['scoring_mode']=='manual':return False
    if question['scoring_mode'] not in RULE_MODES:return True
    if settings is None:
        from ..config import get_settings
        settings=get_settings()
    routed_question={**question,'rule_config':{**question.get('rule_config',{}),'minimum_ocr_confidence':confidence_threshold(question,settings)}}
    return settings.hybrid_fallback_enabled and bool(settings.openai_api_key) and rule_needs_model(question,response,grade_with_rules(routed_question,response))


def _grade_question(question, response, settings, submission_id, prepared=None):
    # Independent questions may run together; a question's stability passes
    # remain sequential and use the original stopping/review rules.
    if response and response.get('blank_confirmed') and not str(response.get('raw_text') or '').strip():
        decision=GradeDecision(0.0,float(question['max_score']),ScoreState.REPRODUCIBLE_EXACT,False,['confirmed_blank'])
        event('rule_grade')
    elif question["scoring_mode"] in RULE_MODES:
        routed_question = {**question, 'rule_config': {**question.get('rule_config', {}), 'minimum_ocr_confidence': confidence_threshold(question, settings)}}
        decision = grade_with_rules(routed_question, response)
        event('rule_grade')
        if settings.hybrid_fallback_enabled and rule_needs_model(question,response,decision):
            previous=decision.reason_codes
            decision=prepared if prepared is not None else OpenAIGrader(settings).grade(grading_question(question),response['raw_text'])
            decision.reason_codes=list(dict.fromkeys([*decision.reason_codes,'rule_fallback_to_model']))
            decision.evidence.append('rule_uncertainty='+','.join(previous))
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
    elif response.get('extraction_source') not in {'manual','local_vlm_consistent'} and (
        response.get('ocr_confidence') is None or not math.isfinite(float(response['ocr_confidence'])) or
        float(response['ocr_confidence']) < 0.5
    ):
        decision = GradeDecision(None, float(question['max_score']), ScoreState.BLOCKED, True, ['extraction_uncertain'])
    else:
        decision = prepared if prepared is not None else OpenAIGrader(settings).grade(question, str(response["raw_text"]))
    return apply_review_policy(decision, question, response, settings, submission_id)


def _grade_submission(submission_id: str, settings: Settings, pending_only: bool,
                      prepared=None, prior_inputs=None, started_at=None,question_ids=None) -> dict[str, Any]:
    started_at = started_at or utc_now()
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
    if question_ids is not None:
        if not set(question_ids)<={q['id'] for q in questions}:raise ValueError('待判分题目不属于该答卷')
        pending=[q for q in pending if q['id'] in question_ids]
    for q in pending:
        if prior_inputs is not None and prior_inputs.get(q['id']) != (q, response_map.get(q['id'])):
            raise ValueError('批改期间作答或评分标准已修改，请重新批改')
    if not pending and existing:
        return existing
    # The pool is shared process-wide, so two batch workers do not multiply
    # the concurrency limit. No database transaction is held while waiting.
    futures = [question_executor().submit(
        copy_context().run, _grade_question, q, response_map.get(q['id']), settings, submission_id,
        (prepared or {}).get(q['id']),
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
    expected = {q['id']: (q, response_map.get(q['id'])) for q in pending}
    if usage_total["input_tokens"] or usage_total["output_tokens"]:
        record_token_usage(submission["assessment_id"], submission_id, "grade", usage_total["input_tokens"], usage_total["output_tokens"], usage_total["cached_tokens"])
    if failures:
        # Drain every future and record known usage before reporting failure.
        # Preserve the original all-or-nothing grade commit.
        raise failures[0]
    replace_active_grades(submission_id, output, partial=pending_only or question_ids is not None, expected_inputs=expected, started_at=started_at)
    return get_submission(submission_id)  # type: ignore[return-value]
