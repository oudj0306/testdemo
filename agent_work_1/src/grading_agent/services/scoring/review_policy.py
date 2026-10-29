"""Deterministic exception routing, independent of model self-confidence."""

import hashlib
import math

from grading_agent.schemas import ScoreState
from grading_agent.services.recognition.answer_quality import confidence_threshold, parse_confidence


def apply_review_policy(decision, question, response, settings, submission_id):
    # Missing extraction is not evidence of a student's blank answer. Confirmed
    # blanks may score zero, but still obey strict, provisional-key and OCR gates.
    if any(r in {"answer_missing", "blank_not_confirmed"} for r in decision.reason_codes):
        decision.score = None
        decision.score_state = ScoreState.BLOCKED
        decision.needs_review = True
    reasons = []
    if question.get("answer_key_state") != "verified":
        reasons.append("provisional_answer_key")
    if (
        settings.grading_profile == "exam"
        or settings.review_mode == "strict"
        or question.get("rule_config", {}).get("force_review")
    ):
        reasons.append("teacher_review_policy")
    from grading_agent.services.recognition.blank_detection import is_local_blank

    if is_local_blank(response):
        decision.evidence.append("local_blank_verified; not an OCR failure")
    elif response and response.get("extraction_source") == "local_vlm_consistent":
        decision.evidence.append(
            "local_visual_two_transcriptions_agree; not calibrated probability or teacher confirmation"
        )
    elif response and response.get("extraction_source") != "manual":
        confidence = parse_confidence(response.get("ocr_confidence"))
        minimum = confidence_threshold(question, settings)
        if confidence is None or confidence < minimum:
            reasons.append("extraction_uncertain")
    if decision.grader_kind == "llm" and question.get("scoring_mode") not in {
        "exact_text",
        "choice_set",
        "numeric",
        "symbolic",
    }:
        rubric = question.get("rubric", [])
        try:
            valid_rubric = bool(rubric) and all(
                str(r.get("criterion", "")).strip()
                and math.isfinite(float(r["score"]))
                and float(r["score"]) >= 0
                for r in rubric
            )
            # Verified rubrics can contain alternative scoring paths or score caps.
            valid_rubric = valid_rubric and sum(float(r["score"]) for r in rubric) > 0
        except (ValueError, TypeError, KeyError):
            valid_rubric = False
        if not valid_rubric:
            reasons.append("rubric_incomplete")
    # Stable across repeated grading; sampling cannot be avoided by clicking regrade.
    sample = (
        int.from_bytes(
            hashlib.sha256(f"{submission_id}:{question['id']}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    if not decision.needs_review and not reasons and sample < settings.review_sample_rate:
        reasons.append("quality_spot_check")
    if reasons:
        decision.needs_review = True
        decision.reason_codes = list(dict.fromkeys([*decision.reason_codes, *reasons]))
    if (
        not decision.needs_review
        and decision.grader_kind == "llm"
        and decision.score_state != ScoreState.AUTO_SINGLE
    ):
        decision.score_state = ScoreState.AUTO_STABLE
    if question.get("answer_key_state") != "verified" and decision.score is not None:
        decision.score_state = ScoreState.CONSENSUS_CANDIDATE
    decision.evidence.append(
        f"review_policy=v2;mode={settings.review_mode};profile={settings.grading_profile};sample_rate={settings.review_sample_rate};minimum_extraction_confidence={confidence_threshold(question, settings)}"
    )
    return decision
