from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from sympy import Symbol, simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..schemas import ScoreState


SAFE_EXPRESSION = re.compile(r"^[0-9A-Za-z_+\-*/^().,=\s]+$")
SAFE_IDENTIFIERS = {"x", "y", "z", "a", "b", "c", "n", "t"}
TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)


@dataclass
class GradeDecision:
    score: float | None
    max_score: float
    score_state: ScoreState
    needs_review: bool
    reason_codes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    grader_kind: str = "rules"
    grader_version: str = "math-v1"
    usage: dict[str, int] | None = None


def normalize_text(value: str) -> str:
    return "".join(value.strip().upper().split())


def normalize_choice_set(value: str) -> set[str]:
    return {char for char in normalize_text(value) if char.isalpha()}


def parse_decimal(value: str) -> Decimal:
    normalized = value.strip().replace("，", ",").replace("％", "%")
    if normalized.endswith("%"):
        return Decimal(normalized[:-1]) / Decimal(100)
    if "/" in normalized and normalized.count("/") == 1:
        numerator, denominator = normalized.split("/", 1)
        return Decimal(numerator) / Decimal(denominator)
    return Decimal(normalized)


def parse_safe_expression(value: str):
    candidate = value.strip().replace("×", "*").replace("÷", "/")
    if not SAFE_EXPRESSION.fullmatch(candidate):
        raise ValueError("expression_contains_unsupported_characters")
    identifiers = set(re.findall(r"[A-Za-z_]+", candidate))
    if not identifiers.issubset(SAFE_IDENTIFIERS):
        raise ValueError("expression_contains_unsupported_identifiers")
    local_dict = {name: Symbol(name, real=True) for name in SAFE_IDENTIFIERS}
    return parse_expr(
        candidate,
        local_dict=local_dict,
        global_dict={"Integer": __import__("sympy").Integer, "Float": __import__("sympy").Float},
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def grade_with_rules(question: dict[str, Any], response: dict[str, Any] | None) -> GradeDecision:
    max_score = float(question["max_score"])
    if response is None:
        return GradeDecision(None, max_score, ScoreState.BLOCKED, True, ["answer_missing"])

    raw_text = str(response.get("raw_text") or "")
    if not raw_text.strip():
        if response.get("blank_confirmed"):
            return GradeDecision(0.0, max_score, ScoreState.REPRODUCIBLE_EXACT, False, ["confirmed_blank"])
        return GradeDecision(None, max_score, ScoreState.BLOCKED, True, ["blank_not_confirmed"])

    confidence = response.get("ocr_confidence")
    minimum_confidence = float(question.get("rule_config", {}).get("minimum_ocr_confidence", 0.85))
    if response.get('extraction_source')!='api_vision_fallback' and confidence is not None and float(confidence) < minimum_confidence:
        return GradeDecision(
            None,
            max_score,
            ScoreState.BLOCKED,
            True,
            ["ocr_confidence_too_low"],
            [f"OCR confidence={float(confidence):.3f}"],
        )

    mode = question["scoring_mode"]
    answer_key = question["answer_key"]
    rule_config = question.get("rule_config", {})

    try:
        if mode == "exact_text":
            accepted = answer_key if isinstance(answer_key, list) else [answer_key]
            matched = normalize_text(raw_text) in {normalize_text(str(item)) for item in accepted}
            return GradeDecision(
                max_score if matched else 0.0,
                max_score,
                ScoreState.REPRODUCIBLE_EXACT,
                False,
                ["exact_match" if matched else "exact_mismatch"],
                [f"normalized_answer={normalize_text(raw_text)}"],
            )

        if mode == "choice_set":
            expected = normalize_choice_set("".join(answer_key) if isinstance(answer_key, list) else str(answer_key))
            actual = normalize_choice_set(raw_text)
            matched = actual == expected
            return GradeDecision(
                max_score if matched else 0.0,
                max_score,
                ScoreState.REPRODUCIBLE_EXACT,
                False,
                ["choice_set_match" if matched else "choice_set_mismatch"],
                [f"selected={','.join(sorted(actual))}"],
            )

        if mode == "numeric":
            expected_value = parse_decimal(str(answer_key))
            actual_value = parse_decimal(raw_text)
            tolerance = Decimal(str(rule_config.get("tolerance", "0")))
            difference = abs(actual_value - expected_value)
            matched = difference <= tolerance
            return GradeDecision(
                max_score if matched else 0.0,
                max_score,
                ScoreState.REPRODUCIBLE_EXACT,
                False,
                ["numeric_match" if matched else "numeric_mismatch"],
                [f"parsed={actual_value}", f"difference={difference}", f"tolerance={tolerance}"],
            )

        if mode == "symbolic":
            expected_expr = parse_safe_expression(str(answer_key))
            actual_expr = parse_safe_expression(raw_text)
            matched = simplify(actual_expr - expected_expr) == 0
            if rule_config.get("required_form") and normalize_text(raw_text) != normalize_text(str(answer_key)):
                return GradeDecision(
                    None,
                    max_score,
                    ScoreState.BLOCKED,
                    True,
                    ["equivalent_but_required_form_unverified"],
                    [f"simplified_difference={simplify(actual_expr - expected_expr)}"],
                )
            return GradeDecision(
                max_score if matched else 0.0,
                max_score,
                ScoreState.REPRODUCIBLE_EXACT,
                False,
                ["symbolic_match" if matched else "symbolic_mismatch"],
                [f"simplified_difference={simplify(actual_expr - expected_expr)}"],
            )

        return GradeDecision(None, max_score, ScoreState.BLOCKED, True, ["non_deterministic_route"])
    except (InvalidOperation, ZeroDivisionError, TypeError, ValueError, SyntaxError) as exc:
        return GradeDecision(
            None,
            max_score,
            ScoreState.BLOCKED,
            True,
            ["answer_parse_failed"],
            [str(exc)[:200]],
        )

