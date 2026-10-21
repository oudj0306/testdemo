from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def infer_question_route(prompt: str, answer_key: Any, rubric: list[dict[str, Any]]) -> tuple[str, str]:
    text = prompt.lower()
    if rubric or any(token in text for token in ("证明", "说明理由", "解答过程", "计算过程", "应用题")):
        if "证明" in text:
            return "proof", "rubric_assisted"
        return "calculation", "rubric_assisted"
    if isinstance(answer_key, list):
        if all(str(item).strip().upper() in {"A", "B", "C", "D", "E", "F"} for item in answer_key):
            return "multiple_choice", "choice_set"
        return "solution_set", "exact_text"
    answer = str(answer_key).strip()
    if answer.upper() in {"A", "B", "C", "D", "E", "F"}:
        return "single_choice", "exact_text"
    if answer.lower() in {"true", "false", "正确", "错误", "对", "错"}:
        return "true_false", "exact_text"
    try:
        numeric = answer.rstrip("%")
        if numeric.count("/") == 1:
            numerator, denominator = numeric.split("/", 1)
            Decimal(numerator) / Decimal(denominator)
        else:
            Decimal(numeric)
        return "numeric", "numeric"
    except (InvalidOperation, ValueError, ZeroDivisionError):
        pass
    if any(character in answer for character in ("x", "y", "z", "^", "+", "-", "*", "/", "(", ")")):
        return "symbolic", "symbolic"
    return "unknown", "exact_text"
