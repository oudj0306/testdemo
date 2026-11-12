from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    publish_assessment,
)

router = APIRouter()


@router.post("/api/demo/bootstrap")
def demo_bootstrap() -> dict[str, Any]:
    assessment = create_assessment("数学演示卷", ["math"])
    examples = [
        {
            "question_no": "1",
            "prompt": "选择正确答案",
            "question_type": "single_choice",
            "capture_mode": "short_handwriting",
            "answer_form": "text",
            "scoring_mode": "exact_text",
            "max_score": 5,
            "answer_key": "B",
            "rule_config": {},
            "rubric": [],
            "review_policy": "auto_if_verified",
        },
        {
            "question_no": "2",
            "prompt": "计算 1/4 + 1/4",
            "question_type": "numeric",
            "capture_mode": "short_handwriting",
            "answer_form": "text",
            "scoring_mode": "numeric",
            "max_score": 5,
            "answer_key": "1/2",
            "rule_config": {"tolerance": "0"},
            "rubric": [],
            "review_policy": "auto_if_verified",
        },
        {
            "question_no": "3",
            "prompt": "化简 2x+2",
            "question_type": "symbolic",
            "capture_mode": "short_handwriting",
            "answer_form": "text",
            "scoring_mode": "symbolic",
            "max_score": 5,
            "answer_key": "2*(x+1)",
            "rule_config": {},
            "rubric": [],
            "review_policy": "auto_if_verified",
        },
        {
            "question_no": "4",
            "prompt": "说明解题过程",
            "question_type": "calculation",
            "capture_mode": "long_handwriting",
            "answer_form": "text",
            "scoring_mode": "rubric_assisted",
            "max_score": 10,
            "answer_key": "",
            "rule_config": {},
            "rubric": [
                {"criterion": "列式正确", "score": 4},
                {"criterion": "推导正确", "score": 4},
                {"criterion": "结论正确", "score": 2},
            ],
            "review_policy": "always_review",
        },
    ]
    for example in examples:
        create_question(assessment["id"], example)
    return publish_assessment(assessment["id"])
