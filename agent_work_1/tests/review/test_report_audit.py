"""Report totals and exports keep rule passes, missing values and untrusted labels safe."""

import json

from grading_agent.storage.db import initialize_database
from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    create_review,
    create_submission_record,
    get_submission,
    publish_assessment,
    replace_active_grades,
)
from grading_agent.web.workspace_views import report, report_html


def report_fixture():
    initialize_database()
    assessment = create_assessment("Report audit", [])
    questions = []
    for number in range(1, 6):
        questions.append(
            create_question(
                assessment["id"],
                {
                    "question_no": str(number),
                    "prompt": "Question",
                    "question_type": "short_answer",
                    "capture_mode": "short_handwriting",
                    "answer_form": "text",
                    "scoring_mode": "exact_text",
                    "max_score": 5,
                    "answer_key": "B",
                    "answer_key_state": "verified",
                    "rule_config": {},
                    "rubric": [],
                },
            )
        )
    publish_assessment(assessment["id"])
    submission = create_submission_record(
        assessment["id"],
        "Student",
        "report-hash",
        "unused-original.png",
        "image/png",
        {},
        [],
    )
    grades = []
    for index, state in enumerate(
        ["AUTO_STABLE", "REPRODUCIBLE_EXACT", "REPRODUCIBLE_EXACT", "BLOCKED"]
    ):
        grades.append(
            {
                "question_id": questions[index]["id"],
                "score": None if state == "BLOCKED" else 5,
                "max_score": 5,
                "score_state": state,
                "needs_review": state == "BLOCKED",
                "reason_codes": ["answer_missing"] if state == "BLOCKED" else [],
                "evidence": ["test"],
                "grader_kind": "llm" if state == "AUTO_STABLE" else "rules",
                "grader_version": "test",
            }
        )
    replace_active_grades(submission["id"], grades)
    human_grade = next(
        g
        for g in get_submission(submission["id"])["grade_items"]
        if g["question_id"] == questions[2]["id"]
    )
    first = create_review(human_grade["id"], "teacher", 4, "checked")
    create_review(human_grade["id"], "teacher", 3, "corrected", first["id"], True)
    return assessment, questions, submission


def chart_payload(html):
    encoded = html.split("<script>const D=", 1)[1].split(";new Chart(", 1)[0]
    return json.loads(encoded)


def test_report_partitions_model_rule_human_and_pending_questions():
    assessment, _, _ = report_fixture()
    result = report(assessment["id"])
    assert result["auto_stable_question_count"] == 1
    assert result["rule_passed_question_count"] == 1
    assert result["human_reviewed_question_count"] == 1
    assert result["pending_question_count"] == 2
    assert result["students"][0]["confirmed_score"] == 13
    assert result["students"][0]["final_score"] is None
    assert result["questions"][2]["average"] == 3
    assert result["questions"][3]["score_rate"] is None
    assert result["questions"][4]["score_rate"] is None


def test_export_uses_same_four_groups_and_keeps_missing_rates_null():
    assessment, _, _ = report_fixture()
    html = report_html(assessment["id"]).body.decode("utf-8")
    payload = chart_payload(html)
    assert payload["mode"] == [1, 1, 1, 2]
    assert payload["q_rates"] == [100, 100, 60, None, None]
    assert "labels:['模型评分','规则评分','人工复核','缺少分数']" in html


def test_export_question_label_cannot_close_the_script_element():
    initialize_database()
    assessment = create_assessment("Untrusted question label", [])
    label = "</script><script>alert(1)</script>"
    create_question(
        assessment["id"],
        {
            "question_no": label,
            "prompt": "Question",
            "question_type": "short_answer",
            "capture_mode": "short_handwriting",
            "answer_form": "text",
            "scoring_mode": "exact_text",
            "max_score": 5,
            "answer_key": "B",
        },
    )
    html = report_html(assessment["id"]).body.decode("utf-8")
    assert label not in html
    assert "\\u003c/script>" in html
    payload = chart_payload(html)
    assert payload["q_labels"] == [f"第{label}题"]
    assert payload["q_rates"] == [None]
