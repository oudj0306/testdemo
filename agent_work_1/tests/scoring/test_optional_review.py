from grading_agent.storage.db import initialize_database, transaction
from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    create_submission_record,
    get_submission,
    list_submissions,
    publish_assessment,
    replace_active_grades,
)
from grading_agent.web.workspace_views import report


def test_numeric_machine_score_is_final_while_review_stays_optional():
    initialize_database()
    assessment = create_assessment("Optional review", [])
    question = create_question(
        assessment["id"],
        {
            "question_no": "1",
            "prompt": "Show your work",
            "question_type": "calculation",
            "capture_mode": "short_handwriting",
            "answer_form": "text",
            "scoring_mode": "rubric_assisted",
            "max_score": 5,
            "answer_key": "4",
            "answer_key_state": "verified",
            "rubric": [{"criterion": "Correct", "score": 5}],
            "rule_config": {},
        },
    )
    publish_assessment(assessment["id"])
    submission = create_submission_record(
        assessment["id"],
        "Student A",
        "optional-review-hash",
        "test.png",
        "image/png",
        {},
        [],
    )
    replace_active_grades(
        submission["id"],
        [
            {
                "question_id": question["id"],
                "score": 4,
                "max_score": 5,
                "score_state": "CONSENSUS_CANDIDATE",
                "needs_review": True,
                "reason_codes": ["quality_spot_check"],
                "evidence": ["model score"],
                "grader_kind": "llm",
                "grader_version": "test",
            }
        ],
    )

    # Existing installations may still contain the old blocking status.
    with transaction() as db:
        db.execute(
            "UPDATE submissions SET status='review_required',finalized_at=NULL WHERE id=?",
            (submission["id"],),
        )
    initialize_database()

    saved = get_submission(submission["id"])
    assert saved["status"] == "finalized"
    assert saved["score_summary"]["current_score"] == 4
    assert saved["score_summary"]["unresolved_count"] == 0
    assert saved["score_summary"]["review_recommended_count"] == 1
    assert list_submissions()[0]["review_recommended_count"] == 1
    assert report(assessment["id"])["students"][0]["final_score"] == 4
