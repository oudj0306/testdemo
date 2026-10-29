from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock

import pytest

from grading_agent.config import get_settings
from grading_agent.infrastructure.work_locks import SubmissionBusyError, _entries, submission_work
from grading_agent.services.scoring import submission_grader as grading
from grading_agent.services.scoring.llm_grader import LLMResult, OpenAIGrader
from grading_agent.services.workflows.batch_queue import BatchQueue, enqueue_saved
from grading_agent.storage.db import initialize_database
from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    create_review,
    create_submission_record,
    get_batch,
    get_submission,
    publish_assessment,
    token_usage_for_assessment,
    upsert_response,
)


def papers(count=1, questions=4):
    initialize_database()
    assessment = create_assessment("Parallel grading", [])
    items = []
    for number in range(1, questions + 1):
        items.append(
            create_question(
                assessment["id"],
                {
                    "question_no": str(number),
                    "prompt": f"Explain {number}",
                    "question_type": "short_answer",
                    "capture_mode": "text",
                    "answer_form": "text",
                    "scoring_mode": "rubric_assisted",
                    "max_score": 10,
                    "answer_key": "Reference",
                    "answer_key_state": "verified",
                    "rule_config": {},
                    "rubric": [{"criterion": "Correct explanation", "score": 10}],
                },
            )
        )
    publish_assessment(assessment["id"])
    submissions = []
    for index in range(count):
        submission = create_submission_record(
            assessment["id"],
            f"s{index}",
            f"h{index}",
            "unused.png",
            "image/png",
            {},
            [],
        )
        for q in items:
            upsert_response(
                submission["id"],
                q["id"],
                f"Answer {index}:{q['question_no']}",
                None,
                False,
                "manual",
            )
        submissions.append(submission)
    return assessment, items, submissions


def settings():
    return replace(
        get_settings(),
        openai_api_key="fake",
        review_sample_rate=0,
        llm_grading_passes=3,
        review_mode="selective",
    )


def test_two_papers_share_bound_preserve_two_passes_order_scores_and_usage(monkeypatch):
    assessment, questions, submissions = papers(count=2)
    guard, release, full = Lock(), Event(), Event()
    active = peak = 0
    calls = {}

    def single(self, question, answer):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            calls[answer] = calls.get(answer, 0) + 1
            if active == 4:
                full.set()
        try:
            assert release.wait(10)
            return LLMResult(
                int(question["question_no"]),
                [],
                [answer],
                False,
                input_tokens=100,
                output_tokens=20,
            )
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    with ThreadPoolExecutor(4) as pool, ThreadPoolExecutor(2) as callers:
        monkeypatch.setattr(grading, "question_executor", lambda: pool)
        futures = [
            callers.submit(grading.grade_submission, s["id"], settings()) for s in submissions
        ]
        try:
            assert full.wait(10)
        finally:
            release.set()
        results = [f.result(timeout=10) for f in futures]
    assert peak == 4
    assert len(calls) == 8 and set(calls.values()) == {2}
    for result in results:
        assert [g["question_no"] for g in result["grade_items"]] == ["1", "2", "3", "4"]
        assert [g["score"] for g in result["grade_items"]] == [1, 2, 3, 4]
        assert result["score_summary"]["current_score"] == 10
        assert all(g["score_state"] == "AUTO_STABLE" for g in result["grade_items"])
        assert all(
            g["snapshot"]["response"]["raw_text"].startswith("Answer")
            for g in result["grade_items"]
        )
    total = token_usage_for_assessment(assessment["id"])["total"]
    assert total["input"] == 1600
    assert total["output"] == 320


def test_overlapping_same_submission_does_not_queue_duplicate_model_work(monkeypatch):
    _, _, submissions = papers(questions=1)
    sid = submissions[0]["id"]
    started, release = Event(), Event()
    calls = []

    def single(self, question, answer):
        calls.append(answer)
        started.set()
        assert release.wait(10)
        return LLMResult(8, [], [answer], False)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    with ThreadPoolExecutor(1) as caller:
        first = caller.submit(grading.grade_submission, sid, settings())
        try:
            assert started.wait(10)
            with pytest.raises(SubmissionBusyError):
                grading.grade_submission(sid, settings())
        finally:
            release.set()
        assert first.result(timeout=10)["score_summary"]["current_score"] == 8
    assert len(calls) == 2
    assert not _entries


def test_busy_submission_batch_job_is_requeued_instead_of_failed():
    assessment, _, submissions = papers(questions=1)
    sid = submissions[0]["id"]
    batch = enqueue_saved(assessment["id"], [sid])
    queue = BatchQueue(lambda *a: pytest.fail("saved submission must be reused"))
    job = queue.claim()
    locked, release = Event(), Event()

    def hold():
        with submission_work(get_settings().database_path, sid):
            locked.set()
            assert release.wait(10)

    with ThreadPoolExecutor(1) as owner:
        future = owner.submit(hold)
        try:
            assert locked.wait(10)
            queue.process(job)
            state = get_batch(batch["id"])
            assert state["status"] == "processing"
            assert state["items"][0]["status"] == "queued"
            assert state["failed_count"] == 0
        finally:
            release.set()
        future.result(timeout=10)


def test_parallel_regrade_preserves_reviewed_and_accepted_items(monkeypatch):
    _, _, submissions = papers(questions=2)
    sid = submissions[0]["id"]
    monkeypatch.setattr(
        OpenAIGrader, "_single_pass", lambda self, q, a: LLMResult(8, [], [a], False)
    )
    first = grading.grade_submission(sid, settings())
    create_review(first["grade_items"][0]["id"], "teacher", 6, "Correction")
    monkeypatch.setattr(
        OpenAIGrader, "_single_pass", lambda *a: pytest.fail("must reuse accepted grades")
    )
    second = grading.grade_submission(sid, settings(), pending_only=True)
    assert [g["id"] for g in second["grade_items"]] == [g["id"] for g in first["grade_items"]]
    assert second["score_summary"]["current_score"] == 14


def test_retry_only_unscored_question_preserves_scored_review_suggestion(monkeypatch):
    _, questions, submissions = papers(questions=2)
    sid = submissions[0]["id"]
    calls = []

    def first_pass(self, question, answer):
        calls.append(question["question_no"])
        if question["question_no"] == "2":
            raise RuntimeError("temporary model failure")
        return LLMResult(8, [], [answer], True)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", first_pass)
    first = grading.grade_submission(sid, settings())
    scored = next(g for g in first["grade_items"] if g["question_no"] == "1")
    assert scored["score"] == 8 and scored["needs_review"]
    assert next(g for g in first["grade_items"] if g["question_no"] == "2")["score"] is None

    calls.clear()
    monkeypatch.setattr(
        OpenAIGrader,
        "_single_pass",
        lambda self, question, answer: (
            calls.append(question["question_no"]) or LLMResult(7, [], [answer], False)
        ),
    )
    second = grading.grade_submission(sid, settings(), pending_only=True)
    assert calls == ["2", "2"]
    assert next(g for g in second["grade_items"] if g["question_no"] == "1")["id"] == scored["id"]
    assert second["score_summary"]["current_score"] == 15


def test_parallel_results_never_commit_over_changed_answer(monkeypatch):
    _, questions, submissions = papers(questions=2)
    sid = submissions[0]["id"]
    started, release = Event(), Event()

    def single(self, question, answer):
        started.set()
        assert release.wait(10)
        return LLMResult(8, [], [answer], False)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    with ThreadPoolExecutor(1) as caller:
        future = caller.submit(grading.grade_submission, sid, settings())
        try:
            assert started.wait(10)
            upsert_response(sid, questions[0]["id"], "Corrected answer", None, False, "manual")
        finally:
            release.set()
        with pytest.raises(ValueError, match="已修改"):
            future.result(timeout=10)
    assert not get_submission(sid)["grade_items"]
