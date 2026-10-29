import io
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grading_agent import config
from grading_agent.infrastructure.telemetry import summary
from grading_agent.services.scoring.bulk_grader import grade_daily_items
from grading_agent.services.scoring.llm_grader import LLMResult, OpenAIGrader
from grading_agent.services.scoring.submission_grader import grade_submission
from grading_agent.services.workflows.batch_queue import BatchQueue, enqueue_saved
from grading_agent.storage.db import initialize_database, transaction
from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    create_review,
    create_submission_record,
    get_assessment,
    get_batch,
    get_submission,
    publish_assessment,
    token_usage_for_assessment,
    update_question,
    upsert_response,
)


def fixture_papers(count=1, profile="daily", mode="rubric_assisted"):
    initialize_database()
    settings = replace(
        config.get_settings(),
        openai_api_key="fake",
        grading_profile=profile,
        review_sample_rate=0,
        llm_grading_passes=1,
    )
    config._settings_cache = settings
    assessment = create_assessment("Template test", [])
    q = create_question(
        assessment["id"],
        dict(
            question_no="1",
            prompt="Explain evaporation",
            question_type="short_answer",
            scoring_mode=mode,
            max_score=10,
            answer_key="evaporation",
            capture_mode="text",
            answer_form="text",
            answer_key_state="verified",
            rule_config={},
            rubric=[{"criterion": "point", "score": 10}],
        ),
    )
    publish_assessment(assessment["id"])
    papers = []
    for i in range(count):
        s = create_submission_record(
            assessment["id"],
            f"PRIVATE_NAME_{i}",
            f"hash-{i}",
            "not-a-real-file.png",
            "image/png",
            {},
            [],
        )
        upsert_response(s["id"], q["id"], f"evaporation {i}", None, False, "manual")
        papers.append(s)
    return settings, assessment, q, papers


def fake_api(monkeypatch, change=None):
    calls = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        calls.append(payload)
        content = payload["input"][-1]["content"][-1]["text"]
        if payload["text"]["format"]["name"] == "bulk_grades":
            items = json.loads(content.split("：", 1)[1])
            result = {
                "results": [
                    dict(
                        answer_id=i["answer_id"],
                        score=8,
                        reason_codes=[],
                        evidence=[i["answer"]],
                        needs_review=False,
                    )
                    for i in items
                ]
            }
        else:
            answer = content.split("：", 1)[1]
            result = dict(score=8, reason_codes=[], evidence=[answer], needs_review=False)
        if change:
            result = change(result, len(calls))
        return io.BytesIO(
            json.dumps(
                {
                    "output_text": json.dumps(result),
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 30},
                    },
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def test_profiles_default_legacy_and_roundtrip():
    assert config.get_settings().grading_profile == "legacy"
    s = config.get_settings()
    updated = config.update_llm_settings(
        None, s.openai_model, s.openai_base_url, 1, True, False, grading_profile="daily"
    )
    assert updated.grading_profile == "daily"


def test_daily_single_is_not_consensus_and_exact_cache_avoids_calls(monkeypatch):
    settings, a, q, papers = fixture_papers()
    calls = fake_api(monkeypatch)
    first = grade_submission(papers[0]["id"], settings)
    grade = first["grade_items"][0]
    assert grade["score_state"] == "AUTO_SINGLE" and grade["score"] == 8
    assert not grade["needs_review"]
    second = OpenAIGrader(settings).grade(q, "evaporation 0")
    assert second.score == 8 and len(calls) == 1
    assert any("cache_hit" in e for e in second.evidence)
    assert token_usage_for_assessment(a["id"])["total"]["input"] == 100


def test_exam_disagreement_never_silently_averages(monkeypatch):
    settings, _, _, papers = fixture_papers(profile="exam")
    values = iter([4, 6])
    monkeypatch.setattr(
        OpenAIGrader, "_single_pass", lambda self, q, a: LLMResult(next(values), [], [a], False)
    )
    result = grade_submission(papers[0]["id"], settings)["grade_items"][0]
    assert result["score"] is None and result["score_state"] == "UNSTABLE"
    assert result["needs_review"] and "teacher_review_policy" in result["reason_codes"]


def test_two_stage_batch_has_eight_limit_tail_and_usage_once(monkeypatch):
    settings, assessment, q, papers = fixture_papers(10)
    calls = fake_api(monkeypatch)
    batch = enqueue_saved(assessment["id"], [s["id"] for s in papers])
    queue = BatchQueue(lambda *a: pytest.fail("No extraction needed"))
    while job := queue.claim():
        assert "daily_group" in job
        queue.process_daily(job["daily_group"])
    assert get_batch(batch["id"])["success_count"] == 10
    assert len(calls) == 2
    assert [
        len(json.loads(c["input"][-1]["content"][-1]["text"].split("：", 1)[1])) for c in calls
    ] == [8, 2]
    measured = summary(batch["id"])
    assert measured["requests"] == 2 and measured["tokens"]["input_tokens"] == 200
    assert measured["stages"]["database"]["count"] > 0
    assert token_usage_for_assessment(assessment["id"])["total"]["input"] == 200
    for s in papers:
        g = get_submission(s["id"])["grade_items"][0]
        assert g["score"] == 8 and g["score_state"] == "AUTO_SINGLE"
        assert any(e.startswith("request_id=") for e in g["evidence"])


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "out_of_range", "wrong_evidence"])
def test_partial_batch_retains_valid_result_and_retries_only_bad(monkeypatch, corruption):
    settings, _, q, _ = fixture_papers()

    def change(result, n):
        if n == 1:
            if corruption == "missing":
                result["results"].pop(0)
            if corruption == "duplicate":
                result["results"].append(result["results"][0].copy())
            if corruption == "out_of_range":
                result["results"][0]["score"] = 1000
            if corruption == "wrong_evidence":
                result["results"][0]["evidence"] = ["another student"]
        return result

    calls = fake_api(monkeypatch, change)
    result = grade_daily_items(
        q, [{"answer_id": "a", "answer": "first"}, {"answer_id": "b", "answer": "second"}], settings
    )
    assert result["a"].score == result["b"].score == 8
    assert len(calls) == 2
    assert calls[1]["input"][-1]["content"][-1]["text"].endswith("first")


def test_failed_recovery_is_scoreless_and_valid_result_kept(monkeypatch):
    settings, _, q, _ = fixture_papers()

    def change(result, n):
        if n == 1:
            result["results"][0]["score"] = -1
        else:
            result["score"] = -1
        return result

    calls = fake_api(monkeypatch, change)
    result = grade_daily_items(
        q, [{"answer_id": "a", "answer": "first"}, {"answer_id": "b", "answer": "second"}], settings
    )
    assert len(calls) == 2 and result["a"].score is None and result["b"].score == 8


def test_exact_duplicate_dedup_and_question_changes_invalidate_cache(monkeypatch):
    settings, _, q, _ = fixture_papers()
    calls = fake_api(monkeypatch)
    result = grade_daily_items(
        q, [{"answer_id": "a", "answer": "same"}, {"answer_id": "b", "answer": "same"}], settings
    )
    assert len(calls) == 1 and result["b"].score == 8
    grade_daily_items(
        {**q, "rule_config": {"region": {"page": 1, "x": 0, "y": 0, "width": 1, "height": 0.5}}},
        [{"answer_id": "c", "answer": "same"}],
        settings,
    )
    assert len(calls) == 2


def test_manual_answer_cannot_be_overwritten_by_delayed_ocr():
    _, _, q, papers = fixture_papers()
    saved = upsert_response(papers[0]["id"], q["id"], "wrong", 0.99, False, "paddleocr_cpu")
    assert saved["raw_text"] == "evaporation 0" and saved["extraction_source"] == "manual"


def test_template_preview_validation_and_training_labels(monkeypatch):
    settings, a, q, papers = fixture_papers()
    fake_api(monkeypatch)
    grade = grade_submission(papers[0]["id"], settings)["grade_items"][0]
    image = settings.data_dir / "template.png"
    Image.new("RGB", (200, 100), "white").save(image)
    with transaction() as db:
        db.execute(
            """INSERT INTO assessment_documents VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                "document",
                a["id"],
                "question_paper",
                "hash-image",
                str(image),
                "image/png",
                "",
                "stored",
                1,
                "2026-10-21",
            ),
        )
    from grading_agent.app import app

    with TestClient(app) as client:
        path = f"/api/assessments/{a['id']}"
        region = dict(page=1, x=0.25, y=0.1, width=0.5, height=0.5, page_aspect=2)
        preview = client.post(path + "/template-preview?document_id=document", json=region)
        assert preview.status_code == 200
        assert Image.open(io.BytesIO(preview.content)).size == (100, 50)
        assert client.get(path + "/template-page?document_id=document&page=2").status_code == 400
        assert (
            client.post(
                path + "/template-preview?document_id=document", json={**region, "x": 0.9}
            ).status_code
            == 422
        )
        assert (
            client.put(
                "/api/questions/" + q["id"], json={**q, "rule_config": {"region": region}}
            ).status_code
            == 200
        )
        assert not client.get(path + "/training-samples").content
        assert (
            client.post(
                f"/api/grade-items/{grade['id']}/training-approval",
                json={"redacted_text": "evaporation", "approved_by": "teacher"},
            ).status_code
            == 200
        )
        pseudo = client.get(path + "/training-samples?kind=pseudo")
        assert json.loads(pseudo.text)["label_source"] == "machine_pseudo"
        assert (
            "PRIVATE_NAME" not in pseudo.text
            and "fake" not in pseudo.text
            and "source_path" not in pseudo.text
        )
        assert not client.get(path + "/training-samples?kind=teacher").content
        review = create_review(grade["id"], "PRIVATE_TEACHER", 7, "private comment")
        assert not client.get(path + "/training-samples?kind=pseudo").content
        assert (
            client.post(
                f"/api/grade-items/{grade['id']}/training-approval",
                json={
                    "redacted_text": "evaporation",
                    "approved_by": "teacher",
                    "expected_review_id": review["id"],
                },
            ).status_code
            == 200
        )
        exported = client.get(path + "/training-samples?kind=teacher")
        assert json.loads(exported.text)["score"] == 7
        assert "PRIVATE_TEACHER" not in exported.text and "private comment" not in exported.text


def test_region_scaling_and_wrong_aspect_fall_back(tmp_path):
    from grading_agent.services.documents.ingestion import NormalizedPage, crop_question_regions

    settings, _, q, _ = fixture_papers()
    image = tmp_path / "page.png"
    Image.new("RGB", (400, 200)).save(image)
    region = dict(page=1, x=0.25, y=0.1, width=0.5, height=0.5, page_aspect=2)
    pages = [NormalizedPage(1, image, 400, 200, "ok", {})]
    question = {**q, "rule_config": {"region": region}}
    crop = crop_question_regions(pages, [question], settings)
    assert Image.open(crop[q["id"]]).size == (200, 100)
    assert not crop_question_regions(
        pages, [{**q, "rule_config": {"region": {**region, "page_aspect": 1}}}], settings
    )
    assert not crop_question_regions(
        pages, [{**q, "rule_config": {"region": {**region, "page": 2}}}], settings
    )


def test_unknown_ids_do_not_replace_known_answers(monkeypatch):
    settings, _, q, _ = fixture_papers()

    def change(result, n):
        result["results"].append({**result["results"][0], "answer_id": "unknown"})
        return result

    calls = fake_api(monkeypatch, change)
    result = grade_daily_items(
        q, [{"answer_id": "a", "answer": "first"}, {"answer_id": "b", "answer": "second"}], settings
    )
    assert set(result) == {"a", "b"} and len(calls) == 1


def test_question_changed_during_batch_does_not_commit_stale_grades(monkeypatch):
    settings, assessment, q, papers = fixture_papers(2)

    def change(result, n):
        update_question(q["id"], {**q, "prompt": "Changed while request was running"})
        return result

    fake_api(monkeypatch, change)
    batch = enqueue_saved(assessment["id"], [p["id"] for p in papers])
    queue = BatchQueue(lambda *a: pytest.fail("Unexpected extraction"))
    job = queue.claim()
    queue.process_daily(job["daily_group"])
    assert get_batch(batch["id"])["failed_count"] == 2
    assert all(not get_submission(p["id"])["grade_items"] for p in papers)


def test_different_questions_do_not_share_answer_cache(monkeypatch):
    settings, a, q, _ = fixture_papers()
    other = create_question(a["id"], {**q, "question_no": "2"})
    calls = fake_api(monkeypatch)
    for question in [q, other]:
        grade_daily_items(question, [{"answer_id": "a", "answer": "same"}], settings)
    assert len(calls) == 2


def test_blank_unconfirmed_and_rules_use_no_model(monkeypatch):
    settings, _, q, papers = fixture_papers(mode="exact_text")
    calls = fake_api(monkeypatch)
    upsert_response(papers[0]["id"], q["id"], "evaporation", None, False, "manual")
    assert grade_submission(papers[0]["id"], settings)["grade_items"][0]["score"] == 10
    upsert_response(papers[0]["id"], q["id"], "", None, False, "manual")
    assert grade_submission(papers[0]["id"], settings)["grade_items"][0]["score"] is None
    assert not calls


def test_local_region_success_skips_vision_and_rules_skip_grading(monkeypatch):
    settings, a, q, _ = fixture_papers(mode="exact_text")
    config._settings_cache = replace(settings, ocr_provider="paddleocr")
    update_question(
        q["id"],
        {
            **q,
            "rule_config": {
                "region": {
                    "page": 1,
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.5,
                    "height": 0.2,
                    "page_aspect": 0.75,
                }
            },
        },
    )
    monkeypatch.setattr(
        "grading_agent.services.documents.ingestion.OCRAdapter.extract",
        lambda self, path: ("evaporation", 0.99, "paddleocr_cpu"),
    )
    calls = fake_api(monkeypatch)
    from grading_agent.app import persist_submission

    image = io.BytesIO()
    Image.new("RGB", (600, 800), "white").save(image, "PNG")
    submission = persist_submission(
        get_assessment(a["id"]), "synthetic-local", image.getvalue(), "image/png"
    )
    result = grade_submission(submission["id"], config.get_settings())
    assert result["grade_items"][0]["score"] == 10 and not calls


def test_invalid_json_response_recovers_only_once(monkeypatch):
    settings, _, q, _ = fixture_papers()
    calls = []

    def bad(request, timeout):
        calls.append(request)
        return io.BytesIO(b"[]")

    monkeypatch.setattr("urllib.request.urlopen", bad)
    result = grade_daily_items(q, [{"answer_id": "a", "answer": "first"}], settings)
    assert len(calls) == 2 and result["a"].score is None


def test_changed_standards_between_students_never_share_request(monkeypatch):
    settings, a, q, papers = fixture_papers(2)
    calls = fake_api(monkeypatch)
    from grading_agent.storage import repository

    original = repository.questions_for_submission
    snapshots = []

    def changed(sid):
        snapshots.append(sid)
        if len(snapshots) == 2:
            update_question(q["id"], {**q, "prompt": "New standard"})
        return original(sid)

    monkeypatch.setattr(repository, "questions_for_submission", changed)
    batch = enqueue_saved(a["id"], [p["id"] for p in papers])
    queue = BatchQueue(lambda *a: pytest.fail("Unexpected extraction"))
    queue.process_daily(queue.claim()["daily_group"])
    assert len(calls) == 2
    assert all(c["text"]["format"]["name"] == "grade_result" for c in calls)
    assert get_batch(batch["id"])["success_count"] == get_batch(batch["id"])["failed_count"] == 1
