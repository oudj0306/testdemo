import copy
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from grading_agent import config
from grading_agent.services.recognition import local_vision as vision
from grading_agent.services.recognition.blank_detection import blank_pixels
from tests.recognition.test_local_layout import generate, mark, upload
from tests.recognition.test_local_layout import workspace as workspace
from tests.recognition.test_local_vision import enable_visual


def test_clear_paper_and_printed_edge_rules_can_be_blank():
    image = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((0, 72, 399, 72), fill="black", width=2)
    assert blank_pixels(image)["confirmed"]
    draw.text((40, 30), "1", fill=(249, 249, 249))
    assert not blank_pixels(image)["confirmed"]  # Very faint writing is not an empty page.


def test_printed_text_requires_a_blank_reference_and_extra_ink_is_not_removed():
    reference = Image.new("RGB", (400, 80), "white")
    ImageDraw.Draw(reference).text((5, 25), "(2)", fill="black")
    assert not blank_pixels(reference)["confirmed"]
    assert blank_pixels(reference, reference)["confirmed"]
    written = reference.copy()
    ImageDraw.Draw(written).point((30, 30), fill="black")
    assert not blank_pixels(written, reference)["confirmed"]
    assert not blank_pixels(Image.new("RGB", (400, 80), (180, 180, 180)))["confirmed"]


def test_complete_row_frame_excludes_clipped_neighbouring_answer_but_not_own_ink():
    image = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, -5), "clipped neighbour", fill="black")
    draw.line((0, 10, 399, 10), fill="black")
    draw.line((0, 70, 399, 70), fill="black")
    evidence = blank_pixels(image)
    assert evidence["confirmed"] and evidence["evaluated_box"] == [0, 11, 400, 70]
    draw.point((40, 50), fill=(249, 249, 249))
    assert not blank_pixels(image)["confirmed"]


def blank_result():
    return {
        "text": "",
        "consistent": True,
        "issues": [],
        "blank_candidate": True,
        "model": {"model": "mock-local"},
        "requests": [],
    }


def test_confirmed_blank_gets_zero_and_is_reused_without_scoring_api(workspace, monkeypatch):
    client, a, qs, doc, source, calls = workspace
    generate(workspace)
    enable_visual(monkeypatch)
    # Include a subjective question path: even it must not invoke the scoring API for a blank.
    q = qs[1]
    assert (
        client.put(
            "/api/questions/" + q["id"],
            json={**q, "question_type": "essay", "scoring_mode": "rubric_assisted"},
        ).status_code
        == 200
    )
    generate(workspace)
    recognition = []

    def transcribe(*args):
        recognition.append(True)
        return blank_result()

    monkeypatch.setattr(vision, "transcribe", transcribe)
    monkeypatch.setattr(
        "grading_agent.services.scoring.submission_grader.OpenAIGrader.grade",
        lambda *a, **kw: pytest.fail("Blank needs no scoring API"),
    )
    student = upload(workspace, mark(source))
    response = next(r for r in student["responses"] if r["question_id"] == q["id"])
    assert response["blank_confirmed"] and response["raw_text"] == ""
    assert (
        response["extraction_source"] == "local_layout_blank" and response["ocr_confidence"] is None
    )
    result = client.post(f"/api/submissions/{student['id']}/grade")
    assert result.status_code == 200, result.text
    grade = next(g for g in result.json()["grade_items"] if g["question_id"] == q["id"])
    assert grade["score"] == 0 and "confirmed_blank" in grade["reason_codes"]
    assert "extraction_uncertain" not in grade["reason_codes"]
    from grading_agent.app import prepare_submission

    prepare_submission(student["id"])
    assert len(recognition) == 1
    correction = client.post(
        f"/api/submissions/{student['id']}/answers",
        json={"answers": [{"question_id": q["id"], "raw_text": "42"}]},
    )
    assert correction.status_code == 200
    corrected = prepare_submission(student["id"])
    answer = next(r for r in corrected["responses"] if r["question_id"] == q["id"])
    assert answer["raw_text"] == "42" and not answer["blank_confirmed"]
    assert len(recognition) == 1


@pytest.mark.parametrize("mode", ["faint", "filled_reference", "failure"])
def test_uncertain_writing_and_failed_ocr_never_become_zero(workspace, monkeypatch, mode):
    client, a, qs, doc, source, calls = workspace
    generate(workspace)
    enable_visual(monkeypatch)
    student = mark(source)
    ImageDraw.Draw(student).text((130, 630), "1", fill=(245, 245, 245))

    def transcribe(*args):
        if mode == "failure":
            raise ValueError("识别失败")
        return {
            "text": "",
            "consistent": False,
            "blank_candidate": False,
            "issues": ["无法确认浅色笔迹"],
            "requests": [],
        }

    monkeypatch.setattr(vision, "transcribe", transcribe)
    if mode == "filled_reference":
        from grading_agent.services.recognition import local_answers

        original = local_answers.recognize_answers

        def filled(layout, *args, **kwargs):
            layout = copy.deepcopy(layout)
            layout["source_review"] = {"answer_content": "filled"}
            return original(layout, *args, **kwargs)

        monkeypatch.setattr(local_answers, "recognize_answers", filled)
    result = upload(workspace, student)
    assert not any(r["question_id"] == qs[1]["id"] for r in result["responses"])
    graded = client.post(f"/api/submissions/{result['id']}/grade").json()
    assert (
        next(g for g in graded["grade_items"] if g["question_id"] == qs[1]["id"])["score"] is None
    )


@pytest.mark.parametrize("second_blank", [True, False])
def test_multiple_blanks_keep_positions_and_only_all_blank_is_zero(
    workspace, monkeypatch, second_blank
):
    client, a, qs, doc, source, calls = workspace
    state = generate(workspace)
    enable_visual(monkeypatch)
    payload = dict(
        expected_template_id=state["template_id"],
        document_id=doc["id"],
        regions=[
            dict(page=1, x=0.1, y=0.5, width=0.3, height=0.1),
            dict(page=1, x=0.5, y=0.5, width=0.3, height=0.1),
        ],
    )
    assert (
        client.put(
            f"/api/assessments/{a['id']}/layout-template/regions/{qs[1]['id']}", json=payload
        ).status_code
        == 200
    )
    parts = [
        blank_result(),
        blank_result()
        if second_blank
        else {"text": "42", "consistent": True, "issues": [], "requests": []},
    ]
    monkeypatch.setattr(vision, "transcribe", lambda *args: parts.pop(0))
    student = upload(workspace, mark(source))
    response = next(r for r in student["responses"] if r["question_id"] == qs[1]["id"])
    assert response["blank_confirmed"] == second_blank
    assert response["raw_text"] == ("" if second_blank else "[未作答]\n42")


def test_local_ocr_empty_result_needs_pixel_proof(workspace, monkeypatch):
    client, a, qs, doc, source, calls = workspace
    generate(workspace)
    monkeypatch.setattr(
        "grading_agent.services.documents.ingestion.OCRAdapter.extract",
        lambda *args: ("", None, "paddleocr_failed"),
    )
    student = upload(workspace, mark(source))  # Synthetic answer region has no ink.
    response = next(r for r in student["responses"] if r["question_id"] == qs[1]["id"])
    assert response["blank_confirmed"] and response["extraction_source"] == "local_layout_blank"


@pytest.mark.parametrize("profile", ["legacy", "daily", "exam"])
def test_confirmed_blank_also_scores_zero_in_manual_mode_with_exam_review_preserved(
    monkeypatch, profile
):
    from grading_agent.services.scoring.submission_grader import _grade_question

    settings = replace(config.get_settings(), grading_profile=profile, review_sample_rate=0)
    monkeypatch.setattr(
        "grading_agent.services.scoring.submission_grader.OpenAIGrader.grade",
        lambda *args: pytest.fail("No scoring API"),
    )
    q = {
        "id": "q",
        "scoring_mode": "manual",
        "max_score": 5,
        "answer_key_state": "verified",
        "rule_config": {},
    }
    r = {
        "raw_text": "",
        "blank_confirmed": True,
        "ocr_confidence": None,
        "extraction_source": "local_layout_blank",
    }
    decision = _grade_question(q, r, settings, "synthetic")
    assert decision.score == 0 and "confirmed_blank" in decision.reason_codes
    assert "extraction_uncertain" not in decision.reason_codes
    if profile == "exam":
        assert decision.needs_review
