import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grading_agent import config
from grading_agent.schemas import QuestionCreate
from grading_agent.storage.db import encode_json, initialize_database, transaction
from grading_agent.storage.repository import (
    create_assessment,
    create_question,
    create_submission_record,
    get_submission,
    publish_assessment,
)
from tests.recognition.test_local_layout import cv as cv
from tests.recognition.test_local_layout import generate, mark, upload
from tests.recognition.test_local_layout import workspace as workspace


def seed(regions=None, pages=1):
    initialize_database()
    settings = config.get_settings()
    a = create_assessment("Read-only evidence", [])
    q = create_question(
        a["id"],
        QuestionCreate(
            question_no="二-1",
            prompt="Read",
            question_type="short_answer",
            scoring_mode="exact_text",
            answer_key="42",
            max_score=5,
            rule_config={"regions": regions} if regions else {},
        ).model_dump(mode="json"),
    )
    publish_assessment(a["id"])
    source = settings.upload_dir / "original.png"
    Image.new("RGB", (200, 300), "white").save(source)
    entries = []
    for n in range(pages):
        path = settings.page_dir / f"page-{n + 1}.png"
        Image.new("RGB", (200, 300), "white").save(path)
        entries.append(
            dict(
                page_no=n + 1,
                image_path=str(path),
                width=200,
                height=300,
                quality_state="ok",
                quality_report={},
            )
        )
    s = create_submission_record(
        a["id"], "synthetic", "hash", str(source), "image/png", {}, entries
    )
    return a, q, s


def test_linked_crop_is_padded_original_and_pages_do_not_change_grades(workspace, monkeypatch):
    client, a, qs, doc, source, calls = workspace
    generate(workspace)
    student = upload(workspace, mark(source))
    sid = student["id"]
    before = get_submission(sid)
    monkeypatch.setattr(
        "grading_agent.services.documents.ingestion.OCRAdapter.extract",
        lambda *a: pytest.fail("viewer must not run OCR"),
    )
    response = client.get(
        f"/api/submissions/{sid}/review-visual", params={"question_id": qs[1]["id"]}
    )
    assert response.status_code == 200, response.text
    manifest = response.json()
    assert manifest["default_mode"] == "crop" and manifest["model_requests"] == 0
    assert manifest["regions"][0]["page"] == 1 and len(manifest["crops"]) == 1
    assert "source_path" not in response.text and "image_path" not in response.text
    crop = client.get(manifest["crops"][0]["url"])
    assert crop.status_code == 200
    with Image.open(io.BytesIO(crop.content)) as image:
        assert image.width > 630 and image.height > 144  # Small context around every boundary.
    page = client.get(manifest["pages"][0]["url"])
    with Image.open(io.BytesIO(page.content)) as image:
        assert image.size == source.size
    after = get_submission(sid)
    assert (
        before["responses"] == after["responses"] and before["grade_items"] == after["grade_items"]
    )
    assert len(calls) == 1


def test_highlight_maps_back_to_shifted_student_page(workspace, cv):
    import numpy as np

    client, a, qs, doc, source, calls = workspace
    generate(workspace)
    moved = Image.fromarray(
        cv.warpAffine(
            np.asarray(mark(source)),
            np.float32([[1, 0, 8], [0, 1, 12]]),
            source.size,
            borderValue=(255, 255, 255),
        )
    )
    s = upload(workspace, moved)
    manifest = client.get(
        f"/api/submissions/{s['id']}/review-visual", params={"question_id": qs[1]["id"]}
    ).json()
    x, y = manifest["regions"][0]["polygon"][0]
    assert x == pytest.approx(0.1 + 8 / 900, abs=0.012)
    assert y == pytest.approx(0.5 + 12 / 1200, abs=0.012)
    # Historical reports have no transform; viewer can rebuild geometry locally.
    report = s["normalization_report"]
    report["local_recognition"]["pages"]["1"].pop("source_to_template", None)
    with transaction() as db:
        db.execute(
            "UPDATE submissions SET normalization_report=? WHERE id=?",
            (encode_json(report), s["id"]),
        )
    rebuilt = client.get(
        f"/api/submissions/{s['id']}/review-visual", params={"question_id": qs[1]["id"]}
    ).json()
    assert rebuilt["regions"][0]["polygon"][0][0] == pytest.approx(x, abs=0.005)


def test_unknown_regions_fall_back_to_full_page_and_foreign_question_is_rejected(monkeypatch):
    from grading_agent.app import app

    monkeypatch.setattr("grading_agent.app.BatchQueue.start", lambda self: None)
    a, q, s = seed()
    with TestClient(app) as client:
        data = client.get(
            f"/api/submissions/{s['id']}/review-visual", params={"question_id": q["id"]}
        ).json()
        assert data["default_mode"] == "page" and data["regions"] == [] and data["crops"] == []
        assert (
            client.get(f"/api/submissions/{s['id']}/review-visual?question_id=foreign").status_code
            == 404
        )
        assert (
            client.get(f"/api/submissions/{s['id']}/review-region?question_id=foreign").status_code
            == 404
        )
        assert client.get(f"/api/submissions/{s['id']}/review-page?page=99").status_code == 404


def test_all_continuation_regions_remain_separate_and_in_order(monkeypatch):
    from grading_agent.app import app

    monkeypatch.setattr("grading_agent.app.BatchQueue.start", lambda self: None)
    boxes = [
        dict(page=1, x=0.1, y=0.2, width=0.7, height=0.3),
        dict(page=2, x=0.1, y=0.1, width=0.7, height=0.5),
    ]
    a, q, s = seed(boxes, 2)
    with TestClient(app) as client:
        data = client.get(f"/api/submissions/{s['id']}/review-visual?question_id={q['id']}").json()
        assert [c["page"] for c in data["crops"]] == [1, 2]
        assert all(client.get(c["url"]).status_code == 200 for c in data["crops"])
        assert [r["page"] for r in data["regions"]] == [1, 2]


def test_external_image_paths_are_not_served(monkeypatch, tmp_path):
    from grading_agent.app import app

    monkeypatch.setattr("grading_agent.app.BatchQueue.start", lambda self: None)
    a, q, s = seed()
    outside = tmp_path.parent / "outside-review.png"
    Image.new("RGB", (30, 30), "black").save(outside)
    with transaction() as db:
        db.execute(
            "UPDATE submission_pages SET image_path=? WHERE submission_id=?",
            (str(outside), s["id"]),
        )
    with TestClient(app) as client:
        response = client.get(f"/api/submissions/{s['id']}/review-page?page=1")
        assert response.status_code == 404


def test_pdf_without_extraction_pages_has_read_only_page_preview(monkeypatch):
    import pypdfium2 as pdfium

    from grading_agent.app import app

    monkeypatch.setattr("grading_agent.app.BatchQueue.start", lambda self: None)
    a, q, _ = seed()
    path = config.get_settings().upload_dir / "original.pdf"
    with pdfium.PdfDocument.new() as pdf:
        for _ in range(2):
            pdf.new_page(200, 300).close()
        pdf.save(str(path))
    s = create_submission_record(
        a["id"], "synthetic", "pdf-hash", str(path), "application/pdf", {}, []
    )
    with TestClient(app) as client:
        data = client.get(f"/api/submissions/{s['id']}/review-visual?question_id={q['id']}").json()
        assert [p["page"] for p in data["pages"]] == [1, 2]
        response = client.get(data["pages"][1]["url"])
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as image:
            assert image.size == (300, 450)


def test_stale_template_shows_original_page_without_false_highlight(workspace):
    client, a, qs, doc, source, calls = workspace
    generate(workspace)
    s = upload(workspace, mark(source))
    client.put("/api/questions/" + qs[1]["id"], json={**qs[1], "prompt": "Changed question"})
    data = client.get(f"/api/submissions/{s['id']}/review-visual?question_id={qs[1]['id']}").json()
    assert data["regions"] == [] and data["default_mode"] == "page"
    assert data["pages"] and data["note"]
