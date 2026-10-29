import copy

import pytest
from PIL import Image, ImageDraw

from grading_agent.services.recognition.layout.geometry import calibrate_regions, padded_frame
from tests.recognition.test_local_layout import box, generate
from tests.recognition.test_local_layout import workspace as workspace


def test_printed_rows_and_continuation_panel(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "sheet.png"
    image = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(image)
    d.rectangle((50, 100, 450, 420), outline="black", width=2)
    items = []
    for i, y in enumerate([170, 230, 290, 350]):
        d.line((80 if i == 0 else 110, y, 420, y), fill="black", width=2)
        items.append(
            dict(question_id=str(i), method="text", regions=[box(0.1, (y - 5) / 1400, 0.3, 0.02)])
        )
    d.rectangle((550, 500, 950, 1300), outline="black", width=2)
    items.append(dict(question_id="continued", method="text", regions=[box(0.55, 0.36, 0.4, 0.12)]))
    image.save(path)
    calibrated, changed = calibrate_regions(items, [dict(page=1, path=path)])
    assert len(changed) == 5
    for i in range(4):
        b = calibrated[i]["regions"][0]
        assert b["y"] < items[i]["regions"][0]["y"]
        assert b["x"] < 0.08 and b["width"] > 0.34
        assert b["y"] * 1400 < [170, 230, 290, 350][i] - 60
        assert (b["y"] + b["height"]) * 1400 > [170, 230, 290, 350][i] + 2
    assert calibrated[-1]["regions"][0]["height"] > 0.55
    panel = calibrated[-1]["regions"][0]
    assert panel["x"] * 1000 < 550 and panel["y"] * 1400 < 500
    assert (panel["x"] + panel["width"]) * 1000 > 950
    assert (panel["y"] + panel["height"]) * 1400 > 1300
    repeated, _ = calibrate_regions(calibrated, [dict(page=1, path=path)])
    assert repeated == calibrated  # enlarged short rows must not turn into whole-panel crops
    assert items[-1]["regions"][0]["height"] == 0.12  # caller untouched
    protected, _ = calibrate_regions(items, [dict(page=1, path=path)], ["continued"])
    assert protected[-1] == items[-1]


def test_margin_at_page_edge_stays_inside_image():
    assert padded_frame((0, 0, 1000, 1400), 1000, 1400) == dict(x=0, y=0, width=1, height=1)
    b = padded_frame((10, 20, 300, 400), 1000, 1400)
    assert b == dict(x=0.007, y=17 / 1400, width=0.306, height=406 / 1400)


def test_no_frame_keeps_candidates(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "blank.png"
    Image.new("RGB", (1000, 1400), "white").save(path)
    items = [dict(question_id="q", method="text", regions=[box(0.1, 0.2, 0.4, 0.2)])]
    assert calibrate_regions(items, [dict(page=1, path=path)]) == (items, set())


def test_edit_multiple_regions_preserves_question_and_rejects_stale_writer(workspace):
    client, a, qs, doc, *_ = workspace
    old = generate(workspace)
    qid = qs[1]["id"]
    url = f"/api/assessments/{a['id']}/layout-template/regions/{qid}"
    before = client.get("/api/assessments/" + a["id"]).json()["questions"]
    regions = [box(0.1, 0.5, 0.3, 0.1), box(0.5, 0.5, 0.3, 0.1)]
    payload = dict(
        expected_template_id=old["template_id"],
        document_id=doc["id"],
        regions=regions,
        confirmed_indices=[0],
    )
    response = client.put(url, json=payload)
    assert response.status_code == 200, response.text
    new = response.json()
    assert new["confirmed_items"] == 0
    assert next(i for i in new["items"] if i["question_id"] == qid)["regions"] == regions
    assert client.put(url, json=payload).status_code == 409
    payload.update(expected_template_id=new["template_id"], confirmed_indices=[1])
    confirmed = client.put(url, json=payload).json()
    assert confirmed["confirmed_items"] == 1
    assert client.get("/api/assessments/" + a["id"]).json()["questions"] == before
    # Removing one part retains the other part and its explicit confirmation.
    payload.update(
        expected_template_id=confirmed["template_id"], regions=regions[1:], confirmed_indices=[]
    )
    removed = client.put(url, json=payload).json()
    assert next(i for i in removed["items"] if i["question_id"] == qid)["regions"] == regions[1:]
    assert removed["confirmed_items"] == 1


@pytest.mark.parametrize("problem", ["page", "source", "index", "outside"])
def test_bad_edit_is_atomic(workspace, problem):
    client, a, qs, doc, *_ = workspace
    old = generate(workspace)
    payload = dict(
        expected_template_id=old["template_id"],
        document_id=doc["id"],
        regions=[box(0.1, 0.5, 0.4, 0.1)],
        confirmed_indices=[0],
    )
    if problem == "page":
        payload["regions"][0]["page"] = 2
    if problem == "source":
        payload["document_id"] = "other"
    if problem == "index":
        payload["confirmed_indices"] = [2]
    if problem == "outside":
        payload["regions"][0]["width"] = 1
    response = client.put(
        f"/api/assessments/{a['id']}/layout-template/regions/{qs[1]['id']}", json=payload
    )
    assert response.status_code in (409, 422), response.text
    assert (
        client.get(f"/api/assessments/{a['id']}/layout-template").json()["template_id"]
        == old["template_id"]
    )


def test_recalibration_protects_manual_positions(workspace, monkeypatch):
    client, a, qs, doc, *_ = workspace
    old = generate(workspace)
    payload = dict(
        expected_template_id=old["template_id"],
        document_id=doc["id"],
        regions=[box(0.1, 0.5, 0.4, 0.1)],
        confirmed_indices=[0],
    )
    assert (
        client.put(
            f"/api/assessments/{a['id']}/layout-template/regions/{qs[1]['id']}", json=payload
        ).status_code
        == 200
    )

    def check(items, pages, protected):
        assert qs[1]["id"] in protected
        return copy.deepcopy(items), set()

    monkeypatch.setattr(
        "grading_agent.services.recognition.layout.geometry.calibrate_regions", check
    )
    response = client.post(f"/api/assessments/{a['id']}/layout-template/calibrate")
    assert response.status_code == 200, response.text
    assert response.json()["confirmed_items"] == 1 and response.json()["adjusted_items"] == 0
