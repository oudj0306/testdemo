from types import SimpleNamespace

import pytest


def test_local_grid_requires_complete_unique_printed_numbers(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from grading_agent.services.recognition.choice_grid import locate_printed_grid

    image = np.full((1500, 1200), 255, np.uint8)
    cv2.rectangle(image, (50, 200), (1100, 340), 0, 2)
    for row in range(2):
        for col in range(5):
            x = 70 + col * 205
            y = 220 + row * 65
            cv2.rectangle(image, (x, y), (x + 5, y + 15), 0, -1)
            for opt in range(4):
                cv2.rectangle(image, (x + 22 + opt * 28, y), (x + 36 + opt * 28, y + 15), 0, 1)
    path = tmp_path / "grid.png"
    cv2.imwrite(str(path), image)
    settings = SimpleNamespace(ocr_provider="none", ocr_device="cpu")
    qs = [dict(id=str(i), question_no="一-" + str(i)) for i in range(1, 11)]
    numbers = iter(range(1, 11))
    result = locate_printed_grid(
        [dict(path=str(path), page=1)], qs, settings, lambda image: (str(next(numbers)), 0.99)
    )
    assert result and len(result) == 10
    assert [i["question_id"] for i in result] == [str(i) for i in range(1, 11)]
    assert (
        locate_printed_grid([dict(path=str(path), page=1)], qs, settings, lambda image: ("1", 0.99))
        is None
    )


def test_printed_number_confidence_excludes_unrelated_mark_noise(monkeypatch, tmp_path):
    from grading_agent.services.recognition import ocr_runtime as runtime

    class Engine:
        def predict(self, path):
            return [{"res": {"rec_texts": ["1[A]", "[B]", "0"], "rec_scores": [0.96, 0.2, 0.12]}}]

    monkeypatch.setattr(runtime, "_engines", {"cpu": (Engine(), "cpu")})
    monkeypatch.setattr(runtime, "_initialization_errors", set())
    assert runtime.recognize(tmp_path / "number.png", "cpu", leading_number=True)[:2] == (
        "1[A]",
        0.96,
    )
    assert runtime.recognize(tmp_path / "answer.png", "cpu")[1] == 0.12
