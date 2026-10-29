import importlib.util
import json
from pathlib import Path

from grading_agent.config import get_settings


def test_case_truth_is_not_given_to_model_and_model_override_is_not_saved(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/verify-local-vision.py"
    spec = importlib.util.spec_from_file_location("visual_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = get_settings()
    manifest = tmp_path / "cases.json"
    output = tmp_path / "report.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "example",
                    "path": "crop.png",
                    "expected": "42",
                    "question": {
                        "question_type": "numeric",
                        "answer_key": "SECRET_REFERENCE",
                        "expected": "SECRET_TRUTH",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def transcribe(path, question, selected):
        calls.append((path, question, selected.local_vision_model))
        return {"text": "42", "consistent": True, "issues": [], "requests": []}

    monkeypatch.setattr(module, "transcribe", transcribe)
    monkeypatch.setattr(module, "runtime", lambda selected: {"model": selected.local_vision_model})
    monkeypatch.setattr(
        "sys.argv",
        [
            str(script),
            "--cases-file",
            str(manifest),
            "--model",
            "local-test",
            "--output",
            str(output),
        ],
    )
    module.main()
    assert calls == [(tmp_path / "crop.png", {"question_type": "numeric"}, "local-test")]
    assert get_settings().local_vision_model == settings.local_vision_model
    assert json.loads(output.read_text(encoding="utf-8"))["cases"][0]["exact_match"] is True
