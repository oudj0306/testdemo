"""Local inference only; never create submissions, grades or training labels."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from grading_agent.config import get_settings
from grading_agent.services.recognition.layout.templates import layout_status
from grading_agent.services.recognition.local_vision import runtime, transcribe
from grading_agent.storage.repository import get_assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment")
    parser.add_argument("--questions", default="二-2,五-1,四-1")
    parser.add_argument(
        "--cases-file",
        type=Path,
        help="Local JSON cases: name, path, question, expected; never used as training labels",
    )
    parser.add_argument(
        "--model", help="Benchmark a downloaded local model without changing application settings"
    )
    parser.add_argument("--output", default="output/local-vision-verification.json")
    args = parser.parse_args()
    settings = replace(get_settings(), local_vision_enabled=True)
    if args.model:
        settings = replace(settings, local_vision_model=args.model)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.cases_file:
        if args.assessment:
            raise ValueError("Use either --assessment or --cases-file")
        specifications = json.loads(args.cases_file.read_text(encoding="utf-8"))
        if not isinstance(specifications, list) or not 1 <= len(specifications) <= 100:
            raise ValueError("Expected 1 to 100 cases")
        for spec in specifications:
            path = Path(spec["path"])
            if not path.is_absolute():
                path = args.cases_file.parent / path
            # Human truth is only compared AFTER inference, never sent to the model.
            question = {
                k: v
                for k, v in spec.get("question", {}).items()
                if k in {"question_type", "answer_form", "prompt"}
            }
            cases.append((spec["name"], path, question, spec.get("expected")))
    elif args.assessment:
        assessment = get_assessment(args.assessment)
        if not assessment:
            raise ValueError("Assessment not found")
        layout = layout_status(assessment, _include_pages=True)
        selected = set(args.questions.split(","))
        pages = {p["page"]: p for p in layout["pages"]}
        for q in assessment["questions"]:
            if q["question_no"] not in selected:
                continue
            item = next(i for i in layout["items"] if i["question_id"] == q["id"])
            for index, b in enumerate(item["regions"]):
                crop = destination.parent / (f"visual-case-{q['id']}-{index}.png")
                with Image.open(pages[b["page"]]["path"]) as image:
                    image.crop(
                        (
                            int(b["x"] * image.width),
                            int(b["y"] * image.height),
                            int((b["x"] + b["width"]) * image.width),
                            int((b["y"] + b["height"]) * image.height),
                        )
                    ).save(crop)
                cases.append((q["question_no"] + ":" + str(index + 1), crop, q, None))
    else:
        crop = destination.parent / "synthetic-verbatim-code.png"
        image = Image.new("RGB", (700, 240), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 36)
        expected = "x = 3\nif x > 2:\n    print(x)"
        draw.multiline_text((35, 25), expected, font=font, fill="black", spacing=10)
        image.save(crop)
        cases.append(("synthetic-code", crop, {"question_type": "code"}, expected))
    report = {
        "runtime": runtime(settings),
        "remote_vision_requests": 0,
        "writes_grades": False,
        "cases": [],
    }
    for name, path, question, expected in cases:
        started = time.perf_counter()
        result = transcribe(path, question, settings)
        value = {"case": name, "seconds": round(time.perf_counter() - started, 3), "result": result}
        if expected is not None:
            value["exact_match"] = result["text"] == expected
        report["cases"].append(value)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": name,
                    "seconds": value["seconds"],
                    "consistent": result["consistent"],
                    "issues": result["issues"],
                    "requests": len(result["requests"]),
                    "exact_match": value.get("exact_match"),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    print("Saved local diagnostics:", destination, flush=True)


if __name__ == "__main__":
    main()
