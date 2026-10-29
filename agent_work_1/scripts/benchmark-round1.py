"""Offline deterministic benchmark. Synthetic token units are NOT provider tokens.

Example: python scripts/benchmark-round1.py --profile daily --data-dir output/bench-daily
Baseline: add --source D:/ou/agent/src --profile legacy --data-dir output/bench-original
No network access or model downloads. Each data directory must be new.
"""

import argparse
import io
import json
import os
import re
import sys
from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


def units(text):
    return len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["legacy", "daily"], default="daily")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--papers", type=int, default=50)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if data_dir.exists():
        parser.error("Data directory must be new to avoid overwriting business data")
    if not 1 <= args.papers <= 1000:
        parser.error("papers must be 1..1000")
    os.environ["GRADING_DATA_DIR"] = str(data_dir)
    os.environ.pop("OPENAI_API_KEY", None)
    sys.path.insert(0, str(args.source.resolve()))
    from grading_agent import config

    if (args.source / "grading_agent/storage/db.py").is_file():
        from grading_agent.storage.db import initialize_database
        from grading_agent.storage.repository import (
            create_assessment,
            create_question,
            create_submission_record,
            get_batch,
            get_submission,
            publish_assessment,
            token_usage_for_assessment,
            upsert_response,
        )
    else:
        from grading_agent.db import initialize_database
        from grading_agent.repository import (
            create_assessment,
            create_question,
            create_submission_record,
            get_batch,
            get_submission,
            publish_assessment,
            token_usage_for_assessment,
            upsert_response,
        )

    # The benchmark deliberately supports both the frozen baseline and the
    # reorganized implementation selected by --source.
    if (args.source / "grading_agent/services/workflows/batch_queue.py").is_file():
        from grading_agent.services.workflows.batch_queue import BatchQueue, enqueue_saved
    else:
        from grading_agent.services.batch_queue import BatchQueue, enqueue_saved

    initialize_database()
    kwargs = dict(
        openai_api_key="offline-placeholder",
        llm_grading_passes=2,
        review_sample_rate=0,
        review_mode="selective",
    )
    if "grading_profile" in {f.name for f in fields(config.Settings)}:
        kwargs["grading_profile"] = args.profile
    elif args.profile != "legacy":
        parser.error("Selected source does not support daily profile")
    settings = replace(config.get_settings(), **kwargs)
    config._settings_cache = settings
    a = create_assessment("Synthetic offline benchmark", [])
    questions = []
    for i, mode in enumerate(["numeric", "exact_text", "rubric_assisted", "rubric_assisted"], 1):
        questions.append(
            create_question(
                a["id"],
                dict(
                    question_no=str(i),
                    prompt="Explain this synthetic concept using the given criterion. " * 8,
                    question_type="short_answer",
                    capture_mode="text",
                    answer_form="text",
                    scoring_mode=mode,
                    max_score=10,
                    answer_key="42" if i == 1 else "reference",
                    answer_key_state="verified",
                    rule_config={},
                    rubric=[
                        {"criterion": "Give a correct explanation for the concept.", "score": 10}
                    ],
                ),
            )
        )
    publish_assessment(a["id"])
    papers = []
    for i in range(args.papers):
        s = create_submission_record(
            a["id"], f"synthetic-{i}", f"hash-{i}", "unused.png", "image/png", {}, []
        )
        for index, q in enumerate(questions, 1):
            answer = (
                "42"
                if index == 1
                else "reference"
                if index == 2
                else (f"explanation-{i % 5}" if index == 3 else f"explanation-{i}")
            )
            upsert_response(s["id"], q["id"], answer, None, False, "manual")
        papers.append(s)
    calls, usage, max_answers = 0, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, 0

    def urlopen(request, timeout):
        nonlocal calls, max_answers
        payload = json.loads(request.data)
        text = payload["input"][-1]["content"][-1]["text"]
        if payload["text"]["format"]["name"] == "bulk_grades":
            items = json.loads(text.split("：", 1)[1])
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
            max_answers = max(max_answers, len(items))
        else:
            answer = text.split("：", 1)[1]
            result = dict(score=8, reason_codes=[], evidence=[answer], needs_review=False)
            max_answers = max(max_answers, 1)
        output_text = json.dumps(result, ensure_ascii=False)
        measured = {
            "input_tokens": units(json.dumps(payload["input"], ensure_ascii=False)),
            "output_tokens": units(output_text),
            "cached_tokens": 0,
        }
        calls += 1
        for k, v in measured.items():
            usage[k] += v
        return io.BytesIO(json.dumps({"output_text": output_text, "usage": measured}).encode())

    queue = BatchQueue(lambda *a: (_ for _ in ()).throw(AssertionError("No extraction needed")))
    started = perf_counter()
    with patch("urllib.request.urlopen", urlopen):
        batch = enqueue_saved(a["id"], [s["id"] for s in papers])
        while job := queue.claim():
            if "daily_group" in job:
                queue.process_daily(job["daily_group"])
            else:
                queue.process(job)
    elapsed = perf_counter() - started
    results = [get_submission(s["id"]) for s in papers]
    scores = [[g["score"] for g in s["grade_items"]] for s in results]
    report = {
        "profile": args.profile,
        "source": str(args.source.resolve()),
        "papers": args.papers,
        "elapsed_seconds": round(elapsed, 4),
        "model_requests": calls,
        "max_answers_per_request": max_answers,
        "synthetic_usage": usage,
        "recorded_usage": token_usage_for_assessment(a["id"])["total"],
        "batch_success": get_batch(batch["id"])["success_count"],
        "score_vectors": scores,
        "note": "Offline mock; synthetic lexical units, not actual provider tokens. No OCR or network latency. Not a real speed/accuracy estimate.",
    }
    report_path = data_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "score_vectors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
