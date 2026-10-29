"""Daily profile only: independent answer IDs, shared rubric, bounded recovery."""

import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from typing import Any

from grading_agent.config import Settings
from grading_agent.model_catalog import apply_provider_options
from grading_agent.repository import record_token_usage
from grading_agent.services.model_io.llm_client import (
    parse_json_object,
    response_output_text,
    response_usage,
)
from grading_agent.services.model_io.model_transport import model_request
from grading_agent.services.scoring.decisions import GradeDecision
from grading_agent.services.scoring.grading_cache import (
    StableGradeCache,
    cache_key,
    review_revision,
)
from grading_agent.services.scoring.llm_grader import (
    GRADE_INSTRUCTIONS,
    GRADE_SCHEMA,
    OpenAIGrader,
    daily_decision,
)
from grading_agent.services.telemetry import event, request_json

VERSION = "daily-bulk-v3-evidence-review"


def _request(question, items, settings):
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    **GRADE_SCHEMA,
                    "properties": {"answer_id": {"type": "string"}, **GRADE_SCHEMA["properties"]},
                    "required": ["answer_id", *GRADE_SCHEMA["required"]],
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }
    fixed = json.dumps(
        {k: question[k] for k in ("prompt", "answer_key", "rubric", "max_score")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    instructions = GRADE_INSTRUCTIONS.split("必须只返回")[0] + (
        "分别独立批改每份答案，不相互比较、不参考其他学生答案。"
        "每项原样返回 answer_id；只返回 JSON 对象 results 数组，每项含 "
        "answer_id、score、reason_codes、evidence、needs_review。证据每条不超过200字符。"
    )
    payload = {
        "model": settings.openai_model,
        "store": False,
        "prompt_cache_key": "daily-" + hashlib.sha256((VERSION + fixed).encode()).hexdigest()[:48],
        "max_output_tokens": 4096,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": instructions}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "固定题目与评分规则：" + fixed},
                    {
                        "type": "input_text",
                        "text": "待评答案：" + json.dumps(items, ensure_ascii=False),
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bulk_grades",
                "strict": True,
                "schema": schema,
            }
        },
    }
    apply_provider_options(payload, settings.openai_base_url, settings.enable_thinking)
    return request_json(
        model_request(payload, settings),
        settings,
        "grade",
        45,
        answer_count=len(items),
        profile="daily",
    )


def grade_daily_items(
    question: dict[str, Any], items: list[dict[str, str]], settings: Settings
) -> dict[str, GradeDecision]:
    """items: [{answer_id: opaque ID, answer: exact transcription}]. Maximum 8."""
    if not isinstance(items, list) or not 1 <= len(items) <= 8:
        raise ValueError("Batch must contain 1..8 unique answer IDs")
    answer_ids = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each batch item must be an answer object")
        answer_id = item.get("answer_id")
        if not isinstance(answer_id, str) or not answer_id.strip() or answer_id in answer_ids:
            raise ValueError("Batch must contain 1..8 unique answer IDs")
        if not isinstance(item.get("answer"), str):
            raise ValueError("Each batch answer must be text")
        answer_ids.add(answer_id)
    answers, output, leaders = {}, {}, {}
    for item in items:
        answers[item["answer_id"]] = item["answer"]
        leaders.setdefault(item["answer"], item["answer_id"])
    unique = [{"answer_id": aid, "answer": answer} for answer, aid in leaders.items()]
    grader, cache = OpenAIGrader(settings), StableGradeCache(settings.data_dir)
    revision = review_revision(settings, question.get("id", ""))
    keys = {
        i["answer_id"]: cache_key(settings, question, i["answer"], VERSION, revision)
        for i in unique
    }
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    # Sorted leases prevent two overlapping batches deadlocking on inverse answer order.
    with ExitStack() as leases:
        held = {
            key: leases.enter_context(cache.single_flight(key))
            for key in sorted(set(k for k in keys.values() if k))
        }
        for item in unique:
            key = keys[item["answer_id"]]
            hit = (
                cache.get(key, float(question["max_score"]), allow_single=True)
                if key and held[key]
                else None
            )
            if hit and revision == review_revision(settings, question.get("id", "")):
                output[item["answer_id"]] = hit
                event("cache_hit")
        pending = [i for i in unique if i["answer_id"] not in output]
        if len(pending) > 1 and grader.available:
            try:
                data, request_id = _request(question, pending, settings)
                usage = response_usage(data)
                rows = parse_json_object(response_output_text(data)).get("results")
                if not isinstance(rows, list):
                    raise ValueError("Missing results")
                counts = Counter(
                    r.get("answer_id")
                    for r in rows
                    if isinstance(r, dict) and isinstance(r.get("answer_id"), str)
                )
                expected = {i["answer_id"] for i in pending}
                invalid = 0
                for row in rows:
                    if not isinstance(row, dict):
                        invalid += 1
                        continue
                    aid = row.get("answer_id")
                    if not isinstance(aid, str) or aid not in expected or counts[aid] != 1:
                        invalid += 1
                        continue
                    try:
                        output[aid] = daily_decision(
                            question,
                            answers[aid],
                            row.get("score"),
                            row.get("reason_codes"),
                            row.get("evidence"),
                            row.get("needs_review"),
                            settings,
                            request_id,
                        )
                    except (ValueError, TypeError):
                        invalid += 1
                if invalid or expected - output.keys():
                    event(
                        "validation_failure",
                        success=False,
                        details={
                            "invalid_items": invalid,
                            "missing_items": len(expected - output.keys()),
                            "request_id": request_id,
                        },
                    )
            except (RuntimeError, ValueError, KeyError, TypeError, OSError):
                event("validation_failure", success=False)
            attempts = 1
        else:
            attempts = 2
        for item in pending:
            aid = item["answer_id"]
            if aid not in output:
                output[aid] = (
                    grader.grade_daily(question, item["answer"], attempts=attempts)
                    if grader.available
                    else grader.grade(question, item["answer"])
                )
                for k, v in (output[aid].usage or {}).items():
                    usage[k] += v
        for item in unique:
            aid, key = item["answer_id"], keys[item["answer_id"]]
            if key and held[key] and revision == review_revision(settings, question.get("id", "")):
                cache.put(key, output[aid], 1, allow_single=True)
    if any(usage.values()):
        record_token_usage(question.get("assessment_id"), None, "grade", **usage)
    result = {}
    for item in items:
        leader = leaders[item["answer"]]
        result[item["answer_id"]] = deepcopy(output[leader])
        result[item["answer_id"]].usage = None  # shared request already recorded once
        if item["answer_id"] != leader:
            result[item["answer_id"]].evidence.append("exact_answer_reused_in_batch")
            event("cache_hit", details={"within_batch": True})
    return result
