from __future__ import annotations

import base64
import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..model_catalog import apply_provider_options
from .ingestion import IngestedDocument, OCRAdapter, docx_embedded_images, extract_reference_text
from .legacy_word import extract_doc_text
from .llm_client import parse_json_object, response_output_text, response_usage
from .model_transport import model_request
from .review_policy import confidence_threshold
from .vision_input import add_visual_document


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_no": {"type": "string"},
                    "raw_text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "blank_confirmed": {"type": "boolean"},
                },
                "required": ["question_no", "raw_text", "confidence", "blank_confirmed"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

ANSWER_LAYOUTS = {"auto", "answer_list", "on_paper"}


def validate_answer_layout(value: str) -> str:
    if value not in ANSWER_LAYOUTS:
        raise ValueError("答卷形式必须是自动判断、仅答案列表或卷面作答")
    return value


def layout_instructions(layout: str) -> str:
    validate_answer_layout(layout)
    common = (
        "只转写学生实际填写、书写、勾选或输入的作答，不解题、不补全、不评分。"
        "印刷题干、选项文字、示例、页眉及标准答案不是学生作答；选择题只记学生明确选中的选项。"
        "同题有多个空或小问时保留原有顺序和标号，主观题保留完整步骤。"
        "只有亲眼看到该题完整作答位置确实空白，才可设置 blank_confirmed=true；"
        "若题目、页码或作答区域未找到、看不清或无法区分印刷与填写内容，返回空 raw_text、"
        "blank_confirmed=false、低 confidence，绝不能把缺失页面当作交白卷。"
    )
    if layout == "answer_list":
        return "本答卷是按题号排列的答案列表，没有重印题干。按题号或清晰的小问顺序对应固定清单。" + common
    if layout == "on_paper":
        return "本答卷直接在印刷试卷上作答。逐题寻找空格、选项标记和题下手写/键入部分，严格区分印刷原题与新增作答。" + common
    return (
        "答卷可能是按题号列出的答案列表，也可能直接在印刷试卷上作答；"
        "先根据可见版式判断每道题的位置，不要假设所有学生使用同一答题卡。" + common
    )


class ExtractedAnswer(BaseModel):
    model_config = ConfigDict(strict=True)
    question_no: str
    raw_text: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    blank_confirmed: bool


def validate_answers(value, questions):
    if not isinstance(value, list):
        raise ValueError('识别结果 answers 必须是数组')
    allowed = {q['question_no'] for q in questions}
    seen, answers = set(), []
    for raw in value:
        item = ExtractedAnswer.model_validate(raw)
        if item.question_no not in allowed or item.question_no in seen:
            raise ValueError('识别结果包含未知或重复题号，不能自动评分')
        if item.blank_confirmed and item.raw_text.strip():
            raise ValueError('识别结果同时声明空白和非空作答')
        seen.add(item.question_no)
        answers.append(item.model_dump())
    return answers


def reusable_answer_ids(questions, responses, settings):
    """Reuse only explicit human input or reliable, non-contradictory extraction."""
    by_id = {question['id']: question for question in questions}
    reusable = set()
    for response in responses:
        question_id = response['question_id']
        if question_id not in by_id:
            continue
        if response.get('extraction_source') == 'manual':
            reusable.add(question_id)
            continue
        text_present = bool(str(response.get('raw_text') or '').strip())
        blank_confirmed = bool(response.get('blank_confirmed'))
        # Empty and unconfirmed is an extraction gap, not a student's blank answer.
        if text_present == blank_confirmed:
            continue
        confidence = response.get('ocr_confidence')
        try:
            valid_confidence = confidence is not None and math.isfinite(float(confidence)) and 0 <= float(confidence) <= 1
            if valid_confidence and float(confidence) >= confidence_threshold(by_id[question_id], settings):
                reusable.add(question_id)
        except (TypeError, ValueError):
            continue
    return reusable


def extract_answers_with_model(
    document: IngestedDocument,
    questions: list[dict[str, Any]],
    settings: Settings,
    local_text: str | None = None,
    *,
    target_question_nos: list[str] | None = None,
    answer_layout: str = "auto",
) -> list[dict[str, Any]]:
    validate_answer_layout(answer_layout)
    requested = None if target_question_nos is None else set(target_question_nos)
    allowed = {item['question_no'] for item in questions}
    if requested is not None and not requested <= allowed:
        raise ValueError('待识别题号不在试卷题目清单中')
    if requested == set():
        return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    if not settings.openai_api_key:
        return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    question_index = [
        ({"question_no": item["question_no"], "answer_form": item["answer_form"]}
         if answer_layout == "answer_list" else
         {"question_no": item["question_no"], "prompt": item["prompt"], "answer_form": item["answer_form"]})
        for item in questions
    ]
    static_context = json.dumps(question_index, ensure_ascii=False, separators=(",", ":"))
    cache_key = hashlib.sha256((answer_layout + ':' + static_context).encode("utf-8")).hexdigest()[:48]
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": "固定题号清单：" + static_context},
        {
            "type": "input_text",
            "text": layout_instructions(answer_layout),
        },
    ]
    if requested is not None:
        # Keep every complete prompt for disambiguation; limit output, not context.
        ordered_targets = [item['question_no'] for item in questions if item['question_no'] in requested]
        content.append({
            'type': 'input_text',
            'text': '本次只返回以下待识别题号，每题返回一项，其余题目已识别，不要重复输出：'
                    + json.dumps(ordered_targets, ensure_ascii=False, separators=(',', ':'))
                    + '。仍结合完整题目清单核对题号；未找到或看不清不得声明已确认空白，应降低 confidence。',
        })
    if local_text is not None:
        content.append({'type': 'input_text', 'text': '本地 OCR 转写（可能存在漏识别，无法可靠对应题号时降低 confidence，不得求解补全）：\n' + local_text})
    elif document.media_type == 'application/msword':
        content.append({'type': 'input_text', 'text': '学生 DOC 答卷：\n' + extract_doc_text(document.source_path)})
    elif document.media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        value, _ = extract_reference_text(document, OCRAdapter(settings.ocr_provider))
        content.append({"type": "input_text", "text": "学生Word答卷：\n" + value})
        for image_type, image_bytes in docx_embedded_images(document.source_path):
            encoded_image = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {"type": "input_image", "image_url": f"data:{image_type};base64,{encoded_image}", "detail": "high"}
            )
    else:
        add_visual_document(content, document.source_path, document.media_type, settings, document.pages)
    payload = {
        "model": settings.openai_model,
        "store": False,
        "prompt_cache_key": f"answer-extract-{cache_key}",
        "max_output_tokens": 8192,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "你只负责忠实转写学生答案，不负责求解或评分。必须只返回一个 JSON 对象，字段为 answers 数组，每项含 question_no、raw_text、confidence(0到1的数字)、blank_confirmed(布尔)，不要输出任何其他文字或代码块标记。"}],
            },
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "student_answers",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            }
        },
    }
    apply_provider_options(payload, settings.openai_base_url, settings.enable_thinking)
    request = model_request(payload, settings)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    output_text = response_output_text(data)
    if not output_text:
        raise RuntimeError("模型响应缺少 output_text")
    expected_questions = questions if requested is None else [item for item in questions if item['question_no'] in requested]
    answers = validate_answers(parse_json_object(output_text)["answers"], expected_questions)
    if requested is not None and {item['question_no'] for item in answers} != requested:
        raise ValueError('识别结果缺少待识别题号，请重新识别或人工校对')
    return answers, response_usage(data)
