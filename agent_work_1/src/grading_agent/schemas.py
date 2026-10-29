from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class QuestionType(StrEnum):
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    PROGRAMMING = "programming"
    READING = "reading"
    AUTO = "auto"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    SOLUTION_SET = "solution_set"
    CALCULATION = "calculation"
    APPLICATION = "application"
    PROOF = "proof"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


class ScoringMode(StrEnum):
    AUTO = "auto"
    EXACT_TEXT = "exact_text"
    CHOICE_SET = "choice_set"
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    RUBRIC_ASSISTED = "rubric_assisted"
    MANUAL = "manual"


class ScoreState(StrEnum):
    AUTO_SINGLE = "AUTO_SINGLE"
    AUTO_STABLE = "AUTO_STABLE"
    REPRODUCIBLE_EXACT = "REPRODUCIBLE_EXACT"
    CONSENSUS_CANDIDATE = "CONSENSUS_CANDIDATE"
    UNSTABLE = "UNSTABLE"
    BLOCKED = "BLOCKED"
    HUMAN_FINAL = "HUMAN_FINAL"


class AssessmentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    subject_codes: list[str] = Field(default_factory=list)


class QuestionCreate(BaseModel):
    question_no: str = Field(min_length=1, max_length=40)
    prompt: str = Field(min_length=1)
    question_type: QuestionType
    capture_mode: str = "short_handwriting"
    answer_form: str = "text"
    scoring_mode: ScoringMode
    max_score: float = Field(gt=0, le=1000)
    answer_key: Any
    answer_key_state: str = "verified"
    rule_config: dict[str, Any] = Field(default_factory=dict)
    rubric: list[dict[str, Any]] = Field(default_factory=list)
    review_policy: str = "auto_if_verified"

    @model_validator(mode="after")
    def valid_region(self):
        if self.rule_config.get("fixed_answers") is not None:
            from grading_agent.services.scoring.fixed_answers import validate_fixed_answers

            validate_fixed_answers(self.rule_config["fixed_answers"], self.max_score)
        if self.rule_config.get("region"):
            self.rule_config["region"] = RegionInput.model_validate(
                self.rule_config["region"]
            ).model_dump(exclude_none=True)
        return self


class AnswerInput(BaseModel):
    question_id: str | None = None
    question_no: str | None = None
    raw_text: str = ""
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    blank_confirmed: bool = False

    @model_validator(mode="after")
    def require_question_reference(self) -> "AnswerInput":
        if not self.question_no and not self.question_id:
            raise ValueError("question_id 或 question_no 至少提供一个")
        if self.blank_confirmed and self.raw_text.strip():
            raise ValueError("未作答不能同时填写作答文字")
        return self


class AnswersUpsert(BaseModel):
    answers: list[AnswerInput]


class AnswerConfirmation(BaseModel):
    raw_text: str = Field(default="", max_length=30000)
    blank_confirmed: bool = False

    @model_validator(mode="after")
    def require_answer_or_blank(self):
        if self.blank_confirmed == bool(self.raw_text.strip()):
            raise ValueError("请填写作答内容，或明确确认未作答，不能同时选择")
        return self


class ReviewCreate(BaseModel):
    reviewer_ref: str = Field(default="", max_length=100)
    final_score: float = Field(ge=0)
    reason: str = Field(default="", max_length=1000)
    expected_review_id: str | None = None
    check_revision: bool = False


class KnowledgeApprove(BaseModel):
    response_id: str
    kind: str = "approved_example"
    redacted_text: str = Field(min_length=1)
    approved_by: str = Field(min_length=1, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentTextUpdate(BaseModel):
    extracted_text: str = Field(min_length=1)


class LLMSettingsUpdate(BaseModel):
    hybrid_fallback_enabled: bool | None = None
    grading_profile: Literal["legacy", "daily", "exam"] | None = None
    api_key: str | None = Field(default=None, max_length=500)
    provider: str | None = Field(default=None, max_length=30)
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=300)
    llm_grading_passes: int = Field(default=1, ge=1, le=3)
    llm_extract_answers: bool = True
    enable_thinking: bool = False
    ocr_provider: Literal["none", "paddleocr"] | None = None
    ocr_device: Literal["auto", "cpu", "gpu:0"] | None = None
    local_ocr_first: bool | None = None
    review_mode: Literal["selective", "strict"] | None = None
    review_sample_rate: float | None = Field(default=None, ge=0, le=1)
    review_confidence_threshold: float | None = Field(default=None, ge=0.5, le=0.99)


class GradeBatchRequest(BaseModel):
    submission_ids: list[str] = Field(default_factory=list, max_length=1000)


class RegionInput(BaseModel):
    page: int = Field(ge=1)
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    page_aspect: float | None = Field(default=None, gt=0, le=20)

    @model_validator(mode="after")
    def within_page(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("答题区域超出页面")
        return self


class TrainingApproval(BaseModel):
    redacted_text: str = Field(min_length=1)
    approved_by: str = Field(min_length=1, max_length=100)
    expected_review_id: str | None = None


class BulkReviewRequest(BaseModel):
    grade_item_ids: list[str] = Field(min_length=1, max_length=1000)
    reviewer_ref: str = Field(default="", max_length=100)
    reason: str = Field(default="", max_length=1000)
