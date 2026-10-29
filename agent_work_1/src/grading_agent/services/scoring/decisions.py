"""评分结果契约；规则与模型评分共用，不执行存储或模型调用。"""

from dataclasses import dataclass, field

from grading_agent.schemas import ScoreState


@dataclass
class GradeDecision:
    """score=None 表示缺分，不能转换为零；needs_review 不阻止已有数值出分。"""

    score: float | None
    max_score: float
    score_state: ScoreState
    needs_review: bool
    reason_codes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    grader_kind: str = "rules"
    grader_version: str = "math-v1"
    usage: dict[str, int] | None = None
