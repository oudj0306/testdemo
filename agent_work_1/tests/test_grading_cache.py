from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3
import threading
import time

import pytest

from grading_agent.config import get_settings
from grading_agent.schemas import ScoreState
from grading_agent.services import llm_grader
from grading_agent.services.grading_cache import StableGradeCache
from grading_agent.services.llm_grader import GradingResponseError, LLMResult, OpenAIGrader
from grading_agent.services.review_policy import apply_review_policy


def question():
    return {
        "id": "q1", "assessment_id": "a1", "question_no": "1",
        "prompt": "解释计算过程", "max_score": 10, "question_type": "short_answer",
        "scoring_mode": "rubric_assisted", "answer_key": "学生证据",
        "answer_key_state": "verified", "rule_config": {},
        "rubric": [{"criterion": "过程正确", "score": 10}],
    }


def settings(**kwargs):
    return replace(get_settings(), openai_api_key="test", review_mode="selective", llm_grading_passes=3,
                   **kwargs)


def mock_consensus(monkeypatch, *, score=8, evidence="学生证据", review=False):
    calls = []

    def single(self, q, answer):
        calls.append((q, answer))
        return LLMResult(score, ["rubric_assisted"], [evidence] if evidence else [], review, 100, 20, 10)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    return calls


def test_persistent_exact_consensus_hit_is_auditable_and_free(monkeypatch):
    calls = mock_consensus(monkeypatch)
    config = settings()
    original = OpenAIGrader(config).grade(question(), "学生证据")
    # A new instance (as in a later request) reads persisted consensus, not a
    # single cached model pass repeated as fake independent agreement.
    reused = OpenAIGrader(config).grade(question(), "学生证据")
    assert len(calls) == 2
    assert original.usage == {"input_tokens": 200, "output_tokens": 20, "cached_tokens": 40}
    assert reused.usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    assert reused.score == original.score == 8
    assert not reused.needs_review
    assert reused.score_state == ScoreState.CONSENSUS_CANDIDATE
    assert "学生证据" in reused.evidence
    assert any(e.startswith("stable_grade_cache_hit=") and "independent_passes=2" in e for e in reused.evidence)
    assert any(e.startswith("cache_source_api_input_tokens=200") for e in reused.evidence)
    assert not any(e.startswith("api_input_tokens=") for e in reused.evidence)


@pytest.mark.parametrize("field,value", [
    ("assessment_id", "a2"), ("id", "q2"), ("prompt", "另一题"),
    ("answer_key", "另一答案"), ("max_score", 11), ("question_type", "essay"),
    ("scoring_mode", "manual"), ("rubric", [{"criterion": "另一评分点", "score": 10}]),
    ("rule_config", {"alternative": True}),
])
def test_question_changes_invalidate(monkeypatch, field, value):
    calls = mock_consensus(monkeypatch)
    grader = OpenAIGrader(settings())
    grader.grade(question(), "学生证据")
    grader.grade({**question(), field: value}, "学生证据")
    assert len(calls) == 4


@pytest.mark.parametrize("field,value,expected", [
    ("openai_model", "other-model", 4), ("openai_base_url", "https://different.invalid/v1", 4),
    ("enable_thinking", True, 4), ("llm_grading_passes", 2, 4),
    ("review_mode", "strict", 5),
])
def test_model_and_policy_changes_invalidate(monkeypatch, field, value, expected):
    calls = mock_consensus(monkeypatch)
    config = replace(settings(), enable_thinking=False)
    OpenAIGrader(config).grade(question(), "学生证据")
    OpenAIGrader(replace(config, **{field: value})).grade(question(), "学生证据")
    assert len(calls) == expected


@pytest.mark.parametrize("field", ["GRADE_INSTRUCTIONS", "GRADING_ALGORITHM_VERSION"])
def test_prompt_or_algorithm_changes_invalidate(monkeypatch, field):
    calls = mock_consensus(monkeypatch)
    grader = OpenAIGrader(settings())
    grader.grade(question(), "学生证据")
    monkeypatch.setattr(llm_grader, field, getattr(llm_grader, field) + " updated")
    grader.grade(question(), "学生证据")
    assert len(calls) == 4


@pytest.mark.parametrize("answer", [" 学生证据", "学生证据 ", "学生证据\n", "学生证据+补充过程"])
def test_answers_are_never_stripped_or_semantically_matched(monkeypatch, answer):
    calls = mock_consensus(monkeypatch)
    grader = OpenAIGrader(settings())
    grader.grade(question(), "学生证据")
    grader.grade(question(), answer)
    assert len(calls) == 4


@pytest.mark.parametrize("review,evidence", [(True, "学生证据"), (False, ""), (False, "伪造的证据")])
def test_review_missing_or_ungrounded_evidence_never_cached(monkeypatch, review, evidence):
    calls = mock_consensus(monkeypatch, review=review, evidence=evidence)
    grader = OpenAIGrader(settings())
    grader.grade(question(), "学生证据")
    grader.grade(question(), "学生证据")
    assert len(calls) == 4


def test_unstable_results_never_cached(monkeypatch):
    calls = []

    def single(*args):
        calls.append(1)
        return LLMResult(7 if len(calls) % 2 else 8, [], ["学生证据"], False)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    grader = OpenAIGrader(settings())
    assert grader.grade(question(), "学生证据").score_state == ScoreState.UNSTABLE
    assert grader.grade(question(), "学生证据").score_state == ScoreState.UNSTABLE
    assert len(calls) == 6


def test_concurrent_identical_answers_share_two_real_passes(monkeypatch):
    calls = []
    barrier = threading.Barrier(4)

    def single(*args):
        calls.append(1)
        time.sleep(0.04)
        return LLMResult(8, [], ["学生证据"], False, 100, 20, 10)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    config = settings()

    def run(_):
        barrier.wait(timeout=5)
        return OpenAIGrader(config).grade(question(), "学生证据")

    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions = list(pool.map(run, range(4)))
    assert len(calls) == 2
    assert all(d.score == 8 and not d.needs_review for d in decisions)
    assert sum(d.usage["input_tokens"] for d in decisions) == 200
    assert sum(any("stable_grade_cache_hit=" in e for e in d.evidence) for d in decisions) == 3


def test_failed_pass_keeps_successful_usage_and_is_not_cached(monkeypatch):
    calls = []

    def single(*args):
        calls.append(1)
        if len(calls) % 2 == 0:
            raise GradingResponseError("bad JSON", {"input_tokens": 40, "output_tokens": 6, "cached_tokens": 2})
        return LLMResult(8, [], ["学生证据"], False, 100, 20, 10)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    grader = OpenAIGrader(settings())
    for _ in range(2):
        decision = grader.grade(question(), "学生证据")
        assert decision.score is None and decision.needs_review
        assert decision.usage == {"input_tokens": 140, "output_tokens": 16, "cached_tokens": 22}
    assert len(calls) == 4


def test_out_of_range_response_still_counts_billed_usage(monkeypatch):
    calls = mock_consensus(monkeypatch, score=11)
    grader = OpenAIGrader(settings())
    for _ in range(2):
        decision = grader.grade(question(), "学生证据")
        assert decision.score_state == ScoreState.BLOCKED
        assert decision.usage == {"input_tokens": 100, "output_tokens": 10, "cached_tokens": 20}
    assert len(calls) == 2


def test_malformed_real_response_retains_known_usage(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"output_text": "invalid JSON", "usage": {
                "input_tokens": 90, "output_tokens": 7, "input_tokens_details": {"cached_tokens": 30},
            }}).encode()

    monkeypatch.setattr(llm_grader.urllib.request, "urlopen", lambda *a, **kw: Response())
    decision = OpenAIGrader(settings()).grade(question(), "学生证据")
    assert decision.score_state == ScoreState.BLOCKED
    assert decision.usage == {"input_tokens": 90, "output_tokens": 7, "cached_tokens": 30}


def test_cache_storage_failure_falls_back_to_grading(monkeypatch):
    calls = mock_consensus(monkeypatch)
    config = settings()
    # A path collision makes only the disposable cache unavailable.
    (config.data_dir / "grading_cache.sqlite3").mkdir()
    decision = OpenAIGrader(config).grade(question(), "学生证据")
    assert len(calls) == 2 and decision.score == 8 and not decision.needs_review


def test_corrupt_cache_entry_is_ignored(monkeypatch):
    calls = mock_consensus(monkeypatch)
    config = settings()
    grader = OpenAIGrader(config)
    grader.grade(question(), "学生证据")
    with sqlite3.connect(config.data_dir / "grading_cache.sqlite3") as db:
        db.execute("UPDATE stable_grades SET decision='invalid'")
    decision = grader.grade(question(), "学生证据")
    assert len(calls) == 4 and decision.score == 8


def create_review_database(config):
    with sqlite3.connect(config.database_path) as db:
        db.execute("CREATE TABLE grade_items (id TEXT, question_id TEXT)")
        db.execute("CREATE TABLE review_events (id TEXT, grade_item_id TEXT, created_at TEXT)")
        db.execute("INSERT INTO grade_items VALUES ('g1','q1')")


def record_review(config):
    with sqlite3.connect(config.database_path) as db:
        db.execute("INSERT INTO review_events VALUES ('r1','g1','2026-10-15T00:00:00Z')")


def test_manual_review_invalidates_previously_cached_consensus(monkeypatch):
    calls = mock_consensus(monkeypatch)
    config = settings()
    create_review_database(config)
    grader = OpenAIGrader(config)
    grader.grade(question(), "学生证据")
    record_review(config)
    decision = grader.grade(question(), "学生证据")
    assert len(calls) == 4
    assert not any("stable_grade_cache_hit=" in e for e in decision.evidence)


def test_review_while_grading_prevents_stale_cache_write(monkeypatch):
    config = settings()
    create_review_database(config)
    calls = []

    def single(*args):
        calls.append(1)
        if len(calls) == 1:
            record_review(config)
        return LLMResult(8, [], ["学生证据"], False)

    monkeypatch.setattr(OpenAIGrader, "_single_pass", single)
    grader = OpenAIGrader(config)
    grader.grade(question(), "学生证据")
    with sqlite3.connect(config.data_dir / "grading_cache.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM stable_grades").fetchone()[0] == 0
    grader.grade(question(), "学生证据")
    assert len(calls) == 4


def test_review_race_during_cache_read_bypasses_stale_hit(monkeypatch):
    calls = mock_consensus(monkeypatch)
    config = settings()
    grader = OpenAIGrader(config)
    revisions = iter(['same', 'same', 'same', 'same', 'same', 'changed', 'changed'])
    monkeypatch.setattr(llm_grader, 'review_revision', lambda *a: next(revisions, 'changed'))
    grader.grade(question(), "学生证据")
    grader.grade(question(), "学生证据")
    # The cached result was read but not returned after the revision changed;
    # two real consensus passes were performed again.
    assert len(calls) == 4


def test_cache_hit_still_obeys_per_submission_review_policy(monkeypatch):
    mock_consensus(monkeypatch)
    config = settings(review_sample_rate=1)
    grader = OpenAIGrader(config)
    grader.grade(question(), "学生证据")
    cached = grader.grade(question(), "学生证据")
    assert cached.usage["input_tokens"] == 0
    apply_review_policy(cached, question(), {"extraction_source": "manual"}, config, "student")
    assert cached.needs_review and "quality_spot_check" in cached.reason_codes
    # Policy edits must not mutate the stored model consensus.
    again = grader.grade(question(), "学生证据")
    assert not again.needs_review and "quality_spot_check" not in again.reason_codes


def test_single_pass_cannot_be_inserted_into_cache():
    config = settings()
    cache = StableGradeCache(config.data_dir)
    from grading_agent.services.math_grader import GradeDecision
    decision = GradeDecision(8, 10, ScoreState.CONSENSUS_CANDIDATE, False, [], ["学生证据"])
    cache.put("single", decision, 1)
    assert cache.get("single", 10) is None
