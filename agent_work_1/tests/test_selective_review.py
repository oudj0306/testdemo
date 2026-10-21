from dataclasses import replace

import pytest

from grading_agent.config import get_settings
from grading_agent.schemas import ScoreState
from grading_agent.services.llm_grader import LLMResult, OpenAIGrader
from grading_agent.services.review_policy import apply_review_policy


def question():
    return {'id': 'q1', 'max_score': 10, 'answer_key_state': 'verified',
            'rubric': [{'criterion': '解释正确', 'score': 10}], 'rule_config': {}}


def run_grading(monkeypatch, scores, *, review=False, evidence=True, mode='selective', passes=3):
    settings = replace(get_settings(), openai_api_key='test', review_mode=mode,
                       llm_grading_passes=passes, review_sample_rate=0)
    calls = []
    def single(*args):
        score = scores[len(calls)]
        calls.append(score)
        return LLMResult(score, [], ['学生证据'] if evidence else [], review)
    grader = OpenAIGrader(settings)
    monkeypatch.setattr(grader, '_single_pass', single)
    return grader.grade(question(), '学生证据'), settings, calls


def test_two_consistent_scores_auto_pass(monkeypatch):
    decision, settings, calls = run_grading(monkeypatch, [8, 8, 8])
    apply_review_policy(decision, question(), {'extraction_source': 'manual'}, settings, 's1')
    assert calls == [8, 8]
    assert decision.score == 8
    assert decision.score_state == ScoreState.AUTO_STABLE
    assert not decision.needs_review


@pytest.mark.parametrize('threshold,expected', [(0.85, False), (0.95, True)])
def test_configurable_extraction_threshold(monkeypatch, threshold, expected):
    decision,settings,_=run_grading(monkeypatch,[8,8])
    settings=replace(settings,review_confidence_threshold=threshold)
    apply_review_policy(decision,question(),{'extraction_source':'openai_vision','ocr_confidence':0.90},settings,'s1')
    assert decision.needs_review is expected


def test_one_configured_pass_still_requires_two_for_auto(monkeypatch):
    decision, _, calls = run_grading(monkeypatch, [8, 8], passes=1)
    assert calls == [8, 8]
    assert not decision.needs_review


@pytest.mark.parametrize('scores,review,evidence,reason', [
    ([8, 7, 8], False, True, 'score_disagreement'),
    ([8, 8], True, True, 'model_requested_review'),
    ([8, 8], False, False, 'grading_evidence_missing'),
    ([11], False, True, 'llm_request_failed'),
    ([float('nan')], False, True, 'llm_request_failed'),
])
def test_anomalies_require_review(monkeypatch, scores, review, evidence, reason):
    decision, _, _ = run_grading(monkeypatch, scores, review=review, evidence=evidence)
    assert decision.needs_review
    assert reason in decision.reason_codes


@pytest.mark.parametrize('kind', ['candidate', 'low_confidence', 'unknown_confidence', 'rubric', 'strict', 'force_review', 'sample'])
def test_policy_gates(monkeypatch, kind):
    decision, settings, _ = run_grading(monkeypatch, [8, 8])
    q = question()
    response = {'extraction_source': 'openai_vision', 'ocr_confidence': 0.99}
    if kind == 'candidate': q['answer_key_state'] = 'candidate'
    if kind == 'low_confidence': response['ocr_confidence'] = 0.8
    if kind == 'unknown_confidence': response['ocr_confidence'] = None
    if kind == 'rubric': q['rubric'] = []
    if kind == 'strict': settings = replace(settings, review_mode='strict')
    if kind == 'force_review': q['rule_config']['force_review'] = True
    if kind == 'sample': settings = replace(settings, review_sample_rate=1)
    apply_review_policy(decision, q, response, settings, 'student')
    assert decision.needs_review
    assert decision.score_state != ScoreState.AUTO_STABLE


def test_auto_pass_updates_report_and_history(monkeypatch):
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from grading_agent.app import app
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.setenv('LLM_EXTRACT_ANSWERS', 'false')
    monkeypatch.setattr(OpenAIGrader, '_single_pass', lambda *args: LLMResult(8, [], ['学生证据'], False))
    with TestClient(app) as client:
        a = client.post('/api/assessments', json={'title': '自动定分'}).json()
        q = {**question(), 'question_no': '1', 'prompt': '解释', 'question_type': 'short_answer',
             'scoring_mode': 'rubric_assisted', 'answer_key': '参考解释'}
        client.post(f"/api/assessments/{a['id']}/questions", json=q)
        client.post(f"/api/assessments/{a['id']}/publish")
        image = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(image, 'PNG')
        s = client.post('/api/submissions', data={'assessment_id': a['id'], 'student_ref': 's1'}, files={'file': ('s.png', image.getvalue(), 'image/png')}).json()
        client.post(f"/api/submissions/{s['id']}/answers", json={'answers': [{'question_no': '1', 'raw_text': '学生证据'}]})
        result = client.post(f"/api/submissions/{s['id']}/grade").json()
        assert result['status'] == 'finalized'
        assert result['grade_items'][0]['score_state'] == 'AUTO_STABLE'
        assert client.get(f"/api/assessments/{a['id']}/report").json()['average'] == 8
        assert client.get(f"/api/submissions/{s['id']}/history").json()[0]['snapshot']['response']['raw_text'] == '学生证据'
        original_id = result['grade_items'][0]['id']
        # UI pending-only regrade must not spend calls or supersede an accepted item.
        def unexpected(*args):
            raise AssertionError('accepted question must not be graded again')
        monkeypatch.setattr(OpenAIGrader, '_single_pass', unexpected)
        again = client.post(f"/api/submissions/{s['id']}/grade?pending_only=true").json()
        assert again['grade_items'][0]['id'] == original_id
        assert again['status'] == 'finalized'


def test_manual_review_preserved_when_another_question_regraded():
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from grading_agent.app import app
    with TestClient(app) as client:
        a = client.post('/api/demo/bootstrap').json()
        image = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(image, 'PNG')
        s = client.post('/api/submissions', data={'assessment_id': a['id'], 'student_ref': 's1'}, files={'file': ('s.png', image.getvalue(), 'image/png')}).json()
        client.post(f"/api/submissions/{s['id']}/answers", json={'answers': [{'question_no': '1', 'raw_text': 'B'}]})
        grades = client.post(f"/api/submissions/{s['id']}/grade").json()['grade_items']
        q1 = next(g for g in grades if g['question_no'] == '1')
        q2 = next(g for g in grades if g['question_no'] == '2')
        client.post(f"/api/grade-items/{q2['id']}/review", json={'final_score': 3, 'reviewer_ref': 't', 'reason': '人工检查'})
        result = client.post(f"/api/submissions/{s['id']}/grade?pending_only=true").json()
        assert next(g for g in result['grade_items'] if g['question_no'] == '1')['id'] == q1['id']
        preserved = next(g for g in result['grade_items'] if g['question_no'] == '2')
        assert preserved['id'] == q2['id']
        assert preserved['effective_score'] == 3


@pytest.mark.parametrize('patch', [{'confidence': float('nan')}, {'confidence': 2}, {'blank_confirmed': 'false'}, {'blank_confirmed': True}, {'question_no': 'unknown'}])
def test_invalid_extraction_not_auto_accepted(patch):
    from grading_agent.services.answer_extractor import validate_answers
    answer = {'question_no': '1', 'raw_text': 'abc', 'confidence': 0.99, 'blank_confirmed': False}
    with pytest.raises(ValueError):
        validate_answers([{**answer, **patch}], [{'question_no': '1'}])


def test_concurrent_answer_edit_cannot_be_overwritten(monkeypatch):
    from grading_agent.db import initialize_database
    from grading_agent.repository import create_assessment, create_question, publish_assessment, create_submission_record, upsert_response, get_submission
    from grading_agent.services.grading import grade_submission
    initialize_database()
    a = create_assessment('并发校对', [])
    q = create_question(a['id'], {**question(), 'question_no': '1', 'prompt': '解释', 'question_type': 'short_answer', 'scoring_mode': 'rubric_assisted', 'answer_key': '答案', 'capture_mode': 'text', 'answer_form': 'text'})
    publish_assessment(a['id'])
    s = create_submission_record(a['id'], 's', 'hash', 'test.png', 'image/png', {}, [])
    upsert_response(s['id'], q['id'], '旧作答', None, False, 'manual')
    settings = replace(get_settings(), openai_api_key='test')
    def grade(*args):
        upsert_response(s['id'], q['id'], '新校对作答', None, False, 'manual')
        return LLMResult(8, [], ['旧作答'], False)
    monkeypatch.setattr(OpenAIGrader, '_single_pass', grade)
    with pytest.raises(ValueError, match='已修改'):
        grade_submission(s['id'], settings)
    result = get_submission(s['id'])
    assert result['responses'][0]['raw_text'] == '新校对作答'
    assert not result['grade_items']
