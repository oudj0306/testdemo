"""Extraction savings preserve full context and never invent missing answers."""
import base64
from dataclasses import replace
import io
import json

import pytest

from grading_agent.config import get_settings
from grading_agent.services.answer_extractor import extract_answers_with_model, reusable_answer_ids
from grading_agent.services.ingestion import IngestedDocument


def questions():
    return [
        {'id': f'q{i}', 'question_no': str(i), 'prompt': f'题目 {i} 完整上下文，包含前后依赖和限定条件',
         'answer_form': 'text', 'rule_config': {}}
        for i in range(1, 4)
    ]


def answer(number='2'):
    return {'question_no': number, 'raw_text': '学生完整作答', 'confidence': 0.98, 'blank_confirmed': False}


def mock_model(monkeypatch, answers):
    from grading_agent.services import answer_extractor
    requests = []

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'output_text': json.dumps({'answers': answers}),
                                    'usage': {'input_tokens': 100, 'output_tokens': 20}}).encode())

    monkeypatch.setattr(answer_extractor.urllib.request, 'urlopen', urlopen)
    return requests


def test_targeted_output_retains_full_questions_and_original_image(monkeypatch, tmp_path):
    source = tmp_path / 'answer.png'
    source.write_bytes(b'unchanged-original-image')
    document = IngestedDocument('digest', source, 'image/png', [])
    calls = mock_model(monkeypatch, [answer()])
    settings = replace(get_settings(), openai_api_key='test')
    result, usage = extract_answers_with_model(document, questions(), settings, target_question_nos=['2'])
    assert result == [answer()]
    assert usage['output_tokens'] == 20
    content = calls[0]['input'][1]['content']
    full_index = json.loads(content[0]['text'].split('：', 1)[1])
    assert [q['prompt'] for q in full_index] == [q['prompt'] for q in questions()]
    assert '本次只返回' in content[2]['text'] and '["2"]' in content[2]['text']
    image = content[-1]
    assert image['detail'] == 'high'
    assert base64.b64decode(image['image_url'].split(',', 1)[1]) == source.read_bytes()
    extract_answers_with_model(document, questions(), settings)
    assert calls[0]['prompt_cache_key'] == calls[1]['prompt_cache_key']
    assert calls[0]['max_output_tokens'] == calls[1]['max_output_tokens']


@pytest.mark.parametrize('returned', [[], [answer('1')], [answer(), answer()]])
def test_incomplete_or_wrong_target_cannot_be_accepted(monkeypatch, tmp_path, returned):
    calls = mock_model(monkeypatch, returned)
    document = IngestedDocument('hash', tmp_path / 'unused.png', 'image/png', [])
    with pytest.raises(ValueError):
        extract_answers_with_model(document, questions(), replace(get_settings(), openai_api_key='test'),
                                   local_text='2. 学生完整作答', target_question_nos=['2'])
    assert len(calls) == 1


def test_empty_target_skips_model_and_file_read(monkeypatch, tmp_path):
    calls = mock_model(monkeypatch, [])
    result, usage = extract_answers_with_model(
        IngestedDocument('hash', tmp_path / 'absent.png', 'image/png', []), questions(),
        replace(get_settings(), openai_api_key='test'), target_question_nos=[])
    assert result == [] and not any(usage.values()) and not calls


def test_missing_api_key_returns_the_normal_tuple_without_reading_file(tmp_path):
    result, usage = extract_answers_with_model(
        IngestedDocument('hash', tmp_path / 'absent.png', 'image/png', []), questions(),
        replace(get_settings(), openai_api_key=None))
    assert result == [] and usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def test_unknown_target_is_rejected_before_model_call(monkeypatch, tmp_path):
    calls = mock_model(monkeypatch, [])
    with pytest.raises(ValueError, match='待识别题号'):
        extract_answers_with_model(
            IngestedDocument('hash', tmp_path / 'absent.png', 'image/png', []), questions(),
            replace(get_settings(), openai_api_key='test'), target_question_nos=['missing'])
    assert not calls


@pytest.mark.parametrize('raw,blank,confidence,source,reusable', [
    ('完整答案', False, 0.99, 'openai_vision', True),
    ('', True, 0.99, 'openai_vision', True),
    ('', False, 0.99, 'openai_vision', False),
    ('矛盾', True, 0.99, 'openai_vision', False),
    ('', True, 0.5, 'openai_vision', False),
    ('答案', False, None, 'openai_vision', False),
    ('答案', False, float('inf'), 'openai_vision', False),
    ('答案', False, float('nan'), 'openai_vision', False),
    ('答案', False, 2.0, 'openai_vision', False),
    ('人工答案', False, None, 'manual', True),
    ('', True, None, 'manual', True),
])
def test_reuse_only_reliable_answers(raw, blank, confidence, source, reusable):
    response = {'question_id': 'q1', 'raw_text': raw, 'blank_confirmed': blank,
                'ocr_confidence': confidence, 'extraction_source': source}
    assert ('q1' in reusable_answer_ids(questions(), [response], get_settings())) is reusable


def test_reuse_respects_question_specific_confidence_threshold():
    qs = questions()
    qs[0]['rule_config'] = {'minimum_ocr_confidence': 0.99}
    response = {'question_id': 'q1', 'raw_text': '答案', 'blank_confirmed': False,
                'ocr_confidence': 0.97, 'extraction_source': 'openai_vision'}
    assert not reusable_answer_ids(qs, [response], get_settings())


def create_saved_submission():
    from grading_agent.db import initialize_database
    from grading_agent.repository import create_assessment, create_question, create_submission_record, get_assessment, publish_assessment
    initialize_database()
    assessment = create_assessment('定向重试', [])
    for number in range(1, 5):
        create_question(assessment['id'], {
            'question_no': str(number), 'prompt': f'完整题目{number}', 'question_type': 'short_answer',
            'capture_mode': 'text', 'answer_form': 'text', 'scoring_mode': 'rubric_assisted',
            'max_score': 5, 'answer_key': '参考答案', 'rule_config': {'region': {'page': 1}},
        })
    publish_assessment(assessment['id'])
    assessment = get_assessment(assessment['id'])
    settings = get_settings()
    source = settings.upload_dir / 'answer.png'
    submission = create_submission_record(assessment['id'], 'student', 'digest', str(source), 'image/png', {}, [], status='uploaded')
    return assessment, submission, source


def test_retry_only_unresolved_answer_preserves_saved_text_blank_and_manual(monkeypatch):
    from grading_agent import app as module
    from grading_agent.repository import upsert_response, get_submission
    assessment, submission, source = create_saved_submission()
    qs = assessment['questions']
    upsert_response(submission['id'], qs[0]['id'], '已识别答案', 0.99, False, 'openai_vision')
    upsert_response(submission['id'], qs[1]['id'], '', 0.99, True, 'openai_vision')
    upsert_response(submission['id'], qs[2]['id'], '人工答案', None, False, 'manual')
    original = get_submission(submission['id'])['responses']
    settings = replace(get_settings(), openai_api_key='test')
    monkeypatch.setattr(module, 'get_settings', lambda: settings)
    monkeypatch.setattr(module, 'ingest_document', lambda *a, **kw: IngestedDocument('digest', source, 'image/png', []))
    cropped_questions = []
    model_targets = []

    def crop(pages, pending, settings):
        cropped_questions.extend(q['question_no'] for q in pending)
        return {qs[3]['id']: source}

    def model(document, full_questions, settings, local_text=None, target_question_nos=None, answer_layout='auto'):
        assert full_questions == qs
        assert answer_layout == 'auto'
        model_targets.extend(target_question_nos)
        return [answer('4')], {'input_tokens': 100, 'output_tokens': 20, 'cached_tokens': 0}

    monkeypatch.setattr(module, 'crop_question_regions', crop)
    monkeypatch.setattr(module.OCRAdapter, 'extract', lambda *a: ('', 0.1, 'paddleocr'))
    monkeypatch.setattr(module, 'extract_answers_with_model', model)
    result = module.persist_submission(assessment, 'student', b'content', 'image/png', submission['id'])
    assert cropped_questions == ['4'] and model_targets == ['4']
    saved = {r['question_id']: r for r in result['responses']}
    for response in original:
        assert saved[response['question_id']] == response
    assert saved[qs[3]['id']]['raw_text'] == '学生完整作答'


def test_prepare_reuses_all_answers_without_reopening_source(monkeypatch):
    from grading_agent import app as module
    from grading_agent.db import transaction
    from grading_agent.repository import upsert_response
    assessment, submission, source = create_saved_submission()
    for q in assessment['questions']:
        upsert_response(submission['id'], q['id'], '', 0.99, True, 'openai_vision')
    with transaction() as db:
        db.execute("UPDATE submissions SET status='extraction_failed' WHERE id=?", (submission['id'],))
    assert not source.exists()

    def unexpected(*a, **kw):
        pytest.fail('已确认作答不应再次识别')

    monkeypatch.setattr(module, 'persist_submission', unexpected)
    result = module.prepare_submission(submission['id'])
    assert result['status'] == 'extracted'
    assert len(result['responses']) == 4


def test_teacher_edit_during_model_request_is_preserved(monkeypatch):
    from grading_agent import app as module
    from grading_agent.repository import upsert_response
    assessment, submission, source = create_saved_submission()
    q = assessment['questions'][0]
    settings = replace(get_settings(), openai_api_key='test')
    monkeypatch.setattr(module, 'get_settings', lambda: settings)
    monkeypatch.setattr(module, 'ingest_document', lambda *a, **kw: IngestedDocument('digest', source, 'image/png', []))
    monkeypatch.setattr(module, 'crop_question_regions', lambda *a: {})

    def model(*args, **kwargs):
        upsert_response(submission['id'], q['id'], '教师校正', None, False, 'manual')
        return [answer('1')], {'input_tokens': 100, 'output_tokens': 20, 'cached_tokens': 0}

    monkeypatch.setattr(module, 'extract_answers_with_model', model)
    result = module.persist_submission(assessment, 'student', b'content', 'image/png', submission['id'])
    assert result['responses'][0]['raw_text'] == '教师校正'
    assert result['responses'][0]['extraction_source'] == 'manual'


@pytest.mark.parametrize('kind,mode', [('single_choice','choice_set'), ('multiple_choice','choice_set'), ('custom','choice_set')])
def test_choice_plain_ocr_is_not_reused_as_selected_marks(kind, mode):
    from grading_agent.services.answer_extractor import is_choice_question
    question = {**questions()[0], 'question_type': kind, 'scoring_mode': mode}
    response = {'question_id': question['id'], 'raw_text': 'ABCD', 'blank_confirmed': False,
                'ocr_confidence': .999, 'extraction_source': 'paddleocr_gpu:0'}
    assert is_choice_question(question)
    assert question['id'] not in reusable_answer_ids([question], [response], get_settings())
    response['extraction_source'] = 'manual'
    assert question['id'] in reusable_answer_ids([question], [response], get_settings())


@pytest.mark.parametrize('old_region', [False, True])
def test_choice_upload_without_manual_boxes_extracts_then_grades_with_rules(monkeypatch, old_region):
    from fastapi.testclient import TestClient
    from PIL import Image
    from grading_agent import config
    from grading_agent.app import app
    from grading_agent.services.ingestion import OCRAdapter
    from grading_agent.services.llm_grader import OpenAIGrader
    monkeypatch.setattr(config, '_settings_cache', replace(get_settings(), openai_api_key='synthetic',
                        ocr_provider='paddleocr', local_ocr_first=True))
    monkeypatch.setattr(OCRAdapter, 'extract', lambda *a: pytest.fail('Choice marks must not use plain OCR'))
    monkeypatch.setattr(OpenAIGrader, 'grade', lambda *a: pytest.fail('Choice scoring must use rules'))
    calls = mock_model(monkeypatch, [{'question_no':'1','raw_text':'B','confidence':.99,'blank_confirmed':False}])
    picture=io.BytesIO()
    Image.new('RGB',(400,600),'white').save(picture,format='PNG')
    with TestClient(app) as client:
        a=client.post('/api/assessments',json={'title':'Synthetic choice','subject_codes':[]}).json()
        region={'region':{'page':1,'x':.1,'y':.1,'width':.5,'height':.2}} if old_region else {}
        q=client.post(f"/api/assessments/{a['id']}/questions",json={
            'question_no':'1','prompt':'Select A/B/C/D','question_type':'single_choice',
            'scoring_mode':'choice_set','max_score':5,'answer_key':'B','rule_config':region})
        assert q.status_code==200
        assert client.post(f"/api/assessments/{a['id']}/publish").status_code==200
        result=client.post('/api/submissions',data={'assessment_id':a['id'],'student_ref':'synthetic','answer_layout':'answer_list'},
                           files={'file':('synthetic.png',picture.getvalue(),'image/png')})
        assert result.status_code==200, result.text
        result=result.json()
        assert result['responses'][0]['raw_text']=='B'
        assert result['responses'][0]['extraction_source']=='openai_vision'
        graded=client.post(f"/api/submissions/{result['id']}/grade")
        assert graded.status_code==200, graded.text
        assert graded.json()['grade_items'][0]['score']==5
        assert graded.json()['grade_items'][0]['grader_kind']=='rules'
        assert len(calls)==1
