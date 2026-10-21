"""Answer-list and in-paper extraction keep numbering and visible inputs intact."""
from dataclasses import replace
import base64
import io
import json

import pypdfium2 as pdfium
from fastapi.testclient import TestClient
from PIL import Image

from grading_agent.config import get_settings
from grading_agent.services.answer_extractor import extract_answers_with_model
from grading_agent.services.assessment_compiler import compile_question_paper
from grading_agent.services.ingestion import IngestedDocument


def _questions():
    return [
        {'question_no': '一-1', 'prompt': '选择正确选项 A 或 B', 'answer_form': 'text'},
        {'question_no': '二-1', 'prompt': '写出完整计算过程', 'answer_form': 'text'},
    ]


def _mock_response(monkeypatch, module, result):
    calls = []

    def urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'output_text': json.dumps(result),
                                      'usage': {'input_tokens': 100, 'output_tokens': 20}}).encode())

    monkeypatch.setattr(module.urllib.request, 'urlopen', urlopen)
    return calls


def _pdf(path):
    with pdfium.PdfDocument.new() as pdf:
        for _ in range(2):
            pdf.new_page(300, 400).close()
        pdf.save(str(path))


def test_qwen_pdf_extraction_sends_visible_pages_not_ignored_pdf(monkeypatch, tmp_path):
    from grading_agent.services import answer_extractor

    path = tmp_path / 'student.pdf'
    _pdf(path)
    answers = [{'question_no': no, 'raw_text': 'B', 'confidence': 0.98, 'blank_confirmed': False}
               for no in ('一-1', '二-1')]
    calls = _mock_response(monkeypatch, answer_extractor, {'answers': answers})
    settings = replace(get_settings(), openai_api_key='test', openai_model='qwen3.8-max')
    document = IngestedDocument('digest', path, 'application/pdf', [], 2)
    result, _ = extract_answers_with_model(document, _questions(), settings, answer_layout='on_paper')
    assert result == answers
    content = calls[0]['input'][1]['content']
    assert not any(item['type'] == 'input_file' for item in content)
    assert [item['text'] for item in content if item['type'] == 'input_text' and item['text'].startswith('PDF 第')] == ['PDF 第 1 页：', 'PDF 第 2 页：']
    images = [item for item in content if item['type'] == 'input_image']
    assert len(images) == 2
    assert all(base64.b64decode(item['image_url'].split(',', 1)[1]).startswith(b'\x89PNG') for item in images)
    assert '印刷题干' in content[1]['text']


def test_answer_list_omits_full_question_prompts_from_model_input(monkeypatch, tmp_path):
    from grading_agent.services import answer_extractor

    path = tmp_path / 'answers.png'
    path.write_bytes(b'image')
    calls = _mock_response(monkeypatch, answer_extractor, {'answers': [
        {'question_no': '一-1', 'raw_text': 'A', 'confidence': 0.98, 'blank_confirmed': False},
        {'question_no': '二-1', 'raw_text': '2+2=4', 'confidence': 0.98, 'blank_confirmed': False},
    ]})
    settings = replace(get_settings(), openai_api_key='test', openai_model='qwen3.8-max')
    extract_answers_with_model(IngestedDocument('digest', path, 'image/png', []),
                               _questions(), settings, answer_layout='answer_list')
    content = calls[0]['input'][1]['content']
    index = json.loads(content[0]['text'].split('：', 1)[1])
    assert [item['question_no'] for item in index] == ['一-1', '二-1']
    assert all('prompt' not in item for item in index)
    assert '按题号排列的答案列表' in content[1]['text']


def test_qwen_pdf_question_compilation_receives_page_images(monkeypatch, tmp_path):
    from grading_agent.services import assessment_compiler

    path = tmp_path / 'paper.pdf'
    _pdf(path)
    calls = _mock_response(monkeypatch, assessment_compiler, {'questions': []})
    settings = replace(get_settings(), openai_api_key='test', openai_model='qwen3.8-max')
    compile_question_paper([{'kind': 'question_paper', 'media_type': 'application/pdf',
                             'source_path': str(path)}], settings)
    content = calls[0]['input'][1]['content']
    assert sum(item['type'] == 'input_image' for item in content) == 2
    assert not any(item['type'] == 'input_file' for item in content)


def test_compilation_uses_uploaded_answer_key_once_and_keeps_paper_primary(monkeypatch, tmp_path):
    from grading_agent.services import assessment_compiler

    paper = tmp_path / 'paper.png'
    answer_key = tmp_path / 'key.png'
    paper.write_bytes(b'paper-image')
    answer_key.write_bytes(b'answer-image')
    calls = _mock_response(monkeypatch, assessment_compiler, {'questions': []})
    settings = replace(get_settings(), openai_api_key='test')
    documents = [
        {'kind': 'question_paper', 'source_hash': 'paper', 'source_path': str(paper), 'media_type': 'image/png'},
        {'kind': 'answer_key', 'source_hash': 'key', 'source_path': str(answer_key), 'media_type': 'image/png'},
        {'kind': 'rubric', 'source_hash': 'key', 'source_path': str(answer_key), 'media_type': 'image/png'},
    ]
    compile_question_paper(documents, settings)
    content = calls[0]['input'][1]['content']
    images = [item for item in content if item['type'] == 'input_image']
    assert len(images) == 2
    assert base64.b64decode(images[0]['image_url'].split(',', 1)[1]) == b'paper-image'
    assert base64.b64decode(images[1]['image_url'].split(',', 1)[1]) == b'answer-image'
    labels = [item['text'] for item in content if item['type'] == 'input_text' and item['text'].startswith('资料类型：')]
    assert '原试卷' in labels[0]
    assert '教师参考答案、教师评分标准' in labels[1]
    assert '教师答案可能仅列各空答案' in content[0]['text']


def test_fill_missing_uses_uploaded_reference_without_replacing_existing_answer(monkeypatch, tmp_path):
    from grading_agent.services import assessment_compiler

    path = tmp_path / 'answer.png'
    path.write_bytes(b'key-image')
    calls = _mock_response(monkeypatch, assessment_compiler, {'items': [
        {'question_no': '1', 'answer_key': '4', 'rubric': [{'criterion': '正确', 'score': 5}]},
    ]})
    questions = [{'id': 'q1', 'question_no': '1', 'prompt': '2+2', 'max_score': 5,
                  'scoring_mode': 'rubric_assisted', 'answer_key': '原答案', 'rubric': []}]
    result = assessment_compiler.fill_missing_parts(questions,
        replace(get_settings(), openai_api_key='test'),
        [{'kind': 'answer_key', 'source_hash': 'key', 'source_path': str(path), 'media_type': 'image/png'}])
    assert result['updates'][0]['answer_key'] == '原答案'
    assert result['updates'][0]['rubric'] == [{'criterion': '正确', 'score': 5}]
    content = calls[0]['input'][1]['content']
    assert sum(item['type'] == 'input_image' for item in content) == 1
    assert any('教师参考答案' in item['text'] for item in content if item['type'] == 'input_text')


def test_uploaded_batch_preserves_on_paper_choice(monkeypatch, tmp_path):
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    from grading_agent.app import app

    image = io.BytesIO()
    Image.new('RGB', (100, 100), 'white').save(image, 'PNG')
    with TestClient(app) as client:
        assessment = client.post('/api/assessments', json={'title': '卷面作答'}).json()
        client.post(f"/api/assessments/{assessment['id']}/questions", json={
            'question_no': '1', 'prompt': '填空', 'question_type': 'numeric',
            'scoring_mode': 'numeric', 'max_score': 5, 'answer_key': '2',
        })
        client.post(f"/api/assessments/{assessment['id']}/publish")
        result = client.post('/api/batches', data={'assessment_id': assessment['id'],
                                                   'answer_layout': 'on_paper'},
                             files=[('files', ('student.png', image.getvalue(), 'image/png'))])
        assert result.status_code == 202
        submission_id = result.json()['items'][0]['submission_id']
        saved = client.get('/api/submissions/' + submission_id).json()
        assert saved['normalization_report']['answer_layout'] == 'on_paper'


def test_answer_list_and_on_paper_both_reach_normal_grading(monkeypatch, tmp_path):
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    from grading_agent import config
    from grading_agent.app import app
    from grading_agent.services import answer_extractor

    settings = replace(config.get_settings(), openai_api_key='test', openai_model='qwen3.8-max')
    monkeypatch.setattr(config, '_settings_cache', settings)
    calls = _mock_response(monkeypatch, answer_extractor, {'answers': [
        {'question_no': '1', 'raw_text': '2', 'confidence': 0.99, 'blank_confirmed': False},
    ]})
    with TestClient(app) as client:
        assessment = client.post('/api/assessments', json={'title': '两种答卷'}).json()
        client.post(f"/api/assessments/{assessment['id']}/questions", json={
            'question_no': '1', 'prompt': '1 + 1 = __', 'question_type': 'numeric',
            'scoring_mode': 'numeric', 'max_score': 5, 'answer_key': '2',
        })
        client.post(f"/api/assessments/{assessment['id']}/publish")
        for layout, color in [('answer_list', 'white'), ('on_paper', 'gray')]:
            image = io.BytesIO()
            Image.new('RGB', (100, 100), color).save(image, 'PNG')
            uploaded = client.post('/api/submissions', data={
                'assessment_id': assessment['id'], 'student_ref': layout,
                'answer_layout': layout,
            }, files={'file': (layout + '.png', image.getvalue(), 'image/png')})
            assert uploaded.status_code == 200, uploaded.text
            graded = client.post(f"/api/submissions/{uploaded.json()['id']}/grade")
            assert graded.status_code == 200, graded.text
            assert graded.json()['score_summary']['current_score'] == 5
            assert graded.json()['normalization_report']['answer_layout'] == layout
    assert len(calls) == 2
    assert '按题号排列的答案列表' in calls[0]['input'][1]['content'][1]['text']
    assert '直接在印刷试卷上作答' in calls[1]['input'][1]['content'][1]['text']


def test_existing_queue_database_adds_answer_layout_without_losing_jobs(monkeypatch, tmp_path):
    import sqlite3
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    database = tmp_path / 'grading.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE batch_jobs(item_id TEXT PRIMARY KEY,content BLOB NOT NULL,media_type TEXT,auto_grade INTEGER NOT NULL)')
        connection.execute("INSERT INTO batch_jobs VALUES ('old-job', X'01', 'image/png', 1)")
    from grading_agent.db import initialize_database
    initialize_database()
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT item_id,answer_layout FROM batch_jobs WHERE item_id='old-job'").fetchone()
    assert row == ('old-job', 'auto')
