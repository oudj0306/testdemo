import threading
import time

from fastapi.testclient import TestClient

from grading_agent.db import transaction
from grading_agent.repository import create_batch, get_batch
from grading_agent.services.batch_queue import BatchQueue, enqueue


def test_queue_returns_before_model_and_limits_parallelism(monkeypatch, tmp_path):
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    from grading_agent import app as module
    release = threading.Event()
    two_running = threading.Event()
    lock = threading.Lock()
    started = []

    def persist(*args):
        with lock:
            started.append(args[1])
            if len(started) == 2:
                two_running.set()
        assert release.wait(5)
        raise ValueError('测试识别失败')

    monkeypatch.setattr(module, 'persist_submission', persist)
    with TestClient(module.app) as client:
        assessment = client.post('/api/demo/bootstrap').json()
        try:
            response = client.post('/api/batches', data={'assessment_id': assessment['id'], 'auto_grade': 'true'},
                                   files=[('files', (f'{i}.png', b'test', 'image/png')) for i in range(3)])
            assert response.status_code == 202
            assert response.json()['status'] == 'processing'
            assert two_running.wait(3)
            assert len(started) == 2
            assert client.get('/api/health').status_code == 200
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get('/api/batches/' + response.json()['id']).json()
            if result['status'] != 'processing':
                break
            time.sleep(0.02)
        assert result['failed_count'] == 3
        assert all('测试识别失败' in item['error'] for item in result['items'])


def test_queue_restart_does_not_repeat_started_requests(monkeypatch, tmp_path):
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    from grading_agent.db import initialize_database
    from grading_agent.repository import create_assessment
    initialize_database()
    assessment = create_assessment('恢复测试', ['math'])
    batch = create_batch(assessment['id'], 2)
    enqueue(batch['id'], 'started.pdf', 'a', b'one', 'application/pdf', True)
    enqueue(batch['id'], 'queued.pdf', 'b', b'two', 'application/pdf', True)
    with transaction() as connection:
        connection.execute("UPDATE grading_batch_items SET status='grading' WHERE student_ref='a'")
    calls = []

    def persist(*args):
        calls.append(args[1])
        raise ValueError('fixture failure')

    queue = BatchQueue(persist)
    queue.start()
    try:
        deadline = time.monotonic() + 5
        while get_batch(batch['id'])['status'] == 'processing' and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        queue.close()
    assert calls == ['b']
    assert get_batch(batch['id'])['failed_count'] == 2


def test_draft_batch_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    from grading_agent.app import app
    with TestClient(app) as client:
        assessment = client.post('/api/assessments', json={'title': '未发布'}).json()
        result = client.post('/api/batches', data={'assessment_id': assessment['id']},
                             files=[('files', ('a.png', b'test', 'image/png'))])
        assert result.status_code == 400
        assert client.get('/api/batches', params={'assessment_id': assessment['id']}).json() == []


def test_pdf_without_regions_does_not_render(monkeypatch, tmp_path):
    monkeypatch.setenv('GRADING_DATA_DIR', str(tmp_path))
    import pypdfium2 as pdfium
    from grading_agent.config import get_settings
    from grading_agent.services.ingestion import ingest_document
    path = tmp_path / 'sample.pdf'
    with pdfium.PdfDocument.new() as pdf:
        page = pdf.new_page(100, 100)
        page.close()
        pdf.save(str(path))
    settings = get_settings()
    result = ingest_document(path.read_bytes(), 'application/pdf', settings, render_pages=False)
    assert result.source_page_count == 1
    assert result.pages == []
    assert not list(settings.page_dir.iterdir())
    rendered = ingest_document(path.read_bytes(), 'application/pdf', settings)
    assert len(rendered.pages) == 1
