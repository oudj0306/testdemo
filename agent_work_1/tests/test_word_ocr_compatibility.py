import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pypdfium2 as pdfium
import pytest

from grading_agent import config
from grading_agent.db import initialize_database
from grading_agent.repository import create_assessment, upsert_assessment_document
from grading_agent.services import document_renderer as renderer, runtime_paths


def pdf_file(path, count=2):
    with pdfium.PdfDocument.new() as pdf:
        for _ in range(count):
            pdf.new_page(200, 300).close()
        pdf.save(str(path))
    return path


def test_word_converter_is_cached_and_keeps_source(monkeypatch, tmp_path):
    source = tmp_path/'paper.doc'
    source.write_bytes(b'synthetic-input-not-an-office-macro')
    calls = []
    monkeypatch.setattr(renderer, 'renderer_status', lambda: dict(available=True, backend='office', executable='powershell'))
    def convert(command, **kwargs):
        calls.append(command)
        assert '-InputPath' in command and '-OutputPath' in command
        pdf_file(Path(command[-1]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(renderer.subprocess, 'run', convert)
    output = renderer.word_to_pdf(source, tmp_path)
    assert renderer.word_to_pdf(source, tmp_path) == output and len(calls) == 1
    assert source.read_bytes() == b'synthetic-input-not-an-office-macro'
    with pdfium.PdfDocument(output) as pdf:
        assert len(pdf) == 2


@pytest.mark.parametrize('media', ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'])
def test_word_template_page_and_crop(monkeypatch, tmp_path, media):
    initialize_database()
    settings = config.get_settings()
    assessment = create_assessment('Word source', [])
    source = settings.upload_dir/'synthetic.doc'
    source.write_bytes(b'fake')
    rendered = pdf_file(tmp_path/'rendered.pdf')
    doc = upsert_assessment_document(assessment['id'], 'question_paper', media, str(source), media, '', 'stored', 0)
    monkeypatch.setattr('grading_agent.optimization_api.word_to_pdf', lambda path, root: rendered)
    from grading_agent.app import app
    with TestClient(app) as client:
        base = f"/api/assessments/{assessment['id']}"
        result = client.get(base+'/template-page', params=dict(document_id=doc['id'], page=2))
        assert result.status_code == 200
        assert Image.open(io.BytesIO(result.content)).size == (300, 450)
        assert client.get(base+'/template-page', params=dict(document_id=doc['id'], page=3)).status_code == 400
        crop = client.post(base+'/template-preview', params=dict(document_id=doc['id']),
                           json=dict(page=2, x=.1, y=.1, width=.5, height=.5))
        assert crop.status_code == 200 and Image.open(io.BytesIO(crop.content)).size == (150, 225)


def test_unavailable_word_renderer_shows_actionable_error(monkeypatch):
    initialize_database()
    settings = config.get_settings()
    assessment = create_assessment('Word missing converter', [])
    source = settings.upload_dir/'synthetic.doc'
    source.write_bytes(b'fake')
    doc = upsert_assessment_document(assessment['id'], 'question_paper', 'hash', str(source), 'application/msword', '', 'stored', 0)
    monkeypatch.setattr(renderer, 'renderer_status', lambda: dict(available=False, message='请用 Word 导出 PDF 后上传'))
    from grading_agent.app import app
    with TestClient(app) as client:
        result = client.get(f"/api/assessments/{assessment['id']}/template-page", params={'document_id': doc['id']})
        assert result.status_code == 422 and 'PDF' in result.json()['detail']


def test_word_conversion_timeout_is_bounded(monkeypatch, tmp_path):
    source = tmp_path/'paper.doc'
    source.write_bytes(b'fake')
    monkeypatch.setattr(renderer, 'renderer_status', lambda: dict(available=True, backend='office', executable='powershell'))
    def timeout(command, **kwargs):
        assert kwargs['timeout'] == 60
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(renderer.subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='超时'):
        renderer.word_to_pdf(source, tmp_path)


def test_ocr_inheritance_has_no_credentials_and_own_options_win(monkeypatch, tmp_path):
    original = tmp_path/'original'
    optimized = original/'optimization'/'repository'/'project'
    for root in [original, optimized]:
        (root/'src/grading_agent').mkdir(parents=True)
        (root/'src/grading_agent/config.py').touch()
        (root/'data').mkdir()
    (original/'data/settings.json').write_text(json.dumps(dict(
        ocr_provider='paddleocr', ocr_device='gpu:0', openai_api_key='DO-NOT-INHERIT', review_mode='strict')))
    inherited = runtime_paths.inherited_ocr_defaults(optimized)
    assert inherited == dict(ocr_provider='paddleocr', ocr_device='gpu:0')
    monkeypatch.setattr(config, 'PROJECT_ROOT', optimized)
    monkeypatch.setenv('GRADING_DATA_DIR', str(optimized/'data'))
    monkeypatch.delenv('OCR_PROVIDER', raising=False)
    monkeypatch.delenv('OCR_DEVICE', raising=False)
    config._settings_cache = None
    assert config.get_settings().ocr_provider == 'paddleocr'
    assert config.get_settings().openai_api_key is None
    (optimized/'data/settings.json').write_text(json.dumps(dict(ocr_provider='none', ocr_device='cpu')))
    config._settings_cache = None
    assert config.get_settings().ocr_provider == 'none' and config.get_settings().ocr_device == 'cpu'


def test_cached_models_reused_without_copying(monkeypatch, tmp_path):
    own = tmp_path/'project'
    original = tmp_path/'original'
    (original/'.cache/paddlex/official_models').mkdir(parents=True)
    monkeypatch.setattr(runtime_paths, 'project_roots', lambda: [own, original])
    assert runtime_paths.model_cache_root() == original/'.cache'


def test_template_button_handles_word_and_empty_questions():
    text = (Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js').read_text(encoding='utf-8')
    block = text[text.index('function setupRegionEditor'):text.index('function drawRegion')]
    assert 'application/msword' in block and 'wordprocessingml.document' in block
    assert "$('editRegions').disabled=false" in block
    assert '请先生成参考答案或录入题目' in text
    assert 'detail.detail' in text


def test_region_dialog_opens_for_missing_setup_and_word_preview():
    import shutil
    node = shutil.which('node') or r'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    if not Path(node).exists():
        pytest.skip('Node runtime unavailable')
    source = Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[1],'utf8');
const code=text.slice(text.indexOf('function openRegionEditor(){'),text.indexOf("$('closeRegions').onclick"));
for(const [current,documents,expected,disabled] of [
 [null,0,'新建或选择',true],
 [{questions:[]},0,'上传原试卷',true],
 [{questions:[]},1,'可先预览',true],
 [{questions:[{id:'q',rule_config:{region:{page:2}}}]},1,'空白位置',false]
]){
 const elements={};let loads=0;
 const $=id=>elements[id]??=( {options:id==='regionDocument'?Array(documents):[],value:'q',open:false,
 showModal(){this.open=true;},click(){loads++;},close(){this.open=false;},scrollIntoView(){},focus(){this.focused=true;}} );
 const context={current,$};vm.createContext(context);vm.runInContext(code+';openRegionEditor();',context);
 assert($('regionDialog').open);assert($('regionStatus').textContent.includes(expected));
 assert.equal($('saveRegion').disabled,disabled);assert.equal(loads,current&&documents?1:0);
 if(!disabled)assert.equal($('regionPage').value,'2');
 $('regionPrepare').onclick();
 const target=!current?'newAssessment':!documents?'paperFiles':!current.questions.length?'generate':'editRegions';
 assert($(target).focused);
}
"""
    subprocess.run([str(node), '-e', script, str(source)], check=True, capture_output=True, text=True)


def test_home_disables_stale_html_cache():
    from grading_agent.app import app
    with TestClient(app) as client:
        response = client.get('/')
        assert response.headers['cache-control'] == 'no-store'
        assert 'region-dialog-v3' in response.text
