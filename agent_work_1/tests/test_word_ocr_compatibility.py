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
    assert '先从原试卷识别题目，不生成参考答案' in text
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
 [{questions:[]},0,'答题卡模板／已填样张',true],
 [{questions:[]},1,'可先预览',true],
 [{questions:[{id:'q',rule_config:{region:{page:2}}}]},1,'填写答案的位置',false]
]){
 const elements={};let loads=0;
 const $=id=>elements[id]??=( {options:id==='regionDocument'?Array(documents):[],value:'q',open:false,
 showModal(){this.open=true;},click(){loads++;},close(){this.open=false;},scrollIntoView(){},focus(){this.focused=true;}} );
 const context={current,$,regionPageRequest:0};vm.createContext(context);vm.runInContext(text.slice(text.indexOf('function editableRegionQuestions(){'),text.indexOf('// Region coordinates always')),context);vm.runInContext(code+';openRegionEditor();',context);
 assert($('regionDialog').open);assert($('regionStatus').textContent.includes(expected));
 assert.equal($('saveRegion').disabled,disabled);assert.equal(loads,current&&documents?1:0);
 if(!disabled)assert.equal($('regionPage').value,'2');
 $('regionPrepare').onclick();
 const target=!current?'newAssessment':!documents?'paperFiles':!current.questions.length?'extractQuestions':'editRegions';
 assert($(target).focused);
}
"""
    subprocess.run([str(node), '-e', script, str(source)], check=True, capture_output=True, text=True)


def test_home_disables_stale_html_cache():
    from grading_agent.app import app
    with TestClient(app) as client:
        response = client.get('/')
        assert response.headers['cache-control'] == 'no-store'
        assert 'answer-card-v9' in response.text


def test_region_pointer_modes_scale_and_work_before_questions():
    import shutil
    node = shutil.which('node') or r'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    if not Path(node).exists():
        pytest.skip('Node runtime unavailable')
    source = Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[1],'utf8');
const code=text.slice(text.indexOf('const regionCanvas='),text.indexOf("action('previewRegion'"));
const elements={},ctx={fillRect(){}};
const $=id=>elements[id]??={value:'drag',width:1000,height:2000,getBoundingClientRect(){return {left:100,top:50,width:500,height:1000}},getContext(){return ctx},setPointerCapture(){}};
const env={$,current:{questions:[]},regionLoadedPage:1,regionSelection:null,regionStart:null,regionClickStart:null,drawRegion(){},window:{addEventListener(){}},editableRegionQuestions(){return []},esc:String};
vm.createContext(env);vm.runInContext(code,env);
const event=(x,y)=>({clientX:x,clientY:y,button:0,pointerId:1,preventDefault(){}});
const canvas=$('regionCanvas');
canvas.onpointerdown(event(200,250));canvas.onpointerup(event(400,650));
assert.equal(env.regionSelection.x,.2);assert.equal(env.regionSelection.y,.2);
assert(Math.abs(env.regionSelection.width-.4)<1e-9);assert(Math.abs(env.regionSelection.height-.4)<1e-9);
assert($('regionStatus').textContent.includes('尚无题目'));
$('regionMode').value='corners';$('regionMode').onchange();
canvas.onpointerdown(event(400,650));assert($('regionStatus').textContent.includes('第一个角'));
canvas.onpointerdown(event(200,250));assert(Math.abs(env.regionSelection.width-.4)<1e-9);
env.regionLoadedPage=null;canvas.onpointerdown(event(100,100));assert($('regionStatus').textContent.includes('先加载'));
"""
    subprocess.run([str(node), '-e', script, str(source)], check=True, capture_output=True, text=True)


def test_region_save_picker_keeps_draft_and_excludes_choices():
    import shutil
    node = shutil.which('node')
    if not node:
        pytest.skip('Node runtime unavailable')
    source = Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[1],'utf8');const elements={};
const $=id=>elements[id]??={value:'q2'};
const selection={width:.3,height:.2};
const env={$,esc:String,layoutQuestionLabel:q=>q.question_no,current:{questions:[{id:'q1',question_no:'1',scoring_mode:'choice_set'},{id:'q2',question_no:'2',rule_config:{}},{id:'q3',question_no:'3',rule_config:{region:{page:1}}}]},regionLoadedPage:1,regionSelection:selection};
vm.createContext(env);
vm.runInContext(text.slice(text.indexOf('function editableRegionQuestions(){'),text.indexOf('// Region coordinates always')),env);
vm.runInContext(text.slice(text.indexOf('function showRegionActions(){'),text.indexOf('function regionSelectionFeedback(){')),env);
vm.runInContext('showRegionActions()',env);
assert.equal($('regionActions').hidden,false);assert(!$('regionSaveQuestion').innerHTML.includes('value="q1"'));
assert($('regionSaveQuestion').innerHTML.includes('value="q3"'));
$('regionSaveQuestion').value='q3';$('regionSaveQuestion').onchange();
assert.strictEqual(env.regionSelection,selection);assert($('regionSaveHint').textContent.length>0);
assert.equal($('saveRegion').disabled,false);
"""
    subprocess.run([node, '-e', script, str(source)], check=True, capture_output=True, text=True)



def test_slow_region_page_cannot_replace_newer_selection():
    import shutil
    node=shutil.which('node')
    if not node:pytest.skip('Node runtime unavailable')
    source=Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[1],'utf8');
const block=text.slice(text.indexOf("action('loadRegionPage',async()=>{"),text.indexOf('const regionCanvas='));
const elements={},pending=[],handlers={};
const $=id=>elements[id]??={value:id==='regionPage'?'1':'doc',open:true};
const env={$,applyRegionScale(){},aid:()=> 'assessment',action:(id,f)=>handlers[id]=f,regionPageRequest:0,regionLoadedPage:null,
 regionImage:null,regionSelection:null,selectSavedRegion(){},URL:{createObjectURL:b=>String(b.number),revokeObjectURL(){}},
 Image:class{async decode(){this.naturalWidth=Number(this.src)*100;this.naturalHeight=200;}},
 fetch:()=>new Promise(resolve=>pending.push(resolve))};
vm.createContext(env);vm.runInContext(block,env);
(async()=>{
 const old=handlers.loadRegionPage();$('regionPage').value='2';const recent=handlers.loadRegionPage();
 pending[1]({ok:true,blob:async()=>({number:2})});await recent;
 assert.equal(env.regionLoadedPage,2);assert.equal($('regionCanvas').width,200);
 pending[0]({ok:false,json:async()=>({detail:'obsolete error'})});await old;
 assert.equal(env.regionLoadedPage,2);assert.equal($('regionCanvas').width,200);
 const closed=handlers.loadRegionPage();$('regionDialog').open=false;
 pending[2]({ok:true,blob:async()=>({number:3})});await closed;
 assert.equal(env.regionLoadedPage,null);
})().catch(e=>{console.error(e);process.exit(1)});
"""
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)
