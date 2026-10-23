import copy
import json
from dataclasses import replace

from PIL import Image
import pytest

from grading_agent import config
from grading_agent.services import local_vision as vision
from test_local_layout import workspace,generate,upload,mark


def candidate(text='for i in range(3)\n    print(i)',nodes=None,edges=None,uncertain=None):
    return dict(text=text,nodes=nodes or [],edges=edges or [],uncertain=uncertain or [])


@pytest.fixture
def local_model(tmp_path,monkeypatch):
    settings=replace(config.get_settings(),local_vision_enabled=True)
    path=tmp_path/'handwriting.png';Image.new('RGB',(200,80),'white').save(path)
    calls=[];values=[candidate(),candidate()]
    monkeypatch.setattr(vision,'runtime',lambda settings:{'model':settings.local_vision_model,'digest':'revision1'})
    def request(base,path,body,timeout):
        assert path=='/api/chat' and base.startswith('http://127.0.0.1:')
        assert 'SECRET_REFERENCE' not in json.dumps(body)
        calls.append(body)
        return {'done':True,'done_reason':'stop','message':{'content':json.dumps(values[len(calls)-1])},'prompt_eval_count':12,'eval_count':5}
    monkeypatch.setattr(vision,'local_json',request)
    return settings,path,calls,values


@pytest.mark.parametrize('url',['https://127.0.0.1:11435','http://remote.example','http://127.0.0.1.evil.test',
                              'http://user:password@localhost:11435','http://localhost/path','http://localhost?proxy=x'])
def test_only_loopback_root_allowed(url):
    with pytest.raises(ValueError):vision.local_url(url)


def test_remote_model_alias_rejected(monkeypatch):
    settings=config.get_settings()
    monkeypatch.setattr(vision,'local_json',lambda base,path,*args:{'models':[{'name':settings.local_vision_model,'digest':'x'}]} if path=='/api/tags' else {'capabilities':['vision'],'remote_host':'https://remote.example'})
    with pytest.raises(ValueError,match='远程模型'):vision.runtime(settings)


def test_verbatim_two_passes_and_exact_cache(local_model):
    settings,path,calls,values=local_model
    result=vision.transcribe(path,{'prompt':'写python程序','answer_key':'SECRET_REFERENCE'},settings)
    assert result['consistent'] and result['text']==values[0]['text'] and 'range(3)\n    ' in result['text']
    assert len(result['requests'])==2 and len(calls)==2
    assert calls[0]['messages'][1]['images']!=calls[1]['messages'][1]['images'] or calls[0]['messages'][0]!=calls[1]['messages'][0]
    again=vision.transcribe(path,{'prompt':'写python程序'},settings)
    assert again['cache_hit'] and not again['requests'] and len(calls)==2


def test_same_question_multiple_images_are_sent_together_in_order(local_model):
    settings,path,calls,values=local_model
    second=path.with_name('continuation.png');Image.new('RGB',(200,80),'gray').save(second)
    result=vision.transcribe([path,second],{'question_type':'code'},settings)
    assert result['consistent'] and len(calls)==2
    assert len(calls[0]['messages'][1]['images'])==2
    assert calls[0]['messages'][1]['images'][0]!=calls[0]['messages'][1]['images'][1]
    assert '连续作答区' in calls[0]['messages'][0]['content']
    with pytest.raises(ValueError,match='最多联合核验'):vision.transcribe([path]*4,{},settings)


@pytest.mark.parametrize('issue',['indent','symbol','uncertain','no_graph','single'])
def test_uncertain_never_becomes_accepted(local_model,issue):
    settings,path,calls,values=local_model
    question={}
    if issue=='indent':values[1]['text']=values[0]['text'].replace('    ','  ')
    if issue=='symbol':values[1]['text']=values[0]['text'].replace('3','8')
    if issue=='uncertain':values[0]['uncertain']=['运算符看不清']
    if issue=='no_graph':question['prompt']='画流程图'
    if issue=='single':settings=replace(settings,local_vision_verify=False)
    result=vision.transcribe(path,question,settings)
    assert not result['consistent'] and result['issues']
    assert result['text']  # candidate is inspectable, not discarded


def test_flowchart_preserves_topology(local_model):
    settings,path,calls,values=local_model
    graph=candidate('说明',nodes=[dict(id='n1',text='x > 0?',shape='diamond'),dict(id='n2',text='print(x)',shape='rectangle')],
                    edges=[dict(source='n1',target='n2',label='是')])
    values[:]=[copy.deepcopy(graph),copy.deepcopy(graph)]
    result=vision.transcribe(path,{'prompt':'画流程图'},settings)
    assert not result['consistent'] and 'n1 -> n2 [是]' in result['text']
    assert any('需人工核对' in issue for issue in result['issues'])
    values[1]['edges'][0]['label']='否'
    # A new revision must not reuse an earlier model's accepted topology.
    settings=replace(settings,local_vision_model='other-local-model');calls.clear()
    assert not vision.transcribe(path,{'prompt':'画流程图'},settings)['consistent']


def test_invalid_graph_or_truncation_rejected(local_model,monkeypatch):
    settings,path,_,values=local_model
    values[0]['edges']=[dict(source='missing',target='also_missing',label='')]
    with pytest.raises(ValueError,match='结构无效'):vision.transcribe(path,{},settings)
    monkeypatch.setattr(vision,'local_json',lambda *a,**kw:{'done':True,'done_reason':'length'})
    with pytest.raises(ValueError,match='未完整结束'):vision.transcribe(path,{},settings)


def enable_visual(monkeypatch):
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),local_vision_enabled=True))
    monkeypatch.setattr(vision,'runtime',lambda *a:{'ready':True,'model':'synthetic-local','digest':'test'})


def test_integration_accepted_without_fake_confidence_and_rules_still_score(workspace,monkeypatch):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_visual(monkeypatch)
    monkeypatch.setattr(vision,'transcribe',lambda *a:{'text':'42','consistent':True,'issues':[],
        'model':{'model':'synthetic-local'},'structured':candidate('42'),'requests':[]})
    student=upload(workspace,mark(source))
    response=next(r for r in student['responses'] if r['question_id']==qs[1]['id'])
    assert response['extraction_source']=='local_vlm_consistent' and response['ocr_confidence'] is None
    assert student['normalization_report']['local_recognition']['missing']==[]
    graded=client.post(f"/api/submissions/{student['id']}/grade");assert graded.status_code==200,graded.text
    grade=next(g for g in graded.json()['grade_items'] if g['question_id']==qs[1]['id'])
    assert grade['score']==5 and 'visual_transcription_spot_check' in grade['reason_codes']
    assert len(calls)==1  # only mocked template request, no remote recognition


def test_complex_continuation_regions_are_joint_and_every_crop_is_accessible(workspace,monkeypatch):
    client,a,qs,doc,source,calls=workspace
    q=qs[1]
    assert client.put('/api/questions/'+q['id'],json={**q,'question_type':'programming'}).status_code==200
    state=generate(workspace);enable_visual(monkeypatch)
    payload=dict(expected_template_id=state['template_id'],document_id=doc['id'],
                 regions=[dict(page=1,x=.1,y=.5,width=.3,height=.1),dict(page=1,x=.5,y=.5,width=.3,height=.1)])
    assert client.put(f"/api/assessments/{a['id']}/layout-template/regions/{q['id']}",json=payload).status_code==200
    received=[]
    def joint(paths,*args):
        received.append(paths)
        return dict(text='42',consistent=True,issues=[],model={'model':'local'},requests=[])
    monkeypatch.setattr(vision,'transcribe',joint)
    student=upload(workspace,mark(source))
    assert len(received)==1 and len(received[0])==2
    detail=student['normalization_report']['local_recognition']['recognitions'][q['id']]
    assert len(detail['parts'])==1 and len(detail['parts'][0]['crops'])==2
    url=f"/api/submissions/{student['id']}/local-recognition-crop?question_id={q['id']}"
    assert client.get(url+'&region=0').status_code==200
    assert client.get(url+'&region=1').status_code==200
    assert client.get(url+'&region=2').status_code==404


@pytest.mark.parametrize('issues',[['符号分歧'],[]])
def test_integration_disagreement_keeps_missing_and_crop_inspectable(workspace,monkeypatch,issues):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_visual(monkeypatch)
    monkeypatch.setattr(vision,'transcribe',lambda *a:{'text':'42?','consistent':False,'issues':issues,
        'model':{'model':'synthetic-local'},'verification':candidate('43')})
    student=upload(workspace,mark(source))
    assert not any(r['question_id']==qs[1]['id'] for r in student['responses'])
    detail=student['normalization_report']['local_recognition']['recognitions'][qs[1]['id']]
    assert detail['parts'][0]['text']=='42?'
    url=f"/api/submissions/{student['id']}/local-recognition-crop?question_id={qs[1]['id']}"
    assert client.get(url).status_code==200 and client.get(url+'&part=8').status_code==404
    # Human correction is retained on a new recognition attempt.
    client.post(f"/api/submissions/{student['id']}/answers",json={'answers':[{'question_id':qs[1]['id'],'raw_text':'43'}]})
    from grading_agent.app import prepare_submission
    retried=prepare_submission(student['id'])
    assert next(r for r in retried['responses'] if r['question_id']==qs[1]['id'])['raw_text']=='43'


def test_configuration_is_separate_and_cloud_or_unverified_mode_rejected(workspace):
    client,*_=workspace
    current=config.get_settings()
    config._settings_file().write_text(json.dumps({'openai_api_key':current.openai_api_key,
        'openai_model':current.openai_model,'openai_base_url':current.openai_base_url,
        'grading_profile':current.grading_profile,'ocr_provider':current.ocr_provider}),encoding='utf-8')
    before=client.get('/api/settings').json()
    assert client.put('/api/runtime/local-vision',json={'enabled':True,'verify':False}).status_code==422
    assert client.put('/api/runtime/local-vision',json={'enabled':True,'model':'qwen-cloud'}).status_code==422
    result=client.put('/api/runtime/local-vision',json={'enabled':False})
    assert result.status_code==200 and not result.json()['enabled']
    assert client.get('/api/settings').json()==before


def test_subjective_grading_routes_agreed_transcription_without_numeric_confidence():
    from grading_agent.services.grading import requires_model
    assert requires_model({'scoring_mode':'rubric_assisted'},dict(raw_text='学生作答',ocr_confidence=None,extraction_source='local_vlm_consistent'))


def test_diagnostics_html_escapes_candidate_text_and_shows_raw_crop():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    source=Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const env={esc:v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),encodeURIComponent};
vm.createContext(env);vm.runInContext(source.slice(source.indexOf('function localRecognitionHtml('),source.indexOf('async function loadLocalVision(')),env);
const html=env.localRecognitionHtml({id:'submission',normalization_report:{local_recognition:{missing:['1'],warnings:[],recognitions:{q:{question_no:'1',parts:[{text:'<script>bad()</script>',model:{model:'local'},consistent:false,issues:['symbol'],verification:{text:'other'}}]}}}}});
assert(html.includes('&lt;script&gt;'));assert(!html.includes('<script>'));
assert(html.includes('local-recognition-crop?question_id=q&part=0'));assert(html.includes('第二次转写'));assert(!html.includes('教师确认标签'));
'''
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)
