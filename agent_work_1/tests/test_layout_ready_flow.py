"""Real user starting state: one uploaded source, no questions, no template."""
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient


def candidate(number='二-1'):
    return {'question_no':number,'prompt':'Synthetic question '+number,'question_type':'numeric',
            'scoring_mode':'numeric','max_score':5,'candidate_answer':'42','confidence':.99,'rubric':[]}


@pytest.mark.parametrize('problem',['duplicate','blank_prompt','unknown_type','nan','missing_key'])
def test_generated_questions_are_atomic_and_usage_is_recorded(monkeypatch,problem):
    from grading_agent.app import app
    from grading_agent.db import transaction
    items=[candidate('一-1'),candidate('二-1')]
    if problem=='duplicate':items[1]['question_no']='一-1'
    if problem=='blank_prompt':items[1]['prompt']=''
    if problem=='unknown_type':items[1]['question_type']='bad'
    if problem=='nan':items[1]['confidence']=float('nan')
    if problem=='missing_key':del items[1]['candidate_answer']
    monkeypatch.setattr('grading_agent.app.compile_question_paper',lambda *a:{'questions':items,'usage':{'input_tokens':123}})
    with TestClient(app) as client:
        a=client.post('/api/assessments',json={'title':'Synthetic empty paper','subject_codes':[]}).json()
        response=client.post(f"/api/assessments/{a['id']}/generate-candidate-key")
        assert response.status_code>=400
        assert client.get(f"/api/assessments/{a['id']}").json()['questions']==[]
        with transaction() as db:
            rows=db.execute('SELECT input_tokens FROM token_usage WHERE assessment_id=?',(a['id'],)).fetchall()
        assert [r['input_tokens'] for r in rows]==[123]


def test_empty_paper_prepares_questions_then_layout_without_manual_numbering(monkeypatch):
    from dataclasses import replace
    from grading_agent import config
    from grading_agent.app import app
    from grading_agent.repository import get_assessment
    from grading_agent.services import layout_templates
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),openai_api_key='synthetic'))
    calls=[]
    def compile(*args, **kwargs):
        calls.append('questions');return {'questions':[candidate('一-1'),candidate('二-1')],'usage':{}}
    monkeypatch.setattr('grading_agent.app.compile_question_paper',compile)
    with TestClient(app) as client:
        a=client.post('/api/assessments',json={'title':'Synthetic no questions','subject_codes':[]}).json()
        image=io.BytesIO();Image.new('RGB',(600,800),'white').save(image,'PNG')
        doc=client.post(f"/api/assessments/{a['id']}/documents",data={'kind':'answer_sheet'},
                        files=[('files',('paper.png',image.getvalue(),'image/png'))]).json()[0]
        assert not get_assessment(a['id'])['questions']
        assert client.post(f"/api/assessments/{a['id']}/extract-questions").status_code==200
        def model(*args):
            calls.append('layout')
            values=[dict(question_id=q['id'],section=q['question_no'].split('-')[0],number='1',method='text',
                         confidence=.99,regions=[dict(page=1,x=.1,y=.1+i*.3,width=.7,height=.2)],options=[])
                    for i,q in enumerate(get_assessment(a['id'])['questions'])]
            return {'output_text':json.dumps({'items':values}),'usage':{}},'synthetic'
        monkeypatch.setattr(layout_templates,'request_json',model)
        response=client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id']})
        assert response.status_code==200,response.text
        assert response.json()['enabled'] and len(response.json()['items'])==2
        assert calls==['questions','layout']
        assert client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id']}).json()['cache_hit']
        assert calls==['questions','layout']


def test_ready_button_and_two_step_browser_orchestration():
    node=shutil.which('node')
    if not node:pytest.skip('Node runtime unavailable')
    source=Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const block=source.slice(source.indexOf('function updateLayoutEntry(){'),source.indexOf("action('toggleLayout'"));
async function scenario(existing=false,fail=false){
 const elements={},calls=[],handlers={};
 const $=id=>elements[id]??={value:'original-source',options:[{}],checked:false};
 const env={$,current:{id:'a',questions:existing?[{id:'q'}]:[]},layoutBusy:false,questionPreparation:null,
  action:(id,f)=>handlers[id]=f,json:body=>({body}),notice(){},
  async api(url,options){calls.push([url,options]);if(fail)throw Error('synthetic configuration error');},
  async loadAssessment(){env.updateLayoutEntry();assert($('generateLayout').disabled);env.current.questions=[{id:'q'}];$('layoutDocument').value='changed-during-refresh';},
  async loadLayoutStatus(){}};
 vm.createContext(env);vm.runInContext(block,env);env.updateLayoutEntry();
 assert.equal($('generateLayout').disabled,false);
 if(!existing)assert($('generateLayout').textContent.length>0);
 if(fail){await assert.rejects(handlers.generateLayout);assert($('layoutState').textContent.includes('synthetic configuration error'));assert.equal(calls.length,1);}
 else{await handlers.generateLayout();assert.equal(calls.length,existing?1:2);assert.equal(calls.at(-1)[1].body.document_id,'original-source');}
 assert.equal(env.layoutBusy,false);assert.equal($('generateLayout').disabled,false);
}
(async()=>{await scenario();await scenario(true);await scenario(false,true)})().catch(e=>{console.error(e);process.exit(1)});
"""
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)



def test_region_draft_survives_question_preparation_and_requires_explicit_save():
    node=shutil.which('node')
    if not node:pytest.skip('Node runtime unavailable')
    source=Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[1],'utf8'),elements={},handlers={};
const $=id=>elements[id]??={value:'q2',open:true};
const draft={page:1,x:.2,y:.3,width:.4,height:.2};let saves=0;
const env={$,current:{id:'a',questions:[]},regionSelection:draft,regionLoadedPage:1,esc:String,
 layoutQuestionLabel:q=>q.question_no,regionStart:null,regionClickStart:null,drawRegion(){},aid:()=> 'a',
 action:(id,f)=>handlers[id]=f,async persistRegion(){saves++},
 async ensureAssessmentQuestions(){env.current.questions=[{id:'q2',question_no:'二-1'}]}};
vm.createContext(env);
vm.runInContext(text.slice(text.indexOf('function editableRegionQuestions(){'),text.indexOf('// Region coordinates always')),env);
vm.runInContext(text.slice(text.indexOf('function showRegionActions(){'),text.indexOf('function regionSelectionFeedback(){')),env);
vm.runInContext(text.slice(text.indexOf("action('saveRegion',async()=>{"),text.indexOf('let trainingGrade=')),env);
(async()=>{
 vm.runInContext('showRegionActions()',env);assert.equal($('saveRegion').disabled,false);
 await handlers.saveRegion();assert.strictEqual(env.regionSelection,draft);assert.equal(saves,0);
 assert.equal($('regionSaveQuestion').disabled,false);assert($('regionSaveQuestion').innerHTML.includes('q2'));
 await handlers.saveRegion();assert.equal(saves,1);
})().catch(e=>{console.error(e);process.exit(1)});
"""
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)


def test_region_page_fit_and_width_fit_use_available_viewport():
    node=shutil.which('node')
    if not node:pytest.skip('Node runtime unavailable')
    source=Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[1],'utf8'),elements={};
const $=id=>elements[id]??={value:'page',clientWidth:1000,clientHeight:500,width:1000,height:2000,style:{}};
const env={$,regionImage:{},window:{addEventListener(){}}};vm.createContext(env);
vm.runInContext(text.slice(text.indexOf('function applyRegionScale(){'),text.indexOf('function regionPoint(')),env);
vm.runInContext('applyRegionScale()',env);assert.equal($('regionCanvas').style.width,'232px');
$('regionScale').value='width';$('regionScale').onchange();assert.equal($('regionCanvas').style.width,'964px');
"""
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)
