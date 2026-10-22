"""Paper content, card geometry and scoring standards are independent."""
import io
import json
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image


def test_structure_prompt_excludes_answers_and_card(monkeypatch,tmp_path):
    from grading_agent.config import get_settings
    from grading_agent.services import assessment_compiler as compiler
    calls=[]
    def respond(request,timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'output_text':json.dumps({'questions':[{
            'question_no':'一-1','prompt':'选择 A.1 B.2','question_type':'single_choice',
            'scoring_mode':'choice_set','max_score':2,'confidence':.9,
            'candidate_answer':'UNREQUESTED','rubric':[{'criterion':'bad','score':2}]}]})}).encode())
    monkeypatch.setattr(compiler.urllib.request,'urlopen',respond)
    added=[]
    monkeypatch.setattr(compiler,'_add_source',lambda content,doc,label,settings:added.append(doc['kind']))
    docs=[{'kind':k,'source_path':str(tmp_path/k)} for k in ['question_paper','answer_sheet','answer_key','rubric']]
    result=compiler.compile_question_paper(docs,replace(get_settings(),openai_api_key='synthetic'),structure_only=True)
    assert added==['question_paper']
    schema=calls[0]['text']['format']['schema']['properties']['questions']['items']
    assert 'candidate_answer' not in schema['properties'] and 'rubric' not in schema['required']
    assert '不解题' in calls[0]['input'][0]['content'][0]['text']
    assert result['questions'][0]['candidate_answer']=='' and result['questions'][0]['rubric']==[]


def test_card_template_does_not_generate_answers(monkeypatch):
    from grading_agent.app import app
    from grading_agent import config
    from grading_agent.repository import get_assessment
    from grading_agent.services import layout_templates
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),openai_api_key='synthetic'))
    calls=[]
    def structure(docs,settings,**kwargs):
        assert kwargs=={'structure_only':True}
        calls.append('structure')
        return {'questions':[dict(question_no='二-1',prompt='计算 6×7',question_type='numeric',
            scoring_mode='numeric',max_score=5,confidence=.99,candidate_answer='MUST NOT SAVE',rubric=[])],'usage':{}}
    monkeypatch.setattr('grading_agent.app.compile_question_paper',structure)
    def fill(questions,*args):
        calls.append('answers')
        return {'updates':[dict(id=questions[0]['id'],answer_key='42',rubric=[],generated_answer=True)],'usage':{}}
    monkeypatch.setattr('grading_agent.app.fill_missing_parts',fill)
    with TestClient(app) as client:
        a=client.post('/api/assessments',json={'title':'Synthetic split workflow'}).json()
        base='/api/assessments/'+a['id']
        buf=io.BytesIO();Image.new('RGB',(600,800),'white').save(buf,'PNG')
        def upload(kind):
            r=client.post(base+'/documents',data={'kind':kind},files=[('files',('source.png',buf.getvalue(),'image/png'))])
            assert r.status_code==200,r.text
            return r.json()[0]
        paper=upload('question_paper');card=upload('answer_sheet')
        assert client.post(base+'/layout-template',json={'document_id':paper['id']}).status_code==404
        assert client.post(base+'/extract-questions').status_code==200
        question=get_assessment(a['id'])['questions'][0]
        assert question['answer_key']=='' and question['answer_key_state']=='missing'
        assert client.post(base+'/confirm-candidate-key').status_code==400
        assert client.post(base+'/publish').status_code==400
        def model(*args):
            calls.append('layout')
            item=dict(question_id=question['id'],section='二、计算题',number='1',method='text',confidence=.99,
                regions=[dict(page=1,x=.1,y=.1,width=.7,height=.2)],options=[])
            return {'output_text':json.dumps({'items':[item]}),'usage':{}},'synthetic'
        monkeypatch.setattr(layout_templates,'request_json',model)
        r=client.post(base+'/layout-template',json={'document_id':card['id']})
        assert r.status_code==200,r.text
        assert calls==['structure','layout']
        from grading_agent.db import transaction
        with transaction() as db:
            db.execute("UPDATE assessment_documents SET kind='reference' WHERE id=?",(card['id'],))
        old=client.get(base+'/layout-template').json()
        assert old['stale'] and any('独立类型' in warning for warning in old['warnings'])
        with transaction() as db:
            db.execute("UPDATE assessment_documents SET kind='answer_sheet' WHERE id=?",(card['id'],))
        assert get_assessment(a['id'])['questions'][0]['answer_key_state']=='missing'
        assert client.post(base+'/fill-missing').status_code==200
        assert calls==['structure','layout','answers']
        assert not client.get(base+'/layout-template').json()['stale']
        assert client.post(base+'/confirm-candidate-key').status_code==200
        assert client.post(base+'/publish').status_code==200
