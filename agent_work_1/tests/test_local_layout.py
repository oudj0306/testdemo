"""Synthetic geometry/marks are real; template API and OCR are mocked unless noted."""
import copy
import io
import json
import random
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from grading_agent import config
from grading_agent.db import transaction, decode_json
from grading_agent.services import layout_templates as templates, local_answers as local


def blank_sheet():
    image=Image.new('RGB',(900,1200),'white')
    draw=ImageDraw.Draw(image)
    rng=random.Random(812)
    for n in range(500):
        x,y=rng.randrange(25,870),rng.randrange(25,1170)
        if .09*900<x<.8*900 and (.14*1200<y<.33*1200 or .48*1200<y<.65*1200):
            continue
        draw.text((x,y),str(n),fill='black')
    draw.text((85,145),'I. CHOICE - 1',fill='black')
    for x in [135,225,315,405]:
        draw.ellipse((x,240,x+36,276),outline='black',width=2)
    draw.text((85,570),'II. NUMERIC - 1',fill='black')
    return image


def box(x,y,w,h):
    return dict(page=1,x=x,y=y,width=w,height=h)


def draft(q1='choice',q2='text'):
    return {'items':[
        dict(question_id=q1,section='一、选择题',number='1',method='omr',confidence=.99,
             regions=[box(.1,.14,.7,.19)],options=[dict(label=label,box=box(x/900,.2,.04,.03))
                for label,x in zip('ABCD',[135,225,315,405])]),
        dict(question_id=q2,section='二、填空题',number='1',method='text',confidence=.99,
             regions=[box(.1,.5,.7,.12)],options=[])]}


def mark(image,label='B',partial=False):
    result=image.copy()
    draw=ImageDraw.Draw(result)
    x={'A':135,'B':225,'C':315,'D':405}[label]
    if partial:
        draw.rectangle((x+15,252,x+20,261),fill='black')
    else:
        draw.ellipse((x+4,244,x+32,272),fill='black')
    return result


def data(image):
    stream=io.BytesIO();image.save(stream,'PNG');return stream.getvalue()


@pytest.fixture
def cv():
    pytest.importorskip('numpy')
    return pytest.importorskip('cv2')


def test_real_registration_handles_shift_and_rejects_wrong_page(cv):
    import numpy as np
    ref=blank_sheet()
    student=mark(ref)
    shift=np.float32([[1,0,8],[0,1,12]])
    moved=Image.fromarray(cv.warpAffine(np.asarray(student),shift,ref.size,borderValue=(255,255,255)))
    aligned,meta=local.register_page(ref,moved)
    assert meta['inliers']>=16
    value=local.read_marks(draft()['items'][0],{1:aligned},{1:ref},{1:meta})
    assert value[:3]==('B',.97,False)
    wrong=Image.new('RGB',ref.size,'white')
    with pytest.raises(ValueError):
        local.register_page(ref,wrong)


@pytest.mark.parametrize('mode,expected', [('blank',''),('one','B'),('multiple',None),('faint',None),('multiselect','BC')])
def test_real_mark_detection_blank_faint_single_and_multiple(cv,mode,expected):
    import numpy as np
    ref=blank_sheet()
    answer=ref if mode=='blank' else mark(ref,partial=mode=='faint')
    if mode in ('multiple','multiselect'):
        answer=mark(answer,'C')
    text,confidence,blank,measurements=local.read_marks(draft()['items'][0],{1:np.asarray(answer)},{1:ref},
                                                     {1:{}},multiple=mode=='multiselect')
    assert len(measurements)==4
    if expected is None:
        assert confidence is None and not blank
    else:
        assert text==expected and confidence>.9 and blank==(mode=='blank')


@pytest.mark.parametrize('problem', ['unknown','duplicate','page','overlap','outside','nan'])
def test_invalid_layout_not_accepted(problem):
    value=draft()
    if problem=='unknown':value['items'][0]['question_id']='other'
    if problem=='duplicate':value['items'].append(copy.deepcopy(value['items'][0]))
    if problem=='page':value['items'][0]['regions'][0]['page']=2
    if problem=='overlap':value['items'][0]['options'][1]['box']=value['items'][0]['options'][0]['box']
    if problem=='outside':value['items'][0]['regions'][0]['x']=.8
    if problem=='nan':value['items'][0]['confidence']=float('nan')
    with pytest.raises(ValueError):
        templates.validate_layout(value,[{'id':'choice','question_no':'一-1','scoring_mode':'choice_set'},
                                        {'id':'text','question_no':'二-1','scoring_mode':'numeric'}],1)


@pytest.fixture
def workspace(monkeypatch):
    from grading_agent.app import app
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),openai_api_key='synthetic',
                        ocr_provider='paddleocr',grading_profile='daily',review_sample_rate=0))
    calls=[]
    monkeypatch.setattr('grading_agent.app.extract_answers_with_model',lambda *a,**kw:pytest.fail('No student vision API'))
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *a:('42',.99,'synthetic_ocr'))
    with TestClient(app) as client:
        a=client.post('/api/assessments',json={'title':'Synthetic local','subject_codes':[]}).json()
        qs=[]
        for number,kind,mode,key in [('一-1','single_choice','choice_set','B'),('二-1','numeric','numeric','42')]:
            response=client.post(f"/api/assessments/{a['id']}/questions",json={'question_no':number,'prompt':number,
                     'question_type':kind,'scoring_mode':mode,'max_score':5,'answer_key':key})
            assert response.status_code==200,response.text
            qs.append(response.json())
        source=blank_sheet()
        document=client.post(f"/api/assessments/{a['id']}/documents",data={'kind':'answer_sheet'},
                             files=[('files',('blank.png',data(source),'image/png'))]).json()[0]
        def fake(request,settings,stage,timeout):
            calls.append(stage)
            return {'output_text':json.dumps(draft(qs[0]['id'],qs[1]['id'])),
                    'usage':{'input_tokens':100,'output_tokens':50}},'synthetic-request'
        monkeypatch.setattr(templates,'request_json',fake)
        client.post(f"/api/assessments/{a['id']}/publish")
        yield client,a,qs,document,source,calls


def generate(ws):
    client,a,qs,doc,source,calls=ws
    result=client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id']})
    assert result.status_code==200,result.text
    return result.json()


def upload(ws,image):
    client,a,*_=ws
    result=client.post('/api/submissions',data={'assessment_id':a['id'],'student_ref':'synthetic'},
                       files={'file':('synthetic.png',data(image),'image/png')})
    assert result.status_code==200,result.text
    return result.json()


def test_template_once_local_answers_rules_and_overlay(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    first=generate(workspace)
    assert first['enabled'] and not first['cache_hit']
    assert generate(workspace)['cache_hit'] and len(calls)==1
    assert first['items'][0]['section']=='一、选择题' and first['items'][1]['number']=='1'
    assert client.get(f"/api/assessments/{a['id']}/layout-template/page").status_code==200
    student=upload(workspace,mark(source))
    assert {r['raw_text'] for r in student['responses']}=={'B','42'}
    report=student['normalization_report']['local_recognition']
    assert report['remote_vision_requests']==0 and report['missing']==[]
    assert {r['extraction_source'] for r in student['responses']}=={'local_layout_ocr','local_layout_omr'}
    grade=client.post(f"/api/submissions/{student['id']}/grade").json()
    assert all(g['score']==5 and g['grader_kind']=='rules' for g in grade['grade_items'])
    with transaction() as db:
        rows=db.execute("SELECT stage,input_tokens FROM token_usage WHERE assessment_id=?",(a['id'],)).fetchall()
    assert [(r['stage'],r['input_tokens']) for r in rows]==[('layout',100)]


def test_wrong_layout_and_stale_template_never_call_vision(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    bad=upload(workspace,Image.new('RGB',source.size,'white'))
    assert not bad['responses'] and bad['status']=='extraction_failed'
    assert len(bad['normalization_report']['local_recognition']['missing'])==2
    q=qs[0]
    assert client.put('/api/questions/'+q['id'],json={**q,'prompt':'changed'}).status_code==200
    assert client.get(f"/api/assessments/{a['id']}/layout-template").json()['stale']
    student=upload(workspace,mark(source))
    assert not student['responses'] and student['status']=='extraction_failed' and len(calls)==1


def test_teacher_answer_survives_retry_and_region_correction_updates_template(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    student=upload(workspace,mark(source))
    client.post(f"/api/submissions/{student['id']}/answers",json={'answers':[{'question_id':qs[1]['id'],'raw_text':'43'}]})
    # Correct the template using the same public question API as manual region UI.
    q=qs[1]
    region=box(.12,.51,.6,.1)
    assert client.put('/api/questions/'+q['id'],json={**q,'rule_config':{'region':region}}).status_code==200
    status=client.get(f"/api/assessments/{a['id']}/layout-template").json()
    item=next(i for i in status['items'] if i['question_id']==q['id'])
    assert item['regions']==[region] and not status['stale']
    from grading_agent.app import prepare_submission
    retried=prepare_submission(student['id'])
    assert next(r for r in retried['responses'] if r['question_id']==q['id'])['raw_text']=='43'


def test_generation_failure_keeps_old_layout_and_missing_dependency_is_explicit(workspace,monkeypatch,cv):
    client,a,qs,doc,source,calls=workspace
    old=generate(workspace)
    monkeypatch.setattr(templates,'request_json',lambda *a:({'output_text':'{"items":[]}','usage':{}},'bad'))
    failed=client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id'],'force':True})
    assert failed.status_code==422
    state=client.get(f"/api/assessments/{a['id']}/layout-template").json()
    assert state['template_id']==old['template_id'] and state['enabled']
    monkeypatch.setattr(local,'dependencies',lambda:{'opencv':False,'numpy':False})
    student=upload(workspace,mark(source))
    assert not student['responses'] and 'OpenCV' in student['extraction_warning']


def test_empty_option_box_is_not_confirmed_blank(cv):
    import numpy as np
    ref=blank_sheet();item=draft()['items'][0]
    for option in item['options']:
        option['box']['y']=.3
    with pytest.raises(ValueError,match='模板选项框'):
        local.read_marks(item,{1:np.asarray(ref)},{1:ref},{1:{}})


def test_missing_page_is_not_blank(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    from grading_agent.repository import get_assessment
    payload=templates.load_layout(get_assessment(a['id']))
    answers,warnings,_=local.recognize_answers(payload,[],get_assessment(a['id'])['questions'],config.get_settings())
    assert not answers and any('缺少' in message for message in warnings)


def test_local_ocr_failure_retains_choice_and_missing_text(workspace,monkeypatch,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *a:('',None,'paddleocr_failed'))
    student=upload(workspace,mark(source))
    assert len(student['responses'])==1 and student['responses'][0]['raw_text']=='B'
    assert student['normalization_report']['local_recognition']['missing']==['二-1']
    graded=client.post(f"/api/submissions/{student['id']}/grade").json()
    assert {g['question_no']:g['score'] for g in graded['grade_items']}=={'一-1':5.,'二-1':None}


def test_subjective_local_text_is_sent_to_scoring_only(workspace,monkeypatch,cv):
    client,a,qs,doc,source,calls=workspace
    q=qs[1]
    client.put('/api/questions/'+q['id'],json={**q,'question_type':'short_answer','scoring_mode':'rubric_assisted',
                'rubric':[{'criterion':'meaning','score':5}]})
    generate(workspace)
    from grading_agent.services.math_grader import GradeDecision
    from grading_agent.schemas import ScoreState
    scoring=[]
    def score(self,question,text):
        scoring.append((question['question_no'],text))
        return GradeDecision(4.,5.,ScoreState.AUTO_SINGLE,False,['synthetic'])
    monkeypatch.setattr('grading_agent.services.llm_grader.OpenAIGrader.grade',score)
    student=upload(workspace,mark(source))
    graded=client.post(f"/api/submissions/{student['id']}/grade")
    assert graded.status_code==200,graded.text
    assert scoring==[('二-1','42')]
    assert len(calls)==1


def test_partial_local_recognition_can_finish_daily_batch(workspace,monkeypatch,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *a:('',None,'paddleocr_failed'))
    student=upload(workspace,mark(source))
    from grading_agent.services.batch_queue import BatchQueue,enqueue_saved
    from grading_agent.app import prepare_submission
    from grading_agent.repository import get_submission
    queue=BatchQueue(None,prepare_submission)
    batch=enqueue_saved(a['id'],[student['id']])
    job=queue.claim()
    assert job and 'daily_group' in job
    queue.process_daily(job['daily_group'])
    result=get_submission(student['id'])
    assert {g['question_no']:g['score'] for g in result['grade_items']}=={'一-1':5.,'二-1':None}
    assert len(calls)==1



def test_remove_auto_region_really_removes_local_mapping(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    q=qs[1]
    response=client.put('/api/questions/'+q['id'],json={**q,'rule_config':{'region':None}})
    assert response.status_code==200,response.text
    state=client.get(f"/api/assessments/{a['id']}/layout-template").json()
    assert not any(i['question_id']==q['id'] for i in state['items'])
    student=upload(workspace,mark(source))
    assert student['normalization_report']['local_recognition']['missing']==['二-1']


def test_correction_from_wrong_document_rolls_back(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    old=generate(workspace)
    q=qs[1]
    result=client.put('/api/questions/'+q['id'],json={**q,'rule_config':{
        'region':box(.12,.51,.6,.1),'region_document_id':'other-document'}})
    assert result.status_code==400,result.text
    state=client.get(f"/api/assessments/{a['id']}/layout-template").json()
    assert state['template_id']==old['template_id']
    assert client.get(f"/api/assessments/{a['id']}").json()['questions'][1]['rule_config']==q['rule_config']


def test_failed_regeneration_never_overwrites_old_page(workspace,monkeypatch,cv):
    from grading_agent.repository import get_assessment
    from pathlib import Path
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    old=templates.load_layout(get_assessment(a['id']))
    path=Path(old['pages'][0]['path']);before=path.stat().st_mtime_ns
    monkeypatch.setattr(templates,'request_json',lambda *a:({'output_text':'{"items":[]}','usage':{}},'bad'))
    assert client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id'],'force':True}).status_code==422
    assert path.stat().st_mtime_ns==before
    assert templates.load_layout(get_assessment(a['id']))['template_id']==old['template_id']


def test_regeneration_does_not_overwrite_concurrent_teacher_correction(workspace,monkeypatch,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    q=qs[1];region=box(.12,.51,.6,.1)
    def concurrent(*args):
        result=client.put('/api/questions/'+q['id'],json={**q,'rule_config':{'region':region}})
        assert result.status_code==200
        return {'output_text':json.dumps(draft(qs[0]['id'],q['id'])),'usage':{}},'synthetic'
    monkeypatch.setattr(templates,'request_json',concurrent)
    response=client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id'],'force':True})
    assert response.status_code==422,response.text
    state=client.get(f"/api/assessments/{a['id']}/layout-template").json()
    assert next(i for i in state['items'] if i['question_id']==q['id'])['regions']==[region]


def test_omitted_known_option_is_rejected():
    value=draft();value['items'][0]['options'].pop()
    questions=[{'id':'choice','question_no':'一-1','scoring_mode':'choice_set','prompt':'A. a B. b C. c D. d','answer_key':'B'},
               {'id':'text','question_no':'二-1','scoring_mode':'numeric'}]
    with pytest.raises(ValueError,match='遗漏'):
        templates.validate_layout(value,questions,1)


@pytest.mark.parametrize('text,confidence', [('BC',.99),('B',1.5),('B',float('nan'))])
def test_single_choice_text_ambiguity_and_bad_confidence_remain_missing(workspace,monkeypatch,cv,text,confidence):
    from grading_agent.repository import get_assessment
    from grading_agent.services.ingestion import NormalizedPage
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    assessment=get_assessment(a['id']);layout=templates.load_layout(assessment)
    layout['items'][0]['method']='text';layout['items'][0]['options']=[]
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *a:(text,confidence,'synthetic'))
    path=config.get_settings().data_dir/'synthetic-student.png';mark(source).save(path)
    answers,warnings,_=local.recognize_answers(layout,[NormalizedPage(1,path,900,1200,'ok',{})],[assessment['questions'][0]],config.get_settings())
    assert not answers and warnings



def test_noncontiguous_option_labels_rejected():
    value=draft();value['items'][0]['options'].pop(1)
    with pytest.raises(ValueError,match='不连续'):
        templates.validate_layout(value,[{'id':'choice','question_no':'1','scoring_mode':'choice_set'},
                                        {'id':'text','question_no':'2','scoring_mode':'numeric'}],1)


def test_old_saved_invalid_template_is_blocked_without_remote_fallback(workspace,cv):
    client,a,qs,doc,source,calls=workspace
    generate(workspace)
    with transaction() as db:
        row=db.execute('SELECT payload FROM assessment_layouts WHERE assessment_id=?',(a['id'],)).fetchone()
        payload=decode_json(row['payload'],{})
        payload['items'][0]['options'].pop(1)
        db.execute('UPDATE assessment_layouts SET payload=? WHERE assessment_id=?',(json.dumps(payload),a['id']))
    state=client.get(f"/api/assessments/{a['id']}/layout-template").json()
    assert state['enabled'] and state['stale'] and state['warnings']
    student=upload(workspace,mark(source))
    assert not student['responses'] and student['status']=='extraction_failed' and len(calls)==1
