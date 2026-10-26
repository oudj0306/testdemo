import json
from dataclasses import replace

import pytest
from PIL import ImageDraw

from grading_agent import config
from grading_agent.repository import get_submission
from grading_agent.services import local_vision as vision
from grading_agent.services.math_grader import GradeDecision
from grading_agent.schemas import ScoreState
from test_local_layout import workspace,generate,upload,mark
from test_local_vision import enable_visual


def enable_hybrid(monkeypatch):
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),hybrid_fallback_enabled=True))


def unreadable_image(source):
    image=mark(source);ImageDraw.Draw(image).text((130,630),'unreadable',fill='black');return image


@pytest.mark.parametrize('blank',[False,True])
def test_uncertain_question_alone_uses_region_api_and_scores(workspace,monkeypatch,blank):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_hybrid(monkeypatch)
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *args:('4?',.3,'ocr'))
    requests=[]
    def extract(document,questions,settings,**kwargs):
        requests.append((questions,kwargs))
        return [dict(question_no=qs[1]['question_no'],raw_text='' if blank else '42',confidence=.99,blank_confirmed=blank)],dict(input_tokens=10,output_tokens=4,cached_tokens=0)
    monkeypatch.setattr('grading_agent.app.extract_answers_with_model',extract)
    student=upload(workspace,unreadable_image(source))
    assert len(requests)==1 and [q['id'] for q in requests[0][0]]==[qs[1]['id']]
    assert set(requests[0][1]['region_images'])=={qs[1]['question_no']}
    assert student['normalization_report']['local_recognition']['remote_vision_requests']==1
    assert student['normalization_report']['local_recognition']['missing']==[]
    result=client.post(f"/api/submissions/{student['id']}/grade").json()
    assert {g['question_no']:g['score'] for g in result['grade_items']}=={'一-1':5.,'二-1':0. if blank else 5.}
    assert len(requests)==1  # Grade reuses the supplemented answer.


def test_api_failure_preserves_local_answer_and_keeps_candidates_visible(workspace,monkeypatch):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_hybrid(monkeypatch)
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *args:('4?',.3,'ocr'))
    attempts=[]
    def failed(*args,**kwargs):attempts.append(True);raise RuntimeError('provider unavailable')
    monkeypatch.setattr('grading_agent.app.extract_answers_with_model',failed)
    student=upload(workspace,unreadable_image(source))
    assert len(attempts)==1 and student['responses'][0]['raw_text']=='B'
    report=student['normalization_report']['local_recognition']
    assert report['recognitions'][qs[1]['id']]['parts'][0]['text']=='4?'
    assert 'provider unavailable' in report['api_fallback'][qs[1]['id']]['error']
    assert report['missing']==['二-1']
    # Manual entry bypasses recognition and processes ONLY this question.
    response=client.post(f"/api/submissions/{student['id']}/questions/{qs[1]['id']}/confirm-answer",json={'raw_text':'42'})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['grade_items'][0]['score']==5 and len(result['grade_items'])==1
    assert next(r for r in result['responses'] if r['question_id']==qs[1]['id'])['extraction_source']=='manual'
    assert len(attempts)==1


def test_manual_blank_direct_zero_preserves_other_human_grade_and_validates_input(workspace):
    client,a,qs,doc,source,calls=workspace;generate(workspace)
    student=upload(workspace,mark(source));sid=student['id']
    first=client.post(f'/api/submissions/{sid}/questions/{qs[1]["id"]}/confirm-answer',json={'raw_text':'42'}).json()
    grade=first['grade_items'][0]
    assert client.post('/api/grade-items/'+grade['id']+'/review',json={'final_score':4,'reason':'teacher check'}).status_code==200
    result=client.post(f'/api/submissions/{sid}/questions/{qs[0]["id"]}/confirm-answer',json={'blank_confirmed':True}).json()
    byq={g['question_id']:g for g in result['grade_items']}
    assert byq[qs[0]['id']]['score']==0
    assert byq[qs[1]['id']]['id']==grade['id'] and byq[qs[1]['id']]['final_score']==4
    assert client.post(f'/api/submissions/{sid}/questions/{qs[0]["id"]}/confirm-answer',json={}).status_code==422
    assert client.post(f'/api/submissions/{sid}/questions/{qs[0]["id"]}/confirm-answer',json={'raw_text':'x','blank_confirmed':True}).status_code==422
    assert client.post(f'/api/submissions/{sid}/questions/foreign/confirm-answer',json={'raw_text':'x'}).status_code==404


def test_two_explicit_blank_observations_are_adopted_without_pixel_veto(workspace,monkeypatch):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_visual(monkeypatch);enable_hybrid(monkeypatch)
    monkeypatch.setattr(vision,'transcribe',lambda *args:dict(text='',consistent=True,issues=[],blank_candidate=True,requests=[]))
    student=upload(workspace,unreadable_image(source))
    response=next(r for r in student['responses'] if r['question_id']==qs[1]['id'])
    assert response['blank_confirmed']
    assert student['normalization_report']['local_recognition']['remote_vision_requests']==0
    graded=client.post(f"/api/submissions/{student['id']}/grade").json()
    assert next(g for g in graded['grade_items'] if g['question_id']==qs[1]['id'])['score']==0


def test_rule_parse_problem_uses_model_but_exact_numeric_wrong_answer_does_not(workspace,monkeypatch):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_hybrid(monkeypatch)
    student=upload(workspace,mark(source));sid=student['id'];requests=[]
    def grade(self,question,answer):
        requests.append((question,answer))
        return GradeDecision(5,5,ScoreState.AUTO_SINGLE,False,['model_numeric_match'],grader_kind='llm')
    monkeypatch.setattr('grading_agent.services.grading.OpenAIGrader.grade',grade)
    result=client.post(f'/api/submissions/{sid}/questions/{qs[1]["id"]}/confirm-answer',json={'raw_text':'42分'}).json()
    assert result['grade_items'][0]['score']==5 and len(requests)==1
    assert 'rule_fallback_to_model' in result['grade_items'][0]['reason_codes']
    result=client.post(f'/api/submissions/{sid}/questions/{qs[1]["id"]}/confirm-answer',json={'raw_text':'43'}).json()
    assert result['grade_items'][0]['score']==0 and len(requests)==1


def test_low_confidence_remote_blank_is_not_zero(workspace,monkeypatch):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_hybrid(monkeypatch)
    monkeypatch.setattr('grading_agent.services.ingestion.OCRAdapter.extract',lambda *args:('4?',.3,'ocr'))
    monkeypatch.setattr('grading_agent.app.extract_answers_with_model',lambda *args,**kwargs:([dict(question_no=qs[1]['question_no'],raw_text='',confidence=.2,blank_confirmed=True)],dict(input_tokens=1,output_tokens=1,cached_tokens=0)))
    student=upload(workspace,unreadable_image(source))
    response=next(r for r in student['responses'] if r['question_id']==qs[1]['id'])
    assert not response['blank_confirmed'] and response['raw_text']==''
    assert student['normalization_report']['local_recognition']['missing']==['二-1']


def test_api_preview_leaves_responses_and_human_scores_unchanged(workspace,monkeypatch):
    from grading_agent.db import transaction
    client,a,qs,doc,source,calls=workspace;generate(workspace)
    student=upload(workspace,mark(source));sid=student['id']
    saved=client.post(f'/api/submissions/{sid}/questions/{qs[1]["id"]}/confirm-answer',json={'raw_text':'42'}).json()
    grade=saved['grade_items'][0]
    client.post('/api/grade-items/'+grade['id']+'/review',json={'final_score':4,'reason':'teacher check'})
    before=get_submission(sid)
    seen=[]
    def extract(document,questions,settings,**kwargs):
        seen.append(kwargs)
        return [dict(question_no=qs[1]['question_no'],raw_text='43',confidence=.99,blank_confirmed=False)],dict(input_tokens=12,output_tokens=3,cached_tokens=0)
    monkeypatch.setattr('grading_agent.services.answer_extractor.extract_answers_with_model',extract)
    response=client.post(f'/api/submissions/{sid}/questions/{qs[1]["id"]}/api-recognition-preview')
    assert response.status_code==200,response.text
    assert response.json()['preview_only'] and not response.json()['writes_grades']
    assert response.json()['candidate']['raw_text']=='43' and response.json()['input_source']=='regions'
    after=get_submission(sid)
    assert before['responses']==after['responses'] and before['grade_items']==after['grade_items']
    assert set(seen[0]['region_images'])=={qs[1]['question_no']}
    with transaction() as db:
        rows=db.execute("SELECT input_tokens,output_tokens FROM token_usage WHERE submission_id=? AND stage='extract_preview'",(sid,)).fetchall()
    assert [(r['input_tokens'],r['output_tokens']) for r in rows]==[(12,3)]


def test_missing_page_cannot_be_sent_to_api_or_assumed_blank(tmp_path):
    from grading_agent.services.hybrid_recognition import extract_uncertain_question
    from grading_agent.services.ingestion import IngestedDocument
    question={'id':'q','question_no':'1'}
    layout={'items':[{'question_id':'q','regions':[{'page':2}]}]}
    document=IngestedDocument('hash',tmp_path/'missing.png','image/png',[])
    with pytest.raises(ValueError,match='页面缺失'):
        extract_uncertain_question(document,question,config.get_settings(),{},layout,lambda *a,**kw:pytest.fail('missing page must not call API'))


@pytest.mark.parametrize('issue',['两次转写存在差异，不能自动确定','检测到涂改，即使两次一致也须补识','文字难以辨认'])
def test_local_special_cases_automatically_use_api_result(workspace,monkeypatch,issue):
    client,a,qs,doc,source,calls=workspace;generate(workspace);enable_visual(monkeypatch);enable_hybrid(monkeypatch)
    monkeypatch.setattr(vision,'transcribe',lambda *args:dict(text='local candidate',consistent=False,issues=[issue],blank_candidate=False,requests=[]))
    requests=[]
    def extract(*args,**kwargs):
        requests.append(kwargs)
        return [dict(question_no=qs[1]['question_no'],raw_text='42',confidence=.3,blank_confirmed=False)],dict(input_tokens=10,output_tokens=2,cached_tokens=0)
    monkeypatch.setattr('grading_agent.app.extract_answers_with_model',extract)
    student=upload(workspace,unreadable_image(source))
    response=next(r for r in student['responses'] if r['question_id']==qs[1]['id'])
    assert response['extraction_source']=='api_vision_fallback' and response['raw_text']=='42'
    assert response['ocr_confidence']==.3  # No fabricated confidence boost.
    info=student['normalization_report']['local_recognition']['api_fallback'][qs[1]['id']]
    assert info['adopted'] and issue in info['trigger_reasons']
    assert student['normalization_report']['local_recognition']['missing']==[]
    result=client.post(f"/api/submissions/{student['id']}/grade").json()
    grade=next(g for g in result['grade_items'] if g['question_id']==qs[1]['id'])
    assert grade['score']==5 and grade['grader_kind']=='rules' and grade['needs_review']
    assert len(requests)==1
