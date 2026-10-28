from dataclasses import replace
import sqlite3

import pytest

from grading_agent import config
from grading_agent.schemas import QuestionCreate
from grading_agent.services.fixed_answers import grade_fixed_answers
from grading_agent.services.grading import _grade_question, requires_model
from grading_agent.services.grading_cache import review_revision
from grading_agent.services import local_vision as vision
from test_local_layout import workspace,generate,upload,mark


def question():
    return dict(id='q',question_no='二-1',prompt='依次填空',question_type='short_answer',
                scoring_mode='rubric_assisted',max_score=4,answer_key='控制器，存储器',
                answer_key_state='verified',rubric=[dict(criterion='每空2分',score=4)],
                rule_config={'fixed_answers':{'blanks':[
                    {'accepted':['控制器'],'score':2},{'accepted':['存储器','内存储器'],'score':2}]}})


def response(text):return dict(raw_text=text,extraction_source='api_vision_fallback',ocr_confidence=None)


@pytest.mark.parametrize('text,score',[('控制器，存储器',4),('控制器\n内存储器',4),('控制器；[未作答]',2)])
def test_configured_blanks_remove_model_calls(monkeypatch,text,score):
    monkeypatch.setattr('grading_agent.services.grading.OpenAIGrader.grade',lambda *a:pytest.fail('rule must not call API'))
    q=question();r=response(text);settings=config.get_settings()
    assert not requires_model(q,r,settings)
    result=_grade_question(q,r,settings,'student')
    assert result.score==score and result.grader_kind=='rules'


@pytest.mark.parametrize('text',['控制器 存储器','控制器，寄存器','控制器，存储器，运算器','(1)控制器，(2)存储器'])
def test_ambiguous_answers_never_silently_split_or_score_zero(text):
    q=question()
    assert grade_fixed_answers(q,response(text)) is None
    assert requires_model(q,response(text),config.get_settings())


def test_strict_mismatch_and_provisional_keys():
    q=question();q['rule_config']['fixed_answers']['strict_mismatch']=True
    assert grade_fixed_answers(q,response('控制器，寄存器')).score==2
    q['answer_key_state']='candidate'
    assert grade_fixed_answers(q,response('控制器，存储器')) is None
    q['scoring_mode']='manual'
    assert not requires_model(q,response('控制器，存储器'))


@pytest.mark.parametrize('problem',['sum','negative','nan','empty','bool'])
def test_invalid_standards_rejected_before_saving(problem):
    q=question();rule=q['rule_config']['fixed_answers']
    if problem=='sum':rule['blanks'][0]['score']=1
    if problem=='negative':rule['blanks'][0]['score']=-2
    if problem=='nan':rule['blanks'][0]['score']=float('nan')
    if problem=='empty':rule['blanks'][0]['accepted']=[' ']
    if problem=='bool':rule['blanks'][0]['score']=True
    with pytest.raises(ValueError):QuestionCreate.model_validate(q)


def test_confirmation_does_not_bust_cache_but_correction_does():
    settings=config.get_settings()
    with sqlite3.connect(settings.database_path) as db:
        db.execute('CREATE TABLE grade_items(id TEXT,question_id TEXT,score REAL)')
        db.execute('CREATE TABLE review_events(id TEXT,grade_item_id TEXT,created_at TEXT,final_score REAL)')
        db.execute("INSERT INTO grade_items VALUES('g','q',4)")
    before=review_revision(settings,'q')
    with sqlite3.connect(settings.database_path) as db:
        db.execute("INSERT INTO review_events VALUES('r','g','t1',4)")
    assert review_revision(settings,'q')==before
    with sqlite3.connect(settings.database_path) as db:
        db.execute("INSERT INTO review_events VALUES('r2','g','t2',2)")
    assert review_revision(settings,'q')!=before


@pytest.mark.parametrize('kind',['programming','short_answer'])
def test_hard_or_explicit_api_route_skips_all_local_inference(workspace,monkeypatch,kind):
    client,a,qs,doc,source,calls=workspace;q=qs[1]
    assert client.put('/api/questions/'+q['id'],json={**q,'question_type':kind,
        'rule_config':{'recognition_route':'api'} if kind=='short_answer' else {}}).status_code==200
    generate(workspace)
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),local_vision_enabled=True,hybrid_fallback_enabled=True))
    monkeypatch.setattr(vision,'runtime',lambda *a:pytest.fail('must not load local VLM'))
    monkeypatch.setattr(vision,'transcribe',lambda *a:pytest.fail('must not transcribe locally'))
    # Exercise real hybrid orchestration while mocking only its paid extractor.
    def extract(*args,**kwargs):
        return ([{'question_no':q['question_no'],'raw_text':'42','confidence':.95,'blank_confirmed':False}],
                {'input_tokens':10,'output_tokens':2,'cached_tokens':0})
    monkeypatch.setattr('grading_agent.app.extract_answers_with_model',extract)
    student=upload(workspace,mark(source))
    report=student['normalization_report']['local_recognition']
    assert report['recognitions'][q['id']]['route']=='api_direct'
    assert report['local_model_usage']['requests']==0
    assert next(r for r in student['responses'] if r['question_id']==q['id'])['extraction_source']=='api_vision_fallback'


def test_numeric_ocr_skips_vlm_and_keeps_honest_source(workspace,monkeypatch):
    _,_,qs,_,source,_=workspace;generate(workspace)
    monkeypatch.setattr(config,'_settings_cache',replace(config.get_settings(),local_vision_enabled=True))
    monkeypatch.setattr(vision,'runtime',lambda *a:pytest.fail('numeric OCR needs no VLM'))
    monkeypatch.setattr(vision,'transcribe',lambda *a:pytest.fail('numeric OCR needs no VLM'))
    student=upload(workspace,mark(source))
    r=next(r for r in student['responses'] if r['question_id']==qs[1]['id'])
    assert r['raw_text']=='42' and r['extraction_source']=='local_layout_ocr'
    assert r['ocr_confidence']==.99
