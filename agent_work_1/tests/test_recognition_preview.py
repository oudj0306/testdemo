import copy
from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from grading_agent import config, optimization_api, repository
from grading_agent.services import local_vision


@pytest.fixture
def preview(monkeypatch):
    settings=replace(config.get_settings(),local_vision_enabled=True)
    monkeypatch.setattr(config,'_settings_cache',settings)
    settings.crop_dir.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(100,60),'white').save(settings.crop_dir/'answer.png')
    submission={'id':'s','assessment_id':'a','responses':[{'raw_text':'人工校正','source':'manual'}],
                'grade_items':[{'score':5}], 'normalization_report':{'local_recognition':{
                    'recognitions':{'q':{'parts':[{'crop':'answer.png','text':'旧转写'}]}}}}}
    question={'id':'q','question_type':'short_answer'}
    monkeypatch.setattr(repository,'get_submission',lambda _:submission)
    monkeypatch.setattr(optimization_api,'get_assessment',lambda _:dict(questions=[question]))
    calls=[]
    def transcribe(paths,q,settings):
        calls.append((paths,q))
        return {'text':'新候选','consistent':False,'issues':['需核对'],'requests':[]}
    monkeypatch.setattr(local_vision,'transcribe',transcribe)
    app=FastAPI();app.include_router(optimization_api.router)
    return TestClient(app),submission,calls,settings


def test_preview_reads_original_crop_and_does_not_change_manual_answers_or_grades(preview):
    client,submission,calls,settings=preview
    before=copy.deepcopy(submission)
    response=client.post('/api/submissions/s/local-recognition-preview?question_id=q&part=0')
    assert response.status_code==200,response.text
    assert response.json()['text']=='新候选'
    assert response.json()['preview_only'] and not response.json()['writes_grades']
    assert submission==before
    assert calls[0][0]==[(settings.crop_dir/'answer.png').resolve()]


def test_preview_validates_question_part_and_crop_path(preview):
    client,submission,calls,settings=preview
    base='/api/submissions/s/local-recognition-preview?question_id='
    assert client.post(base+'other').status_code==404
    assert client.post(base+'q&part=-1').status_code==422
    assert client.post(base+'q&part=1').status_code==404
    submission['normalization_report']['local_recognition']['recognitions']['q']['parts'][0]['crop']='../outside.png'
    assert client.post(base+'q').status_code==404
    assert not calls


def test_historical_joint_fill_in_crops_are_retried_independently(preview):
    client,submission,calls,settings=preview
    Image.new('RGB',(100,60),'white').save(settings.crop_dir/'second.png')
    part=submission['normalization_report']['local_recognition']['recognitions']['q']['parts'][0]
    part['crops']=['answer.png','second.png']
    response=client.post('/api/submissions/s/local-recognition-preview?question_id=q')
    assert response.status_code==200,response.text
    assert [call[0] for call in calls]==[(settings.crop_dir/'answer.png').resolve(),(settings.crop_dir/'second.png').resolve()]
    assert len(response.json()['region_results'])==2
