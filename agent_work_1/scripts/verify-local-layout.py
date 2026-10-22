"""Synthetic end-to-end check with real local OCR/OMR and NO remote model calls.

Run with the existing OCR Python. Template coordinates are a fixture, not an
evaluation of a remote model's ability to generate accurate coordinates.
"""
import io
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from unittest.mock import patch
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['GRADING_DATA_DIR']=str(ROOT/'output/local-layout-real-check')

from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from grading_agent import config
from grading_agent.app import app
from grading_agent.services import layout_templates as templates


def png(image):
    stream=io.BytesIO();image.save(stream,'PNG');return stream.getvalue()


def main():
    settings=config.get_settings()
    config._settings_cache=replace(settings,openai_api_key='synthetic-no-network',ocr_provider='paddleocr',
                                  ocr_device='gpu:0',grading_profile='daily',review_sample_rate=0)
    root=settings.data_dir
    image=Image.new('RGB',(900,1200),'white')
    draw=ImageDraw.Draw(image)
    rng=random.Random(812)
    for n in range(500):
        x,y=rng.randrange(25,870),rng.randrange(25,1170)
        if 80<x<730 and (160<y<400 or 560<y<790):continue
        draw.text((x,y),str(n),fill='black')
    draw.text((85,145),'I. CHOICE - 1',fill='black')
    for x in [135,225,315,405]:draw.ellipse((x,240,x+36,276),outline='black',width=2)
    draw.text((85,570),'II. NUMERIC - 1',fill='black')
    student=image.copy();d=ImageDraw.Draw(student)
    d.ellipse((229,244,257,272),fill='black')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',80)
    d.text((125,625),'42',font=font,fill='black')
    image.save(root/'blank.png');student.save(root/'student.png')
    calls=[]
    def forbidden(*args,**kwargs):raise AssertionError('Remote student recognition/scoring is forbidden in this check')
    with TestClient(app) as client:
        a=client.post('/api/assessments',json={'title':'本地识别验证 · 合成试卷','subject_codes':[]}).json()
        qs=[]
        for number,kind,mode,key in [('一-1','single_choice','choice_set','B'),('二-1','numeric','numeric','42')]:
            q=client.post(f"/api/assessments/{a['id']}/questions",json={'question_no':number,'prompt':number,
                         'question_type':kind,'scoring_mode':mode,'max_score':5,'answer_key':key})
            assert q.status_code==200,q.text
            qs.append(q.json())
        # Register the blank fixture without OCR at upload; answer OCR below is real.
        with patch('grading_agent.app.extract_reference_text',return_value=('', 'synthetic')):
            doc=client.post(f"/api/assessments/{a['id']}/documents",data={'kind':'answer_sheet'},
                            files=[('files',('blank.png',png(image),'image/png'))]).json()[0]
        def b(x,y,w,h):return dict(page=1,x=x,y=y,width=w,height=h)
        result={'items':[
            dict(question_id=qs[0]['id'],section='一、选择题',number='1',method='omr',confidence=.99,
                 regions=[b(.1,.14,.7,.19)],options=[dict(label=label,box=b(x/900,.2,.04,.03))
                     for label,x in zip('ABCD',[135,225,315,405])]),
            dict(question_id=qs[1]['id'],section='二、填空题',number='1',method='text',confidence=.99,
                 regions=[b(.1,.5,.7,.12)],options=[])]}
        def fixed_layout(*args):
            calls.append('fixture_template');result['source_review']={'answer_content':'blank','composite_scan':False,'reason':'Synthetic blank sheet'};return {'output_text':json.dumps(result),'usage':{}},'synthetic'
        with patch.object(templates,'request_json',side_effect=fixed_layout):
            for _ in range(2):
                response=client.post(f"/api/assessments/{a['id']}/layout-template",json={'document_id':doc['id']})
                assert response.status_code==200,response.text
        assert len(calls)==1
        client.post(f"/api/assessments/{a['id']}/publish")
        started=perf_counter()
        with patch('grading_agent.app.extract_answers_with_model',side_effect=forbidden), \
             patch('grading_agent.services.llm_grader.OpenAIGrader.grade',side_effect=forbidden):
            response=client.post('/api/submissions',data={'assessment_id':a['id'],'student_ref':'synthetic'},
                                 files={'file':('student.png',png(student),'image/png')})
            assert response.status_code==200,response.text
            submission=response.json()
            grade=client.post(f"/api/submissions/{submission['id']}/grade")
            assert grade.status_code==200,grade.text
        elapsed=perf_counter()-started
        scores=grade.json()['grade_items']
        answers={r['question_no']:r['raw_text'] for r in submission['responses']}
        assert answers=={'一-1':'B','二-1':'42'},answers
        assert all(g['score']==5 and g['grader_kind']=='rules' for g in scores),scores
        report={'synthetic_only':True,'real_local_ocr':True,'template_model':'mocked known coordinates',
                'answers':answers,'scores':{g['question_no']:g['score'] for g in scores},
                'remote_requests':0,'template_fixture_calls':len(calls),'elapsed_seconds':round(elapsed,3),
                'normalization':submission['normalization_report'],'assessment_id':a['id']}
        (root/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='normalization'},ensure_ascii=True))


if __name__=='__main__':main()
