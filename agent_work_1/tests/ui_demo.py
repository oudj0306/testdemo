"""Synthetic UI acceptance data. Refuses to use a production data directory."""
import io
import os
from pathlib import Path

from PIL import Image

from grading_agent.app import store_submission
from grading_agent.db import initialize_database
from grading_agent.repository import (create_assessment, create_question, publish_assessment,
    upsert_response, replace_active_grades, get_submission, create_review, record_token_usage)


def seed():
    if not Path(os.environ.get('GRADING_DATA_DIR','')).name.startswith('.test-run-ui-'):
        raise RuntimeError('UI fixture requires a separate .test-run-ui-* directory')
    initialize_database()
    a=create_assessment('九年级数学 · 第一单元学习检测',[])
    definitions=[('1','计算：3² + 4² 的值，并写出计算过程。','25',15,'先计算平方各得 5 分，相加得 5 分。'),
                 ('2','解方程：2(x − 3) = 10，写出必要的求解步骤。','2x − 6 = 10；2x = 16；x = 8。',20,'去括号 5 分，移项 5 分，合并同类项 5 分，求解 5 分。'),
                 ('3','一个长方形长 8 cm，宽 5 cm，求它的面积。','8 × 5 = 40 cm²',15,'正确列式与计算 10 分，单位正确 5 分。')]
    for number,prompt,answer,maximum,rubric in definitions:
        create_question(a['id'],dict(question_no=number,prompt=prompt,question_type='calculation',scoring_mode='rubric_assisted',capture_mode='long_handwriting',answer_form='text',answer_key=answer,max_score=maximum,rubric=[{'criterion':rubric,'score':maximum}]))
    a=publish_assessment(a['id'])
    names=['陈予安','林知夏','张一诺','王沐辰','李书涵','周可欣','赵星辰','刘安然','许清和','吴子墨','孙雨桐','郑亦然','何明远']
    for i,name in enumerate(names):
        image=Image.new('RGB',(900,1200),'white');image.putpixel((0,0),(i,0,0));stream=io.BytesIO();image.save(stream,'PNG')
        s=store_submission(a,name,stream.getvalue(),'image/png')
        if i==12:continue
        grades=[]
        for j,q in enumerate(a['questions']):
            text=['3²=9，4²=16，9+16=25。','2x−6=10，2x=16，x=8。','S=8×5=40 cm²。'][j]
            response=upsert_response(s['id'],q['id'],text,.98,False,'openai_vision')
            pending=j==1 and i%3==0
            grades.append(dict(question_id=q['id'],response_id=response['id'],score=q['max_score']-(i%4),max_score=q['max_score'],score_state='CONSENSUS_CANDIDATE' if pending else 'AUTO_STABLE',needs_review=pending,reason_codes=['quality_spot_check'] if pending else ['rubric_assisted'],evidence=['【界面测试数据】评分细则与作答步骤匹配。','independent_pass_scores=[20, 20]'],grader_kind='llm',grader_version='UI-test-fixture'))
        replace_active_grades(s['id'],grades)
        if i==0:
            grade=get_submission(s['id'])['grade_items'][1]
            create_review(grade['id'],'测试教师',18,'界面验收：首次确认')
            create_review(grade['id'],'测试教师',20,'界面验收：再次检查，步骤完整')
        record_token_usage(a['id'],s['id'],'extract',3200+i*20,480,2000)
        record_token_usage(a['id'],s['id'],'grade',1500+i*10,240,1200)
    return a


if __name__=='__main__':
    seed()
    import uvicorn
    uvicorn.run('grading_agent.app:app',host='127.0.0.1',port=8766)
