"""One API supplement per uncertain question; preserve local and human answers."""
from pathlib import Path

from .answer_extractor import confidence_threshold,reusable_answer_ids
from ..repository import get_submission,upsert_response,record_token_usage


def extract_uncertain_question(document,question,settings,diagnostics,layout,extract):
    number=question['question_no'];qid=question['id']
    item=next((i for i in (layout or {}).get('items',[]) if i['question_id']==qid),None)
    if item and not (layout or {}).get('stale'):
        available={p.page_no for p in document.pages}
        needed={b['page'] for b in item['regions']}
        if not needed<=available:raise ValueError('该题页面缺失，不能用API将缺页认定为空白')
    parts=diagnostics.get('recognitions',{}).get(qid,{}).get('parts',[])
    names=[name for part in parts for name in (part.get('crops') or [part.get('crop','')])]
    paths=[(settings.crop_dir/name).resolve() for name in names]
    use_regions=bool(item and not (layout or {}).get('stale') and paths and len(paths)==len(item['regions'])
                     and len(set(paths))==len(paths) and all(p.is_relative_to(settings.crop_dir.resolve()) and p.is_file() for p in paths))
    options={'target_question_nos':[number],'answer_layout':'on_paper'}
    if use_regions:options['region_images']={number:paths}
    answers,usage=extract(document,[question],settings,**options)
    if len(answers)!=1 or answers[0]['question_no']!=number:raise ValueError('API未返回唯一的指定题号')
    return answers[0],usage,'regions' if use_regions else 'pages'


def supplement_local_answers(submission,questions,document,settings,diagnostics,layout,extract):
    diagnostics['api_fallback']={};diagnostics['remote_vision_requests']=0
    if not settings.hybrid_fallback_enabled:return
    reliable=reusable_answer_ids(questions,get_submission(submission['id'])['responses'],settings)
    for question in questions:
        qid=question['id']
        if qid in reliable:continue
        info={'question_no':question['question_no'],'attempted':False}
        diagnostics['api_fallback'][qid]=info
        if not settings.openai_api_key or not settings.llm_extract_answers:
            info['error']='未启用视觉API识别或尚未配置API密钥，可人工确认作答'
            continue
        try:
            # Validate known missing pages before counting a network attempt.
            item=next((i for i in (layout or {}).get('items',[]) if i['question_id']==qid),None)
            if item and not layout.get('stale') and not {b['page'] for b in item['regions']}<={p.page_no for p in document.pages}:
                raise ValueError('该题页面缺失，保留缺分，请补充页面')
            info['attempted']=True;diagnostics['remote_vision_requests']+=1
            answer,usage,source=extract_uncertain_question(document,question,settings,diagnostics,layout,extract)
            record_token_usage(submission['assessment_id'],submission['id'],'extract_fallback',usage['input_tokens'],usage['output_tokens'],usage['cached_tokens'])
            info.update(candidate=answer,input_source=source)
            # A teacher correction committed during the request always wins.
            current=get_submission(submission['id'])
            if qid in reusable_answer_ids(questions,current['responses'],settings):continue
            confirmed_blank=answer['blank_confirmed'] and answer['confidence']>=confidence_threshold(question,settings)
            upsert_response(submission['id'],qid,answer['raw_text'],answer['confidence'],confirmed_blank,'api_vision_fallback')
        except Exception as exc:
            usage=getattr(exc,'usage',None)
            if usage:record_token_usage(submission['assessment_id'],submission['id'],'extract_fallback',usage['input_tokens'],usage['output_tokens'],usage['cached_tokens'])
            info['error']=str(exc)[:500]
