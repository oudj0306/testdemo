"""Automatic API transcription; one wider-page retry for unreadable crops."""
from pathlib import Path

from .answer_extractor import reusable_answer_ids
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
    attempts=[];total={'input_tokens':0,'output_tokens':0,'cached_tokens':0}
    def request(options,source):
        try:
            answers,usage=extract(document,[question],settings,**options)
            for key in total:total[key]+=usage.get(key,0)
            if len(answers)!=1 or answers[0]['question_no']!=number:raise ValueError('API未返回唯一的指定题号')
            answer=answers[0]
            attempts.append({'input_source':source,'success':True,'has_text':bool(answer['raw_text'].strip()),'blank_confirmed':answer['blank_confirmed']})
            return answer
        except Exception as exc:
            usage=getattr(exc,'usage',None)
            if usage:
                for key in total:total[key]+=usage.get(key,0)
            attempts.append({'input_source':source,'success':False})
            exc.usage={**total,'attempts':attempts}
            raise
    source='regions' if use_regions else 'pages'
    answer=request(options,source)
    # A valid API transcription wins immediately, regardless of local disagreement.
    # Only an empty, unconfirmed crop response gets one wider-context request.
    if use_regions and not answer['raw_text'].strip() and not answer['blank_confirmed']:
        needed={b['page'] for b in item['regions']}
        pages=[page for page in document.pages if page.page_no in needed]
        if pages:
            answer=request({'target_question_nos':[number],'answer_layout':'on_paper','page_images':pages},'context_pages')
            source='context_pages'
    return answer,{**total,'attempts':attempts},source


def supplement_local_answers(submission,questions,document,settings,diagnostics,layout,extract):
    diagnostics['api_fallback']={};diagnostics['remote_vision_requests']=0
    diagnostics['policy_version']='api-authoritative-v2'
    if not settings.hybrid_fallback_enabled:return
    reliable=reusable_answer_ids(questions,get_submission(submission['id'])['responses'],settings)
    for question in questions:
        qid=question['id']
        if qid in reliable:continue
        parts=diagnostics.get('recognitions',{}).get(qid,{}).get('parts',[])
        reasons=list(dict.fromkeys(issue for part in parts for issue in part.get('issues',[])))
        reasons+= [w.split('：',1)[-1] for w in diagnostics.get('warnings',[]) if w.startswith(question['question_no']+'：')]
        info={'question_no':question['question_no'],'attempted':False,
              'trigger_reasons':list(dict.fromkeys(reasons)) or ['本地未取得可靠的完整作答']}
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
            info.update(candidate=answer,input_source=source,attempts=usage.get('attempts',[]))
            diagnostics['remote_vision_requests']+=max(0,len(usage.get('attempts',[]))-1)
            # A teacher correction committed during the request always wins.
            current=get_submission(submission['id'])
            if qid in reusable_answer_ids(questions,current['responses'],settings):continue
            confirmed_blank=answer['blank_confirmed']
            upsert_response(submission['id'],qid,answer['raw_text'],answer['confidence'],confirmed_blank,'api_vision_fallback')
            info['adopted']=bool(answer['raw_text'].strip()) or confirmed_blank
        except Exception as exc:
            usage=getattr(exc,'usage',None)
            if usage:
                info['attempts']=usage.get('attempts',[])
                diagnostics['remote_vision_requests']+=max(0,len(usage.get('attempts',[]))-1)
            if usage:record_token_usage(submission['assessment_id'],submission['id'],'extract_fallback',usage['input_tokens'],usage['output_tokens'],usage['cached_tokens'])
            info['error']=str(exc)[:500]


def needs_recognition(submission,questions,settings):
    if submission['status'] in {'uploaded','extraction_failed'}:return True
    if not settings.hybrid_fallback_enabled:return False
    reliable=reusable_answer_ids(questions,submission.get('responses',[]),settings)
    # A retry of an already graded paper must still re-enter recognition for missing answers.
    return any(q['id'] not in reliable for q in questions)


def supplement_saved_submission(submission,assessment,settings,extract):
    """Resume old local-only work without running the resident local model again."""
    from copy import deepcopy
    from .ingestion import IngestedDocument,NormalizedPage
    from .layout_templates import load_layout
    from ..db import transaction,encode_json
    report=deepcopy(submission.get('normalization_report') or {})
    diagnostics=report.get('local_recognition')
    if diagnostics is None or not settings.hybrid_fallback_enabled:return None
    source=Path(submission['source_path']).resolve()
    if not source.is_relative_to(settings.data_dir.resolve()) or not source.is_file():raise ValueError('答卷原件不可访问')
    pages=[]
    for page in submission['pages']:
        path=Path(page['image_path']).resolve()
        if not path.is_relative_to(settings.data_dir.resolve()) or not path.is_file():raise ValueError('答卷页面不可访问')
        pages.append(NormalizedPage(page['page_no'],path,page['width'],page['height'],page['quality_state'],page['quality_report']))
    document=IngestedDocument(submission['source_hash'],source,submission['media_type'],pages)
    supplement_local_answers(submission,assessment['questions'],document,settings,diagnostics,load_layout(assessment),extract)
    current=get_submission(submission['id'])
    reliable=reusable_answer_ids(assessment['questions'],current['responses'],settings)
    missing=[q['question_no'] for q in assessment['questions'] if q['id'] not in reliable]
    diagnostics.update(mode='hybrid',missing=missing)
    with transaction() as db:
        db.execute('UPDATE submissions SET status=?,normalization_report=? WHERE id=?',('extraction_failed' if missing else 'extracted',encode_json(report),submission['id']))
    return get_submission(submission['id'])
