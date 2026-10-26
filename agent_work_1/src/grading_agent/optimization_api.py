"""Template preview, measured performance, explicitly redacted sample exports."""
import io
import json
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import pypdfium2 as pdfium

from .config import get_settings
from .db import transaction, utc_now, decode_json, encode_json
from .repository import get_assessment, row_dict
from .schemas import RegionInput, TrainingApproval, AnswerConfirmation
from .services.ingestion import serialized_document_io
from .services.telemetry import summary
from .services.document_renderer import WORD_MEDIA, renderer_status, word_to_pdf

router = APIRouter()


@router.post('/api/submissions/{submission_id}/questions/{question_id}/confirm-answer')
def confirm_answer(submission_id: str,question_id: str,payload: AnswerConfirmation):
    from .repository import get_submission,upsert_response
    from .services.grading import grade_submission
    from .services.work_locks import submission_work,SubmissionBusyError
    settings=get_settings()
    try:
        with submission_work(settings.database_path,submission_id,blocking=False):
            submission=get_submission(submission_id)
            if not submission:raise HTTPException(404,'答卷不存在')
            assessment=get_assessment(submission['assessment_id'])
            if question_id not in {q['id'] for q in (assessment or {}).get('questions',[])}:
                raise HTTPException(404,'题目不属于该答卷')
            upsert_response(submission_id,question_id,payload.raw_text,None,payload.blank_confirmed,'manual')
            current=get_submission(submission_id)
            report=current.get('normalization_report',{})
            if 'local_recognition' in report:
                from .services.answer_extractor import reusable_answer_ids
                reliable=reusable_answer_ids(assessment['questions'],current['responses'],settings)
                local=report['local_recognition']
                local['missing']=[q['question_no'] for q in assessment['questions'] if q['id'] not in reliable]
                number=next(q['question_no'] for q in assessment['questions'] if q['id']==question_id)
                local['warnings']=[w for w in local.get('warnings',[]) if not w.startswith(number+'：')]
                with transaction() as db:
                    db.execute('UPDATE submissions SET normalization_report=? WHERE id=?',(encode_json(report),submission_id))
            return grade_submission(submission_id,settings,question_ids={question_id})
    except SubmissionBusyError as exc:raise HTTPException(409,str(exc)) from exc


@router.post('/api/submissions/{submission_id}/questions/{question_id}/api-recognition-preview')
def api_recognition_preview(submission_id: str,question_id: str):
    from .repository import get_submission,record_token_usage
    from .services.ingestion import IngestedDocument,NormalizedPage
    from .services.answer_extractor import extract_answers_with_model
    from .services.hybrid_recognition import extract_uncertain_question
    from .services.layout_templates import load_layout
    from pathlib import Path
    settings=get_settings()
    if not settings.openai_api_key:raise HTTPException(409,'请先配置视觉API模型和密钥')
    submission=get_submission(submission_id)
    if not submission:raise HTTPException(404,'答卷不存在')
    assessment=get_assessment(submission['assessment_id'])
    question=next((q for q in (assessment or {}).get('questions',[]) if q['id']==question_id),None)
    if not question:raise HTTPException(404,'题目不属于该答卷')
    source=Path(submission['source_path']).resolve()
    if not source.is_relative_to(settings.data_dir.resolve()) or not source.is_file():raise HTTPException(404,'原件不可访问')
    pages=[]
    for page in submission['pages']:
        path=Path(page['image_path']).resolve()
        if not path.is_relative_to(settings.data_dir.resolve()) or not path.is_file():raise HTTPException(404,'页面不可访问')
        pages.append(NormalizedPage(page['page_no'],path,page['width'],page['height'],page['quality_state'],page['quality_report']))
    document=IngestedDocument(submission['source_hash'],source,submission['media_type'],pages)
    diagnostics=submission.get('normalization_report',{}).get('local_recognition',{})
    try:
        answer,usage,input_source=extract_uncertain_question(document,question,settings,diagnostics,load_layout(assessment),extract_answers_with_model)
        record_token_usage(submission['assessment_id'],submission_id,'extract_preview',usage['input_tokens'],usage['output_tokens'],usage['cached_tokens'])
        return {'preview_only':True,'writes_grades':False,'candidate':answer,'usage':usage,'input_source':input_source}
    except Exception as exc:
        usage=getattr(exc,'usage',None)
        if usage:record_token_usage(submission['assessment_id'],submission_id,'extract_preview',usage['input_tokens'],usage['output_tokens'],usage['cached_tokens'])
        raise HTTPException(422,str(exc)[:500]) from exc


class LocalVisionSettings(BaseModel):
    enabled: bool
    model: str = Field(default='qwen3-vl:4b-instruct',min_length=1,max_length=120,pattern=r'^[A-Za-z0-9._:/-]+$')
    verify: bool = True


@router.get('/api/runtime/local-vision')
def local_vision_status():
    from .services.local_vision import runtime
    settings=get_settings()
    result={'enabled':settings.local_vision_enabled,'model':settings.local_vision_model,
            'verify':settings.local_vision_verify,'url':settings.local_vision_url,
            'hybrid_fallback_enabled':settings.hybrid_fallback_enabled}
    try:return {**result,**runtime(settings)}
    except ValueError as exc:return {**result,'ready':False,'error':str(exc)}


@router.put('/api/runtime/local-vision')
def local_vision_settings(payload: LocalVisionSettings):
    from .config import update_local_vision
    if 'cloud' in payload.model.lower():raise HTTPException(422,'本地识别禁止云端模型标签')
    if payload.enabled and not payload.verify:raise HTTPException(422,'质量优先模式必须启用两次转写核验')
    update_local_vision(payload.enabled,payload.model,payload.verify)
    return local_vision_status()


@router.get('/api/submissions/{submission_id}/local-recognition-crop')
def local_recognition_crop(submission_id: str, question_id: str, part: int = Query(0,ge=0), region: int = Query(0,ge=0)):
    from .repository import get_submission
    from fastapi.responses import FileResponse
    submission=get_submission(submission_id)
    if not submission:raise HTTPException(404,'答卷不存在')
    parts=submission.get('normalization_report',{}).get('local_recognition',{}).get('recognitions',{}).get(question_id,{}).get('parts',[])
    if part>=len(parts):raise HTTPException(404,'裁切不存在')
    crops=parts[part].get('crops') or [parts[part].get('crop','')]
    if region>=len(crops):raise HTTPException(404,'区域不存在')
    path=(get_settings().crop_dir/crops[region]).resolve()
    if not path.is_relative_to(get_settings().crop_dir.resolve()) or not path.is_file():raise HTTPException(404,'裁切不可访问')
    return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store'})


@router.post('/api/submissions/{submission_id}/local-recognition-preview')
def local_recognition_preview(submission_id: str, question_id: str, part: int = Query(0,ge=0)):
    """Re-transcribe a saved crop without changing responses, grades or human labels."""
    from .repository import get_submission
    from .services.local_vision import transcribe,kind_for
    settings=get_settings()
    if not settings.local_vision_enabled:raise HTTPException(409,'请先启用本地视觉识别')
    submission=get_submission(submission_id)
    if not submission:raise HTTPException(404,'答卷不存在')
    assessment=get_assessment(submission['assessment_id'])
    question=next((q for q in (assessment or {}).get('questions',[]) if q['id']==question_id),None)
    if not question:raise HTTPException(404,'题目不属于该答卷')
    parts=submission.get('normalization_report',{}).get('local_recognition',{}).get('recognitions',{}).get(question_id,{}).get('parts',[])
    if part>=len(parts):raise HTTPException(404,'裁切不存在')
    names=parts[part].get('crops') or [parts[part].get('crop','')]
    if not names or len(names)>3:raise HTTPException(422,'单次预览最多3个连续区域')
    paths=[(settings.crop_dir/name).resolve() for name in names]
    if any(not p.is_relative_to(settings.crop_dir.resolve()) or not p.is_file() for p in paths):
        raise HTTPException(404,'裁切不可访问')
    try:
        if len(paths)>1 and kind_for(question)=='text':
            # Historical reports grouped fill-in blanks as code; re-read each blank separately.
            values=[transcribe(path,question,settings) for path in paths]
            result={**values[0],'text':'\n'.join(value['text'] for value in values),
                    'consistent':all(value['consistent'] for value in values),
                    'blank_candidate':all(value.get('blank_candidate',False) for value in values),
                    'issues':list(dict.fromkeys(issue for value in values for issue in value['issues'])),
                    'requests':[request for value in values for request in value.get('requests',[])],
                    'region_results':values}
        else:result=transcribe(paths,question,settings)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return {'preview_only':True,'writes_grades':False,'question_id':question_id,'part':part,**result}


class LayoutGenerate(BaseModel):
    document_id: str = Field(min_length=1)
    force: bool = False


class LayoutSwitch(BaseModel):
    enabled: bool


class RegionEdit(BaseModel):
    expected_template_id: str
    document_id: str
    regions: list[RegionInput] = Field(max_length=20)
    confirmed_indices: list[int] = Field(default_factory=list, max_length=20)


@router.post('/api/assessments/{assessment_id}/layout-template/calibrate')
def calibrate_layout(assessment_id: str):
    from .services.layout_templates import layout_status, validate_layout
    from .services.region_geometry import calibrate_regions
    assessment=get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404,'试卷不存在')
    state=layout_status(assessment, _include_pages=True)
    if not state['exists'] or state['stale']:
        raise HTTPException(409,'请先匹配有效的位置模板')
    protected={qid for qid,r in state.get('region_reviews',{}).items() if r.get('source')=='manual'}
    protected.update(q['id'] for q in assessment['questions'] if q.get('rule_config',{}).get('region'))
    try:
        items,adjusted=calibrate_regions(state['items'],state['pages'],protected)
        validate_layout({'items':items,'source_review':state.get('source_review')},assessment['questions'],len(state['pages']))
    except ImportError as exc:
        raise HTTPException(422,'本地边线校准需要 OpenCV，请使用 GPU 环境启动') from exc
    with transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT payload FROM assessment_layouts WHERE assessment_id=?',(assessment_id,)).fetchone()
        content=decode_json(row['payload'],{}) if row else {}
        if content.get('template_id')!=state['template_id'] or layout_status(get_assessment(assessment_id))['stale']:
            raise HTTPException(409,'校准期间模板已变化，请刷新后重试')
        content['items']=items
        for qid in adjusted:
            content.setdefault('region_reviews',{})[qid]={'source':'printed_lines','confirmed':[]}
        content['template_id']=hashlib.sha256(encode_json(content).encode()).hexdigest()
        db.execute('UPDATE assessment_layouts SET payload=? WHERE assessment_id=?',(encode_json(content),assessment_id))
    return {**layout_status(assessment),'adjusted_items':len(adjusted)}


@router.put('/api/assessments/{assessment_id}/layout-template/regions/{question_id}')
def edit_layout_regions(assessment_id: str, question_id: str, payload: RegionEdit):
    """Change geometry only; never overwrite question text, keys or rubrics."""
    from .services.layout_templates import layout_status, validate_layout
    from .services.region_geometry import region_key
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, '试卷不存在')
    q = next((q for q in assessment['questions'] if q['id']==question_id), None)
    if not q or q['scoring_mode']=='choice_set':
        raise HTTPException(422, '请选择非选择题')
    with transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM assessment_layouts WHERE assessment_id=?',(assessment_id,)).fetchone()
        if not row:
            raise HTTPException(404, '尚无模板，请先匹配答题卡')
        content=decode_json(row['payload'], {})
        if row['document_id']!=payload.document_id or content.get('template_id')!=payload.expected_template_id:
            raise HTTPException(409, '模板或来源已变化，请重新打开区域后保存')
        if any(i<0 or i>=len(payload.regions) for i in payload.confirmed_indices):
            raise HTTPException(422, '确认区域编号不存在')
        regions=[{k:b.model_dump()[k] for k in ('page','x','y','width','height')} for b in payload.regions]
        old=next((i for i in content['items'] if i['question_id']==question_id), {})
        items=[i for i in content['items'] if i['question_id']!=question_id]
        if regions:
            items.append(dict(question_id=question_id,section=old.get('section','人工校正'),number=old.get('number',q['question_no']),
                              method='text',confidence=1.,regions=regions,options=[]))
        try:
            validate_layout({'items':items,'source_review':content.get('source_review')},assessment['questions'],len(content['pages']))
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        reviews=content.setdefault('region_reviews',{})
        previous=reviews.get(question_id,{}).get('confirmed',[])
        keys=[region_key(b) for b in regions]
        reviews[question_id]={'source':'manual','confirmed':list(dict.fromkeys([k for k in previous if k in keys]+[keys[i] for i in payload.confirmed_indices]))}
        content['items']=items
        content['template_id']=hashlib.sha256(encode_json(content).encode()).hexdigest()
        db.execute('UPDATE assessment_layouts SET payload=? WHERE assessment_id=?',(encode_json(content),assessment_id))
    return layout_status(assessment)


@router.get('/api/assessments/{assessment_id}/layout-template')
def get_layout_template(assessment_id: str):
    from .services.layout_templates import layout_status
    from .services.local_answers import dependencies
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, '试卷不存在')
    return {**layout_status(assessment), 'local_dependencies': dependencies()}


@router.post('/api/assessments/{assessment_id}/layout-template')
def create_layout_template(assessment_id: str, payload: LayoutGenerate):
    from .services.layout_templates import generate_layout
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, '试卷不存在')
    with transaction() as db:
        row = db.execute('SELECT * FROM assessment_documents WHERE assessment_id=? AND id=? AND kind=\'answer_sheet\'',
                         (assessment_id,payload.document_id)).fetchone()
    if not row:
        raise HTTPException(404, '请上传并选择资料类型为“空白答题卡模板”的文件；原试卷不作为答题卡定位来源')
    try:
        return generate_layout(assessment,dict(row),get_settings(),payload.force)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception('Template generation failed')
        raise HTTPException(502,'模板模型调用失败，请检查连接；未自动重试，原模板保持可用') from exc


@router.put('/api/assessments/{assessment_id}/layout-template')
def switch_layout_template(assessment_id: str, payload: LayoutSwitch):
    from .services.layout_templates import layout_status
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404, '试卷不存在')
    state = layout_status(assessment)
    if not state['exists']:
        raise HTTPException(404, '尚无模板')
    if payload.enabled and state['stale']:
        raise HTTPException(409, '题目或来源已变更，请重新生成模板')
    with transaction() as db:
        db.execute('UPDATE assessment_layouts SET enabled=? WHERE assessment_id=?',(int(payload.enabled),assessment_id))
    return layout_status(assessment)


@router.get('/api/assessments/{assessment_id}/layout-template/page')
def layout_overlay(assessment_id: str, page: int = Query(1,ge=1)):
    from PIL import ImageDraw, ImageFont
    from .services.layout_templates import layout_status
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(404,'试卷不存在')
    state = layout_status(assessment)
    if not state['exists']:
        raise HTTPException(404,'尚无模板')
    with transaction() as db:
        row = db.execute('SELECT payload FROM assessment_layouts WHERE assessment_id=?',(assessment_id,)).fetchone()
    payload = decode_json(row['payload'],{})
    spec = next((p for p in payload['pages'] if p['page']==page),None)
    if not spec:
        raise HTTPException(404,'模板无此页')
    path = Path(spec['path']).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()) or not path.is_file():
        raise HTTPException(404,'模板页面不可访问')
    numbers = {q['id']:q['question_no'] for q in assessment['questions']}
    with Image.open(path) as raw:
        image=raw.convert('RGB')
    draw=ImageDraw.Draw(image)
    font=ImageFont.load_default()
    for font_path in ('C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if Path(font_path).is_file():
            font=ImageFont.truetype(font_path,18)
            break
    for item in payload['items']:
        for b in item['regions'] + [o['box'] for o in item['options']]:
            if b['page']==page:
                xy=(int(b['x']*image.width),int(b['y']*image.height),int((b['x']+b['width'])*image.width),int((b['y']+b['height'])*image.height))
                draw.rectangle(xy,outline='#20a879' if item['method']=='omr' else '#416ce5',width=3)
                draw.text((xy[0]+3,xy[1]+3),numbers.get(item['question_id'],'?'),fill='#173d80',font=font)
    return png(image)


@serialized_document_io
def page_image(assessment_id, document_id, page):
    with transaction() as db:
        row = db.execute('SELECT * FROM assessment_documents WHERE id=? AND assessment_id=?',
                         (document_id, assessment_id)).fetchone()
    if not row:
        raise HTTPException(404, '资料不存在')
    path = Path(row['source_path']).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()) or not path.is_file():
        raise HTTPException(404, '资料不可访问')
    if row['media_type'] in WORD_MEDIA:
        try:
            path = word_to_pdf(path, get_settings().data_dir)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if row['media_type'] == 'application/pdf' or row['media_type'] in WORD_MEDIA:
        with pdfium.PdfDocument(path) as pdf:
            if not 1 <= page <= len(pdf):
                raise HTTPException(400, '页码不存在')
            p = pdf[page - 1]
            bitmap = p.render(scale=1.5)
            try:
                return bitmap.to_pil().convert('RGB').copy()
            finally:
                bitmap.close()
                p.close()
    if row['media_type'].startswith('image/') and page == 1:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert('RGB')
    raise HTTPException(400, '模板预览支持图片和 PDF；请检查页码')


@router.get('/api/runtime/document-renderer')
def document_renderer_status():
    status = renderer_status()
    return {key: value for key, value in status.items() if key != 'executable'}


def png(image):
    out = io.BytesIO()
    image.save(out, format='PNG')
    return Response(out.getvalue(), media_type='image/png', headers={'Cache-Control': 'no-store'})


@router.get('/api/assessments/{assessment_id}/template-page')
def template_page(assessment_id: str, document_id: str, page: int = Query(1, ge=1)):
    with page_image(assessment_id, document_id, page) as image:
        return png(image)


@router.post('/api/assessments/{assessment_id}/template-preview')
def template_crop(assessment_id: str, region: RegionInput, document_id: str):
    with page_image(assessment_id, document_id, region.page) as image:
        box = (int(region.x * image.width), int(region.y * image.height),
               int((region.x + region.width) * image.width), int((region.y + region.height) * image.height))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise HTTPException(400, '区域过小')
        return png(image.crop(box))


@router.get('/api/batches/{batch_id}/performance')
def performance(batch_id: str):
    try:
        return summary(batch_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post('/api/grade-items/{grade_id}/training-approval')
def approve_training(grade_id: str, payload: TrainingApproval):
    if not payload.redacted_text.strip() or not payload.approved_by.strip():
        raise HTTPException(422, '需要脱敏文本和批准人')
    with transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        grade = db.execute('SELECT * FROM grade_items WHERE id=? AND superseded_at IS NULL', (grade_id,)).fetchone()
        if not grade:
            raise HTTPException(404, '批改记录不存在或已过期')
        if grade['score'] is None and payload.expected_review_id is None:
            raise HTTPException(400, '无有效分数的记录不能作为训练标签')
        review = db.execute('SELECT * FROM review_events WHERE grade_item_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1', (grade_id,)).fetchone()
        if (review['id'] if review else None) != payload.expected_review_id:
            raise HTTPException(409, '复核记录已变化，请刷新后重新批准')
        db.execute('INSERT INTO training_approvals VALUES (?,?,?,?,?,?)',
                   (uuid4().hex, grade_id, payload.expected_review_id, payload.redacted_text,
                    payload.approved_by, utc_now()))
    return {'approved': True, 'label_source': 'teacher_confirmed' if review else 'machine_pseudo'}


@router.get('/api/assessments/{assessment_id}/training-samples')
def training_samples(assessment_id: str, kind: str = Query('teacher', pattern='^(teacher|pseudo)$')):
    if not get_assessment(assessment_id):
        raise HTTPException(404, '试卷不存在')
    output = []
    with transaction() as db:
        rows = db.execute('''SELECT g.* FROM grade_items g JOIN submissions s ON s.id=g.submission_id
                             WHERE s.assessment_id=? AND g.superseded_at IS NULL''', (assessment_id,)).fetchall()
        for row in rows:
            grade = row_dict(row)
            reviews = db.execute('SELECT id,final_score,created_at FROM review_events WHERE grade_item_id=? ORDER BY created_at,rowid', (grade['id'],)).fetchall()
            latest = reviews[-1]['id'] if reviews else None
            approval = db.execute('''SELECT * FROM training_approvals WHERE grade_item_id=? AND review_id IS ?
                                      ORDER BY created_at DESC,rowid DESC LIMIT 1''', (grade['id'], latest)).fetchone()
            if not approval or bool(latest) != (kind == 'teacher'):
                continue
            snapshot = grade.get('snapshot') or {}
            q, response = snapshot.get('question') or {}, snapshot.get('response') or {}
            output.append({'sample_id': grade['id'], 'question_group': q.get('id'),
                'question': q.get('prompt'), 'answer_key': q.get('answer_key'), 'rubric': q.get('rubric'),
                'max_score': grade['max_score'], 'answer': approval['redacted_text'],
                'extraction_source': response.get('extraction_source'),
                'ocr_confidence': response.get('ocr_confidence'), 'model': grade['grader_version'],
                'rubric_version': hashlib.sha256(json.dumps(
                    {key: q.get(key) for key in ('prompt', 'answer_key', 'rubric', 'rule_config')},
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                'grader_kind': grade['grader_kind'], 'model_score': grade['score'],
                'score': reviews[-1]['final_score'] if reviews else grade['score'],
                'label_source': 'teacher_confirmed' if latest else 'machine_pseudo',
                'review_history': [{'score': r['final_score'], 'created_at': r['created_at']} for r in reviews],
                'grading_version': grade['created_at'], 'approval_date': approval['created_at']})
    body = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in output)
    return Response(body, media_type='application/x-ndjson', headers={
        'Content-Disposition': f'attachment; filename="training-{kind}.jsonl"',
        'X-Sample-Count': str(len(output)), 'Cache-Control': 'no-store'})
