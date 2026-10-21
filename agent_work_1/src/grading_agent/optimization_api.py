"""Template preview, measured performance, explicitly redacted sample exports."""
import io
import json
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image, ImageOps
import pypdfium2 as pdfium

from .config import get_settings
from .db import transaction, utc_now, decode_json
from .repository import get_assessment, row_dict
from .schemas import RegionInput, TrainingApproval
from .services.ingestion import serialized_document_io
from .services.telemetry import summary

router = APIRouter()


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
    if row['media_type'] == 'application/pdf':
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
