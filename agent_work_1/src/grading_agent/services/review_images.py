"""Read-only student evidence: original pages and padded question crops. No models."""
from functools import lru_cache
from pathlib import Path
import math

from PIL import Image,ImageOps
import pypdfium2 as pdfium

from ..config import get_settings
from ..repository import get_submission,get_assessment
from .document_renderer import WORD_MEDIA,word_to_pdf
from .layout_templates import load_layout


def safe_file(value):
    path=Path(value).resolve()
    if not path.is_relative_to(get_settings().data_dir.resolve()) or not path.is_file():
        raise ValueError('学生答卷图片不可访问')
    return path


def student_page(submission,number):
    saved=next((p for p in submission['pages'] if p['page_no']==number),None)
    if saved:
        with Image.open(safe_file(saved['image_path'])) as image:return ImageOps.exif_transpose(image).convert('RGB')
    source=safe_file(submission['source_path']);media=submission['media_type']
    if media in WORD_MEDIA:source=word_to_pdf(source,get_settings().data_dir)
    if media=='application/pdf' or media in WORD_MEDIA:
        with pdfium.PdfDocument(source) as pdf:
            if not 1<=number<=len(pdf):raise ValueError('答卷页码不存在')
            page=pdf[number-1];bitmap=page.render(scale=1.5)
            try:return bitmap.to_pil().convert('RGB').copy()
            finally:bitmap.close();page.close()
    if media.startswith('image/') and number==1:
        with Image.open(source) as image:return ImageOps.exif_transpose(image).convert('RGB')
    raise ValueError('答卷暂不能转换为图片，请打开原件查看')


def page_list(submission):
    pages=[]
    for page in submission['pages']:
        try:safe_file(page['image_path'])
        except ValueError:continue
        pages.append({'page':page['page_no'],'width':page['width'],'height':page['height']})
    if pages:return sorted(pages,key=lambda p:p['page'])
    source=safe_file(submission['source_path']);media=submission['media_type']
    if media in WORD_MEDIA:source=word_to_pdf(source,get_settings().data_dir)
    if media=='application/pdf' or media in WORD_MEDIA:
        with pdfium.PdfDocument(source) as pdf:
            for index in range(len(pdf)):
                page=pdf[index]
                try:w,h=page.get_size()
                finally:page.close()
                pages.append({'page':index+1,'width':round(w*1.5),'height':round(h*1.5)})
    elif media.startswith('image/'):
        with student_page(submission,1) as image:pages=[{'page':1,'width':image.width,'height':image.height}]
    return pages


def context(submission_id,question_id):
    submission=get_submission(submission_id)
    if not submission:raise LookupError('答卷不存在')
    assessment=get_assessment(submission['assessment_id'])
    question=next((q for q in assessment['questions'] if q['id']==question_id),None)
    if not question:raise LookupError('题目不属于该答卷')
    return submission,assessment,question


@lru_cache(maxsize=24)
def old_alignment(source,source_stamp,reference,reference_stamp):
    # Old reports did not retain homographies. Rebuild local geometry only, never OCR/API.
    from .local_answers import register_page
    with Image.open(source) as student,Image.open(reference) as template:
        _,meta=register_page(template,ImageOps.exif_transpose(student))
    return meta['source_to_template']


def mapped_regions(submission,assessment,question,pages):
    layout=load_layout(assessment)
    if layout and layout.get('stale'):return [],'模板已改变，显示整页供核对，不沿用旧位置高亮。'
    item=next((i for i in (layout or {}).get('items',[]) if i['question_id']==question['id']),None)
    boxes=item['regions'] if item else question.get('rule_config',{}).get('regions') or []
    if not boxes and question.get('rule_config',{}).get('region'):boxes=[question['rule_config']['region']]
    if not boxes:return [],'未配置本题位置，已切换为学生答卷整页。'
    from ..schemas import RegionInput
    try:boxes=[RegionInput.model_validate(box).model_dump() for box in boxes]
    except (ValueError,TypeError):return [],'本题位置配置无效，显示学生答卷整页。'
    report=submission.get('normalization_report',{}).get('local_recognition',{})
    mapped=[];available={p['page']:p for p in pages}
    for index,box in enumerate(boxes):
        number=box['page']
        if number not in available:return [],'本题部分页面缺失，显示可用整页供核对。'
        page=available[number];points=[(box['x'],box['y']),(box['x']+box['width'],box['y']),(box['x']+box['width'],box['y']+box['height']),(box['x'],box['y']+box['height'])]
        if item:
            spec=next((p for p in layout['pages'] if p['page']==number),None)
            if not spec:return [],'缺少模板页信息，显示原页，不推测定位。'
            transform=report.get('pages',{}).get(str(number),{}).get('source_to_template')
            try:
                reference=safe_file(spec['path'])
                with Image.open(reference) as image:rw,rh=image.size
                if transform is None:
                    saved=next((p for p in submission['pages'] if p['page_no']==number),None)
                    if not saved:raise ValueError('无可用对齐页面')
                    source=safe_file(saved['image_path'])
                    transform=old_alignment(str(source),source.stat().st_mtime_ns,str(reference),reference.stat().st_mtime_ns)
                import numpy as np
                inverse=np.linalg.inv(np.asarray(transform,dtype=float))
                converted=[]
                for x,y in points:
                    value=inverse@np.array([x*rw,y*rh,1.])
                    if not np.isfinite(value).all() or abs(value[2])<1e-9:raise ValueError('无效几何变换')
                    converted.append((float(value[0]/value[2]/page['width']),float(value[1]/value[2]/page['height'])))
                points=converted
            except (ValueError,OSError,ImportError,KeyError):return [],'无法可靠映射到学生原页；显示整页，避免错误高亮。'
        if any(not math.isfinite(v) or v<-.005 or v>1.005 for point in points for v in point):
            return [],'本题区域超出当前原页，显示整页供核对。'
        polygon=[[max(0,min(1,x)),max(0,min(1,y))] for x,y in points]
        mapped.append({'index':index,'page':number,'polygon':polygon})
    return mapped,''


def visual_manifest(submission_id,question_id):
    submission,assessment,question=context(submission_id,question_id)
    try:pages=page_list(submission)
    except (ValueError,OSError):pages=[]
    base=f'/api/submissions/{submission_id}'
    for page in pages:page['url']=f'{base}/review-page?page={page["page"]}'
    regions,note=mapped_regions(submission,assessment,question,pages)
    crops=[{'page':r['page'],'url':f'{base}/review-region?question_id={question_id}&region={r["index"]}',
            'label':f'第 {r["page"]} 页 · 作答区域 {index+1}','kind':'original_crop'} for index,r in enumerate(regions)]
    if not crops:
        parts=submission.get('normalization_report',{}).get('local_recognition',{}).get('recognitions',{}).get(question_id,{}).get('parts',[])
        for part_index,part in enumerate(parts):
            for index,name in enumerate(part.get('crops') or [part.get('crop','')]):
                try:
                    if not name:continue
                    safe_file(get_settings().crop_dir/name)
                except ValueError:continue
                crops.append({'page':None,'url':f'{base}/local-recognition-crop?question_id={question_id}&part={part_index}&region={index}',
                              'label':f'识别时裁切 {len(crops)+1} · 可切整页检查遗漏','kind':'recognition_crop'})
    return {'question_id':question_id,'question_no':question['question_no'],'pages':pages,'regions':regions,'crops':crops,
            'default_mode':'crop' if regions or (crops and not pages) else 'page','default_page':regions[0]['page'] if regions else pages[0]['page'] if pages else None,
            'note':note,'read_only':True,'model_requests':0}


def region_image(submission_id,question_id,index):
    submission,assessment,question=context(submission_id,question_id)
    regions,_=mapped_regions(submission,assessment,question,page_list(submission))
    region=next((r for r in regions if r['index']==index),None)
    if not region:raise LookupError('本题裁切不可用，请切换整页')
    with student_page(submission,region['page']) as image:
        xs=[p[0]*image.width for p in region['polygon']];ys=[p[1]*image.height for p in region['polygon']]
        px=max(8,image.width*.008);py=max(8,image.height*.008)
        box=(max(0,math.floor(min(xs)-px)),max(0,math.floor(min(ys)-py)),min(image.width,math.ceil(max(xs)+px)),min(image.height,math.ceil(max(ys)+py)))
        if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('作答区域无效')
        return image.crop(box)
