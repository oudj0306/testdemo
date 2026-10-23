"""Local page registration, reference-subtracted OMR and region OCR.

Requires matching printed layouts. Low-quality alignment never becomes a blank.
Optional local visual transcription uses a loopback-only service; never remote vision.
"""
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .ingestion import OCRAdapter
from .telemetry import timed, event


def dependencies():
    import importlib.util
    return {'opencv': importlib.util.find_spec('cv2') is not None,
            'numpy': importlib.util.find_spec('numpy') is not None}


def register_page(reference, student):
    import cv2
    import numpy as np
    ref = np.asarray(reference.convert('RGB'))
    src = np.asarray(student.convert('RGB'))
    if ref.shape == src.shape and np.array_equal(ref, src):
        return ref.copy(), {'method':'identical', 'inliers':0}
    h, w = ref.shape[:2]
    scale = min(1., 1400 / max(h, w))
    ratio = min(1., 1400 / max(src.shape[:2]))
    r = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), None, fx=scale, fy=scale)
    s = cv2.resize(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), None, fx=ratio, fy=ratio)
    orb = cv2.ORB_create(nfeatures=6000)
    kr, dr = orb.detectAndCompute(r, None)
    ks, ds = orb.detectAndCompute(s, None)
    if dr is None or ds is None or min(len(kr), len(ks)) < 16:
        raise ValueError('版面特征不足，无法可靠对齐')
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, dr, k=2)
    good = [a for pair in matches if len(pair) == 2 for a,b in [pair] if a.distance < .7*b.distance]
    if len(good) < 16:
        raise ValueError('答卷与模板匹配点不足，请检查版式或页码')
    start = np.float32([ks[m.queryIdx].pt for m in good])
    end = np.float32([kr[m.trainIdx].pt for m in good])
    homography, mask = cv2.findHomography(start, end, cv2.RANSAC, 3.)
    if homography is None or mask is None:
        raise ValueError('答卷对齐失败')
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 16 or inliers.mean() < .5:
        raise ValueError('答卷对齐不可靠')
    points = end[inliers]
    spread = np.ptp(points, axis=0) / np.array([r.shape[1],r.shape[0]])
    hull_area = cv2.contourArea(cv2.convexHull(points)) / (r.shape[0]*r.shape[1])
    if min(spread) < .35 or hull_area < .12:
        raise ValueError('对齐特征集中于局部，不能推断整页位置')
    transform = np.diag([1/scale,1/scale,1]) @ homography @ np.diag([ratio,ratio,1])
    corners = np.float32([[[0,0],[src.shape[1],0],[src.shape[1],src.shape[0]],[0,src.shape[0]]]])
    mapped = cv2.perspectiveTransform(corners, transform)[0]
    if not np.isfinite(mapped).all() or not cv2.isContourConvex(mapped):
        raise ValueError('对齐后页面几何形状异常')
    area = abs(cv2.contourArea(mapped)) / (w*h)
    if not .65 <= area <= 1.5:
        raise ValueError('答卷裁切或透视过大，无法可靠对齐')
    warped = cv2.warpPerspective(src, transform, (w,h), borderValue=(255,255,255))
    coverage = cv2.warpPerspective(np.ones(src.shape[:2],np.uint8), transform, (w,h))
    return warped, {'method':'orb_homography', 'inliers':int(inliers.sum()),
                    'coverage': coverage, 'matched_area':float(hull_area)}


def pixel_box(b, shape):
    h,w = shape[:2]
    x1,y1 = int(b['x']*w),int(b['y']*h)
    x2,y2 = int((b['x']+b['width'])*w),int((b['y']+b['height'])*h)
    if x2-x1 < 8 or y2-y1 < 8 or min(x1,y1)<0 or x2>w or y2>h:
        raise ValueError('作答区域过小或超出页面')
    return x1,y1,x2,y2


def covered(b, aligned, meta):
    x1,y1,x2,y2 = pixel_box(b, aligned.shape)
    if 'coverage' in meta and meta['coverage'][y1:y2,x1:x2].mean() < .995:
        raise ValueError('作答区部分位于照片之外，不能当成空白')
    return x1,y1,x2,y2


def read_marks(item, aligned_pages, references, metadata, multiple=False, prepared=None, reference_mode="blank_difference"):
    import cv2
    import numpy as np
    selected, measurements, uncertain = [], {}, False
    geometry_ratios = []
    prepared = prepared if prepared is not None else {}
    for option in item['options']:
        b = option['box']
        page = b['page']
        image = aligned_pages[page]
        x1,y1,x2,y2 = covered(b, image, metadata[page])
        if page not in prepared:
            ref_gray = cv2.cvtColor(np.asarray(references[page].convert('RGB')),cv2.COLOR_RGB2GRAY)
            answer_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            prepared[page] = (ref_gray,answer_gray,min(190,max(120,float(np.percentile(answer_gray,95))-45)))
        ref_gray,answer_gray,threshold = prepared[page]
        r = ref_gray[y1:y2,x1:x2]
        a = answer_gray[y1:y2,x1:x2]
        if reference_mode == 'geometry':
            # Filled samples supply positions only. Never subtract their answers.
            hh,ww = a.shape
            yy,xx = np.ogrid[:hh,:ww]
            interior = ((xx-(ww-1)/2)/max(ww*.32,1))**2 + ((yy-(hh-1)/2)/max(hh*.32,1))**2 <= 1
            if interior.sum()<15:
                raise ValueError('选项区域分辨率不足，无法可靠检测填涂')
            fill=float(((a<threshold)&interior).sum()/interior.sum())
            measurements[option['label']]=round(fill,4)
            geometry_ratios.append(fill)
            if fill>=.65:selected.append(option['label'])
            elif fill>.45:uncertain=True
            continue
        # Compare added ink only, excluding the printed outline and local lighting.
        rmask = (r < min(190, float(np.percentile(r,95))-45)).astype('uint8')
        amask = (a < threshold).astype('uint8')
        hh,ww = r.shape
        contours,_ = cv2.findContours(rmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        outlines=[]
        for contour in contours:
            bx,by,bw,bh=cv2.boundingRect(contour)
            if (bw>=ww*.65 and bh>=hh*.65 and abs(bx+bw/2-ww/2)<ww*.2
                    and abs(by+bh/2-hh/2)<hh*.2):
                outlines.append(contour)
        if not outlines or not .02<float(rmask.mean())<.5:
            raise ValueError('模板选项框未包含明确气泡或方框轮廓，需重新定位')
        yy,xx = np.ogrid[:hh,:ww]
        interior = ((xx-(ww-1)/2)/(ww*.34))**2 + ((yy-(hh-1)/2)/(hh*.34))**2 <= 1
        excluded = cv2.dilate(rmask,np.ones((3,3),np.uint8))>0
        available = interior & ~excluded
        if available.sum() < .35*interior.sum() or interior.sum()<15:
            raise ValueError('选项区域没有足够空白内部，模板可能框住文字或已填答案')
        fill = float((amask.astype(bool) & available).sum()/available.sum())
        measurements[option['label']] = round(fill,4)
        if fill >= .3:
            selected.append(option['label'])
        elif fill > .06:
            uncertain = True
    if reference_mode=='geometry':
        # Without a blank reference, faint/ambiguous marks cannot establish blankness.
        if not selected:uncertain=True  # A filled reference cannot establish blankness safely.
        if selected and any(.12<r<.65 for r in geometry_ratios):
            clear=[r for r in geometry_ratios if r>=.65]
            other=[r for r in geometry_ratios if r<.65]
            if other and min(clear)-max(other)<.25:uncertain=True
    if uncertain or (not multiple and len(selected)>1):
        return '', None, False, measurements
    return ''.join(sorted(selected)), .97, not selected, measurements


def recognize_answers(layout, pages, questions, settings):
    """Returns candidate answers plus diagnostics; caller preserves human corrections."""
    output, warnings, diagnostics = [], [], {'version':layout['version'], 'template_id':layout.get('template_id'), 'pages':{}, 'marks':{}}
    if layout.get('stale'):
        return [], ['题目或模板来源已改变，请重新生成模板；本地模式不调用视觉 API'], diagnostics
    if not all(dependencies().values()):
        return [], ['当前 Python 缺少 OpenCV / NumPy，请使用优化版 GPU 环境或安装 local_recognition 依赖'], diagnostics
    try:
        import cv2
        import numpy
    except (ImportError,OSError):
        return [], ['本地 OpenCV / NumPy 无法载入，请检查当前 Python 依赖'], diagnostics
    page_map = {p.page_no:p for p in pages}
    aligned, references, metadata = {}, {}, {}
    for spec in layout['pages']:
        number = spec['page']
        if number not in page_map:
            warnings.append(f'缺少第 {number} 页')
            continue
        try:
            path = Path(spec['path']).resolve()
            if not path.is_relative_to(settings.data_dir.resolve()):
                raise ValueError('模板页面不在数据目录内')
            with Image.open(path) as image:
                references[number] = image.convert('RGB').copy()
            with Image.open(page_map[number].path) as image, timed('local_alignment', page=number):
                aligned[number], metadata[number] = register_page(references[number],image)
            diagnostics['pages'][str(number)] = {k:v for k,v in metadata[number].items() if k!='coverage'}
        except (ValueError, OSError, cv2.error) as exc:
            warnings.append(f'第 {number} 页：{exc}')
    by_id = {q['id']:q for q in questions}
    adapter = OCRAdapter(settings.ocr_provider, settings.ocr_device)
    vision_error=None
    if settings.local_vision_enabled and any(i['method']=='text' and i['question_id'] in by_id for i in layout['items']):
        from .local_vision import runtime
        try:diagnostics['visual_runtime']=runtime(settings)
        except ValueError as exc:vision_error=str(exc)
    diagnostics['recognitions']={}
    prepared_marks = {}
    for item in layout['items']:
        qid = item['question_id']
        if qid not in by_id:
            continue
        q = by_id[qid]
        try:
            if item['confidence'] < .9:
                raise ValueError('模板位置置信度不足')
            needed = {b['page'] for b in item['regions']} | {o['box']['page'] for o in item['options']}
            if not needed.issubset(aligned):
                raise ValueError('所需页面未可靠对齐')
            if item['method']=='omr':
                with timed('local_omr'):
                    text,confidence,blank,marks = read_marks(item, aligned, references, metadata,
                                                            q['question_type']=='multiple_choice', prepared=prepared_marks,
                                                            reference_mode=layout.get('omr_reference_mode','geometry' if (layout.get('source_review') or {}).get('answer_content')=='filled' else 'blank_difference'))
                diagnostics['marks'][qid] = marks
                source = 'local_layout_omr'
            else:
                if settings.local_vision_enabled and vision_error:raise ValueError(vision_error)
                texts, scores, blanks = [], [], []
                parts=[]
                joint_crops=[]
                joint_references=[]
                blank_reference=(layout.get('source_review') or {}).get('answer_content')=='blank'
                from .blank_detection import blank_pixels,verify_blank_part
                from .local_vision import kind_for
                joint=settings.local_vision_enabled and len(item['regions'])>1 and kind_for(q) in {'code','flowchart'}
                for b in item['regions']:
                    image = aligned[b['page']]
                    x1,y1,x2,y2 = covered(b,image,metadata[b['page']])
                    crop = settings.crop_dir / (uuid4().hex+'.png')
                    Image.fromarray(image[y1:y2,x1:x2]).save(crop)
                    reference=references[b['page']].crop((x1,y1,x2,y2)) if blank_reference else None
                    if joint:
                        joint_crops.append(crop);joint_references.append(reference);continue
                    if settings.local_vision_enabled:
                        from .local_vision import transcribe
                        part={'crop':crop.name,'text':'','consistent':False,'issues':[],
                              'model':{'model':settings.local_vision_model}}
                        parts.append(part)
                        diagnostics['recognitions'][qid]={'parts':parts,'question_no':q['question_no']}
                        try:
                            with timed('local_visual_transcription'):
                                value=transcribe(crop,q,settings)
                            part.update(value)
                            verify_blank_part(part,[crop],[reference] if blank_reference else None)
                        except ValueError as exc:
                            part['issues']=[str(exc)]
                        # Keep all region candidates available for inspection, even if one failed verification.
                        texts.append(part['text']);blanks.append(bool(part.get('blank_confirmed')));continue
                    with timed('local_layout_ocr'):
                        t,c,_ = adapter.extract(crop)
                    if not t.strip():
                        with Image.open(crop) as pixels:evidence=blank_pixels(pixels,reference)
                        if evidence['confirmed']:
                            texts.append('');blanks.append(True)
                            parts.append({'crop':crop.name,'text':'','consistent':True,'issues':[],
                                          'blank_confirmed':True,'blank_evidence':[evidence]})
                            diagnostics['recognitions'][qid]={'parts':parts,'question_no':q['question_no']}
                            continue
                    if not t.strip() or c is None or not math.isfinite(float(c)) or not 0 <= float(c) <= 1:
                        raise ValueError('OCR 未可靠提取完整作答，不能认定为空白')
                    texts.append(t)
                    scores.append(float(c))
                    blanks.append(False)
                if joint:
                    from .local_vision import transcribe
                    part={'crop':joint_crops[0].name,'crops':[p.name for p in joint_crops],
                          'text':'','consistent':False,'issues':[],'model':{'model':settings.local_vision_model}}
                    parts.append(part);diagnostics['recognitions'][qid]={'parts':parts,'question_no':q['question_no']}
                    try:
                        with timed('local_visual_transcription'):
                            part.update(transcribe(joint_crops,q,settings))
                            verify_blank_part(part,joint_crops,joint_references if blank_reference else None)
                    except ValueError as exc:part['issues']=[str(exc)]
                    texts.append(part['text'])
                    blanks.append(bool(part.get('blank_confirmed')))
                if settings.local_vision_enabled:
                    issues=[issue for part in parts for issue in part['issues']]
                    if any(part.get('consistent') is not True for part in parts) and not issues:
                        issues.append('本地区域未通过转写核验')
                    if issues:raise ValueError('；'.join(dict.fromkeys(issues)))
                    blank=bool(blanks) and all(blanks)
                    text,confidence,source=('' if blank else '\n'.join('[未作答]' if empty else t for t,empty in zip(texts,blanks))),None,('local_layout_blank' if blank else 'local_vlm_consistent')
                else:
                    blank=bool(blanks) and all(blanks)
                    text,confidence,source = ('' if blank else '\n'.join('[未作答]' if empty else t for t,empty in zip(texts,blanks))),(min(scores) if scores else None),('local_layout_blank' if blank else 'local_layout_ocr')
                if q['scoring_mode']=='choice_set' and not blank:
                    normalized = re_choice(text)
                    if normalized is None or (q['question_type']=='single_choice' and len(normalized)!=1):
                        raise ValueError('未识别到明确的选项字母')
                    text = normalized
            if source not in {'local_vlm_consistent','local_layout_blank'} and (confidence is None or confidence < max(.9, settings.review_confidence_threshold)):
                raise ValueError('识别不确定：浅涂、擦改、多涂或 OCR 置信度不足')
            output.append({'question_id':qid,'raw_text':text,'confidence':confidence,
                           'blank_confirmed':blank,'source':source})
            event('local_answer', details={'source':source})
        except (ValueError, KeyError, cv2.error) as exc:
            warnings.append(f"{q['question_no']}：{exc}")
    parts=[p for item in diagnostics['recognitions'].values() for p in item['parts']]
    requests={r['id']:r for p in parts for r in p.get('requests',[])}
    diagnostics['local_model_usage']={'requests':len(requests),'cache_hits':sum(bool(p.get('cache_hit')) for p in parts),
        'input_tokens':sum(r.get('input_tokens') or 0 for r in requests.values()) if all(r.get('input_tokens') is not None for r in requests.values()) else None,
        'output_tokens':sum(r.get('output_tokens') or 0 for r in requests.values()) if all(r.get('output_tokens') is not None for r in requests.values()) else None,
        'remote_vision_requests':0,'seconds':round(sum(r.get('seconds',0) for r in requests.values()),3)}
    return output,warnings,diagnostics


def re_choice(text):
    import re
    candidate = re.sub(r'[\s,，、;；]+','',text).upper()
    return ''.join(sorted(set(candidate))) if re.fullmatch('[A-H]{1,8}',candidate) else None
