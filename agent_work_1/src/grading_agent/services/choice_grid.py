"""Conservative local refinement of boxed, printed choice grids."""
import re
from pathlib import Path
from uuid import uuid4


def locate_printed_grid(pages,questions,settings,recognize_number=None):
    try:
        import cv2
        import numpy as np
        from PIL import Image,ImageOps
        from .ingestion import OCRAdapter
        if settings.ocr_provider!='paddleocr' and recognize_number is None:return None
        lookup={}
        for q in questions:
            match=re.search(r'-(\d+)$',q['question_no'])
            if not match or match[1] in lookup:return None
            lookup[match[1]]=q
        if not lookup:return None
        adapter=OCRAdapter(settings.ocr_provider,settings.ocr_device)
        for spec in pages:
            gray=cv2.imread(str(spec['path']),cv2.IMREAD_GRAYSCALE)
            if gray is None:continue
            mask=(gray<170).astype('uint8')*255
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            rectangles=[]
            for contour in contours:
                x,y,w,h=cv2.boundingRect(contour)
                if w>gray.shape[1]*.25 and 5<w/h<25 and gray.shape[0]*.015<h<gray.shape[0]*.1:
                    rectangles.append((x,y,w,h))
            for x,y,w,h in sorted(rectangles,key=lambda b:b[2]*b[3],reverse=True):
                interior=mask[y+4:y+h-4,x+4:x+w-4]>0
                active=np.where(interior.sum(1)>w*.015)[0]
                if not len(active):continue
                bands=np.split(active,np.where(np.diff(active)>5)[0]+1)
                cells=[]
                for band in bands:
                    if len(band)<6:continue
                    top,bottom=int(band[0]),int(band[-1]+1)
                    cols=np.where(interior[top:bottom].sum(0)>0)[0]
                    groups=np.split(cols,np.where(np.diff(cols)>max(20,(bottom-top)*1.4))[0]+1)
                    for group in groups:
                        if len(group)<3:continue
                        start,end=int(group[0]),int(group[-1]+1)
                        columns=np.where(interior[top:bottom,start:end].sum(0)>0)[0]
                        parts=np.split(columns,np.where(np.diff(columns)>max(8,(bottom-top)*.55))[0]+1)
                        if len(parts)!=5:cells=[];break
                        cells.append((top,bottom,start,[(int(a[0]),int(a[-1]+1)) for a in parts]))
                    if not cells:break
                if len(cells)!=len(questions):continue
                items=[];seen=set()
                for top,bottom,start,parts in cells:
                    left,right=parts[0]
                    digit=gray[y+4+top:y+4+bottom,x+4+start:x+4+start+parts[-1][1]]
                    image=Image.fromarray(digit).resize((max(24,digit.shape[1]*5),digit.shape[0]*5))
                    image=ImageOps.expand(image,border=20,fill='white')
                    if recognize_number:text,confidence=recognize_number(image)
                    else:
                        path=settings.crop_dir/(uuid4().hex+'.png');image.save(path)
                        from .ocr_runtime import recognize
                        text,confidence,_=recognize(path,settings.ocr_device,leading_number=True)
                    match=re.match(r'\s*(\d{1,2})(?!\d)',text)
                    number=match[1] if match else ''
                    if recognize_number is None and (number not in lookup or number in seen or confidence is None or confidence<.8):
                        left,right=parts[0]
                        digit=gray[y+4+top:y+4+bottom,x+4+start+left:x+4+start+right]
                        image=ImageOps.expand(Image.fromarray(digit).resize((max(32,digit.shape[1]*8),digit.shape[0]*8)),border=30,fill='white')
                        path=settings.crop_dir/(uuid4().hex+'.png');image.save(path)
                        text,confidence,_=recognize(path,settings.ocr_device,leading_number=True)
                        match=re.match(r'\s*(\d{1,2})(?!\d)',text);number=match[1] if match else ''
                    if number not in lookup or number in seen or confidence is None or confidence<.8:
                        items=[];break
                    seen.add(number);q=lookup[number];options=[]
                    for label,(left,right) in zip('ABCD',parts[1:]):
                        bx=x+4+start+left-2;by=y+4+top-2
                        bw=right-left+4;bh=bottom-top+4
                        options.append({'label':label,'box':dict(page=spec['page'],x=bx/gray.shape[1],y=by/gray.shape[0],width=bw/gray.shape[1],height=bh/gray.shape[0])})
                    first=options[0]['box'];last=options[-1]['box']
                    region={**first,'width':last['x']+last['width']-first['x']}
                    items.append(dict(question_id=q['id'],section='选择题',number=number,method='omr',confidence=.95,regions=[region],options=options))
                if len(items)==len(questions):return items
    except (ImportError,ValueError,OSError,KeyError):
        return None
    return None
