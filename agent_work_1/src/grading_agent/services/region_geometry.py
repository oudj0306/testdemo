"""Conservative printed-line calibration. No OCR, model calls or answer reading."""
import copy


def padded_frame(frame, page_width, page_height):
    """Include the printed border and a small physical margin for edge strokes."""
    x,y,w,h=frame
    margin=max(3,round(min(page_width,page_height)*.003))
    left,top=max(0,x-margin),max(0,y-margin)
    right,bottom=min(page_width,x+w+margin),min(page_height,y+h+margin)
    return dict(x=left/page_width,y=top/page_height,width=(right-left)/page_width,height=(bottom-top)/page_height)


def calibrate_regions(items, pages, protected=()):
    import cv2
    from PIL import Image
    import numpy as np

    result = copy.deepcopy(items)
    changed = set()
    for page in pages:
        with Image.open(page['path']) as image:
            gray = np.asarray(image.convert('L'))
        height, width = gray.shape
        binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, width//15), 1)))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, height//25))))
        def contours(mask):
            return [cv2.boundingRect(c) for c in cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]]
        frames = []
        for x,y,w,h in contours(horizontal | vertical):
            if w < width*.18 or h < height*.035 or w > width*.9 or h > height*.9:
                continue
            # A contour must have printed support on all four edges.
            if min(np.mean(horizontal[max(0,y-2):y+4,x:x+w]>0),
                   np.mean(horizontal[y+h-4:y+h+2,x:x+w]>0),
                   np.mean(vertical[y:y+h,max(0,x-2):x+4]>0),
                   np.mean(vertical[y:y+h,x+w-4:x+w+2]>0)) < .12:
                continue
            if not any(abs(x-a)+abs(y-b)+abs(w-c)+abs(h-d)<15 for a,b,c,d in frames):
                frames.append((x,y,w,h))
        lines = contours(horizontal)
        groups = {}
        for item in result:
            if item['method'] != 'text' or item['question_id'] in protected:
                continue
            for index,b in enumerate(item['regions']):
                if b['page'] != page['page']:
                    continue
                cx=(b['x']+b['width']/2)*width; cy=(b['y']+b['height']/2)*height
                matches=[f for f in frames if f[0]<cx<f[0]+f[2] and f[1]<cy<f[1]+f[3]]
                if not matches:
                    continue
                frame=min(matches,key=lambda f:f[2]*f[3])
                short_row=(b['height']<=.035 or (b['height']*height<frame[3]*.3
                           and b['width']*width>frame[2]*.25 and frame[3]<height*.4))
                if short_row:
                    groups.setdefault(frame,[]).append((item,index,b))
                else:
                    x,y,w,h=frame
                    # Only expand a plausible coarse candidate within its printed panel.
                    overlap=max(0,min((b['x']+b['width'])*width,x+w)-max(b['x']*width,x))*max(0,min((b['y']+b['height'])*height,y+h)-max(b['y']*height,y))
                    if overlap/(b['width']*width*b['height']*height)<.65:
                        continue
                    item['regions'][index]=dict(page=b['page'],**padded_frame(frame,width,height))
                    changed.add(item['question_id'])
        for frame, candidates in groups.items():
            x,y,w,h=frame
            underlines=[l for l in lines if x+8<=l[0] and l[0]+l[2]<=x+w-8 and y+8<l[1]<y+h-8
                        and w*.25<l[2]<w*.97 and l[3]<8]
            # Require a complete repeated grid; partial or noisy evidence stays unchanged.
            if len(candidates)<3 or len(underlines)!=len(candidates):
                continue
            rows=sorted({round(l[1]/8)*8 for l in underlines})
            gaps=np.diff(rows)
            if len(gaps)<2 or min(gaps)<15 or max(gaps)>min(gaps)*1.3:
                continue
            row_height=float(np.median(gaps))
            ordered=sorted(candidates,key=lambda c:(round((c[2]['y']+c[2]['height']/2)*height/20),c[2]['x']))
            underlines.sort(key=lambda l:(round(l[1]/8),l[0]))
            if any(abs((b['x']+b['width']/2)*width-(l[0]+l[2]/2))>w*.2 or
                   abs((b['y']+b['height']/2)*height-l[1])>row_height*1.5
                   for (_,_,b),l in zip(ordered,underlines)):
                continue
            for (item,index,b),(lx,ly,lw,lh) in zip(ordered,underlines):
                # Include the complete writing row, with overlap only at row edges.
                # Tight underline crops cut ascenders, descenders and strokes at line ends.
                margin=max(3,round(min(width,height)*.003))
                top=max(0,y-margin,ly-row_height-margin/2)
                bottom=min(height,ly+lh+margin)
                # Printed subpart labels shorten some underlines. Use the whole
                # column's writing extent so strokes to the left are not clipped.
                column=[l for l in underlines if (l[0]<x+w/2)==(lx<x+w/2)]
                left=max(0,min(l[0] for l in column)-margin)
                right=min(width,max(l[0]+l[2] for l in column)+margin)
                item['regions'][index]=dict(page=b['page'],x=left/width,y=top/height,
                                           width=(right-left)/width,height=(bottom-top)/height)
                changed.add(item['question_id'])
    return result, changed


def region_key(box):
    import hashlib, json
    return hashlib.sha256(json.dumps({k:box[k] for k in ('page','x','y','width','height')},sort_keys=True).encode()).hexdigest()
