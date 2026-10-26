/* Read-only student evidence; keep image navigation independent of editable grading fields. */
(function(){
 const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const clampZoom=value=>Math.max(.5,Math.min(4,value));
 function pageFocus(regions,page){
  const points=regions.filter(r=>r.page===page).flatMap(r=>r.polygon);
  if(!points.length)return null;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  return {x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys)};
 }
 window.reviewEvidenceGeometry={pageFocus,clampZoom};
 window.createReviewEvidence=function(submissionId,root,options={}){
  let disposed=false,version=0,abort=null,manifest=null,zoom=1,mode='crop',preferredMode='crop',page=null,drag=null;
  root.innerHTML='<div class="evidence-heading"><div><span class="section-kicker">STUDENT ORIGINAL</span><h3>学生答卷原图</h3></div><span class="evidence-question badge neutral"></span></div><div class="evidence-mode" role="group" aria-label="原图查看方式"><button type="button" class="secondary" data-mode="crop">本题区域</button><button type="button" class="secondary" data-mode="page">所在整页</button></div><div class="evidence-toolbar"><label class="evidence-page-label">页码<select aria-label="学生答卷页码"></select></label><div class="evidence-zoom"><button type="button" class="secondary" data-zoom="out" aria-label="缩小原图">−</button><span class="evidence-percent">100%</span><button type="button" class="secondary" data-zoom="in" aria-label="放大原图">＋</button><button type="button" class="text-button" data-zoom="fit">适应宽度</button></div><button type="button" class="text-button evidence-locate">定位本题</button></div><p class="evidence-status" role="status" aria-live="polite"></p><div class="evidence-viewport" tabindex="0" aria-label="学生答卷图片，可滚动或拖动查看"></div><div class="evidence-footer"><span>拖动查看 · 原始笔迹保留</span><button type="button" class="text-button evidence-original">打开答卷原件 ↗</button></div>';
  const viewport=root.querySelector('.evidence-viewport'),status=root.querySelector('.evidence-status'),pageSelect=root.querySelector('select');
  root.querySelector('.evidence-original').onclick=()=>options.onOpenOriginal?.();
  const buttons=[...root.querySelectorAll('[data-mode]')];
  function valid(token){return !disposed&&root.isConnected&&version===token;}
  function scale(){
   const width=Math.max(80,viewport.clientWidth-32)*zoom;
   root.querySelectorAll('.review-image-frame').forEach(frame=>frame.style.width=width+'px');
   root.querySelector('.evidence-percent').textContent=Math.round(zoom*100)+'%';
   root.querySelector('[data-zoom="out"]').disabled=zoom<=.5;
   root.querySelector('[data-zoom="in"]').disabled=zoom>=4;
  }
  function locate(){
   if(!manifest)return;
   if(mode!=='page'){viewport.scrollTo({top:0,left:0});return;}
   const frame=viewport.querySelector('.review-image-frame'),focus=pageFocus(manifest.regions,page);
   if(!frame||!focus){viewport.scrollTo({top:0,left:0});return;}
   const top=frame.offsetTop+(focus.y*frame.clientHeight)-Math.max(20,(viewport.clientHeight-Math.min(focus.height*frame.clientHeight,viewport.clientHeight))/2);
   const left=frame.offsetLeft+(focus.x+focus.width/2)*frame.clientWidth-viewport.clientWidth/2;
   viewport.scrollTo({top:Math.max(0,top),left:Math.max(0,left)});
  }
  function render(){
   if(!manifest)return;
   root.dataset.viewMode=mode;
   const token=version;
   buttons.forEach(b=>{b.classList.toggle('selected',b.dataset.mode===mode);b.setAttribute('aria-pressed',String(b.dataset.mode===mode));b.disabled=b.dataset.mode==='crop'?!manifest.crops.length:!manifest.pages.length;});
   pageSelect.closest('label').hidden=mode!=='page';
   pageSelect.innerHTML=manifest.pages.map(p=>'<option value="'+p.page+'" '+(p.page===page?'selected':'')+'>第 '+p.page+' 页</option>').join('');
   const entries=mode==='crop'?manifest.crops:manifest.pages.filter(p=>p.page===page);
   root.querySelector('.evidence-locate').disabled=mode!=='page'||!pageFocus(manifest.regions,page);
   status.textContent=mode==='crop'?(entries.length>1?'本题有 '+entries.length+' 处作答区域，已按顺序完整展示。':'本题作答区保留少量边缘，可切整页核对是否遗漏。'):(pageFocus(manifest.regions,page)?'蓝框对应当前题目；可拖动查看邻题及续答。':manifest.note||'本页暂无可靠定位，整页显示供核对。');
   viewport.innerHTML=entries.length?entries.map((entry,index)=>{
    const outlines=mode==='page'?manifest.regions.filter(r=>r.page===page).map(r=>'<polygon points="'+r.polygon.map(p=>p.map(v=>v*1000).join(',')).join(' ')+'"/>').join(''):'';
    return '<figure class="review-image-card"><figcaption>'+escape(mode==='crop'?entry.label:'学生答卷 · 第 '+page+' 页')+(mode==='crop'&&entry.page?'<button type="button" class="text-button" data-source-page="'+entry.page+'">查看所在页 ↗</button>':'')+'</figcaption><div class="review-image-frame"><img draggable="false" alt="'+escape('第 '+manifest.question_no+' 题 · '+(mode==='crop'?'作答原图 '+(index+1):'学生答卷整页'))+'" src="'+escape(entry.url)+'">'+(outlines?'<svg class="evidence-highlight" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">'+outlines+'</svg>':'')+'</div></figure>';
   }).join(''):'<div class="evidence-empty">暂不能显示图片，请打开答卷原件查看。此处不会调用识别模型。</div>';
   scale();viewport.scrollTo({top:0,left:0});
   viewport.querySelectorAll('img').forEach(img=>{
    img.onload=()=>{if(valid(token)&&viewport.contains(img)){scale();if(mode==='page')locate();}};
    img.onerror=()=>{
     if(!valid(token)||!viewport.contains(img))return;
     if(mode==='crop'&&manifest.pages.length){mode='page';page=manifest.default_page;render();status.textContent='裁切暂不可用，已自动切换学生答卷整页。';}
     else{img.closest('.review-image-frame').textContent='图片加载失败，可打开答卷原件查看。';status.textContent='原图暂不可用，不影响已保存的校对内容。';}
    };
    if(img.complete&&img.naturalWidth)img.onload();
   });
  }
  buttons.forEach(button=>button.onclick=()=>{preferredMode=button.dataset.mode;mode=preferredMode;zoom=1;render();});
  pageSelect.onchange=()=>{page=Number(pageSelect.value);zoom=1;render();};
  root.querySelector('.evidence-locate').onclick=locate;
  root.querySelectorAll('[data-zoom]').forEach(button=>button.onclick=()=>{
   const previous=zoom,x=viewport.scrollLeft+viewport.clientWidth/2,y=viewport.scrollTop+viewport.clientHeight/2;
   zoom=clampZoom(button.dataset.zoom==='fit'?1:zoom+(button.dataset.zoom==='in'?.25:-.25));scale();
   if(button.dataset.zoom==='fit')locate();else viewport.scrollTo({left:x*zoom/previous-viewport.clientWidth/2,top:y*zoom/previous-viewport.clientHeight/2});
  });
  viewport.addEventListener('click',event=>{const b=event.target.closest('[data-source-page]');if(b){preferredMode=mode='page';page=Number(b.dataset.sourcePage);zoom=1;render();}});
  viewport.addEventListener('pointerdown',event=>{
   if(event.button!==0||event.pointerType==='touch'||!event.target.closest('.review-image-frame'))return;
   drag={id:event.pointerId,x:event.clientX,y:event.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};
   viewport.setPointerCapture(event.pointerId);viewport.classList.add('dragging');event.preventDefault();
  });
  viewport.addEventListener('pointermove',event=>{if(drag&&drag.id===event.pointerId){viewport.scrollLeft=drag.left-(event.clientX-drag.x);viewport.scrollTop=drag.top-(event.clientY-drag.y);}});
  function endDrag(){drag=null;viewport.classList.remove('dragging');}
  viewport.addEventListener('pointerup',endDrag);viewport.addEventListener('pointercancel',endDrag);viewport.addEventListener('lostpointercapture',endDrag);
  const resized=()=>scale(),observer=window.ResizeObserver?new ResizeObserver(resized):null;
  if(observer)observer.observe(viewport);else window.addEventListener('resize',resized);
  return {
   async show(question){
    const token=++version;abort?.abort();abort=new AbortController();manifest=null;zoom=1;
    root.querySelector('.evidence-question').textContent='第 '+question.question_no+' 题';
    buttons.forEach(b=>b.disabled=true);status.textContent='正在加载本题学生作答原图…';viewport.setAttribute('aria-busy','true');viewport.innerHTML='<div class="evidence-empty">正在定位原图…</div>';
    try{
     const response=await fetch('/api/submissions/'+encodeURIComponent(submissionId)+'/review-visual?question_id='+encodeURIComponent(question.id),{signal:abort.signal});
     const data=await response.json();if(!valid(token))return;
     if(!response.ok)throw Error(response.status===404?'联动预览接口或答卷不存在；代码更新后请重启服务，可先打开原件。':typeof data.detail==='string'?data.detail:'原图加载失败');
     if(data.question_id!==question.id)throw Error('原图题号不匹配，已停止展示。');
     manifest=data;page=data.default_page;
     mode=preferredMode==='page'&&data.pages.length?'page':data.default_mode;
     render();
    }catch(error){if(error.name!=='AbortError'&&valid(token)){status.textContent=error.message;viewport.innerHTML='<div class="evidence-empty">原图暂不可用，可打开答卷原件。已编辑的文字和分数不会改变。</div>';}}
    finally{if(valid(token))viewport.removeAttribute('aria-busy');}
   },
   clear(){version++;abort?.abort();manifest=null;root.querySelector('.evidence-question').textContent='';viewport.innerHTML='<div class="evidence-empty">选择题目后同步显示学生原图。</div>';status.textContent='';},
   dispose(){disposed=true;version++;abort?.abort();observer?.disconnect();window.removeEventListener('resize',resized);}
  };
 };
})();
