const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => value == null ? '—' : esc(value);
const json = body => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function currentSubmissionScore(submission){
 const summary=submission?.score_summary||{};
 for(const value of [summary.current_score,summary.confirmed_score]){
  if(value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value)))return Number(value);
 }
 const grades=submission?.grade_items||[];
 if(!grades.length)return null;
 const scores=grades.map(g=>g.effective_score??g.final_score??g.score).filter(value=>value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value)));
 return scores.length?scores.reduce((total,value)=>total+Number(value),0):null;
}
let current = null, selected = null;
let submissionRows = [], activeJobs = new Map(), pendingFiles = [], currentTab = 'paper', openRequest = 0;
let submissionHistory = [];
let assessmentRequest = 0;
let reportCharts = [];
let lastReport = null;
let viewedQuestionIndex = 0, rerenderCurrentQuestion = null;
const pageState = {};
const pg = key => { if(!pageState[key]) pageState[key]={page:1,size:10}; return pageState[key]; };
function pageSlice(items, key){ const st=pg(key); const totalPages=Math.max(1,Math.ceil(items.length/st.size)); if(st.page>totalPages)st.page=totalPages; return items.slice((st.page-1)*st.size,st.page*st.size); }
function sortList(list, key, dir, numeric){ const c=[...list]; c.sort((a,b)=>{ const av=a[key],bv=b[key]; const aNull=av==null||av===''; const bNull=bv==null||bv===''; if(aNull&&bNull)return 0; if(aNull)return 1; if(bNull)return -1; let r; if(numeric){ r=(Number(av)||0)-(Number(bv)||0); if(r===0)r=String(av).localeCompare(String(bv),'zh'); } else { r=String(av).localeCompare(String(bv),'zh'); } return dir==='desc'?-r:r; }); return c; }
function renderPager(containerId,key,total){const el=$(containerId);if(!el)return;if(!total){el.innerHTML='';return;}const st=pg(key),pages=Math.max(1,Math.ceil(total/st.size));st.page=Math.min(st.page,pages);el.innerHTML='<span class="page-count">共 '+total+' 份 · '+st.page+' / '+pages+' 页</span><button class="secondary pg-prev" data-key="'+key+'" '+(st.page<=1?'disabled':'')+'>上一页</button><button class="secondary pg-next" data-key="'+key+'" '+(st.page>=pages?'disabled':'')+'>下一页</button><label>每页<select aria-label="每页显示数量" class="pg-size" data-key="'+key+'">'+[5,10,20,50].map(n=>'<option value="'+n+'" '+(n===st.size?'selected':'')+'>'+n+'</option>').join('')+'</select></label>';}
function onPagerChange(key){if(key==='report')renderReport().catch(e=>notice(e.message,true));else if(key==='token')renderTokenUsage().catch(e=>notice(e.message,true));else submissions().catch(e=>notice(e.message,true));}
document.addEventListener('click',e=>{const p=e.target.closest('.pg-prev');if(p){const st=pg(p.dataset.key);if(st.page>1){st.page--;onPagerChange(p.dataset.key);}}else{const n=e.target.closest('.pg-next');if(n){const st=pg(n.dataset.key);st.page++;onPagerChange(n.dataset.key);}}});
document.addEventListener('change',e=>{if(e.target.classList&&e.target.classList.contains('pg-size')){const st=pg(e.target.dataset.key);st.size=Number(e.target.value);st.page=1;onPagerChange(e.target.dataset.key);}});
let batchTimer = null;
let viewedSubmission = null;
const reviewReasons={quality_spot_check:'质量抽检',extraction_uncertain:'识别置信不足',ocr_confidence_too_low:'识别置信不足',score_disagreement:'多次评分有分歧',model_requested_review:'模型发现疑问',provisional_answer_key:'标准答案未确认',rubric_incomplete:'评分细则不完整',teacher_review_policy:'策略提示复核',insufficient_independent_passes:'评分次数不足',grading_evidence_missing:'缺少评分证据',answer_missing:'作答缺失',llm_request_failed:'模型调用失败'};
Object.assign(reviewReasons,{numeric_match:'数值匹配',numeric_mismatch:'数值不匹配',exact_match:'答案匹配',exact_mismatch:'答案不匹配',choice_set_match:'选项匹配',choice_set_mismatch:'选项不匹配',symbolic_match:'表达式等价',symbolic_mismatch:'表达式不等价',rubric_assisted:'按细则评分',confirmed_blank:'确认空白',blank_not_confirmed:'空白尚未确认',llm_not_configured:'模型未配置'});
function refreshReviewHint(){
 if(!viewedSubmission)return;
 const pending=current.questions.filter(q=>{const g=viewedSubmission.grade_items.find(x=>x.question_id===q.id);return !g||(g.needs_review&&g.final_score==null);}).length;
 $('reviewHint').textContent=$('reviewOnly').value==='all'?'全卷复核：可以逐题检查并修改当前成绩。':pending?'重点复核：系统建议关注 '+pending+' 题，其余成绩已直接生效。':'本答卷没有建议复核题；可切换为全卷复核。';
}
let reviewViewValue=$('reviewOnly').value;
$('reviewOnly').onchange=()=>{if(!mayLeaveDraft()){$('reviewOnly').value=reviewViewValue;return;}reviewViewValue=$('reviewOnly').value;if(rerenderCurrentQuestion)rerenderCurrentQuestion();};
async function api(path, options={}) {const r=await fetch(path,options); const body=await r.json().catch(()=>({})); if(!r.ok) throw Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||'请求失败')); return body;}
function notice(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':'';}
function action(id,fn){$(id).onclick=async()=>{const b=$(id);if(b.disabled)return;b.disabled=true;b.setAttribute('aria-busy','true');notice('正在处理，请稍候…');try{await fn();}catch(e){notice(e.message,true);}finally{b.disabled=false;b.removeAttribute('aria-busy');}};}
function hasDraft(){return [...$('submissionBody').querySelectorAll('[data-initial]')].some(e=>e.value!==e.dataset.initial);}
function mayLeaveDraft(){if(!hasDraft())return true;if(!confirm('当前有未保存的校对或复核内容。确定放弃修改并继续？'))return false;$('submissionBody').querySelectorAll('[data-initial]').forEach(e=>e.value=e.dataset.initial);return true;}
window.addEventListener('beforeunload',e=>{if(hasDraft()){e.preventDefault();e.returnValue='';}});
function aid(){if(!$('assessment').value) throw Error('请先新建或选择试卷');return $('assessment').value;}
function requireSavedAnswers(){if(viewedSubmission&&[...document.querySelectorAll('[data-answer]')].some(t=>t.value!==(viewedSubmission.responses.find(r=>r.question_id===t.dataset.answer)?.raw_text||'')))throw Error('作答校对尚未保存，请先保存并重新批改，再复核新结果。');}
$('submissionBody').addEventListener('submit',e=>{try{requireSavedAnswers();}catch(error){e.preventDefault();e.stopImmediatePropagation();notice(error.message,true);}},true);
$('submissionBody').addEventListener('click',e=>{if(e.target.id!=='bulkConfirm')return;try{requireSavedAnswers();}catch(error){e.preventDefault();e.stopImmediatePropagation();notice(error.message,true);}},true);
function showPreview(kind,id){$('previewFrame').src='/api/preview/'+kind+'/'+id;$('preview').showModal();}
$('closePreview').onclick=()=>{$('preview').close();$('previewFrame').src='about:blank';};
const pageInfo={paper:['试卷与答案','ASSESSMENT LIBRARY','把标准准备好，让每一份答卷有据可依。'],upload:['答卷收集','SUBMISSION INBOX','先收集，后批阅。原件保留，重复文件自动去重。'],mark:['智能批阅','GRADING QUEUE','批阅完成即形成当前成绩；复核是可选纠错流程。'],review:['复核工作台','REVIEW DESK','选择重点复核或全卷复核，最新人工分数覆盖机器分。'],report:['成绩洞察','LEARNING INSIGHTS','从当前成绩出发，看见学习中的进步与薄弱点。'],token:['用量记录','MODEL USAGE','每一次模型调用，都有迹可查。'],settings:['模型与策略','PREFERENCES','连接你的模型，选择合适的评分与复核提示策略。'],runtime:['识别加速','OPTIONAL ACCELERATION','可选的后端识别能力，不是批阅的前置步骤。']};
function navigate(tab){if(tab!==currentTab&&!mayLeaveDraft())return false;currentTab=tab;document.querySelectorAll('.tab').forEach(s=>s.hidden=s.id!==tab);document.querySelectorAll('[data-tab]').forEach(n=>{n.classList.toggle('active',n.dataset.tab===tab);n.setAttribute('aria-current',n.dataset.tab===tab?'page':'false');});$('heading').textContent=pageInfo[tab][0];$('pageEyebrow').textContent=pageInfo[tab][1];$('pageDescription').textContent=pageInfo[tab][2];$('listTools').hidden=!['upload','mark','review'].includes(tab);return true;}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{if(!navigate(b.dataset.tab))return;if(b.dataset.tab==='report'&&current)$('refreshReport').click();if(b.dataset.tab==='token'&&current)$('refreshToken').click();});
$('newAssessment').onclick=()=>{$('createDialog').showModal();$('title').focus();};
$('closeCreate').onclick=()=>$('createDialog').close();
$('title').onkeydown=e=>{if(e.key==='Enter')$('create').click();};
$('studentSearch').oninput=()=>{['upload','mark','review'].forEach(k=>pg(k).page=1);renderSubmissions();};
$('goMark').onclick=()=>navigate('mark');
$('goReview').onclick=()=>navigate('review');
let reviewFilterValue=$('reviewFilter').value;
$('reviewFilter').onchange=()=>{if(!mayLeaveDraft()){$('reviewFilter').value=reviewFilterValue;return;}reviewFilterValue=$('reviewFilter').value;pg('review').page=1;const row=submissionRows.find(s=>s.id===selected);const inScope=row&&(reviewFilterValue==='pending'?(row.review_recommended_count>0||row.status==='review_required'):reviewFilterValue==='reviewed'?row.review_count>0:['review_required','finalized'].includes(row.status));if(selected&&!inScope){selected=null;viewedSubmission=null;submissionHistory=[];rerenderCurrentQuestion=null;openRequest++;$('studentTitle').textContent='选择一份答卷';$('submissionBody').innerHTML='<div class="empty">请选择当前范围内的答卷。</div>';}renderSubmissions();};
action('bulkReviewAll',async()=>{if(hasDraft())throw Error('当前题目有未保存修改，请先保存或放弃后再一键复核。');const id=aid();const e=await api('/api/assessments/'+id+'/review-eligible');if(!e.count)throw Error('当前没有可一键复核的题目（有异常或已复核的题仍需逐题处理）');if(!confirm('将批量确认 '+e.count+' 道有明确建议分、无识别/评分异常的题目，记录为人工复核。确定？'))return;await api('/api/reviews/bulk',json({grade_item_ids:e.grade_item_ids,reviewer_ref:'',reason:''}));await submissions();if(selected)await openSubmission(selected,{force:true});notice('已批量确认 '+e.count+' 道题；异常题仍需逐题处理。');});
async function assessments(id){const list=await api('/api/assessments');$('assessment').innerHTML=list.map(a=>'<option value="'+a.id+'">'+esc(a.title)+' · '+(a.status==='published'?'已发布':'草稿')+'</option>').join('');if(id)$('assessment').value=id;await loadAssessment();}
async function loadAssessment(){
 const request=++assessmentRequest,id=$('assessment').value;openRequest++;rerenderCurrentQuestion=null;submissionRows=[];activeJobs.clear();$('studentSearch').value='';Object.values(pageState).forEach(s=>s.page=1);$('contextStats').innerHTML='';$('downloadReport').setAttribute('aria-disabled','true');
 viewedSubmission=null;
 clearTimeout(batchTimer);
 selected=null;current=null;pg('review').page=1;$('submissionBody').textContent='选择答卷查看详情。';$('studentTitle').textContent='选择一份答卷';$('reportBody').textContent='点击生成报告以读取最新成绩。';$('downloadReport').removeAttribute('href');
 if(!$('assessment').value)return;
 const [a,docs]=await Promise.all([api('/api/assessments/'+id),api('/api/assessments/'+id+'/documents')]);if(request!==assessmentRequest||$('assessment').value!==id)return;current=a;$('assessmentState').textContent=a.status==='published'?'已发布':'草稿';$('assessmentState').className='badge '+(a.status==='published'?'':'neutral');
 $('documents').innerHTML=docs.length?docs.map(d=>'<button data-doc="'+d.id+'">'+esc(({question_paper:'原试卷',answer_key:'答案册',rubric:'评分标准',reference:'参考资料'})[d.kind])+' · '+esc(d.media_type.includes('word')?'Word':d.media_type.split('/')[1])+'　查看原件</button>').join(''):'<p class="muted">还没有资料，请上传原试卷。</p>';
 document.querySelectorAll('[data-doc]').forEach(b=>b.onclick=()=>showPreview('document',b.dataset.doc));
 $('questions').innerHTML=a.questions.length?a.questions.map(q=>'<details class="question" open><summary>第 '+esc(q.question_no)+' 题 · '+q.max_score+' 分 <span class="badge">'+(q.answer_key_state==='candidate'?'待确认':'已确认')+'</span></summary><p>'+esc(q.prompt)+'</p><h3>参考答案</h3><pre>'+esc(typeof q.answer_key==='string'?q.answer_key:JSON.stringify(q.answer_key))+'</pre><h3>评分细则</h3>'+q.rubric.map(r=>'<p>'+esc(r.criterion)+' <b>'+esc(r.score)+' 分</b></p>').join('')+(q.rubric.length?'':'<p class="muted">根据标准答案及题目规则评分。</p>')+'</details>').join(''):'<p class="empty">上传试卷后，点击 AI 生成答案。</p>';
 setupRegionEditor(docs);
 $('exportTeacher').href='/api/assessments/'+a.id+'/training-samples?kind=teacher';
 $('exportPseudo').href='/api/assessments/'+a.id+'/training-samples?kind=pseudo';
 await submissions();
 await refreshBatches(a.id);
}
$('assessment').onchange=()=>{if(!mayLeaveDraft()){$('assessment').value=current?.id||'';return;}loadAssessment().then(()=>{if(currentTab==='report')return renderReport();if(currentTab==='token')return renderTokenUsage();}).catch(e=>notice(e.message,true));};
$('sortMode').onchange=()=>submissions().catch(e=>notice(e.message,true));
$('reportSort').onchange=()=>{if(current)renderReport().catch(e=>notice(e.message,true));};
$('tokenSort').onchange=()=>{if(current)renderTokenUsage().catch(e=>notice(e.message,true));};
const statusText={uploaded:'已上传，待批改',extraction_failed:'识别失败，可重试',extracted:'待批改',review_required:'缺少分数',finalized:'已出分'};
async function submissions(){
 const id=aid(),rows=await api('/api/submissions');
 if($('assessment').value!==id)return;
 submissionRows=rows.filter(s=>s.assessment_id===id);renderSubmissions();
}
function renderSubmissions(){
 const all=submissionRows,term=$('studentSearch').value.trim().toLocaleLowerCase();
 let rows=all.filter(s=>s.student_ref.toLocaleLowerCase().includes(term));
 const mode=$('sortMode').value;
 rows=sortList(rows,mode.startsWith('name')?'student_ref':'created_at',mode.endsWith('asc')?'asc':'desc');
 const pending=all.filter(s=>s.review_recommended_count>0||s.status==='review_required').length;
 $('contextStats').innerHTML='<span><b>'+all.length+'</b>份答卷</span><span><b>'+all.filter(s=>['uploaded','extracted','extraction_failed'].includes(s.status)).length+'</b>待批阅</span><span><b>'+pending+'</b>建议复核</span>';
 $('reviewNavCount').hidden=!pending;$('reviewNavCount').textContent=pending;
 const empty='<div class="empty"><strong>'+ (term?'没有匹配的答卷':'还没有答卷')+'</strong>'+(term?'换个姓名或文件名试试。':'先在答卷收集页上传文件。')+'</div>';
 const state=s=>activeJobs.has(s.id)?({queued:'排队中',extracting:'识别中',grading:'批阅中'}[activeJobs.get(s.id)]||'处理中'):(s.status==='finalized'&&s.review_recommended_count>0?'已出分 · 建议复核':statusText[s.status]||s.status);
 const badge=s=>'<span class="badge '+(s.status==='review_required'||s.status==='extraction_failed'?'warn':s.status==='finalized'?'':'neutral')+'">'+esc(state(s))+'</span>';
 const meta=s=>'<span class="avatar" aria-hidden="true">'+esc(s.student_ref.slice(0,1))+'</span><div class="student-meta"><strong>'+esc(s.student_ref)+'</strong><small>'+esc(new Date(s.created_at).toLocaleString())+(s.normalization_report?.answer_layout==='on_paper'?' · 卷面作答':s.normalization_report?.answer_layout==='answer_list'?' · 仅答案列表':'')+'</small></div>';
 $('uploadedList').innerHTML=pageSlice(rows,'upload').map(s=>'<div class="submission-row">'+meta(s)+badge(s)+'<div class="row-actions"><button class="secondary" data-source="'+s.id+'">预览</button><button class="text-button danger" data-del="'+s.id+'" '+(activeJobs.has(s.id)?'disabled title="处理中不能删除"':'')+'>删除</button></div></div>').join('')||empty;
 renderPager('uploadPager','upload',rows.length);
 $('gradingList').innerHTML=pageSlice(rows,'mark').map(s=>'<div class="submission-row">'+meta(s)+badge(s)+'<div class="row-actions">'+(['review_required','finalized'].includes(s.status)?'<button class="secondary" data-result="'+s.id+'">查看结果</button>':'<button class="secondary" data-source="'+s.id+'">原件</button>')+(['extraction_failed','review_required'].includes(s.status)?'<button class="text-button" data-retry="'+s.id+'" '+(activeJobs.has(s.id)?'disabled':'')+'>重试缺分题</button>':'')+'</div></div>').join('')||empty;
 renderPager('markPager','mark',rows.length);
 const filter=$('reviewFilter').value;
 const visible=rows.filter(s=>filter==='pending'?(s.review_recommended_count>0||s.status==='review_required'):filter==='reviewed'?s.review_count>0:['review_required','finalized'].includes(s.status));
 $('submissions').innerHTML=pageSlice(visible,'review').map(s=>'<button data-sub="'+s.id+'" class="'+(selected===s.id?'selected':'')+'">'+esc(s.student_ref)+'<small>'+esc(state(s))+' · '+(s.review_recommended_count||0)+' 题建议关注 · '+(s.review_count||0)+' 次复核</small></button>').join('')||'<p class="muted small">当前范围没有答卷。可切换“全部已出分答卷”。</p>';
 renderPager('reviewPager','review',visible.length);
 document.querySelectorAll('#submissions [data-sub]').forEach(b=>b.onclick=()=>{if(!navigate('review'))return;openSubmission(b.dataset.sub).catch(e=>notice(e.message,true));});
 document.querySelectorAll('[data-result]').forEach(b=>b.onclick=()=>{
  if(!navigate('review'))return;
  $('reviewFilter').value='all';reviewFilterValue='all';
  $('reviewOnly').value='all';reviewViewValue='all';
  renderSubmissions();
  openSubmission(b.dataset.result).catch(e=>notice(e.message,true));
 });
 document.querySelectorAll('[data-source]').forEach(b=>b.onclick=()=>showPreview('submission',b.dataset.source));
 document.querySelectorAll('[data-del]').forEach(b=>b.onclick=()=>removeSubmission(b.dataset.del,b));
 document.querySelectorAll('[data-retry]').forEach(b=>b.onclick=async()=>{if(!confirm('仅重试尚未取得分数的题目，可能消耗 Token；已有分数和人工复核保持不变。继续？'))return;b.disabled=true;try{await api('/api/assessments/'+aid()+'/grade-batch',json({submission_ids:[b.dataset.retry]}));await refreshBatches(aid());notice('缺分题已加入批阅队列。');}catch(e){notice(e.message,true);b.disabled=false;}});
}
async function removeSubmission(id,button){
 if(id===selected&&!mayLeaveDraft())return;
 if(!confirm('删除后，该答卷及批改、复核记录无法从界面恢复。确定删除？'))return;
 button.disabled=true;
 try{await api('/api/submissions/'+id,{method:'DELETE'});if(selected===id){selected=null;viewedSubmission=null;rerenderCurrentQuestion=null;openRequest++;$('submissionBody').innerHTML='<div class="empty">答卷已删除，请选择其他答卷。</div>';$('studentTitle').textContent='选择一份答卷';}await submissions();notice('答卷记录已删除。');}catch(e){notice(e.message,true);button.disabled=false;}
}
async function openSubmission(id,{force=false}={}){
 if(!force&&!mayLeaveDraft())return;
 const request=++openRequest,assessmentId=aid();
 const [s,history]=await Promise.all([api('/api/submissions/'+id),api('/api/submissions/'+id+'/history')]);
 if(request!==openRequest||aid()!==assessmentId)return;
 if(s.assessment_id!==assessmentId)throw Error('答卷与当前试卷不一致，请重新选择。');
 if(selected!==id)viewedQuestionIndex=0;
 selected=id;viewedSubmission=s;submissionHistory=history;
 $('studentTitle').textContent=s.student_ref;
 const questions=current.questions,summary=s.score_summary||{},currentScore=currentSubmissionScore(s);
 let nav=[];
 const navigable=()=>questions.map((q,i)=>i).filter(i=>{const g=s.grade_items.find(g=>g.question_id===questions[i].id);return $('reviewOnly').value==='all'||!g||(g.needs_review&&g.final_score==null);});
 $('submissionBody').className='';
 $('submissionBody').innerHTML='<div class="toolbar review-actions"><button id="viewSource" class="secondary">查看答卷原件 ↗</button><button id="deleteSubmission" class="text-button danger">删除答卷</button></div><div class="metrics"><div class="metric"><b>'+fmt(currentScore)+'</b>当前成绩</div><div class="metric"><b>'+fmt(summary.max_score)+'</b>试卷满分</div><div class="metric"><b>'+fmt(summary.review_recommended_count)+'</b>建议复核</div></div><div id="bulkArea"></div><div class="question-nav"><button id="prevQ" class="secondary">← 上一题</button><select id="questionSelect" aria-label="选择题号"></select><button id="nextQ" class="secondary">下一题 →</button></div><div id="questionArea"></div><details id="reviewAudit" class="review-audit"><summary>查看批改与复核历史（'+history.reduce((n,g)=>n+g.reviews.length,0)+' 次复核）</summary><div id="auditBody"></div></details>';
 $('viewSource').onclick=()=>showPreview('submission',id);
 $('deleteSubmission').disabled=activeJobs.has(id);
 $('deleteSubmission').onclick=()=>removeSubmission(id,$('deleteSubmission'));
 $('auditBody').innerHTML=history.map(g=>'<details class="trace"><summary>第 '+esc(g.snapshot?.question?.question_no||'?')+' 题 · 机器建议 '+fmt(g.score)+' 分 · '+(g.superseded_at?'历史版本':'当前版本')+'</summary><p>'+esc(new Date(g.created_at).toLocaleString())+' · '+esc(g.grader_kind)+' / '+esc(g.grader_version)+'</p><p>当时作答：'+esc(g.snapshot?.response?.raw_text??'未保存快照')+'</p><p>当时标准：'+esc(typeof g.snapshot?.question?.answer_key==='object'?JSON.stringify(g.snapshot.question.answer_key):g.snapshot?.question?.answer_key??'未保存快照')+'</p><pre>'+esc(g.evidence.join('\n'))+'</pre>'+g.reviews.map((rev,i)=>'<p><b>复核 '+(i+1)+' · '+fmt(rev.final_score)+' 分</b> · '+esc(rev.reviewer_ref||'人工复核')+'<br>'+esc(rev.reason||'确认建议分')+'<br><span class="muted small">'+esc(new Date(rev.created_at).toLocaleString())+'</span></p>').join('')+'</details>').join('')||'<p class="muted small">尚无批改记录。</p>';
 function renderQuestion(){
  const q=questions[viewedQuestionIndex];
  if(!q||!nav.length){$('questionArea').innerHTML='<div class="empty"><strong>当前没有建议复核题</strong>切换为“全卷复核”可以查看并调整任意题成绩。</div>';return;}
  const r=s.responses.find(x=>x.question_id===q.id),g=s.grade_items.find(x=>x.question_id===q.id),reference=g?.snapshot?.question||q;
  const answer=typeof reference.answer_key==='string'?reference.answer_key:JSON.stringify(reference.answer_key,null,2);
  const needs=!g||(g.needs_review&&g.final_score==null);
  const reasons=g?(needs?g.reason_codes.map(code=>reviewReasons[code]||code).join(' / '):g.final_score!=null?'已人工确认 · 可再次调整':'自动通过 · 非人工复核'):'尚未批阅';
  $('questionArea').innerHTML='<div class="question"><b>第 '+esc(q.question_no)+' 题</b><p>'+esc(reference.prompt)+'</p><div class="answer-comparison"><div><label>学生作答<textarea data-answer="'+q.id+'" aria-label="学生作答校对">'+esc(r?.raw_text||'')+'</textarea></label><p class="muted small">可校对识别文字。修改后需保存并重新批阅。</p><button id="saveAnswers" class="text-button">保存本题校对</button></div><div><h3>参考答案'+(g?.snapshot?.question?' · 批改时版本':'')+'</h3><pre>'+esc(answer||'未提供参考答案')+'</pre><h3>评分标准</h3>'+(reference.rubric?.length?reference.rubric.map(rr=>'<p>'+esc(rr.criterion)+' <b>'+esc(rr.score)+' 分</b></p>').join(''):'<p class="muted">按标准答案和题目规则评分</p>')+'</div></div><p class="'+(needs?'pending':'score')+'">'+(g?'当前得分 '+fmt(g.effective_score)+' / '+g.max_score+(needs?' · 建议复核':' · 已出分'):'尚未批阅')+'</p><p class="muted small">'+esc(reasons)+'</p>'+(g?'<details><summary>展开评分依据</summary><pre>'+esc(g.evidence.join('\n'))+'</pre></details><form data-review="'+g.id+'"><div class="toolbar"><label>复核分数<input name="score" aria-label="复核分数" type="number" min="0" max="'+g.max_score+'" step="any" required></label><label style="flex:1">复核说明<input name="reason" aria-label="复核说明" placeholder="可选，建议改分时填写" style="width:100%" maxlength="1000"></label><button>'+(g.latest_review_id?'保存再次复核':'保存人工复核')+'</button></div></form>':'<p class="helper-note">先到智能批阅页处理答卷，再在这里复核成绩。</p>')+'</div>';
  if(g){$('questionArea').querySelector('[name=score]').value=g.effective_score??'';const approve=document.createElement('button');approve.className='secondary';approve.textContent='脱敏并批准训练样本';approve.onclick=()=>openTraining(g,r);$('questionArea').append(approve);if(g.score_state==='AUTO_SINGLE'){const label=document.createElement('p');label.textContent='日常作业 · 单次自动评分（非双评一致）';$('questionArea').append(label);}}
  $('questionArea').querySelectorAll('input,textarea').forEach(e=>e.dataset.initial=e.value);
  action('saveAnswers',async()=>{
   const input=$('questionArea').querySelector('[data-answer]');
   if(input.value===(r?.raw_text||'')){notice('作答未改动，无需保存。');return;}
   await api('/api/submissions/'+id+'/answers',json({answers:[{question_id:q.id,raw_text:input.value}]}));
   await openSubmission(id,{force:true});notice('本题校对已保存，请到智能批阅页重新批阅。其他题目成绩保留。');
  });
  const form=$('questionArea').querySelector('form');
  if(form)form.onsubmit=async e=>{
   e.preventDefault();const button=form.querySelector('button');button.disabled=true;
   try{requireSavedAnswers();const data=new FormData(form);await api('/api/grade-items/'+g.id+'/review',json({reviewer_ref:'',final_score:Number(data.get('score')),reason:data.get('reason')||'',check_revision:true,expected_review_id:g.latest_review_id}));await openSubmission(id,{force:true});notice('复核已保存。旧记录保留，成绩洞察采用最新分数。');}
   catch(error){notice(error.message,true);button.disabled=false;}
  };
 }
 function renderNavigation(){
  nav=navigable();if(!nav.includes(viewedQuestionIndex))viewedQuestionIndex=nav[0];
  $('questionSelect').innerHTML=nav.map(i=>'<option value="'+i+'" '+(i===viewedQuestionIndex?'selected':'')+'>第 '+esc(questions[i].question_no)+' 题 · '+questions[i].max_score+' 分</option>').join('')||'<option>没有待处理题</option>';
  const pos=nav.indexOf(viewedQuestionIndex);$('prevQ').disabled=pos<=0;$('nextQ').disabled=pos<0||pos>=nav.length-1;$('questionSelect').disabled=!nav.length;
  refreshReviewHint();renderQuestion();
 }
 function selectQuestion(index){if(!mayLeaveDraft()){$('questionSelect').value=String(viewedQuestionIndex);return;}viewedQuestionIndex=index;renderNavigation();}
 $('prevQ').onclick=()=>selectQuestion(nav[nav.indexOf(viewedQuestionIndex)-1]);
 $('nextQ').onclick=()=>selectQuestion(nav[nav.indexOf(viewedQuestionIndex)+1]);
 $('questionSelect').onchange=()=>selectQuestion(Number($('questionSelect').value));
 rerenderCurrentQuestion=renderNavigation;
 const eligible=s.grade_items.filter(g=>g.bulk_review_eligible);
 if(eligible.length){
  $('bulkArea').innerHTML='<details class="bulk-review"><summary>本卷 '+eligible.length+' 道题可批量确认</summary><p class="muted">仅包含有明确建议分且无评分、识别异常的题。请检查后确认。</p><p>'+eligible.map(g=>'第 '+esc(g.question_no)+' 题 '+fmt(g.score)+' / '+g.max_score).join('；')+'</p><div class="toolbar"><input id="bulkReason" placeholder="确认依据（可选）" maxlength="1000"><button id="bulkConfirm">确认这 '+eligible.length+' 道题</button></div></details>';
  action('bulkConfirm',async()=>{requireSavedAnswers();if(hasDraft())throw Error('请先保存或放弃本题复核修改，再进行批量确认。');if(!confirm('确认已检查列出的 '+eligible.length+' 道题，并按建议分记录人工复核？'))return;await api('/api/reviews/bulk',json({grade_item_ids:eligible.map(g=>g.id),reviewer_ref:'',reason:$('bulkReason').value.trim()}));await openSubmission(id,{force:true});notice('可确认项已保存，异常题仍需单独处理。');});
 }
 renderNavigation();await submissions();
}
action('create',async()=>{if(!$('title').value.trim())throw Error('请输入试卷名称');if(!mayLeaveDraft())return;const a=await api('/api/assessments',json({title:$('title').value.trim()}));$('createDialog').close();$('title').value='';await assessments(a.id);navigate('paper');notice('试卷已创建，先保存资料，再准备答案与评分标准。');});
['paperFiles','answerFiles'].forEach(id=>$(id).accept='.doc,.docx,.pdf,.jpg,.jpeg,.png,.webp');
action('uploadPaper',async()=>{if(!$('paperFiles').files.length)throw Error('请选择资料');const f=new FormData();f.append('kind',$('kind').value);[...$('paperFiles').files].forEach(x=>f.append('files',x));await api('/api/assessments/'+aid()+'/documents',{method:'POST',body:f});await loadAssessment();notice('资料已入库，可点击预览。');});
action('generate',async()=>{await api('/api/assessments/'+aid()+'/generate-candidate-key',{method:'POST'});await loadAssessment();notice('候选答案已生成，请展开查看答案和评分细则。');});
action('fillMissing',async()=>{const id=aid();const r=await api('/api/assessments/'+id+'/fill-missing',{method:'POST'});if(!r.filled){notice('没有缺失的答案或评分标准，无需补全。');return;}await loadAssessment();notice('已补全 '+r.filled+' 道题的缺失内容，请展开检查并确认。');});
action('confirm',async()=>{if(!confirm('确认已检查每题答案与评分细则？'))return;await api('/api/assessments/'+aid()+'/confirm-candidate-key',{method:'POST'});await loadAssessment();notice('答案已确认。');});
action('publish',async()=>{const id=aid();await api('/api/assessments/'+id+'/publish',{method:'POST'});await assessments(id);notice('试卷已发布，可以上传答卷。');});
async function refreshBatches(id){
 clearTimeout(batchTimer);
 try{
  const rows=await api('/api/batches?assessment_id='+encodeURIComponent(id));
  if($('assessment').value!==id)return;
  activeJobs=new Map(rows.flatMap(b=>b.items).filter(i=>['queued','extracting','grading'].includes(i.status)&&i.submission_id).map(i=>[i.submission_id,i.status]));
  const active=rows.filter(b=>b.status==='processing');
  $('queueState').textContent=active.length?activeJobs.size+' 份处理中':'没有运行中的任务';
  $('queueState').className='badge '+(active.length?'':'neutral');
  const shown=active.length?active:rows.slice(0,1);
  $('batchResults').innerHTML=shown.map(b=>'<div class="batch-progress"><span>'+esc(new Date(b.created_at).toLocaleString())+' · '+(b.status==='processing'?'正在处理':'最近一批')+'</span><progress value="'+(b.success_count+b.failed_count)+'" max="'+Math.max(1,b.total_count)+'"></progress><span>'+b.success_count+' / '+b.total_count+' 处理完成'+(b.duplicate_count?' · '+b.duplicate_count+' 份重复跳过':'')+(b.failed_count?' · '+b.failed_count+' 份失败':'')+'</span>'+'<button class="secondary" data-performance="'+b.id+'">查看耗时与用量</button><div id="perf-'+b.id+'"></div>'+(b.failed_count?'<details><summary>查看失败原因</summary>'+b.items.filter(i=>i.error).map(i=>'<p class="batch-error">'+esc(i.filename)+'：'+esc(i.error)+'</p>').join('')+'</details>':'')+'</div>').join('');
  await submissions();
  document.querySelectorAll('[data-performance]').forEach(button=>button.onclick=async()=>{try{const p=await api('/api/batches/'+button.dataset.performance+'/performance');$('perf-'+button.dataset.performance).textContent='总耗时 '+(p.wall_ms/1000).toFixed(2)+' 秒；请求 '+p.requests+' 次；输入 '+p.tokens.input_tokens+' / 输出 '+p.tokens.output_tokens+' / 缓存 '+p.tokens.cached_tokens+' token；失败 '+p.failed_requests+'；补请求 '+p.retries+'。未配置价格，不估算费用。';}catch(e){notice(e.message,true);}});
  if(active.length)batchTimer=setTimeout(()=>refreshBatches(id),2500);
 }catch(e){if($('assessment').value===id){notice('进度读取失败，稍后自动重试：'+e.message,true);batchTimer=setTimeout(()=>refreshBatches(id),5000);}}
}
function showChosenFiles(files){
 pendingFiles=[...files];$('answerFileHint').textContent=pendingFiles.length?'已选择 '+pendingFiles.length+' 个文件 · 点击下方按钮上传':'每个文件不超过 25 MB；同卷同内容自动去重';
}
$('answerFiles').onchange=()=>showChosenFiles($('answerFiles').files);
$('paperFiles').onchange=()=>{$('paperFileHint').textContent=$('paperFiles').files.length?'已选择 '+$('paperFiles').files.length+' 个文件':'尚未选择文件';};
$('answerDropZone').ondragover=e=>{e.preventDefault();$('answerDropZone').classList.add('dragover');};
$('answerDropZone').ondragleave=()=>$('answerDropZone').classList.remove('dragover');
$('answerDropZone').ondrop=e=>{e.preventDefault();$('answerDropZone').classList.remove('dragover');if(!$('batch').disabled)showChosenFiles(e.dataTransfer.files);};
async function uploadFiles(fileList){
 if(!current||current.status!=='published')throw Error('请先发布试卷，再收集答卷。');
 if(!fileList.length)throw Error('请选择答卷文件。');
 const id=aid(),files=[...fileList],valid=[],errors=[];
 for(const file of files){if(!/\.(docx?|pdf|jpe?g|png|webp)$/i.test(file.name))errors.push(file.name+'：不支持的格式');else if(file.size>25*1024*1024)errors.push(file.name+'：超过 25 MB');else valid.push(file);}
 if(!valid.length)throw Error(errors.slice(0,5).join('；')||'没有可上传的文件');
 let added=0,duplicates=0,failed=errors.length;
 ['assessment','answerFiles','folderFiles','uploadFolder','answerLayout'].forEach(key=>$(key).disabled=true);
 try{
  for(let offset=0;offset<valid.length;offset+=100){
   const body=new FormData();body.append('assessment_id',id);body.append('auto_grade','false');body.append('answer_layout',$('answerLayout').value);
   valid.slice(offset,offset+100).forEach(file=>body.append('files',file));
   const result=await api('/api/batches',{method:'POST',body});
   added+=result.success_count-result.duplicate_count;duplicates+=result.duplicate_count;failed+=result.failed_count;
   errors.push(...result.items.filter(i=>i.error).map(i=>i.filename+'：'+i.error));
   $('uploadResults').textContent='新增 '+added+' 份 · 去重 '+duplicates+' 份 · 失败 / 跳过 '+failed+' 份 · 已处理 '+Math.min(offset+100,valid.length)+' / '+valid.length;
  }
  $('uploadResults').innerHTML='<strong>收集完成</strong> · 新增 '+added+' 份 · 去重 '+duplicates+' 份 · 失败 / 跳过 '+failed+' 份'+(errors.length?'<details><summary>查看 '+errors.length+' 项提示</summary>'+errors.map(e=>'<p>'+esc(e)+'</p>').join('')+'</details>':'');
  $('answerFiles').value='';$('folderFiles').value='';showChosenFiles([]);
  await refreshBatches(id);notice('答卷已保存，尚未识别或批阅。下一步到智能批阅页开始。');
 }catch(error){await refreshBatches(id);throw Error('已保存的答卷不会丢失，可重新上传并自动去重。'+error.message);}
 finally{['assessment','answerFiles','folderFiles','uploadFolder','answerLayout'].forEach(key=>$(key).disabled=false);}
}
action('batch',()=>uploadFiles(pendingFiles));
$('uploadFolder').onclick=()=>$('folderFiles').click();
$('folderFiles').onchange=()=>showChosenFiles($('folderFiles').files);
action('grade',async()=>{await api('/api/assessments/'+aid()+'/grade-batch',json({}));await refreshBatches(aid());notice('待批答卷已加入队列；模型或规则给出分数后即形成当前成绩，可按需进入复核。');});
async function renderReport(){
 const id=aid(),r=await api('/api/assessments/'+id+'/report');if(aid()!==id)return;
 lastReport=r;
 const rm=$('reportSort').value;
 let students=r.students;
 if(rm==='score_desc')students=sortList(students,'final_score','desc',true);
 else if(rm==='score_asc')students=sortList(students,'final_score','asc',true);
 else if(rm==='name_desc')students=sortList(students,'student','desc');
 else students=sortList(students,'student','asc');
 const studentsPaged=pageSlice(students,'report');
 $('reportBody').className='';
 reportCharts.forEach(c=>c.destroy());reportCharts=[];
 $('reportBody').innerHTML='<div class="metrics"><div class="metric"><b>'+r.submission_count+'</b>答卷份数</div><div class="metric"><b>'+r.finalized_count+'</b>已出分</div><div class="metric"><b>'+fmt(r.average)+'</b>当前平均分</div></div><p class="muted">'+esc(r.method)+'</p><div class="charts"><div class="chart-box"><h3>成绩分布</h3><canvas id="distChart"></canvas></div><div class="chart-box"><h3>批改方式</h3><canvas id="modeChart"></canvas></div></div><div class="chart-box"><h3>逐题得分率趋势</h3><canvas id="qRateChart"></canvas></div><h3>学生成绩</h3><table><tr><th>学生</th><th>当前成绩</th><th>建议复核</th><th>缺少分数</th><th>操作</th></tr>'+studentsPaged.map(s=>'<tr><td>'+esc(s.student)+'</td><td>'+fmt(s.final_score)+'</td><td>'+s.review_recommended+'</td><td>'+s.pending+'</td><td><button data-report-sub="'+s.id+'">查看 / 复核</button></td></tr>').join('')+'</table><div id="reportPager" class="pagination"></div><p>优先复习得分率较低的题目。缺少分数的题不会按零分统计。</p>';
 $('reportBody').querySelectorAll('[data-report-sub]').forEach(b=>b.onclick=async()=>{try{$('reviewOnly').value='all';reviewViewValue='all';$('reviewFilter').value='all';reviewFilterValue='all';navigate('review');await openSubmission(b.dataset.reportSub);}catch(e){notice(e.message,true);}});
 renderPager('reportPager','report',students.length);
 if(window.Chart){
  Chart.defaults.font.family='"Segoe UI", "Microsoft YaHei", sans-serif';Chart.defaults.color='#8190a6';
  reportCharts.push(new Chart($('distChart'),{type:'bar',data:{labels:(r.score_distribution||[]).map(d=>d.label),datasets:[{label:'人数',data:(r.score_distribution||[]).map(d=>d.count),backgroundColor:'#718ee4',borderRadius:5,maxBarThickness:42}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,ticks:{precision:0},grid:{color:'#edf0f5'}},x:{grid:{display:false}}}}}));
  reportCharts.push(new Chart($('modeChart'),{type:'doughnut',data:{labels:['模型评分','规则评分','人工复核','缺少分数'],datasets:[{data:[r.auto_stable_question_count||0,r.rule_passed_question_count||0,r.human_reviewed_question_count||0,r.pending_question_count||0],backgroundColor:['#5275d9','#78a999','#c7a575','#dfe5ef'],borderWidth:4}]},options:{cutout:'74%',plugins:{legend:{position:'bottom',labels:{usePointStyle:true,boxWidth:7,padding:16}}}}}));
  reportCharts.push(new Chart($('qRateChart'),{type:'line',data:{labels:r.questions.map(q=>'第'+q.question_no+'题'),datasets:[{label:'得分率 %',data:r.questions.map(q=>q.score_rate??null),borderColor:'#718ee4',backgroundColor:'#718ee4',fill:false,tension:0.2,spanGaps:false}]},options:{scales:{y:{beginAtZero:true,max:100,grid:{color:'#edf0f5'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}}}));
 }
 const detail=document.createElement('details');detail.className='review-audit';detail.innerHTML='<summary>逐题成绩明细（无分数不按零分统计）</summary><table><tr><th>题号</th><th>已出分样本</th><th>平均分</th><th>得分率</th></tr>'+r.questions.map(q=>'<tr><td>'+esc(q.question_no)+'</td><td>'+q.sample_count+'</td><td>'+fmt(q.average)+'</td><td>'+fmt(q.score_rate)+(q.score_rate==null?'':'%')+'</td></tr>').join('')+'</table>';$('reportBody').append(detail);
 $('downloadReport').href='/api/assessments/'+id+'/report.html';$('downloadReport').setAttribute('aria-disabled','false');wrapTables($('reportBody'));notice('统计已更新：机器分直接计入，存在人工复核时采用最新复核分。');
}
action('refreshReport',renderReport);
async function renderTokenUsage(){
 const id=aid(),tu=await api('/api/assessments/'+id+'/token-usage');if(aid()!==id)return;
 $('tokenBody').className='';
 const stageLabels={extract:'抽取答案',grade:'过程题批改',compile:'生成候选答案'};
 const tok=u=>u?((u.input||0)+' / '+(u.output||0)):'—';
 const cacheRate=tu.total.input?Math.round(tu.total.cached/tu.total.input*100):0;
 let html='<div class="metrics"><div class="metric"><b>'+(tu.total.input+tu.total.output)+'</b>总 token（输入+输出）</div><div class="metric"><b>'+tu.total.input+'</b>输入 token</div><div class="metric"><b>'+tu.total.output+'</b>输出 token</div></div>';
 html+='<p class="muted">缓存命中 '+tu.total.cached+' token，命中率 '+cacheRate+'%：命中越高，重复输入越省。生成候选答案按卷计，不计入单张答卷。</p>';
 const stageRows=Object.keys(tu.stages).map(k=>'<tr><td>'+esc(stageLabels[k]||k)+'</td><td>'+tu.stages[k].input+'</td><td>'+tu.stages[k].output+'</td><td>'+tu.stages[k].cached+'</td><td>'+(tu.stages[k].input+tu.stages[k].output)+'</td></tr>').join('');
 html+='<h3>分环节明细</h3><table><tr><th>环节</th><th>输入</th><th>输出</th><th>缓存命中</th><th>合计</th></tr>'+(stageRows||'<tr><td colspan="5" class="muted">暂无记录。</td></tr>')+'</table>';
 const tm=$('tokenSort').value;
 const tokSum=s=>s.total?((s.total.input||0)+(s.total.output||0)):0;
 let tsubs=[...tu.submissions];
 if(tm==='token_desc')tsubs.sort((a,b)=>tokSum(b)-tokSum(a));
 else if(tm==='token_asc')tsubs.sort((a,b)=>tokSum(a)-tokSum(b));
 else if(tm==='name_desc')tsubs.sort((a,b)=>String(b.student||'').localeCompare(String(a.student||''),'zh'));
 else tsubs.sort((a,b)=>String(a.student||'').localeCompare(String(b.student||''),'zh'));
 const tokenPaged=pageSlice(tsubs,'token');
 html+='<h3>分答卷明细（输入 / 输出）</h3><table><tr><th>学生</th><th>抽取答案</th><th>过程题批改</th><th>合计</th><th>缓存命中</th></tr>'+(tokenPaged.length?tokenPaged.map(s=>'<tr><td>'+esc(s.student)+'</td><td>'+tok(s.stages.extract)+'</td><td>'+tok(s.stages.grade)+'</td><td>'+tok(s.total)+'</td><td>'+(s.total? (s.total.cached||0):0)+'</td></tr>').join(''):'<tr><td colspan="5" class="muted">暂无记录，批改后自动统计。</td></tr>')+'</table><div id="tokenPager" class="pagination"></div>';
 $('tokenBody').innerHTML=html;
 wrapTables($('tokenBody'));
 renderPager('tokenPager','token',tsubs.length);
}
action('refreshToken',renderTokenUsage);
async function refreshHealth(){const h=await api('/api/health');$('health').textContent=h.llm_configured?'模型已配置':'模型待配置';const r=h.ocr_runtime;$('gpuStatus').textContent=!r.paddle_installed||!r.paddleocr_installed?'当前服务使用的 Python 未装齐 OCR。请关闭服务，用 start-optimization.ps1 重新启动；它会优先选择原有 OCR 环境。':r.active_device?'最近实际识别设备：'+r.active_device:'后端 OCR 包已安装，可选择检查设备，也可以跳过。';$('gpuStatus').textContent+=' 当前服务环境：'+(r.python_executable||'未知')+'；OCR 选项：'+(h.ocr_provider==='paddleocr'?'已启用':'未启用')+'。';if(Object.keys(r.errors).length)$('gpuStatus').textContent+=' 错误：'+Object.values(r.errors).join('；')+'（排除故障后重启服务）';if(r.compatibility_warning)$('gpuStatus').textContent+=' '+r.compatibility_warning;}
let modelProviders=[], savedConnection=null, customConnectionDraft={model:'',base_url:''}, displayedProvider='custom';
function selectedModelConnection(){const id=$('llmProvider').value,p=modelProviders.find(x=>x.id===id);return {provider:id,model:p?$('llmPresetModel').value:$('llmModel').value.trim(),base_url:p?p.base_url:$('llmBaseUrl').value.trim()};}
function renderModelProvider(keepModel){const id=$('llmProvider').value,p=modelProviders.find(x=>x.id===id),custom=!p;
 $('llmPresetModelWrap').hidden=custom;$('llmCustomModelWrap').hidden=!custom;$('llmBaseUrlWrap').hidden=!custom;
 if(p){$('llmPresetModel').innerHTML=p.models.map(m=>'<option value="'+esc(m)+'">'+esc(m)+'</option>').join('');if(keepModel&&p.models.includes(keepModel))$('llmPresetModel').value=keepModel;$('llmBaseUrl').value=p.base_url;}
 else {$('llmModel').value=customConnectionDraft.model;$('llmBaseUrl').value=customConnectionDraft.base_url;}
 $('llmProviderNote').textContent=p?p.note:(savedConnection?.provider==='custom'&&savedConnection?.model?.startsWith('qwen3.8-')&&customConnectionDraft.base_url===savedConnection.base_url?'已保留你原有的 Qwen 自定义地址与密钥；不切换服务商即可照常使用。':'填写兼容 Responses API、支持图像输入的模型和基础地址。自定义接口的实际能力需自行测试；Chat 接口请使用上方预设。');
 for(const [element,url] of [['llmDocs',p?.docs_url],['llmKeyDocs',p?.key_url]]){$(element).hidden=!url;if(url)$(element).href=url;}
 const changed=savedConnection&&selectedModelConnection().base_url!==savedConnection.base_url;
 $('llmKey').placeholder=changed?'切换服务地址，请输入该服务的 API 密钥':savedConnection?.api_key_set?'已配置（留空保持不变）':'请输入 API 密钥';
 displayedProvider=id;
}
$('llmProvider').addEventListener('change',()=>{if(displayedProvider==='custom')customConnectionDraft={model:$('llmModel').value,base_url:$('llmBaseUrl').value};renderModelProvider();});
async function loadSettings(){const [s,c]=await Promise.all([api('/api/settings'),api('/api/model-catalog')]);modelProviders=c.providers||[];savedConnection=s;
 $('llmProvider').innerHTML=modelProviders.map(p=>'<option value="'+esc(p.id)+'">'+esc(p.name)+'</option>').join('')+'<option value="custom">自定义连接</option>';
 $('llmProvider').value=s.provider||'custom';customConnectionDraft=s.provider==='custom'?{model:s.model,base_url:s.base_url}:{model:'',base_url:''};$('llmModel').value=s.model;$('llmBaseUrl').value=s.base_url;renderModelProvider(s.model);
 $('gradingProfile').value=s.grading_profile||'legacy';$('llmPasses').value=String(s.llm_grading_passes);$('llmExtract').checked=s.llm_extract_answers;$('llmThinking').checked=s.enable_thinking;$('llmKey').value='';$('settingsStatus').textContent=s.api_key_set?'当前密钥：'+s.api_key_masked:'尚未配置密钥，过程题与 AI 生成答案将不可用。';$('ocrProvider').value=s.ocr_provider;$('ocrDevice').value=s.ocr_device;$('localOcrFirst').checked=s.local_ocr_first;$('reviewMode').value=s.review_mode;$('sampleRate').value=String(Math.round(s.review_sample_rate*100));$('confidenceThreshold').value=s.review_confidence_threshold;}
action('saveSettings',async()=>{const key=$('llmKey').value.trim(),connection=selectedModelConnection();await api('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key:key||null,...connection,grading_profile:$('gradingProfile').value,llm_grading_passes:Number($('llmPasses').value),llm_extract_answers:$('llmExtract').checked,enable_thinking:$('llmThinking').checked,review_mode:$('reviewMode').value,review_sample_rate:Number($('sampleRate').value)/100,review_confidence_threshold:Number($('confidenceThreshold').value)})});await loadSettings();await refreshHealth();$('llmTestResult').textContent='设置已保存。可测试连接后再批阅。';notice('模型与复核策略已保存；仅影响后续批改，不自动改动已有成绩。');});
action('testLlm',async()=>{const result=await api('/api/settings/test-connection',{method:'POST'});$('llmTestResult').textContent=result.message;notice(result.message);});
action('saveRuntime',async()=>{const saved=await api('/api/settings');await api('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:saved.model,base_url:saved.base_url,llm_grading_passes:saved.llm_grading_passes,llm_extract_answers:saved.llm_extract_answers,enable_thinking:saved.enable_thinking,ocr_provider:$('ocrProvider').value,ocr_device:$('ocrDevice').value,local_ocr_first:$('localOcrFirst').checked})});await refreshHealth();notice('识别设置已保存。环境检查为可选，不影响远程模型流程。');});
action('checkOcr',async()=>{const result=await api('/api/runtime/ocr-check',{method:'POST'});await refreshHealth();$('ocrCheckResult').textContent=result.ok?'实际设备：'+result.device+'；本次首次调用 '+result.durations_seconds[0]+' 秒，第二次 '+result.durations_seconds[1]+' 秒。'+result.message:result.message;notice(result.ok?'本地自检通过，当前服务中的 OCR 模型已预热。':'本地自检失败，请查看设备错误信息。',!result.ok);});
function wrapTables(container){container.querySelectorAll('table').forEach(table=>{const wrapper=document.createElement('div');wrapper.className='table-scroll';wrapper.tabIndex=0;wrapper.setAttribute('aria-label','可横向滚动的数据表格');table.replaceWith(wrapper);wrapper.append(table);});}
(async()=>{await Promise.all([refreshHealth(),assessments(),loadSettings()].map(p=>p.catch(e=>notice(e.message,true))));})();

// Region coordinates always refer to the source page, independent of CSS scaling.
let regionImage=null, regionSelection=null, regionStart=null, regionDocumentId=null, regionLoadedPage=null, regionPreviewUrl=null, regionSaved=false;
function setupRegionEditor(docs){
 const supported=docs.filter(d=>d.kind==='question_paper'&&(d.media_type==='application/pdf'||d.media_type.startsWith('image/')||d.media_type==='application/msword'||d.media_type==='application/vnd.openxmlformats-officedocument.wordprocessingml.document'));
 $('regionDocument').innerHTML=supported.map((d,i)=>'<option value="'+esc(d.id)+'">原试卷 · 第 '+(i+1)+' 份资料</option>').join('');
 $('regionQuestion').innerHTML=current.questions.map(q=>'<option value="'+esc(q.id)+'">第 '+esc(q.question_no)+' 题'+(q.rule_config?.region?' · 已配置':'')+'</option>').join('');
 $('editRegions').disabled=false;$('editRegions').title=!supported.length?'请先上传原试卷':!current.questions.length?'请先生成或录入题目':'支持图片、PDF 和 Word 原卷';
}
function drawRegion(){
 const canvas=$('regionCanvas'),ctx=canvas.getContext('2d');if(!regionImage)return;
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(regionImage,0,0);
 if(regionSelection){const r=regionSelection;ctx.fillStyle='rgba(44,100,240,.15)';ctx.fillRect(r.x*canvas.width,r.y*canvas.height,r.width*canvas.width,r.height*canvas.height);ctx.strokeStyle='#245de0';ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(r.x*canvas.width,r.y*canvas.height,r.width*canvas.width,r.height*canvas.height);}
}
function selectSavedRegion(){
 const q=current?.questions?.find(q=>q.id===$('regionQuestion').value),r=q?.rule_config?.region;
 regionSelection=r&&r.page===regionLoadedPage?{...r}:null;drawRegion();
 $('regionStatus').textContent=!current?.questions?.length?'原试卷已加载。请先生成参考答案或录入题目，再为题号保存答题区域。':regionSelection?'已载入该题区域，可重新拖动修改':'拖动鼠标框选学生实际作答区域，避免印刷题干';
}
function openRegionEditor(){
 regionSaved=false;regionImage=null;regionLoadedPage=null;regionSelection=null;regionStart=null;
 $('regionCanvas').width=0;$('regionCrop').hidden=true;
 const canRender=!!current&&$('regionDocument').options.length>0,hasQuestions=!!current?.questions?.length;
 const problem=!current?'请先新建或选择试卷。':!canRender?'请先在试卷资料中上传原试卷（Word、PDF 或图片）。':!hasQuestions?'可先预览原试卷；请先生成参考答案或录入题目，再为题号保存答题区域。':'';
 for(const id of ['loadRegionPage','regionDocument','regionPage'])$(id).disabled=!canRender;
 for(const id of ['regionQuestion','previewRegion','saveRegion','removeRegion'])$(id).disabled=!canRender||!hasQuestions;
 $('regionStatus').textContent=problem||'在原试卷上框选学生将填写答案的空白位置。';
 $('regionStatus').className=problem?'region-guidance':'region-guidance ready';
 if(!$('regionDialog').open)$('regionDialog').showModal();
 if(canRender){$('regionPage').value=String(current.questions?.find(q=>q.id===$('regionQuestion').value)?.rule_config?.region?.page||1);$('loadRegionPage').click();}
}
$('editRegions').onclick=openRegionEditor;
$('regionPrepare').onclick=()=>{$('regionDialog').close();const target=!current?$('newAssessment'):!$('regionDocument').options.length?$('paperFiles'):!current.questions.length?$('generate'):$('editRegions');target.scrollIntoView({block:'center'});target.focus();};
$('closeRegions').onclick=()=>{$('regionDialog').close();};
$('regionDialog').addEventListener('close',()=>{if(regionPreviewUrl)URL.revokeObjectURL(regionPreviewUrl);if(regionSaved)loadAssessment().catch(e=>notice(e.message,true));});
$('regionDocument').onchange=()=>{regionLoadedPage=null;regionSelection=null;$('regionStatus').textContent='请重新加载页面';};
$('regionPage').oninput=()=>{regionLoadedPage=null;regionSelection=null;$('regionStatus').textContent='请重新加载页面';};
$('regionQuestion').onchange=selectSavedRegion;
action('loadRegionPage',async()=>{
 const doc=$('regionDocument').value,page=Number($('regionPage').value);if(!Number.isInteger(page)||page<1)throw Error('请输入有效页码');$('regionStatus').textContent='正在加载页面；Word 原卷首次需要本机转换为 PDF，请稍候。';
 const response=await fetch('/api/assessments/'+aid()+'/template-page?document_id='+encodeURIComponent(doc)+'&page='+page);
 if(!response.ok){const detail=await response.json().catch(()=>({}));const message=detail.detail||'页面加载失败，请检查页码和资料类型';$('regionStatus').textContent=message;throw Error(message);}
 const url=URL.createObjectURL(await response.blob());
 try{const image=new Image();image.src=url;await image.decode();regionImage=image;regionDocumentId=doc;regionLoadedPage=page;$('regionCanvas').width=image.naturalWidth;$('regionCanvas').height=image.naturalHeight;selectSavedRegion();}finally{URL.revokeObjectURL(url);}
});
const regionCanvas=$('regionCanvas');
function regionPoint(e){const rect=regionCanvas.getBoundingClientRect();return {x:Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),y:Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height))};}
regionCanvas.onpointerdown=e=>{if(!regionLoadedPage||!current?.questions?.length)return;regionStart=regionPoint(e);regionCanvas.setPointerCapture(e.pointerId);};
regionCanvas.onpointermove=e=>{if(!regionStart)return;const end=regionPoint(e);regionSelection={page:regionLoadedPage,x:Math.min(end.x,regionStart.x),y:Math.min(end.y,regionStart.y),width:Math.abs(end.x-regionStart.x),height:Math.abs(end.y-regionStart.y),page_aspect:regionCanvas.width/regionCanvas.height};drawRegion();};
regionCanvas.onpointerup=()=>{regionStart=null;};regionCanvas.onpointercancel=()=>{regionStart=null;};
function selectedRegion(){if(!regionSelection||!regionLoadedPage||regionSelection.width<=0||regionSelection.height<=0)throw Error('请先框选有效区域');return regionSelection;}
action('previewRegion',async()=>{
 const response=await fetch('/api/assessments/'+aid()+'/template-preview?document_id='+encodeURIComponent(regionDocumentId),json(selectedRegion()));
 if(!response.ok)throw Error('裁切失败，请检查区域');if(regionPreviewUrl)URL.revokeObjectURL(regionPreviewUrl);regionPreviewUrl=URL.createObjectURL(await response.blob());$('regionCrop').src=regionPreviewUrl;$('regionCrop').hidden=false;
});
async function persistRegion(remove){
 const region=remove?null:selectedRegion(),assessment=await api('/api/assessments/'+aid()),q=assessment.questions.find(q=>q.id===$('regionQuestion').value);
 if(!q)throw Error('题目已变化，请重新打开模板');const config={...q.rule_config};if(remove)delete config.region;else config.region=region;
 await api('/api/questions/'+q.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({...q,rule_config:config})});
 q.rule_config=config;current.questions=assessment.questions;regionSaved=true;selectSavedRegion();$('regionStatus').textContent=remove?'区域已移除':'区域已保存，下次识别复用；已保存作答保持原样';
}
action('saveRegion',()=>persistRegion(false));action('removeRegion',()=>persistRegion(true));
let trainingGrade=null;
function openTraining(g,response){trainingGrade=g;$('trainingText').value=response?.raw_text||'';$('trainingApprover').value='';$('trainingConsent').checked=false;$('trainingDialog').showModal();}
$('closeTraining').onclick=()=>{$('trainingDialog').close();};
action('approveTraining',async()=>{if(!$('trainingConsent').checked)throw Error('请先检查脱敏文本并勾选批准');await api('/api/grade-items/'+trainingGrade.id+'/training-approval',json({redacted_text:$('trainingText').value,approved_by:$('trainingApprover').value,expected_review_id:trainingGrade.latest_review_id||null}));$('trainingDialog').close();notice('已批准脱敏样本。标签来源由实际人工复核记录决定。');});
