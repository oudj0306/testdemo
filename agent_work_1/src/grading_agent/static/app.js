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
let current = null, selected = null, activeLayout = null, layoutBusy = false, questionPreparation = null;
let submissionRows = [], activeJobs = new Map(), pendingFiles = [], currentTab = 'paper', openRequest = 0;
let submissionHistory = [];
let assessmentRequest = 0;
let reportCharts = [];
let lastReport = null;
let reviewEvidence=null;
let viewedQuestionIndex = 0, rerenderCurrentQuestion = null, rerenderAnswerOverview = null;
const expandedPerformance=new Set(),performanceCache=new Map();
let batchControlsSupport=null,batchCapabilityPromise=null;
const pendingBatchControls=new Set();
const pageState = {};
const pg = key => { if(!pageState[key]) pageState[key]={page:1,size:10}; return pageState[key]; };
function pageSlice(items, key){ const st=pg(key); const totalPages=Math.max(1,Math.ceil(items.length/st.size)); if(st.page>totalPages)st.page=totalPages; return items.slice((st.page-1)*st.size,st.page*st.size); }
function sortList(list, key, dir, numeric){ const c=[...list]; c.sort((a,b)=>{ const av=a[key],bv=b[key]; const aNull=av==null||av===''; const bNull=bv==null||bv===''; if(aNull&&bNull)return 0; if(aNull)return 1; if(bNull)return -1; let r; if(numeric){ r=(Number(av)||0)-(Number(bv)||0); if(r===0)r=String(av).localeCompare(String(bv),'zh'); } else { r=String(av).localeCompare(String(bv),'zh'); } return dir==='desc'?-r:r; }); return c; }
function renderPager(containerId,key,total,unit='份'){const el=$(containerId);if(!el)return;if(!total){el.innerHTML='';return;}const st=pg(key),pages=Math.max(1,Math.ceil(total/st.size));st.page=Math.min(st.page,pages);el.innerHTML='<span class="page-count">共 '+total+' '+unit+' · '+st.page+' / '+pages+' 页</span><button class="secondary pg-prev" data-key="'+key+'" '+(st.page<=1?'disabled':'')+'>上一页</button><button class="secondary pg-next" data-key="'+key+'" '+(st.page>=pages?'disabled':'')+'>下一页</button><label>每页<select aria-label="每页显示数量" class="pg-size" data-key="'+key+'">'+[5,10,20,50].map(n=>'<option value="'+n+'" '+(n===st.size?'selected':'')+'>'+n+'</option>').join('')+'</select></label>';}
function onPagerChange(key){if(key.startsWith('answers:')){if(rerenderAnswerOverview)rerenderAnswerOverview();}else if(key==='report')renderReport().catch(e=>notice(e.message,true));else if(key==='token')renderTokenUsage().catch(e=>notice(e.message,true));else submissions().catch(e=>notice(e.message,true));}
document.addEventListener('click',e=>{const p=e.target.closest('.pg-prev');if(p){const st=pg(p.dataset.key);if(st.page>1){st.page--;onPagerChange(p.dataset.key);}}else{const n=e.target.closest('.pg-next');if(n){const st=pg(n.dataset.key);st.page++;onPagerChange(n.dataset.key);}}});
document.addEventListener('change',e=>{if(e.target.classList&&e.target.classList.contains('pg-size')){const st=pg(e.target.dataset.key);st.size=Number(e.target.value);st.page=1;onPagerChange(e.target.dataset.key);}});
let batchTimer = null;
let viewedSubmission = null;
const reviewReasons={quality_spot_check:'质量抽检',extraction_uncertain:'识别置信不足',ocr_confidence_too_low:'识别置信不足',score_disagreement:'多次评分有分歧',model_requested_review:'模型发现疑问',provisional_answer_key:'标准答案未确认',rubric_incomplete:'评分细则不完整',teacher_review_policy:'策略提示复核',insufficient_independent_passes:'评分次数不足',grading_evidence_missing:'缺少评分证据',answer_missing:'作答缺失',llm_request_failed:'模型调用失败'};
Object.assign(reviewReasons,{numeric_match:'数值匹配',numeric_mismatch:'数值不匹配',exact_match:'答案匹配',exact_mismatch:'答案不匹配',choice_set_match:'选项匹配',choice_set_mismatch:'选项不匹配',symbolic_match:'表达式等价',symbolic_mismatch:'表达式不等价',rubric_assisted:'按细则评分',confirmed_blank:'确认空白',blank_not_confirmed:'空白尚未确认',llm_not_configured:'模型未配置'});
Object.assign(reviewReasons,{fixed_answers_match:'逐空答案匹配',fixed_answers_partial:'逐空规则计分',correct:'答案正确',partial:'部分得分',incorrect:'答案错误',insufficient_evidence:'作答证据不足',ambiguous_answer:'作答有歧义',rubric_issue:'评分细则需检查',rule_fallback_to_model:'规则疑难已交模型判分'});

function fixedAnswerEditor(q){
 if(['single_choice','multiple_choice'].includes(q.question_type))return '';
 const rules=q.rule_config?.fixed_answers;
 const row=(b={accepted:[],score:''})=>'<div class="fixed-answer-row"><label>可接受答案（同义答案用竖线 | 分隔）<input data-fixed-accepted value="'+esc((b.accepted||[]).join(' | '))+'" placeholder="例如：存储器 | 内存储器"></label><label>本空分值<input data-fixed-score type="number" min="0.01" step="any" value="'+esc(b.score)+'"></label><button type="button" class="text-button" data-fixed-remove aria-label="删除此空">移除</button></div>';
 return '<details class="fixed-answer-editor"><summary>固定答案规则与识别方式'+(rules?' · 已配置':'')+'</summary><form data-fixed-question="'+esc(q.id)+'"><p class="muted small">按题目空序填写答案及分值，各空分值之和须等于 '+q.max_score+' 分。命中规则无需调用评分模型；其他答案默认交模型判断。不修改已有成绩。</p><label class="inline"><input type="checkbox" data-fixed-enabled '+(rules?'checked':'')+'> 启用逐空答案规则</label><div data-fixed-rows>'+(rules?.blanks||[{accepted:[],score:q.max_score}]).map(row).join('')+'</div><button type="button" class="secondary" data-fixed-add>＋ 添加一空</button><label class="inline"><input type="checkbox" data-fixed-strict '+(rules?.strict_mismatch?'checked':'')+'> 未列出的答案直接判错（仅适用于答案完全确定的题）</label><label>内容识别<select data-recognition-route><option value="auto">自动：本地识别；困难题和不可靠结果用 API</option><option value="api" '+(q.rule_config?.recognition_route==='api'?'selected':'')+'> 本题直接使用视觉 API</option></select></label><div class="toolbar"><button type="submit">保存本题规则</button><span data-fixed-status role="status" class="muted small"></span></div></form></details>';
}
function bindFixedAnswerEditors(){
 $('questions').querySelectorAll('[data-fixed-question]').forEach(form=>{
  form.querySelector('[data-fixed-add]').onclick=()=>{const source=document.createElement('div');source.innerHTML=fixedAnswerEditor({id:'row',question_type:'short_answer',max_score:'',rule_config:{}});form.querySelector('[data-fixed-rows]').append(source.querySelector('.fixed-answer-row'));};
  form.addEventListener('click',e=>{if(e.target.closest('[data-fixed-remove]'))e.target.closest('.fixed-answer-row').remove();});
  form.onsubmit=async e=>{e.preventDefault();const button=form.querySelector('[type=submit]'),status=form.querySelector('[data-fixed-status]');button.disabled=true;
   try{const health=await api('/api/health');if(!health.capabilities?.includes('fixed_answers_v1'))throw Error('当前服务还是旧版本，请在原服务窗口按 Ctrl+C，然后重新运行启动命令。');const q=current.questions.find(q=>q.id===form.dataset.fixedQuestion),config={...q.rule_config};
    if(form.querySelector('[data-fixed-enabled]').checked){config.fixed_answers={blanks:[...form.querySelectorAll('.fixed-answer-row')].map(row=>({accepted:row.querySelector('[data-fixed-accepted]').value.split('|').map(s=>s.trim()).filter(Boolean),score:Number(row.querySelector('[data-fixed-score]').value)})),strict_mismatch:form.querySelector('[data-fixed-strict]').checked};}else delete config.fixed_answers;
    config.recognition_route=form.querySelector('[data-recognition-route]').value;
    await api('/api/questions/'+q.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({...q,rule_config:config})});q.rule_config=config;
    status.textContent='已保存，仅用于后续识别与判分；已有成绩保留。';
   }catch(error){status.textContent=error.message;notice(error.message,true);}finally{button.disabled=false;}
  };
 });
}
function refreshReviewHint(){
 if(!viewedSubmission)return;
 const pending=current.questions.filter(q=>{const g=viewedSubmission.grade_items.find(x=>x.question_id===q.id);return !g||(g.needs_review&&g.final_score==null);}).length;
 $('reviewHint').textContent=$('reviewOnly').value==='all'?'全卷复核：可以逐题检查并修改当前成绩。':pending?'重点复核：系统建议关注 '+pending+' 题，其余成绩已直接生效。':'本答卷没有建议复核题；可切换为全卷复核。';
}
let reviewViewValue=$('reviewOnly').value;
$('reviewOnly').onchange=()=>{if(!mayLeaveDraft()){$('reviewOnly').value=reviewViewValue;return;}reviewViewValue=$('reviewOnly').value;if(rerenderCurrentQuestion)rerenderCurrentQuestion();};
async function api(path, options={}) {const r=await fetch(path,options); const body=await r.json().catch(()=>({})); if(!r.ok) throw Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||'请求失败')); return body;}
function notice(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':'';}
function action(id,fn){$(id).onclick=async()=>{const b=$(id);if(b.disabled)return;b.disabled=true;b.setAttribute('aria-busy','true');notice('正在处理，请稍候…');try{await fn();}catch(e){notice(e.message,true);}finally{b.disabled=false;b.removeAttribute('aria-busy');if(id==='generateLayout')updateLayoutEntry();if(id==='saveRegion')refreshRegionSaveButton();}};}
function draftValue(e){return e.type==='checkbox'?String(e.checked):e.value;}
function hasDraft(){return [...$('submissionBody').querySelectorAll('[data-initial]')].some(e=>draftValue(e)!==e.dataset.initial);}
function mayLeaveDraft(){if(!hasDraft())return true;if(!confirm('当前有未保存的校对或复核内容。确定放弃修改并继续？'))return false;$('submissionBody').querySelectorAll('[data-initial]').forEach(e=>{if(e.type==='checkbox')e.checked=e.dataset.initial==='true';else e.value=e.dataset.initial;});return true;}
window.addEventListener('beforeunload',e=>{if(hasDraft()){e.preventDefault();e.returnValue='';}});
function aid(){if(!$('assessment').value) throw Error('请先新建或选择试卷');return $('assessment').value;}
function requireSavedAnswers(){if(viewedSubmission&&[...document.querySelectorAll('[data-answer]')].some(t=>{const r=viewedSubmission.responses.find(r=>r.question_id===t.dataset.answer);return t.value!==(r?.raw_text||'')||Boolean($('answerBlank')?.checked)!==Boolean(r?.blank_confirmed);}))throw Error('请先确认作答并判分，再确认该题评分。');}
$('submissionBody').addEventListener('submit',e=>{try{requireSavedAnswers();}catch(error){e.preventDefault();e.stopImmediatePropagation();notice(error.message,true);}},true);
$('submissionBody').addEventListener('click',e=>{if(e.target.id!=='bulkConfirm')return;try{requireSavedAnswers();}catch(error){e.preventDefault();e.stopImmediatePropagation();notice(error.message,true);}},true);
function showPreview(kind,id){$('previewFrame').src='/api/preview/'+kind+'/'+id;$('preview').showModal();}
$('closePreview').onclick=()=>{$('preview').close();$('previewFrame').src='about:blank';};
const pageInfo={paper:['试卷与答案','ASSESSMENT LIBRARY','把标准准备好，让每一份答卷有据可依。'],upload:['答卷收集','SUBMISSION INBOX','先收集，后批阅。原件保留，重复文件自动去重。'],mark:['智能批阅','GRADING QUEUE','批阅完成即形成当前成绩；复核是可选纠错流程。'],review:['复核工作台','REVIEW DESK','选择重点复核或全卷复核，最新人工分数覆盖机器分。'],report:['成绩洞察','LEARNING INSIGHTS','从当前成绩出发，看见学习中的进步与薄弱点。'],token:['用量记录','MODEL USAGE','每一次模型调用，都有迹可查。'],settings:['模型与策略','PREFERENCES','连接你的模型，选择合适的评分与复核提示策略。'],runtime:['识别加速','OPTIONAL ACCELERATION','可选的后端识别能力，不是批阅的前置步骤。']};
function navigate(tab){if(tab!==currentTab&&!mayLeaveDraft())return false;if(tab==='review'&&currentTab!=='review'){$('reviewOnly').value='focused';reviewViewValue='focused';if(rerenderCurrentQuestion)rerenderCurrentQuestion();}currentTab=tab;document.querySelectorAll('.tab').forEach(s=>s.hidden=s.id!==tab);document.querySelectorAll('[data-tab]').forEach(n=>{n.classList.toggle('active',n.dataset.tab===tab);n.setAttribute('aria-current',n.dataset.tab===tab?'page':'false');});$('heading').textContent=pageInfo[tab][0];$('pageEyebrow').textContent=pageInfo[tab][1];$('pageDescription').textContent=pageInfo[tab][2];$('listTools').hidden=!['upload','mark','review'].includes(tab);return true;}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{if(!navigate(b.dataset.tab))return;if(b.dataset.tab==='report'&&current)$('refreshReport').click();if(b.dataset.tab==='token'&&current)$('refreshToken').click();});
$('newAssessment').onclick=()=>{$('createDialog').showModal();$('title').focus();};
$('closeCreate').onclick=()=>$('createDialog').close();
$('title').onkeydown=e=>{if(e.key==='Enter')$('create').click();};
$('studentSearch').oninput=()=>{['upload','mark','review'].forEach(k=>pg(k).page=1);renderSubmissions();};
$('goMark').onclick=()=>navigate('mark');
$('goReview').onclick=()=>navigate('review');
let reviewFilterValue=$('reviewFilter').value;
$('reviewFilter').onchange=()=>{if(!mayLeaveDraft()){$('reviewFilter').value=reviewFilterValue;return;}reviewFilterValue=$('reviewFilter').value;pg('review').page=1;const row=submissionRows.find(s=>s.id===selected);const inScope=row&&(reviewFilterValue==='pending'?(row.review_recommended_count>0||row.status==='review_required'):reviewFilterValue==='reviewed'?row.review_count>0:true);if(selected&&!inScope){reviewEvidence?.dispose();reviewEvidence=null;selected=null;viewedSubmission=null;submissionHistory=[];rerenderCurrentQuestion=null;openRequest++;$('studentTitle').textContent='选择一份答卷';$('submissionBody').innerHTML='<div class="empty">请选择当前范围内的答卷。</div>';}renderSubmissions();};
action('bulkReviewAll',async()=>{if(hasDraft())throw Error('当前题目有未保存修改，请先保存或放弃后再一键复核。');const id=aid();const e=await api('/api/assessments/'+id+'/review-eligible');if(!e.count)throw Error('当前没有可一键复核的题目（有异常或已复核的题仍需逐题处理）');if(!confirm('将批量确认 '+e.count+' 道有明确建议分、无识别/评分异常的题目，记录为人工复核。确定？'))return;await api('/api/reviews/bulk',json({grade_item_ids:e.grade_item_ids,reviewer_ref:'',reason:''}));await submissions();if(selected)await openSubmission(selected,{force:true});notice('已批量确认 '+e.count+' 道题；异常题仍需逐题处理。');});
async function assessments(id){const list=await api('/api/assessments');$('assessment').innerHTML=list.map(a=>'<option value="'+a.id+'">'+esc(a.title)+' · '+(a.status==='published'?'已发布':'草稿')+'</option>').join('');if(id)$('assessment').value=id;await loadAssessment();}
async function loadAssessment(){
 reviewEvidence?.dispose();reviewEvidence=null;
 const request=++assessmentRequest,id=$('assessment').value;openRequest++;rerenderCurrentQuestion=null;submissionRows=[];activeJobs.clear();$('studentSearch').value='';Object.values(pageState).forEach(s=>s.page=1);$('contextStats').innerHTML='';$('downloadReport').setAttribute('aria-disabled','true');
 viewedSubmission=null;
 clearTimeout(batchTimer);
 selected=null;current=null;activeLayout=null;pg('review').page=1;$('submissionBody').textContent='选择答卷查看详情。';$('studentTitle').textContent='选择一份答卷';$('reportBody').textContent='点击生成报告以读取最新成绩。';$('downloadReport').removeAttribute('href');
 if(!$('assessment').value)return;
 const [a,docs]=await Promise.all([api('/api/assessments/'+id),api('/api/assessments/'+id+'/documents')]);if(request!==assessmentRequest||$('assessment').value!==id)return;current=a;$('assessmentState').textContent=a.status==='published'?'已发布':'草稿';$('assessmentState').className='badge '+(a.status==='published'?'':'neutral');
 $('documents').innerHTML=docs.length?docs.map(d=>'<button data-doc="'+d.id+'">'+esc(({question_paper:'原试卷',answer_sheet:'答题卡模板／已填样张',answer_key:'答案册',rubric:'评分标准',reference:'参考资料'})[d.kind])+' · '+esc(d.media_type.includes('word')?'Word':d.media_type.split('/')[1])+'　查看原件</button>').join(''):'<p class="muted">还没有资料，请上传原试卷。</p>';
 document.querySelectorAll('[data-doc]').forEach(b=>b.onclick=()=>showPreview('document',b.dataset.doc));
 await loadLayoutStatus(docs);
 $('questions').innerHTML=a.questions.length?a.questions.map(q=>'<details class="question" open><summary>'+esc(layoutQuestionLabel(q))+' · '+q.max_score+' 分 <span class="badge">'+(q.answer_key_state==='missing'?'未配置答案':q.answer_key_state==='candidate'?'待确认':'已确认')+'</span></summary><p>'+esc(q.prompt)+'</p><h3>参考答案</h3><pre>'+esc(typeof q.answer_key==='string'?q.answer_key:JSON.stringify(q.answer_key))+'</pre><h3>评分细则</h3>'+q.rubric.map(r=>'<p>'+esc(r.criterion)+' <b>'+esc(r.score)+' 分</b></p>').join('')+(q.rubric.length?'':'<p class="muted">根据标准答案及题目规则评分。</p>')+fixedAnswerEditor(q)+'</details>').join(''):'<p class="empty">先上传原试卷并识别题目；参考答案在独立步骤配置。</p>';
 bindFixedAnswerEditors();
 setupRegionEditor(docs);
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
 document.querySelector('#review .workbench')?.classList.toggle('with-evidence',Boolean(selected));
 const all=submissionRows,term=$('studentSearch').value.trim().toLocaleLowerCase();
 let rows=all.filter(s=>s.student_ref.toLocaleLowerCase().includes(term));
 const mode=$('sortMode').value;
 rows=sortList(rows,mode.startsWith('name')?'student_ref':'created_at',mode.endsWith('asc')?'asc':'desc');
 const pending=all.filter(s=>s.review_recommended_count>0||['review_required','uploaded','extracted','extraction_failed'].includes(s.status)).length;
 $('contextStats').innerHTML='<span><b>'+all.length+'</b>份答卷</span><span><b>'+all.filter(s=>['uploaded','extracted','extraction_failed'].includes(s.status)).length+'</b>待批阅</span><span><b>'+pending+'</b>建议复核</span>';
 $('reviewNavCount').hidden=!pending;$('reviewNavCount').textContent=pending;
 const empty='<div class="empty"><strong>'+ (term?'没有匹配的答卷':'还没有答卷')+'</strong>'+(term?'换个姓名或文件名试试。':'先在答卷收集页上传文件。')+'</div>';
 const state=s=>activeJobs.has(s.id)?({queued:'排队中',extracting:'识别中',grading:'批阅中',paused:'已暂停',stopped:'已停止 · 可继续'}[activeJobs.get(s.id)]||'处理中'):(s.status==='finalized'&&s.review_recommended_count>0?'已出分 · 建议复核':statusText[s.status]||s.status);
 const badge=s=>'<span class="badge '+(s.status==='review_required'||s.status==='extraction_failed'?'warn':s.status==='finalized'?'':'neutral')+'">'+esc(state(s))+'</span>';
 const meta=s=>'<span class="avatar" aria-hidden="true">'+esc(s.student_ref.slice(0,1))+'</span><div class="student-meta"><strong>'+esc(s.student_ref)+'</strong><small>'+esc(new Date(s.created_at).toLocaleString())+(s.normalization_report?.answer_layout==='on_paper'?' · 卷面作答':s.normalization_report?.answer_layout==='answer_list'?' · 仅答案列表':'')+'</small></div>';
 $('uploadedList').innerHTML=pageSlice(rows,'upload').map(s=>'<div class="submission-row">'+meta(s)+badge(s)+'<div class="row-actions"><button class="secondary" data-source="'+s.id+'">预览</button><button class="text-button danger" data-del="'+s.id+'" '+(activeJobs.has(s.id)?'disabled title="处理中不能删除"':'')+'>删除</button></div></div>').join('')||empty;
 renderPager('uploadPager','upload',rows.length);
 $('gradingList').innerHTML=pageSlice(rows,'mark').map(s=>'<div class="submission-row">'+meta(s)+badge(s)+'<div class="row-actions">'+('<button class="secondary" data-result="'+s.id+'">作答校对 / 复核</button>')+(['extraction_failed','review_required'].includes(s.status)?'<button class="text-button" data-retry="'+s.id+'" '+(activeJobs.has(s.id)?'disabled':'')+'>重试缺分题</button>':'')+'</div></div>').join('')||empty;
 renderPager('markPager','mark',rows.length);
 const filter=$('reviewFilter').value;
 const visible=rows.filter(s=>filter==='pending'?(s.review_recommended_count>0||['review_required','uploaded','extracted','extraction_failed'].includes(s.status)):filter==='reviewed'?s.review_count>0:true);
 $('submissions').innerHTML=pageSlice(visible,'review').map(s=>'<button data-sub="'+s.id+'" class="'+(selected===s.id?'selected':'')+'">'+esc(s.student_ref)+'<small>'+esc(state(s))+' · '+(s.review_recommended_count||0)+' 题建议关注 · '+(s.review_count||0)+' 次复核</small></button>').join('')||'<p class="muted small">当前范围没有答卷。可切换“全部答卷”。</p>';
 renderPager('reviewPager','review',visible.length);
 document.querySelectorAll('#submissions [data-sub]').forEach(b=>b.onclick=()=>{if(!navigate('review'))return;openSubmission(b.dataset.sub).catch(e=>notice(e.message,true));});
 document.querySelectorAll('[data-result]').forEach(b=>b.onclick=()=>{
  if(!navigate('review'))return;
  $('reviewFilter').value='all';reviewFilterValue='all';
  $('reviewOnly').value='focused';reviewViewValue='focused';
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
 try{await api('/api/submissions/'+id,{method:'DELETE'});if(selected===id){reviewEvidence?.dispose();reviewEvidence=null;selected=null;viewedSubmission=null;rerenderCurrentQuestion=null;openRequest++;$('submissionBody').innerHTML='<div class="empty">答卷已删除，请选择其他答卷。</div>';$('studentTitle').textContent='选择一份答卷';}await submissions();notice('答卷记录已删除。');}catch(e){notice(e.message,true);button.disabled=false;}
}
function answerCandidate(submission,qid){
 const report=submission.normalization_report?.local_recognition||{};
 return report.api_fallback?.[qid]?.candidate?.raw_text??(report.recognitions?.[qid]?.parts||[]).map(p=>p.text||'').join('\n');
}
function answerRecognitionState(submission,q){
 const r=submission.responses.find(r=>r.question_id===q.id);
 if(r?.extraction_source==='manual'&&(r.raw_text?.trim()||r.blank_confirmed))return r.blank_confirmed?'人工确认未作答':'人工确认作答';
 const fallback=submission.normalization_report?.local_recognition?.api_fallback?.[q.id];
 if(r?.extraction_source==='api_vision_fallback')return r.blank_confirmed?'API补识 · 确认空白':r.raw_text?.trim()?'API补识 · 已采用':'API补识 · 仍不确定';
 if(fallback?.attempted)return fallback.error?'API补识 · 调用失败':'API补识 · 仍不确定';
 if(submission.normalization_report?.local_recognition?.missing?.includes(q.question_no))return answerCandidate(submission,q.id)?'本地候选 · 待API补识':'等待自动API补识';
 if(r?.blank_confirmed)return '自动识别未作答';
 if(r?.raw_text?.trim())return r.extraction_source==='api_vision_fallback'?'API补识已采用':'已自动采用';
 return answerCandidate(submission,q.id)?'本地候选 · 待API补识':'等待自动API补识';
}
function reviewQuestionOrder(questions){return [...questions].sort((a,b)=>a.question_no.localeCompare(b.question_no,'zh',{numeric:true}));}
function answerOverviewHtml(submission,questions,items=questions.map((q,i)=>({q,i})),expanded=true){
 return '<details class="answer-overview" '+(expanded?'open':'')+'><summary><span>作答校对 <small>'+questions.length+' 题 · 含未识别题</small></span><span class="overview-toggle">展开 / 收起</span></summary><div class="table-scroll"><table><thead><tr><th>题号</th><th>识别状态</th><th>作答 / 待核对候选</th><th>操作</th></tr></thead><tbody>'+items.map(({q,i})=>{
  const r=submission.responses.find(r=>r.question_id===q.id),text=r?.blank_confirmed?'[未作答]':r?.raw_text||answerCandidate(submission,q.id)||(submission.normalization_report?.local_recognition?.api_fallback?.[q.id]?.error?'API调用失败，系统未取得转写':submission.normalization_report?.local_recognition?.api_fallback?.[q.id]?.attempted?'API整页补识后未返回有效转写':'等待自动API补识');
  return '<tr><td>'+esc(q.question_no)+'</td><td><span class="recognition-tag '+(r?.extraction_source==='api_vision_fallback'?'api':r?.extraction_source==='manual'?'manual':'')+'">'+esc(answerRecognitionState(submission,q))+'</span></td><td class="answer-overview-text">'+esc(text.slice(0,180))+(text.length>180?'…':'')+'</td><td><button class="secondary" data-answer-question="'+i+'">校对本题</button></td></tr>';
 }).join('')+'</tbody></table></div><div id="answerOverviewPager" class="pagination"></div></details>';
}

async function openSubmission(id,{force=false}={}){
 if(!force&&!mayLeaveDraft())return;
 const request=++openRequest,assessmentId=aid();
 const [s,history]=await Promise.all([api('/api/submissions/'+id),api('/api/submissions/'+id+'/history')]);
 if(request!==openRequest||aid()!==assessmentId)return;
 if(s.assessment_id!==assessmentId)throw Error('答卷与当前试卷不一致，请重新选择。');
 if(selected!==id){viewedQuestionIndex=0;$('reviewOnly').value='focused';reviewViewValue='focused';}
 selected=id;viewedSubmission=s;submissionHistory=history;
 $('studentTitle').textContent=s.student_ref;
 reviewEvidence?.dispose();
 const questions=reviewQuestionOrder(current.questions),summary=s.score_summary||{},currentScore=currentSubmissionScore(s);
 let nav=[];
 const navigable=()=>questions.map((q,i)=>i).filter(i=>{const g=s.grade_items.find(g=>g.question_id===questions[i].id);return $('reviewOnly').value==='all'||!g||(g.needs_review&&g.final_score==null);});
 $('submissionBody').className='';
 $('submissionBody').innerHTML=localRecognitionHtml(s)+'<div class="toolbar review-actions"><button id="viewSource" class="secondary">查看答卷原件 ↗</button><button id="deleteSubmission" class="text-button danger">删除答卷</button></div><div class="metrics"><div class="metric"><b>'+fmt(currentScore)+'</b>当前成绩</div><div class="metric"><b>'+fmt(summary.max_score)+'</b>试卷满分</div><div class="metric"><b>'+fmt(questions.filter(q=>{const g=s.grade_items.find(g=>g.question_id===q.id);return !g||(g.needs_review&&g.final_score==null);}).length)+'</b>重点关注</div></div><p class="helper-note">未出分 '+questions.filter(q=>!s.grade_items.some(g=>g.question_id===q.id&&(g.score!=null||g.final_score!=null))).length+' / '+questions.length+' 题；未识别题不会自动当作0分。</p><div id="bulkArea"></div><div class="question-nav"><button id="prevQ" class="secondary">← 上一题</button><select id="questionSelect" aria-label="选择题号"></select><button id="nextQ" class="secondary">下一题 →</button></div><div id="reviewQuestionWorkspace" class="review-question-workspace"><aside id="reviewEvidence" class="review-evidence" aria-label="学生作答原图"></aside><div id="questionArea"></div></div><div id="answerOverview"></div><details id="reviewAudit" class="review-audit"><summary>查看批改与复核历史（'+history.reduce((n,g)=>n+g.reviews.length,0)+' 次复核）</summary><div id="auditBody"></div></details>';
 reviewEvidence=createReviewEvidence(id,$('reviewEvidence'),{onOpenOriginal:()=>showPreview('submission',id)});
 $('viewSource').onclick=()=>showPreview('submission',id);
 $('deleteSubmission').disabled=activeJobs.has(id);
 $('deleteSubmission').onclick=()=>removeSubmission(id,$('deleteSubmission'));
 $('auditBody').innerHTML=history.map(g=>'<details class="trace"><summary>第 '+esc(g.snapshot?.question?.question_no||'?')+' 题 · 机器建议 '+fmt(g.score)+' 分 · '+(g.superseded_at?'历史版本':'当前版本')+'</summary><p>'+esc(new Date(g.created_at).toLocaleString())+' · '+esc(g.grader_kind)+' / '+esc(g.grader_version)+'</p><p>当时作答：'+esc(g.snapshot?.response?.raw_text??'未保存快照')+'</p><p>当时标准：'+esc(typeof g.snapshot?.question?.answer_key==='object'?JSON.stringify(g.snapshot.question.answer_key):g.snapshot?.question?.answer_key??'未保存快照')+'</p><pre>'+esc(g.evidence.join('\n'))+'</pre>'+g.reviews.map((rev,i)=>'<p><b>复核 '+(i+1)+' · '+fmt(rev.final_score)+' 分</b> · '+esc(rev.reviewer_ref||'人工复核')+'<br>'+esc(rev.reason||'确认建议分')+'<br><span class="muted small">'+esc(new Date(rev.created_at).toLocaleString())+'</span></p>').join('')+'</details>').join('')||'<p class="muted small">尚无批改记录。</p>';
 function renderQuestion(){
  const q=questions[viewedQuestionIndex];
  if(!q||!nav.length){reviewEvidence.clear();$('reviewEvidence').hidden=true;$('reviewQuestionWorkspace').classList.add('no-question');$('questionArea').innerHTML='<div class="empty"><strong>当前没有建议复核题</strong>切换为“全卷复核”可以查看并调整任意题成绩。</div>';return;}
  $('reviewEvidence').hidden=false;$('reviewQuestionWorkspace').classList.remove('no-question');reviewEvidence.show(q);
  const r=s.responses.find(x=>x.question_id===q.id),candidate=answerCandidate(s,q.id),g=s.grade_items.find(x=>x.question_id===q.id),reference=g?.snapshot?.question||q;
  const answer=typeof reference.answer_key==='string'?reference.answer_key:JSON.stringify(reference.answer_key,null,2);
  const needs=!g||(g.needs_review&&g.final_score==null);
  const reasons=g?(needs?g.reason_codes.map(code=>reviewReasons[code]||code).join(' / '):g.final_score!=null?'已人工确认 · 可再次调整':'自动通过 · 非人工复核'):'尚未批阅';
  $('questionArea').innerHTML='<div class="question"><div class="question-title-line"><b>第 '+esc(q.question_no)+' 题</b><span class="recognition-tag '+(r?.extraction_source==='api_vision_fallback'?'api':r?.extraction_source==='manual'?'manual':'')+'">'+esc(answerRecognitionState(s,q))+'</span></div><p>'+esc(reference.prompt)+'</p><div class="answer-comparison"><div><label>学生作答<textarea data-answer="'+q.id+'" aria-label="学生作答校对">'+esc(r?.blank_confirmed?'':r?.raw_text||candidate||'')+'</textarea></label><p class="muted small">'+esc(answerRecognitionState(s,q))+'。未识别题也可直接输入；候选文字须核对。</p><label class="inline"><input id="answerBlank" type="checkbox" '+(r?.blank_confirmed?'checked':'')+'> 确认本题未作答</label><div class="toolbar answer-actions"><button id="saveAnswers">确认作答并判分</button><button id="confirmBlankAnswer" class="secondary">确认未作答并记0分</button></div><p id="answerActionStatus" role="status" class="muted small">本地不可靠题自动补调 API 并标记来源；此处只需校对或确认空白。</p></div><div><h3>参考答案'+(g?.snapshot?.question?' · 批改时版本':'')+'</h3><pre>'+esc(answer||'未提供参考答案')+'</pre><h3>评分标准</h3>'+(reference.rubric?.length?reference.rubric.map(rr=>'<p>'+esc(rr.criterion)+' <b>'+esc(rr.score)+' 分</b></p>').join(''):'<p class="muted">按标准答案和题目规则评分</p>')+'</div></div><p class="'+(needs?'pending':'score')+'">'+(g?'当前得分 '+fmt(g.effective_score)+' / '+g.max_score+(needs?' · 建议复核':' · 已出分'):'尚未批阅')+'</p><p class="muted small">'+esc(reasons)+'</p>'+(g?'<details><summary>展开评分依据</summary><pre>'+esc(g.evidence.join('\n'))+'</pre></details><form data-review="'+g.id+'"><div class="toolbar"><label>复核分数<input name="score" aria-label="复核分数" type="number" min="0" max="'+g.max_score+'" step="any" required></label><label style="flex:1">复核说明<input name="reason" aria-label="复核说明" placeholder="可选，建议改分时填写" style="width:100%" maxlength="1000"></label><button type="submit" class="secondary">'+(g.latest_review_id?'保存再次复核':'确认本题评分')+'</button><button type="submit" data-review-next>保存并下一题 →</button></div></form>':'<p class="helper-note">可直接填写或确认未作答，再点击确认作答并判分，无需先批阅整卷。</p>')+'</div>';
  if(g){
   const area=$('questionArea'),panel=document.createElement('section'),score=area.querySelector('.score,.pending');
   panel.className='review-decision-panel';panel.setAttribute('aria-label','本题评分与复核操作');
   const reason=score?.nextElementSibling,reviewForm=area.querySelector('form');
   if(score)panel.append(score);if(reason)panel.append(reason);if(reviewForm)panel.append(reviewForm);
   area.querySelector('.answer-comparison').before(panel);
   area.querySelector('[name=score]').value=g.effective_score??'';
   if(g.score_state==='AUTO_SINGLE'){const label=document.createElement('p');label.textContent='日常作业 · 单次自动评分（非双评一致）';panel.prepend(label);}
  }
  const input=$('questionArea').querySelector('[data-answer]'),blank=$('answerBlank');
  let previousText=input.value;input.disabled=blank.checked;
  blank.onchange=()=>{if(blank.checked){previousText=input.value;input.value='';}else input.value=previousText;input.disabled=blank.checked;};
  $('questionArea').querySelectorAll('input,textarea').forEach(e=>e.dataset.initial=draftValue(e));
  async function confirmAnswer(isBlank){
   if(!isBlank&&!input.value.trim())throw Error('请输入学生作答，或点击“确认未作答并记0分”。');
   if(r&&Boolean(r.blank_confirmed)===isBlank&&(isBlank||r.raw_text===input.value)&&(g?.score!=null||g?.final_score!=null)){notice('作答未修改，已有分数；无需重复识别或判分。可直接确认评分。');return;}
   const controls=[...$('questionArea').querySelectorAll('input,textarea,button')];controls.forEach(e=>e.disabled=true);
   try{
    const updated=await api('/api/submissions/'+id+'/questions/'+q.id+'/confirm-answer',json({raw_text:isBlank?'':input.value,blank_confirmed:isBlank}));
    const graded=updated.grade_items.find(item=>item.question_id===q.id);
    if(selected===id&&aid()===assessmentId)await openSubmission(id,{force:true});
    notice(isBlank?'已人工确认未作答，本题按规则记0分。':graded?.score==null?'作答已确认；本题尚未出分，请检查评分错误或人工给分。':'作答已人工确认并完成本题判分；其他题成绩保留。');
   }finally{controls.forEach(e=>e.disabled=false);input.disabled=blank.checked;}
  }
  action('saveAnswers',async()=>confirmAnswer(blank.checked));
  action('confirmBlankAnswer',async()=>confirmAnswer(true));
  const form=$('questionArea').querySelector('form');
  if(form)form.onsubmit=async e=>{
   e.preventDefault();const buttons=[...form.querySelectorAll('button')];if(buttons.some(b=>b.disabled))return;buttons.forEach(b=>b.disabled=true);const moveNext=Boolean(e.submitter?.hasAttribute('data-review-next')),nextIndex=nav[nav.indexOf(viewedQuestionIndex)+1];
   try{requireSavedAnswers();const data=new FormData(form);await api('/api/grade-items/'+g.id+'/review',json({reviewer_ref:'',final_score:Number(data.get('score')),reason:data.get('reason')||'',check_revision:true,expected_review_id:g.latest_review_id}));if(moveNext&&nextIndex!==undefined)viewedQuestionIndex=nextIndex;await openSubmission(id,{force:true});notice('复核已保存。旧记录保留，成绩洞察采用最新分数。');}
   catch(error){notice(error.message,true);buttons.forEach(b=>b.disabled=false);}
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
 let overviewExpanded=$('reviewOnly').value==='all';
 function renderAnswerOverview(){
  const key='answers:'+id,items=pageSlice(questions.map((q,i)=>({q,i})),key);
  $('answerOverview').innerHTML=answerOverviewHtml(s,questions,items,overviewExpanded);
  renderPager('answerOverviewPager',key,questions.length,'题');
  $('answerOverview').querySelector('details').ontoggle=e=>{overviewExpanded=e.currentTarget.open;};
  $('answerOverview').querySelectorAll('[data-answer-question]').forEach(button=>button.onclick=()=>{if(!mayLeaveDraft())return;$('reviewOnly').value='all';reviewViewValue='all';viewedQuestionIndex=Number(button.dataset.answerQuestion);renderNavigation();$('questionArea').scrollIntoView({block:'start',behavior:'smooth'});});
 }
 rerenderAnswerOverview=renderAnswerOverview;renderAnswerOverview();
 renderNavigation();await submissions();
}
action('create',async()=>{if(!$('title').value.trim())throw Error('请输入试卷名称');if(!mayLeaveDraft())return;const a=await api('/api/assessments',json({title:$('title').value.trim()}));$('createDialog').close();$('title').value='';await assessments(a.id);navigate('paper');notice('试卷已创建，先保存资料，再准备答案与评分标准。');});
['paperFiles','answerFiles'].forEach(id=>$(id).accept='.doc,.docx,.pdf,.jpg,.jpeg,.png,.webp');
action('uploadPaper',async()=>{if(!$('paperFiles').files.length)throw Error('请选择资料');const f=new FormData();f.append('kind',$('kind').value);[...$('paperFiles').files].forEach(x=>f.append('files',x));await api('/api/assessments/'+aid()+'/documents',{method:'POST',body:f});await loadAssessment();notice('资料已入库，可点击预览。');});
action('extractQuestions',async()=>{await ensureAssessmentQuestions(aid());notice('题目清单已提取，未生成参考答案。请核对题号、题型及分值。');});
action('generate',async()=>{await ensureAssessmentQuestions(aid());const r=await api('/api/assessments/'+aid()+'/fill-missing',{method:'POST'});await loadAssessment();notice(r.filled?'候选答案已生成，请核对后确认。':'已有答案保持不变，没有缺失项需要生成。');});
action('fillMissing',async()=>{const id=aid();const r=await api('/api/assessments/'+id+'/fill-missing',{method:'POST'});if(!r.filled){notice('没有缺失的答案或评分标准，无需补全。');return;}await loadAssessment();notice('已补全 '+r.filled+' 道题的缺失内容，请展开检查并确认。');});
action('confirm',async()=>{if(!confirm('确认已检查每题答案与评分细则？'))return;await api('/api/assessments/'+aid()+'/confirm-candidate-key',{method:'POST'});await loadAssessment();notice('答案已确认。');});
action('publish',async()=>{const id=aid();await api('/api/assessments/'+id+'/publish',{method:'POST'});await assessments(id);notice('试卷已发布，可以上传答卷。');});
function batchControlsHtml(b){
 if(!['processing','paused','stopped'].includes(b.status))return '';
 if(batchControlsSupport===false)return '<div class="batch-controls batch-control-unavailable" role="status"><p><b>当前服务尚未载入暂停、停止接口</b><br>请在原服务终端按 Ctrl+C，重新运行启动命令，再刷新网页。当前批阅仍按原流程运行。</p></div>';
 const button=(action,label,kind='secondary')=>'<button class="'+kind+'" data-batch-action="'+action+'" data-batch-id="'+b.id+'" '+(b.settling||pendingBatchControls.has(b.id)?'disabled':'')+'>'+label+'</button>';
 return '<div class="batch-controls"><p>'+(b.settling?'正在保存当前题目结果，完成后即可继续。':b.status==='processing'?'暂停或停止会保留进度，当前题目完成后生效。':'已保存识别与评分进度，继续时跳过已完成题目。')+'</p><div>'+(b.status==='processing'?button('pause','暂停批阅')+button('stop','停止批阅','text-button'):button('resume',b.status==='stopped'?'从停止处继续':'继续批阅')+(b.status==='paused'?button('stop','停止批阅','text-button'):''))+'</div><div class="batch-control-feedback" role="status" aria-live="polite"></div></div>';
}
function batchCardHtml(b){
 const active=b.status==='processing',label=b.settling?(b.status==='paused'?'正在暂停':'正在停止'):(b.status==='paused'?'已暂停':b.status==='stopped'?'已停止 · 进度保留':active?'正在处理':'已结束');
 return '<section class="batch-progress"><div class="batch-heading"><div><span class="batch-eyebrow">批阅批次</span><time>'+esc(new Date(b.created_at).toLocaleString())+'</time></div><span class="badge '+(active?'':'neutral')+'">'+label+'</span></div><progress aria-label="批阅进度" value="'+(b.success_count+b.failed_count)+'" max="'+Math.max(1,b.total_count)+'"></progress><div class="batch-counts"><strong>'+b.success_count+' <span>/ '+b.total_count+' 份已完成</span></strong><span>'+(b.duplicate_count?b.duplicate_count+' 份重复跳过':'无重复答卷')+(b.failed_count?' · '+b.failed_count+' 份失败':'')+'</span></div>'+(b.question_count?'<p class="batch-question-progress">已保存评分 <b>'+(b.saved_question_count||0)+'</b> / '+b.question_count+' 题</p>':'')+batchControlsHtml(b)+'<details class="batch-usage" data-batch-usage="'+b.id+'" '+(expandedPerformance.has(b.id)?'open':'')+'><summary><span>耗时与用量</span><span class="usage-disclosure">查看明细 <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 3 5 5-5 5"/></svg></span></summary><div class="batch-usage-panel" role="status">正在读取本批次统计…</div></details>'+(b.failed_count?'<details class="batch-errors"><summary>查看失败原因</summary>'+b.items.filter(i=>i.error).map(i=>'<p class="batch-error">'+esc(i.filename)+'：'+esc(i.error)+'</p>').join('')+'</details>':'')+'</section>';
}
function performanceHtml(p){
 const metrics=[['总耗时',(p.wall_ms/1000).toFixed(2)+' 秒'],['模型请求',p.requests+' 次'],['输入 token',p.tokens.input_tokens],['输出 token',p.tokens.output_tokens],['缓存 token',p.tokens.cached_tokens],['失败 / 补请求',p.failed_requests+' / '+p.retries]];
 return '<div class="usage-metrics">'+metrics.map(([label,value])=>'<div><span>'+esc(label)+'</span><strong>'+esc(value)+'</strong></div>').join('')+'</div><p class="usage-note">仅统计实际请求用量；总耗时包含暂停等待，未配置价格时不估算费用。</p>';
}
async function ensureBatchControls(){
 if(!batchCapabilityPromise)batchCapabilityPromise=(async()=>{
  try{const spec=await api('/openapi.json');batchControlsSupport=Boolean(spec.paths?.['/api/batches/{batch_id}/{action}']?.post);}
  catch(e){batchCapabilityPromise=null;throw Error('无法核对批阅控制接口，请检查服务连接：'+e.message);}
 })();
 await batchCapabilityPromise;
}
async function handleBatchControlClick(event){
 const button=event.target.closest('[data-batch-action]');if(!button||button.disabled)return;
 const batchId=button.dataset.batchId,action=button.dataset.batchAction,id=aid();
 if(pendingBatchControls.has(batchId))return;
 pendingBatchControls.add(batchId);clearTimeout(batchTimer);
 const card=button.closest('.batch-progress'),feedback=card.querySelector('.batch-control-feedback');
 const buttons=[...card.querySelectorAll('[data-batch-action]')];buttons.forEach(b=>b.disabled=true);
 feedback.textContent='正在发送'+(action==='pause'?'暂停':action==='stop'?'停止':'继续')+'请求…';
 try{
  await api('/api/batches/'+batchId+'/'+action,{method:'POST'});performanceCache.delete(batchId);
  feedback.textContent=action==='resume'?'已收到请求，正在继续批阅。':'请求已收到，正在保存当前题目结果。';
  pendingBatchControls.delete(batchId);await refreshBatches(id);
  notice(action==='resume'?'已从保存处继续批阅，已完成题目不会重批。':'已收到'+(action==='pause'?'暂停':'停止')+'请求；当前题目完成并保存后生效。');
 }catch(e){
  const message=e.message==='Not Found'?'当前服务没有载入批阅控制接口，请在原服务终端按 Ctrl+C 后重新启动，再刷新网页。':e.message;
  feedback.textContent=message;feedback.classList.add('error');notice(message,true);buttons.forEach(b=>b.disabled=false);
 }finally{
  pendingBatchControls.delete(batchId);
  if(!$('batchResults').contains(button))return;
  batchTimer=setTimeout(()=>refreshBatches(id),2500);
 }
}
$('batchResults').addEventListener('click',handleBatchControlClick);
async function refreshBatches(id){
 clearTimeout(batchTimer);
 if(pendingBatchControls.size){batchTimer=setTimeout(()=>refreshBatches(id),500);return;}
 try{
  const [rows]=await Promise.all([api('/api/batches?assessment_id='+encodeURIComponent(id)),ensureBatchControls()]);
  if($('assessment').value!==id)return;
  activeJobs=new Map(rows.flatMap(b=>b.items.map(i=>({...i,displayStatus:['paused','stopped'].includes(b.status)?b.status:i.status}))).filter(i=>['queued','extracting','grading'].includes(i.status)&&i.submission_id).map(i=>[i.submission_id,i.displayStatus]));
  const active=rows.filter(b=>b.status==='processing'),held=rows.filter(b=>['paused','stopped'].includes(b.status)),settling=held.some(b=>b.settling);
  const runnable=active.reduce((n,b)=>n+b.items.filter(i=>['queued','extracting','grading'].includes(i.status)).length,0);
  $('queueState').textContent=active.length?runnable+' 份处理中':held.length?held.length+' 批已暂停或停止 · 进度保留':'没有运行中的任务';
  $('queueState').className='badge '+(active.length?'':'neutral');
  const outstanding=rows.filter(b=>['processing','paused','stopped'].includes(b.status));
  const shown=outstanding.length?outstanding:rows.slice(0,1);
  $('batchResults').innerHTML=shown.map(batchCardHtml).join('');
  await submissions();
  document.querySelectorAll('[data-batch-usage]').forEach(details=>{
   const batch=shown.find(b=>b.id===details.dataset.batchUsage);
   let loading=false;
   async function loadPerformance(){
    if(!details.open||loading)return;loading=true;
    expandedPerformance.add(batch.id);
    const panel=details.querySelector('.batch-usage-panel');
    try{
     const cached=performanceCache.get(batch.id),data=cached||await api('/api/batches/'+batch.id+'/performance');
     if(!details.isConnected||$('assessment').value!==id)return;
     if(!['processing','paused','stopped'].includes(batch.status))performanceCache.set(batch.id,data);
     panel.innerHTML=performanceHtml(data);
    }catch(e){if(details.isConnected)panel.textContent='用量读取失败：'+e.message+'。收起后展开可重试。';}finally{loading=false;}
   }
   details.ontoggle=()=>{if(details.open)loadPerformance();else expandedPerformance.delete(batch.id);};
   if(details.open)loadPerformance();
  });
  if(active.length||settling)batchTimer=setTimeout(()=>refreshBatches(id),2500);
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
 $('reportBody').querySelectorAll('[data-report-sub]').forEach(b=>b.onclick=async()=>{try{$('reviewOnly').value='focused';reviewViewValue='focused';$('reviewFilter').value='all';reviewFilterValue='all';navigate('review');await openSubmission(b.dataset.reportSub);}catch(e){notice(e.message,true);}});
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
 const stageLabels={extract_fallback:'疑难题自动补识',extract_preview:'识别预览',extract:'抽取答案',grade:'过程题批改',compile:'生成候选答案',structure:'提取题目结构',layout:'匹配答题卡位置',answer_key:'生成／读取参考答案'};
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
async function refreshHealth(){const h=await api('/api/health');const warning=$('serviceVersionWarning');if(warning){warning.hidden=h.app_build==='20261028-routing-rules-1';warning.textContent=h.app_build?'页面与服务版本不同，请重启服务并刷新页面后使用新功能。':'服务仍为旧版本。请在原服务窗口按 Ctrl+C，再重新启动；新规则和识别路由须重启后生效。';}$('health').textContent=h.llm_configured?'模型已配置':'模型待配置';const r=h.ocr_runtime;$('gpuStatus').textContent=!r.paddle_installed||!r.paddleocr_installed?'当前服务使用的 Python 未装齐 OCR。请关闭服务，用 start-optimization.ps1 重新启动；它会优先选择原有 OCR 环境。':r.active_device?'最近实际识别设备：'+r.active_device:'后端 OCR 包已安装，可选择检查设备，也可以跳过。';$('gpuStatus').textContent+=' 当前服务环境：'+(r.python_executable||'未知')+'；OCR 选项：'+(h.ocr_provider==='paddleocr'?'已启用':'未启用')+'。';if(Object.keys(r.errors).length)$('gpuStatus').textContent+=' 错误：'+Object.values(r.errors).join('；')+'（排除故障后重启服务）';if(r.compatibility_warning)$('gpuStatus').textContent+=' '+r.compatibility_warning;}
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
 $('hybridFallback').checked=s.hybrid_fallback_enabled!==false;$('gradingProfile').value=s.grading_profile||'legacy';$('llmPasses').value=String(s.llm_grading_passes);$('llmExtract').checked=s.llm_extract_answers;$('llmThinking').checked=s.enable_thinking;$('llmKey').value='';$('settingsStatus').textContent=s.api_key_set?'当前密钥：'+s.api_key_masked:'尚未配置密钥，过程题与 AI 生成答案将不可用。';$('ocrProvider').value=s.ocr_provider;$('ocrDevice').value=s.ocr_device;$('localOcrFirst').checked=s.local_ocr_first;$('reviewMode').value=s.review_mode;$('sampleRate').value=String(Math.round(s.review_sample_rate*100));$('confidenceThreshold').value=s.review_confidence_threshold;}
action('saveSettings',async()=>{const key=$('llmKey').value.trim(),connection=selectedModelConnection();await api('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key:key||null,...connection,hybrid_fallback_enabled:$('hybridFallback').checked,grading_profile:$('gradingProfile').value,llm_grading_passes:Number($('llmPasses').value),llm_extract_answers:$('llmExtract').checked,enable_thinking:$('llmThinking').checked,review_mode:$('reviewMode').value,review_sample_rate:Number($('sampleRate').value)/100,review_confidence_threshold:Number($('confidenceThreshold').value)})});await loadSettings();await refreshHealth();$('llmTestResult').textContent='设置已保存。可测试连接后再批阅。';notice('模型与复核策略已保存；仅影响后续批改，不自动改动已有成绩。');});
action('testLlm',async()=>{const result=await api('/api/settings/test-connection',{method:'POST'});$('llmTestResult').textContent=result.message;notice(result.message);});
action('saveRuntime',async()=>{const saved=await api('/api/settings');await api('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:saved.model,base_url:saved.base_url,llm_grading_passes:saved.llm_grading_passes,llm_extract_answers:saved.llm_extract_answers,enable_thinking:saved.enable_thinking,ocr_provider:$('ocrProvider').value,ocr_device:$('ocrDevice').value,local_ocr_first:$('localOcrFirst').checked,hybrid_fallback_enabled:$('hybridFallback').checked})});await refreshHealth();notice('识别设置已保存。环境检查为可选，不影响远程模型流程。');});
action('checkOcr',async()=>{const result=await api('/api/runtime/ocr-check',{method:'POST'});await refreshHealth();$('ocrCheckResult').textContent=result.ok?'实际设备：'+result.device+'；本次首次调用 '+result.durations_seconds[0]+' 秒，第二次 '+result.durations_seconds[1]+' 秒。'+result.message:result.message;notice(result.ok?'本地自检通过，当前服务中的 OCR 模型已预热。':'本地自检失败，请查看设备错误信息。',!result.ok);});
function wrapTables(container){container.querySelectorAll('table').forEach(table=>{const wrapper=document.createElement('div');wrapper.className='table-scroll';wrapper.tabIndex=0;wrapper.setAttribute('aria-label','可横向滚动的数据表格');table.replaceWith(wrapper);wrapper.append(table);});}
(async()=>{await Promise.all([refreshHealth(),assessments(),loadSettings(),loadLocalVision()].map(p=>p.catch(e=>notice(e.message,true))));})();

function editableRegionQuestions(){return (current?.questions||[]).filter(q=>q.scoring_mode!=='choice_set'&&!['single_choice','multiple_choice'].includes(q.question_type));}
function savedQuestionRegion(q){if(!q)return null;if(activeLayout?.exists)return activeLayout.items?.find(i=>i.question_id===q.id)?.regions?.[0]||null;return q.rule_config?.region||null;}
// Region coordinates always refer to the source page, independent of CSS scaling.
let regionImage=null, regionSelection=null, regionStart=null, regionDocumentId=null, regionLoadedPage=null, regionPreviewUrl=null, regionSaved=false,regionClickStart=null,regionPageRequest=0;
function setupRegionEditor(docs){
 const supported=docs.filter(d=>d.kind==='answer_sheet'&&(d.media_type==='application/pdf'||d.media_type.startsWith('image/')||d.media_type==='application/msword'||d.media_type==='application/vnd.openxmlformats-officedocument.wordprocessingml.document'));
 $('regionDocument').innerHTML=supported.map((d,i)=>'<option value="'+esc(d.id)+'">答题卡 · 第 '+(i+1)+' 份资料</option>').join('');
 $('regionQuestion').innerHTML=editableRegionQuestions().map(q=>'<option value="'+esc(q.id)+'">'+esc(layoutQuestionLabel(q))+(q.rule_config?.region?' · 已配置':'')+'</option>').join('');
 if(activeLayout?.document_id&&supported.some(d=>d.id===activeLayout.document_id))$('regionDocument').value=activeLayout.document_id;
 $('regionSaveQuestion').innerHTML=$('regionQuestion').innerHTML;
 if($('regionDialog').open){$('regionQuestion').disabled=!editableRegionQuestions().length;$('removeRegion').disabled=!editableRegionQuestions().length;refreshRegionSaveButton();}
 $('editRegions').disabled=false;$('editRegions').title=!supported.length?'请先上传答题卡模板／已填样张':!current.questions.length?'请先识别或录入题目':'支持图片、PDF 和 Word 答题卡';
}
function drawRegion(){
 const canvas=$('regionCanvas'),ctx=canvas.getContext('2d');if(!regionImage)return;
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(regionImage,0,0);
 if(regionSelection){const r=regionSelection;ctx.fillStyle='rgba(44,100,240,.15)';ctx.fillRect(r.x*canvas.width,r.y*canvas.height,r.width*canvas.width,r.height*canvas.height);ctx.strokeStyle='#245de0';ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(r.x*canvas.width,r.y*canvas.height,r.width*canvas.width,r.height*canvas.height);const size=10*canvas.width/Math.max(1,canvas.getBoundingClientRect().width);ctx.fillStyle='#245de0';for(const x of [r.x,r.x+r.width])for(const y of [r.y,r.y+r.height])ctx.fillRect(x*canvas.width-size/2,y*canvas.height-size/2,size,size);}
}
function questionRegions(qid){
 const item=activeLayout?.document_id===$('regionDocument').value?activeLayout?.items?.find(i=>i.question_id===qid):null;
 const q=current?.questions?.find(q=>q.id===qid),manual=q?.rule_config?.region;
 if(activeLayout?.exists&&activeLayout.document_id===$('regionDocument').value)return item?.regions||[];
 return (!q?.rule_config?.region_document_id||q.rule_config.region_document_id===$('regionDocument').value)&&manual?[manual]:[];
}
function refreshRegionParts(index=0){
 const boxes=questionRegions($('regionQuestion').value);
 $('regionPart').innerHTML=boxes.map((b,i)=>'<option value="'+i+'">区域 '+(i+1)+' · 第 '+b.page+' 页</option>').join('')+'<option value="new">新作答区</option>';
 $('regionPart').value=boxes[index]?String(index):'new';
}
function selectSavedRegion(){
 regionClickStart=null;regionStart=null;$('regionActions').hidden=true;
 const boxes=questionRegions($('regionQuestion').value);if(!$('regionPart').options.length)refreshRegionParts();
 const r=boxes[Number($('regionPart').value)];
 regionSelection=r&&r.page===regionLoadedPage?{...r}:null;drawRegion();
 focusRegion();
 $('regionStatus').textContent=!current?.questions?.length?'页面已加载，可以框选并预览；保存前请先从原试卷识别题目，不生成参考答案。':regionSelection?'已载入区域：拖动框内移动、四角缩放；也可在框外重新框选。点击框后可预览并保存确认。':'可拖动框选；拖动不便时切换“两点框选”，依次点击区域两个对角。';
}
function openRegionEditor(){
 regionPageRequest++;regionSaved=false;regionImage=null;regionLoadedPage=null;regionSelection=null;regionStart=null;regionClickStart=null;$('regionActions').hidden=true;
 $('regionCanvas').width=0;$('regionCrop').hidden=true;
 const canRender=!!current&&$('regionDocument').options.length>0,hasQuestions=editableRegionQuestions().length>0;
 const problem=!current?'请先新建或选择试卷。':!canRender?'请先上传类型为“答题卡模板／已填样张”的 Word、PDF 或图片。':!hasQuestions?(current?.questions?.length?'本卷选择题自动识别，无需设置区域。':'可先预览答题卡；从原试卷识别题目后，为对应题号保存答题区域，不生成参考答案。'):'';
 for(const id of ['loadRegionPage','regionDocument','regionPage','regionMode','regionScale','previewRegion'])$(id).disabled=!canRender;
 for(const id of ['regionQuestion','saveRegion','removeRegion'])$(id).disabled=!canRender||!hasQuestions;
 $('regionStatus').textContent=problem||'在答题卡上框选学生将填写答案的位置，松开后选择题号保存。';
 $('regionStatus').className=problem?'region-guidance':'region-guidance ready';
 if(!$('regionDialog').open)$('regionDialog').showModal();
 refreshRegionParts();
 if(canRender){$('regionPage').value=String(savedQuestionRegion(current.questions?.find(q=>q.id===$('regionQuestion').value))?.page||1);$('loadRegionPage').click();}
}
$('editRegions').onclick=openRegionEditor;
$('regionPrepare').onclick=()=>{$('regionDialog').close();const target=!current?$('newAssessment'):!$('regionDocument').options.length?$('paperFiles'):!current.questions.length?$('extractQuestions'):$('editRegions');target.scrollIntoView({block:'center'});target.focus();};
$('closeRegions').onclick=()=>{regionPageRequest++;$('regionDialog').close();};
$('regionDialog').addEventListener('close',()=>{regionPageRequest++;if(regionPreviewUrl)URL.revokeObjectURL(regionPreviewUrl);if(regionSaved)loadAssessment().catch(e=>notice(e.message,true));});
$('regionDocument').onchange=()=>{regionPageRequest++;regionLoadedPage=null;regionSelection=null;regionStart=null;regionClickStart=null;refreshRegionParts();$('regionActions').hidden=true;$('regionStatus').textContent='请重新加载页面';};
$('regionPage').oninput=()=>{regionPageRequest++;regionLoadedPage=null;regionSelection=null;regionStart=null;regionClickStart=null;$('regionActions').hidden=true;$('regionStatus').textContent='请重新加载页面';};
$('regionQuestion').onchange=()=>{refreshRegionParts();$('regionPart').onchange();};
$('regionPart').onchange=()=>{const region=questionRegions($('regionQuestion').value)[Number($('regionPart').value)];if(region&&region.page!==regionLoadedPage){$('regionPage').value=String(region.page);$('loadRegionPage').click();}else selectSavedRegion();};
$('addRegionPart').onclick=()=>{$('regionPart').value='new';selectSavedRegion();$('regionStatus').textContent='正在添加本题续答区，可切换页码后框选；原有区域会保留。';};
action('loadRegionPage',async()=>{
 const request=++regionPageRequest,assessmentId=aid();regionLoadedPage=null;regionImage=null;regionSelection=null;$('regionActions').hidden=true;
 const doc=$('regionDocument').value,page=Number($('regionPage').value);if(!Number.isInteger(page)||page<1)throw Error('请输入有效页码');$('regionStatus').textContent='正在加载页面；Word 原卷首次需要本机转换为 PDF，请稍候。';
 const response=await fetch('/api/assessments/'+assessmentId+'/template-page?document_id='+encodeURIComponent(doc)+'&page='+page);
 if(request!==regionPageRequest||!$('regionDialog').open||aid()!==assessmentId)return;
 if(!response.ok){const detail=await response.json().catch(()=>({}));const message=detail.detail||'页面加载失败，请检查页码和资料类型';$('regionStatus').textContent=message;throw Error(message);}
 const url=URL.createObjectURL(await response.blob());
 try{const image=new Image();image.src=url;await image.decode();if(request!==regionPageRequest||!$('regionDialog').open||aid()!==assessmentId)return;regionImage=image;regionDocumentId=doc;regionLoadedPage=page;$('regionCanvas').width=image.naturalWidth;$('regionCanvas').height=image.naturalHeight;applyRegionScale();selectSavedRegion();}finally{URL.revokeObjectURL(url);}
});
const regionCanvas=$('regionCanvas');
let regionGesture=null;
function hitRegion(point){
 if(!regionSelection)return null;const r=regionSelection,rect=regionCanvas.getBoundingClientRect(),dx=10/rect.width,dy=10/rect.height;
 for(const [name,x,y] of [['nw',r.x,r.y],['ne',r.x+r.width,r.y],['sw',r.x,r.y+r.height],['se',r.x+r.width,r.y+r.height]])if(Math.abs(point.x-x)<dx&&Math.abs(point.y-y)<dy)return name;
 return point.x>=r.x&&point.x<=r.x+r.width&&point.y>=r.y&&point.y<=r.y+r.height?'move':null;
}
function adjustRegion(point){
 const {start,box:r,handle}=regionGesture;
 if(handle==='move')regionSelection={...r,x:Math.max(0,Math.min(1-r.width,r.x+point.x-start.x)),y:Math.max(0,Math.min(1-r.height,r.y+point.y-start.y))};
 else{const fixed={x:handle.includes('w')?r.x+r.width:r.x,y:handle.includes('n')?r.y+r.height:r.y};updateRegionSelection(fixed,point);}
 drawRegion();
}
function applyRegionScale(){
 if(!regionImage)return;const surface=$('regionSurface'),canvas=$('regionCanvas');
 const width=Math.max(1,surface.clientWidth-36),height=Math.max(1,surface.clientHeight-36);
 canvas.style.width=Math.round($('regionScale').value==='page'?Math.min(width,height*canvas.width/canvas.height):width)+'px';
 drawRegion();focusRegion();
}
function focusRegion(){
 if(!regionSelection||!regionImage||$('regionScale').value==='page')return;
 const surface=$('regionSurface'),rect=$('regionCanvas').getBoundingClientRect();
 surface.scrollTop=Math.max(0,regionSelection.y*rect.height-surface.clientHeight*.25);
}
$('regionScale').onchange=applyRegionScale;window.addEventListener('resize',applyRegionScale);
function regionPoint(e){const rect=regionCanvas.getBoundingClientRect();return {x:Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),y:Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height))};}
function updateRegionSelection(start,end){
 regionSelection={page:regionLoadedPage,x:Math.min(end.x,start.x),y:Math.min(end.y,start.y),width:Math.abs(end.x-start.x),height:Math.abs(end.y-start.y),page_aspect:regionCanvas.width/regionCanvas.height};drawRegion();
}
function showRegionActions(){
 const choices=editableRegionQuestions();
 $('regionActions').hidden=false;$('regionCrop').hidden=true;$('regionActionStatus').textContent='';
 $('regionSaveQuestion').innerHTML=choices.length?choices.map(q=>'<option value="'+esc(q.id)+'">'+esc(layoutQuestionLabel(q))+(q.rule_config?.region?' · 将替换已存区域':'')+'</option>').join(''):'<option value="">暂无可保存题目</option>';
 if(choices.some(q=>q.id===$('regionQuestion').value))$('regionSaveQuestion').value=$('regionQuestion').value;
 $('regionSaveQuestion').disabled=!choices.length;refreshRegionSaveButton();
 $('regionSelectionInfo').textContent='第 '+regionLoadedPage+' 页 · 宽 '+Math.round(regionSelection.width*100)+'% · 高 '+Math.round(regionSelection.height*100)+'%';
 updateRegionSaveHint();
}
function refreshRegionSaveButton(){
 $('saveRegion').disabled=!editableRegionQuestions().length&&!!current?.questions?.length;
 $('saveRegion').textContent=!current?.questions?.length?'识别题目后选择题号':'保存并确认此区域';
}
function updateRegionSaveHint(){
 const q=editableRegionQuestions().find(q=>q.id===$('regionSaveQuestion').value);
 $('regionSaveHint').textContent=!current?.questions?.length?'尚未识别题目：点击下方按钮先识别（会调用模型），当前选区会保留；完成后选择题号保存。':q?($('regionPart').value==='new'?'新增一个区域，保留本题已有区域。':'只替换所选区域，保留本题其他小问与续答区。'):'选择题无需框选；非选择题须先生成或录入题目。';
}
$('regionSaveQuestion').onchange=updateRegionSaveHint;
$('cancelRegion').onclick=()=>{regionSelection=null;regionStart=null;regionClickStart=null;$('regionActions').hidden=true;drawRegion();$('regionStatus').textContent='已取消本次框选，可重新选择区域。';};
function regionSelectionFeedback(){
 const valid=regionSelection?.width>0&&regionSelection?.height>0;
 if(valid)showRegionActions();
 $('regionStatus').textContent=valid?'已框选区域，可预览裁切。'+(current?.questions?.length?'在弹出卡片中选择题号，再点击“保存区域”。':'尚无题目：可预览，生成或录入题目后才能保存。'):'区域过小，请重新选择两个不同的对角。';
}
$('regionMode').onchange=()=>{regionStart=null;regionClickStart=null;$('regionActions').hidden=true;$('regionStatus').textContent=$('regionMode').value==='corners'?'两点框选：先点击左上角，再点击右下角（也可反向）。':'拖动框选：按住鼠标左键拖动，再松开。';};
regionCanvas.onpointerdown=e=>{
 if(!regionLoadedPage){$('regionStatus').textContent='请先加载页面，页面显示后再框选。';return;}
 if(e.button!==undefined&&e.button!==0)return;
 e.preventDefault();const point=regionPoint(e);$('regionActions').hidden=true;
 if($('regionMode').value==='corners'){
  if(regionClickStart){updateRegionSelection(regionClickStart,point);regionClickStart=null;regionSelectionFeedback();}
  else{regionClickStart=point;regionSelection=null;drawRegion();const ctx=regionCanvas.getContext('2d');ctx.fillStyle='#e14b32';ctx.fillRect(point.x*regionCanvas.width-5,point.y*regionCanvas.height-5,10,10);$('regionStatus').textContent='第一个角已选中，请点击另一个对角。';}
  return;
 }
 const handle=hitRegion(point);if(handle){regionGesture={start:point,box:{...regionSelection},handle};regionCanvas.setPointerCapture(e.pointerId);return;}
 regionStart=point;regionSelection=null;drawRegion();regionCanvas.setPointerCapture(e.pointerId);
 $('regionStatus').textContent='正在框选，拖到区域另一个角后松开鼠标。';
};
regionCanvas.onpointermove=e=>{if(regionGesture)adjustRegion(regionPoint(e));else if(regionStart)updateRegionSelection(regionStart,regionPoint(e));else{const handle=hitRegion(regionPoint(e));regionCanvas.style.cursor=handle==='move'?'move':handle?handle+'-resize':'crosshair';}};
regionCanvas.onpointerup=e=>{if(regionGesture){adjustRegion(regionPoint(e));regionGesture=null;regionSelectionFeedback();return;}if(!regionStart)return;updateRegionSelection(regionStart,regionPoint(e));regionStart=null;regionSelectionFeedback();};
regionCanvas.onpointercancel=()=>{if(regionGesture){regionSelection=regionGesture.box;regionGesture=null;drawRegion();}regionStart=null;};
function selectedRegion(){if(!regionSelection||!regionLoadedPage||regionSelection.width<=0||regionSelection.height<=0)throw Error('请先框选有效区域');return regionSelection;}
action('previewRegion',async()=>{
 const selection=selectedRegion(),request=regionPageRequest;
 const response=await fetch('/api/assessments/'+aid()+'/template-preview?document_id='+encodeURIComponent(regionDocumentId),json(selection));
 if(request!==regionPageRequest||regionSelection!==selection||!$('regionDialog').open)return;
 if(!response.ok){$('regionActionStatus').textContent='裁切失败，请检查区域后重试。';throw Error('裁切失败，请检查区域');}if(regionPreviewUrl)URL.revokeObjectURL(regionPreviewUrl);regionPreviewUrl=URL.createObjectURL(await response.blob());$('regionCrop').src=regionPreviewUrl;$('regionCrop').hidden=false;
});
async function persistRegion(remove){
 const targetId=remove?$('regionQuestion').value:$('regionSaveQuestion').value;
 const region=remove?null:selectedRegion(),assessment=await api('/api/assessments/'+aid()),q=assessment.questions.find(q=>q.id===targetId);
 if(!q)throw Error('请选择有效题号；如尚无题目，请先生成或录入。');if(q.scoring_mode==='choice_set'||['single_choice','multiple_choice'].includes(q.question_type))throw Error('选择题自动识别，无需保存区域');
 if(activeLayout?.exists){
  if(activeLayout.document_id!==regionDocumentId)throw Error('请选择当前位置模板对应的答题卡');
  const regions=questionRegions(targetId).map(b=>({...b})),part=$('regionPart').value;
  const index=part==='new'?regions.length:targetId!==$('regionQuestion').value?0:Math.min(Number(part),regions.length);
  if(remove){if(index>=regions.length)throw Error('请选择已保存的区域');regions.splice(index,1);}else regions[index]=region;
  await api('/api/assessments/'+aid()+'/layout-template/regions/'+q.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_template_id:activeLayout.template_id,document_id:regionDocumentId,regions,confirmed_indices:remove?[]:[index]})});
  regionSaved=true;await loadLayoutStatus();$('regionQuestion').value=targetId;refreshRegionParts(Math.min(index,regions.length-1));selectSavedRegion();$('regionStatus').textContent=remove?'当前区域已移除，其他区域已保留':'当前区域已保存并确认；本题其他区域保持原样。';return;
 }
 const config={...q.rule_config};if(remove)config.region=null;else{config.region=region;config.region_document_id=regionDocumentId;}
 await api('/api/questions/'+q.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({...q,rule_config:config})});
 q.rule_config=config;current.questions=assessment.questions;regionSaved=true;await loadLayoutStatus();$('regionQuestion').innerHTML=editableRegionQuestions().map(item=>'<option value="'+esc(item.id)+'">'+esc(layoutQuestionLabel(item))+(item.rule_config?.region?' · 已配置':'')+'</option>').join('');$('regionQuestion').value=targetId;selectSavedRegion();$('regionStatus').textContent=remove?'区域已移除':'区域已保存，下次识别复用；已保存作答保持原样';
}
action('saveRegion',async()=>{try{if(!current?.questions?.length){const id=aid(),draft=regionSelection;$('regionActionStatus').textContent='正在识别题目，请稍候；当前选区已保留。';await ensureAssessmentQuestions(id);if(regionSelection!==draft||!$('regionDialog').open)return;showRegionActions();$('regionActionStatus').textContent='题目已识别，请选择对应题号，再点击保存区域。';return;}await persistRegion(false);}catch(e){$('regionActionStatus').textContent=e.message;throw e;}});action('removeRegion',()=>persistRegion(true));
let trainingGrade=null;
function openTraining(g,response){trainingGrade=g;$('trainingText').value=response?.raw_text||'';$('trainingApprover').value='';$('trainingConsent').checked=false;$('trainingDialog').showModal();}
$('closeTraining').onclick=()=>{$('trainingDialog').close();};
action('approveTraining',async()=>{if(!$('trainingConsent').checked)throw Error('请先检查脱敏文本并勾选批准');await api('/api/grade-items/'+trainingGrade.id+'/training-approval',json({redacted_text:$('trainingText').value,approved_by:$('trainingApprover').value,expected_review_id:trainingGrade.latest_review_id||null}));$('trainingDialog').close();notice('已批准脱敏样本。标签来源由实际人工复核记录决定。');});

function layoutQuestionLabel(q){
 const item=activeLayout?.items?.find(i=>i.question_id===q.id);
 return item?.section?item.section+' / '+item.number:'题号 '+q.question_no;
}
async function loadLayoutStatus(docs){
 const id=aid(),state=await api('/api/assessments/'+id+'/layout-template');if(current?.id!==id)return;
 activeLayout=state;
 if(docs)$('layoutDocument').innerHTML=docs.filter(d=>d.kind==='answer_sheet').map((d,i)=>'<option value="'+esc(d.id)+'">答题卡 · '+(i+1)+'</option>').join('');
 if(state.document_id)$('layoutDocument').value=state.document_id;
 $('layoutBadge').textContent=state.enabled?(state.stale?'模板待更新':state.confirmed_items===state.items.length?'位置已确认':'候选区域待核对'):state.exists?'已停用':'尚未配置';
 $('layoutBadge').className='badge '+(state.enabled&&!state.stale?'':'neutral');
 updateLayoutEntry();
 $('previewLayout').disabled=!state.exists;$('toggleLayout').disabled=!state.exists;
 $('calibrateLayout').disabled=!state.exists||state.stale;
 $('toggleLayout').textContent=state.enabled?'停用并恢复原识别路径':'启用本地模板';
 const deps=state.local_dependencies||{},missing=Object.entries(deps).filter(([_,v])=>!v).map(([k])=>k);
 $('layoutState').textContent=!current?.questions?.length?'先上传原试卷和答题卡模板／已填样张。尚无题目时先提取题目结构，再匹配答题卡位置，两步模型请求均不生成参考答案。':state.enabled?(state.stale?'题目或来源已变更：建议重新生成；疑难题按API补处理设置执行。':'模板已启用：'+state.items.length+' 道题，'+state.page_count+' 页。学生答卷本地优先；未可靠识别时按设置补一次API，仍可人工填写。'):'请先预览来源，支持空白卡或已填样张；拼图需保持同样版式。首次解析使用模型；可靠作答复用本地结果，疑难项按API补处理设置执行。';
 if(state.exists&&!state.stale)$('layoutState').textContent+=' 坐标合法仅表示可裁切，不代表位置正确；已逐区确认 '+(state.confirmed_items||0)+' / '+state.items.length+' 题。可先本地边线校准，再预览或修正区域。';
 if(missing.length)$('layoutState').textContent+=' 当前服务缺少 '+missing.join(' / ')+'，请用优化版 GPU 环境启动。';
 if(state.warnings?.length)$('layoutState').textContent+=' '+state.warnings.join('；');
 const failed=state.last_attempt;if(failed?.version==='local-layout-v3'&&failed?.status==='rejected'&&failed.document_id===$('layoutDocument').value){$('layoutState').textContent+=' 上次检查：'+failed.error;if(!state.enabled)$('layoutBadge').textContent='来源待检查';}
 $('layoutQuestions').innerHTML=(state.items||[]).map(item=>{const q=current.questions.find(q=>q.id===item.question_id),review=state.region_reviews?.[item.question_id];return q?'<span class="layout-question-chip"><b>'+esc(item.section||'未分组')+'</b> / '+esc(item.number||q.question_no)+' · '+(item.method==='omr'?'填涂检测':'作答内容识别')+' · '+item.regions.length+' 区'+(review?.source==='manual'?' · 已人工调整':review?.source==='printed_lines'?' · 已按边线校准，待核对':' · 候选待核对')+'</span>':'';}).join('');
}
action('calibrateLayout',async()=>{const state=await api('/api/assessments/'+aid()+'/layout-template/calibrate',json({}));await loadLayoutStatus();notice('本地校准 '+state.adjusted_items+' 题，未调用 API；请预览核对。');});
function updateLayoutEntry(){
 const button=$('generateLayout');button.disabled=layoutBusy;
 if(layoutBusy)$('layoutBadge').textContent='正在检查并定位…';
 else if($('layoutBadge').textContent==='正在检查并定位…')$('layoutBadge').textContent=activeLayout?.enabled?(activeLayout.stale?'模板待更新':'本地识别已启用'):activeLayout?.exists?'已停用':'尚未配置';
 button.textContent=layoutBusy?'正在处理…':!current?.questions?.length?'识别题目并匹配答题卡':'匹配答题卡并启用模板';
 button.title=!current?'请先新建或选择试卷':!$('layoutDocument').options.length?'请先上传答题卡模板／已填样张':!current.questions.length?'先提取原试卷题目结构，再定位答题卡；两步模型请求，不生成答案':'首次请求模型生成位置；有效模板直接复用';
}
async function ensureAssessmentQuestions(assessmentId){
 if(current?.id!==assessmentId)throw Error('试卷已切换，请重新操作');
 if(current.questions.length)return;
 if(!questionPreparation){
  questionPreparation=(async()=>{
   await api('/api/assessments/'+assessmentId+'/extract-questions',{method:'POST'});
   if(current?.id!==assessmentId)throw Error('试卷已切换，题目结果已保存，请返回原试卷查看');
   await loadAssessment();
   if(!current?.questions?.length)throw Error('原卷中未识别到题目，请检查资料可读性，或上传含题干的原试卷后重试');
  })().finally(()=>{questionPreparation=null;});
 }
 await questionPreparation;
}
action('generateLayout',async()=>{
 const id=current?.id,documentId=$('layoutDocument').value,force=$('forceLayout').checked;
 if(!id){$('layoutState').textContent='请先新建或选择试卷，再上传原卷。';throw Error($('layoutState').textContent);}
 if(!documentId){$('layoutState').textContent='请按“答题卡模板／已填样张”类型上传答题卡，再选择模板来源。';throw Error($('layoutState').textContent);}
 layoutBusy=true;updateLayoutEntry();
 try{
  if(!current.questions.length){$('layoutState').textContent='第 1/2 步：正在提取原试卷题目结构，不生成答案…';await ensureAssessmentQuestions(id);}
  if(current?.id!==id)throw Error('试卷已切换，请返回原试卷继续生成模板');
  $('layoutState').textContent='正在定位作答区与填涂位置，生成本地识别模板…';
  await api('/api/assessments/'+id+'/layout-template',json({document_id:documentId,force}));
  if(current?.id===id){$('forceLayout').checked=false;await loadAssessment();notice('答题卡模板已启用。请核对题号与位置；参考答案需在独立步骤配置。');}
 }catch(e){if(current?.id===id){$('layoutState').textContent=e.message;$('layoutBadge').textContent=activeLayout?.enabled?'生成失败 · 保留旧模板':'模板未通过检查';}throw e;}
 finally{layoutBusy=false;updateLayoutEntry();}
});
action('toggleLayout',async()=>{await api('/api/assessments/'+aid()+'/layout-template',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:!activeLayout.enabled})});await loadLayoutStatus();});
let layoutPreviewSource=false;
$('previewLayoutSource').onclick=()=>{if(!$('layoutDocument').value){notice('请先上传并选择答题卡模板／已填样张。',true);return;}layoutPreviewSource=true;$('layoutPreviewPage').value='1';$('layoutPreviewDialog').showModal();$('loadLayoutPreview').click();};
$('previewLayout').onclick=()=>{layoutPreviewSource=false;$('layoutPreviewPage').value='1';$('layoutPreviewDialog').showModal();$('loadLayoutPreview').click();};
$('closeLayoutPreview').onclick=()=>{$('layoutPreviewDialog').close();};
action('loadLayoutPreview',async()=>{
 const page=Number($('layoutPreviewPage').value);if(!Number.isInteger(page)||page<1)throw Error('请输入有效页码');
 $('layoutPreviewTitle').textContent=layoutPreviewSource?'来源答题卡 · 输入检查':'自动模板 · 位置核对';
 $('layoutPreviewHint').textContent=layoutPreviewSource?'已填样张只提供位置，不提供标准答案；拼图按整张定位；这里仅预览，不调用模型。':'蓝框为文字区域，绿框为填涂位置。请核对题号与实际位置。';
 $('layoutPreviewImage').alt=layoutPreviewSource?'上传的来源答题卡原图':'自动模板区域标注预览';
 $('layoutPreviewStatus').textContent=layoutPreviewSource?'正在加载来源图片…':'正在加载标注页…';
 const response=await fetch('/api/assessments/'+aid()+(layoutPreviewSource?'/template-page?document_id='+encodeURIComponent($('layoutDocument').value)+'&page=':'/layout-template/page?page=')+page);
 if(!response.ok){const e=await response.json();$('layoutPreviewStatus').textContent=e.detail;throw Error(e.detail);}
 const url=URL.createObjectURL(await response.blob());const image=$('layoutPreviewImage');image.src=url;
 try{await image.decode();$('layoutPreviewStatus').textContent='第 '+page+' 页 · '+(layoutPreviewSource?'来源答题卡（仅预览，不调用模型）':'自动定位结果');}finally{URL.revokeObjectURL(url);}
});

function localRecognitionHtml(submission){
 const report=submission.normalization_report?.local_recognition;if(!report)return '';
 const missing=report.missing||[],warnings=report.warnings||[];
 const candidates=Object.entries(report.recognitions||{}).map(([qid,value])=>'<details><summary>题号 '+esc(value.question_no)+' · 转写与原图核对</summary>'+value.parts.map((part,index)=>'<div class="transcription-part"><p>区域 '+(index+1)+' · '+esc(part.model?.model||'')+' · '+(part.blank_confirmed?'已确认未作答':part.consistent?(part.method==='ocr'?'本地OCR已采用':'两次转写一致，已自动采用（非人工确认）'):'待校正：'+(part.issues||[]).map(esc).join('；'))+'</p>'+(part.crops||[part.crop]).map((_,r)=>'<a target="_blank" rel="noopener" href="/api/submissions/'+encodeURIComponent(submission.id)+'/local-recognition-crop?question_id='+encodeURIComponent(qid)+'&part='+index+'&region='+r+'">查看原图裁切 '+(r+1)+' ↗</a>').join(' · ')+'<pre>'+esc(part.text)+'</pre>'+(part.verification&&!part.consistent?'<p>第二次转写：</p><pre>'+esc(JSON.stringify(part.verification,null,2))+'</pre>':'')+'<button type="button" class="secondary" data-local-retry="'+esc(submission.id)+'" data-question="'+esc(qid)+'" data-part="'+index+'">重新识别预览</button><div class="local-retry-result" aria-live="polite"></div></div>').join('')+'</details>').join('');
 return '<details class="local-recognition-note '+(missing.length?'incomplete':'')+'"><summary>'+((report.remote_vision_requests||0)>0?'本地优先识别 · 视觉API补识 '+report.remote_vision_requests+' 次':'本地答卷识别 · 本次未调用视觉API')+'<span class="overview-toggle">识别记录</span></summary><p>'+(missing.length?'尚未可靠识别：'+missing.map(esc).join('、')+'。这些题保留缺分，可在下方校对并立即判分。':'所有题均已有通过当前校验或人工校正的识别结果；模型一致仍可能出错。')+'</p>'+(warnings.length?'<details><summary>识别详情</summary>'+warnings.slice(0,20).map(w=>'<p>'+esc(w)+'</p>').join('')+'</details>':'')+candidates+Object.values(report.api_fallback||{}).map(info=>'<details><summary>第 '+esc(info.question_no)+' 题 · API自动补识'+(info.error?'失败':info.adopted?'已采用':'待核对')+'</summary>'+(info.trigger_reasons?.length?'<p class="muted small">触发原因：'+info.trigger_reasons.map(esc).join('；')+'</p>':'')+(info.candidate?'<pre>'+esc(info.candidate.blank_confirmed?'[未作答]':info.candidate.raw_text)+'</pre>':'')+(info.error?'<p>'+esc(info.error)+'</p>':'')+'</details>').join('')+'</details>';
}


async function loadLocalVision(){
 const state=await api('/api/runtime/local-vision');$('localVisionEnabled').value=String(state.enabled);$('localVisionModel').value=state.model;
 $('localVisionStatus').textContent=state.ready?'模型已就绪：'+state.model+' · '+(state.quantization||'')+'。两次一致自动采用，仍可人工校对；'+(state.hybrid_fallback_enabled?'疑难题允许API补识。':'API补处理已关闭。')+'':state.error;
 return state;
}
document.addEventListener('click',async event=>{
 const button=event.target.closest('[data-local-retry]');if(!button||button.disabled)return;
 const output=button.parentElement.querySelector('.local-retry-result');button.disabled=true;
 output.textContent='正在用当前本地模型重新转写，仅预览，不修改作答与成绩…';
 try{
  const result=await api('/api/submissions/'+encodeURIComponent(button.dataset.localRetry)+'/local-recognition-preview?question_id='+encodeURIComponent(button.dataset.question)+'&part='+encodeURIComponent(button.dataset.part),{method:'POST'});
  output.innerHTML='<p>新候选 · '+esc(result.model?.model)+' · '+esc(result.version)+' · '+(result.blank_candidate?'空白候选；仅预览，尚未确认出分':result.consistent?'通过当前校验，仍需核对原图':'待核对：'+(result.issues||[]).map(esc).join('；'))+'</p><pre>'+esc(result.text)+'</pre><p>仅预览候选。请对照原图，在作答编辑区保存校正内容；已有成绩未改变。</p>';
 }catch(error){output.textContent=error.message;}finally{button.disabled=false;}
});

action('checkLocalVision',async()=>{await loadLocalVision();});
action('saveLocalVision',async()=>{const state=await api('/api/runtime/local-vision',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:$('localVisionEnabled').value==='true',model:$('localVisionModel').value.trim(),verify:true})});await loadLocalVision();notice(state.enabled?'已启用本地优先识别；疑难项按API补处理设置执行，不修改已有成绩。':'已恢复原有 OCR 路径。');});
