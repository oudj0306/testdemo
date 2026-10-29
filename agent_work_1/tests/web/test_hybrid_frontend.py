import shutil
import subprocess
from pathlib import Path

import pytest


def test_overview_includes_missing_questions_candidates_and_manual_blank_without_xss():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    source = Path(__file__).resolve().parents[2] / "src/grading_agent/static/app.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(process.argv[1],'utf8');
const env={esc:v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))};
vm.createContext(env);
vm.runInContext(code.slice(code.indexOf('function answerCandidate('),code.indexOf('async function openSubmission(')),env);
const q=[{id:'a',question_no:'1'},{id:'b',question_no:'2'},{id:'c',question_no:'3'}];
const s={responses:[{question_id:'a',raw_text:'',blank_confirmed:true,extraction_source:'manual'}],normalization_report:{local_recognition:{missing:['2','3'],recognitions:{b:{parts:[{text:'<script>bad()</script>'}]}}}}};
const html=env.answerOverviewHtml(s,q);
assert.equal((html.match(/data-answer-question=/g)||[]).length,3);
assert(html.includes('[未作答]'));assert(html.includes('等待自动API补识'));
assert(html.includes('&lt;script&gt;'));assert(!html.includes('<script>'));
assert(html.includes('人工确认未作答'));assert(html.includes('本地候选 · 待API补识'));
vm.runInContext(code.slice(code.indexOf('function draftValue('),code.indexOf('function hasDraft(')),env);
assert.equal(env.draftValue({type:'checkbox',value:'on',checked:false}),'false');
assert.equal(env.draftValue({type:'checkbox',value:'on',checked:true}),'true');
assert.deepEqual(Array.from(env.reviewQuestionOrder([{question_no:'二-10'},{question_no:'二-2'},{question_no:'二-1'}]),q=>q.question_no),['二-1','二-2','二-10']);
// Pagination keeps original question indices, so page two cannot edit page one's answers.
const fields={pager:{innerHTML:''}};env.$=id=>fields[id];
vm.runInContext(code.slice(code.indexOf('const pageState ='),code.indexOf('function sortList(')),env);
env.pg=vm.runInContext('pg',env);
vm.runInContext(code.slice(code.indexOf('function renderPager('),code.indexOf('function onPagerChange(')),env);
const many=Array.from({length:23},(_,i)=>({q:{id:'q'+i,question_no:String(i+1)},i}));
env.pg('answers:one').page=2;
const second=env.pageSlice(many,'answers:one');assert.equal(second.length,10);assert.equal(second[0].i,10);
const paged=env.answerOverviewHtml({responses:[]},many.map(x=>x.q),second,false);
assert(paged.includes('data-answer-question="10"'));assert(!paged.includes('data-answer-question="0"'));
assert(!paged.includes('answer-overview" open'));
env.renderPager('pager','answers:one',23,'题');assert(fields.pager.innerHTML.includes('共 23 题 · 2 / 3 页'));
env.pg('answers:one').page=3;assert.equal(env.pageSlice(many,'answers:one').length,3);
env.pg('answers:one').size=20;env.pg('answers:one').page=1;assert.equal(env.pageSlice(many,'answers:one').length,20);
assert.equal(env.answerRecognitionState({responses:[{question_id:'q',raw_text:'42',extraction_source:'api_vision_fallback'}]},{id:'q'}),'API补识 · 已采用');
assert.equal(env.answerRecognitionState({responses:[],normalization_report:{local_recognition:{api_fallback:{q:{attempted:true,error:'timeout'}}}}},{id:'q'}),'API补识 · 调用失败');
env.expandedPerformance=new Set();env.batchControlsSupport=true;env.pendingBatchControls=new Set();
vm.runInContext(code.slice(code.indexOf('function batchControlsHtml('),code.indexOf('async function ensureBatchControls(')),env);
const batch=env.batchCardHtml({id:'b',status:'completed',created_at:'2026-10-26',success_count:2,total_count:2,duplicate_count:0,failed_count:0});
assert(batch.includes('data-batch-usage="b"'));assert(!batch.includes('<button'));
const stats=env.performanceHtml({wall_ms:1200,requests:1,tokens:{input_tokens:100,output_tokens:5,cached_tokens:0},failed_requests:0,retries:0});
assert(stats.includes('1.20 秒'));assert(stats.includes('usage-metrics'));
const running=env.batchControlsHtml({id:'b',status:'processing'});assert(running.includes('data-batch-action="pause"'));assert(running.includes('data-batch-action="stop"'));
const stopped=env.batchControlsHtml({id:'b',status:'stopped'});assert(stopped.includes('从停止处继续'));assert(stopped.includes('data-batch-action="resume"'));
const settling=env.batchControlsHtml({id:'b',status:'paused',settling:true});assert(settling.includes('disabled'));
env.batchControlsSupport=false;const unavailable=env.batchControlsHtml({id:'b',status:'processing'});assert(unavailable.includes('Ctrl+C'));assert(!unavailable.includes('<button'));
env.batchControlsSupport=true;env.pendingBatchControls.add('b');assert(env.batchControlsHtml({id:'b',status:'processing'}).includes('disabled'));


"""
    subprocess.run(
        [node, "-e", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_control_handler_is_delegated_immediate_and_blocks_duplicate_requests():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    source = Path(__file__).resolve().parents[2] / "src/grading_agent/static/app.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(process.argv[1],'utf8');
let finish,requests=0,refreshes=0,handler;
const feedback={textContent:'',classList:{add(){}}};
const button={disabled:false,dataset:{batchId:'one',batchAction:'pause'},closest:()=>card};
const card={querySelector:()=>feedback,querySelectorAll:()=>[button]};
const root={addEventListener:(name,callback)=>handler=callback,contains:()=>true};
const env={$:()=>root,aid:()=> 'assessment',pendingBatchControls:new Set(),performanceCache:new Map(),batchTimer:null,clearTimeout(){},setTimeout(){},notice(){},api:async()=>{requests++;return new Promise(resolve=>finish=resolve);},refreshBatches:async()=>{refreshes++;}};
vm.createContext(env);
vm.runInContext(code.slice(code.indexOf('async function handleBatchControlClick('),code.indexOf('async function refreshBatches(')),env);
assert.equal(typeof handler,'function');
(async()=>{
 const event={target:{closest:()=>button}};
 const pending=handler(event);
 assert(button.disabled);assert(feedback.textContent.length>0);assert.equal(requests,1);
 await handler(event);assert.equal(requests,1);
 finish({status:'paused'});await pending;
 assert.equal(refreshes,1);assert.equal(env.pendingBatchControls.size,0);
 env.batchCapabilityPromise=null;env.api=async()=>({paths:{}});
 vm.runInContext(code.slice(code.indexOf('async function ensureBatchControls('),code.indexOf('async function handleBatchControlClick(')),env);
 await env.ensureBatchControls();assert.equal(env.batchControlsSupport,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    subprocess.run(
        [node, "-e", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
