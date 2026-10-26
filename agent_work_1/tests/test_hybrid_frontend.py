import shutil
import subprocess
from pathlib import Path

import pytest


def test_overview_includes_missing_questions_candidates_and_manual_blank_without_xss():
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    source=Path(__file__).resolve().parents[1]/'src/grading_agent/static/app.js'
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(process.argv[1],'utf8');
const env={esc:v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))};
vm.createContext(env);
vm.runInContext(code.slice(code.indexOf('function answerCandidate('),code.indexOf('async function openSubmission(')),env);
const q=[{id:'a',question_no:'1'},{id:'b',question_no:'2'},{id:'c',question_no:'3'}];
const s={responses:[{question_id:'a',raw_text:'',blank_confirmed:true,extraction_source:'manual'}],normalization_report:{local_recognition:{missing:['2','3'],recognitions:{b:{parts:[{text:'<script>bad()</script>'}]}}}}};
const html=env.answerOverviewHtml(s,q);
assert.equal((html.match(/data-answer-question=/g)||[]).length,3);
assert(html.includes('[未作答]'));assert(html.includes('尚未识别'));
assert(html.includes('&lt;script&gt;'));assert(!html.includes('<script>'));
assert(html.includes('人工确认未作答'));assert(html.includes('候选待人工确认'));
vm.runInContext(code.slice(code.indexOf('function draftValue('),code.indexOf('function hasDraft(')),env);
assert.equal(env.draftValue({type:'checkbox',value:'on',checked:false}),'false');
assert.equal(env.draftValue({type:'checkbox',value:'on',checked:true}),'true');
'''
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)
