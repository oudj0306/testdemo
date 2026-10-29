import shutil
import subprocess
from pathlib import Path

import pytest


def test_review_viewer_switching_zoom_and_stale_requests_do_not_touch_edits():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    source = Path(__file__).resolve().parents[2] / "src/grading_agent/static/review-evidence.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
 constructor(){this.innerHTML='';this.textContent='';this.attrs={};this.disabled=false;this.listeners={};this.dataset={};this.clientWidth=500;this.clientHeight=400;this.scrollLeft=0;this.scrollTop=0;this.isConnected=true;this.style={};this.classList={toggle(){},add(){},remove(){}};}
 addEventListener(name,fn){this.listeners[name]=fn;}
 removeAttribute(name){delete this.attrs[name];}
 setAttribute(name,value){this.attrs[name]=value;}
 scrollTo({top=0,left=0}){this.scrollTop=top;this.scrollLeft=left;}
 closest(){return this.parent||null;}
 querySelectorAll(){return [];}
 querySelector(){return null;}
 setPointerCapture(){}
}
const root=new Element(),viewport=new Element(),select=new Element();select.parent=new Element();
const modes=['crop','page'].map(mode=>{const b=new Element();b.dataset.mode=mode;return b;});
const zooms=['out','in','fit'].map(zoom=>{const b=new Element();b.dataset.zoom=zoom;return b;});
const singles={'.evidence-viewport':viewport,'select':select,'.evidence-status':new Element(),'.evidence-question':new Element(),'.evidence-percent':new Element(),'.evidence-original':new Element(),'.evidence-locate':new Element()};
for(const b of zooms)singles['[data-zoom="'+b.dataset.zoom+'"]']=b;
root.querySelector=s=>singles[s];root.querySelectorAll=s=>s==='[data-mode]'?modes:s==='[data-zoom]'?zooms:[];
const requests=[],window={addEventListener(){},removeEventListener(){}};
const env={window,AbortController,fetch:(url,options)=>new Promise(resolve=>requests.push({url,options,resolve}))};
vm.createContext(env);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),env);
const g=window.reviewEvidenceGeometry;
assert.equal(g.clampZoom(10),4);assert.equal(g.clampZoom(.1),.5);
assert.equal(g.pageFocus([],1),null);
const focus=g.pageFocus([{page:1,polygon:[[.1,.2],[.4,.2],[.4,.5],[.1,.5]]},{page:2,polygon:[[0,0],[1,1]]}],1);
assert(Math.abs(focus.x-.1)<1e-8);assert(Math.abs(focus.height-.3)<1e-8);
const viewer=window.createReviewEvidence('paper',root);
const manifest=id=>({question_id:id,question_no:id,default_page:1,default_mode:'crop',note:'',pages:[{page:1,url:'/page-'+id},{page:2,url:'/page2-'+id}],regions:[{page:1,polygon:[[.1,.2],[.4,.2],[.4,.5],[.1,.5]]}],crops:[{page:1,url:'/crop-'+id,label:'<script>unsafe</script>'},{page:2,url:'/crop2-'+id,label:'续答区域'}]});
(async()=>{
 const first=viewer.show({id:'one',question_no:'1'}),second=viewer.show({id:'two',question_no:'2'});
 assert.equal(requests.length,2);assert(requests[0].options.signal.aborted);
 requests[1].resolve({ok:true,json:async()=>manifest('two')});await second;
 assert(viewport.innerHTML.includes('/crop-two'));assert(viewport.innerHTML.includes('/crop2-two'));
 assert(viewport.innerHTML.includes('&lt;script&gt;'));assert(!viewport.innerHTML.includes('<script>unsafe'));
 requests[0].resolve({ok:true,json:async()=>manifest('one')});await first;
 assert(viewport.innerHTML.includes('/crop-two'));assert(!viewport.innerHTML.includes('/crop-one'));
 modes[1].onclick();assert(viewport.innerHTML.includes('/page-two'));assert(viewport.innerHTML.includes('<polygon'));
 select.value='2';select.onchange();assert(viewport.innerHTML.includes('/page2-two'));assert(!viewport.innerHTML.includes('<polygon'));
 for(let i=0;i<20;i++)zooms[1].onclick();assert.equal(singles['.evidence-percent'].textContent,'400%');
 zooms[2].onclick();assert.equal(singles['.evidence-percent'].textContent,'100%');
 const third=viewer.show({id:'three',question_no:'3'});requests[2].resolve({ok:true,json:async()=>manifest('three')});await third;
 assert(viewport.innerHTML.includes('/page-three')); // Whole-page preference follows the reviewer.
 const fourth=viewer.show({id:'four',question_no:'4'});viewer.dispose();requests[3].resolve({ok:true,json:async()=>manifest('four')});await fourth;
 assert(!viewport.innerHTML.includes('/page-four'));
 assert(requests.every(r=>r.url.includes('/review-visual?question_id=')&&!r.options.method));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    result = subprocess.run(
        [node, "-e", script, str(source)], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
