const $ = id => document.getElementById(id);
const colors={cyan:'#32a9c7',red:'#d84f4a',yellow:'#e7bf35',green:'#5caa71',blue:'#446bc6',orange:'#dc873b',purple:'#9563bd',white:'#eee'};
let scenes=[],current=null,timer=null,step=0;
function node(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function toast(text){$('toast').textContent=text;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',4000);}
function resetPlan(){clearInterval(timer);timer=null;step=0;$('play-plan').textContent='Preview build order ▶';$('plan-status').textContent='Ready to explore';}
function showScenes(tier='all'){
  $('scene-list').replaceChildren();
  for(const s of scenes.filter(s=>tier==='all'||s.tier===tier)){
    const btn=node('button',undefined,'scene-card'+(s.id===current?.id?' selected':''));btn.type='button';btn.setAttribute('aria-pressed',String(s.id===current?.id));
    const img=node('img');img.src=`assets/scenes/${s.id}.png`;img.alt='';img.loading='lazy';
    const label=node('span');label.append(node('strong',s.structure),node('small',`${s.id} · ${s.target.blocks.length} blocks`));btn.append(img,label);btn.addEventListener('click',()=>select(s));$('scene-list').append(btn);
  }
}
function select(s){
  resetPlan();current=s;$('target-image').src=`assets/scenes/${s.id}.png`;$('target-image').alt=`Benchmark target: ${s.structure}, ${s.target.blocks.length} blocks`;$('scene-id').textContent=s.id;$('target-tier').textContent=`${s.tier} / DEVELOPMENT BENCHMARK`;$('target-name').textContent=s.structure.replaceAll('_',' ');$('block-count').textContent=s.target.blocks.length;
  $('plan-list').replaceChildren();for(const [i,id] of s.oracle_order.entries()){
    const b=s.target.blocks.find(b=>b.id===id);const row=node('div',undefined,'plan-step');const swatch=node('span',undefined,'swatch');swatch.style.background=colors[b.color]||'#888';row.append(node('span',String(i+1).padStart(2,'0'),'step-num'),swatch,node('span',`${b.color} ${b.type}`),node('small',id));$('plan-list').append(row);
  }showScenes(document.querySelector('[data-tier].active').dataset.tier);
}
document.querySelectorAll('[data-tier]').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('[data-tier]').forEach(b=>b.classList.toggle('active',b===btn));showScenes(btn.dataset.tier);}));
$('play-plan').addEventListener('click',()=>{
  if(!current)return;if(timer){resetPlan();document.querySelectorAll('.plan-step').forEach(el=>el.classList.remove('highlight','complete'));return;}
  document.querySelectorAll('.plan-step').forEach(el=>el.classList.remove('highlight','complete'));step=0;$('play-plan').textContent='Stop preview ■';
  const advance=()=>{const rows=[...document.querySelectorAll('.plan-step')];if(step>=rows.length){clearInterval(timer);timer=null;rows.forEach(r=>{r.classList.remove('highlight');r.classList.add('complete');});$('plan-status').textContent='Build order preview complete · no simulation run';$('play-plan').textContent='Replay build order ↻';return;}rows.forEach((r,i)=>{r.classList.toggle('highlight',i===step);r.classList.toggle('complete',i<step);});$('plan-status').textContent=`Plan step ${step+1} of ${rows.length}: ${current.oracle_order[step]}`;step++;};advance();timer=setInterval(advance,1100);
});
$('download-scene').addEventListener('click',()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=`${current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
const roles=[['Principal investigator','Coordinates the goal, budget, and next research decision.','CLAUDE'],['Literature scout','Finds prior work and records evidence for the research goal.','CLAUDE'],['Scene designer','Creates development scenes and checks target stability.','CLAUDE'],['Method designer','Proposes approaches and versions the assembly method.','CLAUDE'],['Experiment planner','Compares candidate tests before selecting an experiment.','CLAUDE'],['Robotics engineer','Implements and validates the robot’s assembly method.','CODEX'],['Experiment runner','Runs selected experiments in the simulation environment.','CLAUDE'],['Analyst','Measures outcomes and explains what failed and why.','CLAUDE'],['Reviewer','Checks whether the evidence supports the claimed results.','CODEX']];
for(const [i,r] of roles.entries()){const card=node('article',undefined,'agent-card');card.append(node('span',String(i+1).padStart(2,'0'),'agent-number'),node('h3',r[0]),node('p',r[1]),node('span',`${r[2]} · CONFIGURED ROLE`,'model'));$('agent-grid').append(card);}
function renderRecord(record){
  if(!Array.isArray(record)||record.some(o=>!o||typeof o!=='object'||typeof o.id!=='string'||typeof o.kind!=='string'||!o.data||typeof o.data!=='object'||Array.isArray(o.data)))throw new Error('Expected the array exported by record/export.json.');
  const runs=record.filter(o=>o.kind==='run').sort((a,b)=>(b.created||0)-(a.created||0));const summary=runs[0]?.data.summary;const reviews=record.filter(o=>o.kind==='result'&&o.data.review);const numeric=n=>typeof n==='number'&&Number.isFinite(n);
  $('run-count').textContent=runs.length;$('review-count').textContent=reviews.length;$('success-rate').textContent=numeric(summary?.success_rate)?`${(summary.success_rate*100).toFixed(1)}%`:'—';$('placement-error').textContent=numeric(summary?.median_pos_err_cm)?`${summary.median_pos_err_cm.toFixed(1)} cm`:'—';$('record-status').textContent=`${record.length} record objects loaded · ${runs.length} measured runs. Results are an imported snapshot, not live activity.`;
  $('record-list').replaceChildren();const interesting=record.filter(o=>['run','decision','result','hypothesis','experiment'].includes(o.kind)).sort((a,b)=>(b.created||0)-(a.created||0)).slice(0,50);
  for(const o of interesting){const row=node('article',undefined,'record-row');const meta=node('div');meta.append(node('span',o.kind.toUpperCase(),'tag'),node('p',o.id));const body=node('div');let description=o.data.interpretation||o.data.change||o.data.statement||o.data.expected_learning||`Method: ${o.data.method||'unspecified'}`;if(typeof description!=='string')description=JSON.stringify(description);body.append(node('p',description.slice(0,3000)));const date=numeric(o.created)?new Date(o.created*1000).toLocaleString():'';body.append(node('small',`${o.author||'Unknown author'}${date?' · '+date:''}${o.data.status?' · '+o.data.status:''}`));if(o.kind==='run'&&numeric(o.data.summary?.success_rate))body.append(node('p',`${(o.data.summary.success_rate*100).toFixed(1)}% success · ${o.data.summary.episodes??'—'} episodes`));row.append(meta,body);$('record-list').append(row);}
}
$('record-file').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>10*1024*1024)throw new Error('Please import a record under 10 MB.');renderRecord(JSON.parse(await file.text()));toast('Research record imported.');}catch(e){toast(`Could not import: ${e.message}`);}event.target.value='';});
try{const res=await fetch('data/scenes.json');if(!res.ok)throw new Error('Scene data could not be loaded');scenes=await res.json();$('scene-total').textContent=`${scenes.length} development benchmarks`;select(scenes.find(s=>s.id==='T3-dev-03')||scenes[0]);}catch(e){$('scene-list').textContent='Could not load benchmark data. Serve this folder through HTTP; do not open index.html as a local file.';toast(e.message);}
try{const res=await fetch('data/record.json');if(res.ok){const record=await res.json();if(Array.isArray(record)&&record.length)renderRecord(record);}}catch{}
