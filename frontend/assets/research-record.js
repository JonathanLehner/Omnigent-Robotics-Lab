const $ = id => document.getElementById(id);
const el = (tag, text, className) => {
 const n = document.createElement(tag);
 if (text !== undefined) n.textContent = text;
 if (className) n.className = className;
 return n;
};
const numeric = n => typeof n === 'number' && Number.isFinite(n);
const rate = n => numeric(n) ? `${(100*n).toFixed(1)}%` : '—';
const filters = [['all','All evidence'],['run','Runs'],['result','Reviews'],['decision','Decisions'],['hypothesis','Hypotheses'],['experiment','Experiments'],['evidence','Literature']];
const kinds = new Set(filters.slice(1).map(([k]) => k));
function detail(parent, title, text) {
 if (!text) return;
 const d = el('details', undefined, 'record-detail');
 d.append(el('summary', title), el('p', typeof text === 'string' ? text : JSON.stringify(text)));
 parent.append(d);
}
function badge(verdict) {
 return el('span', verdict === 'accept' ? 'Review accepted' : verdict === 'reject' ? 'Review rejected' : 'Review pending', `review-badge review-${['accept','reject'].includes(verdict) ? verdict : 'pending'}`);
}
export function renderResearchRecord(record, metadata) {
 if (!Array.isArray(record) || record.some(o => !o || typeof o.id !== 'string' || typeof o.kind !== 'string' || !o.data || typeof o.data !== 'object' || Array.isArray(o.data))) throw new Error('Expected the array exported by record/export.json.');
 const byId = new Map(record.map(o => [o.id,o]));
 const newest = [...record].sort((a,b) => (b.created || 0)-(a.created || 0));
 const runs = newest.filter(o => o.kind === 'run'), results = newest.filter(o => o.kind === 'result');
 const summary = runs[0]?.data.summary;
 $('run-count').textContent = runs.length;
 $('review-count').textContent = results.filter(o => o.data.review).length;
 $('success-rate').textContent = rate(summary?.success_rate);
 $('placement-error').textContent = numeric(summary?.median_pos_err_cm) ? `${summary.median_pos_err_cm.toFixed(1)} cm` : '—';
 for (const id of ['record-status','snapshot-source','latest-run-detail']) {$(id).replaceChildren();$(id).hidden=true;}
 const decision = newest.find(o => o.kind === 'decision');
 const latestDecision = $('latest-decision'); latestDecision.replaceChildren(); latestDecision.hidden = !decision;
 if (decision) {
  latestDecision.append(el('div', `LATEST PI DECISION · ${decision.id}`, 'eyebrow'));
  latestDecision.append(el('p', String(decision.data.change || '').split('\n')[0]));
  detail(latestDecision,'Read the decision and limitations',decision.data.change);
  detail(latestDecision,'Why this decision was made',decision.data.rationale);
  detail(latestDecision,'Proposed next experiments',decision.data.next_experiment);
 }
 let active = 'all';
 const toolbar = $('record-filters'); toolbar.replaceChildren();
 for (const [kind,label] of filters) {
  const b = el('button', `${label} (${kind === 'all' ? newest.filter(o => kinds.has(o.kind)).length : newest.filter(o => o.kind === kind).length})`);
  b.type = 'button'; b.dataset.kind = kind; b.setAttribute('aria-pressed', String(kind === active));
  b.addEventListener('click', () => {active=kind; render();}); toolbar.append(b);
 }
 function references(parent, ids, label) {
  if (!Array.isArray(ids) || !ids.length) return;
  const links = el('div', undefined, 'record-references'); links.append(el('span',label));
  for (const id of [...new Set(ids)]) {
   const b = el('button', id, 'record-link'); b.type='button';
   b.disabled = !byId.has(id);
   b.addEventListener('click', () => {active='all';render(); document.getElementById(`record-${id}`)?.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});});
   links.append(b);
  } parent.append(links);
 }
 function render() {
  toolbar.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.kind === active)));
  const list = $('record-list'); list.replaceChildren();
  for (const o of newest.filter(o => kinds.has(o.kind) && (active === 'all' || o.kind === active))) {
   const d=o.data, row=el('article',undefined,'record-row'); row.id=`record-${o.id}`;
   const meta=el('div',undefined,'record-meta'); meta.append(el('span',o.kind.toUpperCase(),'tag'),el('strong',o.id));
   const body=el('div',undefined,'record-body');
   if(o.kind === 'run') {
    body.append(el('h3',d.method || 'Unknown method'));
    body.append(el('p',`${rate(d.summary?.success_rate)} observed success · n=${d.episodes ?? '—'} · 95% CI ${Array.isArray(d.summary?.success_ci95)?d.summary.success_ci95.map(rate).join('–'):'unavailable'}`));
    const tiers=d.summary?.by_tier || {};
    body.append(el('p',Object.entries(tiers).map(([tier,v]) => `${tier}: ${v.success}/${v.n}`).join(' · '),'record-secondary'));
    const linked = results.filter(r => r.data.run_ids?.includes(o.id));
    if(!linked.length)body.append(el('span','No linked review','review-badge review-pending'));
    references(body,linked.map(r=>r.id),'Result reviews:');
    references(body,[d.experiment_id],'Experiment:');
    detail(body,'Remaining idealizations',d.idealizations?.join(', '));
    detail(body,'Failure categories',d.summary?.failure_categories);
    detail(body,'Reproducibility', {commit:d.commit,fingerprint:d.run_fingerprint?.id,scenes:d.scene_ids,seeds:d.seeds,final_eval:!!d.final_eval});
   } else if(o.kind === 'result') {
    body.append(badge(d.review?.verdict));
    body.append(el('p',String(d.interpretation || 'No interpretation recorded').split('\n')[0]));
    detail(body,'Full analysis',d.interpretation);
    detail(body,'Reviewer verdict and reasons',d.review?.reasons);
    detail(body,'Failure analysis',d.failure_analysis);
    references(body,d.run_ids,'Runs:');
    references(body,[...(d.supports || []),...(d.refutes || [])],'Hypotheses:');
   } else if(o.kind === 'decision') {
    body.append(el('p',String(d.change || '').split('\n')[0]));
    detail(body,'Decision',d.change); detail(body,'Rationale',d.rationale); detail(body,'Next experiment',d.next_experiment);
    references(body,d.result_ids,'Reviewed results:');
   } else if(o.kind === 'hypothesis') {
    body.append(el('span',d.status || 'open','research-state'),el('p',d.statement || ''));
    detail(body,'Predicted effect',d.predicted_effect); references(body,d.evidence_ids,'Evidence:');
   } else if(o.kind === 'experiment') {
    body.append(el('span',d.status || 'candidate','research-state'),el('p',d.expected_learning || d.method || ''));
    detail(body,'Method and planned test',`${d.method || ''}\n${d.episodes ?? '—'} planned episodes\n${(d.scene_ids || []).join(', ')}`);
    detail(body,'Selection rationale',d.selection_rationale); references(body,d.hypothesis_ids,'Hypotheses:');
   } else {
    body.append(el('p',d.claim || ''),el('small',d.source || ''));
    detail(body,'Relevance',d.relevance);
   }
   body.append(el('small',`${o.author || 'Unknown author'} · ${numeric(o.created) ? new Date(o.created*1000).toLocaleString() : ''}`));
   row.append(meta,body); list.append(row);
  }
  if(!list.children.length)list.append(el('p','No objects in this category.','muted'));
 }
 render();
}
