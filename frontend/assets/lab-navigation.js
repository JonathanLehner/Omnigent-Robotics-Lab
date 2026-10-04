(() => {
 const screens = new Set(['workspace','demos','research','team']);
 const stages = [
  ['Principal investigator', 'Choosing direction and setting the budget'],
  ['Literature scout', 'Finding relevant research and evidence'],
  ['Method & scene designers', 'Proposing a method and building test scenes'],
  ['Planner & robotics engineer', 'Choosing a test and preparing the robot method'],
  ['Experiment runner', 'Testing the method in simulation'],
  ['Analyst & reviewer', 'Interpreting results and checking the evidence'],
  ['Principal investigator', 'Updating the plan for the next experiment']
 ];
 const dialog = document.createElement('dialog');
 dialog.className = 'workflow-loading';
 dialog.setAttribute('aria-labelledby','workflow-title');
 dialog.innerHTML = '<div class="workflow-terminal"><div class="terminal-chrome" aria-hidden="true"><span class="terminal-new">＋</span><span class="terminal-app-title">opal labs</span><span class="terminal-window-icons"><i>⌕</i><i>☰</i><i>−</i><i>□</i><i>×</i></span></div><div class="terminal-tab" aria-hidden="true">research@opal-labs: ~/robotics-lab <span>×</span></div><div class="workflow-boot-brand">OPAL LABS / RESEARCH OS</div><div class="workflow-heading"><span class="workflow-spinner" aria-hidden="true"></span><h2 id="workflow-title">what would you like to research?</h2></div><div class="terminal-command"><span>opal&gt;</span><p>Can a simulated robot assemble a structure from a reference picture without task-specific training?</p></div><ol class="workflow-steps"></ol><div class="workflow-bottom"><span class="workflow-status" role="status" aria-live="polite"></span></div></div>';
 const list = dialog.querySelector('ol');
 stages.forEach(([name,description]) => {
  const row = document.createElement('li');
  const icon = document.createElement('span'); icon.className = 'workflow-icon'; icon.setAttribute('aria-hidden','true');
  const text = document.createElement('div');
  const label = document.createElement('strong'); label.textContent = name;
  const detail = document.createElement('span'); detail.textContent = description;
  const state = document.createElement('span');state.className = 'workflow-row-state';state.setAttribute('aria-hidden','true');
  text.append(label,detail); row.append(icon,text,state); list.append(row);
 });
 document.body.append(dialog);
 let pending = null;
 let transition = null;
 function cancelPreview() {
  if (pending !== null) clearTimeout(pending);
  pending = null;
  transition?.remove(); transition = null;
  document.body.classList.remove('is-revealing-workspace');
  if (dialog.open) dialog.close();
  dialog.classList.remove('is-finishing');
 }
 function show(screen, updateHistory = true) {
  cancelPreview();
  document.querySelectorAll('#demos video').forEach(video => video.pause());
  const next = screens.has(screen) ? screen : 'home';
  document.body.dataset.screen = next;
  window.dispatchEvent(new CustomEvent("opal:screen-change", {detail: {screen: next}}));
  document.querySelector('main').scrollTop = 0;
  document.querySelectorAll('.lab-nav a').forEach(link => {
   if (link.hash === '#' + next) link.setAttribute('aria-current','page');
   else link.removeAttribute('aria-current');
  });
  if (updateHistory) history.pushState(null, '', next === 'home' ? location.pathname + location.search : '#' + next);
  const target = next === 'home' ? document.querySelector('#enter-lab') : document.querySelector('#' + next + ' h2');
  if (target) { if (next !== 'home') target.setAttribute('tabindex','-1'); target.focus({preventScroll:true}); }
 }
 function preview() {
  if (dialog.open) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || typeof dialog.showModal !== 'function') { show('workspace'); return; }
  const rows = [...list.children];
  rows.forEach(row => { row.className = ''; row.querySelector('.workflow-icon').textContent = ''; });
  dialog.classList.remove('is-finishing');
  dialog.showModal();
  let step = 0;
  rows.forEach(row => {row.querySelector('div > span').textContent='';row.querySelector('.workflow-row-state').textContent='';});
  function advance() {
   if (!dialog.open) return;
   if (step === stages.length) {
    dialog.querySelector('.workflow-status').textContent = 'RESEARCH LOOP INITIALIZED — ENTERING WORKSPACE';
    dialog.classList.add('is-finishing');
    transition = document.createElement('div');
    transition.className = 'workspace-transition';
    transition.setAttribute('aria-hidden','true');
    dialog.append(transition);
    pending = setTimeout(() => {
     const layer = transition;
     transition = null;
     document.body.append(layer);
     show('workspace');
     transition = layer;
     document.body.classList.add('is-revealing-workspace');
     layer.classList.add('is-revealing');
     pending = setTimeout(() => {
      layer.remove(); transition = null; pending = null;
      document.body.classList.remove('is-revealing-workspace');
     },750);
    },650);return;
   }
   const row = rows[step], description=stages[step][1];
   row.classList.add('active');
   row.querySelector('.workflow-icon').textContent='>';
   row.querySelector('.workflow-row-state').textContent='RUNNING';
   dialog.querySelector('.workflow-status').textContent=stages[step][0]+' · '+(step+1)+' / '+stages.length;
   let char=0;
   function type() {
    if(!dialog.open)return;
    row.querySelector('div > span').textContent=description.slice(0,++char);
    if(char<description.length){pending=setTimeout(type,12);return;}
    pending=setTimeout(() => {
     row.classList.remove('active');row.classList.add('done');
     row.querySelector('.workflow-row-state').textContent='READY';
     step++;pending=setTimeout(advance,120);
    },130);
   }
   type();
  }
  advance();
  dialog.setAttribute('tabindex','-1');dialog.focus();
 }
 document.querySelector('#enter-lab').addEventListener('click', preview);
 dialog.addEventListener('cancel',event => {event.preventDefault();show('workspace');});
 document.querySelectorAll('.lab-nav a').forEach(link => link.addEventListener('click', event => {event.preventDefault(); show(link.hash.slice(1));}));
 document.querySelectorAll('[data-home]').forEach(link => link.addEventListener('click', event => {event.preventDefault();show('home');}));
 window.addEventListener('popstate', () => show(location.hash.slice(1), false));
 if (screens.has(location.hash.slice(1))) show(location.hash.slice(1), false);
})();
