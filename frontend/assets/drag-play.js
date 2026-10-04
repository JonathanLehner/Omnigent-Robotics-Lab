(() => {
 const finePointer = matchMedia('(pointer:fine)').matches;
 // Wrap text nodes individually, preserving spaces, line breaks, and icons.
 const textBlocks = document.querySelectorAll('.hero .eyebrow,.hero-bottom p,.powered-by-label,.section-heading h2,.plan-panel .panel-title');
 if (finePointer) textBlocks.forEach(block => {
  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
   acceptNode: node => node.parentElement.closest('svg,[aria-hidden="true"]') || !/[\p{L}\p{N}]/u.test(node.textContent)
    ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(node => {
   const fragment = document.createDocumentFragment();
   node.textContent.split(/(\s+)/u).forEach(part => {
    if (!part) return;
    if (/\S/u.test(part)) {
     const word = document.createElement('span');
     word.className = 'play-word';word.textContent = part;fragment.append(word);
    } else fragment.append(document.createTextNode(part));
   });
   node.replaceWith(fragment);
  });
 });
 const elements = [...document.querySelectorAll(finePointer ? '.play-word,.landing-logo,.powered-brand,#enter-lab,#download-scene,.landing-block' : '.landing-block')];
 const positions = new Map(elements.map(el => [el, {x:0,y:0}]));
 let drag = null, suppress = null, blockLayer = 3;
 const raiseBlock = el => {if(el.classList.contains('landing-block')) el.style.zIndex = String(++blockLayer);};
 const place = (el,pos) => {positions.set(el,pos);el.style.transform = `translate(${pos.x}px,${pos.y}px)`;};
 function reset(el) {if(el.classList.contains('landing-block')) el.style.zIndex = '';el.classList.add('is-snapping');place(el,{x:0,y:0});setTimeout(()=>el.classList.remove('is-snapping'),260);}
 elements.forEach(el => {
  el.classList.add('play-draggable');
  el.title = el.classList.contains('landing-block') ? 'Drag and drop to arrange blocks. Escape resets the stack.' : 'Drag to move; return near the starting point to snap. Escape resets the layout.';
  el.addEventListener('pointerdown',e => {
   if(e.button !== 0 || (e.pointerType === 'touch' && !el.classList.contains('landing-block'))) return;
   const pos = positions.get(el), rect = el.getBoundingClientRect();
   drag = {el,id:e.pointerId,startX:e.clientX,startY:e.clientY,pos:{...pos},rect,moved:false};
   el.classList.remove('is-snapping');
   if(el.classList.contains('landing-block')) e.preventDefault();
  });
  if(el.classList.contains('landing-block')) el.addEventListener('keydown',e => {
   const directions = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
   if(e.key === 'Home'){e.preventDefault();reset(el);return;}
   const vector = directions[e.key]; if(!vector)return;
   e.preventDefault();const pos=positions.get(el),rect=el.getBoundingClientRect(),amount=e.shiftKey?30:10;
   el.classList.remove('is-snapping');
   raiseBlock(el);
   place(el,{x:pos.x+Math.max(-rect.left,Math.min(innerWidth-rect.right,vector[0]*amount)),y:pos.y+Math.max(-rect.top,Math.min(innerHeight-rect.bottom,vector[1]*amount))});
  });
 });
 document.addEventListener('pointermove',e => {
  if(!drag || e.pointerId !== drag.id) return;
  const dx=e.clientX-drag.startX,dy=e.clientY-drag.startY;
  if(!drag.moved && Math.hypot(dx,dy)<6) return;
  if(!drag.moved){drag.moved=true;raiseBlock(drag.el);drag.el.classList.add('is-grabbed');drag.el.setPointerCapture?.(e.pointerId);}
  e.preventDefault();
  const x=drag.pos.x+Math.max(-drag.rect.left,Math.min(innerWidth-drag.rect.right,dx));
  const y=drag.pos.y+Math.max(-drag.rect.top,Math.min(innerHeight-drag.rect.bottom,dy));
  place(drag.el,{x,y});
 },{passive:false});
 function end(e,cancel=false){
  if(!drag || (e && e.pointerId !== drag.id)) return;
  const current=drag;drag=null;current.el.classList.remove('is-grabbed');
  if(current.el.hasPointerCapture?.(current.id))current.el.releasePointerCapture(current.id);
  if(current.moved){
   suppress={el:current.el,until:performance.now()+500};
   if(cancel)place(current.el,current.pos);
   else if(!current.el.classList.contains('landing-block') && Math.hypot(positions.get(current.el).x,positions.get(current.el).y)<42)reset(current.el);
  }
 }
 document.addEventListener('pointerup',e=>end(e));
 document.addEventListener('pointercancel',e=>end(e,true));
 document.addEventListener('click',e=>{
  if(suppress && performance.now()<suppress.until && suppress.el.contains(e.target)){e.preventDefault();e.stopImmediatePropagation();suppress=null;}
 },true);
 document.addEventListener('keydown',e=>{if(e.key==='Escape'){end(null,true);elements.forEach(reset);}});
 window.addEventListener('blur',()=>end(null,true));
 window.addEventListener('resize',()=>{end(null,true);elements.forEach(reset);});
})();
