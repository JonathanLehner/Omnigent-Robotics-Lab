(() => {
 if (!matchMedia('(pointer:fine)').matches) return;
 function gripper(p) {
 const d=7*p, l=5+d, r=27-d;
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="black" aria-hidden="true"><defs><mask id="grip-cut"><rect width="32" height="32" fill="white"/><path d="M14 7h4" stroke="black" stroke-width="1"/></mask></defs><g mask="url(#grip-cut)"><rect x="13" y="2" width="6" height="4" rx="1"/><rect x="9" y="5" width="14" height="8" rx="3"/><g fill="none" stroke="black" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M10 11 L${l-2} 21 M13 12 L${l+2} 21 M22 11 L${r+2} 21 M19 12 L${r-2} 21"/></g><rect x="${l-2.5}" y="20" width="5" height="10" rx="1.2"/><rect x="${r-2.5}" y="20" width="5" height="10" rx="1.2"/></g></svg>`;
}
 const style=document.createElement('style');
 style.textContent='html.gripper-active,html.gripper-active *{cursor:none!important}.robot-gripper-cursor{position:fixed;top:0;left:0;width:32px;height:32px;pointer-events:none;z-index:2147483647;visibility:hidden}.robot-gripper-cursor svg{display:block;width:100%;height:100%;transform:rotate(90deg);transform-origin:50% 50%}';
 document.head.append(style);
 const el=document.createElement('div'); el.className='robot-gripper-cursor';el.setAttribute('aria-hidden','true');el.innerHTML=gripper(0);document.body.append(el);
 let amount=0, target=0, raf=0, last=0;
 let demoPlayed=false, pointerSeen=false, demoTimers=[];
 function stopDemo(){demoTimers.forEach(clearTimeout);demoTimers=[];}
 const reduced=matchMedia('(prefers-reduced-motion:reduce)');
 function tick(time){const dt=Math.min(40,time-(last||time));last=time;amount+=(target-amount)*(1-Math.exp(-dt/45));if(Math.abs(target-amount)<.002)amount=target;el.innerHTML=gripper(amount);if(amount!==target)raf=requestAnimationFrame(tick);else{raf=0;last=0;}}
 function setGrip(value){target=value;if(reduced.matches){cancelAnimationFrame(raf);raf=0;amount=target;el.innerHTML=gripper(amount);}else if(!raf)raf=requestAnimationFrame(tick);}
 document.addEventListener('pointermove',e=>{pointerSeen=true;el.style.transform=`translate3d(${e.clientX-3}px,${e.clientY-16}px,0)`;el.style.visibility='visible';document.documentElement.classList.add('gripper-active');},{passive:true});
 document.addEventListener('pointerdown',e=>{if(e.button===0){stopDemo();demoPlayed=true;setGrip(1);}});
 window.addEventListener('pointerup',()=>setGrip(0));
 window.addEventListener('pointercancel',()=>setGrip(0));
 function reset(){stopDemo();setGrip(0);el.style.visibility='hidden';document.documentElement.classList.remove('gripper-active');}
 function loadPinches(){
  if(demoPlayed || reduced.matches || document.hidden)return;
  demoPlayed=true;
  demoTimers.push(setTimeout(() => {
   if(!pointerSeen){el.style.transform=`translate3d(${innerWidth*.7}px,${innerHeight*.55}px,0)`;el.style.visibility='visible';}
   [1,0,1,0].forEach((value,i)=>demoTimers.push(setTimeout(()=>setGrip(value),i*330)));
   demoTimers.push(setTimeout(()=>{demoTimers=[];if(!pointerSeen)el.style.visibility='hidden';},1500));
  },750));
 }
 document.addEventListener('opal:intro-complete',loadPinches,{once:true});
 if(!document.querySelector('script[src*="opal-intro"]'))loadPinches();
 document.documentElement.addEventListener('pointerleave',reset);
 window.addEventListener('blur',reset);
 document.addEventListener('visibilitychange',()=>{if(document.hidden)reset();});
})();