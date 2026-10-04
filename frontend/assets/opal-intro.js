(() => {
 const base = new URL('.', document.currentScript.src);
 const logo = new URL('opal-logo.png', base).href;
 const css = document.createElement('style');
 css.textContent = `
 .opal-intro{--logo-width:min(250px,65vw);position:fixed;inset:0;z-index:99999;overflow:hidden;background:#fafcff;color:#181b22;isolation:isolate;transition:opacity .65s ease,visibility .65s}
 .opal-intro.is-finished{opacity:0;visibility:hidden;pointer-events:none}
 .opal-intro__backdrop{position:absolute;inset:-15%;background:radial-gradient(ellipse at 15% 95%,#72d5ff 0%,#2266d0 20%,transparent 50%),radial-gradient(ellipse at 100% 0%,#1c4fa3 0%,transparent 52%),linear-gradient(145deg,#00102f 10%,#092759 65%,#16478f);animation:opal-backdrop 1.1s .15s ease both}
 .opal-intro__shade{position:absolute;inset:0;background:radial-gradient(ellipse at 100% 100%,#e9f1ff 0%,transparent 60%);pointer-events:none}
 .opal-intro__sweep{position:absolute;inset:-100%;background:linear-gradient(125deg,transparent 43%,#b9deff33 49%,transparent 55%);animation:opal-sweep 2.5s ease both;pointer-events:none}
 .opal-intro__stage{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;margin:0;padding:0}
 .opal-intro__lockup{position:relative;flex:0 0 var(--logo-width);width:var(--logo-width);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:22px;box-sizing:content-box;margin:0;padding:0}
 .opal-intro__bricks{position:relative;width:142px;height:74px;flex-shrink:0}
 .opal-intro__brick{position:absolute;width:66px;height:32px;border-radius:5px;background:#181b22;animation:opal-drop .65s cubic-bezier(.22,1,.36,1) both}
 .opal-intro__brick:nth-child(1){left:38px;top:0;background:#163b85;animation-delay:1.2s}
 .opal-intro__brick:nth-child(2){left:0;top:42px;animation-delay:.72s}
 .opal-intro__brick:nth-child(3){left:76px;top:42px;animation-delay:.96s}
 .opal-intro__wordmark{display:block;flex:none;width:var(--logo-width);height:calc(var(--logo-width) * .780239);margin:0;padding:0;overflow:hidden;animation:opal-logo-in .65s 2.2s ease both}
 .opal-intro__logo{display:block;width:100%;height:100%;max-width:none;margin:0;padding:0;overflow:hidden}
 @keyframes opal-drop{0%{opacity:0;transform:translateY(-100px)}15%{opacity:1}75%{opacity:1;transform:translateY(3px)}100%{opacity:1;transform:translateY(0)}}
 .opal-intro__sparkle{position:absolute;right:-12%;top:43%;width:clamp(22px,3.6vw,40px);height:auto;fill:#163b85;animation:opal-sparkle .8s 2.65s cubic-bezier(.22,1,.36,1) both}
 .opal-intro__skip{position:absolute;right:24px;bottom:24px;background:#fafcff;border:1px solid #d8e0ee;color:#163b85;border-radius:30px;padding:9px 17px;font:13px system-ui;cursor:pointer}.opal-intro__skip:focus-visible{outline:2px solid #163b85;outline-offset:4px}
 @keyframes opal-logo-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
 @keyframes opal-sparkle{0%{opacity:0;transform:scale(0) rotate(-80deg)}55%{opacity:1;transform:scale(1.16) rotate(8deg)}100%{opacity:1;transform:scale(1) rotate(0)}}
 @keyframes opal-backdrop{0%,10%{opacity:1;transform:scale(1.03)}100%{opacity:0;transform:scale(1)}}
 @keyframes opal-sweep{from{transform:translateX(-28%);opacity:0}25%{opacity:1}to{transform:translateX(28%);opacity:0}}
 @keyframes opal-progress{from{transform:scaleX(0)}to{transform:scaleX(1)}}
 @media(prefers-reduced-motion:reduce){.opal-intro *{animation:none!important}.opal-intro__sweep,.opal-intro__backdrop{display:none}.opal-intro{transition:none}}
 `;
 document.head.append(css);
 let active=null;
 window.playOpalIntro = () => {
  if(active)active.finish();
  const el=document.createElement('section');el.className='opal-intro';el.setAttribute('role','dialog');el.setAttribute('aria-modal','true');el.setAttribute('aria-label','Opal Labs introduction');
  el.innerHTML='<div class="opal-intro__backdrop"></div><div class="opal-intro__shade"></div><div class="opal-intro__sweep"></div><div class="opal-intro__stage"><div class="opal-intro__lockup"><div class="opal-intro__bricks" aria-hidden="true"><i class="opal-intro__brick"></i><i class="opal-intro__brick"></i><i class="opal-intro__brick"></i></div><div class="opal-intro__wordmark"><svg class="opal-intro__logo" viewBox="1106 171 587 458" preserveAspectRatio="xMidYMid meet" role="img" aria-label="opal labs"><image class="opal-intro__source" x="0" y="0" width="1962" height="802"/></svg></div><svg class="opal-intro__sparkle" viewBox="0 0 40 40" aria-hidden="true"><path d="M20 0C22 13 27 18 40 20C27 22 22 27 20 40C18 27 13 22 0 20C13 18 18 13 20 0Z"/></svg></div></div><button class="opal-intro__skip">Skip intro</button>';
  el.querySelector('.opal-intro__source').setAttribute('href',logo);
  const siblings=[...document.body.children].filter(n=>n.tagName!=='SCRIPT'&&n.tagName!=='STYLE');const states=siblings.map(n=>n.inert);siblings.forEach(n=>n.inert=true);
  const previous=document.activeElement,oldOverflow=document.documentElement.style.overflow;document.documentElement.style.overflow='hidden';document.body.append(el);
  const skip=el.querySelector('button');skip.focus({preventScroll:true});let finished=false,timer;
  function key(e){if(e.key==='Escape')finish();if(e.key==='Tab'){e.preventDefault();skip.focus();}}
  function finish(){if(finished)return;finished=true;clearTimeout(timer);document.removeEventListener('keydown',key);siblings.forEach((n,i)=>n.inert=states[i]);document.documentElement.style.overflow=oldOverflow;el.classList.add('is-finished');if(previous?.isConnected)previous.focus({preventScroll:true});setTimeout(()=>el.remove(),700);active=null;document.dispatchEvent(new CustomEvent('opal:intro-complete'));}
  active={finish};skip.onclick=finish;document.addEventListener('keydown',key);
  timer=setTimeout(finish,matchMedia('(prefers-reduced-motion:reduce)').matches?700:3800);
 };
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',window.playOpalIntro,{once:true});else window.playOpalIntro();
})();
