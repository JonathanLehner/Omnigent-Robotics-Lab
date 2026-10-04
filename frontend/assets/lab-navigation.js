(() => {
 const screens = new Set(['workspace','research','team']);
 function show(screen, updateHistory = true) {
  const next = screens.has(screen) ? screen : 'home';
  document.body.dataset.screen = next;
  document.querySelector('main').scrollTop = 0;
  document.querySelectorAll('.lab-nav a').forEach(link => {
   if (link.hash === '#' + next) link.setAttribute('aria-current','page');
   else link.removeAttribute('aria-current');
  });
  if (updateHistory) history.pushState(null, '', next === 'home' ? location.pathname + location.search : '#' + next);
  const target = next === 'home' ? document.querySelector('#enter-lab') : document.querySelector('#' + next + ' h2');
  if (target) { if (next !== 'home') target.setAttribute('tabindex','-1'); target.focus({preventScroll:true}); }
 }
 document.querySelector('#enter-lab').addEventListener('click', () => show('workspace'));
 document.querySelectorAll('.lab-nav a').forEach(link => link.addEventListener('click', event => {event.preventDefault(); show(link.hash.slice(1));}));
 document.querySelectorAll('[data-home]').forEach(link => link.addEventListener('click', event => {event.preventDefault();show('home');}));
 window.addEventListener('popstate', () => show(location.hash.slice(1), false));
 if (screens.has(location.hash.slice(1))) show(location.hash.slice(1), false);
})();
