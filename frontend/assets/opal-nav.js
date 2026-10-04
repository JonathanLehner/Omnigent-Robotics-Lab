(() => {
 const nav = document.querySelector('.topbar');
 if (!nav) return;
 let timer;
 function flash() {
  if (matchMedia('(prefers-reduced-motion:reduce)').matches) return;
  nav.classList.remove('nav-flash');
  void nav.offsetWidth;
  nav.classList.add('nav-flash');
  clearTimeout(timer);
  timer = setTimeout(() => nav.classList.remove('nav-flash'), 2000);
 }
 // Play once the intro reveals the page so the navbar animation is visible.
 document.addEventListener('opal:intro-complete', flash);
})();
