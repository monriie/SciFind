// Make module-load failures visible, including opening the app through file://.
window.addEventListener('error', event => {
  const root=document.getElementById('root');
  if (!root || !root.querySelector('.boot-screen')) return;
  const box=document.createElement('div');
  box.className='boot-screen';
  const title=document.createElement('h1');title.textContent='Aplikasi belum dapat dimuat';
  const message=document.createElement('p');message.textContent=event.message || 'Periksa seluruh file aplikasi dan gunakan server HTTP/HTTPS.';
  const hint=document.createElement('p');hint.textContent='Untuk preview: python -m http.server 8765 --directory interface';
  box.append(title,message,hint);root.replaceChildren(box);
});
