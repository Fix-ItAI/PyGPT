(function(){
  const vscode = acquireVsCodeApi ? acquireVsCodeApi() : null;
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const messages = document.getElementById('messages');

  function appendMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'msg ' + role;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  send.addEventListener('click', () => {
    const txt = input.value.trim();
    if (!txt) return;
    appendMessage('user', txt);
    if (vscode) vscode.postMessage({ type: 'send', text: txt });
    input.value = '';
  });

  window.addEventListener('message', event => {
    const msg = event.data;
    if (msg.type === 'response') {
      appendMessage('bot', msg.text);
    }
  });
})();
