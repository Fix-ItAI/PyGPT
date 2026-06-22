const vscode = require('vscode');
const { execFile } = require('child_process');

function getWorkspacePath() {
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length > 0) {
    return folders[0].uri.fsPath;
  }
  return process.cwd();
}

function runPyGptCommand(command, args) {
  return new Promise((resolve, reject) => {
    const cmd = 'pygpt';
    execFile(cmd, [command, ...args], { cwd: getWorkspacePath() }, (error, stdout, stderr) => {
      if (error) {
        return reject({ error, stderr: stderr && stderr.toString() });
      }
      resolve(stdout && stdout.toString());
    });
  });
}

class PyGptSidebar {
  constructor(context) {
    this.context = context;
    this._view = null;
  }

  resolveWebviewView(webviewView) {
    this._view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this._getHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (msg) => {
      if (msg.type === 'send') {
        const text = msg.text;
        // Check for local model preference
        const cfg = vscode.workspace.getConfiguration('pygpt');
        const useLocal = cfg.get('useLocalModel', false);
        const localModelPath = cfg.get('localModelPath', 'model.pth');
        try {
          if (useLocal) {
            this._appendMessage('user', text);
            const args = ['--model-path', localModelPath, '--prompt', text, '--max-new-tokens', '150'];
            const out = await runPyGptCommand('generate', args);
            this._appendMessage('bot', out || '');
            webviewView.webview.postMessage({ type: 'response', text: out || '' });
          } else {
            // Call external API (OpenAI-compatible)
            const apiKey = cfg.get('apiKey', '');
            const provider = cfg.get('apiProvider', 'openai');
            if (!apiKey) {
              webviewView.webview.postMessage({ type: 'response', text: 'No API key set. Use the command "PyGPT: Set API Key".' });
              return;
            }
            this._appendMessage('user', text);
            const responseText = await this._callOpenAI(apiKey, text, cfg.get('model', 'gpt-3.5-turbo'));
            this._appendMessage('bot', responseText);
            webviewView.webview.postMessage({ type: 'response', text: responseText });
          }
        } catch (e) {
          const msgErr = (e && e.stderr) ? e.stderr : (e && e.message) ? e.message : String(e);
          webviewView.webview.postMessage({ type: 'response', text: `Error: ${msgErr}` });
        }
      }
    });
  }

  _appendMessage(role, text) {
    // store in workspaceState for session memory
    const messages = this.context.workspaceState.get('pygpt.messages', []);
    messages.push({ role, text, ts: Date.now() });
    this.context.workspaceState.update('pygpt.messages', messages.slice(-200));
  }

  async _callOpenAI(apiKey, prompt, model) {
    // Use node fetch if available, otherwise built-in https
    const fetch = global.fetch || (await import('node-fetch')).default;
    const body = {
      model: model,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 256
    };
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`API error: ${res.status} ${txt}`);
    }
    const data = await res.json();
    const out = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
    return out || '';
  }

  _getHtml(webview) {
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'main.js'));
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'styles.css'));
    return `<!doctype html>
      <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="stylesheet" href="${styleUri}">
      </head>
      <body>
        <div id="messages"></div>
        <div id="controls">
          <input id="input" placeholder="Type a message" />
          <button id="send">Send</button>
        </div>
        <script src="${scriptUri}"></script>
      </body>
      </html>`;
  }
}

function activate(context) {
  const sidebarProvider = new PyGptSidebar(context);
  context.subscriptions.push(vscode.window.registerWebviewViewProvider('pygpt-sidebar', sidebarProvider));

  context.subscriptions.push(vscode.commands.registerCommand('pygpt.setApiKey', async () => {
    const key = await vscode.window.showInputBox({ prompt: 'Enter API key for external provider (will be stored in workspace settings)' });
    if (key !== undefined) {
      await vscode.workspace.getConfiguration('pygpt').update('apiKey', key, vscode.ConfigurationTarget.Workspace);
      vscode.window.showInformationMessage('PyGPT: API key saved to workspace settings.');
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('pygpt.setLocalModel', async () => {
    const path = await vscode.window.showInputBox({ prompt: 'Enter local model file path (relative to workspace root)', value: 'model.pth' });
    if (path !== undefined) {
      await vscode.workspace.getConfiguration('pygpt').update('localModelPath', path, vscode.ConfigurationTarget.Workspace);
      await vscode.workspace.getConfiguration('pygpt').update('useLocalModel', true, vscode.ConfigurationTarget.Workspace);
      vscode.window.showInformationMessage('PyGPT: Local model path saved and local model enabled.');
    }
  }));

  // Keep backwards-compatible simple commands
  context.subscriptions.push(vscode.commands.registerCommand('pygpt.train', async () => {
    const files = await vscode.window.showInputBox({ prompt: 'Enter file paths or directories for training data, separated by spaces', placeHolder: 'data/text1.txt data/text2.txt' });
    if (files === undefined) return;
    const args = files.trim() ? files.trim().split(/\s+/) : [];
    try {
      const out = await runPyGptCommand('train', args);
      vscode.window.showInformationMessage('pygpt train finished');
      console.log(out);
    } catch (e) {
      vscode.window.showErrorMessage('pygpt train failed: ' + (e && e.stderr ? e.stderr : e && e.error && e.error.message ? e.error.message : String(e)));
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('pygpt.generate', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Enter prompt text for generation', placeHolder: 'Once upon a time' });
    if (prompt === undefined) return;
    try {
      const out = await runPyGptCommand('generate', ['--prompt', prompt]);
      vscode.window.showInformationMessage('Generation complete. See console for output.');
      console.log(out);
    } catch (e) {
      vscode.window.showErrorMessage('pygpt generate failed: ' + (e && e.stderr ? e.stderr : e && e.error && e.error.message ? e.error.message : String(e)));
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('pygpt.download', async () => {
    const output = await vscode.window.showInputBox({ prompt: 'Enter output path for the default dataset', value: 'PyGPT/data/default.txt' });
    if (output === undefined) return;
    try {
      const out = await runPyGptCommand('download', ['--output', output]);
      vscode.window.showInformationMessage('Downloaded default dataset.');
      console.log(out);
    } catch (e) {
      vscode.window.showErrorMessage('pygpt download failed: ' + (e && e.stderr ? e.stderr : e && e.error && e.error.message ? e.error.message : String(e)));
    }
  }));
}

function deactivate() {}

module.exports = {
  activate,
  deactivate
};
