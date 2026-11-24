import * as vscode from 'vscode';
import { randomBytes } from 'node:crypto';

const VIEW_ID = 'arena.sidebar';
const DEFAULT_URL = 'https://arena.ai';

export function activate(context: vscode.ExtensionContext): void {
  const provider = new ArenaViewProvider();
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW_ID, provider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    vscode.commands.registerCommand('arena.reload', () => provider.reload()),
    vscode.commands.registerCommand('arena.openInBrowser', () => provider.openInBrowser()),
  );
}

class ArenaViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this.getHtml(this.getConfiguredUrl());
    view.onDidDispose(() => {
      if (this.view === view) this.view = undefined;
    });
  }

  reload(): void {
    if (!this.view) {
      void vscode.commands.executeCommand('workbench.view.extension.arena');
      return;
    }
    this.view.webview.html = this.getHtml(this.getConfiguredUrl());
  }

  async openInBrowser(): Promise<void> {
    const url = this.getConfiguredUrl();
    await vscode.env.openExternal(vscode.Uri.parse(url));
  }

  private getConfiguredUrl(): string {
    const configured = vscode.workspace.getConfiguration().get<string>('arena.sidebar.url', DEFAULT_URL).trim();
    try {
      const parsed = new URL(configured);
      if (parsed.protocol === 'https:') return parsed.toString();
    } catch {
      // Display the safe default when the setting is malformed.
    }
    void vscode.window.showWarningMessage('Arena Sidebar requires an HTTPS URL. Using https://arena.ai instead.');
    return DEFAULT_URL;
  }

  private getHtml(url: string): string {
    const nonce = getNonce();
    const safeUrl = escapeAttribute(url);
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src https:; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
  <title>Arena Chat</title>
  <style nonce="${nonce}">
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: var(--vscode-sideBar-background); color: var(--vscode-foreground); font: 13px var(--vscode-font-family); }
    .shell { display: grid; grid-template-rows: auto 1fr; width: 100%; height: 100%; }
    .toolbar { min-height: 38px; padding: 6px 10px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--vscode-panel-border); }
    .brand { font-weight: 600; flex: 1; }
    button { color: var(--vscode-foreground); background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px 7px; cursor: pointer; }
    button:hover { background: var(--vscode-toolbar-hoverBackground); }
    .frame-wrap { position: relative; min-height: 0; }
    iframe { width: 100%; height: 100%; border: 0; background: #fff; }
  </style>
</head>
<body>
  <main class="shell">
    <header class="toolbar">
      <span class="brand">Arena</span>
      <button id="reload" type="button" title="Reload Arena">Reload</button>
    </header>
    <section class="frame-wrap">
      <iframe id="arena" src="${safeUrl}" title="Arena.ai" allow="clipboard-read; clipboard-write; microphone" referrerpolicy="strict-origin-when-cross-origin"></iframe>
    </section>
  </main>
  <script nonce="${nonce}">
    document.getElementById('reload').addEventListener('click', () => {
      const frame = document.getElementById('arena');
      frame.src = frame.src;
    });
  </script>
</body>
</html>`;
  }
}

function escapeAttribute(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function getNonce(): string {
  return randomBytes(16).toString('base64');
}

export function deactivate(): void {}
