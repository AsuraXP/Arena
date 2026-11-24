# Arena Sidebar for VS Code

A small VS Code extension prototype that adds an **Arena** icon to the Activity Bar and displays `https://arena.ai` in a sidebar webview. The page is embedded in VS Code rather than opened in a normal browser window.

## Run locally

1. Open this `vscode-arena-sidebar` folder in VS Code.
2. Run `npm install`.
3. Press **F5** to launch an Extension Development Host.
4. Select the **Arena** icon in the Activity Bar.

To compile from a terminal, run `npm run compile`.

## Important embedding limitation

This extension hosts the Arena website in an iframe. Whether it works depends on Arena's current security headers, login/cookie behavior, and VS Code's embedded webview environment. If Arena sends a `Content-Security-Policy: frame-ancestors` or `X-Frame-Options` policy that disallows framing, the page cannot be made to load by changing this extension; the website owner would need to permit embedding, or provide a supported API/extension integration. The extension includes an explicit browser fallback command for diagnosis, but the intended experience is the in-sidebar website.

The URL is configurable with `arena.sidebar.url` (HTTPS only). The current checkout is a prototype and has not yet been verified inside VS Code against Arena's live embedding policy.
