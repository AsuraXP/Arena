#!/usr/bin/env python3
"""Static server for the demo player, built for proxied/live-preview hosting.

  GET /            -> site/index.html   (the player page)
  GET /media/demo.mp4, /media/poster.jpg
  GET /site/*      -> site assets
  GET /healthz     -> "ok"

Range requests are implemented properly so the video can seek/scrub.
"""
import http.server
import os
import re
import socketserver
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".ttf": "font/ttf",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "demo-player/1.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    # --- quieter, safer logging -------------------------------------------
    def log_message(self, fmt, *args):
        try:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))
            sys.stderr.flush()
        except Exception:
            pass

    def log_error(self, fmt, *args):
        pass

    # --- routing -----------------------------------------------------------
    def translate_path(self, path):
        p = path.split("?")[0].split("#")[0]
        if p in ("", "/", "/index.html"):
            p = "/site/index.html"
        elif p == "/healthz":
            p = "/__health"
        elif p == "/media/demo.mp4":
            p = "/out.mp4"
        elif p == "/media/demo2.mp4":
            p = "/demo2.mp4"
        elif p == "/media/poster2.jpg":
            p = "/site/poster2.jpg"
        elif p == "/media/poster.jpg":
            p = "/site/poster.jpg"
        elif not p.startswith("/site/"):
            p = "/site" + p
        safe = [x for x in p.split("/") if x not in ("", ".", "..")]
        return os.path.join(ROOT, *safe)

    # --- shared headers ----------------------------------------------------
    def _common(self):
        # Preview proxies embed the app in an iframe: never block framing,
        # and stay permissive about cross-origin subresources.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("X-Frame-Options", "ALLOWALL")
        self.send_header("Connection", "close")
        self.close_connection = True

    def _ctype(self, path):
        return MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")

    def send_head(self):
        path = self.translate_path(self.path)

        if path.endswith("__health"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._common()
            self.end_headers()
            self.wfile.write(body)
            return None

        if not os.path.isfile(path):
            body = b"404 not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self._common()
            self.end_headers()
            self.wfile.write(body)
            return None

        try:
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            partial = False
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:  # suffix range: bytes=-500
                        start = max(size - int(m.group(2)), 0)
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */%d" % size)
                        self.send_header("Content-Length", "0")
                        self._common()
                        self.end_headers()
                        return None
                    partial = True

            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", self._ctype(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            # never cache the page or JSON, so edits show up on reload
            no_store = path.endswith((".html", ".json"))
            self.send_header("Cache-Control", "no-store" if no_store else "public, max-age=300")
            self._common()
            self.end_headers()

            f = open(path, "rb")
            f.seek(start)
            self._remaining = end - start + 1
            return f
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass
            return None

    def copyfile(self, source, outputfile):
        rem = getattr(self, "_remaining", None)
        try:
            if rem is None:
                return super().copyfile(source, outputfile)
            while rem > 0:
                chunk = source.read(min(64 * 1024, rem))
                if not chunk:
                    break
                outputfile.write(chunk)
                rem -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self._remaining = None

    # browsers sometimes probe with HEAD
    def do_HEAD(self):
        f = self.send_head()
        if f:
            f.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


def main():
    with Server((HOST, PORT), Handler) as httpd:
        print("demo player: http://%s:%d  (root=%s)" % (HOST, PORT, ROOT), flush=True)
        print("open  /  to watch, /healthz to probe", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
