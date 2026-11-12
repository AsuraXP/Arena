#!/usr/bin/env python3
"""Tiny static server with HTTP Range support (needed for video seeking).
Serves:
  /               -> site/index.html
  /media/poster.jpg, /media/demo.mp4  (from demo/)
  /site/*         -> site files
"""
import http.server, os, re, socketserver, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def translate_path(self, path):
        p = path.split("?")[0].split("#")[0]
        if p == "/" or p == "":
            p = "/site/index.html"
        elif p == "/media/demo.mp4":
            p = "/out.mp4"
        elif p == "/media/poster.jpg":
            p = "/site/poster.jpg"
        elif not p.startswith("/site/"):
            p = "/site" + p
        parts = [x for x in p.split("/") if x not in ("", ".", "..")]
        return os.path.join(ROOT, *parts)

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "Not found")
            return None
        if not rng:
            return super().send_head()
        try:
            size = os.path.getsize(path)
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            f = open(path, "rb")
            f.seek(start)
            self._remaining = end - start + 1
            return f
        except Exception as e:
            self.send_error(500, str(e))
            return None

    def copyfile(self, source, outputfile):
        rem = getattr(self, "_remaining", None)
        if rem is None:
            return super().copyfile(source, outputfile)
        while rem > 0:
            chunk = source.read(min(64 * 1024, rem))
            if not chunk:
                break
            outputfile.write(chunk)
            rem -= len(chunk)


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with S(("0.0.0.0", PORT), H) as httpd:
        print(f"serving {ROOT} on 0.0.0.0:{PORT}", flush=True)
        httpd.serve_forever()
