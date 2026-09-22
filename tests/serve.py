"""A range-capable static file server, run as a subprocess by the `ct_origin` fixture.

`python -m http.server` answers every GET with the whole file, which is wrong for a SHARDED zarr: the
reader asks for the byte range of one chunk inside a shard and would silently get the shard's first
bytes instead. This handler implements single-range `Range: bytes=a-b` requests, which is all zarr asks
for.

Usage: `python tests/serve.py <port> <directory>`.
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()
        size = os.path.getsize(path)
        a, _, b = rng[len("bytes="):].partition("-")
        try:
            start = int(a) if a else max(0, size - int(b))
            end = (int(b) if b else size - 1) if a else size - 1
        except ValueError:
            self.send_error(400, "bad range")
            return None
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return _Slice(f, end - start + 1)

    def log_message(self, *a):  # quiet
        pass


class _Slice:
    """A file object that yields at most `n` bytes, so `copyfile` stops at the end of the range."""

    def __init__(self, f, n):
        self.f, self.n = f, n

    def read(self, k=-1):
        if self.n <= 0:
            return b""
        k = self.n if k is None or k < 0 else min(k, self.n)
        b = self.f.read(k)
        self.n -= len(b)
        return b

    def close(self):
        self.f.close()


if __name__ == "__main__":
    port, root = int(sys.argv[1]), sys.argv[2]
    os.chdir(root)
    HTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
