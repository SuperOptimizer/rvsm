"""Pull every published PHerc Paris 4 surface (segment) mesh to a local directory.

Two sources, both public, read over ONE keep-alive aiohttp session with at most 16 connections:

  volpkg   https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/<id>/
           283 legacy VC segment dirs (obj / ppm / vcps / tif). They carry NO tifxyz; only each dir's
           listing and its meta.json (plus author.txt / area_cm2.txt) are pulled -- the obj/ppm are
           tens of GB and the tifxyz below are the same surfaces.
  s3       https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/<id>/mesh/
           the tifxyz triplets (x.tif, y.tif, z.tif float32, meta.json) of the ~82 segments that were
           re-published, each in up to three frames: `<id>-on-20230205180739-7.91um` (legacy volume
           frame), `<id>-on-20260411134726-2.4um` (the fine 2.4 um volume) and `-45.532um`.

Layout: <out>/<segid>/{meta.json, author.txt, area_cm2.txt, listing.json, <frame>.tifxyz/{x,y,z}.tif,meta.json}
and <out>/manifest.json (every file, its size, its source). A file already present at its listed size
is skipped; a download goes to `.part` and is renamed only after its size checks out.

    python -m rvsm.tools.refine.pull_paths --out /home/forrest/refine/paths [--jobs 16] [--no-mesh]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import aiohttp

VOLPKG = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/"
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
S3_PREFIX = "PHercParis4/segments/"
SMALL = ("meta.json", "author.txt", "area_cm2.txt")
UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
ROW = re.compile(r'<a href="([^"]+)"[^>]*>.*?</a></td><td class="size">([^<]*)</td><td class="date">([^<]*)<')


def parse_listing(html):
    """nginx fancyindex html -> [(name, approx bytes or None, date string)]."""
    out = []
    for name, size, date in ROW.findall(html):
        if name.startswith("..") or name.startswith("?"):
            continue
        b = None
        m = re.match(r"([0-9.]+)\s*([KMGT]?i?B)", size.strip())
        if m:
            b = int(float(m.group(1)) * UNITS.get(m.group(2), 1))
        out.append((name, b, date.strip()))
    return out


class Puller:
    def __init__(self, out, jobs=16, retries=4):
        self.out, self.jobs, self.retries = out, int(jobs), int(retries)
        self.sem = asyncio.Semaphore(self.jobs)
        self.done_bytes = self.got_bytes = self.skipped = self.failed = 0
        self.errors = []

    async def text(self, url):
        for a in range(self.retries):
            try:
                async with self.sem, self.sess.get(url) as r:
                    if r.status == 404:
                        return None
                    r.raise_for_status()
                    return await r.text()
            except Exception as e:  # noqa: BLE001 - retried, then reported
                err = e
                await asyncio.sleep(1 + 2 * a)
        self.errors.append((url, repr(err)))
        return None

    async def file(self, url, path, size=None):
        """Download url -> path unless it is already there at `size` (None: any non-empty file)."""
        if os.path.exists(path) and (size is None and os.path.getsize(path) > 0
                                     or size is not None and os.path.getsize(path) == size):
            self.skipped += 1
            self.done_bytes += os.path.getsize(path)
            return True
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part = path + ".part"
        err = None
        for a in range(self.retries):
            try:
                async with self.sem, self.sess.get(url) as r:
                    r.raise_for_status()
                    n = 0
                    with open(part, "wb") as f:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            f.write(chunk)
                            n += len(chunk)
                if size is not None and n != size:
                    raise IOError(f"size {n} != listed {size}")
                os.replace(part, path)
                self.got_bytes += n
                self.done_bytes += n
                return True
            except Exception as e:  # noqa: BLE001
                err = e
                await asyncio.sleep(1 + 2 * a)
        self.failed += 1
        self.errors.append((url, repr(err)))
        try:
            os.remove(part)
        except OSError:
            pass
        return False

    async def s3_list(self, prefix, delimiter=None):
        """(keys [(key, size, last_modified)], common prefixes) under prefix (ListObjectsV2, paginated)."""
        keys, pre, token = [], [], None
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        while True:
            q = f"?list-type=2&prefix={quote(prefix)}" + (f"&delimiter={quote(delimiter)}" if delimiter else "") \
                + (f"&continuation-token={quote(token, safe='')}" if token else "")
            root = ET.fromstring(await self.text(S3 + q))
            for c in root.findall("s:Contents", ns):
                keys.append((c.find("s:Key", ns).text, int(c.find("s:Size", ns).text),
                             c.find("s:LastModified", ns).text))
            pre += [c.find("s:Prefix", ns).text for c in root.findall("s:CommonPrefixes", ns)]
            if root.find("s:IsTruncated", ns).text != "true":
                return keys, pre
            token = root.find("s:NextContinuationToken", ns).text

    async def s3_mesh_keys(self):
        """Every tifxyz key: the segment prefixes first (one delimited listing), then each segment's
        `mesh/` prefix in parallel -- a flat listing of `segments/` walks every ink-label zarr chunk."""
        _k, segs = await self.s3_list(S3_PREFIX, "/")
        got = await asyncio.gather(*(self.s3_list(s + "mesh/") for s in segs))
        return [k for keys, _p in got for k in keys], len(segs)

    async def run(self, mesh=True):
        conn = aiohttp.TCPConnector(limit=self.jobs, limit_per_host=self.jobs, keepalive_timeout=60)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=120, sock_connect=30)
        manifest = {"volpkg": {}, "s3": {}}
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as self.sess:
            # ---- volpkg paths: listing + small files of every dir
            top = parse_listing(await self.text(VOLPKG))
            dirs = [(n.rstrip("/"), d) for n, _b, d in top if n.endswith("/")]
            print(f"[pull] volpkg: {len(dirs)} path dirs", flush=True)

            async def one_dir(seg, date):
                html = await self.text(VOLPKG + seg + "/")
                files = parse_listing(html or "")
                rec = {"date": date, "files": {n: b for n, b, _ in files}}
                d = os.path.join(self.out, seg)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "listing.json"), "w") as f:
                    json.dump(rec, f, indent=1)
                for n in SMALL:
                    if n in rec["files"]:
                        await self.file(VOLPKG + seg + "/" + n, os.path.join(d, n))
                # a volpkg dir that does carry a tifxyz triplet (none did on 2026-09-26)
                trip = [n for n in ("x.tif", "y.tif", "z.tif") if n in rec["files"]]
                if mesh and trip:
                    for n in trip:
                        await self.file(VOLPKG + seg + "/" + n, os.path.join(d, "volpkg.tifxyz", n))
                manifest["volpkg"][seg] = rec
            await asyncio.gather(*(one_dir(s, dt) for s, dt in dirs))

            # ---- S3 segments: every key under */mesh/*.tifxyz/
            keys, nseg = await self.s3_mesh_keys()
            mk = [(k, s, t) for k, s, t in keys if "/mesh/" in k and ".tifxyz/" in k]
            print(f"[pull] s3: {nseg} segments under {S3_PREFIX}, {len(mk)} tifxyz files, "
                  f"{sum(s for _k, s, _t in mk) / 2**30:.2f} GiB", flush=True)
            for k, s, t in mk:
                seg = k[len(S3_PREFIX):].split("/")[0]
                manifest["s3"].setdefault(seg, {})[k] = {"bytes": s, "modified": t}
            if mesh:
                t0 = time.time()

                async def one_key(k, s):
                    rel = k[len(S3_PREFIX):]
                    seg, _mesh, frame, name = rel.split("/")[:4]
                    return await self.file(S3 + k, os.path.join(self.out, seg, frame, name), s)
                big = sorted(mk, key=lambda q: -q[1])       # biggest first: the pool stays busy
                await asyncio.gather(*(one_key(k, s) for k, s, _t in big))
                dt = time.time() - t0
                print(f"[pull] s3 meshes: {self.got_bytes / 2**30:.2f} GiB fetched in {dt:.0f} s "
                      f"({self.got_bytes / 2**20 / max(dt, 1e-6):.1f} MiB/s), {self.skipped} already "
                      f"complete, {self.failed} failed", flush=True)
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest


def frames_of(out, seg):
    """{frame name: meta.json dict} of the tifxyz dirs downloaded for one segment."""
    d = os.path.join(out, seg)
    got = {}
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else ():
        p = os.path.join(d, n, "meta.json")
        if n.endswith(".tifxyz") and os.path.exists(p):
            with open(p) as f:
                got[n] = json.load(f)
    return got


def report(out, manifest):
    """Counts, sizes, frames declared, and the largest / oldest / newest tables."""
    vol = {}
    for seg in manifest["volpkg"]:
        p = os.path.join(out, seg, "meta.json")
        if os.path.exists(p):
            try:
                vol[json.load(open(p)).get("volume")] = vol.get(json.load(open(p)).get("volume"), 0) + 1
            except ValueError:
                vol["<bad json>"] = vol.get("<bad json>", 0) + 1
    print(f"\nvolpkg paths: {len(manifest['volpkg'])} dirs; meta.json 'volume' ids: {vol}")
    legacy = sum(sum(v for v in r["files"].values() if v) for r in manifest["volpkg"].values())
    print(f"  (their obj/ppm/vcps would be ~{legacy / 2**40:.2f} TiB: not pulled)")
    segs = manifest["s3"]
    tot = sum(v["bytes"] for s in segs.values() for v in s.values())
    print(f"S3 re-published segments with tifxyz: {len(segs)}; {tot / 2**30:.2f} GiB of tifxyz files")
    fr = {}
    for seg in segs:
        for k in segs[seg]:
            fr.setdefault(k.split("/")[4].split("-on-")[-1], set()).add(seg)
    for k, v in sorted(fr.items()):
        print(f"  frame {k}: {len(v)} segments")
    rows = []
    for seg, files in segs.items():
        b = sum(v["bytes"] for v in files.values())
        fine = frames_of(out, seg)
        m24 = next((m for n, m in fine.items() if "2.4um" in n), {})
        bb = m24.get("bbox")
        zr = (bb[0][2], bb[1][2]) if bb else None
        rows.append((seg, b, min(v["modified"] for v in files.values()), zr, m24.get("area_vx2")))
    fmt = lambda r: (f"  {r[0]:<22} {r[1] / 2**20:8.1f} MiB  s3 {r[2][:10]}  "
                     f"fine z {'%.0f-%.0f' % r[3] if r[3] else '-':<13} "
                     f"area {r[4] / 1e6 if r[4] else 0:8.1f} Mvox^2")
    print("largest:")
    for r in sorted(rows, key=lambda r: -r[1])[:10]:
        print(fmt(r))
    print("oldest (by segment id = creation time):")
    for r in sorted(rows)[:5]:
        print(fmt(r))
    print("newest:")
    for r in sorted(rows)[-5:]:
        print(fmt(r))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/home/forrest/refine/paths")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-mesh", action="store_true", help="listings + meta.json only")
    a = ap.parse_args(argv)
    assert a.jobs <= 16, "be gentle: at most 16 connections"
    os.makedirs(a.out, exist_ok=True)
    p = Puller(a.out, a.jobs)
    t0 = time.time()
    man = asyncio.run(p.run(mesh=not a.no_mesh))
    print(f"[pull] done in {time.time() - t0:.0f} s; {len(p.errors)} errors", flush=True)
    for u, e in p.errors[:20]:
        print("  ERR", u, e)
    report(a.out, man)
    return 1 if p.failed else 0


if __name__ == "__main__":
    sys.exit(main())
