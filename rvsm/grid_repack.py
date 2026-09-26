"""Rewrite a spilled validation grid's item files from pickle protocol 2 to protocol 4, in place.

    python -m rvsm.grid_repack DIR [DIR ...] [--jobs N] [--dry-run] [--force] [--wait S]
    rvsm grid-repack DIR ...                     (the same)

`sample._save_item` wrote every grid item with torch.save's default pickle protocol 2 until
2026-09-26, which pickles each Blosc buffer as a latin-1 str: ~1.2 s of GIL-held work per 240 MB item
on load, so the evaluation's prefetch threads ran one at a time. Protocol 4 loads the same item in
~0.25 s. New and rebuilt items are protocol 4 already; this converts the ones a run keeps reusing.

Per file (`item_*.pt`): already protocol 4 (the zip's data.pkl header) -> skipped. Otherwise it is
loaded, saved to `.repack.<name>.tmp` in the same directory (a name `sample._drop_orphans` never
deletes), fsynced, loaded back and compared with the original record (every Blosc buffer byte for
byte, every raw value equal), then `os.replace`d over the original and the directory fsynced. The
decoded arrays are therefore identical, and the file NAME is unchanged -- names and grid.json are
digests of what an item is BUILT from (`sample._item_name`, `grid_global`, the region sources), never
of the file's bytes, so the manifest still validates; the tool checks that every file grid.json lists
still exists afterwards, and that `val_grid`'s reuse test (file present, record unchanged) holds.

WHEN TO RUN. Between evaluations. The trainer opens item files only inside an evaluation, and since
this commit it marks that with `<grid dir>/.evaluating` (pid, host, step); the tool waits while the
marker's process is alive (at most `--wait` seconds per check, then gives up on the remaining files),
and treats a marker whose pid is dead as stale. A trainer from BEFORE this commit writes no marker:
run the tool right after an `eval.jsonl` line, it takes minutes and an evaluation is hours apart.
Even a replace during an evaluation is not corrupting (a reader holds the old inode), only slower.
COST (measured, 240 MB protocol-2 items): ~2.2 s per item per job, and ~1.5 GB peak RSS per job
(the latin-1 str, the record and the read-back); `--jobs 4` is ~6 GB. The protocol-4 file is ~27 %
smaller (protocol 2 wrote each byte >= 0x80 as two UTF-8 bytes). Exit status 1 when a file stayed
`busy`, else 0; a JSON summary line (files, per-status counts, bytes, seconds) is printed last.
It must not run alongside `val_grid` rebuilding the same directory (a refresh writes new names and
drops orphans; a file that vanishes under the tool is reported as `gone`, never recreated).
"""
from __future__ import annotations

import json
import os
import sys
import time

PKL_P4 = b"\x80\x04"
TMP_PREFIX = ".repack."


def protocol_of(path):
    """The pickle protocol byte of a torch zip file's data.pkl (2, 4, ...), or None when unreadable."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
            with z.open(name) as f:
                head = f.read(2)
    except (OSError, zipfile.BadZipFile, StopIteration):
        return None
    return head[1] if len(head) == 2 and head[0] == 0x80 else None


def _same(a, b):
    import numpy as np
    import torch
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if x[0] != y[0]:
            return False
        if x[0] == "blosc":
            if bytes(x[1]) != bytes(y[1]) or tuple(x[2]) != tuple(y[2]) or x[3:] != y[3:]:
                return False
        else:
            u, v = x[1], y[1]
            if torch.is_tensor(u) or torch.is_tensor(v):
                if not (torch.is_tensor(u) and torch.is_tensor(v) and u.dtype == v.dtype
                        and torch.equal(u, v)):
                    return False
            elif isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
                if not (isinstance(u, np.ndarray) and isinstance(v, np.ndarray)
                        and u.dtype == v.dtype and np.array_equal(u, v)):
                    return False
            elif u != v:
                return False
    return True


def _marker_live(d):
    """The `.evaluating` marker's record when its process is alive on this host, else None."""
    from rvsm.sample import EVAL_MARKER
    import socket
    m = os.path.join(d, EVAL_MARKER)
    try:
        with open(m) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    pid = int(rec.get("pid", 0) or 0)
    if rec.get("host") not in (None, socket.gethostname()):
        return rec                      # another host's trainer: cannot check, so assume it is live
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None                     # stale: the trainer that wrote it is gone
    except PermissionError:
        pass
    return rec


def wait_idle(d, wait_s, poll=5.0):
    """True once no live evaluation marks `d` (immediately, usually); False after `wait_s` seconds."""
    t0 = time.time()
    while _marker_live(d) is not None:
        if time.time() - t0 >= wait_s:
            return False
        time.sleep(poll)
    return True


def repack_one(path, dry_run=False, wait_s=3600.0, force=False):
    """(status, bytes before, bytes after, seconds) for one item file. status: p4 (already), would
    (dry run), repacked, gone (vanished under us), busy (an evaluation outlasted `wait_s`). `force`:
    never wait on an `.evaluating` marker."""
    import torch
    t0 = time.time()
    try:
        n0 = os.path.getsize(path)
    except OSError:
        return "gone", 0, 0, 0.0
    if protocol_of(path) == 4:
        return "p4", n0, n0, time.time() - t0
    if dry_run:
        return "would", n0, 0, time.time() - t0
    d, name = os.path.split(path)
    if not force and not wait_idle(d, wait_s):
        return "busy", n0, 0, time.time() - t0
    try:
        rec = torch.load(path, weights_only=False)
    except FileNotFoundError:
        return "gone", 0, 0, time.time() - t0
    tmp = os.path.join(d, f"{TMP_PREFIX}{name}.tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(rec, f, pickle_protocol=4)
            f.flush()
            os.fsync(f.fileno())
        if protocol_of(tmp) != 4 or not _same(rec, torch.load(tmp, weights_only=False)):
            raise RuntimeError(f"{path}: the protocol-4 copy does not read back identical")
        if not os.path.exists(path):
            os.remove(tmp)
            return "gone", 0, 0, time.time() - t0
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return "repacked", n0, os.path.getsize(path), time.time() - t0


def _job(args):
    path, dry_run, wait_s, force = args
    return (path,) + repack_one(path, dry_run=dry_run, wait_s=wait_s, force=force)


def check_manifest(d):
    """(n listed, missing files) of `d/grid.json`: what `sample.val_grid`'s reuse test needs on disk
    (a record's file present; its global / source fields are JSON the tool never touches)."""
    man = os.path.join(d, "grid.json")
    if not os.path.exists(man):
        return 0, []
    with open(man) as f:
        m = json.load(f)
    files = [r["file"] for r in (m.get("records") or [])] or list(m.get("items") or [])
    return len(files), [n for n in files if not os.path.exists(os.path.join(d, n))]


def repack(dirs, jobs=1, dry_run=False, wait_s=3600.0, force=False, log=print):
    """Repack every `item_*.pt` of `dirs`; returns the summary dict (also printed)."""
    import multiprocessing as mp
    t0 = time.time()
    todo, before = [], {}
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"grid_repack: {d} is not a directory")
        live = _marker_live(d)
        if live is not None and not force and not dry_run:
            log(f"grid_repack: an evaluation is reading {d} ({live}); waiting up to {wait_s:.0f} s per file")
        before[d] = check_manifest(d)
        todo += [os.path.join(d, n) for n in sorted(os.listdir(d))
                 if n.startswith("item_") and n.endswith(".pt")]
    args = [(p, bool(dry_run), float(wait_s), bool(force)) for p in todo]
    res = []
    if int(jobs) > 1 and len(args) > 1:
        with mp.get_context("spawn").Pool(int(jobs)) as pool:
            for r in pool.imap_unordered(_job, args):
                res.append(r)
                if r[1] in ("repacked", "busy"):
                    log(f"  {r[1]:8s} {os.path.basename(r[0])} {r[2] / 2**20:.0f} MB {r[4]:.1f} s")
    else:
        for a in args:
            r = _job(a)
            res.append(r)
            if r[1] in ("repacked", "busy"):
                log(f"  {r[1]:8s} {os.path.basename(r[0])} {r[2] / 2**20:.0f} MB {r[4]:.1f} s")
    count = {}
    for r in res:
        count[r[1]] = count.get(r[1], 0) + 1
    after = {d: check_manifest(d) for d in dirs}
    for d in dirs:
        n, missing = after[d]
        if missing and len(missing) > len(before[d][1]):
            raise SystemExit(f"grid_repack: {d}/grid.json lists {len(missing)} missing files after the "
                             f"repack (before: {len(before[d][1])}): {missing[:5]}")
    out = {"dirs": list(dirs), "files": len(res), **count,
           "bytes_rewritten": sum(r[2] for r in res if r[1] == "repacked"),
           "bytes_after": sum(r[3] for r in res if r[1] == "repacked"),
           "bytes_to_rewrite": sum(r[2] for r in res if r[1] == "would"),
           "manifest_files": {d: after[d][0] for d in dirs},
           "manifest_missing": {d: len(after[d][1]) for d in dirs},
           "dry_run": bool(dry_run), "jobs": int(jobs), "s": round(time.time() - t0, 1)}
    log(json.dumps(out))
    return out


USAGE = __doc__.split("\n\n")[1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dirs, jobs, dry, force, wait_s = [], 1, False, False, 3600.0
    it = iter(argv)
    for a in it:
        if a in ("-h", "--help"):
            print(__doc__)
            return 0
        if a == "--jobs":
            jobs = int(next(it))
        elif a == "--dry-run":
            dry = True
        elif a == "--force":
            force = True
        elif a == "--wait":
            wait_s = float(next(it))
        elif a.startswith("--"):
            print(USAGE)
            return 2
        else:
            dirs.append(a)
    if not dirs:
        print(USAGE)
        return 2
    out = repack(dirs, jobs=jobs, dry_run=dry, wait_s=wait_s, force=force)
    return 1 if out.get("busy") else 0


if __name__ == "__main__":
    sys.exit(main())
