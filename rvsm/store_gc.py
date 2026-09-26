"""List, and optionally delete, SUPERSEDED generations of a run's region stores.

    python -m rvsm.store_gc --out DIR [--round R] [--delete] [--gen0] [--min-age-min 30] [--log F]
    rvsm store-gc --out DIR ...                  (the same)

A regeneration (the verso's, a teacher reteach, the fields rebuilt from either) writes a NEW directory
beside the old one (`stores.gen_path`: `region_<z>_<y>_<x>[.g<N>].zarr`) and `run.commit_sources`
moves the region's bundle (`<out>/stores/round_<r>/bundle/region_<z>_<y>_<x>.json`, `stores.bundle_state`)
to it; nothing ever deletes the old one. Per region and channel the COMMITTED generation is the one
`stores.current_path` resolves -- recto / rw / band -> bundle "recto", verso / verso_r* -> "verso",
midline* / thickness* -> "gen" -- and:

    g <  committed   SUPERSEDED: no reader resolves it again (commits only move forward)
    g == committed   the store every reader uses: never touched
    g >  committed   IN PROGRESS (a reteach or verso regeneration not committed yet): never touched

Any other channel has one generation and is left alone. A superseded store is still skipped when its
`zarr.json` lacks done=true, when any file in it (or the directory) was modified within `--min-age-min`,
when the region's bundle was committed within `--min-age-min` (a visit or a unit that resolved the old
path just before the commit may still be reading it: a sampler visit is ~8 min on paris4), when the
committed store itself is not finished, and when the path is the committed path of any channel.

GENERATION 0 (`region_<z>_<y>_<x>.zarr`, no suffix) is only a candidate with `--gen0`: before commit
`regions.Catalog.list_done` learnt to parse `.g<N>` names, it was the region's only existence marker
(the recto regeneration's work list, the verso count, the round rollover), so a producer or trainer
running OLDER code must not see it go. Use `--gen0` only once every process runs this commit or later.

What does NOT pin an old generation: the evaluation grid's identity (`sample.grid_sources`) digests
`Catalog.path`, i.e. the committed stores, and its item files are self-contained tensors; the
`Catalog` resolves `path` from the bundle on EVERY call (its TTL caches only a MISS of that resolved
path, and its opened-array cache is keyed by path), so the moment a bundle moves no catalog hands out
the old generation again. Only a read already in flight holds it -- hence the bundle-age guard.

Deletion (`--delete`): each store is first `os.replace`d to `<name>.gc-<ts>` (a name no reader or
writer resolves: a concurrent reader gets a clean ENOENT, never a half-deleted store), then removed
with `shutil.rmtree`. A `.gc-*` left by an interrupted run is removed too. A JSON log of everything
removed is written to `--log` (default `<out>/logs/store_gc_<ts>.json`). The default is a dry run
that deletes and writes nothing and prints, per channel, the candidates' count and bytes.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

from rvsm import stores

GC_TAG = ".gc-"


def committed_gen(channel, bundle):
    """The committed generation of `channel` under a `stores.bundle_state` dict, or None for a channel
    that has one generation (never collected). Mirrors `stores.current_path`."""
    c = str(channel)
    if c in stores.TEACHER_BUNDLED:
        return int(bundle["recto"])
    if c == "verso" or c.startswith("verso_r"):
        return int(bundle["verso"])
    if stores.is_bundled(c):
        return int(bundle["gen"])
    return None


def _tree(path):
    """(apparent bytes, allocated bytes, newest mtime) of a store directory, its own mtime included."""
    size = alloc = 0
    newest = os.stat(path).st_mtime
    for d, _dirs, files in os.walk(path):
        try:
            newest = max(newest, os.stat(d).st_mtime)
        except OSError:
            pass
        for f in files:
            try:
                st = os.stat(os.path.join(d, f))
            except OSError:
                continue
            size += st.st_size
            alloc += st.st_blocks * 512
            newest = max(newest, st.st_mtime)
    return size, alloc, newest


def _rounds(out, round_=None):
    root = os.path.join(str(out), "stores")
    if round_ is not None:
        return [int(round_)]
    got = []
    for n in os.listdir(root) if os.path.isdir(root) else ():
        if n.startswith("round_") and n[len("round_"):].isdigit():
            got.append(int(n[len("round_"):]))
    return sorted(got)


def scan(out, round_=None, gen0=False, min_age_s=1800.0, now=None):
    """Every superseded store of the run: a list of dicts {round, channel, region, gen, committed,
    path, bytes, alloc}, plus the skipped ones (`skipped`: same fields and `why`) and the leftover
    `.gc-*` directories (`leftover`). Reads only."""
    now = time.time() if now is None else float(now)
    cands, skipped, leftover = [], [], []
    for r in _rounds(out, round_):
        rd = os.path.join(str(out), "stores", f"round_{r}")
        chans = sorted(n for n in os.listdir(rd)
                       if n != "bundle" and not n.endswith(".zarr") and os.path.isdir(os.path.join(rd, n)))
        bundles = {}
        for ch in chans:
            d = os.path.join(rd, ch)
            names = os.listdir(d)
            for n in names:
                if GC_TAG in n:
                    leftover.append(os.path.join(d, n))
            if committed_gen(ch, {"recto": 0, "verso": 0, "gen": 0}) is None:
                continue                                # a single-generation channel
            by_region = {}
            for n in names:
                got = stores.parse_region_name(n)
                if got is not None:
                    by_region.setdefault(got[0], []).append(got[1])
            for lo, gs in sorted(by_region.items()):
                if lo not in bundles:
                    bf = stores._bundle_file(out, lo, r)
                    try:
                        bm = os.stat(bf).st_mtime
                    except OSError:
                        bm = None
                    bundles[lo] = (stores.bundle_state(out, lo, r), bm)
                b, bm = bundles[lo]
                cg = committed_gen(ch, b)
                base = stores.store_path(out, ch, lo, r)
                for g in sorted(gs):
                    if g >= cg:
                        continue                        # committed, or in progress
                    p = stores.gen_path(base, g)
                    rec = {"round": r, "channel": ch, "region": list(lo), "gen": g, "committed": cg,
                           "path": p}
                    why = None
                    if bm is None:
                        why = "no bundle file"          # cannot be: committed > 0 needs one
                    elif now - bm < min_age_s:
                        why = "bundle committed recently"
                    elif not stores.is_done(stores.gen_path(base, cg)):
                        why = "committed store not finished"
                    elif not stores.is_done(p):
                        why = "not done"
                    elif os.path.islink(p) or not os.path.isdir(p):
                        why = "not a directory"
                    elif os.path.abspath(p) == os.path.abspath(stores.current_path(out, ch, lo, r)):
                        why = "committed path"          # paranoia: `current_path` agrees with `cg`
                    try:
                        size, alloc, newest = _tree(p)
                    except OSError:
                        continue                        # vanished under the scan
                    rec.update(bytes=size, alloc=alloc)
                    if why is None and now - newest < min_age_s:
                        why = "modified recently"
                    if why is None and g == 0 and not gen0:
                        why = "gen0"                    # would go with --gen0: every guard passed
                    if why is None:
                        cands.append(rec)
                    else:
                        skipped.append({**rec, "why": why})
    return {"candidates": cands, "skipped": skipped, "leftover": leftover}


def summary(res):
    """Per (round, channel): count and bytes of the candidates, and of the skipped ones by reason."""
    per = {}
    for c in res["candidates"]:
        k = f"round_{c['round']}/{c['channel']}"
        e = per.setdefault(k, {"n": 0, "bytes": 0, "alloc": 0, "by_gen": {}, "skipped": {}})
        e["n"] += 1
        e["bytes"] += c["bytes"]
        e["alloc"] += c["alloc"]
        e["by_gen"][str(c["gen"])] = e["by_gen"].get(str(c["gen"]), 0) + 1
    for s in res["skipped"]:
        k = f"round_{s['round']}/{s['channel']}"
        e = per.setdefault(k, {"n": 0, "bytes": 0, "alloc": 0, "by_gen": {}, "skipped": {}})
        q = e["skipped"].setdefault(s["why"], {"n": 0, "bytes": 0})
        q["n"] += 1
        q["bytes"] += s.get("bytes", 0)
    return per


def delete(res, log_path, log=print):
    """Remove every candidate: rename to `<path>.gc-<ts>`, then rmtree; and the leftovers. Writes the
    JSON log (what was removed, with its bytes) and returns it."""
    ts = int(time.time())
    removed, failed = [], []
    for c in res["candidates"]:
        p = c["path"]
        q = f"{p}{GC_TAG}{ts}"
        try:
            os.replace(p, q)
        except OSError as e:
            failed.append({**c, "error": repr(e)})
            continue
        shutil.rmtree(q, ignore_errors=True)
        removed.append(c)
    for q in res["leftover"]:
        shutil.rmtree(q, ignore_errors=True)
    rec = {"t": time.time(), "removed": removed, "failed": failed, "leftover_removed": res["leftover"],
           "bytes": sum(c["bytes"] for c in removed), "alloc": sum(c["alloc"] for c in removed)}
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(log_path + ".tmp", log_path)
    log(f"store-gc: removed {len(removed)} stores ({rec['alloc'] / (1 << 30):.2f} GiB), "
        f"{len(failed)} failed, {len(res['leftover'])} leftovers; log {log_path}")
    return rec


def _gib(b):
    return f"{b / (1 << 30):8.2f} GiB"


def report(res, log=print):
    per = summary(res)
    tot_n = tot_b = 0
    for k in sorted(per):
        e = per[k]
        tot_n += e["n"]
        tot_b += e["alloc"]
        sk = ", ".join(f"{w}: {v['n']} ({v['bytes'] / (1 << 30):.2f} GiB)" for w, v in sorted(e["skipped"].items()))
        log(f"{k:32s} {e['n']:6d} superseded {_gib(e['alloc'])}  gens {e['by_gen']}"
            + (f"  | skipped {sk}" if sk else ""))
    log(f"{'TOTAL':32s} {tot_n:6d} superseded {_gib(tot_b)}"
        + (f"  (+ {len(res['leftover'])} leftover .gc-* dirs)" if res["leftover"] else ""))
    return per


USAGE = __doc__.split("\n\n")[1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out, round_, do, gen0, age, logp, as_json = None, None, False, False, 30.0, None, False
    it = iter(argv)
    for a in it:
        if a in ("-h", "--help"):
            print(__doc__)
            return 0
        if a == "--out":
            out = next(it)
        elif a == "--round":
            round_ = int(next(it))
        elif a == "--delete":
            do = True
        elif a == "--dry-run":
            do = False
        elif a == "--gen0":
            gen0 = True
        elif a == "--min-age-min":
            age = float(next(it))
        elif a == "--log":
            logp = next(it)
        elif a == "--json":
            as_json = True
        else:
            print(USAGE)
            return 2
    if not out:
        print(USAGE)
        return 2
    res = scan(out, round_=round_, gen0=gen0, min_age_s=age * 60.0)
    per = report(res)
    if as_json:
        print(json.dumps(per))
    if not do:
        print("store-gc: dry run, nothing deleted (--delete to remove the superseded stores above)")
        return 0
    logp = logp or os.path.join(str(out), "logs", f"store_gc_{int(time.time())}.json")
    rec = delete(res, logp)
    return 1 if rec["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
