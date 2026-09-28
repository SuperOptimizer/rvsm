"""List, and optionally delete, SUPERSEDED generations of a run's region stores.

    python -m rvsm.store_gc --out DIR [--round R] [--delete] [--gen0] [--min-age-min 30] [--log F]
                            [--superseded-by-round R]
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

`--superseded-by-round R` lists (and with `--delete` removes) instead the ROUND-(R-1) stores of every
region whose round-R stores cover them (`superseded_by_round`): once a region's round-R recto is
finished the trainer reads the whole region from round R (`regions.RoundCatalog`), so its round-(R-1)
stores of every channel round R has (and round 0's rw / band beside the recto) are dead. Held-out
regions are never touched (their round-0 stores are every round's reference), nor a region whose
round-R recto finished within `--min-age-min`. The producer does the same per region as it goes
(`run.SUPERSEDE_GRACE_S` after the region's round-R fields are in, a `supersede_gc` line); this is for
what a restart or an older producer left behind.

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


# ------------------------------------------------------------------ superseded by a newer ROUND

SUPERSEDE_ANCHOR = "recto"   # the channel whose round-r store moves a region's readers to round r
                             # (`regions.RoundCatalog.ANCHOR`)


def pinned_regions(out):
    """The held-out regions (`<out>/eval/heldout.json`): their ROUND-0 stores are the fixed reference
    every gate and evaluation of every round is scored against, never collected."""
    try:
        with open(os.path.join(str(out), "eval", "heldout.json")) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return set()
    return {tuple(int(v) for v in h["lo"]) for h in (d or {}).get("regions", [])}


def _region_dirs(out, channel, lo, round_):
    """Every directory of a region's store in one channel and round: generation 0 and each `.g<N>`."""
    base = stores.store_path(out, channel, lo, round_)
    return [(g, stores.gen_path(base, g)) for g in [0] + stores.gens(base)
            if os.path.isdir(stores.gen_path(base, g))]


def superseded_by_round(out, lo, round_):
    """The round-(`round_` - 1) store directories of region `lo` that its round-`round_` stores cover:
    every generation of each lower-round channel whose round-`round_` store (`stores.current_path`) is
    finished -- and round 0's teacher companions (rw, band: `stores.TEACHER_BUNDLED`), which only mean
    anything beside round 0's recto, once the round-`round_` recto is. A list of dicts {round, channel,
    region, gen, path}; nothing when the round-`round_` anchor (recto) is not finished, since readers
    (`regions.RoundCatalog`) keep reading the lower round until it is. Reads only."""
    r, old = int(round_), int(round_) - 1
    lo = tuple(int(v) for v in lo)
    if old < 0 or not stores.is_done(stores.current_path(out, SUPERSEDE_ANCHOR, lo, r)):
        return []
    rd = os.path.join(str(out), "stores", f"round_{old}")
    if not os.path.isdir(rd):
        return []
    got = []
    for ch in sorted(os.listdir(rd)):
        if ch == "bundle" or ch.endswith(".zarr") or not os.path.isdir(os.path.join(rd, ch)):
            continue
        cover = SUPERSEDE_ANCHOR if ch in stores.TEACHER_BUNDLED else ch
        if not stores.is_done(stores.current_path(out, cover, lo, r)):
            continue
        for g, p in _region_dirs(out, ch, lo, old):
            got.append({"round": old, "channel": ch, "region": list(lo), "gen": g, "path": p})
    return got


def remove(paths_or_recs):
    """Rename each store to `<path>.gc-<ts>` (a name no reader resolves), then rmtree it. Returns
    (removed records with their `bytes` / `alloc`, failed records)."""
    ts = int(time.time())
    removed, failed = [], []
    for c in paths_or_recs:
        c = c if isinstance(c, dict) else {"path": c}
        p = c["path"]
        try:
            size, alloc, _ = _tree(p)
        except OSError:
            continue
        q = f"{p}{GC_TAG}{ts}"
        try:
            os.replace(p, q)
        except OSError as e:
            failed.append({**c, "error": repr(e)})
            continue
        shutil.rmtree(q, ignore_errors=True)
        removed.append({**c, "bytes": size, "alloc": alloc})
    return removed, failed


def supersede(out, lo, round_, pinned=None):
    """Delete region `lo`'s round-(`round_` - 1) stores that its round-`round_` stores cover
    (`superseded_by_round`), unless it is a held-out region (`pinned`, default `pinned_regions`).
    Returns (removed records, bytes allocated that were freed)."""
    pinned = pinned_regions(out) if pinned is None else {tuple(int(v) for v in p) for p in pinned}
    if tuple(int(v) for v in lo) in pinned:
        return [], 0
    removed, _ = remove(superseded_by_round(out, lo, round_))
    return removed, sum(c["alloc"] for c in removed)


def scan_superseded(out, round_, min_age_s=1800.0, now=None):
    """`scan`'s result shape for the round-(`round_` - 1) stores superseded by round `round_`: every
    region with a finished round-`round_` anchor store, its covered lower-round stores
    (`superseded_by_round`) as candidates -- or skipped, `why` "held out" (`pinned_regions`) or
    "superseded recently" (the round-`round_` anchor finished less than `min_age_s` ago: a reader that
    resolved the region before may still be reading the old round)."""
    now = time.time() if now is None else float(now)
    cands, skipped, leftover = [], [], []
    r = int(round_)
    pinned = pinned_regions(out)
    old_rd = os.path.join(str(out), "stores", f"round_{r - 1}")
    if r >= 1 and os.path.isdir(old_rd):
        for ch in os.listdir(old_rd):
            d = os.path.join(old_rd, ch)
            if os.path.isdir(d) and not ch.endswith(".zarr"):
                leftover += [os.path.join(d, n) for n in os.listdir(d) if GC_TAG in n]
    d = os.path.join(str(out), "stores", f"round_{r}", SUPERSEDE_ANCHOR)
    los = sorted({got[0] for got in (stores.parse_region_name(n) for n in
                                     (os.listdir(d) if r >= 1 and os.path.isdir(d) else ()))
                  if got is not None})
    for lo in los:
        recs = superseded_by_round(out, lo, r)
        if not recs:
            continue
        anchor = stores.current_path(out, SUPERSEDE_ANCHOR, lo, r)
        try:
            t_new = os.stat(os.path.join(anchor, "zarr.json")).st_mtime
        except OSError:
            continue
        why = "held out" if lo in pinned else ("superseded recently" if now - t_new < min_age_s else None)
        for c in recs:
            try:
                size, alloc, _ = _tree(c["path"])
            except OSError:
                continue
            c.update(bytes=size, alloc=alloc, committed=None)
            (cands if why is None else skipped).append(c if why is None else {**c, "why": why})
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
    by_round = None
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
        elif a == "--superseded-by-round":
            by_round = int(next(it))
        else:
            print(USAGE)
            return 2
    if not out:
        print(USAGE)
        return 2
    if by_round is not None:
        res = scan_superseded(out, by_round, min_age_s=age * 60.0)
    else:
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
