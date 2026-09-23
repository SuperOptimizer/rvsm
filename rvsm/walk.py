"""The trainer's sampler for `rvsm run`: `sample.Patches`' walk with the lookahead rule on top.

Its own module because the DataLoader's forkserver workers unpickle the dataset BY NAME, so the class must
be importable at module level -- and `rvsm/run.py` must not import torch at its top (the spawned producer
sets CUDA_VISIBLE_DEVICES first). `run.walk_patches()` returns `WalkPatches`; see its docstring there.
"""
from __future__ import annotations

import os
import time

import numpy as np

from rvsm import sample
from rvsm.run import WAIT_S, _write_json, cursor_dir, jlog, stop_requested


class WalkPatches(sample.Patches):
    """`start` is a saved walk (`run.walk_snapshot`: `{"stride": W, "workers": {w: {"pos", "done",
    "pass"}}}`): a worker whose stride matches resumes its own share of the walk at `pos`, skipping
    the positions in `done` it had already visited past it, instead of replaying from 0 -- a restart
    must never repeat training data. A snapshot for a different worker count is ignored."""

    def __init__(self, cfg, out, *, lookahead_n=8, wait_s=WAIT_S, start=None, **kw):
        super().__init__(cfg, **kw)
        self.out = str(out)
        self.L = int(lookahead_n)
        self.wait_s = float(wait_s)
        self.start = start or None

    def _publish(self, w, W, pos, region_s, done=(), npass=0):
        _write_json(os.path.join(cursor_dir(self.out), f"w{int(w)}.json"),
                    {"pos": int(pos), "stride": int(W), "worker": int(w),
                     "region_s": float(region_s), "t": time.time(),
                     "done": sorted(int(q) for q in done), "pass": int(npass)})

    def _region_lo(self, rec):
        k = int(rec["k"])
        lo2 = np.array(rec["lo"], np.int64) << max(k - 2, 0)
        return tuple(int(v) // int(self.cfg.region) * int(self.cfg.region) for v in lo2)

    def _resume_at(self, w, W, n):
        """(pos, done positions, pass) this worker starts from: the saved walk, or 0."""
        s = self.start or {}
        if int(s.get("stride", -1)) != int(W):
            return 0, set(), 0
        e = (s.get("workers") or {}).get(str(int(w))) or (s.get("workers") or {}).get(int(w))
        if not e:
            return 0, set(), 0
        pos = min(max(int(e.get("pos", 0)), 0), n)
        return pos, {int(q) for q in e.get("done", ()) if int(q) >= pos}, int(e.get("pass", 0))

    def __iter__(self):
        import torch
        if self.pyr is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        w, W = (info.id, info.num_workers) if info else (0, 1)
        mine = [int(i) for i in self.order[w::W]] or [int(i) for i in self.order]
        pos, visited, npass = self._resume_at(w, W, len(mine))
        # a resumed walk draws other windows than the first pass over the same visits would have
        rng = np.random.default_rng(self.seed + 1000 * w + 7919 * npass + pos)
        pend, revisit, seen_no_verso = [], [], {}     # pend / revisit hold (position, visit)
        t_last, region_s = time.time(), 0.0

        def floor():        # a revisit (position -1) is not a place in the walk
            return min([q for q, _ in pend if q >= 0] + [pos])

        # this walk starts where the saved one stopped (0 on a fresh run): overwrite whatever an
        # earlier process left, so the producer's window follows THIS walk
        self._publish(w, W, pos, 0.0, visited, npass)
        while True:
            while len(pend) < max(self.L, 1) and pos < len(mine):
                if pos not in visited and not self._dead(self.visits[mine[pos]]):
                    pend.append((pos, mine[pos]))    # held-out home: never a target
                pos += 1
            if not pend:                       # the walk is exhausted: start it again
                pos, npass, visited = 0, npass + 1, set()
                continue
            pick = next((j for j, (_, i) in enumerate(pend) if self._visitable(self.visits[i])), None)
            if pick is None:
                if stop_requested(self.out):
                    return
                jlog(self.out, "train", {"kind": "wait", "worker": int(w),
                                         "train_wait_s": self.wait_s, "pending": len(pend)},
                     echo=False)
                time.sleep(self.wait_s)
                self.cat = type(self.cat)(self.root, self.round)   # drop the cached MISSes
                continue
            q, i = pend.pop(pick)
            rec = self.visits[i]
            lo = self._region_lo(rec)
            had_verso = self.cat.done("verso", lo)
            if not had_verso and i not in seen_no_verso:
                seen_no_verso[i] = True
                revisit.append((q, i))
            # the visit counts as made from the moment it STARTS: a snapshot taken mid-visit (or
            # with its windows still in the loader's prefetch queue) skips the rest of it on resume
            # rather than replaying the part already trained on
            visited.add(q)
            now = time.time()
            region_s = 0.5 * region_s + 0.5 * (now - t_last) if region_s else now - t_last
            t_last = now
            f = floor()
            visited = {v for v in visited if v >= f}
            self._publish(w, W, f, region_s, visited, npass)
            left, fails, air = self.windows, 0, self.air_budget()
            while left > 0 and fails < 8 * max(self.windows, 1):
                got = self._draw(rng, rec, air_ok=air > 0)
                if got is None:
                    fails += 1
                    continue
                air -= int(self._last_air)
                left, fails = left - 1, 0
                yield got
            # a region whose verso landed after its visit earns exactly one revisit
            for j in list(revisit):
                if self.cat.done("verso", self._region_lo(self.visits[j[1]])):
                    revisit.remove(j)
                    pend.insert(0, (-1, j[1]))
