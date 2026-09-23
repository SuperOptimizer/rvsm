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
    def __init__(self, cfg, out, *, lookahead_n=8, wait_s=WAIT_S, **kw):
        super().__init__(cfg, **kw)
        self.out = str(out)
        self.L = int(lookahead_n)
        self.wait_s = float(wait_s)

    def _publish(self, w, W, pos, region_s):
        _write_json(os.path.join(cursor_dir(self.out), f"w{int(w)}.json"),
                    {"pos": int(pos), "stride": int(W), "worker": int(w),
                     "region_s": float(region_s), "t": time.time()})

    def _region_lo(self, rec):
        k = int(rec["k"])
        lo2 = np.array(rec["lo"], np.int64) << max(k - 2, 0)
        return tuple(int(v) // int(self.cfg.region) * int(self.cfg.region) for v in lo2)

    def __iter__(self):
        import torch
        if self.pyr is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        w, W = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed + 1000 * w)
        mine = [int(i) for i in self.order[w::W]] or [int(i) for i in self.order]
        pos, pend, revisit, seen_no_verso = 0, [], [], {}
        t_last, region_s = time.time(), 0.0
        self._publish(w, W, 0, 0.0)     # this walk starts at 0: overwrite whatever an earlier one left
        while True:
            while len(pend) < max(self.L, 1) and pos < len(mine):
                pend.append(mine[pos])
                pos += 1
            if not pend:                       # the walk is exhausted: start it again
                pos = 0
                continue
            pick = next((j for j, i in enumerate(pend) if self._visitable(self.visits[i])), None)
            if pick is None:
                if stop_requested(self.out):
                    return
                jlog(self.out, "train", {"kind": "wait", "worker": int(w),
                                         "train_wait_s": self.wait_s, "pending": len(pend)},
                     echo=False)
                time.sleep(self.wait_s)
                self.cat = type(self.cat)(self.root, self.round)   # drop the cached MISSes
                continue
            i = pend.pop(pick)
            rec = self.visits[i]
            lo = self._region_lo(rec)
            had_verso = self.cat.done("verso", lo)
            if not had_verso and i not in seen_no_verso:
                seen_no_verso[i] = True
                revisit.append(i)
            left, fails = self.windows, 0
            while left > 0 and fails < 8 * max(self.windows, 1):
                got = self._draw(rng, rec)
                if got is None:
                    fails += 1
                    continue
                left, fails = left - 1, 0
                yield got
            now = time.time()
            region_s = 0.5 * region_s + 0.5 * (now - t_last) if region_s else now - t_last
            t_last = now
            self._publish(w, W, pos - len(pend), region_s)
            # a region whose verso landed after its visit earns exactly one revisit
            for j in list(revisit):
                if self.cat.done("verso", self._region_lo(self.visits[j])):
                    revisit.remove(j)
                    pend.insert(0, j)

