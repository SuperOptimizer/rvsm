import os, sys, json, tempfile, shutil, time
work = tempfile.mkdtemp(prefix="m7w_")
os.environ.update(M7W_WORK=work, M7W_HOME=work, M7W_NO_EVICT="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import m7_wholevol as M
from concurrent.futures import ThreadPoolExecutor

shape = (1300, 600, 530)
rng = np.random.default_rng(0)
ct = np.zeros(shape, np.uint8)
zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
inside = ((yy - 300) ** 2 + (xx - 260) ** 2) < 240 ** 2
ct[:] = np.where(inside, rng.integers(1, 255, shape, dtype=np.uint8), 0)
ct[:, :, :40] = 0
ct[500:560] = 0           # an all-air band
v = dict(sample="S", name="20990101000000-8.640um-x", shape=list(shape), pitch=8.64, stem="S_test", url="http://invalid")
json.dump([dict(sample="S", name=v["name"], shape=list(shape))], open(os.path.join(work, "volumes.json"), "w"))
codec = M.Codec(); pool = ThreadPoolExecutor(8)
ctd = os.path.join(work, "ct", "S_test")
for zi in range(2):
    for yi in range(1):
        for xi in range(1):
            blk = np.ascontiguousarray(ct[zi*1024:(zi+1)*1024, :, :])
            # lossless CT so the test compares exactly
            orig = M.Q
            M.write_shard(os.path.join(ctd, "c", str(zi), str(yi), str(xi)), blk, 1024,
                          type("C", (), {"encode": lambda self, c: codec.encode(c, 0.0)})(), pool)

class Mock:
    def __init__(self, dev):
        self.dev = dev; self.stream = torch.cuda.Stream(dev); self.plan = "mock"
        z = torch.arange(M.W, device=dev, dtype=torch.float32)
        self.pos = (0.6 + 0.4 * torch.cos(z / 17.0))[:, None, None] * (0.7 + 0.3 * torch.sin(z / 23.0))[None, :, None] \
            * (0.8 + 0.2 * torch.cos(z / 29.0))[None, None, :]
    def __call__(self, u8):
        return torch.sigmoid(u8.float() / 40 - 3) * self.pos

dev = torch.device("cuda:0")
m7 = Mock(dev)
vv = M.load_vols(os.path.join(work, "volumes.json"))[0]
assert vv["stem"] == "S_20990101000000", vv["stem"]
vv["stem"] = "S_test"
t = time.time()
for (r0, r1) in [(1, 2), (0, 1)]:           # units out of order, one shard row each
    M.run_unit(vv, r0, r1, m7, codec, pool, 144, {})
print("units", time.time() - t)
root = M.store_dir(vv)
out = M.read_block(os.path.join(root, M.lvl_path(8.64, 0)), shape, 1024, (0, 0, 0), shape, codec, pool)
cts = M.CTSource(vv, codec, pool)
ref, cbox = M.ref_blend(vv, (0, 0, 0), shape, m7, cts, 144)
d = np.abs(out.astype(int) - ref.astype(int))
print("vs direct blend: max", d.max(), "mean", d[ct > 0].mean(), "frac>1", (d > 1).mean(), "nonzero out", (out > 0).mean())
lossless = (out == ref).mean()
print("exact frac", lossless)
# the quantised q8 store vs the lossless ref: q8 error, so compare decoded L0 against ref within tolerance
for k in range(1, 4):
    sh = M.lvl_shape(shape, k)
    a = M.read_block(os.path.join(root, M.lvl_path(8.64, k)), sh, M.lvl_shard(k), (0, 0, 0), sh, codec, pool)
    print("level", k, a.shape, "mean", a.mean())
shutil.rmtree(work)
