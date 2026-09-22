# rvsm

**rvsm** turns a raw CT zarr and an umbilicus into a self-distilled recto/verso surface model, on one
machine. It starts from nothing but the volume: it runs the upstream teachers itself over the regions it
needs, trains a unified multi-rung student on their output, infers the verso by running the same student
with the radial sign flipped, and then keeps going — the student becomes its own teacher, round after
round. Production is on demand, one 1024³ region at a time, and every region's state is derived from what
is on disk, so a run resumes by looking at its own output directory. There is no cluster, no planner
process, no published mask or store lookup, and every v2 recipe improvement is a default from the first
commit rather than a research flag.

## CLI

```
rvsm run      cfg.toml | --ct URL|PATH --umbilicus PATH|auto --out DIR --gpus 0[,1] --mode auto|resident|timeshare
              [--rounds N] [--steps N] [--size 30m6] [--init ckpt.pt]
rvsm produce  --out DIR (--teacher recto,m7 | --student ckpt [--sign -1]) --region Z Y X
rvsm train    cfg.toml
rvsm eval     --out DIR [--ckpt P] [--round R] [--tifxyz DIR] [--json]
rvsm export   --out DIR --ckpt P --box Z Y X DZ DY DX --dest DIR
rvsm calibrate --out DIR --ckpt P
rvsm pretrain cfg.toml [--steps N]
rvsm ladder   cfg.toml --sizes 15m,30m6,60m
rvsm status | stop | ledger --rebuild | umbilicus --ct URL --out umbilicus.json
```

Only the skeleton is in place so far (`rvsm.config`, `rvsm.ladder`, `rvsm.axis`, `rvsm.scanmeta`); the
subcommands land with the modules that drive them.

## Install

```sh
uv venv
uv pip install -e '.[dev]'          # torch comes from the cu130 index configured in pyproject.toml
```

Extras: `mesh` (marching cubes and tifxyz meshes for export/eval), `trt` (TensorRT teacher engines,
with a torch bf16 fallback), `dev` (pytest).

### libvolcomp

Every CT volume rvsm reads is volcomp-encoded, so `volcomp_zarr` needs the shared library. `rvsm.ladder`
raises with the reason if it cannot load it — it never silently drops the levels it cannot decode. Build
it and point `VOLCOMP_LIB` at the result:

```sh
git clone https://github.com/SuperOptimizer/volume-compressor && cd volume-compressor
cmake --preset release && cmake --build --preset release --target volcomp_shim
export VOLCOMP_LIB=$PWD/build/release/libvolcomp.so
```

(`volcomp_zarr` also finds a `libvolcomp.so` sitting beside the installed package, in which case
`VOLCOMP_LIB` is unnecessary. The test suite points it at a local build when the variable is unset.)

## Tests

```sh
.venv/bin/python -m pytest -q
```

Everything is CPU-only and synthetic: a sharded zarr-v3 CT pyramid with a tilted slab in it, served by a
range-capable HTTP server so the ladder is exercised over the network as well as locally.

## Layout

| module | what it owns |
|---|---|
| `config.py` | the ONE `Config` dataclass and the `Layout` that derives the channel contract (cin 21, cout 14) |
| `ladder.py` | the rung ladder: one CT pyramid, read at any rung, coarse rungs pooled from the top |
| `axis.py` | the umbilicus (parse, derive, write) and the radial/radius/scale channels built from it |
| `scanmeta.py` | the upstream `metadata.json`, flattened, as five conditioning planes and augmentation ranges |
