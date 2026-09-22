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

Status: every commit of the plan is on main, and the whole suite runs on the CPU. Nothing has run on a GPU or on a real scroll yet: see the progress table in docs/plan.md and the known gaps in docs/review_checklist.md before trusting any of it in production.
subcommands land with the modules that drive them.

## Documentation

Everything about *why* rvsm is built this way is under [`docs/`](docs/); each document cites the usrm2
design sections and the measured runs behind it, so a reviewer can reconstruct any decision.

| document | what it is for |
|---|---|
| [`docs/plan.md`](docs/plan.md) | the implementation plan, approved 2026-09-22, verbatim, plus the user decisions it records and a table mapping its 7 commits to the actual history on `main` |
| [`docs/rationale.md`](docs/rationale.md) | why rvsm exists: the usrm2 pipeline's failure modes with their incidents, the measured numbers a reviewer needs (u1/u3 surface metrics vs the label ceiling, training speeds, store-format facts, the TensorRT findings), and every design decision with its evidence and the alternative rejected |
| [`docs/recipe.md`](docs/recipe.md) | every fixed default in `rvsm/config.py`, one line of justification each, with a pointer to the research or design section that supports it, and the residual experiment list |
| [`docs/research/README.md`](docs/research/README.md) | the 18-document research corpus indexed one paragraph at a time, with the consolidated bibliography |
| [`docs/review_checklist.md`](docs/review_checklist.md) | what to verify before a production run: contracts, store rules, weighting rules, gates, VRAM budgets, resumability, the test covering each, and the known gaps |

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
| `run.py` | the driver: the supervisor, the producer loop, the lookahead window, the verso and round gates, the rounds and the GPU modes |
