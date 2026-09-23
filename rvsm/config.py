"""The ONE place the run's contract lives: every tunable, the whole fixed v2 recipe, and the channel layout.

`Config` is a single flat dataclass. The first block is what a run may sensibly change (the CT, the
output directory, the model size, the GPUs); everything after it is the *fixed recipe* -- the v2 union of
the design's sections 26.6 and 29.9 -- which is a default from the first commit rather than a research
flag. A field is still a field so a test can shrink it (`small_cfg`) and so `config.json` records exactly
what ran, but nothing in the pipeline chooses between recipes at runtime.

`Layout` derives the input/output channel contract from the Config. It is the only module that knows the
stem channel ORDER, so the sampler, the prep, the model stem, warm-start and the exporter all agree by
construction:

    in  [CT, ctx_1..ctx_9, CASCADE, radius, meta(5), scale, rz, ry, rx]                   -> cin  21
    out [recto, verso, midline, thickness, logvar, aff8_{z,y,x}, aff16_*, aff32_*]        -> cout 14

Resume compares `fingerprint()`: everything but the fields a restart is allowed to change.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields


@dataclass(frozen=True)
class Layout:
    """The channel contract of one Config: stem order with named indices, head order with names."""

    nctx: int                 # number of coarse context cubes in the stem (rungs k+1 .. k+nctx)
    n_meta: int               # number of scan-metadata planes (scanmeta.META_RANGE)
    channels: tuple           # probability head names, in order
    aff_offsets: tuple        # affinity offsets in voxels; each contributes 3 heads (z, y, x)

    # ---- stem
    @property
    def i_ct(self):
        """Index of the CT channel (always 0)."""
        return 0

    @property
    def i_ctx(self):
        """Index of the first coarse context cube."""
        return 1

    @property
    def i_cas(self):
        """Index of the CASCADE channel (the coarser rung's prediction, upsampled)."""
        return 1 + self.nctx

    @property
    def i_planes(self):
        """Index of the first conditioning plane (the radius plane, then the metadata planes)."""
        return self.i_cas + 1

    @property
    def n_planes(self):
        """Number of conditioning planes: the radius plane plus the scan-metadata planes."""
        return 1 + self.n_meta

    @property
    def i_radius(self):
        """Index of the normalised-radius plane (`axis.radius`)."""
        return self.i_planes

    @property
    def i_meta(self):
        """Index of the first scan-metadata plane (`scanmeta.scan_planes`)."""
        return self.i_planes + 1

    @property
    def i_scale(self):
        """Index of the constant scale plane, (rung - 2) / 9."""
        return self.i_planes + self.n_planes

    @property
    def i_rad(self):
        """Index of the first of the three radial unit-vector channels (rz, ry, rx); always last."""
        return self.i_scale + 1

    @property
    def cin(self):
        """Total stem channel count."""
        return self.i_rad + 3

    # ---- head
    @property
    def nprob(self):
        """Number of probability heads (recto, verso)."""
        return len(self.channels)

    @property
    def cout_t(self):
        """Number of target-field heads: the probabilities plus midline and thickness."""
        return self.nprob + 2

    @property
    def i_mid(self):
        """Index of the signed-midline-distance head."""
        return self.nprob

    @property
    def i_thick(self):
        """Index of the thickness head."""
        return self.nprob + 1

    @property
    def i_log(self):
        """Index of the heteroscedastic log-variance head."""
        return self.cout_t

    @property
    def i_aff(self):
        """Index of the first affinity head."""
        return self.i_log + 1

    @property
    def n_aff(self):
        """Number of affinity heads: three (z, y, x) per offset."""
        return 3 * len(self.aff_offsets)

    @property
    def cout(self):
        """Total head count."""
        return self.i_aff + self.n_aff

    def head_names(self):
        """Every head in the canonical order the multi-head pass stacks them in."""
        out = [str(c) for c in self.channels] + ["midline", "thickness", "logvar"]
        for off in self.aff_offsets:
            out += [f"aff{int(off)}_{a}" for a in "zyx"]
        return out

    def stem_names(self):
        """Every stem channel in order (for logs and warm-start reports)."""
        out = ["CT"] + [f"ctx_{i + 1}" for i in range(self.nctx)] + ["cascade", "radius"]
        out += [f"meta_{i}" for i in range(self.n_meta)]
        return out + ["scale", "rz", "ry", "rx"]

    def to_json(self):
        return {"cin": self.cin, "cout": self.cout, "nprob": self.nprob, "cout_t": self.cout_t,
                "i_cas": self.i_cas, "i_planes": self.i_planes, "n_planes": self.n_planes,
                "i_scale": self.i_scale, "i_rad": self.i_rad, "i_log": self.i_log, "i_aff": self.i_aff,
                "stem": self.stem_names(), "heads": self.head_names()}


# The keys of one sampled training item. `sample.rung_item` builds exactly these and the collate walks
# them, so a key added here without a producer fails loudly instead of silently vanishing.
RUNG_ITEM_KEYS = ("ct", "tgt", "w", "lo", "cyx", "sym", "rung", "norm", "cm", "cx", "lo1", "cyx1",
                  "rmax", "meta")

# Fields a resume is allowed to differ in: a longer run, a different eval cadence, different hardware.
FINGERPRINT_EXCLUDE = ("steps", "eval_every", "workers", "gpus", "rounds", "ct_seed", "ckpt_act", "compile",
                       "vram_train_gb", "vram_produce_gb", "pin_memory", "gpu_prefetch")


@dataclass
class Config:
    # ------------------------------------------------------------------ tunables
    ct: str = ""                       # CT zarr URL or local pyramid directory (integer level names)
    umbilicus: str = "auto"            # umbilicus json / volpkg umbilicus.txt path, or "auto" to derive
    out: str = "out"                   # the run directory; everything else is derived from it
    size: str = "30m6"                 # student preset (1m, 5m, 15m, 30m6, 60m)
    patch: int = 256                   # training patch edge, in rung-k voxels
    batch: int = 2                     # patches per forward per GPU
    accum: int = 1                     # forwards accumulated per optimiser step (batch 1 x accum 2 is
                                       # batch 2's step at half the activation memory)
    ckpt_act: int = 1                  # activation-checkpointing level (0 = off, higher = more recompute)
    gpus: tuple = (0,)                 # CUDA device ordinals available to the run
    mode: str = "auto"                 # auto | resident (one big card) | timeshare (alternating phases)
    rounds: int = 3                    # self-distillation rounds, round 0 = the teacher bootstrap
    steps: int = 20000                 # optimiser steps per round
    workers: int = 4                   # sampler worker processes
    tifxyz: str = ""                   # optional directory of human tifxyz meshes, for eval only
    teacher_ckpts: dict = field(default_factory=dict)  # {"recto": path, "m7": path} local teacher weights
    cache_gb: float = 64.0             # CT shard cache budget on disk/RAM
    gpu_prefetch: bool = True          # the trainer copies batch i+1 to the device on a side stream
    pin_memory: bool = False           # the trainer's loader pins its batches. Off: on Thunder's A100 the
                                       # trainer hung inside CUDA calls in every run that pinned (none that
                                       # did not), and pinned H2D was slower there anyway (2.2 vs 2.7 GB/s)
    vram_train_gb: float = 46.0        # the resident budget table (GB on an 80 GB card): the trainer ...
    vram_produce_gb: float = 30.0      # ... and the producer; each becomes a per-process memory fraction
    ct_seed: str = ""                  # a local, possibly partial mirror of a URL `ct`: shards it has are
                                       # hard-linked into the cache instead of fetched
# (ckpt_act and compile are in FINGERPRINT_EXCLUDE: they change the speed and the memory, never the math)

    # ------------------------------------------------------------------ fixed recipe: the ladder
    ctx: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9)   # context cube offsets, in rungs above the sample's own
    rungs: tuple = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  # the rungs a run trains on
    rung_boost: dict = field(default_factory=lambda: {2: 2})  # extra draw weight per rung
    region: int = 1024                 # region edge in rung-2 voxels: the unit of production and walk
    windows_per_region: int = 128      # training windows drawn from a region per visit
    visits_max: int = 64               # how often the walk may return to one region
    air_keep: float = 0.1              # probability of keeping a drawn window that is pure air
    occ_min_fine: float = 0.05         # minimum CT occupancy fraction for a rung-2 region
    occ_min_coarse: float = 0.01       # ... for rungs >= 5
    heldout: int = 8                   # rung-2 regions held out of the walk at every rung

    # ------------------------------------------------------------------ fixed recipe: optimisation
    lr: float = 3e-4                   # peak learning rate
    warmup: int = 200                  # linear warmup steps from cold
    rewarm: int = 800                  # warmup steps after a warm start / round change
    new_param_lr_mult: float = 3.0     # lr multiplier for tensors a warm start could not copy
    sched: str = "wsd"                 # warmup-stable-decay
    cooldown: float = 0.1              # fraction of the run spent in the WSD cooldown
    ema: str = "auto"                  # auto = horizon chosen from the step count
    ema_k: int = 50                    # the auto-EMA constant (horizon = steps / k)
    aug: str = "full2"                 # augmentation preset, ranges centred on the scan metadata

    # ------------------------------------------------------------------ fixed recipe: cascade + planes
    cascade: str = "mix"               # off | mask | self | mix: source of the cascade channel
    self_p_lo: float = 0.1             # cascade self-prediction probability at step 0
    self_p_hi: float = 0.7             # ... annealed to this by the end of the run
    cascade_drop: float = 0.1          # probability of blanking the cascade channel
    cascade_noise: bool = True         # jitter the cascade channel
    planes: str = "radius+meta"        # the conditioning planes (radius plane + 5 scan-metadata planes)
    channels: tuple = ("recto", "verso")  # the probability heads, in order
    aff_offsets: tuple = (8, 16, 32)   # affinity offsets in voxels; 3 heads (z, y, x) each

    # ------------------------------------------------------------------ fixed recipe: losses
    loss_excl: float = 0.1             # recto/verso mutual exclusion
    loss_selfcons: float = 0.1         # cross-rung self-consistency
    loss_skel: float = 0.05            # soft-skeleton (clDice) term
    skel_iters: int = 4                # soft-skeleton erosion iterations
    loss_affinity: float = 0.1         # multi-offset affinity term
    loss_sdist: float = 1.0            # signed midline distance + thickness
    loss_eikonal: float = 0.1          # |grad d| = 1 on the distance head
    pair_band: float = 1.5             # pair-construction band half-width, in voxels
    pair_tau: float = 0.5              # pair-construction probability threshold
    loss_ect: float = 0.05             # Euler-characteristic-transform term (rung 2 only)
    ect_n: int = 1                     # ECT directions per step
    ect_rung: int = 2                  # the rung the ECT term applies at
    ect_block: int = 64                # ECT block edge

    # ------------------------------------------------------------------ fixed recipe: rounds + inference
    verso_source: str = "flip"         # verso stores come from the student run with the radial sign flipped
    verso_after_steps: int = 10000     # round-0 verso starts here if the gate has not already fired
    verso_gate_dice: float = 0.6       # ... or earlier, once the held-out recto reaches this dice
    eval_every: int = 2000             # held-out evaluation cadence, in steps
    calibrate: bool = True             # refit the temperatures after every eval
    infer_window: int = 256            # sliding-window edge at inference
    infer_halo: int = 32               # window overlap discarded on each side
    cascade_depth: int = 3             # coarse-to-fine passes per inference region
    tta: int = 1                       # test-time augmentations (1 = none)
    min_regions_before_train: int = 8  # cold start: regions the producer finishes before training begins
    lookahead_extra: int = 4           # the "+4" of L = ceil(T_produce / T_train) * K_active + 4
    train_min: float = 20.0            # timeshare: minutes of training per phase
    produce_max_min: float = 10.0      # timeshare: maximum minutes of production per phase
    reserve_gb: float = 50.0           # production pauses below this much free disk
    round_steps: int = 20000           # a round ends here if the plateau fit has not ended it
    compile: bool = True               # torch.compile the student (and the teachers, when teacher_bf16)
    teacher_bf16: bool = True          # teachers under bf16 autocast (+ compile): their fp32 forward was
                                       # the whole cost of a teacher region

    # ------------------------------------------------------------------ derived
    def layout(self):
        """The channel contract implied by this config."""
        from rvsm.scanmeta import META_RANGE
        n_meta = len(META_RANGE) if "meta" in self.planes else 0
        return Layout(nctx=len(self.ctx), n_meta=n_meta, channels=tuple(self.channels),
                      aff_offsets=tuple(self.aff_offsets))

    def to_json(self):
        """A JSON-safe dict of the config plus its layout: what `<out>/config.json` holds."""
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        return {"config": d, "layout": self.layout().to_json(), "fingerprint": self.fingerprint()}

    def fingerprint(self):
        """A hash of every field a resume must match. Fields in FINGERPRINT_EXCLUDE are left out, so a
        longer run, a different eval cadence or different hardware may resume an existing checkpoint."""
        d = {k: v for k, v in asdict(self).items() if k not in FINGERPRINT_EXCLUDE}
        blob = json.dumps(d, sort_keys=True, default=lambda v: list(v) if isinstance(v, tuple) else str(v))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


_TYPES = {f.name: f.type for f in fields(Config)}


def _coerce(name, v):
    """TOML / CLI value -> the field's type (tuple fields accept a list, int fields a float)."""
    cur = getattr(Config(), name)
    if isinstance(cur, tuple):
        return tuple(v) if isinstance(v, (list, tuple)) else (v,)
    if isinstance(cur, bool):
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(v)
    if isinstance(cur, float):
        return float(v)
    if isinstance(cur, dict):
        d = dict(v)
        # TOML keys are always strings; a field whose defaults are keyed by RUNG (`rung_boost`) must
        # come back keyed by int, or the boost silently applies to nothing (and the fingerprint, which
        # json-encodes the keys as strings either way, would not show it).
        if cur and all(isinstance(k, int) for k in cur):
            d = {int(k): q for k, q in d.items()}
        return d
    return type(cur)(v)


def load(path=None, overrides=None):
    """The run's Config: a TOML file (stdlib `tomllib`, flat or one `[rvsm]` table) then `overrides`.

    Both sources may leave anything out; an unknown key is an error, because a typo in a recipe field
    would silently train something else. `path=None` means the defaults."""
    data = {}
    if path:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        data = dict(raw.get("rvsm", raw))
        for k, v in list(data.items()):          # allow [phase] / [disk] style sub-tables to flatten
            if isinstance(v, dict) and k not in _TYPES:
                data.pop(k)
                data.update({f"{k}_{a}" if f"{k}_{a}" in _TYPES else a: b for a, b in v.items()})
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    bad = [k for k in data if k not in _TYPES]
    if bad:
        raise ValueError(f"unknown config key(s) {bad}; known keys: {sorted(_TYPES)}")
    return Config(**{k: _coerce(k, v) for k, v in data.items()})
