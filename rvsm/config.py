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
RUNG_ITEM_KEYS = ("ct", "tgt", "w", "tch", "lo", "cyx", "sym", "rung", "norm", "cm", "cx", "lo1", "cyx1",
                  "rmax", "meta")

# Fields a resume is allowed to differ in: a longer run, a different eval cadence, different hardware.
FINGERPRINT_EXCLUDE = ("infer_margin", "steps", "eval_every", "workers", "gpus", "rounds", "ct_seed", "ckpt_act", "compile",
                       "vram_train_gb", "vram_produce_gb", "pin_memory", "gpu_prefetch", "verso_min_dice",
                       "self_p_mid_step", "self_p_end", "self_p_end_step",
                       # the cascade source: paris4 switched mix -> self (drop 0.1 -> 0.3) at step 12000 on
                       # a user decision; a resume may change it, and the switch is recorded in the run log
                       "cascade", "cascade_drop", "self_p_lo", "self_p_hi",
                       # loss weights a running experiment may retune on resume (pass-3 P3-01)
                       "loss_prob_dice", "loss_pair",
                       # scheduling of the verso passes and of the round gate (pass-3 item 11, P3-03)
                       "verso_regen_gain", "round_min_steps_after_verso", "verso_min_regions",
                       "ram_trainer_gb", "ram_host_exit",
                       # the producer's GPU fields batch and whether it regenerates old verso stores
                       "fields_batch", "verso_regen",
                       # the checkpoint-only cadence: WHEN the resume state is written, never what is
                       # trained (a checkpoint boundary runs no evaluation, calibration or gate)
                       "ckpt_every",
                       # the round-0 teacher set (its KEYS pick the teachers): paris4 dropped the 2.4 um
                       # `recto` teacher and went m7-only from ~step 56000 on a user decision
                       # (2026-09-25). A deliberate mid-run change of the recto target definition: the
                       # resume logs a `teacher_switch` sched line and the producer regenerates every
                       # recto another set made (`run.recto_needs_regen`)
                       "teacher_ckpts",
                       # the continuity terms, retunable on a resume (a deliberate mid-run retuning,
                       # 2026-09-25, paris4: values unchanged for now); the trainer reads them from the
                       # live config and the resume logs a `loss_switch` sched line naming any that moved
                       "loss_skel", "loss_affinity", "loss_ect", "loss_selfcons",
                       # the skeleton PRECISION term (off by default) and its two knobs: switchable on a
                       # resume like the continuity terms above (a `loss_switch` line when the weight moves)
                       "loss_skel_prec", "skel_prec_iters", "skel_prec_gate",
                       # the GroupNorm+SiLU outputs in bf16 under autocast (model.NormAct): it changes
                       # the numerics (the stored activations and the upsample round to bf16), so it is
                       # recorded in config.json and in every checkpoint's cfg, but a resume may flip it
                       # as a deliberate mid-run switch, like the cascade change; the resume logs a
                       # `precision_switch` sched line (`run.log_switches`)
                       "gn_bf16",
                       # the overlap-crop consistency term (docs/recipe.md): off by default, and switched
                       # on / retuned on a resume as a deliberate mid-run change (`loss_switch` line)
                       "overlap_p", "overlap_sub", "loss_overlap",
                       # per-rung teacher ROUTING with gap-fill (`route_spec`, docs/recipe.md §7): like the
                       # m7 switch, a change of the recto target SOURCE that the producer's regeneration
                       # handles (a routed store records `route`, `run.recto_needs_regen` compares it); the
                       # resume logs a `teacher_switch` sched line. The gap-fill loss knobs go with it
                       "teacher_route", "loss_band", "band_dilate", "band_eps",
                       # the THINNED-BAND rung-2 target (docs/recipe.md §6): computed on the device at step
                       # time from the stored recto, switched on at a resume (`loss_switch` line)
                       "thin_band", "thin_band_width", "thin_band_soft")

# The loss weights a resume may retune (`run.log_switches` logs a `loss_switch` line when one moved).
LOSS_SWITCH_FIELDS = ("loss_prob_dice", "loss_pair", "loss_skel", "loss_affinity", "loss_ect",
                      "loss_selfcons", "loss_skel_prec", "loss_overlap", "loss_band", "band_dilate", "band_eps",
                      "thin_band", "thin_band_width", "thin_band_soft")

# The per-rung teacher routing (`Config.teacher_route`). ONE fine teacher may own the recto targets of
# rung 2 (`ROUTE_FINE_RUNGS`: the only rung the sampler and the producer know how to route); every other
# rung keeps the BASE teacher's targets, and the base teacher's probability rides along at rung 2 as the
# gap-fill `band`. `ROUTE_RULE` versions the coverage rule c_A (`infer.route_coverage_u8`): a new rule
# is a new store identity, so the producer regenerates.
ROUTE_FINE_RUNGS = (2,)
ROUTE_RULE = "cA-v1"
ROUTE_BASE_DEFAULT = "m7"


@dataclass(frozen=True)
class RouteSpec:
    """An ACTIVE teacher route: `fine` makes the recto probability at `fine_rungs` (rung 2), `base` makes
    it at every other rung and the gap-fill band at the fine rungs. `sig` is the identity a routed store
    records (`route` attr) and the producer compares against."""
    fine: str
    base: str
    fine_rungs: tuple
    sig: str


def route_spec(cfg):
    """The config's teacher route as a `RouteSpec`, or None when routing is off.

    `teacher_route` maps rung -> teacher name (TOML keys are strings: `{ "2" = "recto", "3" = "m7",
    "4" = "m7" }`). OFF: an empty map, or one whose rung 2 names the same teacher as the other rungs
    (that is the one-teacher mode of `teacher_ckpts`, nothing to route). ON: rung 2 names the FINE
    teacher and every other rung given names one BASE teacher (`m7` when no other rung is given). A
    fine teacher at another rung, or two base teachers, is an error: only rung 2 can be routed."""
    r = getattr(cfg, "teacher_route", None) or {}
    if not r:
        return None
    try:
        m = {int(k): str(v) for k, v in dict(r).items()}
    except (TypeError, ValueError):
        raise ValueError(f"teacher_route: the keys must be rungs, got {r!r}") from None
    fine = m.get(2)
    others = sorted({v for k, v in m.items() if k != 2})
    if len(others) > 1:
        raise ValueError(f"teacher_route: every rung but 2 must name the one base teacher, got {m}")
    base = others[0] if others else ROUTE_BASE_DEFAULT
    if fine is None:
        raise ValueError(f"teacher_route: only rung 2 can be routed to a fine teacher, got {m}")
    if fine == base:
        return None
    return RouteSpec(fine=fine, base=base, fine_rungs=ROUTE_FINE_RUNGS,
                     sig=f"r2={fine};band={base};{ROUTE_RULE}")


@dataclass
class Config:
    # ------------------------------------------------------------------ tunables
    ct: str = ""                       # CT zarr URL or local pyramid directory (integer level names)
    umbilicus: str = "auto"            # umbilicus json / volpkg umbilicus.txt path, or "auto" to derive
    out: str = "out"                   # the run directory; everything else is derived from it
    size: str = "30m6"                 # student preset (1m, 5m, 15m, 30m6, 60m)
    patch: int = 256                   # training patch edge, in rung-k voxels
    batch: int = 1                     # patches per forward per GPU
    accum: int = 2                     # forwards accumulated per optimiser step (batch 1 x accum 2 is
                                       # batch 2's step at half the activation memory)
    ckpt_act: int = 0                  # activation-checkpointing level (0 = off, higher = more recompute)
    gpus: tuple = (0,)                 # CUDA device ordinals available to the run
    mode: str = "auto"                 # auto | resident (one big card) | timeshare (alternating phases)
    rounds: int = 3                    # self-distillation rounds, round 0 = the teacher bootstrap
    steps: int = 20000                 # optimiser steps in TOTAL, over every round (the global budget;
                                       # a round is promoted only with round_steps of it left)
    workers: int = 6                   # sampler worker processes
    tifxyz: str = ""                   # optional directory of human tifxyz meshes, for eval only
    teacher_ckpts: dict = field(default_factory=dict)  # {"recto": path, "m7": path} local teacher weights;
                                       # its KEYS are the teacher set ({} = recto + m7; {"m7": path} alone =
                                       # m7-only recto targets, rw = 1)
    teacher_route: dict = field(default_factory=dict)  # per-rung teacher routing with gap-fill; {} = off.
                                       # {"2": "recto", "3": "m7", "4": "m7"}: rung-2 recto targets from the
                                       # 2.4 um recto teacher where it is confident (c_A), the m7 band beside
                                       # them for the gaps, rungs >= 3 from m7 as before (`route_spec`;
                                       # both teachers must be keys of `teacher_ckpts`)
    cache_gb: float = 64.0             # CT shard cache budget on disk/RAM
    gpu_prefetch: bool = True          # the trainer copies batch i+1 to the device on a side stream
    pin_memory: bool = False           # the trainer's loader pins its batches. Off: on Thunder's A100 the
                                       # trainer hung inside CUDA calls in every run that pinned (none that
                                       # did not), and pinned H2D was slower there anyway (2.2 vs 2.7 GB/s)
    vram_train_gb: float = 50.0        # the resident budget table (GB on an 80 GB card): the trainer ...
    vram_produce_gb: float = 26.0      # ... and the producer; each becomes a per-process memory fraction
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
    cascade: str = "self"              # off | mask | self | mix: source of the cascade channel
    self_p_lo: float = 0.1             # cascade self-prediction probability at step 0
    self_p_hi: float = 0.7             # ... annealed linearly to this at `self_p_mid_step`
    self_p_mid_step: int = 20000       # (0: the old schedule, self_p_lo -> self_p_hi over the whole run)
    self_p_end: float = 0.9            # ... then linearly to this at `self_p_end_step`, held after
    self_p_end_step: int = 30000
    cascade_drop: float = 0.3          # probability of blanking the cascade channel
    cascade_noise: bool = True         # jitter the cascade channel
    planes: str = "radius+meta"        # the conditioning planes (radius plane + 5 scan-metadata planes)
    channels: tuple = ("recto", "verso")  # the probability heads, in order
    aff_offsets: tuple = (8, 16, 32)   # affinity offsets in voxels; 3 heads (z, y, x) each

    # ------------------------------------------------------------------ fixed recipe: losses
    loss_prob_dice: float = 1.0        # weight of the LEARNED probability heads' soft dice (bce stays 1)
    loss_pair: float = 1.0             # weight of the constructed pair's bce + dice
    loss_excl: float = 0.1             # recto/verso mutual exclusion
    loss_selfcons: float = 0.1         # cross-rung self-consistency
    loss_skel: float = 0.05            # soft-skeleton (clDice) term
    skel_iters: int = 4                # soft-skeleton erosion iterations
    loss_skel_prec: float = 0.0        # soft-clDice PRECISION of the prediction's skeleton vs the target
                                       # band + 1 voxel (spurs / bridges); 0 = off (`losses.skel_precision`)
    skel_prec_iters: int = 3           # its soft-thinning iterations (2 * (iters + 1) pooling ops)
    skel_prec_gate: float = 0.3        # only predicted voxels with p >= this are scored (early-training haze)
    loss_affinity: float = 0.1         # multi-offset affinity term
    loss_sdist: float = 1.0            # signed midline distance + thickness
    loss_eikonal: float = 0.1          # |grad d| = 1 on the distance head
    pair_band: float = 1.5             # pair-construction band half-width, in voxels
    pair_tau: float = 0.5              # pair-construction probability threshold
    loss_ect: float = 0.05             # Euler-characteristic-transform term (rung 2 only)
    ect_n: int = 1                     # ECT directions per step
    ect_blocks: int = 1                # interior ECT blocks per sample, drawn at random (seeded by step)
    ect_rung: int = 2                  # the rung the ECT term applies at
    ect_block: int = 64                # ECT block edge
    # the OVERLAP-CROP CONSISTENCY term (`losses.overlap_loss`, docs/recipe.md "Overlap-crop
    # consistency"): with probability `overlap_p` a training draw also carries a SECOND window of the same
    # visit, shifted by p/4..p/2 (even) along one or more axes; the trainer runs the EMA net on it (no
    # grad, bf16) and scores the student's crop against it on the shared voxels where the EMA window sees
    # MORE context than the student's. Off by default (both 0).
    overlap_p: float = 0.0             # probability that a draw carries the second window
    overlap_sub: int = 0               # EMA forward on a q^3 sub-crop of the second window (0 = all of it)
    loss_overlap: float = 0.0          # weight of the term (0 = not computed, even with overlap_p > 0)
    loss_band: float = 0.0             # routed rung 2 only (`teacher_route`): mean relu(p - band_eps)^2 over
                                       # the voxels OUTSIDE the dilated base (m7) band where the fine teacher
                                       # is not confident (`losses.band_penalty`); 0 = off
    band_dilate: int = 2               # the base band's dilation radius at the routed rung, in voxels
    band_eps: float = 0.05             # the band penalty's free margin
    thin_band: int = 0                 # 1: the rung-2 recto target is the THINNED band (`losses.thin_band`):
                                       # a sheet about the m7 band's medial surface, the band's flanks weight 0
    thin_band_width: float = 4.0       # the thinned sheet's full width, in rung-2 voxels
    thin_band_soft: float = 1.0        # its linear fade at the edge, in voxels

    # ------------------------------------------------------------------ fixed recipe: rounds + inference
    verso_source: str = "flip"         # verso stores come from the student run with the radial sign flipped
    verso_after_steps: int = 10000     # round-0 verso starts here if the gate has not already fired
    verso_min_dice: float = 0.3        # ... and even then only once the eval's RUNG-2 dice reaches this
                                       # on two consecutive evaluations
    verso_regen_gain: float = 0.15     # rung-2 dice >= verso_min_dice + this, first time: round 0's verso
                                       # stores from older checkpoints are regenerated once (new generation)
    round_min_steps_after_verso: int = 8000   # round 0 is promoted only this long after verso_on
    verso_min_regions: int = 200       # ... and with at least this many finished verso stores
    verso_regen: bool = True           # regenerate round 0's old verso stores once (verso_regen_gain)
    fields_batch: int = 0              # blocks per GPU fields batch in the producer (~0.83 GB each);
                                       # 0: 1 when the producer's VRAM budget is under 30 GB, else 3
    ram_trainer_gb: float = 40.0       # the trainer's own RSS past this: checkpoint and exit cleanly
    ram_host_exit: float = 0.90        # ... or the host's memory in use past this share of MemTotal
    verso_gate_dice: float = 0.6       # ... or earlier, once the held-out recto reaches this dice
    eval_every: int = 2000             # held-out evaluation cadence, in steps
    ckpt_every: int = 1000             # checkpoint-only cadence, in steps (0 = checkpoint at evals only):
                                       # the resume state an evaluation writes, without the evaluation
    calibrate: bool = True             # refit the temperatures after every eval
    infer_window: int = 256            # sliding-window edge at inference
    infer_halo: int = 32               # window overlap discarded on each side
    infer_margin: int = 16             # rung-2 voxels of fine CT read around a region at inference (the
                                       # region's windows are spread over the padded box, the accumulators
                                       # stay the region): no seam at a region face, and no extra windows
                                       # (`infer.spread_starts`: 5^3 per 1024^3 at 256/32 up to margin 64,
                                       # stride 200 at 16). Scaled by rung (`infer.margin_at`); 0 = none
    cascade_depth: int = 3             # coarse-to-fine passes per inference region
    tta: int = 1                       # test-time augmentations (1 = none)
    min_regions_before_train: int = 8  # cold start: regions the producer finishes before training begins
    lookahead_extra: int = 4           # the "+4" of L = ceil(T_produce / T_train) * K_active + 4
    train_min: float = 20.0            # timeshare: minutes of training per phase
    produce_max_min: float = 10.0      # timeshare: maximum minutes of production per phase
    reserve_gb: float = 50.0           # production pauses below this much free disk
    round_steps: int = 20000           # the MINIMUM steps a round trains (counted from its own start)
                                       # before the round gate may promote it
    compile: bool = True               # torch.compile the student (and the teachers, when teacher_bf16)
    teacher_bf16: bool = True          # teachers under bf16 autocast (+ compile): their fp32 forward was
                                       # the whole cost of a teacher region
    gn_bf16: bool = False              # the student's GroupNorm+SiLU outputs leave in bf16 under autocast
                                       # (float32 statistics; `model.NormAct`): about half the activation
                                       # memory. The trainer, its eval/cascade nets and the producer's
                                       # student (from the checkpoint's cfg) all follow it

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


def stored_fingerprint(stored):
    """The fingerprint of a STORED config (`Config.to_json()`: `<out>/config.json`, a checkpoint's
    `cfg`) as THIS code computes it: its field values rebuilt into a Config (fields it predates take
    their defaults) and hashed with today's FINGERPRINT_EXCLUDE. Comparing that, not the hash written
    at the time, is what "everything but FINGERPRINT_EXCLUDE must match" means: moving a field into
    the exclude set changes every hash, and paris4's resume after `cascade` / `cascade_drop` were
    excluded failed on the stale one although every remaining field matched. A stored config without
    its field values falls back to the hash it carries."""
    d = (stored or {}).get("config")
    if not isinstance(d, dict):
        return (stored or {}).get("fingerprint")
    return Config(**{k: _coerce(k, v) for k, v in d.items() if k in _TYPES}).fingerprint()


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
