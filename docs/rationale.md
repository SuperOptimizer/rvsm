# Why rvsm exists, and why each choice was made

This document is the evidence trail behind [`plan.md`](plan.md). Every number in it was measured on a
real run; the source is cited as a section of `usrm2/docs/unified_design.md` (the usrm2 design document,
2508 lines, `/home/forrest/usrm2/docs/unified_design.md`) or as one of the project memory notes under
`/home/forrest/.claude/projects/-home-forrest-usrm2/memory/`. Nothing here is a projection unless it says
so.

Reading order for a reviewer: §1 (what usrm2 was and how it failed), §2 (the measured numbers that must
be beaten), §3 (each design decision, its evidence, and the alternative rejected), §4 (non-goals).

---

## 1. What rvsm replaces

### 1.1 usrm2's shape

usrm2 trains a single multi-rung student over a 12-rung power-of-two resolution ladder (rung k = 0.6·2^k
µm; 2.4 µm = rung 2) on PHerc Paris 4, supervised by upstream surface predictions
(`unified_design.md` §1-§11). It works: the numbers in §2 below are usrm2's. What stopped working was
everything around the training loop.

By 2026-09-22 a single logical run spanned four hosts:

| host | role |
|---|---|
| `forlindesk2` (2×5060 Ti) | recto teacher region stores, the publisher loop, the verso mirror, the distance-pyramid jobs, and the **only ssh hop to the A100** |
| Thunder A100 (`tnr-0`) | the trainer (`u1` → `u2` → `u3` → `u4`) |
| RunPod RTX 5090 pod | the verso label producer (`runpod-5090-verso.md`) |
| `dl.ash2txt.org` | the exchange: every store moved between hosts through an sftp mirror |

(`usrm2-runs-state.md`, 2026-09-20 through 2026-09-22; `usrm2-unified-plan.md` 2026-09-21.)

### 1.2 The failure modes, each with its incident

These are the concrete events the plan's Context section refers to as "most incidents of the last two
days were the coordination itself".

**(a) A repack under a live reader corrupted the reader.** `cloud/repack_regions.py --strip-zstd`
rewrote finished stores in place to fix the codec chain (§24). It ran while the A100 trainer was reading
those same stores. The trainer died at 18:01 UTC on 2026-09-21 and had to be resumed from step 19k with
`~/u2_a100_resume.sh`. `unified_design.md` §24 records it under the heading "An in-place conversion under
a live reader corrupts that reader (2026-09-21, learned the hard way)".
*rvsm's answer:* a store is written once, to `.zarr.tmp/`, renamed, and `done` is set last; a finished
store is never rewritten in place, and a new round writes a **new** `stores/round_<r+1>/` tree rather than
touching round r (plan §2, §10).

**(b) Fetch starvation and network contention on a shared link.** Pulling a 718 MB checkpoint off the
A100 while it was training stalled the run to ~30 s/step for ~20 minutes — Thunder's proxied GPU shares
the instance's network link (`usrm2-findings.md`, 2026-09-21). Separately, the origin serves ~10.6 MiB/s
with 16 keep-alive connections in **one** session but ~3 MB/s plus errors with 64 churning jobs, and
desk→pod direct is 0.07 MB/s, so large files had to be relayed through the sftp server
(`runpod-5090-verso.md`, "Network").
*rvsm's answer:* one keep-alive aiohttp session, 16-32 connections, no publishing at all in v1 (plan §1,
§9); everything a run needs is on the machine it runs on.

**(c) sftp operation counts, not bandwidth, were the bottleneck.** An unsharded 1024³ store is 514 files,
so publishing one cost ~590 sftp operations for a median 10 MB of data; thousands of regions hammered the
server. Sharding it to one shard file made a store 2 files and 7 sftp operations, with a whole batch per
session (§18.1b, `usrm2-store-format.md`).
*rvsm's answer:* sharded zarr v3 is the only store format, from the first commit (plan §2, §10).

**(d) The desk outage cut the only hop.** On 2026-09-22 ~14:05 UTC `forlindesk2` went unreachable
(Tailscale, no ping or ssh). It is the only route to the A100 and runs the teacher writers, publisher,
sync loop, verso mirror and the distance-pyramid jobs, so the outage stopped all of them
(`usrm2-runs-state.md`, 2026-09-22). The plan's §10 notes it was still down when the plan was approved.
*rvsm's answer:* one machine, no hops (plan Context).

**(e) Two libvolcomp builds decode the same bytes differently.** The desk's `libvolcomp.so` is
`-march=native` (SIGILL on Xeon instances), and "the same volcomp bytes decode to slightly different
arrays on the desk and the A100 (different libvolcomp builds)" — pre-existing, recorded in
`usrm2-runs-state.md` 2026-09-21 and `usrm2-streaming.md`.
*rvsm's answer:* every store records `volcomp_build` (the sha256 of the shared library that wrote it) in
its attrs, and §8's cross-check between the A100 and the desk teacher store is stated as "dice > 0.99
within one volcomp build", not as bit equality (plan §2, §8, §10).

**(f) Silent degradation when the codec was missing.** usrm2's rung ladder silently dropped the levels it
could not decode when `libvolcomp.so` was absent.
*rvsm's answer:* `rvsm.ladder` raises with the reason (plan §10; README "libvolcomp").

### 1.3 Two more things that had to change

**The published-mask bootstrap is a ceiling, not a target.** usrm2's whole `u1` line was trained on
upstream published binary masks (`usrm2-unified-plan.md`, 2026-09-19 evening). §2 shows the model passing
its labels on localisation and matching them on merges. rvsm runs the teachers itself and then distils
itself, so there is no published-mask dependency at all (plan Context, §3).

**"I don't want to train over the same data multiple times."** The user's directive of 2026-09-20 evening
(`usrm2-unified-plan.md`) turned training into a streaming region walk, every region once. rvsm keeps the
walk but makes it the *production* order too, so the teacher and the trainer traverse the same list —
which in usrm2 was a contract two separate services had to be configured identically to honour (§18.1).

---

## 2. The measured numbers a reviewer needs

### 2.1 Surface metrics vs the label ceiling (val box, 2.4 µm)

Box `34432 15104 18432 + 256 1024 1024`, 11 published surfaces, 23 724 mesh points, **2.14 m of meshed
surface**, thr 0.5, window 128 halo 16, 200 seeded bootstrap draws over the surfaces (`unified_design.md`
§25.6). "Ceiling" is the *published recto mask itself* scored the same way: the best any model trained on
those labels could score.

| metric | u1 @60k | u3 @21k | ceiling (labels) | 95% CI (u1) |
|---|---|---|---|---|
| `recall@4` | 0.806 | **0.822** | 0.878 | [0.709, 0.848] |
| `continuity` | 0.656 | **0.705** | 0.729 | [0.589, 0.680] |
| `merge_frac` | 0.440 | **0.331** | 0.397 | [0.408, 0.501] |
| `offset_le3` | 0.363 | **0.397** | 0.326 | [0.291, 0.387] |
| `erl_um` | 237 | **409** | 406 | [137, 282] |
| `lost_merge_frac` | 0.350 | **0.236** | 0.353 | [0.315, 0.369] |
| `betti0_err` | 236 | 108 | 92 | (ref b0 21) |

u1 = `u1_30m6_p4` @60k, 30m6 at 256³, published masks only, `geo` augmentation (§19, §25.6).
u3 = `u3_30m6_cv` @21k, `--cascade mix --verso --cout 2 --aug full2`, streaming, verso stores from the
5090 pod, **no** recto teacher soft targets (`usrm2-runs-state.md` 2026-09-22 03:43;
`/vesuvius/usrm2/eval/u3_0343.json`).

Four conclusions a reviewer should carry into the review:

1. **Every headline CI is 4-10 points wide.** A 2-point difference between checkpoints on this box is not
   evidence (§25.6 conclusion 4, §25.7). rvsm's round gate is therefore written against the bootstrap CI,
   not against a point estimate (plan §1 "Round 0 verso gate, and self-distillation rounds").
2. **The model already beats its labels on localisation.** `offset_le3` 0.363 vs a ceiling of 0.326, and
   `offset_mean` -1.23 vs -3.15: the published band sits ~3 voxels inside the mesh (§25.6 conclusion 1).
   More teacher data cannot fix localisation; a distance head measured against meshes can.
3. **The labels merge as much as the model does.** `lost_merge_frac` 0.350 for u1 against **0.353 for the
   ceiling** — statistically the same number — while `lost_break_frac` is 0.189 vs 0.120 (§25.6
   conclusion 2). A third of the meshed surface has a second sheet on the same normal ray *in the labels
   too*. This is the single most important measured fact in the project: **merges are a property of the
   supervision, so label-free losses, not more teacher data, are the lever.** Every Phase-A/B/C term in
   rvsm's fixed recipe exists because of this line.
4. **The cascade + verso run beats the ceiling's merge behaviour.** u3 merges *less than its own labels*
   (0.331 vs 0.397) and its ERL equals the ceiling (409 vs 406) — evidence that the direction is right
   and that a self-distillation loop has room above the label source.

### 2.2 Validation dice (A100, 30m6, 256³)

`u1` bootstrap: 5k 0.626 → 20k 0.686 → 40k 0.717 → 60k 0.720 (cosine to 60k; +0.003 over the last 15k);
final per-rung r2/r3/r4 = 0.740/0.740/0.682 (§19, `usrm2-findings.md`). A warm start with `--aug full`
drops to ~0.68 first, then climbs. `u3` reached rung-2 dice 0.744 at 12k — above u1's final — and 0.704
overall / 0.744 rung-2 at 28k (`usrm2-runs-state.md`). Streaming never repeats data (2.6 M windows per
epoch = 1.3 M steps at batch 2), so a plateau is set by the LR schedule and capacity, not by data.

Earlier, smaller reference points (`usrm2-findings.md`): the desk 5m `u1_5m_p4` reached recall@4 0.753 /
continuity 0.600 / merge 0.44 (§13); round-3 raw 0.729/0.584/0.42; the recto teacher itself scores
recall@4 0.843, the m7 teacher 0.863. The 5m/128³ configuration establishes the **sign** of a change,
never its final numbers (synthesis_v2 §5 cost anchors).

### 2.3 Training speed, by card

| card / config | throughput | source |
|---|---|---|
| A100, 30m6, 256³, b2, before the fixes | 11.4 Mvox/s (2942 ms/step) | §14 table |
| A100, same, `ckpt_act 0` + `up2x` + NCDHW + compile | **32.6 Mvox/s** | §14, §14b, `usrm2-runs-state.md` |
| A100, `--aug full` | ~22-30 Mvox/s (the aug costs ~25 % on average) | `usrm2-findings.md` 2026-09-21 |
| A100, u4 at `--ckpt-act 1` | 20 Mvox/s, 56.5 GB VRAM | `usrm2-runs-state.md` 2026-09-22 |
| Thunder A6000, 12m at 256³ region streaming | 6.5-9 Mvox/s | `usrm2-streaming.md` |
| Thunder A6000, u4 (30m6, Phase A, `--ckpt-act 2`) | ~4.4 Mvox/s — **4.7× slower than the A100** | `usrm2-runs-state.md` 2026-09-22 |
| desk 2×5060 Ti, 5m at 128³ DDP | 15.8 Mvox/s | `usrm2-runs-state.md` |
| RTX 5090 pod, 30m6 inference, window 256 halo 32 | 12.5 s per occupied region, 10.1 s/region marginal, 99 % GPU duty | `runpod-5090-verso.md` |
| desk 5060 Ti, recto teacher, TRT fp16, one 1024 tile | 137.8 s → ~48 s/region, 13.4 GB peak | `usrm2-findings.md`, `usrm2-runs-state.md` 2026-09-20 |

Where the A100 step went, and what fixed it (§14, §14b) — all four are load-bearing for rvsm's ported
`model.py` and `train.py`:

- `F.interpolate(mode="trilinear")` at the decoder's widest stage (2×64×256³) costs 91 ms forward but
  **1194 ms forward+backward**: its backward is a scatter-add with atomics. `model.up2x` writes the exact
  2× case as gathers, so the backward is a gather: 1762 → **651 ms**. Numerically exact to 2e-7 in fp32;
  one bf16 ulp under autocast, measured effect on step-8000 dice 0.65403 vs 0.65416.
- `channels_last_3d` was a **pessimisation**: compiled GroupNorm+SiLU on 2×32×256³ costs 96.2 ms
  forward+backward in `channels_last_3d` against **15.7 ms contiguous** (§14b). The net runs NCDHW.
- `--ckpt-act 0` is both faster and no more expensive in memory **with `torch.compile`** (46.8 GiB);
  without compile the same configuration needs 67 GiB and OOMs. Compiling is what makes it fit.
- Rejected with numbers (§14): `cudnn.benchmark` (no gain, peak 41 → 71 GiB), an H2D prefetch stream
  (3 % *slower*; the proxied GPU link is 2.3-2.5 GiB/s pinned or not, ~150-167 ms/step irreducible),
  batch 3, fused hand-written GroupNorm, bf16 loss and fused AdamW (< 1.5 %).

Streaming (`usrm2-streaming.md`, §16-§18): plain random windows fetch **39.5 bytes per training voxel**
at a 0.24 shard hit rate and stall; `--region 1024 --windows-per-region 64` fetches **1.37 bytes per
voxel** at a 0.98 hit rate, `stream_wait` ~1-6 ms/step. That ratio is why `region = 1024` and
`windows_per_region = 64` are fixed defaults and not tunables.

### 2.4 Store format facts

From `usrm2-store-format.md`, §18.1b, §24 and §29.1:

- **Every array is a zarr v3 sharded array**: one shard per 1024³ box (`shard_shape = min(1024, shape
  rounded up to the 128 chunk)`), 128³ inner chunks. A finished region store is 2 files (`zarr.json` +
  `c/0/0/0`) instead of 514. Measured cost of sharding: filling a 1024³ store in 256³ blocks takes 41.6 s
  sharded vs 41.3 s unsharded; 200 random 128³ reads cost 1.93 s vs 1.76 s — i.e. free.
- **The codec chain is exactly `[volcomp]`, `compressors=None`.** zarr-python's
  `create_array(serializer=VolcompCodec)` silently appends its default zstd; the measured saving is
  **0.2 %** on volcomp output, for a decode step on every read, and the C exports and the CT are
  volcomp-only. (User: "+ zstd? thats not a thing".)
- **q8 for probabilities, q0 (lossless) for distance fields.** Writing a spiral test field at q8 and
  reading it back turned a stored `0` into a `6` — which decodes to a **-30.5-voxel distance** that
  nothing downstream can distinguish from a real one, and the rounding compounds under the partial-chunk
  writes that a block smaller than 128 makes (§29.1). A rounded probability is harmless; a rounded
  distance is a wrong distance and a rounded no-data marker is a lie. Hence `q8` on
  `recto/rw/verso/conf` and `q0` on `midline/thickness` (plan §2).
- **Encoding of the distance fields** (the tracer contract): signed `code = 128 + round(d/0.25)` clamped
  to 1..255; unsigned `code = round(t/0.25)`; **`code 0` = NO DATA**, everywhere, which is why the cap is
  ±31.75 rather than ±32 (§29.1).
- **One write per shard, never a rewrite in place** (§24, incident 1.2a).
- **`volcomp_build`** — the sha256 of the `libvolcomp.so` that produced the bytes — is a store attr
  (incident 1.2e).
- The teacher run on volcomp q=8 CT input scores dice **0.87-0.91** against the published prediction made
  from raw data, and deblocking changes nothing (`usrm2-findings.md`, 2026-09-16). That is the price of
  training off the compressed mirror, and it is paid once.

### 2.5 TensorRT: where it works and where it does not

- **Teachers, yes.** The desk's fp16 TRT recto engine plus a single 1024 tile took the region teacher
  from 137.8 s to ~48 s/region at 13.4 GB peak, output byte-identical to the 2×2 tiling it replaced; m7
  went 14.8 s (torch) → 10.0 s (TRT) (`usrm2-findings.md`, `usrm2-runs-state.md` 2026-09-20).
- **The student, no.** On the 5090 pod, the TRT engine build **fails on the `model.up2x` gather graph**:
  Myelin reports "Could not find any implementation ... Slice" (`runpod-5090-verso.md`, "Rejected"). The
  student runs compiled torch (`max-autotune-no-cudagraphs`, bf16, NCDHW, window 256/halo 32, batch 1,
  fp16 accumulators).
- **No INT8**, by explicit user decision 2026-09-20: keep fp16 TRT, because the probability ramps matter
  (`usrm2-findings.md`).
- Window size is bounded by the builder, not the model: windows 384/512 OOM on 16 GB (the TRT builder
  wants 8.9/25.8 GB for one window).
- **Reproducibility floor**: two eager bf16+cuDNN runs of the same region agree only to dice 0.994 /
  max 0.11, so every tolerance check must be looser than that (`runpod-5090-verso.md`).

This is exactly why the plan puts TensorRT in v1 **for the teachers only**, behind a fallback ladder to
torch bf16 on any build or load failure (plan Context, §5 `trt.py`), and why `test_infer_export` must
exercise the fallback path.

### 2.6 Verso: why the flip trick is a student trick, not a teacher trick

- **Flipping the teacher does not produce verso labels.** Mirror flips of the recto teacher keep the band
  on the **same** face (dice 0.77-0.81 against the unflipped run); only sheet-sideways axis swaps move
  it, and those are out of distribution. The teacher is using intrinsic cues (fibre texture, curvature),
  so there are no free verso labels from teacher symmetry (`usrm2-findings.md`, 2026-09-16).
- **Negating the radial channel of the *student* does move it.** Both student heads move to the far side
  of the sheet; at round 3 the flipped output was a broad blob filling the outer half, 2-3× the recto band
  volume, whose outer skin is a thin continuous line, 72 % of it within 3 voxels of a CT papyrus-ends edge
  (`usrm2-findings.md`, 2026-09-18). A "next recto band minus 3" rule is wrong wherever a gap exists.
  Paris 4 sheets: median CT-march thickness ~10 voxels (p90 25); the next sheet is ~35 voxels away.
- **The raw flipped labels worked better than a cleaned-up version.** `r3_5m_vraw` beat `r3_5m_vskin`
  (recall@4 0.729 vs 0.722, continuity 0.584 vs 0.548), and the added verso heads did **not** hurt the
  recto heads (`usrm2-findings.md`, `usrm2-runs-state.md`).
- **But only once the recto is good enough.** The user's own deferral (2026-09-21): a strict manifold
  recto/verso loss is wanted, but applied while the verso channel is still a diffuse copy of the recto it
  "would push the channels apart arbitrarily"; the precondition is that the verso lands on the far face on
  its own (`usrm2-unified-plan.md`, "Deferred").

Those three facts are the whole of rvsm's verso design: `verso_source = "flip"`, a **gate** before verso
production starts in round 0 (`verso_after_steps = 10000` as the fallback), verso loss weight 0 until
then, and mutual exclusion / construction-based pairing as recipe terms rather than a separate label
source (plan §1, §4).

---

## 3. Each design decision: evidence, and the alternative rejected

### 3.1 A greenfield standalone project rather than another usrm2 run

**Evidence:** §1.2 — every incident of 2026-09-21/22 was coordination, not modelling. usrm2 carries a
~350-line planner process, a queue-replay protocol with per-worker RNG state, DDP, sftp publishing, mirror
conventions, 25 augmentation ablation presets and a `--flag` for each of ~30 research decisions that have
since been settled by measurement.
**Rejected:** deleting the coordination from usrm2 in place. Rejected because the settled decisions are
*defaults* in rvsm and *flags* in usrm2, and the resume/`grow` machinery exists precisely to let old flag
combinations keep running — stripping it would break every existing checkpoint while leaving the
four-host topology intact.

### 3.2 Two processes, filesystem as the bus

**Evidence:** proxied GPUs (Thunder) scale across **processes**, not threads (plan §1); the A6000 hung
twice on a futex under a proxied GPU (`usrm2-streaming.md`). Region state derived from disk is what made
usrm2's producer/consumer contract auditable: a store is done iff its `zarr.json` says `done: true`
(§18.1). `rvsm ledger --rebuild` is then just a directory scan.
**Rejected:** a sqlite job table or a third planner process — explicit user decision (plan Context #4).
A planner process is what usrm2 had; it required the planner and the teacher service to be configured
with *identical* stores, rungs, boosts, patch, region, exclude and seed, since every one of them feeds
`region_list`/`walk_order` (§18.1). One process owning the cursor removes the class of bug.

### 3.3 On demand, per 1024³ region, with a lookahead

**Evidence:** §2.3 streaming numbers — region mode is a 29× reduction in bytes fetched per training voxel
and takes the shard hit rate from 0.24 to 0.98. Precomputing instead: the recto teacher over all 24 192
Paris 4 regions is ~5.6 days on two 5060 Ti and ~220 GB, and the verso pass over 25 093 regions is ~70 h
and ~$48 on a rented 5090 (`usrm2-findings.md`, `runpod-5090-verso.md`). A run that needs a few thousand
regions should not pay for the corpus.
**Rejected:** precomputing the whole scroll's teacher stores first (what usrm2 did, and what required the
publish/sync/mirror machinery). Also rejected: letting the trainer block on production — the trainer
takes the first ready region in the lookahead window and logs `train_wait_s` instead (plan §1).

### 3.4 Never look up published masks or published stores

**Evidence:** §2.1 conclusion 2 and 3 — the published mask is 3 voxels off the mesh and merges as much as
the model. §25.6 exists to say that the published mask is a **ceiling to be measured against**, not a
supervision signal to be improved by adding more of. And operationally, the published-store dependency is
what produced incidents (b) and (c).
**Rejected:** the usrm2 bootstrap (`usrm2-unified-plan.md`, 2026-09-19 evening: "Bootstrap training uses
ONLY the exported upstream mask pyramids"). It was the right call for usrm2 — cheap dense coverage of
every rung without new inference — and it delivered u1. It is the wrong call for a system that runs its
own teachers, because it reintroduces a network dependency for a signal with a known, measured ceiling.

### 3.5 Round-0 teachers = recto + m7, fused by agreement

**Evidence:** the two lineages' bands differ *systematically* — 32 % vs 23 % coverage of a mid-scroll cube
— so a "last writer wins" rule decides per voxel by lineage order, arbitrarily (§26.3). Agreement fusion
computes a confidence-weighted mean `p = (a_s p_s + a_m p_m)/(a_s + a_m)` with `a_x = w_x(c(p_x)+0.05)`,
`c(p) = 1 - H2(p)/ln 2`, and sets the loss weight to `w_mul = 1 - |p_s - p_m|`: a disagreed voxel is
smoothly down-weighted rather than decided (UA-MT's "gate by agreement, not by one threshold"). Reference
scores: recto teacher recall@4 0.843, m7 0.863; the teacher band is fat, m7 thinner
(`usrm2-findings.md`).
**Rejected:** (i) per-lineage heads — the user settled on ONE recto head fed by both, 2026-09-19
(`usrm2-unified-plan.md`); the two-head `r2_5m_all3` run measured the m7 head as the laggard (0.552 vs
0.687 recall@4). (ii) GLC-measured per-source weights as a round-0 default — `usrm2 glc-weights` exists
but §26.3 flags two caveats it cannot fix (`fpr` is a proxy; the verified boxes are where labelling was
easy), so a fitted weight is a prior, and §26.6 says to run it as a separate arm, one change at a time.
(iii) More teachers (ink, fiber, lasagna) — out of scope (plan §9).
**rvsm keeps the `rw` channel** (the agreement weight) as a first-class store channel precisely so the
fusion is auditable after the fact (plan §2).

### 3.6 Verso by the flipped radial sign, behind a gate

**Evidence:** §2.6 in full. The flip works on the student and not on the teacher; the raw flipped labels
beat a cleaned "skin" variant; the verso heads did not hurt the recto heads; and the user's deferral names
the precondition (the verso must land on the far face on its own).
**Rejected:** (i) a dedicated verso teacher — none exists, and §2.6 shows flips of the recto teacher do
not make one. (ii) a geometric "next recto band minus 3" rule — measured wrong wherever a gap exists
(lands across the air). (iii) starting verso at step 0 — that is the configuration the user explicitly
deferred. The gate is `recall@4` and continuity against the fused-teacher reference within the bootstrap
CI, with `verso_after_steps = 10000` as an unconditional fallback (plan §1).

### 3.7 Self-distillation rounds, gated on merges and topology

**Evidence:** u3, with no recto soft targets at all, beat the published-label ceiling on `merge_frac`
(0.331 vs 0.397) and matched it on ERL (409 vs 406) — a student out-performing its own supervision, which
is the entire premise of a self-distillation loop (§2.1). Conversely §25.6 conclusion 4 says a 2-point
move is noise, so a round must be gated on the CI, and the discipline of "drop any term whose loss curve
moves while its own target metric does not" is §26.6's decision rule.
**Rejected:** ungated rounds. The gate requires `merge_frac` **and** `betti0_err` not worse than the
round-0 reference beyond the CI *and* `recall@4` within CI, and a failing round is discarded with the
previous teacher kept and training extended — which WSD makes free (plan §1). At most two rounds live on
disk at once.

### 3.8 Every v2 improvement is a fixed default, not a flag

**Evidence:** §26.5 and §29.7 measured each Phase A/B/C term's real cost as a percentage of a whole
training step on the same card under the same protocol. L3, L4 and L8 are each inside the ±2 % noise
floor; the whole Phase B stack (six planes, six head channels, Huber, Eikonal, thickness, normals, the
constructed pair) is inside a ±5 % pass-to-pass spread; only ECT costs anything (+18 % at `ect_n 1`).
When the cost of a term is below the measurement noise and the evidence is positive, a flag is a way to
accidentally run without it.
**Rejected:** keeping them as flags. usrm2 carries ~30 of these and 25 augmentation ablation presets; the
plan drops the ablation presets outright (§5 `aug.py`, §9).

### 3.9 `Layout` as the single channel contract

**Evidence:** this is the class of bug usrm2 kept hitting as the stem and head grew: `cin` 14 → 15 with
the cascade channel needed a bespoke `warm_start(cascade=, src_scale=)` rather than the naive rule (§22);
`cout` 2 → 2+9 with affinities needed an `ncopy` cut so probability filters were not copied into affinity
rows (§26.1); `cout_t` → distance rows needed `ncopy` to stop at `cout_p`, because "copy mod n" is right
for a probability row and **wrong** for a distance one (§29.2); and `--planes meta,radius` had to be
canonicalised to `radius,meta` so a checkpoint is unambiguous (§29.5). `rvsm/config.py` puts the stem
order, the head order, `RUNG_ITEM_KEYS` and `head_names` in one frozen dataclass, and the sampler, prep,
model stem, warm start and exporter derive from it.
**Rejected:** deriving channel indices at each use site (usrm2's `train.stem_map` matched stem slots by
name at warm-start time). `RUNG_ITEM_KEYS` exists for the same reason on the loader side: a key added
without a producer fails loudly instead of vanishing.

### 3.10 Occupancy from the CT, not from a mask

**Evidence:** usrm2's `data.occupancy` took a block **max** over the published mask pyramid to decide
which 1024³ tiles are worth visiting. rvsm has no mask, so `tiles_fraction` is the block **mean** of
`ct_level > 0` at the occupancy rung, thresholded at ≥ 0.05 at rung 2 and ≥ 0.01 at rungs ≥ 5, with draw
weight ∝ √fraction (plan §3, `config.occ_min_fine` / `occ_min_coarse`).
**Rejected:** keeping the mask-derived occupancy (impossible without the published masks) and uniform
tiling (§18.2 measured what happens when the walk's weights are wrong: on the first 968 windows of the
desk soak, rung 2 got 120 windows although it is 69 % of the regions).

### 3.11 The `mix` walk, `visits_max 64`

**Evidence:** §18.2, measured. A weighted shuffle front-loads the heavy items: Paris 4 at rung 9 is ONE
region, `--rung-boost 9=16` asks for a large share, and all ten such regions across the 8-scroll stores
file are drawn in the first few hundred — after ~640 windows rung 9 never appears again. `--walk mix`
gives a region `round(w · regions)` visits of weight `w/visits` (capped at 64), so almost every entry
weighs `1/regions` and a group's share of the walk is its intended share *all the way through*. A second
visit draws its own windows from its own RNG: denser sampling, never a repeat.
**Rejected:** a single weighted shuffle (the prefix mix is right, but it is right once).

### 3.12 Sharded zarr v3, volcomp only, q8/q0, one write, never in place

Covered by §1.2(a,c,e) and §2.4. **Rejected:** zarr v2 OME output (deleted 2026-09-21 rather than
exempted, §18.1b); `[volcomp, zstd]` (0.2 % for a decode step on every read); q8 distance fields (a
stored 0 came back as 6); in-place repacks (killed a training run).

### 3.13 TensorRT for the teachers, with a fallback ladder

Covered by §2.5. **Rejected:** TRT for the student (the build fails on `up2x`); INT8 (user decision);
larger inference windows (builder OOM at 384/512 on 16 GB; on the 5090, windows 320/384 also change the
per-window z-score enough to give dice 0.93 against window 256, *and* are slower end to end).

### 3.14 Two GPU modes, with a VRAM table checked before the run starts

**Evidence:** the production configuration measures 46-57 GB for the trainer (30m6, 256³, batch 2,
`ckpt_act` 1-2) — u4 at `--ckpt-act 1` was 56.5 GB on the A100, and u1 at `--ckpt-act 0` 46.8 GiB — and
Phase A adds ~0.6-0.8 GB on top (§26.5 arithmetic for the affinity targets and skeleton temporaries).
On a 48 GB A6000 that forces `--ckpt-act 1` to keep batch 2, with `--batch 1 --accum 2` as the fallback
(~15 % slower, §26.6). The teacher slot is ~13 GB TRT / ~14 GB torch bf16 (13.4 GB measured peak on the
desk), the student ~16 GB. The user's decision was to make the defaults fit **both** 80 GB resident and
32 GB timeshare.
**Rejected:** a single mode. Also rejected: leaving the budget implicit — the plan requires the process
to refuse to start when the table sum exceeds `vram_total - 4 GB`, and to set
`torch.cuda.set_per_process_memory_fraction` per process (plan §1).
For **timeshare**, both processes stay alive across a phase swap because the compile cache survives and a
recompile on `.cuda()` is ~1 min, under 5 % of a 20-minute phase; TRT engines are built once per host and
deserialised in seconds.

### 3.15 The recipe terms (short form; the long form is [`recipe.md`](recipe.md))

| decision | evidence | alternative rejected |
|---|---|---|
| `loss_excl 0.1`, `loss_selfcons 0.1`, `loss_skel 0.05 / iters 4` | §26.5: each inside the ±2 % step-cost noise floor; §25.6 conclusion 2 says merges are the target | omitting them; villa's `NormalGatedRepulsionLoss` (L1), demoted — unablated anywhere (synthesis_v2 §6.2) |
| `aff_offsets (8,16,32)`, `loss_affinity 0.1` | sheet pitch is 15-35 voxels **at rung 2**, so 32 brackets rung 2, 16 rung 3, 8 rung 4 — the three rungs carrying almost all sampling mass (§26.1, §26.6); the EM-connectomics affinity lineage is measured at CREMI/FIB-25 scale | a single offset (brackets one rung only); radial rather than axis offsets (a gather per voxel per offset, and no clean rule under the spatial augs); `--affinity-all` (>95 % trivial "one end is air") |
| `loss_sdist 1.0`, `loss_eikonal 0.1`, midline+thickness construction pairing | the cortical white/pial literature moved from penalties to construction; §29.3 *proves* `relu(p_r+p_v-1) ≡ 0` for every `t ≥ 2·half`, so crossing is unrepresentable rather than discouraged | a face-signed distance plus a repulsion penalty (v1's recommendation); `pair construct-only` — it starves the deep-supervision heads, so `construct` keeps the learned rows as the control |
| `loss_ect 0.05`, `ect_n 1`, `ect_rung 2` | §29.7: +18 % at `ect_n 1`, +43 % at `ect_n 4`, against experiment 7's own "< +20 %" budget | `ect_n 4` (over budget before it has shown anything); `--loss-warp` homotopy warping — needs a per-step GPU EDT plus a search, deliberately left undone (§29.4) |
| `sched wsd`, `cooldown 0.1`, `ema auto` / `ema_k 50`, `rewarm 800`, `new_param_lr_mult 3` | §26.2: the step budget need not be committed at run start, which is how these runs are actually managed (u1→u2→u3→u4 were all warm starts or extensions); `ema_k 50` = a 2 % window, the middle of the literature's 1-3 % | a fixed cosine to a committed step count, and a fixed `ema 0.999` (1.7 % of a 60k run, 0.5 % of a 200k one) |
| `calibrate = True` after every eval | dice-trained networks are measurably overconfident, and every uncertainty-gated mechanism downstream needs the sigmoid to be a probability (§26.4) | leaving it to a manual pass |
| `aug full2` | §27: blur sigmas in microns not voxels (the same config is a different physical blur at every rung), `_paganin_jitter`, shuffled artefact order; the histogram-matching result (teacher recall@4 on 1667 0.48 → 0.62 after matching to Paris 4) shows the teacher is sensitive to the per-volume uint8 window | `geo` only (48 symmetries), which is what u1 ran; the 25 ablation presets, dropped |
| `cascade mix`, `self_p 0.1 → 0.7`, `drop 0.1`, `noise on`, `cascade_depth 3` | a fixed self-p of 0.5 is textbook exposure bias (§26.1); the measured step cost is mask 1.07×, self 1.54×, mix 1.35× (§22) | `cascade self` (1.54×, and no coarse-target control); unconstrained multi-step self-feedback — the one-level truncation is kept deliberately, it diverges without damping |
| `planes radius+meta` (5 meta planes) | §29.5: the radial *vector* gives direction, not how far out a voxel sits, which is what sets sheet spacing, curvature and damage; the metadata planes restore the absolute intensity the per-window z-score throws away | angular position and absolute z — they break the symmetry augmentations and carry little (§21); local filters (sharpen/Sobel/sheetness) — the first 3×3×3 layers learn them (§21) |
| `infer_window 256`, `infer_halo 32`, `tta 1` | the 5090 production settings, measured (`runpod-5090-verso.md`); teacher 4-flip TTA buys recall@4 0.851 vs 0.844 at 4× cost, and student 8-flip TTA +1 pt | larger windows (dice 0.93 and slower); TTA in production (measured gain inside the eval CI, at 4-8× cost) |

---

## 4. Non-goals for v1, and why

From plan §9, each with the reason it is out of scope rather than merely unbuilt.

| non-goal | why |
|---|---|
| **No planner process, no queue replay** | §3.2: the process pair with the filesystem as the bus replaces both; the replay protocol (per-worker RNG state, `queue.jsonl`, torn-line recovery, §16) exists to make a *separate* planner reproduce the sampler, and there is no separate planner |
| **No DDP, no multi-node** | one machine is the premise (§1.2d); DDP in usrm2 exists for the desk's 2×5060 Ti, and the first real run is a single A100 |
| **No multi-scroll runs** | the user scoped the streaming run to Paris 4 only, with a whole-corpus fine-tune as a later, separate run (`usrm2-unified-plan.md`, 2026-09-21); cross-scroll transfer is unsettled — after the 2026-09-17 axis-bug correction, *every* augmentation preset landed within noise of `geo` on PHerc1667, deleting the earlier "lowres helps cross-scroll" conclusion (`usrm2-findings.md`) |
| **No mask-pyramid targets, no published ceilings** | §3.4; `evalsurf --ceiling` on published masks is explicitly dropped (plan §5 `evalsurf.py`), replaced by `compare_stores(a, b)` on held-out regions and by `tifxyz` meshes when available |
| **No sftp publishing or mirror conventions** | §1.2(b, c); `--publish rsync://` is named as v2 |
| **Teachers recto + m7 only** | §3.5; ink/fiber/lasagna teachers and the encoder feature helpers are dropped from the tsm port (plan §5 `teachers.py`) |
| **No explicit normals head** | §29.9: three more channels buy nothing until something reads them, and the export derives the normal from the stored field by Scharr anyway. Normals and `gmag` are derived at export, never stored (plan §2) |
| **No `pair construct-only`** | §29.3: it drops the learned term, so the deep-supervision heads receive no gradient; `construct` keeps them as the control the constructed pair is measured against |
| **No Betti matching** | §29.4 / experiment 7's decision rule: adopt a topology term only if Betti error improves *and* step cost < +20 %; on a null result, do **not** escalate to a C++ Betti-matching dependency |
| **No `--loss-warp`** | §29.4: homotopy warping needs a per-step GPU EDT at 256³ plus a critical-voxel search — each a day's work with its own correctness suite — and the survey ranks fast ECT ahead of it. Left deliberately undone rather than half-built |
| **No muP** | synthesis_v2 §7.4: no 3D-conv evidence for any of Schedule-Free / Muon / SOAP / Sophia / Lion / muP; they would displace a working recipe for nothing |
| **No tifxyz refinement tool, no `glc-weights`, no ablation presets** | `glc-weights` is a separate experiment arm with caveats it cannot fix (§26.3); the ablation presets are research scaffolding, and the settled answers are defaults (§3.8) |

Also out of scope and worth stating because it is *not* in §9: the plan makes no attempt to reproduce
usrm2's cross-scroll results. The only cross-scroll facts rvsm inherits are negative (the axis-bug
correction above) or operational (histogram-matching a new volume to Paris 4 before teacher inference is a
cheap lever: teacher recall@4 on 1667 0.48 → 0.62, offset bias +0.95 → -0.11; on Paris 4 itself the same
trick *hurts*, 0.844 → 0.798).
