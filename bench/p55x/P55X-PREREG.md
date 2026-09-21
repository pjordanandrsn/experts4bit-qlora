# P55x — a replacement licensed int4 expert pack: built, gated, fingerprinted, and kept

Written before any P55x data exists. Record: [#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405)
(the open item this lane closes or narrows) and [#530](https://github.com/pjordanandrsn/experts4bit-qlora/issues/530)
(why the recipe cannot carry a licence). Owner directive (Jordan, 2026-09-21, chat): build the pack with the
streamed recipe dumping the artifact, K8 it, publish it if it passes, put `pack_fingerprint` on the licence rows
and supersede the fingerprint-less ones, and test same-box determinism — at or under $8.

**The name.** `P55x`, not `P55`. The P-numbers after P54 are reserved for the throughput levers
[#652](https://github.com/pjordanandrsn/experts4bit-qlora/issues/652) queues (fused dispatch permute/unpermute,
`_reduce_partials` into the GEMV epilogue, 4.5 → 4.125 bits/weight, the GEMV itself). This is a quality and
provenance lane and takes a name that cannot collide with them.

## The question

`docs/STATUS.md`'s first open bullet says it: **no licensed artifact hash is in the register.** The Qwen3-30B-A3B
serving licence — `e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`, K8 PASS on both texts
(wikitext −0.0528 ppl, c4val1 −0.0662) — names a *recipe*, and P37 showed the recipe does not carry that licence
to another host: the same recipe on a fresh 5090 read **11522 gptq / 766 rtn** expert matrices against the
licensed **11512 / 776**, and its K8 **failed c4val1 at +0.1093 ppl** against the +0.05 budget. #530 explains why
this is structural rather than incidental — the per-expert choice is `routed_rows >= min_rows`, a threshold on a
quantity that sits at the router-flip noise floor.

#405 and #531 built the machinery to license **bytes** instead: a hash-pinned artifact
(`experts4bit_qlora.engines.pack_manifest`), a root `pack_fingerprint` over the ordered `(path, size, sha256)`
payload tuples including the identity and gptq/rtn-assignment payloads, and a loader that **refuses** a
fingerprint mismatch and never rebuilds from the recipe. What is missing is the one thing the machinery cannot
manufacture: **a pack whose bytes were retained and which passed the gate.** bo6c's bytes were not kept.

So: **build one, gate it, keep it, and name it in the register.**

## Arms, in the order the box runs them

All on `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, one RTX 5090, P39's harness pieces
(`step_decomp.py`, `k8_bake.py`, `calib.json`) and P42's hook staged byte-for-byte against `staged.sha256`.
Recipe knobs are set explicitly rather than derived, so the pack's conditions are readable off the run and not
off a memory-budget heuristic: `E4B_SERVE_EXP_INT4_CALIB=1` (the calibrated expert path, which is what makes
this a GPTQ pack rather than an RTN one), `E4B_CALIB_NSEQ=128` (32 batches × 4 × 512 = 64k C4-validation tokens),
`E4B_CALIB_LAYERS_PER_PASS=10` (48 layers in 5 passes — the same chunking the 24 GiB budget derives, now stated),
`E4B_INT4_HESSIAN_BUDGET_GB=24`, `min_rows` 32, damping 0.01, `E4B_INT4_GPTQ_DEVICE=cuda`. K8 is the registered
instrument unchanged: teacher-forced NLL through the paged decode path, 2048 steps, 512-token prompt,
`--b1d-loop eager`, `--no-fuse-qkv`, fp8 paged KV, placement all-vram.

| # | arm | what it is |
|---|---|---|
| K0 | refusals | GPU class, free disk on `/root`, and an **upload probe** — before any fetch |
| 1 | `nf4` / wikitext | the K8 reference on this box |
| 2 | `nf4` / c4val1 | the K8 reference on this box |
| 3 | `build1` | the streamed 64k recipe, dumping the artifact → `FP1`; then the fetch marker. It also scores **wikitext from the live stores**, which is free in the same process and gives the dump → load round trip a cross-check against arm 4's score of the same bytes. That score is a cross-check, **never** a gate input |
| 4 | `lic` / wikitext | **primary**: artifact pinned by `FP1` + calibrated int4 attention + folds — bo6c's configuration |
| 5 | `lic` / c4val1 | same |
| 6 | `licrtn` / wikitext | **secondary**: artifact pinned by `FP1` + **RTN** int4 attention + folds |
| 7 | `licrtn` / c4val1 | same |
| 8 | `build2` | the identical recipe a second time → `FP2`, for same-box byte determinism |

`build2` runs last so a slow host loses the determinism measurement before it loses the gate.

**Every gate arm loads from the artifact, none of them rebuilds.** It would be ~10 minutes cheaper to let
`build1` score both texts from the live stores it just built, and P39's box 2 did exactly that. This lane does
not, because its entire subject is byte identity: the gate must score the bytes as the loader installs them,
which is what a user gets and what the fingerprint names. `build1`'s wikitext score is kept beside arm 4's as
the round-trip check that those are the same thing — which is the assumption the cheaper design would have had
to make silently.

### Why arm 6/7 exists, and what it is not

**The artifact pins the experts. It cannot pin the attention.** `engines/int4_attn_calib.py` has no
serialisation of any kind — no artifact, no manifest, no fingerprint — so the 192 calibrated attention
projections in bo6c's configuration are re-derived from a Hessian pass on **every** load, on whatever box does
the loading. A `pack_fingerprint` attached to that configuration therefore identifies one half of it. That is a
real hole in the plan as specified, and saying it is not enough: arm 6/7 measures the cheapest configuration
that does not have it. RTN int4 attention (`E4B_SERVE_ATTN_INT4=1`) is round-to-nearest from the revision-pinned
bf16 weights with no Hessian and no routing, so its bytes are a deterministic function of the checkpoint — the
whole quantised stack is then identified by `pack_fingerprint` plus the model revision. Arm 6/7 costs about
16 minutes because it skips the attention Hessian pass that arm 4/5 pays.

**Arm 6/7 licenses nothing by inheritance.** If it passes it is a NEW configuration with its own claim row; it
never supersedes bo6c, whose configuration is arm 4/5's. If arm 4/5 passes and arm 6/7 fails, the calibrated
attention is load-bearing and the residual stays open. The RTN attention path's own byte determinism is argued
above, **not measured here** — this lane has no instrument for it.

## Predictions, registered before the data

- **Q1 — the gate on arm 4/5 (`lic`).** Both texts within the +0.05 ppl budget. **I do not predict the
  direction with confidence and will not pretend to:** two builds of this recipe have been gated, bo6c passed
  (−0.053 / −0.066) and P37 failed c4val1 (+0.109), and c4val1 specifically has now failed three lanes (bo5
  `all`, bo6 Mixtral `lic_calibexp`, P37) while wikitext passed. The registered prediction is therefore about
  **which text binds**: if this pack fails, it fails on **c4val1**, not wikitext. *Refuted by:* a wikitext
  failure, with or without a c4val1 failure.
- **Q2 — same-box byte determinism.** `FP2 == FP1`, and the two artifacts' `method_map_hash`, row-count-vector
  hash and gptq/rtn counts all agree. Basis: `e4b.serve.buildout.bo6c.qwen3.calib-deterministic.5090.2026-09-05`
  — but read what it actually measured. It compared **mean_nll (spread 0.0 nats) and counts (10820 / 1468)** for
  the **16k**-token arm across two runs on one box. It never compared pack **bytes** — the fingerprint machinery
  did not exist until #405, a day later — and it was not the 64k recipe. So byte identity for this recipe is
  genuinely open and this is the prediction I am willing to stake. *Refuted by:* `FP2 != FP1`.
- **Q3 — the gptq/rtn split.** `build1` reads **11512 / 776** (bo6c's and bo7's split) or it does not. Reported
  as a measurement with no prediction attached; a third distinct split would be further evidence for #530's
  noise-floor reading, and equality would be evidence that the flips are host-class-specific rather than
  universal. It is **not** a gate: under #405 the counts are diagnostics and identity is the fingerprint.
- **Q4 — the NF4 reference travels.** Arms 1–2 land within 0.01 ppl of bo6c's box (wikitext 6.41984, c4val1
  16.49703; P37's box agreed to 0.0002). *Refuted by:* a larger drift, which is reported and does not void the
  gate — the gate is a within-box delta — but does say the instrument moved.

## How the registered gate actually decides, read off the code before the run

`experts4bit_qlora.k8_gate.verdict` is the rule, applied by the library so every lane applies the same one, and
it has a clause worth stating before the data rather than discovering in the verdict. For a **calibrated** pack:

- every text must be within `delta <= +0.05`; and
- **if any text improves, every text must improve**, on at least two texts, one outside the calibration domain.

So a pack whose deltas are **mixed in sign — say wikitext −0.02 and c4val1 +0.03 — FAILS, although both are
comfortably inside the budget.** `tests/test_k8_gate.py::test_calibrated_one_sided_needs_corroboration` pins
exactly that. The reasoning is sound (an improvement that moves with the calibration text is fitting it), and
this lane does not touch it — but it means "the gate failed" has two quite different meanings, and this lane
will say which one it read: *outside the budget* or *inside the budget, mixed in sign*. They are not the same
result and the second is not a quality failure.

Worth noting beside it: the register's own prose for the bo6c row describes the rule as "a calibrated pack
passes when dppl <= +0.05 on every text, an improvement is claimable only with the same sign on >= 2 texts" —
which reads as though the same-sign clause governs the *claim* and not the *pass*. The code makes it govern the
pass. That divergence is filed, not resolved here; this lane quotes the code, because the code is what runs.

## Decision rule, registered in advance

- **Arm 4/5 passes both texts, the fetched artifact re-verifies against its manifest, and the bytes land on the
  NAS** → this pack is the licence basis. A new K8 verdict claim row carrying `pack_fingerprint` is added; the
  fingerprint-less bo6c rows are marked `superseded_by` it and it `supersedes` them; `docs/STATUS.md`'s open
  bullet is rewritten to say where the bytes are; #405 narrows to the attention residual.
- **Arm 4/5 fails either text** → there is no licensed artifact and the register does not change except to gain
  a measured row recording the failure. #405 stays open with a third data point, and the c4val1 pattern gets its
  own issue. **No second box, no re-roll, no knob moved to recover a pass** — `min_rows`, damping, the K8 budget
  and the calibration corpus are outside this lane by construction.
- **`FP2 != FP1`** → a bigger finding than the gate: the recipe is not byte-deterministic even on one box under
  this cut, which would mean no build can ever be re-derived and the artifact is the *only* possible licence
  basis. Its own issue, and the gate verdict is reported beside it either way.
- **The bytes do not reach the NAS** → the gate result is an observation and **cannot** become a licence,
  because a licence whose bytes were not retained is precisely the state #405 describes. Said plainly, not
  glossed.

## What this lane cannot say

Nothing about throughput — no timed arm runs, no step time is quoted, no ratio against any framework. Nothing
about **cross-box** reproduction: it builds and gates on one box and does not re-test #405's cross-host question
(what it removes is the *need* for that question on the licensed path, by shipping bytes instead of a recipe).
Nothing about the calibrated attention component's own reproducibility. Nothing about any model but Qwen3-30B-A3B.
It does not compare against bo6c's pack, which does not exist.

## Budget and STOP rules

- **One box, RTX 5090 (Vast verified/secure), ≤ 5 h wallclock, estimate ≤ $3.30 at $0.66/h; lane ceiling $6,
  hard stop $8** (the owner's figure). Expected shape, from P37's and P39's timings: install ~8 min, bf16 fetch
  ~15, bake ~5, the two NF4 arms ~8, `build1` + dump ~60, the four candidate arms ~45, `build2` ~55, reduce ~2
  — about 3.2 h, with the artifact fetch overlapped.
- **The path proof.** The standing rule wants a cheap proving run before any guard over an hour. `p54-fqkv-1`
  ran this path **today** (2026-09-21, instance 51951835, same launcher head `8542e8d`, same provider class and
  image, staged by the same drive skeleton, reached `TP_DONE` and fetched its receipts), so the path is proven
  on hardware hours old. What differs is the e4b head, because this lane adds files — and that difference is
  covered where it is cheap: `tests/test_p55x_staged_pin.py` in CI, and a controller-side
  `P55X_DRIVE_DRYRUN=1` run of the driver before launch. This is a deliberate, stated deviation, not an
  oversight.
- **STOP-0 (K0, before any fetch)** — a GPU that is not a 5090, free disk on `/root` under
  `P55X_MIN_DISK_GB` (200; the working set is ~110 GB and a Vast instance's container overlay is not the
  machine disk it was ordered on), or an **upload probe under `P55X_MIN_UP_MBS` (8 MB/s)**: refuse with rc 13
  before anything is fetched. The upload floor is this lane's own, and it exists because the lane's product is
  15.2 GiB of bytes that have to leave the box: consumer-hosted Vast boxes have been measured at 0.2–0.5 MB/s
  in the push direction, at which rate the artifact would need 11 hours. 8 MB/s is a disaster detector — 32 min
  for the artifact — not a throughput estimate, and the probe's own reading is recorded whatever it says.
- **STOP-1** — arms 1–2 outside 0.01 ppl of bo6c's box: recorded, the lane continues, within-box deltas only.
- **STOP-2** — an arm that cannot finish 10 min before the launcher's deadline is skipped and recorded
  host-limited, never shortened.
- **STOP-3** — `build1` or `build2` producing no calibration chunk within `P55X_FIRST_CHUNK_S` (1500 s; a good
  host takes ~520 s) is killed as host-limited, bounding a latency-bound host to about $0.30 instead of the
  whole rental (p39-box1b-4 burned 90 min on exactly this).
- **STOP-4** — the artifact fetch incomplete when the lane ends: the run is not a success, the bytes are
  reported unpublished, and the gate verdict is an observation.
- **STOP-5** — no second box on a disappointing result.

## Receipts

Fetched to `receipts/experts4bit-qlora/<date>/p55x-packlic-1/` in the private record:
`logs/run_<arm>.log`, `qwen3_ppl_<arm>_<src>.json` (six K8 receipts), `gate_verdict.json`,
`determinism.json`, `artifact1/manifest.json`, `artifact1/payloads/{identity,assignment}.json`,
`summary.txt`, `forensics.txt`, `versions.txt`, `k0.json`. **The 15.2 GiB of payload bytes do not go in the
receipts tree** — they go to the NAS (see below), and the receipts carry the fingerprint that names them.
`RESULTS-p55x.md` is generated by `p55x_reduce.py` from those files and nothing else.

**Where a reader can get the bytes.** The artifact is ~15.2 GiB, and the receipts tree (git) and this
repository are both the wrong home for it. `p55x_publish.sh` places it on an **archive host named by the
operator** — `P55X_NAS_HOST` is a required parameter with no default, because a default in a public tree would
be either someone's machine name or a wrong guess — and verifies it twice: the library's `verify_artifact` on
the local copy before the transfer, and a re-hash plus an independent recomputation of the root fingerprint on
the archive host after it. Two implementations agreeing on one artifact is the check; either alone is not. For
this run the operator's copy is on the house NAS, and the path is recorded in the private receipts rather than
here.

What goes in public is the **identity**: the register and `docs/STATUS.md` carry the `pack_fingerprint`, and
the receipts carry the manifest with the hashed identity and assignment payloads. A reader can therefore check
any copy they are given, and a loader pinned to the fingerprint refuses anything else. The bytes are available
on request. **This is a private holding stated as one, not dressed up as a publication** — and it is a weaker
thing than publishing them. The CSO note on #405 proposed a dedicated Hugging Face artifact repository pinned
by commit, which would let a reader fetch as well as check; that remains the better answer and is not this
lane's to take, because pushing 15.2 GiB of model-derived weights to a public host is the owner's decision and
not a step inside a measurement lane.

---

## Amendment 1 (2026-09-21, after `p55x-prove-1`, before any registered data)

The proving run happened, and it did not go the way the section above assumed. Recorded here because a
pre-registration that quietly acquires a better story after the fact is not a pre-registration.

**What the section above said.** That `p54-fqkv-1` had proven the path that morning, so a separate proving
run "adds nothing". **That was wrong when written and I did not check it.** Two launcher commits landed
*after* p54 ran — adertha#125 (the pre-flight now measures free disk with a real `df -Pk` on the box, and
probes the **Hugging Face CDN** specifically at a 20 MB/s floor) and #124 (host-limited refusals name the
machine). Both touch the rental path. "P54 proved it" had stopped being true of the code that would run,
and the standing rule wanted a proving run on its own terms anyway.

**What the proving run proved.** The launcher path works on hardware under the new code: pre-flight
recorded `vast_free_disk_gb 320`, `vast_hf_cdn_mb_s 99.3`, `vast_bandwidth_mb_s 180.8`, 110 GB RAM; the
box was destroyed on completion with `instance_absent: true` and an empty list after; **$0.0272**. The
320 GB of *actual* overlay disk also clears this lane's 200 GB floor with room, on a box of the class the
registered run will draw.

**What it found, which is the point of doing it.** Two defects, both mine, both on the controller side,
both invisible to `bash -n` and to every test that existed:

1. **The controller is macOS, whose `rsync` is openrsync** ("rsync version 2.6.9 compatible"), and it
   rejects `--no-compress` with a usage error. The probe's transfer never ran, and the lane reported
   **`upload 0.00 MB/s`** on a box with 99.3 MB/s of measured CDN egress. Under STOP-0 as written, that
   reading would have **refused a perfectly good box** — a floor that can only say no.
2. The Vast image prints a two-line login banner ahead of the command's output, so values read by line
   number came back as `"Welcome to vast.ai..."` and `0`. Same family as the ssh banner fused to a curl
   HTTP status.

And a third, in the proving script itself: it ended in `exit 0` regardless, so the receipt recorded
**OK / pass** over a transfer that never happened. "Does not refuse on a low measurement" and "reports
success having measured nothing" are different things and only the first was intended.

**One condition this lane had left unstated, caught by the gnf4 #319 session on 2026-09-21.** These arms
pass no `grouped-nf4-gemm` compute-mode kwarg, so the mode comes from the capability-conditional default,
and on sm_120 that default resolves to **fp8**, not f32 (`gnf4#319`: the suite's own f32 arms were
resolving to fp8 while carrying an f32 tolerance; `compute_counts()` on a 5090 read `{'f32': 0, 'fp8': 4}`).
**The gate is unaffected** — the NF4 reference arm and every candidate arm go through the identical harness
invocation, so all of them take the same default and the delta is a within-box comparison either way. What
inherits the mode is the **absolute** perplexities, and the conditions line will say "gnf4 compute mode at
its capability-conditional default (fp8 on sm_120)" rather than implying f32. Reading the tally per arm
would mean instrumenting `step_decomp.py`, which is P39's pinned shared harness and outside this lane.

**Changes.** Values are read by `KEY=` marker, never by line number; no rsync invocation in this lane uses
a flag the controller's openrsync rejects (`tests/test_p55x_controller_portability.py` now asserts both in
CI); and the proving script exits non-zero when it obtained no measurement. **No STOP rule, floor, band,
arm or prediction in the sections above is changed** — the 8 MB/s floor and the 200 GB floor stand exactly
as registered. The proving run is re-run after these fixes, and the registered run launches only once a
real upload number has been read.

---

## Amendment 2 (2026-09-21, after `p55x-prove-2`, before any registered data)

**The upload probe works and reads 6.40 MB/s, which fails this lane's own 8 MB/s floor.** Written before the
registered run, and written plainly because it is a change made *after* seeing a measurement, in the direction
that lets my run proceed — the shape of an amendment that deserves suspicion.

**The measurement.** `p55x-prove-2` (RTX 5090, $0.0143, torn down with proof): rsync rc 0, the full
268,435,456 bytes, **6.40 MB/s**, 320 GB free, and the marker parsing returning real values through the Vast
banner. At that rate the 15.2 GiB pack takes **~41 minutes**, which is comfortable inside a 5 h guard and
overlaps the gate arms. The house WAN is 2.5 Gbps, so this is the **box's egress**, not the controller's
downlink.

**Why 8 was wrong, on its own terms.** I justified it as "a disaster detector — 32 min for the artifact". But
32 minutes is not a constraint anything has to satisfy; it was a number that sounded comfortable, and it put
the floor *inside* the acceptable region instead of at the boundary. The documented disaster is the
consumer-host regime of **0.2–0.5 MB/s**, where this artifact needs 11 hours. A floor at 8 rejects a box that
does the job in 41 minutes — and on the evidence of the one 5090 measured tonight, it may reject most boxes,
which is the "a floor no host measured today could pass" failure by another route.

**Re-derived from the constraint that actually binds** — the background transfer must not threaten the guard:

| probe reading | 15.2 GiB takes | verdict |
|---|---|---|
| 0.5 MB/s | 8.9 h | disaster (documented consumer-host regime) |
| 1 MB/s | 4.4 h | threatens a 5 h guard |
| 2 MB/s | 2.2 h | survivable |
| **3 MB/s** | **1.4 h** | **the floor**, against ~3.3 h of arms to overlap |
| 6.40 MB/s | 41 min | what was measured |

**The floor is now 3 MB/s.** Two things make this a re-derivation rather than a fit to the sample. The
number comes from the transfer-versus-guard arithmetic above, not from the observation — and **6.40 clears 3,
4, 5 and 6 alike**, so nothing was tuned to admit it. Had I wanted to fit, 6.3 was available and is not what
this says.

**And the instrument is pessimistic, which cuts the same way.** A 256 MB probe is dominated by connection
ramp-up: a prior lane read **14.76 MB/s on the probe against 38.70 MB/s sustained over 23.6 GB**, 2.6× higher.
So the floor is applied to a reading that understates what the real pull will do. The lane therefore now
**records the sustained rate of the actual 15.2 GiB transfer** (`artifact_fetch.json`, printed beside the
probe in the results), because no lane has ever recorded that number and the next one should size a budget
from it rather than from a probe.

**What does not change.** The 200 GB disk floor, every arm, every prediction, the K8 budget, `min_rows`,
damping, and the decision rule. STOP-0 still refuses before any checkpoint is fetched; it refuses at a
defensible place now.
