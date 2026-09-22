# P319 — does grouped-nf4-gemm#319 reproduce on sm_120, and what fixes it

**Lane:** P319 · **Work:** grouped-nf4-gemm#319 (with the coverage half of #324)
**Registered:** 2026-09-21 · **Box:** one RTX 5090, vast verified-secure
**Estimate:** 0.75 h × $0.66/h = **$0.495** · **Guard:** 0.75 h (under the
one-hour line, so no proving run is registered)

## The claim under test

gnf4#319 (2026-09-03) reports that on an RTX 5090 (sm_120, torch 2.8.0+cu128,
triton 3.4.0) `kernel/test_fp8_paged_attn.py` fails 10 of 35 cases on unmodified
`main`: every `split` and `packed` case — the **f32 compute modes** — misses the
2e-2 tolerance by up to 0.074 on ~5% of elements, while every `f8dot`/`pf8` case
passes. The issue's own suspects are triton 3.4's default `input_precision` for
an fp32 `tl.dot`, and a tolerance calibrated on an older toolchain.

## What is already known, and how (free, no rented compute)

Run on the lab's **RTX A2000 (sm_86)** — the class where the f32 path is the
*serving default*, `_compute_default` selecting fp8 only on sm_89+ — carrying the
**identical toolchain** named in the issue (torch 2.8.0+cu128, triton 3.4.0):

* the suite **passes**, 35/35 non-skipped, f32 modes included;
* the worst residual is **0.015625**, exactly one ULP of the bf16 output at that
  magnitude — the dot contributes nothing visible above output rounding;
* `input_precision="ieee"` drops the split error to **0.000000** and throughput
  to **4.5 GB/s from 77.9** (0.06×); `tf32x3` is likewise exact at **39.5 GB/s**
  (0.51×);
* in the compiled PTX: `tf32` → 32 `mma.sync`; `tf32x3` → 96 `mma.sync`;
  `ieee` → **zero** `mma.sync` and 1045 `fma.rn.f32`, a scalar fallback.

So the toolchain is **not** the discriminating variable, and #319's suggested
remedy is priced out of being a default. What is unverified is the architecture.

## Hypothesis

H1. The failure reproduces on sm_120 at the release commit `9206352` — i.e. it is
architecture-specific, not toolchain-specific, and not already fixed by anything
that landed between 2026-09-03 and v0.32.1.

H2. The mechanism is the score/`P·V` dot's precision: on sm_120 triton 3.4 lowers
the unqualified fp32 `tl.dot` to something materially less exact than the
`mma.sync` tf32 it picks on sm_86, and naming `tf32x3` restores agreement.

## Arms (one box, no head-to-head, no default changes on the box)

| arm | what runs |
|---|---|
| A | full suite at `9206352` (release, unmodified) |
| B | full suite at `cf021d1` (branch; default precision `tf32`) |
| C | accuracy × {tf32, tf32x3, ieee} × {split, packed} over six shapes; throughput × precision; the emitted PTX per precision |
| D | the f32 half of the suite under `GNF4_ATTN_F32_PRECISION` = tf32x3, then ieee |

Arm A is what makes B interpretable: the branch's default is tf32, which is what
the compiler already picked, so **A and B must agree**. If they do not, the
change is not the no-op it claims to be.

## Registered predictions

* **P1.** Arm A fails ≥ 1 f32 case on sm_120. *Refuted if A is green* — then #319
  is stale against v0.32.1 and closes as fixed-by-something-else, with the run as
  the evidence.
* **P2.** Arm A and arm B report the **same** pass/fail set. A difference falsifies
  the claim that naming `tf32` preserves behaviour.
* **P3.** If P1 holds, at least one of `tf32x3` / `ieee` turns the f32 half green
  in arm D. If neither does, the cause is **not** dot precision and H2 is refuted —
  the deliverable becomes a typed refusal on the affected architecture, never a
  widened tolerance.
* **P4.** Arm C's sm_120 PTX for `tf32` differs from sm_86's `mma.sync` tf32 in a
  way that explains the error gap. Soft prior; C is descriptive.

## Success / failure criteria

**Success:** arms A–D complete with logs and a receipt; P1 and P2 read either way;
if P1 holds, P3 decided.
**Failure:** a refusal row (GPU class, egress, disk), an arm without a log, the
handshake timing out, or the deadline.

## Stop rules

* **S1.** Arm A green ⇒ stop after arm B. The remaining arms answer a question
  about a failure that is not there; do not spend clock characterising it.
* **S2.** Any arm exceeding 15 min of wall clock ⇒ record and move on.
* **S3.** 80% of the guard consumed ⇒ fetch whatever exists and finish.

## What this lane will NOT do

Widen a tolerance. The tolerance is the property; a number chosen to make a red
suite green records nothing. If the f32 modes cannot meet it on sm_120 and no
precision fixes them, they refuse there with the numbers, and the register row
says so.
