# DQ11 MLP-adapter BUILD — amendment 5: the Phase 2 kit (#1083)

**Status: ruled by the maintainer on 2026-10-20 (RED), before any Phase 2 code or data.**
- It is registered on merge, after one review by a seat that does not consume it and the maintainer's final pass.
- No Phase 2 code is written before it merges.
- The $0 A2000 shakedown (section 6) runs after the merge, under a `gpu:a2000` claim.

`DQ11-MLP-BUILD-PREREG.md` and Amendments 1–4a stay as merged. Where this file differs from them, it governs Phase 2
only.

## Where Phase 1 stands

`dq11-mlp-phase1-a2000-2` read **`PHASE1_PASS`**. ACCURACY_OVER_REORDER names all 30 tensors other than dh, and
RANK1_UNRESOLVED names 0:out_down. The read was re-derived byte-identical and independently reviewed, and the maintainer
ruled on 2026-10-20 that it stands as the registered Phase 1 gate. It is a gate, not a science claim: it ran on the A2000
rehearsal lock.

**How to quote it from now on.** The review found that U holds duplicate paths. A redraw that does not move a tensor
gives a path identical to one already in U. Per tensor, the 16 paths in U have only k = 2–8 distinct errors; the forward
outputs at layers 0 and 15 have k = 2.
- So the evidence bound is 1/(k + 1) = 1/3 to 1/9, not the stated 1/17.
- Counted by distinct paths, Amendment 4a's floor means no (b) failure in that design could have ruled `PHASE1_FAIL`.
- A calibration over twin paths cannot fail: its worst pseudo-candidate ratio is exactly 1.
- None of this changes the PASS. The band is a maximum, so de-duplication leaves it unchanged, and a de-duplicated
  calibration also passes (top distinct error / second ≤ 1.000031).
- **The wording to quote:** on the outputs, ACCURACY_OVER_REORDER means "no worse than today and the R1/R5 family",
  which is 2 distinct orderings.

`-2` is not amended. The fixes below bind Phase 2.

## What Phase 2 adds to the registration

The registered Phase 2 is unchanged:
- arms F0 (knob off) and F1 (fused), ABBA twice, each a fresh process of 40 updates;
- T_arm = the median of updates 6..40;
- D4's minimum H over the four block pairs;
- NOISY self-pairs (a VOID and a strike);
- QUALITY by K8's 0.05 PPL band;
- GAIN / NO_GAIN;
- the ~9% ceiling and the registered prediction.

Everything below is added.

### 1. The comparison is on the accuracy bar, never bitwise

F1 differs from F0 by more than reorder noise (Phase 1's ACCURACY_OVER_REORDER), so no Phase 2 check compares F1 with
F0 bit for bit or against reorder equality. Correctness is judged only by (b)'s FP64 bar (section 2) and K8. (a) is
computed and reported, and it names ACCURACY_OVER_REORDER; it never rules.

### 2. (b) at the science lock, on distinct paths

The box's first measured part repeats Phase 1's (b) read on the RTX 5090, at the science lock (torch 2.12.1+cu130,
sm_120): the same three layers, captured `x` and `gy`, FP64 reference and adapter state. A GAIN would license this
stack, and its kernels and epilogue differ from the A2000's by architecture.

On that read:
- **Distinct paths.** For each tensor T, U_T is the set of distinct results among today's path and the post-(a) draws.
  Two paths are the same iff their result tensors are equal bit for bit (sha256 of the bytes). k_T = |U_T|.
- **De-duplicate before calibrating.** The leave-one-out calibration runs over U_T, never over U with twins.
- **The bound reported per tensor:** k_T and 1/(k_T + 1).
- **4a's floor applies to k_T, not |U|.** A tensor with k_T < 9 reads `UNJUDGED_AT_K: k = n`. Its calibration is
  not run, and it never reads FAIL. Section 7 says what that does to the read.
- **Two names, two causes.** The calibration runs only on tensors with k_T ≥ 9. A failed calibration there reads
  `CALIBRATION_INDETERMINATE`. So one tensor never carries both names, and below 9 `UNJUDGED_AT_K` is the only one.
- **"Judged"** means k_T ≥ 9 with a calibration that passed.
- **Stated up front.** The bar is never gated on an instrument that cannot fail.
  - The $0 shakedown (section 6) counts k_T for every tensor with the families in section 3.
  - If any tensor has k_T < 9 there, families are added until none does.
  - The final list is fixed in this file before the paid read.
- **Landing is judged per tensor.** Amendment 4 judged "lands on the candidate" on the projection outputs only. So a
  family that moves only gradients (R3, R4, R9) could never land, even if it reproduced the fused path's gradient
  arithmetic.
  - Here, a draw lands on T if, on T, it is closer to the fused path than to today's on both metrics.
  - U_T leaves out every combination that contains a family landing on T at any checked layer.
  - The `to_fused` and `to_today` distances are recorded for every draw and tensor (section 4).
- **Engagement**, as in Phase 1: 96 adapters, nothing refused, fallback 0, and 0 PEFT MLP LoRA forwards in F1. It is
  checked in the (b) part and in every F1 arm. Section 7 places it first.

### 3. Redraw families that move every tensor (the shakedown fixes the list)

Phase 1's R1–R5 are binary, and only R1 moves the forward outputs. That is why k = 2 there.

Each new family changes only an accumulation order inside the unfused path.

**The unfused path's dtypes,** which no family changes (PEFT 0.21.2's `Linear4bit` LoRA path, as Phase 1's reference
repeats it):
- The base output is bf16, from bitsandbytes.
- The LoRA input is cast to FP32. The A GEMM, the B GEMM and the scale run in FP32 (the rank compute stays FP32, as the
  PREREG says).
- The delta is cast to bf16 once and added to the base output in bf16.
- In the backward, the LoRA GEMMs and the A/B weight gradients run in FP32. Each projection's LoRA input gradient is
  cast to bf16 once at its leaf.
- dx's four bf16 terms (two base, two LoRA) are summed in bf16.

**The families:** each splits ONE of those ops into chunks, computed in that op's own dtype and combined in that same
dtype, left to right, before the cast the path already has:
- **R1(c):** the forward A GEMM, split along K into c ∈ {1, 2, 4, 8} chunks. The partials and their sum are FP32. c = 2
  is Phase 1's R1.
- **R6(c), new:** the forward B GEMM, split along the rank into c ∈ {1, 2, 4} chunks. FP32 partials and sum.
- **R9(c), new:** the first LoRA input-gradient GEMM (gy·B), split along its K (the out-features) into
  c ∈ {1, 2, 4, 8} chunks. FP32 partials and sum, before that projection's own cast at its leaf.
- **R3(c):** the A/B weight-gradient GEMMs, split along the tokens. c ∈ {1, 2, 4}; FP32 partials and sum.
- **R4(c):** the LoRA input-gradient GEMM (·A), split along the rank. c ∈ {1, 2, 4}; FP32 partials and sum, before the
  leaf's cast.
- **R5:** dx's four bf16 terms in the other order, in bf16. Unchanged.
- **R2:** stays excluded.

**The exclusion, stated against the fused kernel.** No family:
- adds the base and the delta in FP32 before one cast (R2, the fused epilogue);
- sums gate's and up's LoRA input-gradient terms in FP32 before one cast (the fused path's dx arithmetic, Amendment 4);
- moves, adds or removes any cast of the unfused path.

Within one FP32 op, re-associating FP32 partials changes no rounding point of the path. Every cast stays where PEFT puts
it, so no family has the fused path's single-rounding property.

**The draws: a stratified subset.** For every tensor, the draws contain the full level grid of two families that move
it, with every other family at c = 1. On top of that union, a seeded sample of the full product is added.
- The outputs: R1 × R6 (12).
- dh: R9 × R4 (12).
- dx: R9 × R4 × R5 (24).
- dA and dB: R1 × R3 (12).

The shakedown checks which families move which tensor. k_T is counted on the subset as run. The seeded sample's size
and seed are fixed after the shakedown.

**What the grids give:** up to 12 distinct orderings for the outputs and for dh, so one collapsed level still clears
9. If the shakedown shows two levels collapsing to the same bits, that level is dropped rather than counted twice.

**The shakedown predicts; the paid read decides.** k_T counted on the A2000 only predicts the 5090's k_T, because
sm_86 and sm_120 run different kernels and their collapse pattern can differ. The paid read counts k_T itself, and
section 2's rule (k_T < 9 reads `UNJUDGED_AT_K`, never a FAIL) governs there.

**Size.** The shakedown also times the (b) part with the full families. The seeded subset's size is then fixed so
that (b) stays within its budget, and the guard fit (section 8) is rerun with that number.

### 4. Receipt fields the review asked for

- **F2.** For each redraw and tensor, record the `to_fused` and `to_today` distances that `landed()` compares. Then
  "which redraws land" can be checked from the receipt.
- **F3.** Record `ref_norm` (‖ref(T)‖) for each tensor, so e_min can be recomputed exactly.
- **RANK1_UNRESOLVED** is named again with every tensor's e_min. Phase 1's 0:out_down (e_min 0.0664) is named in the
  prediction below.

### 5. A reported drift check on the two-update state (never ruling)

Phase 1's reported two-update read failed (a)'s rule on 0:dA_up. Phase 2 registers a REPORTING check on that tensor across
its own updates, so a drift shows up instead of surprising anyone:
- **Two untimed drift arms after the timed block:** D0 (F0) and D1 (F1), from the same initializer, batches and seeds as
  the timed arms.
- **They run under configuration C** (Amendment 4 (c)), which held bit for bit across processes in `-2`. So D1 against
  D0 is not mixed with cross-process noise, and C's speed never enters a timing.
- **What they record:**
  - layer 0 `up_proj`'s A and B after updates 1, 2, 4, 8, 16, 32 and 40;
  - ‖θ_D1(t) − θ_D0(t)‖ / ‖θ_D0(t)‖ for A and for B, plus the max |Δloss| per update.
- **Label:** "under C (deterministic settings), not the timed configuration".
- **C's settings** are set only in the drift processes' own environment and are never inherited. The timed arms refuse
  an inherited C setting, as Phase 1's read environment did.
- **REPORTED only.** It never enters GAIN, NO_GAIN or QUALITY.
- **A drift failure can't touch the read.**
  - The timed block's records are written, and fetched off the box, before D0 starts.
  - A drift arm that fails, times out or meets the guard reads `drift: not read (<reason>)`. It never VOIDs the read and
    never changes the box's outcome.
  - The reducer reads the timed records without the drift records.

### 6. A $0 shakedown on the A2000 first

Before any paid second, the whole Phase 2 driver runs on the A2000 at the rehearsal lock, under a `gpu:a2000` claim:
- the (b) part with the section-3 families, which yields every tensor's k_T;
- the eight arms at 8 updates;
- the two drift arms;
- the reducer, end to end.

Its records are a shakedown. They fix the family list, size the (b) part and prove the driver, and they are never
read as Phase 2 data. The (b) part's A2000 time is used only to size the subset and fit the guard. No A2000 timing is
reported as a result.

### 7. How the read combines

**The first rung that applies decides,** and rungs 1–5 are terminal. The outcome names every tensor that meets any of
rungs 3–5, whichever rung decides.
1. **Engagement fails:** fewer than 96 adapters engaged, anything refused, fallback > 0, or a PEFT MLP LoRA forward in
   an F1 arm or the (b) part. F1 isn't fused, so the read is a VOID (harness).
2. **NOISY** stays a VOID and a strike.
3. **A (b) FAIL** on any judged tensor (k_T ≥ 9, calibration passed) makes the read `ACCURACY_FAIL`, whatever H is.
4. **Any tensor CALIBRATION_INDETERMINATE:** the read is `NO_GAIN (accuracy indeterminate: <tensors>)`. The timing is
   still reported, and the maintainer rules what comes next.
5. **Any UNJUDGED_AT_K tensor:** the read is `NO_GAIN (accuracy unjudged: <tensors>)`. A license can't rest on a tensor
   the instrument couldn't fail.
6. **Otherwise:** GAIN iff minimum H ≥ 0.05 and quality passes, with every tensor judged and passing (b). Anything else is
   NO_GAIN, with each failed bar named.

(b) is a correctness read in its own part and doesn't depend on the timed arms, so its result is reported under every
outcome. When NOISY and an ACCURACY_FAIL both hold, the record names ACCURACY_FAIL beside the VOID.

### 8. Launch, budget and guard

- **Launch:** a plain launch manifest through the shared launcher, under the standing rule.
  - It is checked with `--validate`, pinned by inclusion, and the pin is posted before create.
  - A dedicated launcher would be more code to review and more ways to break, for no gain the shared one lacks. It is
    revisited only if the shakedown shows a concrete gap.
- **New files only:** `dq11_mlp_phase2_run.py` (the parts), `dq11_mlp_phase2_box.sh` and the manifest. `science.sha256`
  and the census launcher are untouched, so nothing is resealed.
- **Box:** one RTX 5090 at the science lock. Box-side, it reuses the census flow's bootstrap and prepare scripts by
  file.
- **Time:** about 5 min for the (b) part, 8 × about 5 min for the timed arms and 2 × about 5 min for the drift arms,
  plus load. That is about 65–75 min of parts.
- **The guard is 2 h.**
  - The static fit is (boot + the measured part times) × 1.25 ≤ the guard.
  - With a 5 min boot, 1.5 h fails it at the 75 min estimate: (5 + 75) × 1.25 = 100 > 90. 2 h fits: 100 ≤ 120.
  - The fit is checked by a test, and rerun with the times the shakedown measures once it sets (b)'s size.
  - If the shakedown's times don't fit 2 h, the read goes back to the maintainer before any paid second.
- **Cost:** the worst case is about $2.05, from 2 h at $0.85/h plus the download, on the PREREG's own basis ($1.40 at
  1.25 h). That is inside the $15 standing rule and the maintainer's approval, which covers up to 2 h.

## Prediction (registered; graded, never ruling)

- **H:** unchanged, 0.02–0.05, and NO_GAIN is the likelier outcome.
- **(b) at the science lock:** PASS on every tensor with k_T ≥ 9. 0:out_down is RANK1_UNRESOLVED again.
- **k_T:** with the section-3 families, every tensor reaches k_T ≥ 9 on the A2000 shakedown, and dh is the last to get there.
- **Drift:** D1 against D0 is nonzero from update 1 in B, and stays below 1e-2 relative at update 40.

## Rulings (maintainer, 2026-10-20, RED)

- **P2-D1, YES:** (b) runs at the science lock on the 5090. A GAIN would license that stack, and its kernels differ by
  architecture, so correctness is read where the license applies.
- **P2-D2, YES**, with one addition: the shakedown's k_T only predicts the 5090's. The paid read counts k_T itself, and
  the shakedown times the (b) part to size it (section 3). After review, R9 has an 8-chunk level, so dh has slack
  above 9, and the subset is stratified per tensor.
- **P2-D3, YES**, with two changes:
  - `UNJUDGED_AT_K` and `CALIBRATION_INDETERMINATE` are named apart;
  - GAIN needs every tensor judged and passing (section 7).
- **P2-D4, YES:** the drift arms D0/D1 run under C, labelled as such, and are REPORTED only.
- **P2-D5:** a plain launch manifest through the shared launcher, not a new launcher (section 8).
- **P2-D6, YES:** a guard of 1.5 h and a worst case of about $1.60, subject to the static fit, with up to 2 h inside the
  approval. 1.5 h fails that fit at this file's own 75 min estimate, so the registered guard is 2 h (section 8).
