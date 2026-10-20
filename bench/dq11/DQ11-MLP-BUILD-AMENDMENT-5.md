# DQ11 MLP-adapter BUILD — amendment 5: the Phase 2 kit (#1083)

**Status: DRAFT, for the maintainer's ruling (RED), before any Phase 2 code or data.** Once ruled, it is registered
after one review by a seat that does not consume it and the maintainer's final pass. `DQ11-MLP-BUILD-PREREG.md` and
Amendments 1–4a stay as merged; where this file differs, it governs Phase 2 only.

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
- **4a's floor applies to k_T, not |U|.** A (b) failure on T with k_T < 9 reads `INDETERMINATE: bar too weak at k = n`,
  never a FAIL. A PASS still reads.
- **Stated up front.** The $0 shakedown (section 6) measures k_T for every tensor with the families in section 3. If any
  tensor has k_T < 9 there, families are added until none does, and the final list is fixed here before the paid read.
  The bar is never gated on an instrument that cannot fail.
- **Engagement**, as in Phase 1: 96 adapters, nothing refused, fallback 0, and 0 PEFT MLP LoRA forwards in F1.

### 3. Redraw families that move every tensor (proposal; the shakedown fixes the list)

Phase 1's R1–R5 are binary, and only R1 moves the forward outputs. That is why k = 2 there.

Each new family changes only an accumulation order inside the unfused path. None may contain the candidate's own
arithmetic: Amendment 4's lesson from R2, the FP32 add then one cast, stands. The proposal gives the K-splits levels:
- **R1(c):** the forward A GEMM split along K into c ∈ {1, 2, 4, 8} chunks, summed left to right. c = 2 is Phase 1's R1.
- **R6(c), new:** the forward B GEMM split along the rank into c ∈ {1, 2, 4} chunks.
- **R9(c), new:** the first LoRA input-gradient GEMM (gy·B) split along its K (the out-features) into c ∈ {1, 2, 4}.
  Together with R4(c), this moves dh, where k = 2 today.
- **R3(c), R4(c):** Phase 1's token split and rank split, with c ∈ {1, 2, 4}.
- **R5:** unchanged.
- **R2:** stays excluded.
- **The draws:** the post-(a) draws are the level combinations, taken on a fixed seeded subset if the product is too
  large. The count and the seed are fixed after the shakedown.
- **What the outputs get:** up to c_A × c_B = 12 distinct orderings, and dh up to 9. If the shakedown shows two levels
  collapsing to the same bits, that level is dropped rather than counted twice.

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
- **REPORTED only.** It never enters GAIN, NO_GAIN or QUALITY.

### 6. A $0 shakedown on the A2000 first

Before any paid second, the whole Phase 2 driver runs on the A2000 at the rehearsal lock, under a `gpu:a2000` claim:
- the (b) part with the section-3 families, which yields every tensor's k_T;
- the eight arms at 8 updates;
- the two drift arms;
- the reducer, end to end.

Its records are a shakedown. They fix the family list and prove the driver, and they are never read as Phase 2 data.
No A2000 timing is reported.

### 7. How the read combines (proposal)

- **GAIN** needs everything the PREREG names (minimum H ≥ 0.05, quality) plus (b) at the science lock reading PASS on
  every tensor. Tensors with k_T < 9 are named `UNJUDGED_AT_K`.
- **A (b) FAIL** on any tensor with k_T ≥ 9 makes the read `ACCURACY_FAIL`, whatever H is.
- **(b) INDETERMINATE** from the calibration means the timing still reads and is reported, but it can only be
  `NO_GAIN (accuracy indeterminate)`. The maintainer rules what comes next.
- **NOISY** stays a VOID and a strike.

### 8. Launch, budget and guard (proposal)

- **New files only:** `dq11_mlp_phase2_run.py` (parts), `dq11_mlp_phase2_box.sh`, `dq11_mlp_phase2_drive.sh` and a launcher.
  - The launcher reuses the census launcher's generic pieces: host discovery and the continuity window, the pin plan, the
    journal and permit, and the transport.
  - It has its own constants: run ids `dq11-mlp-phase2-N`, the caps below, and the template.
  - So `science.sha256` and the census launcher's bytes are not resealed.
- **Box:** one RTX 5090, at the science lock, through the bootstrap and prepare already used for the census.
- **Time:** about 5 min for the (b) part, 8 × about 5 min for the timed arms, 2 × about 5 min for the drift arms, plus
  load. That is about 65–75 min, so the guard is **1.5 h**.
- **Cost:** the worst case is about $1.60: 1.5 h at $0.85/h plus the download, on the PREREG's own basis ($1.40 at
  1.25 h). That is inside the $15 standing rule, and the maintainer's approval covers it.

## Prediction (registered; graded, never ruling)

- **H:** unchanged, 0.02–0.05, and NO_GAIN is the likelier outcome.
- **(b) at the science lock:** PASS on every tensor with k_T ≥ 9. 0:out_down is RANK1_UNRESOLVED again.
- **k_T:** with the section-3 families, every tensor reaches k_T ≥ 9 on the A2000 shakedown, and dh is the last to get there.
- **Drift:** D1 against D0 is nonzero from update 1 in B, and stays below 1e-2 relative at update 40.

## Decisions for the maintainer

- **P2-D1:** does (b) run at the science lock on the 5090 (section 2)? Proposed: yes.
- **P2-D2:** the families: K-split levels R1(c), R6(c), R9(c), R3(c) and R4(c), with the list fixed by the $0 k count
  (section 3). Proposed: yes.
- **P2-D3:** how (b) combines with GAIN (section 7). Proposed: as written.
- **P2-D4:** the drift check as two untimed arms under C (section 5). Proposed: yes.
- **P2-D5:** the launch path, a new launcher built from the census launcher's generic parts (section 8). The alternative
  is a plain launch manifest under the standing rule.
- **P2-D6:** the guard of 1.5 h and the worst-case cost (section 8).
