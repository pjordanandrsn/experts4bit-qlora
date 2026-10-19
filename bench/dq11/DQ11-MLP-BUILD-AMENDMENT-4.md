# DQ11 MLP-adapter BUILD — amendment 4: a bar that can judge the candidate, and a null for the knob check (#1083)

Prospective. It is registered and reviewed before any new data, under `DQ11-MLP-BUILD-PREREG.md` and Amendments 1–3,
which stay as merged. The maintainer ruled on 2026-10-19 (RED), after re-deriving `dq11-mlp-phase1-a2000-1`
byte-identical:
- that read stands as `PHASE1_FAIL`;
- no paid box runs, and Phase 2 stays blocked;
- the lever is not closed;
- a new $0 read on the A2000 must first state (a)–(d) below.

That read's id is **`dq11-mlp-phase1-a2000-2`**. Nothing from `-1`, or from its unregistered diagnostic `-diag-1`, is
re-read into it.

**Words.** "Today's path" is PEFT's own unfused path (`peft_read`), which earlier amendments called the truth. "The
reference" is (b)'s FP64 computation. Every path below reads the same captured `x` and `gy` at each checked layer
(Amendment 1).

## Why

- **Self-inclusion.** Redraw R2 (`(result.float() + delta).to(bf16)`, an FP32 add then one cast) is the fused
  epilogue's own arithmetic. So the floors on the outputs and the A/B gradients partly contained the candidate. Fused/floor
  read 0.994–1.004 on both metrics, both sides of 1, and pass or fail was decided in the 4th–5th digit.
- **dx.** dx was the one tensor genuinely outside its floor: relative RMS 1.067 / 1.049 / 1.100 at layers 0 / 15 / 31.
  The fused path sums gate's and up's LoRA terms in FP32 before one cast. That may be MORE accurate, and a floor of
  bf16 orderings cannot tell more accurate from wrong.
- **No null.** KNOB_OFF compared two processes bit for bit, but two identical runs of today's path were never compared
  first.

## (a) The floor leaves out its own candidate

- **By arithmetic:** R2 equals the fused epilogue, so every draw containing R2 leaves the floor. That removes 16 of the
  31 draws. The floor is the worst of the remaining 15, the non-empty combinations of R1, R3, R4 and R5.
- **Mechanically, before any floor is read:**
  - At each checked layer, each single redraw R1–R5 is read, and its distance to the fused read is compared with its
    distance to today's path on the three projection outputs.
  - A redraw **lands on the candidate** if, on any projection output at any checked layer, it is strictly closer to the
    fused read than to today's path on both max-abs and relative RMS.
  - Every redraw that lands leaves the floor too, with every draw containing it. The record names each one.
  - R2 is excluded by its arithmetic whether or not it lands. The record says which.
- **The post-(a) set** is what remains. It is also the set of unfused draws that (b) uses.
- **If no draw remains** (R1, R3, R4 and R5 all land), (a) reads `NO_FLOOR`. That is recorded and decides nothing.

(a) is recorded for every tensor: the fused path's deviation from today's path, against the post-(a) floor, on both
metrics, as before.

## (b) An accuracy judge that shares nothing with the candidate

**The reference.** For each checked module, a FP64 computation of the same function:
- **base weights:** each NF4 base's weights dequantized to the exact bf16 values the base layer uses
  (`bitsandbytes.functional.dequantize_4bit`), then upcast to FP64;
- **adapters and inputs:** the A and B matrices, `x` and `gy`, upcast;
- **computation:** `act_fn` in FP64, every output and gradient by autograd in FP64, with no intermediate rounding.

**The error.** For each checked tensor `T` and path P, `err_P(T)` is the relative RMS of P's tensor against the reference:
‖P(T) − ref(T)‖ / ‖ref(T)‖, in FP64. The max-abs error is recorded too, but it does not decide. It is an extreme value
and too noisy across orderings to compare by maximum. That gives K = 3 layers × 11 tensors = 33 deciding comparisons.

**The unfused band.** U = today's path plus the post-(a) draws, so up to 16 paths. `band(T) = max over U of err_U(T)`.

**The bar.** The fused path passes a tensor iff `err_fused(T) ≤ (1 + m) · band(T)`, with **m = 0.05**. **dx is read on
this bar.**

**The false-reject control.** This calibration runs on the read itself, before the fused path is judged:
- **The pseudo-candidates:** each of the 16 paths in U takes a turn as a neutral pseudo-candidate. It is judged against
  the band of the other 15, with the same margin, on all 33 comparisons.
- **If ANY neutral pseudo-candidate fails,** the bar cannot judge this read. The read is **`INDETERMINATE`**, never
  `FAIL`, and the record gives every failing pseudo-candidate and comparison.
- **The registered false-reject rate.** Suppose the calibration passes and the fused path is neutral, that is,
  exchangeable with the 16. Then it fails only if it alone exceeds the other paths by more than the margin. Among 17
  exchangeable paths that has probability at most 1/17 ≈ 5.9%, and the margin makes it smaller.
- **Fewer than two paths in U** (only today's path plus at most one draw) leaves nothing to calibrate against. The read
  is `INDETERMINATE`.

**What a PASS excludes (Amendment 1, restated on (b)).** For each tensor, the record gives
`e_min = ((1 + m) · band(T) · ‖ref(T)‖) / adapter-term norm`. Wherever `e_min` > 1/r = 1/16, the outcome names the tensor
`RANK1_UNRESOLVED`.

**Which decides:**
- **(b) decides the per-module outcome**, after its calibration.
- **If they disagree:**
  - (a) and (b) both pass: per-module PASS.
  - (b) passes and (a) fails or reads `NO_FLOOR`: per-module PASS, and the outcome names every such tensor as
    `ACCURACY_OVER_REORDER: <layer:tensor ...>`. The fused path differs from today's by more than non-self reorder
    noise, at no worse accuracy than the neutral unfused orderings. It is never a silent PASS.
  - (b) fails on any tensor, with the calibration passed: per-module FAIL, whatever (a) says.
  - The calibration fails: `INDETERMINATE`, whatever (a) and the fused comparison say.

## (c) KNOB_OFF's null comes first

Each step part is one update of the shipped loop in its own process. Off-vs-today is always judged inside one
environment, against that environment's own null. The two environments are never mixed.

1. **The read environment** (the registered L environment):
   - three `step-today` processes, giving three null pairs, and one `step-off`;
   - if the three `step-today` processes agree bit for bit (every adapter slot's sha256, and the loss), `step-off` must
     equal them bit for bit.
2. **If they do not agree,** the check moves to DQ11's registered deterministic configuration,
   `dq11_proof_policy.CONFIGURATION_C`, under `observer_policy()`:
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before CUDA initializes;
   - deterministic algorithms, warn-only, with every warned op recorded;
   - the math SDPA backend.

   Under C: three `step-today-c` processes and one `step-off-c`. If the three agree bit for bit, `step-off-c` must
   equal them bit for bit. These parts run only when step 1's null fails. The plan is fixed here, before data.
3. **If C's three `step-today-c` processes do not agree either,** the check is an aggregate in C, against C's null.
   - **The measure:** for two processes P and Q, D(P, Q) is taken over all 224 B slots concatenated (the only tensors
     one update from B = 0 moves):
     - the relative RMS ‖B_P − B_Q‖ / ‖B_Q‖;
     - the max-abs |B_P − B_Q|.
   - **The null spread** is the larger of D over the three today-vs-today pairs, per metric.
   - **The check passes iff** max over the three of D(`step-off-c`, `step-today-c`_i) ≤ 2 × the null spread, on both
     metrics. That is a margin of 1.0, and the outcome is named `KNOB_OFF_BY_SPREAD`.
   - **The margin's false-reject rate.** A knob that does nothing makes `step-off-c` exchangeable with the three
     `step-today-c`. Its three pair distances then come from the same distribution as the null's three. With the null's
     own pairs within 2× of each other, a doubled maximum is exceeded with probability well under 5%.
   - **If the null's three pairs differ from each other by more than 2× on either metric,** the spread cannot set the
     margin. KNOB_OFF is `INDETERMINATE`.

A cross-process bitwise comparison is never made without a null that passed, and never across environments.

## (d) One independent review before data

One reviewer who is not the author: tc-campaign or docs-upkeep. Then the maintainer's final pass, then the A2000 under a
`gpu:a2000` bus claim at $0.

## The outcome of `-2`

- **`PHASE1_PASS`** needs all of these:
  - the per-module PASS of (b), with `ACCURACY_OVER_REORDER` / `RANK1_UNRESOLVED` named where they apply;
  - KNOB_OFF held by 1, 2 or 3;
  - every registered part below.
- **`INDETERMINATE`** (from (b)'s calibration or (c) 3) means no paid box runs. It is not a FAIL, and the maintainer
  rules on what comes next.
- **Any other failure** is `PHASE1_FAIL`, with each failure named.

## Unchanged

- The registered parts all run again in `-2`, so the read stands on its own:
  - the stage check;
  - engagement (96, nothing refused, fallback 0);
  - 0 PEFT MLP LoRA forwards, with q/k/v/o unchanged;
  - streaming counters and recompute call for call;
  - K8 within 0.05 PPL at both points on both texts;
  - the reported two-update state.
- The adapter state, σ ladder and shares (Amendment 1), the counters (Amendment 2), and the version of record and wheel
  identity (Amendment 3).

## Prediction (registered; graded, never ruling)

- **(a)** fails on the projection outputs: with R2 out, the epilogue's single-rounding change is outside every remaining
  reorder.
- **(b)'s calibration** passes. Relative RMS errors of neutral orderings agree to well within 5%.
- **(b)** passes every tensor, dx included: one FP32 sum and one cast are no less accurate than two bf16 roundings. The
  outcome is `PHASE1_PASS, ACCURACY_OVER_REORDER: …` on the outputs.
- **KNOB_OFF:** the read environment's null fails; C's three `step-today-c` agree bit for bit; and `step-off-c` equals
  them.
- **R2 lands on the candidate,** and no other redraw does.
