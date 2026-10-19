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
    distance to the truth (PEFT's path) on the three projection outputs.
  - A redraw **lands on the candidate** if, on any projection output at any checked layer, it is strictly closer to the
    fused read than to the truth on both max-abs and relative RMS.
  - Every redraw that lands leaves the floor too, with every draw containing it. The record names each one.
  - R2 is excluded by its arithmetic whether or not it lands. The record says which.
- The same post-(a) draw set is the "unfused draws" of (b).

## (b) An accuracy judge that shares nothing with the candidate

**The reference.** For each checked module, a **FP64 reference** of the same function, on the same captured `x` and `gy`:
- **base weights:** each NF4 base's weights dequantized to the exact bf16 values the base layer uses
  (`bitsandbytes.functional.dequantize_4bit`), then upcast to FP64;
- **adapters and inputs:** the A and B matrices, `x` and `gy`, upcast;
- **computation:** `act_fn` in FP64, every output and gradient by autograd in FP64, with no intermediate rounding.

**The errors.** For each checked tensor `T`, `err_P(T)` is the max-abs and the relative RMS of path P's tensor against
the reference's. P ranges over:
- the fused read;
- the truth;
- every draw in the post-(a) set.

**The bar (b).** The fused path passes a tensor iff, on both metrics, `err_fused(T)` ≤ the largest `err_U(T)` over the
unfused paths U (the truth and the post-(a) draws). That is, the fused path is no less accurate than the least accurate
neutral unfused ordering. **dx is read on this bar.**

**What a PASS excludes (Amendment 1, restated on (b)).** For each tensor, the record gives
`e_min = (that largest relative RMS × ‖T_ref‖) / adapter-term norm`. Wherever `e_min` > 1/r = 1/16, the outcome names
the tensor `RANK1_UNRESOLVED`.

**Which decides:**
- **(b) decides the per-module outcome.**
- **(a) is read and recorded** with the same machinery: the fused deviation from the truth against the post-(a) floor.
- **If they disagree:**
  - (a) and (b) both pass: per-module PASS.
  - (b) passes and (a) fails: per-module PASS, and the outcome names every such tensor as
    `ACCURACY_OVER_REORDER: <layer:tensor ...>`. The fused path differs from today's by more than non-self reorder noise,
    at no worse accuracy than any neutral unfused ordering. It is never a silent PASS.
  - (b) fails on any tensor: per-module FAIL, whatever (a) says.

## (c) KNOB_OFF's null comes first

1. **The null, in the registered L environment.** Two `step-today` processes (the engine never imported; one update).
   If their adapters (every slot's sha256) and their losses are equal bit for bit, the registered check holds:
   `step-off` against `step-today`, bit for bit, cross-process.
2. **If the null fails there,** the check moves to DQ11's registered deterministic configuration,
   `dq11_proof_policy.CONFIGURATION_C`, under `observer_policy()`:
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before CUDA initializes;
   - deterministic algorithms, warn-only, with every warned op recorded;
   - the math SDPA backend;
   - a bitwise comparison.

   The null is re-run there first: two `step-today` processes under C. If they are equal, the check is `step-off`
   under C against `step-today` under C, bit for bit.
3. **If the null fails under C too,** the check is the null's spread. For every adapter slot, the max-abs and relative
   RMS of `step-off` against `step-today` must be at or below the larger of the two `step-today`-vs-`step-today`
   spreads (the read environment's and C's). That outcome is named `KNOB_OFF_BY_SPREAD`. A cross-process bitwise
   comparison is never made without a null that passed.

Each step process is its own process, as registered. All the null and C parts run every time, so the plan is fixed
before data, and the reducer applies 1–3 in order.

## (d) One independent review before data

One reviewer who is not the author: tc-campaign or docs-upkeep. Then the maintainer's final pass, then the A2000 under a
`gpu:a2000` bus claim at $0.

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
- **(b)** passes every tensor, dx included: one FP32 sum and one cast are no less accurate than two bf16 roundings.
- **KNOB_OFF:** the null fails in the read environment and holds under C, so `step-off` equals `step-today` bit for
  bit under C.
- **R2 lands on the candidate,** and no other redraw does.
