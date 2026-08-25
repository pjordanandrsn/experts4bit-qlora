# RESULTS — F1 Stage B, arm B2: PASS (94.2 → 133.4 tok/s)

Measured 2026-08-25 under PREREG-f1-stageB + AMENDMENT-f1-stageB-b2 +
AMENDMENT-f1-stageB-b2-frame. Receipts in `receipts-f1/stageB-b2/`
(box 5: RTX 5090, driver 595.71, torch 2.13.0+cu130 / triton 3.7.1;
e4b `812b6e0`, gnf4 `4f685c0`, clean merged-main trees; instance
destroyed, vast verified zero).

```
F1 STAGE B VERDICT: PASS
  B2 (fused KV append over B1) gain 2.08 ms >= 1.0
  (step 7.50 ms, 133.4 tok/s)
```

| arm | step | tok/s |
|---|---|---|
| C0 (shipped B1 config) | 9.589 ms | 104.3 |
| C0 (repeat) | 9.575 ms | 104.4 |
| **C1 (+ fused KV append)** | **7.497 ms** | **133.4** |

## Gates, all discharged

| gate | result |
|---|---|
| bitwise kernel gate, on-box GPU | **13/13** (`f1c_gate0.log`) |
| token identity C1 vs C0 | exact — 127 tokens, 78 distinct, element-wise equal |
| A/A | 0.014 ms (bar 0.5) |
| capture neutrality | 127 steps both arms |
| recompiles in window | 0 |

## The frame refusal, and why the verdict is still prereg-clean

The first adjudication REFUSED: AMENDMENT-b2 had set ABSOLUTE bars
(PASS ≤ 9.6 ms) derived from box 4's 10.62 ms base, and box 5 runs the
identical shipped config at 9.57 ms (a 1.05 ms cross-box shift on
unchanged code; driver 580.119 → 595.71). The degenerate-frame refusal
fired exactly as designed. AMENDMENT-b2-frame re-expressed the SAME
registered quantity — "≥ 1.0 ms of the 1.58 ms block realized", which
is how the 9.6 number was justified in the first place — as a
within-box gain bar, and the same untouched receipts adjudicate PASS
with a 2× margin. No re-measurement, no bar moved toward the data:
the gain bar (1.0 ms) predates the arms in the b2 amendment's own
text, and the observed 2.08 ms clears it either way.

## Why the gain beat the census

The census priced the fp8 KV block at 1.58 ms; the arm returned 2.08.
The fused kernel also removed the device-side address arithmetic
(pos/blk/fill/row gathers) the census had classified under `other`,
and the freed launch slots compound under graph replay.

## Shipped

`E4B_FUSED_KV_APPEND` now defaults **ON** (this PR); the env is the
rollback. The eager append path remains intact behind it.

## Ladder

14.1 → 20.1 (collapse) → 65.8 (graph loop) → 74.3 (K1 configs) →
94.2 (compiled layer body) → **133.4 (fused KV append)**.
Roofline ~480; goal 425. The census's remaining visible blocks:
RoPE+RMSNorm already inside compile; next candidates are the nf4
wrapper's copy+sum (~0.3 ms) and the router/other tail — but the
program's registered arithmetic still says the step-cheapening lane
tops out near ~260 tok/s and 425 needs >1 token/step (speculative
decoding, acceptance-rate first).
