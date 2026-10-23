# D0 results: a trained expert-LoRA delta survives repacking into NF4, INT4 and MXFP4 (OLMoE-1B-7B, one fixture)

**The read** (`d0-p224s3`, CPU, 2026-10-22; receipts under [`receipts/d0-p224s3/`](receipts/d0-p224s3/)). On one
OLMoE-1B-7B fixture trained 200 steps on Alpaca, the expert delta's held-out improvement L(Q) − L(REF) is
**0.3366 ± 0.0303** nats (`fixture-dq`) and **0.3287 ± 0.0297** (`fixture-nodq`). That is RESOLVED at about 11 SE.
**All 12 cells** (3 formats × 2 sources × 2 fixtures) are **RETAINED** by loss:
- R runs from 0.992 to 1.116, and every 90 % CI lower bound is at least 0.984.
- The merged-and-repacked model sits **−0.0036 to +0.0167 nats** from REF (gap_to_ref). That is at most about 5 % of the
  improvement.

R is measured against each cell's own repacked base, so read it beside gap_to_ref, never alone. An R above 1 means that
cell's base lost more to the repack and the merge recovered most of it; the gap to REF is the direct "how close" number.

**Prediction (1) MISSED.** The registration predicted that NF4 from the trained base Q would LOSE the delta, because Q sits on
NF4's grid. It did not: ρ̄ is 0.71, and by loss the cell is RETAINED in both fixtures. The registration's confidence note had
named this open quantity: ‖ΔW‖ against the code gap after 200 steps. Here ‖ΔW‖ / ‖Q‖ is 0.081 (dq) and 0.082 (nodq).

**Scope:** one fixture (OLMoE-1B-7B, 200 Alpaca steps, r = 8). No claim is made that this generalizes. A second fixture
(longer training or another family) is its own registered item, never a re-read of this one.

**Provenance:** the registration is [`PREREG-d0.md`](PREREG-d0.md) with its Amendment 1. It did not pin the tools' hashes,
and no copy of them was stored before the run. That the computation tools were unchanged since 2026-10-21 therefore rests on
the author's statement; the tools as run are in the receipt. One stated deviation (D-1, the tokenizer's recorded name in the
data step, before any pass) is described in the receipt's [README](receipts/d0-p224s3/README.md). Both summaries were
re-derived twice, independently, from the private receipt store, with zero differences. They also re-derive from the
public record here ([`REDERIVE.md`](REDERIVE.md)). Cost: USD 0.

## The numbers

Step 1 is per-expert retention: ρ̄ is the share of the delta carried in its own direction, c̄ its cosine, and f the
fraction of changed values. Step 2 is held-out loss on 64 unseen rows (7,642 scored tokens). R is
[L(T(S)) − L(T(S+ΔW))] / [L(Q) − L(REF)], with a paired 90 % bootstrap CI.

| cell (format from source) | fixture | ρ̄ | c̄ | f | R [90 % CI] | gap_to_ref (nats) | label |
|---|---|---|---|---|---|---|---|
| NF4 from Q | dq | 0.712 | 0.662 | 0.925 | 0.9968 [0.9860, 1.0086] | +0.0011 | RETAINED |
| | nodq | 0.717 | 0.665 | 0.925 | 1.0110 [0.9992, 1.0244] | −0.0036 | RETAINED |
| NF4 from W | dq | 0.995 | 0.584 | 0.935 | 1.0006 [0.9927, 1.0083] | +0.0007 | RETAINED |
| | nodq | 0.995 | 0.586 | 0.936 | 0.9917 [0.9840, 0.9992] | +0.0027 | RETAINED |
| INT4 from Q | dq | 0.949 | 0.573 | 0.909 | 0.9964 [0.9863, 1.0064] | +0.0125 | RETAINED |
| | nodq | 0.950 | 0.575 | 0.909 | 1.1162 [1.1035, 1.1298] | +0.0085 | RETAINED |
| INT4 from W | dq | 1.001 | 0.559 | 0.884 | 1.0664 [1.0530, 1.0798] | +0.0139 | RETAINED |
| | nodq | 1.001 | 0.561 | 0.884 | 1.0656 [1.0520, 1.0804] | +0.0117 | RETAINED |
| MXFP4 from Q | dq | 0.970 | 0.528 | 0.192 | 1.0252 [1.0124, 1.0375] | +0.0100 | RETAINED |
| | nodq | 0.970 | 0.531 | 0.193 | 0.9972 [0.9844, 1.0093] | +0.0150 | RETAINED |
| MXFP4 from W | dq | 0.990 | 0.534 | 0.208 | 0.9993 [0.9858, 1.0131] | +0.0167 | RETAINED |
| | nodq | 0.990 | 0.536 | 0.209 | 1.0192 [1.0064, 1.0320] | +0.0095 | RETAINED |

| loss | dq | nodq |
|---|---|---|
| BASE0 (no adapter) | 2.8671 | 2.8624 |
| Q (base + attention LoRA) | 1.4901 | 1.4826 |
| REF (Q + expert LoRA, unmerged fp32) | 1.1535 | 1.1540 |

- **Carried in expectation, noisy per element.** Off the NF4-from-Q cell, ρ̄ is 0.95-1.00 while c̄ is 0.53-0.59: the delta
  survives on average, under rounding noise of its own size. By loss the noise does not cost this fixture: every such cell is
  RETAINED. The registration asked for this case to be named in words, and it is.
- **ρ̄ weighted by routing** (each expert weighted by its held-out top-8 count) agrees with ρ̄ to within 0.004 in every cell.

## The predictions, graded

| prediction | read | grade |
|---|---|---|
| NF4 from Q: LOST, ρ̄ < 0.5 and f < 0.2 | ρ̄ 0.712 / 0.717, f 0.925; R about 1.00 | **MISSED** |
| NF4 from W: ρ̄ 0.8-1.2, c̄ < 0.6; its `fixture-nodq` base equals Q | ρ̄ 0.995, c̄ 0.58; the nodq base reproduces Q exactly (relative error 0) | MET |
| INT4 and MXFP4, either source: ρ̄ 0.8-1.2, c̄ < 0.6 | ρ̄ 0.949-1.001, c̄ 0.53-0.58 | MET |
| By loss, the off-grid cells PARTIAL or RETAINED; NF4 from Q LOST | off-grid cells all RETAINED; NF4 from Q RETAINED | MET for the off-grid cells; **MISSED** for NF4 from Q |
| `fixture-dq` and `fixture-nodq` give the same labels | the same 6 labels | MET |
| The expert delta's improvement is RESOLVED | about 11 SE in both | MET |

## What follows

The outcome is not one of the registered branches: they assume NF4 from Q is LOST, and here every cell is RETAINED. The
ruling on it:
- Path (a), a merged re-quantized arena, is viable on this fixture even from the trained NF4 base.
- The registered cross-format served read goes ahead, to price the repack on the served model. It is a separate
  registration, with the tools' hashes committed before its run.
- The arena code is its own PR.
- A second fixture is not a precondition for either.

## Checks

- **Plumbing gate:** a 2-layer truncation of this forward matches transformers' `OlmoeForCausalLM` to max |Δlogit| 5.72e-06.
- **Unseen, per fixture:** re-encoding rows `[0, examples_used)` reproduces each receipt's `token_stream_sha256`, and
  training read only rows below 51,938.
- **NF4 from Q reproduces Q** (the registration's expected equality):
  - `fixture-nodq`: 2,048 of 2,048 expert projections.
  - `fixture-dq`: 2,046 of 2,048. Layer 0's gate_up for experts 19 and 38 differ, and the base's relative error is
    4.7e-07. This is reported as a finding.
  - No arm reused L(Q): all 15 arms were computed for each fixture (960 rows of 64 × 15).
- **Wall time** on 2 CPUs: 6,571 s (dq) and 6,781 s (nodq), inside the 8 h cap.
