# P141 — results: **LICENSED**. K33's bandwidth GEMV decodes Mixtral-8x7B one request 2.41× as fast on an RTX 5090 (2026-10-18)

Registration: `bench/p141/PREREG-p141.md` (#1754, merged `d2bf9f64`). The registered consequence is grouped-nf4-gemm#556:
Mixtral's two expert shapes join `GNF4_GEMV_BW=auto` on ≥ 160-SM parts, at the plans this read ran.

Code under test:
- e4b 0.55.0 at `d2bf9f64`;
- grouped-nf4-gemm 0.46.1 at `70bc03cb` (K33's kernel and contract as 0.46.0 shipped them);
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2;
- `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302`: the default `serve_paged` server at 16 slots × 768 tokens,
  graphs on, Mixtral's folds on, the NF4 arena baked on the box.

**Verdict by `p141_reduce.py`: LICENSED.** No VOID condition fired:
- phase 1's record re-derives to its own choice, and every B1 record ran its plan string;
- the commits, revision and server size are the registered ones;
- every bucket was captured;
- engagement was exact: B0 dispatched only the scalar GEMV (and its split-K form), and B1 only `bw_prmt32`, with
  `bw_splitk` beside it under down's split-K plan;
- the quality integrity held, and the scale mutant fails the bar.

## Phase 1: the plans (reported, never ruled)

All twelve of K33's plans met K33's numerics contract at both shapes. The lowest median over 40 rounds chose:

| shape | selected plan | per-call, 32 layers (ms) | the scalar GEMV (ms) | copy floor (ms) |
|---|---|---:|---:|---:|
| gate_up (28672, 4096) | (16, 1024, 4, 1) | 3.086 | 10.309 | 2.821 |
| down (4096, 14336) | (16, 512, 4, 2) | 1.950 | 6.105 | 1.410 |

## Phase 2: the reading (`p141-5090-2`)

**Host:** one RTX 5090 (sm_120, 170 SMs, driver 595.71.05) on an Intel Xeon Gold 6240 (36 CPUs), Vast machine 40549.
**Cost:** $0.999. Teardown proven at 17:13:24Z.

| arm | W1 ms / step | W16 ms / step |
|---|---:|---:|
| B0a (scalar GEMV) | 19.535 | 53.161 |
| B1a (K33) | 8.114 | 53.354 |
| B1b (K33) | 8.126 | 53.306 |
| B0b (scalar GEMV) | 19.624 | 53.279 |

- **g1 = min(2.4075, 2.4148) = 2.4075**, against the bar of 1.03.
- **g16 = min(0.9964, 0.9995) = 0.9964**: no regression (the line, fixed before data, is 0.99).
- **Self-pairs:** 0.9955 and 0.9985 at W1 (band [0.985, 1.015]); 0.9978 and 1.0009 at W16 (band [0.96, 1.04]).
- **Quality** (P110's bar, 24 windows per text):

  | text | bias_ON | bar | spread_ON | bar | K8 Δppl |
  |---|---:|---:|---:|---:|---:|
  | wikitext | +0.00228 | 0.01165 | 0.00826 | 0.01244 | +0.0103 (≤ 0.05) |
  | c4val1 | −0.00096 | 0.01038 | 0.00565 | 0.01335 | −0.0067 (not gated) |

- **Reported:** B1's tokens equal B0's on the W1 row at both lengths, and on 15 of 16 W16 rows at 160 tokens.

## Predictions

| # | prediction | outcome |
|---|---|---|
| Q1 | W1 fits at 16 x 768; no VOID | **HIT** |
| Q2 | 12 of 12 plans meet the contract; engagement exact; B1 ≠ B0 on some W1 rows; self-pairs bitwise | **HIT** except B1 ≠ B0: the W1 tokens were identical |
| Q3 | g1 ∈ [1.25, 2.00] | **FALSIFIED high**: 2.4075 |
| Q4 | gW ∈ [0.99, 1.03] | **HIT** (0.9964) |
| Q5 | quality within the bar | **HIT** |
| Q6 | phase 1, reported: per-call medians 0.3–0.5× the incumbent's | **HIT** (0.30, 0.32) |

Q3's band leaned on P136's class census, which put 15.27 ms in the expert GEMVs. The served saving was 11.4 ms a step,
nearer the scalar-to-K33 difference the phase-1 timings report (about 11.4 ms over both shapes) than the band's lower
transfer allowed.

## Attempts

- `p141-5090-1` ($0.007): NOT_RUN. The instance sat in `loading` for 600 s and the launcher's pre-flight destroyed it;
  nothing ran (`receipts/p141-5090-1/`). A host fault, not a strike.
- `p141-5090-2` ($0.999): the reading (`receipts/p141-5090-2/`).

Lane total: **$1.006**.

## Consequence

LICENSED, as registered: grouped-nf4-gemm#556 adds (28672, 4096) and (4096, 14336) to `_BW_SHAPES`, with (16, 1024, 4, 1)
and (16, 512, 4, 2) in `_BW_PLANS`. Until a release carries it, `GNF4_GEMV_BW=1` with
`GNF4_GEMV_BW_PLAN="28672,4096=16,1024,4,1;4096,14336=16,512,4,2"` reproduces the read.

## What this cannot say

Other cards, prefill, the int4 route, and Mixtral-8x22B (its shapes differ).

## Evidence

- `receipts/p141-5090-2/`:
  - `select.json` (phase 1);
  - the arm, fit and quality records;
  - `verdict_mixtral.json`;
  - `summary.txt`, the logs and `teardown-proof.json`;
  - `SHA256SUMS` covers each directory.
- The reviewers and the maintainer each re-derived the verdict from the receipt store (`279938dd`). One local
  re-reduction was byte-identical; the others differ only at the last ulp of unruled floats.
- Claim: `e4b.serve.p141.gemv-bw.mixtral.5090.2026-10-18` (`docs/claims.json`).
