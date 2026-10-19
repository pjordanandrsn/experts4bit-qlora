# P153 results: D2's served read, grouped-nf4-gemm K36's int8 lm_head on Qwen3-30B-A3B (e4b#1313)

**Status:** closed. **FLIP** on the one read. `E4B_INT8_HEAD_BW` stays opt-in until a separate default PR. That PR waits
on a grouped-nf4-gemm release carrying K36 being e4b's pin. Claim: `e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19`.

[P148](../p148/RESULTS-p148.md) read D2 GO in a microbench: an int8 copy of Qwen3-30B-A3B's lm_head (151936 x 2048,
per-row fp32 scale) halves the head's bytes. Projected onto P142's in-situ head time, P148's GEMV saves **0.189546 ms**
a step. [P150](../p150/RESULTS-p150.md) ruled the grid GO on quality at both of P125's gates. K36 shipped the kernel and
quantizer verbatim, and e4b#1820 added the knob: the head rounded in place to `bf16(q * s)`, with one-row calls through
`gemv_i8_bw`. [The registration](PREREG-p153.md) ran P151's served form with the head in place of D1, plus P131
Amendment 2's unscored warm-up process.

| run | what | cost | verdict |
|---|---|---|---|
| [`p153-5090-1`](receipts/p153-5090-1/README.md) | a warm-up process, eight arms x 5 reps and SANE | $0.445 | **FLIP**: g1 1.0520, g16 0.9918 |

Total $0.445 on one RTX 5090 launch, at the registered fallback tier: a full loop at the $0.80/h tier found no stable
offer. The public receipt copy is under [`receipts/`](receipts/), with a README and `SHA256SUMS`. The launcher's receipt
and ledger row are in the private receipt store. The maintainer re-derived `verdict_p153.json` byte for byte from the
store (the reducer at `585cf68b`, equal to the staged pin; self-test 52/52).

## The read (`p153-5090-1`)

- **The box:** one RTX 5090 at a 575 W power limit (the registered floor), e4b `585cf68b`, grouped-nf4-gemm `929a6906`.
  The premise read 28 passed, K36's compiled contract on the 5090 among them.
- **On shape:** the warm-up, all eight arms and both SANE phases completed.
  - S1 swapped one module, the `[151936, 2048]` head. It routed at one row and passed at every multi-row count (2 to 16,
    and the 512-row prefill), with no one-row fallback.
  - S0 swapped nothing. The dense attention GEMV was off in every arm, and the split cap read its `qwen3_moe` default.
- **Power and clocks:** every timed block read 575 W either side of it, at most 59 C, with SM clocks at least 2760 MHz
  in every arm.

| | S1a/S0a | S1b/S0b | S1c/S0c | S1d/S0d | min |
|---|---|---|---|---|---|
| W1 | 1.0530 | 1.0520 | 1.0522 | 1.0527 | **1.052019** |
| W16 | 0.9918 | 0.9970 | 0.9924 | 0.9929 | **0.991794** |

- **Noise:** W1 self-pairs 0.9993 to 1.0003; W16 0.9958 to 1.0010.
- **The warm-up process:** the box's first process loaded slow again: 38.0 s, against 20.6-21.8 s for every later arm.
  Its timed decode did not run slow: W1 at 0.9989 of S0a, W16 at 1.0004. It was integrity-checked and never scored.
- **Speed:** one request ran **263.66 -> 277.5 tok/s** (+5.2 %). The mean step fell from 3.793 to 3.604 ms. That is
  **0.18912 ms, 0.998 of P148's projection.** By comparison, D1 realized about 0.73 of its projection in P151's clean pairs.
- **Quality:** SANE inside the registered bars (|bias| ≤ 0.02, argmax ≥ 0.95). It read bias +0.002158, argmax 0.988932
  and KL 0.000601. That is in line with P150's I8HEAD at t1 (bias +0.000747, argmax 0.98904), read on other windows.
- **Tokens:** at W1, S1 is identical to S0 at both lengths. At W16, 11 of 16 rows are identical at 32 tokens and 4 of 16
  at 160. The knob rounds the head at every row count, as registered, so multi-row outputs change too.

## The W16 cost: observed, not explained

At W16 the int8 head is **passed, never routed**, yet S1's W16 rate is below S0's in every pair: 0.9918, 0.9970, 0.9924
and 0.9929 (geomean 0.9935). The head's own GEMV cannot be the cause, since it never runs at 16 rows. The S0 W16
self-pairs drift by 0.42 % and 0.07 %, and three of the four pairs move further than that. The cost sits inside the 0.99
no-regression line, so the rule reads it as no regression. It is recorded here so a default does not hide it.

What the knob changes even when the head is passed:
- **Memory:** the int8 copy stays resident alongside the bf16 weight. `memory_allocated` is 311,772,672 bytes higher at
  load, exactly the codes plus their fp32 scales. `memory_reserved` is 400 MiB higher (nvidia-smi 24487 vs 24087 MiB).
- **The dispatch:** the module's row check, which picks the one-row GEMV or `F.linear`.

A W16-only A/B, with the int8 copy built but the dispatch check skipped, would separate the two. It is $0 on the A2000
for correctness, or a later lane for the speed. This read cannot attribute the cost.

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| every W1 pair in [1.025, 1.055] | 1.0520-1.0530 | **hit** |
| g16 in [0.99, 1.01] | 0.991794 | **hit** |
| every W1 self-pair within [0.99, 1.01] | 0.9993-1.0003 | **hit** |
| SANE \|bias\| ≤ 0.005 and argmax ≥ 0.98 | +0.002158; 0.988932 | **hit** |
| S1's W1 tokens differ from S0's within 160 | identical at both lengths | **missed** |
| verdict FLIP | FLIP | **hit** |

## What it changes, and what it doesn't

- **The default:** none here. A separate PR may make `E4B_INT8_HEAD_BW=1` `qwen3_moe`'s default. It needs a
  grouped-nf4-gemm release carrying K36 as e4b's pin. Until then, the knob stays opt-in.
- **The measured basis is Qwen3-30B-A3B's head on an RTX 5090 (170 SMs).** That is one module, 151936 x 2048 and untied,
  at K36's tuned plan for that shape.
  - Other heads take K36's untuned default plan (`PLAN_I8_DEFAULT`), so their speed is not read.
  - P150's quality GO covers this checkpoint's head only.
  - Other cards are not read.
- **The W16 cost** above is real in this read and unexplained. A default PR should carry it forward rather than claim
  W16 unchanged.
- **Not read here:** contexts beyond the 4096-token slots, speculative verify, a smaller server, and memory headroom.
  The int8 copy is allocated after the server sized its slots; at this read's 16 slots of 4096 tokens it fit.
