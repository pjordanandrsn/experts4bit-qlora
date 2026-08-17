# RESULTS — G7 quality clause: what FP8 KV costs (two models)

Gate G7's quality clause is **mandatory and non-negotiable**: perplexity
delta ≤0.5% vs FP16 on a held-out set, plus at least one downstream task.
Its stop condition is explicit — *do not ship degraded KV to hit a batch
number*. So this clause was measured before the fused kernel was written,
and it is reported here on its own.

Method: held-out wikitext-2 test, 24,576 tokens per model, fed in 256-token
pieces so the cache is **read across calls** rather than only within one
forward; LAMBADA last-word accuracy over 200 items as the downstream task,
chosen because it cannot be answered without the long-range context a KV
format damages. All arms share one loaded model and identical token
streams, so every delta is paired. Everything is deterministic (greedy,
fixed seed); the absolute perplexity of a small model is uninteresting, the
paired difference between storage formats is the result.

## Controls (the verdict is void without them)

| control | Llama-3.2-1B | Qwen3-0.6B | meaning |
|---|---|---|---|
| null (`off` vs stock cache) | **0.0000%** | **0.0000%** | plumbing is neutral; deltas are the format |
| positive (`crush`, 2-bit) | **+288×** | **+17,095×** | the harness demonstrably SEES KV damage |
| downstream baseline | 0.585 | 0.435 | the task is answerable unquantized |

Without all three the harness reports `FAILED_TO_MEASURE`, never `PASS` —
"FP8 costs nothing" and "the KV was never exercised" produce the same
number, and only a control separates them.

## Results

| arm | Llama-3.2-1B Δppl | Qwen3-0.6B Δppl | Δlambada (L / Q) | compression |
|---|---|---|---|---|
| `fp8` — E4M3 K+V, per-row scale | **+0.574%** ✗ | +0.080% ✓ | +0.005 / −0.005 | 1.88× / 1.94× |
| `fp8_kg64` — key scale per 64 | +0.574% ✗ | +0.226% ✓ | +0.005 / −0.005 | 1.88× / 1.91× |
| **`fp8_kg32` — key scale per 32** | **+0.411%** ✓ | **+0.085%** ✓ | +0.015 / −0.010 | 1.83× / 1.86× |
| `fp8_vonly` — bf16 keys, E4M3 values | +0.009% ✓ | +0.126% ✓ | 0.000 / +0.005 | 1.31× / 1.32× |
| `int4` — symmetric 4-bit K+V | +32.8% ✗ | +154,322% ✗ | −0.065 / −0.435 | 3.56× / 3.77× |

**The directive's literal format misses the mandatory bar on one of two
models** (Llama-3.2-1B, +0.574% against ≤0.5%). A gate is not passed by
choosing the model where it passes, so `fp8` as specified is a MISS.

`fp8_kg64` is identical to `fp8` on Llama by construction, not by
coincidence: that model's head_dim is 64, so a 64-wide group *is* the full
row. The arm is a no-op there and is kept only so the table is readable.

## Where the cost lives, measured

The `fp8_vonly` arm localizes it: keeping keys in bf16 drops Llama's cost
**62×** (0.574% → 0.009%). Essentially the entire quality cost of FP8 KV is
in the **keys**. Two independent lines of evidence agree:

- This repo's earlier NF4-KV work measured the same asymmetry on a
  different model and format — K-only +0.083 perplexity vs V-only +0.013.
- The Phase-7 outlier probe (`probe_kv_outliers.py`) measures why keys are
  hard for *uniform* formats: median key rows carry `amax/rms` ≈ **10.05**
  (Qwen) and **4.43** (Llama), against ≈ 3 for value rows.

That probe also explains the INT4 catastrophe quantitatively. INT4's step
is `amax/7`, so at `amax/rms` = 10 a **typical** key element spans **0.70
levels — it rounds to zero**. The median key value is annihilated and
attention collapses. The metric predicts the damage across models: Qwen
(ratio 10.05) loses 154,322%, Llama (ratio 4.43, 1.61 levels) loses 32.8%.
This is also why NF4 KV worked historically where INT4 fails here — NF4 is
a *non-uniform* codebook, dense near zero — and why FP8 is largely immune:
floating point carries **relative** precision, so an outlier costs its row
nothing.

Note the corollary, since it cuts against the intuition that finer is
always better: grouping keys more finely does **not** monotonically help.
On Qwen, per-64 grouping (+0.226%) is worse than the full row (+0.080%),
and per-32 (+0.085%) merely returns to it. Finer groups recover only ~30%
of Llama's key cost, which says the residue is E4M3's 3-bit mantissa on
keys rather than dynamic range — something a scale grid cannot fix.

## Verdict and the format Phase 7 carries forward

- **`fp8` as literally specified: MISS** (+0.574% on Llama-3.2-1B).
- **`fp8_kg32`: PASS on both models** (+0.411%, +0.085%), at a byte cost of
  ~3% against the literal format (1.855× vs 1.939× compression on the
  235B-class shape). This is still E4M3 for K and V with a
  per-token-per-head scale — the scale is simply subdivided along the head
  dimension — so it is a refinement of the specified format rather than a
  different one, and it is what the remaining Phase-7 work targets.
- **`int4`: measured, catastrophic, stays default-off** with its number and
  its mechanism recorded, exactly as the directive requires.

Margins are stated plainly rather than rounded away: Llama's `fp8_kg32`
headroom is 18% of the bar (0.411% against 0.500%), and the same format's
`B_max` headroom is thin too (below). Neither is comfortable.

## Consequence for the capacity clause (arithmetic, not yet measured)

For a Qwen3-235B-class model (94 layers, 4 KV heads, head_dim 128) the byte
model reproduces the documented 188.0 KB/token for bf16 exactly, which is
the check that it is right:

| format | bytes/token | per 4K sequence | KV budget for `B_max` = 25 |
|---|---|---|---|
| bf16 | 192,512 (188.0 KB) | 770 MB | 19.3 GB |
| `fp8` | 99,264 (96.9 KB) | 407 MB | 9.9 GB |
| **`fp8_kg32`** | 103,776 (101.3 KB) | 425 MB | **10.4 GB** |
| `fp8_vonly` | 145,888 (142.5 KB) | 598 MB | 14.6 GB |

`fp8_vonly` — the variant with the best quality — cannot reach `B_max` ≥25
inside a plausible KV budget, which is precisely the trade the directive's
stop condition forbids resolving in the batch number's favour. `fp8_kg32`
is the configuration that can satisfy both clauses, and whether it actually
does is a measurement on a real box, not this table.

## Still owed for G7

1. `B_max` ≥25 at 4K on a 235B-class model — a box run, against the
   arithmetic above.
2. Attention kernel ≥70% of measured `B_vram` — the Triton paged-attention
   kernel with in-register dequant, which is the risky clause of this
   phase: the in-tree precedent for a *fused* KV attention kernel is one
   measuring **11.6× slower** than bf16 SDPA invoked properly, after an
   earlier 0.82× claim turned out to be a baseline error.

Receipts in this directory: per-model quality JSON (all arms, controls,
byte accounting), outlier probes, and both harness scripts.
