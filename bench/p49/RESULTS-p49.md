# Results — P49: which store survives Gemma-4's layer 0 (H100 NVL, 2026-09-19)

Pre-registration: [`P49-PREREG.md`](P49-PREREG.md) (+ amendment 1: the activation probe's own bug, below). Run `p49-gemma4fmt` (H100 NVL, vast 51572740, e4b `e4f25ad`, gnf4 v0.32.0), receipt `receipts/experts4bit-qlora/2026-09-19/p49-gemma4fmt/p47/`, read by `bench/p49/p49_reduce.py`. K0 passed; control (i) refused the decode scorer again (reference self-KL 0.2787) → prefill on both sides, the same cached bf16 reference and 200 prompts / 5,879 tokens as P44-b / P47 / P48. Every row carries exactly 1 quantised / 29 bf16 expert stacks AND the store the builder named (`kl_serve._builder_check`).

## The stores, on layer 0 alone

| arm | store | KL nats/token | top-1 | general / technical / code / longctx |
|---|---|---|---|---|
| `L00_int8` | int8, blockwise 64 | **0.6928** | 0.729 | 0.549 / 0.857 / 0.977 / 0.523 |
| `L00_fp8` | fp8 (e4m3), blockwise 64 | 0.7744 | 0.699 | 0.792 / 0.983 / 0.975 / 0.606 |
| `L00_nf4b32` | NF4, block 32 | 0.8684 | 0.686 | 0.769 / 1.007 / 1.122 / 0.712 |
| `L00_nf4` | NF4, block 64 (the P48 anchor) | 0.8916 | 0.680 | 1.078 / 1.043 / 1.111 / 0.683 |
| `L00_fp4` | FP4, block 64 | 1.0514 | 0.652 | 1.106 / 1.342 / 1.260 / 0.852 |

For scale: the bf16 reference against itself under the other forward shape is 0.279, gpt-oss's NF4 requant on this instrument is 0.022, and **NF4 on layer 27 alone is 0.0056**.

## The same store (NF4 b64) at four depths — P48's profile, re-measured here

| arm | KL | top-1 |
|---|---|---|
| `L07_nf4` | 0.4436 | 0.767 |
| `L14_nf4` | 0.1007 | 0.891 |
| `L21_nf4` | 0.0144 | 0.952 |
| `L27_nf4` | 0.0056 | 0.970 |

Identical to P48's rows to four digits on a different box — the profile reproduces exactly.

## Verdicts

- **P0 (anchor) — HOLDS**: `L00_nf4` 0.8916, P48's 0.892.
- **P1 (int8 survives layer 0) — REFUTED, and this is the lane's result.** `L00_int8` reads **0.6928** against a registered ≤ 0.02 for "survives" and ≥ 0.20 for "does not". **Sixteen times finer quantisation buys a 22 % reduction in KL** (0.892 → 0.693) and leaves the model 124× further from the checkpoint than NF4 on layer 27. Under the registered decision rule: **Gemma-4's early expert layers cannot be quantised at all by any store e4b ships — they stay bf16.**
- **P2 (fp8 ≤ 0.05) — REFUTED**: 0.7744.
- **P3 (fp4 is no better than NF4's class) — HOLDS**: 1.0514, in fact worse than NF4 (its codebook is the wrong shape for these weights, as everywhere else).
- **P4 (block size does not rescue) — HOLDS**: block 32 reads 0.8684 = 0.97× block 64 (rule ≥ 0.55×). Halving the block halves the scale-sharing and buys 2.6 %.
- **P5 (post-norm amplification) — REFUTED, together with its registered alternative** (run `p49-gemma4fmt-2`; the section below). The expert-branch error is flat across depth and the post-norm does not amplify it; the sensitivity is positional, not magnitudinal.

## P5 — the activation probe (run `p49-gemma4fmt-2`, probe only, 11:37Z)

The redrawn probe (amendment 1's fix) captured the bf16 reference's MoE branch on every text layer over 8 prompts, then rebuilt each arm and compared the SAME layer's branch on the same tokens — the layers below are bf16 in both models, so the layer's inputs are identical and the difference is the store alone. `raw` = `experts(...)` before Gemma-4's `post_feedforward_layernorm_2`; `norm2` = after it, i.e. what is added to the residual.

### The same store (NF4 b64) at five depths

| layer | rel err, raw | rel err, post-norm | amplification | cos(raw) | end-to-end KL |
|---|---|---|---|---|---|
| 0 | 0.0793 | 0.0630 | 0.79 | 0.9967 | **0.8916** |
| 7 | 0.0922 | 0.0735 | 0.80 | 0.9959 | **0.4436** |
| 14 | 0.0962 | 0.0639 | 0.66 | 0.9951 | **0.1007** |
| 21 | 0.0845 | 0.0917 | 1.09 | 0.9969 | **0.0144** |
| 27 | 0.0796 | 0.1158 | 1.46 | 0.9969 | **0.0056** |

### Five stores, all on layer 0

| store | rel err, raw | rel err, post-norm | cos(raw) | end-to-end KL |
|---|---|---|---|---|
| int8 b64 | 0.0093 | 0.0079 | 0.9999 | **0.6928** |
| fp8 b64 | 0.0214 | 0.0170 | 0.9998 | **0.7744** |
| NF4 b32 | 0.0747 | 0.0589 | 0.9971 | **0.8684** |
| NF4 b64 | 0.0793 | 0.0630 | 0.9967 | **0.8916** |
| FP4 b64 | 0.1071 | 0.0841 | 0.9941 | **1.0514** |

### The reference's own branch, for scale

| layer | rms residual in | rms raw MoE out | rms post-norm | norm gain | post-norm / residual | dense / MoE |
|---|---|---|---|---|---|---|
| 0 | 1.578 | 0.1462 | 9.930 | 68× | 6.29 | 1.69 |
| 7 | 1.956 | 0.1055 | 13.507 | 128× | 6.90 | 1.92 |
| 14 | 2.205 | 0.0713 | 7.521 | 105× | 3.41 | 1.36 |
| 21 | 1.493 | 0.0639 | 3.874 | 61× | 2.60 | 2.49 |
| 27 | 1.146 | 0.0250 | 1.893 | 76× | 1.65 | 3.92 |
| 29 | 0.982 | 0.0277 | 0.918 | 33× | 0.93 | 5.31 |

### Verdict: **P5 REFUTED — and so is its registered alternative.**

- P5 predicted `rel_err_raw` flat across depth (it is: 0.0793–0.0962, a 1.21× spread, inside the registered 2×) **while `rel_err_norm2` tracked the KL**. It does not: Spearman(KL, `rel_err_norm2`) = **-0.900** — strongly NEGATIVE, the wrong direction — and the first/last ratio is 0.54 against a registered ≥ 5. The post-norm error *grows* with depth (0.063 at layer 0 → 0.116 at layer 27) while the KL *collapses* (0.892 → 0.0056).
- The registered alternative (the raw expert-sum error itself spreads and tracks the KL) is refuted too: Spearman(KL, `rel_err_raw`) = -0.100, i.e. nothing.

### What the rows say instead (an observation, NOT a tested prediction)

Quantising any single expert layer damages that layer's branch output by the same 8–10 % relative, wherever it sits, and the layer norm that follows does not amplify it — at layer 0 it slightly attenuates (0.79×) and only at depth does it mildly amplify (1.46×). **The end-to-end effect is therefore not a property of the perturbation's size at all; it is a property of how many layers the perturbation is processed by afterwards.** The two measurements that pin this down are in this lane:

- **Across depth, at fixed local damage**: the same ~8 % branch error costs 0.892 nats at layer 0 and 0.0056 at layer 27 — **159×** for an identical local injury.
- **Within layer 0, at varying local damage**: int8 makes the branch error **8.5× smaller** than NF4 (0.0093 vs 0.0793, cosine 0.9999 vs 0.9967) and buys **22 %** of the KL (0.892 → 0.693).

Position beats magnitude by two orders of magnitude, which is exactly why P49's store sweep found nothing: **no store can fix a layer-0 problem, because the problem is that it is layer 0.** It also explains P48's sub-additivity (30 singles summing to 7.6 against 1.08 together — once the early stream is perturbed, perturbing it again downstream adds little) and it forecloses "activation-aware scaling of the early experts" as a cheap fix: the branch output is already reproduced to cosine 0.997 and the model still moves 0.9 nats.

The reference columns add the mechanism's setting rather than the mechanism: the MoE branch's raw output shrinks with depth (rms 0.146 → 0.028), its post-norm contribution relative to the residual falls 6.3× → 0.94×, and the parallel dense branch grows from 1.7× to 5.3× the MoE branch. A late MoE layer is a small correction on a settled stream; an early one is a large fraction of a stream that 29 more layers will read.

**Consequence for e4b**: the remedy stands as P49 stated it — Gemma-4's early expert layers stay bf16 — and its cost is exactly what P50 (`bench/p50/P50-PREREG.md`, running) measures. No cheaper lever is implied by these rows, and none is registered.

## What this says about #597

**The store is not the variable.** Across a 4-bit codebook (NF4), a different 4-bit codebook (FP4), half the block size, an 8-bit integer store and an 8-bit float store, layer 0's KL moves between 0.69 and 1.05 — while the SAME NF4 store at layer 27 reads 0.0056. A 16× finer grid changing the answer by 22 % is the signature of a quantity that does not scale with quantisation error at all. Combined with P48's two weight-space probes (block-64 NF4 does the same ~9 % relative damage in every layer, per expert and per block), the conclusion is forced: **what makes layer 0 special is what happens to its expert outputs downstream, not how its weights are stored.** P5's probe is the instrument for that and is redrawn.

The remedy for e4b is therefore the one the registered rule names for a refuted P1: **keep Gemma-4's first k expert layers in bf16** (`quantize_layers` = all but 0..k−1, which the loader already supports and P47/P48 exercised). The cost is ~1.5 GB per layer against ~0.4 GB at NF4; P48's profile says k ≈ 10 covers 93 % of the gap and k ≈ 15 covers 96 %. That default belongs to a lane with its own K8 + KL gate — **nothing about Gemma-4 is quotable until it passes one**, and no such default ships from this lane.

## Cost

≈ 23 min on an H100 NVL at ≤ $3.10/h ≈ $1.20 against a registered $4.65; the probe redraw `p49-gemma4fmt-2` (probe only, `P47_KL_SKIP=1`) ≈ 20 min ≈ $1.05.
