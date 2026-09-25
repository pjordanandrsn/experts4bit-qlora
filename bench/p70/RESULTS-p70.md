# Results — P70: the fused router's weights cast to the model's dtype (RTX 5090, 2026-09-25)

Pre-registration: [`P70-PREREG.md`](P70-PREREG.md) (#744, merged before any rental), with amendment 1 (#745,
before the reading). Record: [#726](https://github.com/pjordanandrsn/experts4bit-qlora/issues/726). Receipts:
[`receipts/`](receipts/).

The reducer ran on the box. Its output is the record, and every number below is quoted from it:
[`receipts/RESULTS-p70-generated.md`](receipts/RESULTS-p70-generated.md) and
[`receipts/p70_rep.json`](receipts/p70_rep.json).
- **What cannot be recomputed here.** The KLs come from per-pass logits, which stayed on the box by registration.
- **What can be checked from the committed files.** Every pass's census, the prompt rows, the pack's identity, the
  build's K8 and the proof's result.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p70-prove-1` | proving rental | REFUSED before any rental: the approval evidence was a blob URL | $0 | `9751a87` |
| `p70-prove-2` | proving rental at `a37e95a` | rc 27: the cast check failed on an instrument defect (amendment 1) | $0.0345 | `6b3a192` |
| `p70-prove-3` | proving rental at `c77aab6` | NOT_RUN: a dud box, ssh never answered | $0.0304 | `6364451` |
| `p70-prove-4` | proving rental at `c77aab6` | **PASS rc 0, `PROVED`** | $0.0256 | `6a8b555` |
| `p70-5090-1` | the reading at `c77aab6` | NOT_RUN: a dud box, ssh publickey denied | $0.0326 | `f27c0bf` |
| **`p70-5090-2`** | **the reading**, same commit `c77aab6` | **rc 0, VALID** | **$0.6925** | `44fe199` |

- **Lane total:** $0.8156 against a $2.50 ceiling. Teardown is proven on every rented run
  ([`receipts/teardown-proof.json`](receipts/teardown-proof.json) for the reading).
- **The launcher defect.** `p70-prove-1` exposed a defect: a refused launch left a receipt that its own ledger check
  rejected, which blocked every later launch. It was fixed in adertha-agents#129 before `p70-prove-2`.
- **The proof** ([`receipts/proof/prove_cast.json`](receipts/proof/prove_cast.json)). It ran on an RTX 5090, on the
  real `int4_b32.router_epilogue`, at Qwen3's router shape.
  - With the switch off, the weights are fp32 at ≤ 64 rows. With it on, they are bf16.
  - At 64 rows and at 1 row, the fused path picks the same experts as upstream at that row count, in both states.
  - Cast-on weights are **bit-equal to upstream's (fraction 1.0)** at 64 and at 1 row.
  - Above 64 rows the forward is upstream's, in both states.
  - The top-k slot order differs from `torch.topk`'s, and the GEMM itself varies with the row count. Both were
    reported, as amendment 1 registered.
- **Box.** One RTX 5090 (driver 575.57.08, power limit 600 W; nothing here is timed) on an AMD Ryzen 9 7950X host
  with 124 GiB of RAM, not shared (3 GiB in use at start; [`receipts/forensics.txt`](receipts/forensics.txt)).
- **Software.** e4b 0.37.4 at `c77aab6`; grouped-nf4-gemm 0.33.0 at `5ca1897` (the registered pin); torch
  2.8.0+cu128, Triton 3.4.0, transformers 5.16.1, bitsandbytes 0.50.1.
- **The pack is the licensed one.** `pack.json` records `sha256:0c9955a9…`, equal to P55x's licensed pack. The
  build's wikitext K8 read ppl **6.36709**, P55x's figure to five decimals, and P64's.
- **Where the time went.**
  - The fetch and bake ran 04:02–04:17Z.
  - The build took 29 min, with a first calibration chunk of 360 s, so the host was not limiting.
  - All twelve passes ran 04:46–05:08Z, at 74–112 s per pass and text for 2,048 decode steps.
  - The reducer ran at 05:08Z.
  - Nothing was skipped. The run used 69 min of its 2.5 h guard.

## Validity: VALID (read first)

- **P1, determinism.** `KL(ep32 ‖ ep32_rep)` is exactly 0 on all 2,048 decode positions of both texts.
- **P2, the cast touches no > 64-row forward.** The prefill-last logits of every pass are bit-identical to `ep32`'s.
- **P3, engagement.** Every pass's counts equal the registered ones, and all 48 routers are hooked (see the census
  files under `receipts/out/`).
  - `ep32` decoded through fp32 router weights, and `ep16` through bf16: 98,304 router calls per pass and text
    (48 layers × 2,048 steps).
  - Every prefill of the primary pair and of both floor samples ran through the original router, above 64 rows:
    2,304 calls at chunk 128, 3,072 at 96 and 768 at 384.
  - Only `ep32_pc64` prefilled through the fused router, with 4,608 fp32 calls.
- **Rows.** The same 16 rows × 512 tokens were used in every pass: wikitext = P59's rows `f67e7e4d…`; c4val1 =
  `17e68b42…`.

## The read

| text | G = KL(ep32 ‖ ep16) | top-1 | floor F (mean of pc96, pc384) | G / F | class | dNLL (95 % CI) | F_NLL |
|---|---|---|---|---|---|---|---|
| wikitext | **0.003309** | 0.9775 | 0.005523 (0.005581, 0.005465) | **0.60** | BELOW FLOOR | +0.00096 (−0.00401, +0.00589) | 0.00330 |
| c4val1 | **0.002806** | 0.9761 | 0.003045 (0.003002, 0.003089) | **0.92** | BELOW FLOOR | +0.00216 (−0.00111, +0.00566) | 0.00180 |

**VERDICT (the cast, G): INDISTINGUISHABLE.**
- **What it means.** Rounding the fused router's 8 routing weights per token to bf16 changes the served model's
  output distribution by less than this instrument's own arithmetic-order floor, on both texts. This covers every
  decode step of the 48 MoE layers.
- **dNLL.** Both intervals span 0, and |dNLL| is under the family's 0.0095 K8 floor.

### Predictions

| prediction | registered | read | verdict |
|---|---|---|---|
| P1 determinism | `KL(ep32 ‖ ep32_rep)` exactly 0 | 0 on 4,096 / 4,096 positions | **HELD** |
| P2 no > 64-row forward touched | prefill-last logits bit-identical | identical | **HELD** |
| P3 engagement | counts as registered, every router hooked | as registered, 48 / 48 | **HELD** |
| P4 the primary | G ≤ F on both texts | 0.60 F / 0.92 F | **HELD** |
| P5 NLL | \|dNLL\| ≤ F_NLL on both texts | wikitext 0.00096 ≤ 0.00330; **c4val1 0.00216 > 0.00180** | **BROKEN on c4val1** |
| P6 P64's floor sample | pc64 / F ∈ [0.5, 2] | 1.01 / 1.08 | **HELD** |

- **P5 broke on c4val1, and it broke in the rehearsal on wikitext.** In both cases the cast's |dNLL| exceeded the
  floor pair's NLL difference: by 1.2× here, and by 6.5× on the A2000 rehearsal's unlicensed pack.
- **It moves no class.** MATERIAL needs G > 2F, dNLL > F_NLL, *and* an interval excluding 0. Here G ≤ F on both
  texts, and c4val1's interval (−0.00111, +0.00566) spans 0.
- **What it says about the prediction.** A two-sample floor's NLL difference is a tight bar for a point estimate
  whose own sampling width is ±0.0034 nats. P5 as written was stricter than the instrument can resolve. It is
  reported as broken, not re-derived.
- **P6 held.** P64's `pc64` floor sample read 1.01 F and 1.08 F. The router switch its prefill straddles did not
  dominate it, so P64's reading on that floor stands as registered, and no correction issue is filed.

## What follows (the decision rule)

- **INDISTINGUISHABLE → the cast becomes the default for the `softmax_topk` kind** (Qwen3-MoE, OLMoE, Mixtral) in
  the next e4b release.
  - `E4B_ROUTER_EPI_CAST=0` restores the fp32 weights for one release.
  - The `topk_softmax` kind (gpt-oss, GraniteMoe) keeps fp32 until it is read.
  - The flip lands as its own change, after this read.
- **A register row records G, F, the class and dNLL per text:**
  `e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25`.
- **#726 closes** when the default flip merges. The fused path then computes the upstream routing at every row count,
  as a map from expert to weight (amendment 1).

## What this read does not say

- **Verify and batch size.** Nothing about verify (T = 16/17) or B = 16. Every scored forward is B = 1. Lane P68's
  machinery reads verify.
- **Other router kinds.** Nothing about `topk_softmax` (gpt-oss, GraniteMoe), or about Gemma-4, which already casts.
- **Speed.** A bf16 cast of 8 values per row is not timed.
- **No bf16 reference.** The comparison is the served stack against itself.
- **Slot layout.** Nothing about whether any consumer's accumulation order depends on it. Every pass here used one
  kernel.
- **Scope.** One family, two texts, 2,048 decode positions each.
