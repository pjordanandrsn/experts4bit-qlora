# Serving throughput — every supported family under the Qwen3-30B campaign's protocol

*2026-09-04. Companion to [`SERVING-PARITY.md`](SERVING-PARITY.md) (quality) — this page is speed. Receipt: [`bench/hybrid-g9/throughput-20260904/`](../bench/hybrid-g9/throughput-20260904/README.md) (raw per-arm JSON, run logs, host forensics, the reduce script). Every row is an entry in [`claims.json`](claims.json) at tier **measured**.*

## Protocol

One rented RTX 5090 per lane. Per family: NF4 bake of the released checkpoint; K8 teacher-forced NLL (2048 steps, wikitext, sha-matched) for NF4 / int4 experts / + calibrated int4 attention; B=1 decode (512-token prompt, 128 generated, graph loop, timed window) for NF4, int4 experts, + C4-calibrated int4 attention, and the full fused stack (+ round-1/2 glue folds + router epilogue); B=16 aggregate for NF4 and int4 experts. `--no-fuse-qkv`, fp8 paged KV, all-VRAM placement, e4b 0.32.0 + grouped-nf4-gemm 0.26.0 (PyPI), transformers 5.16.1. **A refused arm is a result**: it names the fusion this family cannot license yet, and that list is the build-out.

Two lanes on two host classes, and **B=1 is host-bound**: OLMoE, Granite and gpt-oss ran on a Ryzen 9 9900X host (tpA); Qwen3, Gemma-4 and Mixtral on an EPYC 9755 host (tpB). Ratios to the Qwen3 reference across that line are indicative, not certified; the same-host re-measurement of Granite and gpt-oss on the EPYC host reproduced their NF4 B=1 and every B=16 number to within 1% (private receipt P30, validation lane), so the rows are comparable at that level.

## Per-family table (tok/s; K8 in nats)

| family | host | K8 nll nf4 / int4exp / calib | B=1 nf4 | B=1 int4exp | B=1 calib | B=1 fused | B=16 nf4 | B=16 int4exp |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (reference) | tpB | 1.8621 / 1.8578 / 1.8448 | 97 | 116 | 112 | 155 | 483 | 944 |
| OLMoE-1B-7B | tpA | 1.9380 / 1.9337 / 1.9295 | 248 | 346 | 336 | 452 | 1294 | 2412 |
| Granite-3.1-3B-A800M | tpA | 1.6741 / 1.6859 / 1.7003 | 191 | 285 | 218 | refused | 1447 | 2210 |
| gpt-oss-20b | tpA | 6.3354 / refused / refused | 124 | refused | refused | refused | 732 | refused |
| Gemma-4-26B-A4B | tpB | 4.8103 / refused / refused | 71 | refused | refused | refused | 572 | refused |
| Mixtral-8x7B-Instruct | tpB | 1.1805 / 1.1690 / 1.1916 | 48 | 99 | 107 | refused | 186 | 371 |

## Ratios to the reference (same protocol)

| family | B=1 NF4 | B=1 best licensed | B=16 NF4 | B=16 best |
|---|---|---|---|---|
| OLMoE-1B-7B | 2.54× | 2.92× | 2.68× | 2.56× |
| Granite-3.1-3B-A800M | 1.96× | 1.84× | 3.00× | 2.34× |
| gpt-oss-20b | 1.28× | 0.80× | 1.52× | 0.78× |
| Gemma-4-26B-A4B | 0.73× | 0.46× | 1.18× | 0.61× |
| Mixtral-8x7B-Instruct | 0.49× | 0.69× | 0.38× | 0.39× |

## What each family gets from the campaign's levers today (0.32.0)

- **Qwen3-30B-A3B (reference):** NF4 → full stack ×1.59 at B=1 (97 → 155); NF4 → int4 experts ×1.95 at B=16 (483 → 944). The campaign's 204.6 / 1,238 tok/s were a faster host and are quoted as measured-private in `STATUS.md`; this page is one host class, one day.
- **OLMoE-1B-7B:** every lever licenses; ×1.83 at B=1 (248 → 452), ×1.86 at B=16.
- **Granite-3.1-3B-A800M:** int4 experts ×1.49 / ×1.53 — **a quality FAIL by the registered gate, corrected 2026-09-04**: 1.6741 → 1.6859 nats is +0.063 ppl, over the 0.05-ppl uncalibrated budget (`experts4bit_qlora.k8_gate`); the table above kept the row unflagged because it was read against the family's noise floor only. The int4 rows for this family are speed measurements of an unlicensed configuration, not a best; calibrated attention is a quality FAIL here too (+0.026 nats over NF4 at the f32 attention path) and the fused arm refused: the layer body scales its residuals and names its MoE `block_sparse_moe`, which the round-2 fold did not license (build-out: e4b#371 + grouped-nf4-gemm#328).
- **gpt-oss-20b:** NF4 only. int4 experts refused by name (interleaved MXFP4 gate/up rows + bias epilogue; build-out e4b#372); the fused arm never reached the folds because the router probe refused first (its router selects on the logits with a bias; build-out e4b#370 + grouped-nf4-gemm#327). Raw-text perplexity is out of regime on this family (≈564), so its K8 is a same-arm delta instrument only.
- **Gemma-4-26B-A4B:** NF4 only. int4 experts refused for want of an adjudicated MoE convention (build-out e4b#369); the fused arm refused at the router probe (normed/scaled router with a per-expert scale; e4b#370). The one thing measured since: the round-1 norm fusion alone is ×1.30 at B=1 on this family (private receipt P30; quality gate pending — this model has no 512-token instrument, see `SERVING-PARITY.md`).
- **Mixtral-8x7B-Instruct:** int4 experts ×2.08 / ×2.00; calibrated attention ×1.076 more at B=1 and +0.011 nats on this window (the 8192-step gate from the earlier campaign stands until re-run); the fused arm refused at the router probe (`MixtralTopKRouter` renormalises without the attribute the matcher read; build-out e4b#370).

The build-out PRs above are **in review and unvalidated** at this date; their numbers, when they land, go in a second table here with their own receipt, never into this one.

## Build-out validation (0.33.0 → 0.34.0), measured in-repo

The refused arms in the table above became the build-out; one lane (bo3, the same RTX 5090 class on an
EPYC 9755 host, 17 phases over 2026-09-04) validated each piece with one arm per fusion. The full
receipt — every run JSON and log, the kernel and op censuses, the reducer, and the per-arm verdict table
in the registered gate's own units — is [`bench/hybrid-g9/throughput-20260904/bo3/`](../bench/hybrid-g9/throughput-20260904/bo3/README.md).
The licensed best per family after it (B=1 tok/s / B=16 tok/s, ratio to that family's NF4 baseline on
the same box):

| family | licensed configuration | B=1 | B=16 | K8 vs NF4 | gate |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | int4 experts + calibrated int4 attention + round-1/2 folds (now engaging, #375) + epilogue | **177.9** (×1.81) | 1089.6 (`all`) | −0.081 ppl | pass |
| Granite-3.1-3B-A800M | NF4 experts + round-1/2 folds (rotary-only fold, #379) + epilogue | **259.1** (×1.37) | **1689.6** (×1.18) | +0.019 ppl | pass |
| Gemma-4-26B-A4B | int4 experts + round-1 folds + epilogue | **121.1** (×1.69) | **962.8** (×1.68) | — | no instrument |
| Mixtral-8x7B | int4 experts + calibrated int4 attention + round-1/2 folds + epilogue | **110.0** (×2.29) | **373.4** (×2.00) | +0.045 ppl | pass (one text) |
| gpt-oss-20b | NF4 experts + round-1/2 folds | **133.3** (×1.08) | 833.9 (`stack`, int4 — FAILS) / 726.2 (NF4) | +0.049 nats | no instrument (OOD text) |

Two rows that were quoted in 0.33.0 are not here. Granite's "302 tok/s, ×1.59" carried int4 experts
that fail the registered 0.05-ppl gate (+0.063 ppl) — retracted in `docs/STATUS.md`; the row above is
the licensed one. gpt-oss's int4-expert rows fail on a grid mismatch (+0.63 nats); its native MXFP4
store is exact against the checkpoint's own bytes and reaches **151.1 tok/s at B=1** through
`gemv_mxfp4_b32` (×1.22), but this family's raw-text perplexity cannot rank an exact arm against a
noisy one (its NF4 arm scores *better* than bf16 upstream), so that number is quoted as speed with the
quality gate open, not as a licensed best.

## Reading the numbers

- Quote a ratio or quote the card and the host. The 5090 class carries ~8.5% inter-box dispersion; B=1 moves with the host CPU.
- The K8 column is paged-vs-paged: it says what a lever costs relative to NF4 on the same path, read against that family's floor (`METHODOLOGY.md` §13.1). It is not absolute parity — that is `SERVING-PARITY.md`.
- A refused arm is not a zero and is not a failure of the model; it is the next item on the list.
