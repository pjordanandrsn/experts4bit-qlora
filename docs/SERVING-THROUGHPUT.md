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
The best per family after it (B=1 tok/s / B=16 tok/s, ratio to that family's NF4 baseline on the same
box), with the registered gate's reading — a calibrated pack is licensed only with the same sign on a
second text, which these lanes did not score (the second text was scored on lane bo5, next section: both
calibrated rows FAIL as registered, and Mixtral's uncalibrated `stack` fails it by 0.008 ppl):

| family | licensed configuration | B=1 | B=16 | K8 vs NF4 | gate |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | int4 experts + calibrated int4 attention + round-1/2 folds (now engaging, #375) + epilogue | **177.9** (×1.81) | 1089.6 (`all`) | −0.081 ppl | calibrated: one text scored, the rule needs two |
| Granite-3.1-3B-A800M | NF4 experts + round-1/2 folds (rotary-only fold, #379) + epilogue | **259.1** (×1.37) | **1689.6** (×1.18) | +0.019 ppl | pass |
| Gemma-4-26B-A4B | int4 experts + round-1 folds + epilogue | **121.1** (×1.69) | **962.8** (×1.68) | — | no instrument |
| Mixtral-8x7B | int4 experts + calibrated int4 attention + round-1/2 folds + epilogue | **110.0** (×2.29) | **373.4** (×2.00) | +0.045 ppl | calibrated: one text scored, the rule needs two (uncalibrated `stack` 102.8 / 372.0 passes) |
| gpt-oss-20b | NF4 experts + round-1/2 folds | **133.3** (×1.08) | 833.9 (`stack`, int4 — FAILS) / 726.2 (NF4) | +0.049 nats | no instrument (OOD text) |

Two rows that were quoted in 0.33.0 are not here. Granite's "302 tok/s, ×1.59" carried int4 experts
that fail the registered 0.05-ppl gate (+0.063 ppl) — retracted in `docs/STATUS.md`; the row above is
the licensed one. gpt-oss's int4-expert rows fail on a grid mismatch (+0.63 nats); its native MXFP4
store is exact against the checkpoint's own bytes and reaches **151.1 tok/s at B=1** through
`gemv_mxfp4_b32` (×1.22), but this family's raw-text perplexity cannot rank an exact arm against a
noisy one (its NF4 arm scores *better* than bf16 upstream), so that number is quoted as speed with the
quality gate open, not as a licensed best.

## The second text and the optimisation pass (bo5), measured in-repo

Lane bo5 — box 49841214, the same RTX 5090 class on an EPYC 9755 host, 2026-09-04 08:5x–12:33Z; e4b
integration-6 @0535930 (0.34.0 + #372 + #384 + #385) and integration-7 @d090940 (+ #387), grouped-nf4-gemm
0.29.0 + `combine_rows` — scored the second text the registered rule asks for (C4 validation, `c4val1`) on
every calibrated pack bo3 had left at "one text", re-measured the shipped positions on one box, and A/B'd the
drafts. Receipt: [`bench/hybrid-g9/throughput-20260904/bo5/`](../bench/hybrid-g9/throughput-20260904/bo5/README.md);
the per-arm table with the verdict column is its
[`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo5/RESULTS.md). The gate column below is the
registered rule in perplexity on every text scored; the nats beside it never change it. `i6` / `i7` = the cut.

| family | configuration | cut | B=1 | B=16 | K8 vs NF4 (wikitext / c4val1) | gate (registered, ppl) |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | int4 experts + calibrated int4 attention + round-1/2 folds + epilogue + #385 glue (`all`) | i6 | **204.1** | **1251.6** | −0.105 / **+0.063** ppl (−0.0164 / +0.0038 nats) | FAIL as registered on c4val1; attributes to noise (every attribution arm sub-floor, the exact folds alone −0.073) — measured speed, NOT licensed pending the user's decision; not retuned |
| Granite-3.1-3B-A800M | NF4 experts + round-1/2 folds (rotary-only fold) + epilogue (`nf4_r12epi`, the licensed stack) | i6 | **294.1** | **1736.1** | this lane's baseline (bo3: +0.019 ppl vs NF4) | licensed |
| Granite-3.1-3B-A800M | licensed stack + C4-calibrated int4 experts (#384 draft, `calibexp_r12epi`) | i6 | 409.8 | 3050.1 | +0.014 / **+0.387** ppl (10× the floor) | FAIL — refused |
| Granite-3.1-3B-A800M | licensed stack + #387 fused q/k/v + fused rope-only fold (`nf4_r12epi_fq2`) | i7 | 286.5 (×0.968 vs 295.9 unfused) | 1748.4 (×1.014 vs 1724.6) | −0.006 ppl (−0.0011 nats) / — | pass; buys nothing — draft |
| gpt-oss-20b | native MXFP4 store: GEMV for single rows, NF4 kept for batched rows (`store_r12`) | i6 | **173.3** (×1.270 vs `r12` 136.4) | **719.7** (×0.971 vs 741.4) | — | no instrument (exact bytes, bo3q); the ×0.81 B=16 penalty is recovered |
| Mixtral-8x7B | int4 experts + calibrated int4 attention + round-1/2 folds + epilogue (`all`) | i6 | **123.3** | **377.3** | +0.033 / **+0.116** ppl | FAIL as registered on c4val1 — measured speed, NOT licensed |
| Mixtral-8x7B | int4 experts + round-1/2 folds + epilogue, no calibrated pack (`lic`) | i7 | **112.0** | **376.5** | −0.046 / **+0.058** ppl (−0.0142 / +0.0070 nats; floor unmeasured) | FAIL as registered on c4val1 by 0.008 — the "licensed stack" label is WITHDRAWN pending the user's decision on a nats gate against a measured floor; NOT licensed |
| Mixtral-8x7B | `lic` + #387 fused q/k/v, 32 modules (`lic_fq`) | i7 | 115.6 (×1.032) | 378.3 | +0.0002 nats vs `lic` / — | pass vs `lic`; inherits `lic`'s verdict — draft |

What this settles: Granite's calibrated-expert route (#384) fails the second text and is refused; Mixtral's
int4-expert stack is one-text pass / one-text FAIL-as-registered, so the label P30 and the table above gave it
is withdrawn (the gate is not retuned to fit; re-registering it in nats against a *measured* floor is a user
decision); Qwen3's `all` fails the second text by a margin every attribution arm shows to be noise, same
disposition; gpt-oss's route rule holds ×1.27 at B=1 without the B=16 penalty; #387 is quality-clean and buys
nothing (on Granite the census shows GPU time −4% and wall +3% — unexplained non-GPU time on the fused path).
bo3's 177.9 / 1089.6 and bo5's 204.1 / 1251.6 for Qwen3 are different boxes of the same class (~8.5%
dispersion; B=1 is host-bound) and different cuts (bo5 carries #385's glue, ×1.057 measured on the bo5 box):
quote either with its box, never the ratio between them.

## Closing the gap, not the gate (bo6), measured in-repo

Lane bo6 + bo6b — box 49861751, an RTX 5090 on a Threadripper PRO 7975WX host, 2026-09-04 13:25Z onward; e4b
integration-8 @db2a070 (attempt 3) and @ae9dc122 (bo6b: streamed calibration), grouped-nf4-gemm main @0b25d13. The
registered gate stayed in perplexity; the int4-expert arms bo5 had failed on their second text with RTN experts were
re-run with per-expert GPTQ calibration (e4b#384), against NF4 **re-scored on this box** (Qwen3 wikitext 1.85939 /
c4val1 2.80318 — repeated after a fresh bake, bit-identical; Mixtral 1.18214 / 2.10973). Receipt:
[`bench/hybrid-g9/throughput-20260904/bo6/`](../bench/hybrid-g9/throughput-20260904/bo6/README.md); the per-arm
table with the method and verdict columns is its
[`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo6/RESULTS.md). **Method** is read from each arm's run log:
*all-at-once* = every layer's Hessian accumulated against the unquantised prefix, then packed (the two-step API);
*streamed* = each layer chunk packed before the next chunk's Hessians accumulate, GPTQ's sequential convention — the
mechanism 0.35.0 ships. Speed is rental-measured on this box with **no ratio** (the lane has no NF4 speed arm; bo7
measures it).

| family | configuration | method | cut | B=1 | B=16 | K8 vs NF4 (wikitext / c4val1) | gate (registered, ppl) |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | calibrated int4 experts alone (`calibexp`) | all-at-once, 16k tok, damp 0.01 | db2a070 | — | — | — / **+0.150** ppl (+0.0090 nats) | FAIL as registered — a verdict on the all-at-once method (`e4b.serve.buildout.bo6.qwen3.calibexp-allatonce.c4val1.2026-09-04`) |
| Qwen3-30B-A3B | calibrated int4 experts + calibrated int4 attention + round-1/2 folds + epilogue + #385 glue (`all_calibexp`) | all-at-once, 16k tok | db2a070 | **158.0** | **993.6** | −0.060 / +0.035 ppl (−0.0093 / +0.0021 nats) | **pass on both texts**; the wikitext improvement is not claimed (signs differ) (`…bo6.qwen3.all-calibexp-allatonce.k8…`, `…bo6.qwen3.b1…`, `…bo6.qwen3.b16…`) |
| Qwen3-30B-A3B | calibrated int4 experts alone (`calibexp_c4val_rep1`) | **streamed**, 16k tok, damp 0.01 | ae9dc122 | — | — | — / **−0.050** ppl (−0.0031 nats) | pass on 1 text; improvement not claimable until wikitext agrees (bo6c) (`…bo6.qwen3.calibexp-streamed-16k.c4val1…`); the order alone moves c4val1 by 0.200 ppl (`…bo6.qwen3.calibration-order.c4val1…`) |
| Qwen3-30B-A3B | calibrated int4 experts alone, damping 0.1 (`calibexp_d01`) | streamed, 16k tok, damp 0.1 | ae9dc122 | — | — | — / +0.054 ppl (+0.0033 nats) | FAIL as registered by 0.004 (`…bo6.qwen3.calibexp-streamed-16k-damp0.1.c4val1…`) |
| Qwen3-30B-A3B | calibrated int4 experts alone, 4× the calibration set (`calibexp_n128`) | streamed, 64k tok | ae9dc122 | — | — | — / **−0.211** ppl (−0.0129 nats, 1.4× floor) | pass on 1 text; the sweep's best point, improvement not claimable until wikitext agrees (`…bo6.qwen3.calibexp-streamed-64k.c4val1…`) |
| Qwen3-30B-A3B | calibrated int4 experts alone, 16× the calibration set (`calibexp_n512`) | streamed, 256k tok | ae9dc122 | — | — | — / −0.141 ppl (−0.0086 nats) | pass on 1 text; non-monotonic sweep at one measurement per point (`…bo6.qwen3.calibexp-streamed-256k.c4val1…`) |
| Mixtral-8x7B | calibrated int4 experts + round-1/2 folds + epilogue, no calibrated pack — bo5's `lic` with calibrated experts (`lic_calibexp`) | streamed, 16k tok, 8 GiB budget | ae9dc122 | not measured (alarm) | not measured (alarm) | **+0.077** / +0.039 ppl (+0.0234 / +0.0047 nats; floor unmeasured) | **FAIL as registered** on wikitext — measured, not licensed; the mirror image of bo5's RTN `lic`. B=1 / B=16: the arms were killed by the lane's 3600-s alarm at pass 23 of the 32-pass streamed calibration (~85 min) — a harness limit, no number; bo7 dropped them too (`e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`) |

What this settles: the gap closes by *method*, not by moving the gate. Attempt 3's "calibrated experts alone FAIL
+0.150" is a verdict on all-at-once calibration; the streamed (sequential) method — what e4b#384 shipped — reads
−0.050 on the same text, box and batches, and Qwen3's full calibrated stack passes both texts as registered. **It did
not close Mixtral's:** the calibrated stack passes the in-domain text (+0.039) and fails the out-of-domain one
(+0.077, over the +0.05 budget) — FAIL as registered, measured, not licensed; not a reference shift (same window sha as
bo5c, NF4 within 0.001 nats); its speed was not measured on this lane (the arms alarmed mid-calibration) and bo7
dropped those arms. Next levers are not gate changes: a per-expert NF4 fallback for the largest-residual experts, or
the 64k calibration set scored on wikitext. Granite stays NF4 (bo5's +0.387 is not closable with this
lever). Nothing here is claimed as an improvement over NF4: every
streamed arm has one text, and the rule needs wikitext with the same sign (lane bo6c, separate). The Qwen3 speed on
this box (158.0 / 993.6) and bo5's on its EPYC box (204.1 / 1251.6, RTN experts) are different hosts and cuts with
no same-box NF4 arm and no bandwidth probe on this one: quote each with its box, never the ratio between them.
Qwen3's NF4 c4val1 reference reads 0.006 nats from bo5's on the identical window (Mixtral's agree to 0.001); the K8
instrument is bit-for-bit deterministic within a box, so that shift is between installs or boxes and stays open — no
sub-0.01-nat number is compared across lanes.

## Reading the numbers

- Quote a ratio or quote the card and the host. The 5090 class carries ~8.5% inter-box dispersion; B=1 moves with the host CPU.
- The K8 column is paged-vs-paged: it says what a lever costs relative to NF4 on the same path, read against that family's floor (`METHODOLOGY.md` §13.1). It is not absolute parity — that is `SERVING-PARITY.md`.
- A refused arm is not a zero and is not a failure of the model; it is the next item on the list.
