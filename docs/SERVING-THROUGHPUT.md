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

## Closing the gap, not the gate (bo6, bo6b, bo6c), measured in-repo

Lane bo6 + bo6b + bo6c — box 49861751, an RTX 5090 on a Threadripper PRO 7975WX host, 2026-09-04 13:25Z to
2026-09-05 03:18Z; e4b integration-8 @db2a070 (attempt 3) and @ae9dc122 (bo6b and bo6c: streamed calibration; bo6c's
cut and hook v6 verified live by its tripwire, 24 GiB Hessian budget), grouped-nf4-gemm main @0b25d13. The
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
| Qwen3-30B-A3B | calibrated int4 experts alone, 4× the calibration set (`calibexp_n128`; wikitext from bo6c) | streamed, 64k tok | ae9dc122 | — | — | −0.0002 / **−0.211** ppl (−0.00003 / −0.0129 nats) | pass on both texts; the improvement clause met on its letter (same sign) but the out-of-domain delta is inside the floor — parity out of domain, −0.211 in-domain, no improvement by a number (`…bo6.qwen3.calibexp-streamed-64k.c4val1…`, `e4b.serve.buildout.bo6c.qwen3.calibexp-streamed-64k.k8.2026-09-05`) |
| Qwen3-30B-A3B | calibrated int4 experts alone, 16× the calibration set (`calibexp_n512`) | streamed, 256k tok | ae9dc122 | — | — | — / −0.141 ppl (−0.0086 nats) | pass on 1 text; non-monotonic sweep at one measurement per point (`…bo6.qwen3.calibexp-streamed-256k.c4val1…`) |
| Qwen3-30B-A3B | **the shipping stack**: streamed 64k calibrated int4 experts + calibrated int4 attention + round-1/2 folds + epilogue + #385 glue (`all_calibexp_n128`, bo6c) | **streamed**, 64k tok | ae9dc122 | not on this lane (bo7, at the 16k default) | not on this lane | **−0.053 / −0.066** ppl (−0.0083 / −0.0040 nats, both sub-floor) | **pass on both texts — LICENSED under the unchanged gate**; at parity or better on both texts, no improvement by a number (`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`); streamed calibration is run-to-run deterministic (`…bo6c.qwen3.calib-deterministic…`) |
| Mixtral-8x7B | calibrated int4 experts + round-1/2 folds + epilogue, no calibrated pack — bo5's `lic` with calibrated experts (`lic_calibexp`) | streamed, 16k tok, 8 GiB budget | ae9dc122 | not measured (alarm) | not measured (alarm) | **+0.077** / +0.039 ppl (+0.0234 / +0.0047 nats; floor unmeasured) | **FAIL as registered** on wikitext — measured, not licensed; the mirror image of bo5's RTN `lic`. B=1 / B=16: the arms were killed by the lane's 3600-s alarm at pass 23 of the 32-pass streamed calibration (~85 min) — a harness limit, no number; bo7 dropped them too (`e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`) |

What this settles: the gap closes by *method*, not by moving the gate. Attempt 3's "calibrated experts alone FAIL
+0.150" is a verdict on all-at-once calibration; the streamed (sequential) method — what e4b#384 shipped — reads
−0.050 on the same text, box and batches, and Qwen3's full calibrated stack passes both texts as registered — under
the all-at-once pack on bo6, and **under the shipping (streamed, 64k-token) pack on bo6c, which is Qwen3-30B-A3B's
licensed serving configuration: at parity or better on both texts, both deltas inside the floor, no improvement
claimed by a number.** Its speed is not on this lane: bo7 measures the calibrated stack at the hook's 16k-token
default, not 64k. **It did not close Mixtral's:** the calibrated stack passes the in-domain text (+0.039) and fails the out-of-domain one
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

## Throughput census (bo7), measured in-repo

Lane bo7 — box 49916675, an RTX 5090 (driver 595.84) on an AMD EPYC 7Q83 host (128 threads, 251 GB), 2026-09-05
02:00Z onward; **the shipped code**: e4b main @f4b639fd2640 (= 0.35.0 + docs) + grouped-nf4-gemm main @ddcb850e05c3
(= 0.30.0 + docs), hook v6 at its 16k-token default; torch 2.8.0+cu128, transformers 5.16.1, bitsandbytes 0.50.1,
triton 3.4.0. **Speed only**: per family an NF4 bake, then every arm at B=1 (512-token prompt, 128 generated, graph
loop, 127 timed steps, `--no-fuse-qkv`, fp8 paged KV, all-VRAM) and B=16 (graph loop, 70 steps, aggregate) — 48 arms
over six families; the calibrated arms under a 5400-s alarm, the rest 3600 s; an alarmed or refused arm is a row.
Receipt: [`bench/hybrid-g9/throughput-20260904/bo7/`](../bench/hybrid-g9/throughput-20260904/bo7/README.md); the
per-arm tables with the register's licence label on every row are its
[`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo7/RESULTS.md). Pre-registered reading rules, applied verbatim:
(1) every ratio is the family's own NF4 arm on this box, same session — never a cross-lane ratio; (2) the licence label
of each arm is this register's at bundle time (bo3, bo5, bo6, bo6c) — bo7 licenses nothing, and where the register is
silent the label is "no quality verdict on record"; (3) three axes for every licensed best — ratio ×N over NF4 on this
box, rental-measured tok/s on this box, and the anchor-class projection marked as a projection, which exists only for
Qwen3-30B at B=1 (159.2 tok/s × the ratio; the class was never certified); (4) unlicensed arms are "measured, not
licensed", never a position; (5) bo3/bo5/bo6 numbers are cited beside, never divided into, bo7's. **All 50 arms ran: the 48
lane arms (`TP_DONE` 2026-09-05T07:00:26Z, 5.0 h of wall; Gemma-4-it loaded without the #344 fault) and the two arms of
P35 amendment 2 (pre-registered 06:05Z): `qwen3/calibexp_all_n128` at B=1 and B=16 — the licensed Qwen3 configuration
(streamed 64k pack) — run on the same box and install after `TP_DONE` (`TP2_DONE` 08:52:27Z); no alarm, refusal or
traceback on any of them.**

Licensed best per family, the three axes (tok/s = rental-measured on this box; the third axis is *no anchor projection*
for every family but Qwen3 and for every B=16 — no anchor-class measurement exists there):

- **Granite-3.1-3B-A800M** — `r12epi`, NF4 experts + round-1/2 folds (rotary-only fold) + router epilogue (licensed:
  +0.019 ppl wikitext, pass, `e4b.serve.buildout.granite.b1.5090.2026-09-04`): **×1.341 at B=1 (3.28 ms = 304.9 tok/s;
  NF4 4.40 ms = 227.3)** and **×1.160 at B=16 (1836.8; NF4 1583.2)** — `e4b.serve.census.bo7.granite.b1` / `.b16`.
  Beside, with their boxes: bo3 259.1 / 1689.6, bo5 294.1 / 1736.1 (both EPYC 9755 hosts).
- **OLMoE-1B-7B** — the position is **NF4** (×1.000: 3.54 ms = 282.5 tok/s B=1; 1347.5 B=16), because nothing above
  it is licensed on this register: the tp row's "best licensed" label predates the two-text clause's application to
  calibrated packs (#386), its calibrated attention is refused on this family (+0.60 ppl C4-val), and the folds alone
  have no K8 — `e4b.serve.census.bo7.olmoe.b1` / `.b16`.
- **gpt-oss-20b** — the quoted best is the lane's own reference arm, `nf4_r12` (NF4 experts + exact folds; no raw-text
  instrument): ×1.000, **6.92 ms = 144.5 tok/s B=1, 761.6 B=16** — `e4b.serve.census.bo7.gptoss.b1` / `.b16`.
- **Qwen3-30B-A3B** — the licensed configuration, the streamed **64k** calibrated pack + calibrated int4 attention +
  round-1/2 folds + router epilogue + glue (pass on both texts, bo6c,
  `e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`), measured on this box under amendment 2
  (`calibexp_all_n128`, `E4B_CALIB_NSEQ=128`, after `TP_DONE`): **×2.067 at B=1 (4.20 ms = 238.1 tok/s; NF4 8.68 ms =
  115.2) — anchor-class projection 159.2 × 2.067 ≈ 329 tok/s, a PROJECTION from a class never certified — and ×2.602 at
  B=16 (1327.5 tok/s; NF4 510.3; no anchor projection)** — `e4b.serve.census.bo7.qwen3.b1` / `.b16`. The lane's own
  calibrated arm had run the hook's 16k default (`calibexp_all`, ×2.067 / ×2.624, measured, not the licensed pack); the
  64k arm's speed is identical to it (4.20 vs 4.20 ms; 1327.5 vs 1338.8, within 1%) and to the RTN stack: a calibrated
  pack's kernels do not depend on the calibration size, as the amendment predicted.
- **Gemma-4-26B-A4B** — no K8 instrument exists for this family, so no arm carries a K8 licence; the register's position
  with that caveat is `r1epi` (the exact round-1 norm fold + router epilogue on NF4 experts): **×1.281 at B=1 (9.65 ms =
  103.6 tok/s; NF4 12.36 ms = 80.9)** and **×1.106 at B=16 (675.8; NF4 611.2)** — `e4b.serve.census.bo7.gemma4.b1` /
  `.b16`. The register's quoted best `int4_r1epi` (bo3's `stack`) reads ×1.705 / ×1.697 (137.9 / 1037.4) measured with no
  quality verdict — its claim text says "licensed configuration" and its notes say "no instrument"; the census reads the
  notes; beside, its box: bo3 121.1 / 962.8 (×1.69 / ×1.68). Calibrated attention on `r1epi` ×1.398 / ×1.120, no verdict.
- **Mixtral-8x7B-Instruct** — the position is **NF4** (×1.000: 19.87 ms = 50.3 tok/s B=1; 191.4 B=16;
  `e4b.serve.census.bo7.mixtral.b1` / `.b16`): `lic` and `all` FAIL as registered on bo5, the calibrated stack FAILS on
  bo6b and was dropped from this lane, and the combined folds arm has no receipt. Measured, not licensed: the exact
  folds ×1.062 / ×1.018, `lic` ×2.329 / ×1.959 (117.2 / 374.9), `all` ×2.597 / ×1.962 (130.7 / 375.5); beside, with their
  boxes: bo3 102.8 / 372.0 and 110.0 / 373.4, bo5 112.0 / 376.5 and 123.3 / 377.3.

Ratios on this box, every arm to its family's NF4 (B=1 / B=16); labels are the register's — no absolute is compared
across families:

| family | licensed best | exact folds (+ epilogue) | + calibrated int4 attention | int4-b32 experts (RTN) stack | calibrated int4 experts stack | other |
|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | `r12epi` ×1.341 / ×1.160 | = licensed best | — | `int4_r12epi` ×2.146 / ×2.094 (FAIL +0.063 ppl) | `calibexp_r12epi` ×2.157 / ×2.093 (FAIL c4val1 +0.387) | — |
| OLMoE-1B-7B | NF4 ×1.000 | `folds` ×1.192 / ×1.081 (no verdict on record) | `calattn` ×1.216 / ×1.077 (refused: +0.60 C4-val) | `int4all` ×2.070 / ×2.289 (not licensed: one text, calibrated pack) | — | — |
| gpt-oss-20b | `nf4_r12` ×1.000 (reference) | = reference | — | — | — | MXFP4 store, route rule `store_r12` ×1.293 / ×0.970 (gate open) |
| Qwen3-30B-A3B | `calibexp_all_n128` ×2.067 / ×2.602 (licensed: bo6c, both texts) | `folds` ×1.409 / ×1.130 (bo5: FAIL by improving, sub-floor) | `calattn` ×1.444 / ×1.101 (no verdict alone) | `int4all` ×2.067 / ×2.617 (FAIL c4val1 +0.063) | `calibexp_all` (16k) ×2.067 / ×2.624; `calibexp_folds` (16k, no calibrated attention) ×2.000 / ×2.628; `calibexp_all_n128` (64k, licensed) ×2.067 / ×2.602 | — |
| Gemma-4-26B-A4B | `r1epi` ×1.281 / ×1.106 (position with the no-instrument caveat, not a K8 licence) | = position | `calattn_r1epi` ×1.398 / ×1.120 (no verdict: unreadable K8) | `int4_r1epi` ×1.705 / ×1.697 (no verdict: no instrument; the register's quoted best) | — | — |
| Mixtral-8x7B-Instruct | NF4 ×1.000 | `folds` ×1.062 / ×1.018 (combined arm unscored) | `all` ×2.597 / ×1.962 (FAIL c4val1 +0.116) | `lic` ×2.329 / ×1.959 (FAIL c4val1 +0.0575; label withdrawn) | dropped (amendment 1) | — |

What the box adds to the register: a calibrated int4-expert pack costs nothing in speed over an RTN one (Granite 2.04 vs
2.05 ms, Qwen3 4.197 vs 4.204 unrounded; same kernel, same bytes, different values), and the streamed calibration's
pack counts reproduce across hosts (Granite 2524 gptq / 36 rtn = bo5's; Qwen3 10820 / 1468 = bo6b's and bo6c's). The
int4-expert lever reads ×1.71–2.60 at B=1 and ×1.70–2.63 at B=16 across the five families that ran it here and is
licensed on exactly one (Qwen3, with the 64k pack — amendment 2's arm, ×2.067 / ×2.602). The decode-attention compute path the kernel
tallied is `fp8` on OLMoE, Qwen3, Gemma-4 and Mixtral and `f32` on Granite and gpt-oss — chosen per family, identical
across a family's arms, as on bo5's receipts.

## Reading the numbers

- Quote a ratio or quote the card and the host. The 5090 class carries ~8.5% inter-box dispersion; B=1 moves with the host CPU.
- The K8 column is paged-vs-paged: it says what a lever costs relative to NF4 on the same path, read against that family's floor (`METHODOLOGY.md` §13.1). It is not absolute parity — that is `SERVING-PARITY.md`.
- A refused arm is not a zero and is not a failure of the model; it is the next item on the list.
