# bo7 — throughput census, results (2026-09-05, box 49916675, one RTX 5090 on an EPYC 7Q83 host)

The tables below are the verbatim output of `python census_reduce.py .` over the JSON receipts in this directory (box, cut, the amendments, protocol, the reading rules and the layout: [`README.md`](README.md)). **bo7 measures speed only; it licenses nothing.** The `licence (register)` column is the register as it stood when this bundle was written — `docs/claims.json` and the K8 verdicts on record from bo3, bo5, bo6 and bo6c — with the claim id that carries each verdict; where the register is silent the cell says `no quality verdict on record`. Every ratio is to the family's own NF4 arm on this box in this session; bo3 / bo5 / bo6 numbers are cited beside, never divided into, these. All 48 lane arms are in (`TP_DONE` 07:00:26Z); the only `pending` rows are amendment 2's two `calibexp_all_n128` arms (`logs/bo7b.sh`, running after `TP_DONE`), which arrive before merge.

### Granite-3.1-3B-A800M
Ratios are to `nf4` on this box (NF4 experts, bf16 attention, no folds (this lane's reference)). B=1 tok/s = 1000 / the timed step (ms to 0.01, the logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, not this lane's.
| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |
|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention, no folds (this lane's reference) | baseline (reference); paged decode at parity with the model's own attention, 0.00229 nats (`e4b.parity.granite.paged-vs-own-attention`) | — | 4.40 | 227.3 | 1.000 | 1583.2 | 1.000 | ok |
| `r12epi` | NF4 experts + round-1 + round-2 (rotary-only) folds + router epilogue | LICENSED -- K8 +0.019 ppl vs NF4 on wikitext, pass under the uncalibrated rule (bo3, `e4b.serve.buildout.granite.b1.5090.2026-09-04` / `.b16`; re-measured on bo5, `e4b.serve.buildout.bo5.granite.b1.5090.2026-09-04` / `.b16`) | — | 3.28 | 304.9 | 1.341 | 1836.8 | 1.160 | ok |
| `int4_r12epi` | RTN int4-b32 experts + round-1/2 folds + router epilogue | measured, not licensed -- RTN int4 experts FAIL as registered: +0.063 ppl on wikitext, over the 0.05 budget (bo3 `int4exp`, retracted in #381; bo5 `rtnexp_r12epi` +0.0634; `e4b.serve.tp.granite.b1.5090.2026-09-04` CORRECTION and `e4b.serve.buildout.granite.b1.5090.2026-09-04` notes) | INT4EXP 32 layers (granitemoe) | 2.05 | 487.8 | 2.146 | 3315.0 | 2.094 | ok |
| `calibexp_r12epi` | streamed GPTQ-calibrated int4 experts (16k C4 tokens) + round-1/2 folds + router epilogue | measured, not licensed -- calibrated int4 experts FAIL as registered on the second text: c4val1 +0.387 ppl (10x the 0.0033-nat floor), wikitext +0.014 (bo5; `e4b.serve.buildout.bo5.granite.b1.5090.2026-09-04` notes: refused) | streamed GPTQ 16k tok, 24 GiB budget, 1 passes, 2524 gptq / 36 rtn · INT4EXP 32 layers (granitemoe) | 2.04 | 490.2 | 2.157 | 3313.8 | 2.093 | ok |

Instrument (Granite-3.1-3B-A800M, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention compute path {"f32": 128, "fp8": 0}; NF4 GEMV dispatch `calibexp_r12epi` none (int4 / MXFP4 path), `int4_r12epi` none (int4 / MXFP4 path), `nf4` {'scalar': 256}, `r12epi` {'scalar': 256}.

### OLMoE-1B-7B
Ratios are to `nf4` on this box (NF4 experts, bf16 attention, no folds (this lane's reference)). B=1 tok/s = 1000 / the timed step (ms to 0.01, the logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, not this lane's.
| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |
|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention, no folds (this lane's reference) | baseline (reference); no paged-vs-own-attention parity receipt on this family | — | 3.54 | 282.5 | 1.000 | 1347.5 | 1.000 | ok |
| `folds` | NF4 experts + round-1/2 folds + router epilogue (exact arithmetic) | no quality verdict on record -- the folds + epilogue were never scored on this family (the tp lane scored nf4 / int4exp / calib only, `e4b.serve.tp.olmoe.b1.5090.2026-09-04` notes); exact arithmetic (moves no weight); not licensed as a position | — | 2.97 | 336.7 | 1.192 | 1457.2 | 1.081 | ok |
| `calattn` | NF4 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue | measured, not licensed -- calibrated int4 attention on OLMoE is refused on quality: +0.60 ppl on C4-val (private receipt; `e4b.serve.b1.qwen3-30b.int4attn-calib.5090` notes: Qwen3-specific, not a general lever); the tp lane's one wikitext reading (int4exp + calib 1.9295 vs NF4 1.9380) does not license a calibrated pack (the rule needs two texts) | ATTNINT4 64 projections | 2.91 | 343.6 | 1.216 | 1450.8 | 1.077 | ok |
| `int4all` | RTN int4-b32 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue | measured, not licensed under the rule as written -- `e4b.serve.tp.olmoe.b1.5090.2026-09-04` calls this stack 'best licensed' on ONE text (wikitext, 2048 steps: int4exp -0.029 ppl, + calib -0.058 vs NF4); its calibrated attention pack needs two texts and carries the +0.60 C4-val FAIL above, and int4-b32 experts at <=1B active are the class STATUS records at ~1.2-1.8% ppl (Granite's retraction, #381); no second text on record | INT4EXP 16 layers (olmoe) · ATTNINT4 64 projections | 1.71 | 584.8 | 2.070 | 3084.0 | 2.289 | ok |

Instrument (OLMoE-1B-7B, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention compute path {"f32": 0, "fp8": 64}; NF4 GEMV dispatch `calattn` {'scalar': 128}, `folds` {'scalar': 128}, `int4all` none (int4 / MXFP4 path), `nf4` {'scalar': 128}.

### gpt-oss-20b
Ratios are to `nf4_r12` on this box (NF4 experts + round-1/2 folds, bf16 attention (this lane's reference)). B=1 tok/s = 1000 / the timed step (ms to 0.01, the logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, not this lane's.
| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |
|---|---|---|---|---|---|---|---|---|---|
| `nf4_r12` | NF4 experts + round-1/2 folds, bf16 attention (this lane's reference) | the register's quoted best -- `e4b.serve.buildout.gptoss.b1.5090.2026-09-04` ('licensed configuration' in its claim text; its notes: no raw-text ppl instrument on this family, the folds' +0.049 nats cannot be read against the budget); exact arithmetic (norm/rotary folds); this lane's reference arm, ratio x1.000 by construction | — | 6.92 | 144.5 | 1.000 | 761.6 | 1.000 | ok |
| `store_r12` | native MXFP4 store: gemv_mxfp4_b32 for single rows, NF4 kept for batched rows (E4B_INT4_KEEP_NF4=1) + round-1/2 folds | measured, not licensed -- quality gate OPEN on this family (no instrument; the MXFP4 store is exact against the checkpoint's own bytes, bo3o/bo3q): `e4b.serve.buildout.bo5.gptoss.b1.5090.2026-09-04` / `.b16` quote it as speed with the gate open | INT4EXP 24 layers (gpt_oss) | 5.35 | 186.9 | 1.293 | 739.0 | 0.970 | ok |

Instrument (gpt-oss-20b, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention compute path {"f32": 96, "fp8": 0}; NF4 GEMV dispatch `nf4_r12` {'scalar': 192}, `store_r12` none (int4 / MXFP4 path).

### Qwen3-30B-A3B
Ratios are to `nf4` on this box (NF4 experts, bf16 attention, no folds (this lane's reference)). B=1 tok/s = 1000 / the timed step (ms to 0.01, the logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, not this lane's.
| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |
|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention, no folds (this lane's reference) | baseline (reference); paged decode at parity with the model's own attention, 0.00173 nats (`e4b.parity.qwen3.paged-vs-own-attention`) | — | 8.68 | 115.2 | 1.000 | 510.3 | 1.000 | ok |
| `folds` | NF4 experts + round-1/2 (rope-only) folds + router epilogue (exact arithmetic) | measured, not licensed as a position -- exact arithmetic (moves no weight); the verdict on record is bo5's: FAIL as registered on c4val1 by improving (-0.073 ppl, sub-floor; `e4b.serve.buildout.bo5.qwen3.b1.5090.2026-09-04` notes) | — | 6.16 | 162.3 | 1.409 | 576.5 | 1.130 | ok |
| `calattn` | NF4 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue | no quality verdict on record for calibrated attention on NF4 experts alone (the register scores it inside int4-expert stacks: bo5 `calib` = int4 experts + calibrated attention, c4val1 +0.048, one text); a calibrated pack needs two texts; not licensed | ATTNINT4 192 projections | 6.01 | 166.4 | 1.444 | 561.7 | 1.101 | ok |
| `int4all` | RTN int4-b32 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue + #385 glue (bo5's `all`) | measured, not licensed -- bo5's `all` (RTN experts): FAIL as registered on c4val1 +0.063 ppl (noise-attributed, not retuned; `e4b.serve.buildout.bo5.qwen3.b1.5090.2026-09-04` / `.b16`) | INT4EXP 48 layers (qwen3_moe) · ATTNINT4 192 projections | 4.20 | 238.1 | 2.067 | 1335.4 | 2.617 | ok |
| `calibexp_all` | streamed GPTQ-calibrated int4 experts at the hook's 16k default + C4-calibrated int4 attention + round-1/2 folds + router epilogue + glue | measured; the LICENSED configuration is the streamed 64k pack (bo6c, `e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`) -- this arm ran the hook's 16k default, and the 16k streamed full stack has no two-text verdict (bo6's two-text pass is the all-at-once 16k pack, `e4b.serve.buildout.bo6.qwen3.all-calibexp-allatonce.k8.2026-09-04`; the streamed 16k experts-only pack passed c4val1 only, `e4b.serve.buildout.bo6.qwen3.calibexp-streamed-16k.c4val1.2026-09-04`); the licensed stack's speed is amendment 2's `calibexp_all_n128` arm on this box (pending until its receipt lands) | streamed GPTQ 16k tok, 24 GiB budget, 5 passes, 10820 gptq / 1468 rtn · INT4EXP 48 layers (qwen3_moe) · ATTNINT4 192 projections | 4.20 | 238.1 | 2.067 | 1338.8 | 2.624 | ok |
| `calibexp_folds` | streamed GPTQ-calibrated int4 experts (16k) + round-1/2 folds + router epilogue, bf16 attention | measured, not licensed -- streamed 16k calibrated experts without calibrated attention: the 16k experts-only pack passed c4val1 (-0.0505) with wikitext unscored at 16k (`e4b.serve.buildout.bo6.qwen3.calibexp-streamed-16k.c4val1.2026-09-04`: pass on 1 text, needs 2); the 64k experts-only pack passes both texts (`e4b.serve.buildout.bo6c.qwen3.calibexp-streamed-64k.k8.2026-09-05`) but is not this arm's pack | streamed GPTQ 16k tok, 24 GiB budget, 5 passes, 10820 gptq / 1468 rtn · INT4EXP 48 layers (qwen3_moe) | 4.34 | 230.4 | 2.000 | 1340.8 | 2.628 | ok |
| `calibexp_all_n128` | THE LICENSED CONFIGURATION (bo6c): streamed GPTQ-calibrated int4 experts at 64k C4 tokens (E4B_CALIB_NSEQ=128) + C4-calibrated int4 attention + round-1/2 folds + router epilogue + glue -- run under P35 amendment 2 (pre-registered 06:05Z) after TP_DONE, same box, same install, `logs/bo7b.sh`, 5400-s alarms | the licensed configuration (bo6c) -- pass on both texts under the unchanged gate, at parity or better, no improvement by a number (`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`); this arm is its speed on this box (amendment 2) | — | — | — | — | — | — | B=1 pending; B=16 pending |

Instrument (Qwen3-30B-A3B, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention compute path {"f32": 0, "fp8": 192}; NF4 GEMV dispatch `calattn` {'dotpad': 384}, `calibexp_all` none (int4 / MXFP4 path), `calibexp_folds` none (int4 / MXFP4 path), `folds` {'dotpad': 384}, `int4all` none (int4 / MXFP4 path), `nf4` {'dotpad': 384}.

### Gemma-4-26B-A4B
Ratios are to `nf4` on this box (NF4 experts, bf16 attention, no folds (this lane's reference)). B=1 tok/s = 1000 / the timed step (ms to 0.01, the logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, not this lane's.
| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |
|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention, no folds (this lane's reference) | baseline (reference); no reference exists for this family at 0.1-nat / 512-token resolution (`e4b.parity.gemma4.no-reference`, #359) | — | 12.36 | 80.9 | 1.000 | 611.2 | 1.000 | ok |
| `r1epi` | NF4 experts + round-1 norm fold + router epilogue (round 2 refuses by design on this family; exact arithmetic) | exact arithmetic on NF4 experts (round-1 norm fold + router epilogue; the fold's kernel matches the module within 1 ULP per SERVING-THROUGHPUT) -- no K8 instrument exists for Gemma-4, so the register carries NO verdict for any Gemma-4 arm; quoted as the position that quantises nothing beyond NF4, with the register's caveat | — | 9.65 | 103.6 | 1.281 | 675.8 | 1.106 | ok |
| `int4_r1epi` | RTN int4-b32 experts + round-1 fold + router epilogue (bo3's `stack`) | measured, no quality verdict (no instrument) -- the register's quoted best for this family (bo3 `stack`; `e4b.serve.buildout.gemma4.b1.5090.2026-09-04` / `.b16`: 'licensed configuration' in the claim text, 'NO quality instrument on this family' in its notes and in SERVING-THROUGHPUT's gate column); not licensed by a verdict | INT4EXP 30 layers (gemma4_text) | 7.25 | 137.9 | 1.705 | 1037.4 | 1.697 | ok |
| `calattn_r1epi` | NF4 experts + C4-calibrated int4 attention + round-1 fold + router epilogue | measured, no quality verdict (no instrument) -- a calibrated pack with an unreadable K8 (bo3 `best`, +0.235 nats vs `stack` inside the family's +-0.1-0.27 band; `e4b.serve.buildout.gemma4.b1.5090.2026-09-04` notes) | ATTNINT4 115 projections | 8.84 | 113.1 | 1.398 | 684.4 | 1.120 | ok |

Instrument (Gemma-4-26B-A4B, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention compute path {"f32": 0, "fp8": 120}; NF4 GEMV dispatch `calattn_r1epi` {'scalar': 240}, `int4_r1epi` none (int4 / MXFP4 path), `nf4` {'scalar': 240}, `r1epi` {'scalar': 240}.

### Mixtral-8x7B-Instruct
Ratios are to `nf4` on this box (NF4 experts, bf16 attention, no folds (this lane's reference)). B=1 tok/s = 1000 / the timed step (ms to 0.01, the logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, not this lane's.
| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |
|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention, no folds (this lane's reference) | baseline (reference); no paged-vs-own-attention parity receipt on this family (the bf16 reference does not fit 32 GB) | — | 19.87 | 50.3 | 1.000 | 191.4 | 1.000 | ok |
| `folds` | NF4 experts + round-1/2 (rope-only) folds + router epilogue (exact arithmetic) | no quality verdict on record for the combined arm -- bo3 scored the pieces on wikitext, one text each: r12 +0.0009 ppl, epilogue -0.0087 ppl vs NF4 (bo3 receipts `mixtral_ppl_r12.json`, `mixtral_ppl_epi.json`; each within the uncalibrated 0.05 budget); exact arithmetic; not licensed as a position | — | 18.71 | 53.4 | 1.062 | 194.9 | 1.018 | ok |
| `lic` | RTN int4-b32 experts + round-1/2 folds + router epilogue (bo5's `lic`, the withdrawn 'licensed stack') | measured, not licensed -- RTN int4 experts + folds + epilogue: FAIL as registered on c4val1 +0.0575 ppl (by 0.008; floor unmeasured); the P30 'licensed stack' label is WITHDRAWN (`e4b.serve.buildout.bo5.mixtral.b1.5090.2026-09-04` / `.b16`) | INT4EXP 32 layers (mixtral) | 8.53 | 117.2 | 2.329 | 374.9 | 1.959 | ok |
| `all` | RTN int4-b32 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue (bo5's `all`) | measured, not licensed -- with calibrated int4 attention: FAIL as registered on c4val1 +0.116 ppl (`e4b.serve.buildout.mixtral.b1.5090.2026-09-04` / `.b16`, bo5 notes) | INT4EXP 32 layers (mixtral) · ATTNINT4 128 projections | 7.65 | 130.7 | 2.597 | 375.5 | 1.962 | ok |
| `calibexp_lic` | — | — | — | — | — | — | — | — | DROPPED under P35 amendment 1 before the box was rented: FAIL as registered on bo6b (wikitext +0.077 ppl, `e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`) and ~85 min of streamed calibration per arm inside a 10-h guard; not run, no number |

Instrument (Mixtral-8x7B-Instruct, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention compute path {"f32": 0, "fp8": 128}; NF4 GEMV dispatch `all` none (int4 / MXFP4 path), `folds` {'scalar': 128, 'scalar_splitk': 128}, `lic` none (int4 / MXFP4 path), `nf4` {'scalar': 128, 'scalar_splitk': 128}.

### Licensed best per family -- the three axes (rule 3)
Ratio = the family's own NF4 arm on this box; tok/s = rental-measured on this box (one RTX 5090, driver 595.84, on an AMD EPYC 7Q83 host, Vast.ai instance 49916675); the anchor-class projection is `159.2 tok/s x the B=1 ratio` and exists ONLY for Qwen3-30B-A3B at B=1 (the anchor class was never certified: 12 refusals) -- every other cell says so.
| family | licensed configuration | licence basis (register) | ×NF4 (B=1) | B=1 tok/s (this box) | anchor-class projection (B=1) | ×NF4 (B=16) | B=16 tok/s (this box) | anchor (B=16) |
|---|---|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | `r12epi` -- NF4 experts + round-1 + round-2 (rotary-only) folds + router epilogue | K8 verdict: +0.019 ppl wikitext, pass (`e4b.serve.buildout.granite.b1.5090.2026-09-04`) | 1.341 | 304.9 | no anchor projection (no anchor-class measurement exists for this family) | 1.160 | 1836.8 | no anchor projection (B=16 has no anchor-class measurement) |
| OLMoE-1B-7B | `nf4` -- NF4 experts, bf16 attention, no folds (this lane's reference) | nothing above NF4 is licensed on the register: the tp row's calibrated pack has one text (needs 2) and a +0.60 C4-val FAIL on record; the folds alone have no receipt | 1.000 (reference arm) | 282.5 | no anchor projection (no anchor-class measurement exists for this family) | 1.000 (reference arm) | 1347.5 | no anchor projection (B=16 has no anchor-class measurement) |
| gpt-oss-20b | `nf4_r12` -- NF4 experts + round-1/2 folds, bf16 attention (this lane's reference) | the register's quoted best (`e4b.serve.buildout.gptoss.b1.5090.2026-09-04`); exact folds on NF4; no instrument on this family -- and it is this lane's reference arm | 1.000 (reference arm) | 144.5 | no anchor projection (no anchor-class measurement exists for this family) | 1.000 (reference arm) | 761.6 | no anchor projection (B=16 has no anchor-class measurement) |
| Qwen3-30B-A3B | `calibexp_all_n128` -- THE LICENSED CONFIGURATION (bo6c): streamed GPTQ-calibrated int4 experts at 64k C4 tokens (E4B_CALIB_NSEQ=128) + C4-calibrated int4 attention + round-1/2 folds + router epilogue + glue -- run under P35 amendment 2 (pre-registered 06:05Z) after TP_DONE, same box, same install, `logs/bo7b.sh`, 5400-s alarms | K8 verdict: pass on both texts, wikitext -0.053 / c4val1 -0.066 ppl, both sub-floor (`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`); the arm runs under P35 amendment 2 after TP_DONE | pending (amendment 2) | pending | not computed until the licensed ratio is measured (rule: 159.2 x the B=1 ratio, a projection) | pending (amendment 2) | pending | no anchor projection (B=16 has no anchor-class measurement) |
| Qwen3-30B-A3B (nearest measured arm, `calibexp_all` -- the same stack with the 16k pack: measured, NOT the licensed pack) | streamed GPTQ-calibrated int4 experts at the hook's 16k default + C4-calibrated int4 attention + round-1/2 folds + router epilogue + glue | measured, not the licensed configuration | 2.067 | 238.1 | not quoted (an unlicensed arm gets no projection) | 2.624 | 1338.8 | no anchor projection (B=16) |
| Gemma-4-26B-A4B | `r1epi` -- NF4 experts + round-1 norm fold + router epilogue (round 2 refuses by design on this family; exact arithmetic) | exact arithmetic (round-1 fold + epilogue) on NF4 experts; no K8 instrument on this family -- no verdict exists, the register's caveat applies | 1.281 | 103.6 | no anchor projection (no anchor-class measurement exists for this family) | 1.106 | 675.8 | no anchor projection (B=16 has no anchor-class measurement) |
| Mixtral-8x7B-Instruct | `nf4` -- NF4 experts, bf16 attention, no folds (this lane's reference) | nothing above NF4 is licensed on the register: `lic` and `all` FAIL as registered (bo5), the calibrated stack FAILS (bo6b), the combined folds arm has no receipt | 1.000 (reference arm) | 50.3 | no anchor projection (no anchor-class measurement exists for this family) | 1.000 (reference arm) | 191.4 | no anchor projection (B=16 has no anchor-class measurement) |

### Cross-family summary -- ratios only (families differ in size; no absolute is compared across families)
| family | licensed best: ×NF4 B=1 / B=16 | fastest measured arm on this box: ×NF4 B=1 (arm) | ×NF4 B=16 (arm) | that arm's label |
|---|---|---|---|---|
| Granite-3.1-3B-A800M | 1.341 / 1.160 (`r12epi`) | 2.157 (`calibexp_r12epi`) | 2.094 (`int4_r12epi`) | measured, not licensed |
| OLMoE-1B-7B | 1.000 / 1.000 (`nf4`) | 2.070 (`int4all`) | 2.289 (`int4all`) | measured, not licensed under the rule as written |
| gpt-oss-20b | 1.000 / 1.000 (`nf4_r12`) | 1.293 (`store_r12`) | 1.000 (`nf4_r12`) | measured, not licensed |
| Qwen3-30B-A3B | pending (`calibexp_all_n128`) | 2.067 (`calibexp_all`) | 2.628 (`calibexp_folds`) | measured; the LICENSED configuration is the streamed 64k pack (bo6c, `e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`) |
| Gemma-4-26B-A4B | 1.281 / 1.106 (`r1epi`) | 1.705 (`int4_r1epi`) | 1.697 (`int4_r1epi`) | measured, no quality verdict (no instrument) |
| Mixtral-8x7B-Instruct | 1.000 / 1.000 (`nf4`) | 2.597 (`all`) | 1.962 (`all`) | measured, not licensed |

Every ratio above is to the family's own NF4 arm on this box in this session (rule 1). Labels are the register's (rule 2); `measured, not licensed` rows are speed of configurations the register does not license (rule 4) and are never quoted as a position. No number here is divided into a bo3 / bo5 / bo6 number (rule 5).
`pending` = an arm the lane script runs whose receipt is not in this snapshot yet (arrives before merge).

## Reading it

**All 48 lane arms are in** (`TP_DONE` 2026-09-05T07:00:26Z; no alarm, no refusal, no traceback). The only rows still
`pending` are the two arms of **P35 amendment 2** (pre-registered 06:05Z): `qwen3/calibexp_all_n128` at B=1 and B=16,
the licensed Qwen3 configuration, run on this box after `TP_DONE` by `logs/bo7b.sh` (same session, same install, 5400-s
alarms; started ≈ 07:01Z, `TP2_DONE` ≈ 09:00Z). Every one of the 48 receipts reproduces its console line
(`summary.txt` / `logs/outer.log`: step ms to 0.01, aggregate tok/s to 0.1, step counts) — no receipt contradicts its
console line. Every receipt carries `fuse_qkv: false` and `recompiles_in_window: 0`.

- **Granite-3.1-3B-A800M — the licensed stack `r12epi` (NF4 experts + round-1/2 folds with the rotary-only fold + router
  epilogue) is ×1.341 at B=1 (4.40 → 3.28 ms, 304.9 tok/s) and ×1.160 at B=16 (1583.2 → 1836.8 tok/s) over NF4 on this
  box.** The register's K8 verdict for that stack is +0.019 ppl on wikitext, pass (bo3). Beside it, with their boxes and
  never divided into these: bo3 259.1 / 1689.6 on instance 49817195 and bo5 294.1 / 1736.1 on 49841214, both EPYC 9755
  hosts. The two int4-expert arms are measured and refused: RTN `int4_r12epi` ×2.146 / ×2.094 (+0.063 ppl, FAIL;
  retracted in #381) and calibrated `calibexp_r12epi` ×2.157 / ×2.093 (c4val1 +0.387 ppl, FAIL, bo5). The calibrated
  pack is speed-identical to RTN (2.04 vs 2.05 ms; 3313.8 vs 3315.0 tok/s), as it must be — same kernel, same bytes,
  different values. Instrument: the streamed calibration on this box packed 2524 gptq / 36 rtn, the counts bo5's
  all-at-once run reported on its box.
- **OLMoE-1B-7B — nothing above NF4 is licensed on the register, so the position is NF4: 282.5 tok/s B=1 (3.54 ms),
  1347.5 B=16.** The exact folds + epilogue read ×1.192 / ×1.081 with no K8 on record for this family (the tp lane scored
  nf4 / int4exp / calib only); calibrated int4 attention on top of the folds adds ×1.021 at B=1 (2.97 → 2.91 ms) and
  nothing at B=16 (1457.2 → 1450.8) and is refused on quality (+0.60 ppl on C4-val, `int4attn-calib` notes); the full
  int4 stack `int4all` reads ×2.070 / ×2.289 — measured, not licensed under the rule as written: the tp claim
  (`e4b.serve.tp.olmoe.b1.5090.2026-09-04`) called it "best licensed" on one wikitext text before the two-text clause
  was applied to calibrated packs (#386), and int4-b32 experts at ≤1B active are the class STATUS records at ~1.2–1.8% ppl.
  Beside, its box only: the tp lane's Ryzen 9 9900X host read 247.7 → 452.5 B=1, 1294.0 → 2412.5 B=16 for NF4 → that stack.
- **gpt-oss-20b — this lane's reference arm is `nf4_r12` (NF4 experts + the exact round-1/2 folds; the register's quoted
  best, no instrument on this family): 144.5 tok/s B=1 (6.92 ms), 761.6 B=16.** The native MXFP4 store under the route
  rule (`store_r12`: `gemv_mxfp4_b32` for single rows, NF4 kept for batched rows) reads **×1.293 at B=1 (186.9 tok/s)
  and ×0.970 at B=16 (739.0)** — the same shape bo5 measured on its EPYC 9755 box (×1.270 / ×0.971, 173.3 / 719.7;
  cited beside, not divided). Its quality gate stays open (exact bytes, no raw-text instrument): measured, not licensed.
- **Qwen3-30B-A3B — the licensed stack's speed is pending on this box (amendment 2).** The licensed configuration is
  bo6c's streamed **64k**-token calibrated stack (`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`);
  bo7's `calibexp_all` ran the hook's 16k default (`E4B_CALIB_NSEQ` unset), whose streamed full stack has no two-text
  verdict, and amendment 2 adds the 64k arm (`calibexp_all_n128`, `E4B_CALIB_NSEQ=128`) at B=1 and B=16 after
  `TP_DONE` — `pending` in this snapshot; when it lands the licensed-best row carries its three axes for real, ratio
  to this box's own `nf4` arms. What this box measures so far: NF4 8.68 ms = 115.2 tok/s B=1, 510.3 B=16; the exact folds ×1.409 / ×1.130 (bo5's
  verdict on record: FAIL as registered on c4val1 by improving, −0.073 ppl, sub-floor); calibrated attention on top of
  the folds ×1.025 at B=1 (6.16 → 6.01 ms) and ×0.974 at B=16; RTN `int4all` (bo5's `all`) ×2.067 / ×2.617 — FAIL on
  c4val1 +0.063, measured, not licensed; the 16k streamed calibrated stack `calibexp_all` **×2.067 / ×2.624 (4.20 ms
  = 238.1 tok/s B=1; 1338.8 B=16)** — speed-identical to the RTN stack (4.197 vs 4.204 ms unrounded; 1338.8 vs 1335.4),
  the calibrated pack changes the values in the int4 store and nothing about the kernel or the bytes; and the same
  calibrated experts without calibrated attention (`calibexp_folds`) 4.34 ms = ×2.000 at B=1 and 1340.8 tok/s = ×2.628
  at B=16, so calibrated int4 attention is worth ×1.033 on top of calibrated experts + folds at B=1 on this box and
  nothing at B=16 (1340.8 → 1338.8, ×0.999) — int4 attention is a B=1 lever only, as on Qwen3's earlier campaign. The 64k pack differs
  from the 16k one only in the packed values (11512 gptq / 776 rtn against 10820 / 1468 on bo6b/bo6c), not in the kernel
  or the bytes read, so the *expectation* is that the licensed stack's speed equals the 16k arm's — an expectation, not a
  measurement; amendment 2's arm measures it, and until it lands no ratio, tok/s or projection is quoted for the
  licensed stack. Beside, each with its box and never divided into these: bo5's RTN `all` 204.1 / 1251.6 on an EPYC 9755 host
  (49841214), bo6's all-at-once calibrated stack 158.0 / 993.6 on a Threadripper PRO 7975WX host (49861751), bo3's
  177.9 / 1089.6 on 49817195 (EPYC 9755, without #385's glue). Instrument: the 16k streamed calibration on this EPYC
  7Q83 box packed 10820 gptq / 1468 rtn on both calibrated arms — the counts bo6b and bo6c reported on their Threadripper
  box — so the calibration set and the `min_rows` fallbacks are host-independent at the count level.
- **Gemma-4-26B-A4B — no K8 instrument exists for this family, so no arm carries a K8 licence; the register's position
  with that caveat is `r1epi` (exact arithmetic: round-1 norm fold + router epilogue on NF4 experts): ×1.281 at B=1
  (12.36 → 9.65 ms, 103.6 tok/s) and ×1.106 at B=16 (611.2 → 675.8 tok/s).** Gemma-4-it loaded on this host without the
  #344 fault (bake 05:44–05:57Z). The register's quoted best, `int4_r1epi` (bo3's `stack`: RTN int4 experts + round-1 fold
  + epilogue), reads ×1.705 at B=1 (7.25 ms = 137.9 tok/s) and ×1.697 at B=16 (1037.4) — measured, no quality verdict
  (its claim text says "licensed configuration", its notes "NO quality instrument"; the census reads the notes); beside,
  its box only: bo3's 121.1 / 962.8 (×1.69 / ×1.68) on instance 49817195 — the same shape, cited, not divided.
  Calibrated int4 attention on top of `r1epi` (`calattn_r1epi`, 115 projections) reads ×1.398 / ×1.120 — ×1.092 at B=1
  over `r1epi` and ×1.013 at B=16 — measured, no quality verdict (an unreadable K8 on this family). Round 2 refuses by
  design on this family and was not run. Instrument: decode-attention compute `fp8` (120), NF4 GEMV dispatch scalar (240).
- **Mixtral-8x7B-Instruct — nothing above NF4 is licensed on the register, so the position is NF4: 50.3 tok/s at B=1
  (19.87 ms), 191.4 at B=16.** The exact folds + epilogue (`folds`, r1 + r2 rope-only + epilogue on NF4 experts) read
  ×1.062 at B=1 (18.71 ms, 53.4 tok/s) and ×1.018 at B=16 (194.9) — the same ×1.062 the rope-only fold bought on bo5's
  calibrated stack (cited beside, not divided); no receipt scores the combined arm (bo3 scored r12 +0.0009 and the
  epilogue −0.0087 ppl separately, one text each), so it is not a licensed position. The RTN int4-expert stack `lic`
  (bo5's withdrawn "licensed stack") reads ×2.329 at B=1 (8.53 ms = 117.2 tok/s) and ×1.959 at B=16 (374.9); with
  calibrated int4 attention (`all`, 128 projections) ×2.597 / ×1.962 (7.65 ms = 130.7; 375.5) — both measured, not
  licensed (c4val1 +0.0575 and +0.116 ppl, FAIL as registered on bo5). Calibrated attention is worth ×1.115 on top of
  `lic` at B=1 and ×1.002 at B=16 here (bo5: ×1.101 / ×1.002 on its box). Beside, with their boxes: bo3 `stack` 102.8 /
  372.0 and `all` 110.0 / 373.4 (49817195); bo5 `lic` 112.0 / 376.5 and `all` 123.3 / 377.3 (49841214). Mixtral's
  `calibexp_lic` was dropped under amendment 1 before the box was rented (FAIL as registered on bo6b; ~85-min streamed
  calibration per arm) and is a row that says so. Instrument: decode-attention compute `fp8` (128); this is the one
  family whose NF4 GEMV dispatch tally shows split-K beside scalar (128 + 128) — the 8-expert, 14336-wide shape.
- **Cross-family, ratios only.** NF4 → the register's position on this box: Granite ×1.341 / ×1.160 (licensed by K8);
  OLMoE ×1.000 (NF4 is the position); gpt-oss ×1.000 (the reference arm is the quoted best); Qwen3 pending amendment 2
  (its nearest measured arm, the same stack with the 16k pack, ×2.067 / ×2.624); Gemma-4 ×1.281 / ×1.106 (exact
  arithmetic, no instrument — a position with a caveat, not a K8 licence); Mixtral ×1.000 (NF4 is the position; the
  exact folds ×1.062 / ×1.018 are unscored as a combined arm). The int4-expert lever itself reads ×1.71–2.60 at B=1 and
  ×1.70–2.63 at B=16 across the five families that ran it here and is licensed on exactly one of them (Qwen3, with the
  64k streamed pack — amendment 2's arm).
- **Anchor axis.** An anchor-class projection exists only for Qwen3-30B at B=1 (159.2 tok/s × the ratio; the class was
  never certified, 12 refusals). It is not computed in this snapshot: the only arm it applies to — the licensed stack,
  amendment 2's `calibexp_all_n128` — has no receipt yet, and an unlicensed arm gets no projection. The reducer prints it,
  marked as a projection, the moment that arm's B=1 receipt lands. Every other family and B=16: no anchor projection.

## Instrument notes

- **Console lines match receipts on all 31 arms.** Checked mechanically: each JSON's `step_ms_clean` rounds to the
  `step=` of its `B1D_TIMED_GRAPH` / `BV3_GRAPH` line, each B=16 `aggregate_tok_s` rounds to its `agg=`, step counts
  match (127 timed B=1 steps, 70 B=16 steps). `summary.txt` holds the 31 result lines in lane order.
- **The decode-attention compute path differs BY FAMILY, not by arm:** the receipts' `mech.compute` tally reads `fp8`
  on OLMoE (64) and Qwen3 (192) and `f32` on Granite (128) and gpt-oss (96) — the same per-family selection bo5's and the
  tp lane's receipts carry (an unset `GNF4_ATTN_COMPUTE` selects capability-conditionally). It is identical across every
  arm of a family, so no ratio on this page sees it; it is why "fp8 paged KV" in the protocol line describes the cache,
  and the kernel's own tally describes the compute. The NF4 GEMV dispatch tally likewise differs by family (dot-pad on
  Qwen3's shape, scalar on the others) and is zero on int4 / MXFP4 arms, which take other kernels.
- **Fusion arms print no banner.** The `E4B_FUSE_*` fusions refuse aloud when nothing structurally matches (#333) and
  are silent when they engage; the run logs of `r12epi` / `folds` / `nf4_r12` carry no lane banner, and their engagement
  is read from the step time (Granite 4.40 → 3.28 ms, Qwen3 8.68 → 6.16 ms). The int4 and calibrated arms print theirs
  (`INT4EXP enabled`, `ATTNINT4 calibrated`, the streamed-calibration pass lines), reproduced in the `engaged` column.
- **The alarms were never reached.** The calibrated arms ran under a 5400-s alarm (amendment 1; Qwen3's streamed 16k
  calibration takes ~40 min per arm on this host: 02:51 → 03:34Z, 03:34 → 04:13Z, 04:24 → 05:05Z), the others under
  3600 s; no `Alarm clock`, no `REFUSED`, no `Traceback` in any log of this snapshot.
- **Speed only.** No K8 arm ran on this lane; every verdict in the `licence` column is copied from the register.
- **Amendment 2 (pre-registered 06:05Z, before its arms ran).** Two arms on this box after `TP_DONE`, same session and
  install: `qwen3/calibexp_all_n128` at B=1 and B=16 — the licensed Qwen3 configuration (streamed 64k pack + calibrated
  attention + folds + epilogue + glue), `logs/bo7b.sh`, 5400-s alarms, marker `TP2_DONE`. Their receipts, or their
  `alarm` / `refused` rows, arrive with the final snapshot.
