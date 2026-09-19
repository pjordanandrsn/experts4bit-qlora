# Results — P44: serving quality instruments (2026-09-19)

Pre-registration: [`P44-PREREG.md`](P44-PREREG.md) (2026-09-19, amendments 1–5 dated before the data they touch). Every number below is read from a committed receipt by `bench/p44/p44_reduce.py`; the reducer's output is the table, this file adds the sentences. Runs: `p44-a-olmoe-2` (RTX 5090; the six OLMoE K8 rows), `p44-a-census` (RTX 5090; Granite complete, Mixtral cut by its alarm at 16 of 32 layers), `p44-b-kl80-5` (H100 NVL; the KL rows under the scorer control (i) admits). Runs 1–4 of P44-b and run 1 of P44-a were harness faults, each receipted and each fixed in an amendment (hook armed at import; gpt-oss dequant check; prefused stacks; census OOM; staged-hash re-pin; scorer control).

## P44-a — OLMoE two-text K8 (budget 0.05 ppl, 2048 steps, `k8_gate.verdict`)

| arm | wikitext ppl (Δ vs nf4) | c4val1 ppl (Δ vs nf4) | rule | verdict |
|---|---|---|---|---|
| `nf4` (control, the licensed position) | 6.9118 | 18.9060 | — | — |
| `int4all` (RTN int4 experts + C4-calibrated int4 attention + folds + epilogue) | 6.8703 (-0.0414) | 19.1606 (**+0.2546**) | two-sided ≤ 0.05 | **FAIL** |
| `calibexp_all` (streamed 64k sequential GPTQ experts calibrated on wikitext-train + int4 attention + folds + epilogue) | 6.8567 (-0.0550) | 19.3487 (**+0.4427**) | one-sided ≤ +0.05, calibration domain wikitext | **FAIL** |

- **P1 (int4all FAILS c4val1) — HOLDS.** RTN int4 experts fail the second text on OLMoE (+0.255) as on Qwen3 (+0.063) and Granite (+0.063); wikitext alone (−0.041) would have passed — the two-text clause is what catches it.
- **P2 (calibexp_all PASSES both texts) — REFUTED.** The Qwen3 recipe does not transfer: calibrating on wikitext-train reads −0.055 on wikitext (same-domain, not trusted by the rule) and **+0.443 on c4val1 — worse than RTN's +0.255**. Fitting the calibration text is what the OOD-flattery clause exists to catch, and here it shows in the other direction. **OLMoE's quoted position stays NF4**; its measured speed levers (×2.07 / ×2.29, bo7) stay unlicensed.

## P44-a — per-expert residual census (P3: ≤ 10 % of experts carry ≥ 50 % of the summed activation-weighted reconstruction error)

| family | method read | layers | routed experts | experts for 50 % | fraction | top-10 % share | verdict |
|---|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M (prefused stacks) | recipe (GPTQ where ≥ 32 rows) | 32 / 32 | 1270 | 126 | **0.0992** | 0.5027 | **P3 HOLDS — by 0.08 points** |
| Mixtral-8x7B (per-expert projections) | RTN (its timed arm) | **16 / 32 (cut by the arm alarm)** | 128 | 29 | 0.2266 | 0.2485 | **P3 REFUTED on the half measured** |

- Granite: 126 of 1,270 routed experts (9.92 %) carry 50.3 % of the summed GPTQ error — the registered threshold is 10 %, so P3 holds, marginally; a per-expert NF4 fallback (#530's third method) has a premise on this family, thin as it is. GPTQ halves the median per-expert residual against RTN (rel_act 0.088 → 0.049).
- Mixtral: on 16 of 32 layers the RTN error is spread evenly — 29 of 128 experts (22.7 %) for 50 %, the top 10 % carry 24.9 % — the distribution is flat, not heavy-tailed. The census was cut by the arm alarm (≈ 3 min per layer for 8 × 14336² Hessians and their GPTQ solves on a 32 GB card); the half measured refutes P3 on this family and the other half would have to be extraordinary to reverse it. Recorded as partial; no redraw is scheduled unless a per-expert fallback is pursued for Mixtral, which this row argues against.

## P44-b — KL from the family's reference (reading rule: every stratum ≤ 1.10 × NF4 and pooled Δ ≤ 0.005 nats/token)

**Scorer per family, chosen by control (i)** (the bf16 reference against itself, decode-shaped vs prefill-shaped, first 40 prompts): gpt-oss self-KL **0.0005** → decode admitted; Gemma-4 self-KL **0.279** → decode REFUSED, prefill on both sides (amendment 5). K0 controls passed on the box.

### gpt-oss-20b (reference = dequant-to-bf16 of the SAME shipped MXFP4 bytes; decode-shaped both sides)

| arm | KL nats/token | top-1 | general / technical / code / longctx | verdict |
|---|---|---|---|---|
| `nf4_r12` (NF4 requant control + folds) | 0.0222 | 0.9366 | 0.0236 / 0.0156 / 0.0114 / 0.0287 | control |
| `store_r12` (native MXFP4 store for single rows + folds) | **0.0019** | 0.9814 | 0.0016 / 0.0014 / 0.0014 / 0.0024 | **LICENSES (`kl-vs-bf16`)** — every stratum ≤ 0.13× the control |

- **P6 (store at the instrument floor < 1e-3) — not met literally** (0.0019), **the reading rule licenses** it by an order of magnitude. This is the first quality verdict on gpt-oss that is not OOD flattery; the register row carries `licensed_by: kl-vs-bf16`, distinct from a K8 licence. Reproduced to four digits between runs 3 and 5.

### Gemma-4-26B-A4B-it (reference = the bf16 checkpoint; prefill-shaped both sides)

| arm | KL nats/token | top-1 | general / technical / code / longctx | verdict |
|---|---|---|---|---|
| `nf4` (control) | **1.0770** | 0.6380 | 1.032 / 1.234 / 1.388 / 0.868 | control — **itself a defect (#597)** |
| `r1epi` | 1.1267 | 0.6358 | 1.079 / 1.585 / 1.417 / 0.868 | P4 NOT READ (prefill) |
| `int4_r1epi` | 1.1560 | 0.6328 | 0.967 / 1.635 / 1.445 / 0.929 | does not license (pooled Δ +0.079; technical ×1.33) |
| `calattn_r1epi` | 1.2782 | 0.6018 | 1.189 / 1.615 / 1.608 / 1.032 | does not license (every stratum over) |

- **P5 (int4_r1epi licenses) — REFUTED** under the rule, but the rule is not the finding here. **The NF4 control itself sits 1.08 nats/token from the bf16 checkpoint with 36 % top-1 disagreement**, where the same instrument reads gpt-oss NF4 at 0.022 and (`KL-FINDINGS.md`) OLMoE / Granite / Mixtral NF4 at 0.02–0.1. Lever-independent, reproduced to four digits across runs 2, 3 and 5, and now under a scorer the control admits. **This is a served-model defect on the Gemma-4 family, filed as #597** with a diagnostic plan (training-loader model vs served stack vs unquantised-expert e4b model) to pre-register as P47. It is the same symptom the family's K8 "chaos band" (#359) and training parity failure (#558) showed from inside e4b; this is the first measurement against the checkpoint. **No Gemma-4 serving position is quoted** — none was.
- **P4 (r1epi = nf4 to < 1e-4) — NOT READ**: under prefill the round-1 fold does not engage; and r1epi differs from nf4 by +0.05 nats here where it should not differ at all — a second item under #597.
- Mixtral KL: NOT_RUN by registration (93 GB bf16 does not fit an 80 GB card).

## What moves in the register

- `e4b.serve.p44.gptoss.store-r12.kl-vs-bf16.2026-09-19` — **measured, licensed_by kl-vs-bf16**: gpt-oss's `store_r12` position (bo7 ×1.29 at B=1) is licensed.
- `e4b.serve.p44.olmoe.two-text-k8.2026-09-19` — measured: `int4all` and `calibexp_all` FAIL the second text; OLMoE stays NF4.
- `e4b.serve.p44.census.granite-mixtral.2026-09-19` — measured: P3 holds on Granite (9.92 %), refuted on Mixtral (16/32 layers, 22.7 %).
- `e4b.serve.p44.gemma4.nf4-vs-bf16.2026-09-19` — measured: the served NF4 control is 1.08 nats from bf16 (defect #597); no licence, no position.
- Instrument findings carried in `KL-FINDINGS.md` terms: the decode scorer must pass the reference self-consistency control per family (Gemma-4 fails it at 0.279 nats).

## Cost

P44-a: run 1 (controller refusal) ≈ $0.01; run 2 ≈ 74 min ≈ $0.80; census redraw ≈ 108 min ≈ $1.17. P44-b: runs 2–5 on the H100 NVL ≈ 24 + 61 + 2 + 45 min ≈ $6.8. Lane total ≈ $8.8 against a registered $11.90 (2.60 + 9.30). Every rental torn down with proof.
