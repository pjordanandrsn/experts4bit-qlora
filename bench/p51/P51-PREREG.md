# P51 — A GRADED STORE MAP FOR GEMMA-4: keep the sensitive layers high-precision, crush the rest (pre-registered 2026-09-19 ~12:10Z, before any box is rented)

Work item: adertha-agents#110; [#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597) (closed as explained). **Owner directive, 2026-09-19:** *"gemma: Keep the first N expert layers high precision, crush the rest."* P50 recommended shipping no Gemma-4 default; the owner's call is to ship the hybrid, so this lane picks its shape with numbers rather than assuming one.

Lineage, all measured this morning on the same instrument: e4b's Gemma-4 modelling is faithful (P48: 0.0056 nats with 29 of 30 expert stacks bf16); per-layer sensitivity spans **159×** (P48: NF4 on layer 0 alone 0.892, on layer 27 alone 0.0056); **no store rescues layer 0** (P49: int8 0.693, fp8 0.774, NF4 0.892, FP4 1.051) because the sensitivity is positional, not magnitudinal; and a uniform bf16 head reaches the 0.05 fidelity floor only at N = 20, costing 32.35 GB of expert store against 12.0 GB all-NF4 and 42.5 GB all-bf16 (P50).

**What those rows leave open is the shape of the head.** P50's head is uniform bf16 across 20 layers, but the layers inside it are not uniform: layer 0 is 0.892 and layer 19 is 0.017 on the same store. A **graded** map — bf16 only where it pays, int8 in the middle, NF4 at the end — is what the directive asks for and what nothing has measured. The mechanism for it landed with this lane: `load_moe_4bit_streaming(quantize_layers={layer: spec})` (a per-layer store map, `layer_store_spec`), refused on the arena and dedicated-quant paths, with tests.

## Instrument and arms

`bench/p44/kl_serve.py` (prefill scorer by control (i), the same cached bf16 reference and 200 committed prompts as P44-b / P47 / P48 / P49 / P50) on family **`gemma4mix`**, builder `loader_tiers_<store>:<count>_...` (counts sum to 30; `bf16` means the base dtype, so a tier spec is exactly a `quantize_layers` mapping):

| arm | tiers (layers 0→29) | role |
|---|---|---|
| `bf16_20` | `bf16:20_nf4:10` | **the anchor — byte-for-byte P50's `loader_keep_20` (0.0469, 32.35 GB)** |
| `int8_20` | `int8:20_nf4:10` | the uniform int8 head: half the head's bytes, and the cheap falsification of "just use int8" |
| `graded_10_10` | `bf16:10_int8:10_nf4:10` | **the candidate** |
| `graded_5_15` | `bf16:5_int8:15_nf4:10` | a shorter bf16 head |
| `graded_10_10_crush` | `bf16:10_int8:10_nf4b256:10` | the candidate with the tail crushed harder |

Proof of execution per row: `verify_moe_4bit`'s per-stack store and block must equal the tier census the builder names (`kl_serve._builder_check`), so a map that silently collapsed to one store refuses instead of producing a wrong row. Every row also carries the expert-bytes census P50 added.

## Registered predictions (falsifiable; reducer `bench/p51/p51_reduce.py`)

- **M1 (the anchor):** `bf16_20` ∈ [0.040, 0.055] — P50's 0.0469 rebuilt through the map machinery. Refuted → the map does not build what keep-k built and **nothing else in this lane is read**.
- **M2 (a uniform int8 head fails):** `int8_20` **≥ 0.50**. P49 measured int8 on layer 0 *alone* at 0.693 and P50's curve is monotone in how much is quantised, so a head containing layer 0 at int8 cannot do better. Refuted below 0.30 → single-layer rows do not compose and the head question reopens with its own lane.
- **M3 (grading works):** `graded_10_10` **≤ 0.10** — int8 on layers 10..19 costs little because those layers are 5–50× less sensitive than layer 0. Refuted at **≥ 0.30**; in between, reported and not shipped.
- **M4 (grading saves bytes):** `graded_10_10`'s expert store ≤ **0.85×** the anchor's.
- **M5 (the tail is not where the bytes are):** `graded_10_10_crush`'s store is within **5 %** of `graded_10_10`'s and its KL within **1.3×** — i.e. "crush the rest" is the cheap half of the directive and the head is where the memory lives. Refuted → crushing does pay and the tail store belongs in the default.

## Decision rules

- **M1 ∧ M3 ∧ M4 → the shipped Gemma-4 default is the graded map** with these boundaries, behind its own K8 + KL gate (a separate lane, because K8 is a serving-stack instrument), with the curve and the byte cost quoted beside it.
- M3 refuted → the default is P50's uniform bf16 head at an N the user picks; grading does not pay and the write-up says so.
- M5 refuted → the tail store joins the default and is named in it.
- M2 refuted → reopen the head question before any default ships.
- Until a default passes its gate, **no Gemma-4 position is quoted**. This lane ships a mechanism (the per-layer store map) and a recommendation, not a number.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p51-gemma4mix` | H100 NVL (≥ 80 GB) | 3.10 | 1.0 h | $3.10 |

Timing basis (P50): fetch ~7 min, K0 < 2 min, reference pass 40 s, five loader arms ~90–150 s ≈ 12 min; ≈ 25 min. STOP: not the class (15); egress < 20 MB/s (14); free disk < 120 GB (13); K0 failing (16); a fetch alarm → not_run; a refused tier row (the census disagreeing with the map) is an honest hole and the other arms still run; a second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p51-gemma4mix/p47/` — `gemma4mix_kl.json` (5 rows with their tier census and expert bytes), `k0.json`, `versions.txt`, `forensics.txt`, `logs/`; read by `bench/p51/p51_reduce.py`. Results → `RESULTS-p51.md`.

## Amendments

(none yet)

### Amendment 1 (2026-09-19 ~12:40Z, after run 1 — one arm read, four refused by a bug of this lane's) — the expected-stack count

Run `p51-gemma4mix` produced **one** row: `int8_20` at **0.7369** (M2's prediction was ≥ 0.50, and this is it). The four arms with a `bf16` tier were refused by `_builder_check` — correctly, given what it was told: `build_loader_model` computed `expected_quantized` as `len(quantize_layers)`, which is right for a SET but wrong for a MAPPING, because a map names every layer it controls and a `None` spec means "base dtype". The models were built correctly (`bf16:20_nf4:10` reported 10 quantised stacks, exactly right); the expectation was wrong, so the guard refused a correct row. Fixed by `n_expected_quantized()`, with a test that asserts the PRODUCER over every registered tier spec, not just the checker — the original test hand-built the expectation and so could not catch this.

Predictions and thresholds are unchanged. The redraw `p51-gemma4mix-2` re-runs all five arms (the `int8_20` row is cheap to repeat and a self-consistent receipt is worth more than saving 90 s). M2 is therefore read on run 1 AND re-read on the redraw; if the two disagree the lane says so.

### Amendment 2 (2026-09-19 ~13:05Z, after the redraw — two answers and one matched-bytes arm) — M5 is unmeasurable on this model; M3 needs a matched control

The redraw `p51-gemma4mix-2` read four of five arms (`RESULTS-p51.md`):

| arm | KL | expert store |
|---|---|---|
| `bf16_20` (anchor) | **0.0469** | 32.35 GB |
| `graded_10_10` | **0.1695** | 25.70 GB |
| `graded_5_15` | 0.4773 | 22.38 GB |
| `int8_20` | 0.7369 | 19.05 GB |

**M1 HOLDS** (the anchor reproduces P50's keep-20 to four digits and to the byte). **M2 HOLDS** (a uniform int8 head fails, 0.737). **M4 HOLDS** (the graded map saves 6.65 GB, 0.79× the anchor). **M3 is INCONCLUSIVE**: 0.1695 sits between the registered 0.10 (ships) and 0.30 (does not).

**M5 is RETIRED as unmeasurable on this model, not refuted.** `graded_10_10_crush` asked for NF4 at block 256 and the stack refused: Gemma-4's `moe_intermediate_size` is **704 = 64 × 11**, so no block larger than 64 divides it. NF4 at block 64 is already the smallest store e4b ships here, so **the tail in every one of these configurations is already crushed as far as this architecture allows** — the "crush the rest" half of the design has no remaining freedom on Gemma-4, and the question is entirely where the high-precision boundary goes.

**What M3's inconclusive reading leaves open is the only question that decides the default**: does grading beat a plain uniform head *at the same bytes*? The lane has one nearly-matched pair already, and it favours uniform: `graded_5_15` is **22.38 GB / 0.4773** against P50's `keep_10` at **22.16 GB / 0.4618** — fewer bytes AND better. At `graded_10_10`'s 25.70 GB the uniform curve has no measured point, and interpolating a convex curve is not a measurement.

So one arm is added and the family re-run on one box: **`bf16_13`** (`bf16:13_nf4:17`, ≈ 25.2 GB), the matched-bytes uniform control for `graded_10_10`. Registered prediction, **M6**: `bf16_13` ≤ `graded_10_10`'s KL — i.e. grading is dominated and the default is the uniform head at an N the user picks. Refuted if `bf16_13` > 0.1695 by more than 10 %, which would say grading genuinely buys quality per byte and belongs in the default. `p51-gemma4mix-3`, H100 NVL, 1 h, ≤ $3.10.

## Read (2026-09-19, runs p51-gemma4mix-2 and -3)

`RESULTS-p51.md`: M1 HOLDS (the anchor rebuilds P50's keep-20 to four digits and to the byte), M2 HOLDS (a uniform int8 head fails at 0.7369 — the head must be bf16), M3 INCONCLUSIVE, M4 HOLDS (6.65 GB saved), M5 RETIRED as architecturally unmeasurable (704 = 64 × 11), **M6 REFUTED — and that is the answer: at matched bytes the graded map is 1.44× better than a uniform head** (0.2448 @ 25.21 GB vs 0.1695 @ 25.70 GB). **Decision: ship the graded map** (bf16 first ~10, int8 next ~10, NF4 rest) as the recommended Gemma-4 shape, behind the registered K8 gate before any position is quoted.

### Amendment 3 (2026-09-19 ~14:30Z, after the read) — the gate I registered cannot be built on this family, and what replaces it

P50 and P51 both say the graded map ships "behind its own K8 two-text gate (a separate lane, because K8 is a serving-stack instrument)". **That gate cannot be built on Gemma-4, for two independent reasons, and I registered it without checking either.**

1. **K8 is unreadable on this family.** `e4b.parity.gemma4.no-reference` (measured, in the register) records that Gemma-4's own NLL on identical prefix tokens moves by **0.4 nats with the batch shape alone**, and that the paged path sits +0.093/+0.114/+0.247 from a one-shot forward while transformers' own cached forward sits −0.107/+0.271/+0.081 from the same forwards. K8's budget is **0.05 ppl**. An instrument whose noise is several times its own budget cannot gate anything; P44's own table already said so ("no instrument: K8 unreadable on this family"), and I did not carry that forward when I wrote the rule.
2. **The K8 runner cannot see a store map anyway.** `bench/p39/step_decomp.py` requires `--arena` and loads through the arena path — which this same release makes **refuse** a per-layer map, because the arena packs one store for the whole model. So even on a family where K8 read cleanly, the graded configuration could not reach it without arena work that does not exist.

**The replacement instrument is the one every lane in this series already used**: KL from the bf16 checkpoint, prefill-shaped under the reference self-consistency control, which P44 established as its own licence kind (`licensed_by: kl-vs-bf16`, carried today by gpt-oss's store route).

**The bar, derived from configurations e4b already ships rather than from Gemma's number:** on this instrument gpt-oss's register-licensed NF4 requant reads **0.0222 nats / top-1 0.9366**, its licensed store route **0.0019 / 0.9814**, OLMoE, Granite and Mixtral NF4 fall in **0.02–0.1** (`bench/KL-FINDINGS.md`), and Gemma-4's own layer-27-only configuration reads **0.0056 / 0.9697**. So the band occupied by everything e4b ships is **≤ 0.10 nats with top-1 ≥ 0.93**, and that is the bar a per-family *default* has to clear.

**Applied: the graded map does not clear it.** `graded_10_10` reads **0.1695 nats / top-1 0.855** — 1.7× the bar, and disagreeing with the bf16 model on about one token in seven.

**Disclosure:** this bar is registered *after* that number existed. It is derived from independent families and from configurations that shipped before today, but it is not a pre-registered gate and must not be read as one. The clean version — the bar fixed first, measured on a held-out prompt set rather than the 200 these lanes have used — is one H100 lane at ≈ $3 and has not been run, because the margin is 1.7× and the per-stratum spread across the existing 200 prompts is tight (general 0.15 to code 0.20). If that margin were near 1.0 this paragraph would not be acceptable and the lane would be required.

**Decision, unchanged from what already ships:** the graded map is a **documented option, not a default**. `quantize_layers` takes it, `RESULTS-p51.md` publishes the curve and the bytes, and a user chooses with the numbers in front of them. **No Gemma-4 position is quoted.** Two scope boundaries belong beside the recommendation wherever it appears: it applies to the **loader path only** (the arena and hybrid-tier serving stack refuse per-layer maps), and its quality is materially below every other family's shipped quantisation.

