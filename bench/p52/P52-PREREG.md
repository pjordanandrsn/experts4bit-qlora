# P52 — THE GEMMA-4 GATE, RUN PROPERLY: the bar fixed first, measured on held-out prompts (pre-registered 2026-09-19 ~14:50Z, before any box is rented)

Work item: adertha-agents#110. Owner instruction, 2026-09-19: *"run the gate properly."*

## Why this lane exists

P50 and P51 promised the Gemma-4 graded store map would ship "behind its own K8 two-text gate". P51 amendment 3 recorded that **that gate cannot be built on this family** — K8's budget is 0.05 ppl and Gemma-4's own NLL moves **0.4 nats with batch shape alone** (`e4b.parity.gemma4.no-reference`), and the K8 runner requires the arena path, which refuses per-layer store maps by design. The same amendment named the replacement instrument (KL from the bf16 checkpoint, the `licensed_by: kl-vs-bf16` kind P44 established) and a bar derived from configurations e4b already ships — but it applied that bar to a number that already existed, and said so. **This lane runs it the other way round: the bar is already registered on `main`, and the measurement is new.**

Two contaminations this lane removes:

1. **The prompts.** Every lane from P44 to P51 scored `bench/kl_prompts.py`'s 200 prompts, and **P48's per-layer sensitivity profile — the measurement that chose the graded map's 10/10/10 tier boundaries — was taken on them.** Scoring the gate on the same prompts would make it partly a fitted metric. This lane scores **`bench/kl_prompts_heldout.py`**, 100 new prompts in the same four strata and proportions, written for this purpose, committed before the run, mechanically checked disjoint (`assert_disjoint_from_committed`), and never used to choose anything.
2. **The bar's provenance.** The bar comes from gpt-oss's shipped NF4 requant reading 0.0222 nats / top-1 0.9366 **on the committed set**. Applying it to a Gemma-4 number measured on a different set assumes the two sets are equivalent. This lane measures that reference point on the **same held-out prompts as its subject**, so the comparison is within-set.

## The bar (registered on `main` before this lane, in P51 amendment 3)

A per-family **default** must sit in the band every configuration e4b already ships occupies on this instrument: **KL ≤ 0.10 nats/token AND top-1 agreement ≥ 0.93.** Reference points, all on the committed set: gpt-oss NF4 requant 0.0222 / 0.9366; gpt-oss licensed store route 0.0019 / 0.9814; OLMoE, Granite and Mixtral NF4 0.02–0.10 (`bench/KL-FINDINGS.md`); Gemma-4's own layer-27-only configuration 0.0056 / 0.9697.

## Instrument and arms

`bench/p44/kl_serve.py --prompt-set heldout`, prefill scorer chosen by control (i) per family, K0-gated, one child per arm, each row carrying the per-store stack census its tier spec names.

| family | arm | tiers | role |
|---|---|---|---|
| `gemma4mix` | `graded_10_10` | `bf16:10_int8:10_nf4:10` | **the subject: the map the gate decides** |
| `gemma4mix` | `bf16_20` | `bf16:20_nf4:10` | the anchor, to calibrate the held-out set against the committed one |
| `gemma4mix` | `bf16_13` | `bf16:13_nf4:17` | the matched-bytes uniform alternative, for the same comparison on fresh data |
| `gptoss` | `nf4_r12` | — | **the bar's provenance point, on the same prompts** |

## Registered predictions

- **G1 (the held-out set is comparable):** `bf16_20` reads within **±15 %** of its committed-set value **0.0469** nats. Refuted → the two sets are not interchangeable, every number below is read on the held-out set alone, and the committed-set figures are not carried across.
- **G2 (the bar's provenance replicates):** gpt-oss `nf4_r12` reads within **±15 %** of **0.0222** nats and **top-1 ≥ 0.93**. Refuted → the bar's derivation does not transfer to this set and **the gate is not read at all**; the lane reports the discrepancy and stops.
- **G3 — THE GATE:** `graded_10_10` reads **≤ 0.10 nats AND top-1 ≥ 0.93** → the graded map **may ship as the Gemma-4 default**. Otherwise it stays a documented option. (Its committed-set value was 0.1695 / 0.855; this prediction is that the held-out measurement confirms a failure, and G1 is what makes a differing result interpretable rather than convenient.)
- **G4 (the matched-bytes result replicates):** `graded_10_10` < `bf16_13` on the held-out set too, i.e. P51's M6 was not a property of the committed prompts. Refuted → P51's recommendation of the graded *shape* is weakened and says so.

## Decision rules

- **G2 refuted → nothing is read.** The bar cannot be applied to a set on which its own provenance does not hold.
- G1 ∧ G2 ∧ **G3 holds** → the graded map becomes the Gemma-4 default, shipped in a follow-up PR with the bar, both prompt sets' numbers and the loader-path scope boundary stated beside it.
- G1 ∧ G2 ∧ **G3 refuted** → the graded map stays a **documented option**; the gate is recorded as *run and not passed*, which is a stronger statement than today's *not run*.
- G4 refuted → P51's shape recommendation is re-opened with its own lane; the gate's verdict is unaffected.
- No Gemma-4 position is quoted either way until G3 passes.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p52-gemma4gate` | H100 NVL (≥ 80 GB) | 3.10 | 1.5 h | $4.65 |

Two families, two fetches (49 GB + 13 GB), four arms, two reference passes over 100 prompts. Timing basis P49/P51: ≈ 35 min. STOP: not the class (15); egress < 20 MB/s (14); free disk < 120 GB (13); K0 failing (16); a fetch alarm → not_run; a refused arm is an honest hole; **the held-out set failing its disjointness assertion refuses before any scoring**; a second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p52-gemma4gate/p47/` — `gemma4mix_kl.json` and `gptoss_kl.json`, each recording `prompt_set.name = "kl_prompts_heldout"` and its digest, plus `k0.json`, `versions.txt`, `forensics.txt`, `logs/`; read by `bench/p52/p52_reduce.py`. Results → `RESULTS-p52.md`.

## Amendments

(none yet)

## Read (2026-09-19, run p52-gemma4gate)

`RESULTS-p52.md`: **G2 HOLDS** (gpt-oss 0.0217 vs 0.0222, -2.1 % — the bar transfers, so the rest is readable); **G1 REFUTED** (the Gemma anchor moved -19.8 % while gpt-oss moved 2 % — the family is the unstable thing, not the sets — so everything is read on held-out alone); **G3 THE GATE FAILS** (0.1319 nats / top-1 0.874 against ≤ 0.10 and ≥ 0.93, missing on both axes and on the more favourable set); **G4 HOLDS** (1.39× at matched bytes, against 1.44× on the committed set). **Decision: documented option, no default, no quoted position — the gate is now RUN AND NOT PASSED.**

