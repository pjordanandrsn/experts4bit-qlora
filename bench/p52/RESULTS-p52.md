# Results — P52: the Gemma-4 gate, run properly (H100 NVL, 2026-09-19)

Pre-registration: [`P52-PREREG.md`](P52-PREREG.md). Owner instruction: *"run the gate properly."* Run `p52-gemma4gate`, receipt `receipts/experts4bit-qlora/2026-09-19/p52-gemma4gate/p47/`, read by `bench/p52/p52_reduce.py`.

**The bar was registered before this measurement existed** (P51 amendment 3, on `main`): a per-family *default* must reach **KL ≤ 0.10 nats and top-1 ≥ 0.93**, the band every configuration e4b already ships occupies. **The prompts are held out**: `bench/kl_prompts_heldout.py`, 100 new prompts (`sha 2030ae10e0fe537f`) in the committed set's four strata and proportions, written for this gate and mechanically checked disjoint — the check caught six accidental overlaps on the first build and they were replaced. This matters because every lane from P44 to P51 scored the committed 200, **including P48's per-layer profile, which chose this map's 10/10/10 boundaries**.

## The rows (held-out set, 3,856 scored tokens, prefill scorer)

| arm | tiers | KL nats/token | top-1 | general / technical / code / longctx | expert store |
|---|---|---|---|---|---|
| `bf16_20` (anchor) | `bf16:20_nf4:10` | 0.0376 | 0.924 | 0.040 / 0.058 / 0.043 / 0.029 | 32.35 GB |
| **`graded_10_10`** (the subject) | `bf16:10_int8:10_nf4:10` | **0.1319** | **0.874** | 0.146 / 0.249 / 0.126 / 0.108 | 25.70 GB |
| `bf16_13` (matched-bytes uniform) | `bf16:13_nf4:17` | 0.1830 | 0.849 | 0.289 / 0.310 / 0.168 / 0.146 | 25.21 GB |
| gpt-oss `nf4_r12` (the bar's provenance) | — | **0.0217** | **0.9477** | — | — |

## Verdicts

- **G2 (the bar's provenance replicates) — HOLDS, and it is what makes the rest readable.** gpt-oss's shipped NF4 requant reads **0.0217** here against **0.0222** on the committed set — a difference of **-2.1 %** — with top-1 0.9477. The bar was derived from that configuration, and it lands in the same place on this set. Registered consequence of a refutation was that **nothing** would be read; it does not apply.
- **G1 (the two sets are interchangeable) — REFUTED.** The Gemma anchor reads 0.0376 here against 0.0469 on the committed set, **-19.8 %**, outside the registered ±15 %. Per the registered rule, **every number above is read on the held-out set alone** and no committed-set figure is carried across. Worth noting what this is and is not: gpt-oss moved 2 % between the same two sets while Gemma-4 moved 20 %, and Gemma's own reference self-inconsistency also moved (0.279 → 0.218). The prompt sets are not the unstable thing; **this family is**, which is the same instability #359 and P44's scorer control already recorded.
- **G3 — THE GATE — FAILS.** `graded_10_10` reads **0.1319 nats and top-1 0.874** against a bar of ≤ 0.10 and ≥ 0.93. It misses on both axes, on every stratum (technical is worst at 0.249), and it misses on the **more favourable** of the two sets — the anchor shows this set reads ~20 % lower for Gemma-4, so the committed-set measurement would have been worse. **The graded map does not ship as the Gemma-4 default.**
- **G4 (the matched-bytes result replicates) — HOLDS.** Graded 0.1319 against uniform 0.1830 at the same expert store: **1.39×**, where the committed set read 1.44×. P51's finding — that the owner's graded shape beats a plain shorter high-precision head at equal bytes — is not an artefact of the prompts it was measured on.

## What this settles

**The gate is now run and not passed**, which is a materially stronger statement than the *not run* it replaced. The graded store map stays a **documented option**: the loader takes it, `bench/p51/RESULTS-p51.md` publishes the curve and the bytes, and a user chooses with the numbers in front of them. **No Gemma-4 position is quoted.**

Two things survive the gate's failure and should not be lost in it:

1. **The shape recommendation holds.** G4 says that if you are going to quantise this family's experts at all, grade them — bf16 where sensitivity is extreme, int8 through the middle, 4-bit at the end — rather than using a shorter uniform high-precision head. That was the owner's directive and it is measured correct on two independent prompt sets.
2. **The mechanism holds.** The per-layer store map shipped in 0.36.3 is what makes any of this expressible, and it is not Gemma-specific.

What fails is narrower: *this particular configuration is not good enough to be a default for an unaware user.* The reason is the one P47–P51 established — Gemma-4's expert sensitivity is positional and spans 159×, so any configuration cheap enough to be worth having is also far enough from bf16 to be visible.

## Cost

≈ 31 min on an H100 NVL at ≤ $3.10/h ≈ **$1.60** against a registered $4.65, covering two families, four arms and two reference passes.
