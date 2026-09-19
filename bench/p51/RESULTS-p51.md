# Results — P51: a graded store map for Gemma-4 (H100 NVL, 2026-09-19)

Pre-registration: [`P51-PREREG.md`](P51-PREREG.md) (+ amendments 1 and 2). Owner directive, 2026-09-19: *"gemma: Keep the first N expert layers high precision, crush the rest."* Runs `p51-gemma4mix` (four arms lost to a harness fault of this lane's — amendment 1), `p51-gemma4mix-2` (four arms read) and **`p51-gemma4mix-3`** (all five, the matched-bytes control). Same instrument as every other #597 lane: `bench/p44/kl_serve.py`, prefill scorer chosen by control (i), the same cached bf16 reference and 200 committed prompts. Every row carries the per-store stack census its tier spec names, so a map that collapsed to one store refuses rather than producing a wrong row. Read by `bench/p51/p51_reduce.py`.

## The rows

| arm | tiers (layers 0→29) | KL nats/token | top-1 | expert store |
|---|---|---|---|---|
| `bf16_20` (anchor) | `bf16:20_nf4:10` | **0.0469** | 0.916 | 32.35 GB |
| **`graded_10_10`** | **`bf16:10_int8:10_nf4:10`** | **0.1695** | 0.855 | **25.70 GB** |
| `bf16_13` (matched-bytes control) | `bf16:13_nf4:17` | **0.2448** | 0.827 | **25.21 GB** |
| `graded_5_15` | `bf16:5_int8:15_nf4:10` | 0.4773 | 0.771 | 22.38 GB |
| `int8_20` | `int8:20_nf4:10` | 0.7369 | 0.716 | 19.05 GB |

For scale, from the neighbouring lanes: all-NF4 is 1.0837 nats at 12.0 GB, all-bf16 experts are 42.5 GB, and P50's uniform curve runs `keep_10` 0.4618 @ 22.16 GB → `keep_15` 0.1334 @ 27.25 → `keep_20` 0.0469 @ 32.35.

## Verdicts

- **M1 (the anchor) — HOLDS.** `bf16_20` reads 0.0469 at 32.35 GB: P50's `keep_20` to four digits **and to the byte**, rebuilt through entirely different machinery (a per-layer store map rather than a layer set). The map is sound.
- **M2 (a uniform int8 head fails) — HOLDS.** 0.7369, against a registered ≥ 0.50. P49 measured int8 *on layer 0 alone* at 0.693 and this composes as that implied: **the first layers must be bf16**, and the directive's "high precision" cannot be read as 8-bit.
- **M3 (grading ships at ≤ 0.10) — INCONCLUSIVE.** 0.1695 sits between the registered 0.10 and 0.30. On its own this decided nothing, which is why amendment 2 added the control below.
- **M4 (grading saves bytes) — HOLDS.** 25.70 GB against the anchor's 32.35, a saving of 6.65 GB (0.79×).
- **M5 (crushing the tail) — RETIRED as unmeasurable on this model, not refuted.** Gemma-4's `moe_intermediate_size` is **704 = 64 × 11**, so no block larger than 64 divides it and the stack refuses; NF4 at block 64 is already the smallest store e4b ships. **Every configuration here already has a maximally crushed tail** — on this architecture the "crush the rest" half of the directive has no remaining freedom, and the whole design question is where the high-precision boundary sits.
- **M6 (the matched-bytes control) — REFUTED, and the refutation is the answer.** At essentially the same expert store, the graded map is **1.44× better than a plain uniform head**: `bf16_13` reads 0.2448 at 25.21 GB against `graded_10_10`'s 0.1695 at 25.70 GB (2 % more bytes, 31 % less divergence, top-1 0.855 vs 0.827). M6 was registered so that "uniform at least as good" would retire the idea; it is not, by a wide margin.

## Decision: ship the graded map

**The owner's directive is measured and correct, and my prior reading was wrong.** Before the control I had one nearly-matched pair pointing the other way (`graded_5_15` at 22.38 GB / 0.4773 against P50's `keep_10` at 22.16 GB / 0.4618, where uniform wins) and I expected the matched control to confirm it. It did the opposite at the point that matters. Both readings are real, and together they say grading pays only once the bf16 head is long enough to cover the layers that actually need it: **at a 5-layer head grading loses; at a 10-layer head it wins by 1.44×.** P48's sensitivity profile explains the crossover — layer 0 reads 0.892 and layer 10 reads 0.332, so an int8 tier starting at layer 5 still lands on layers that cannot take it, while one starting at layer 10 does not.

So, for Gemma-4:

- **The recommended shape is bf16 for the first ~10 expert layers, int8 for the next ~10, NF4 for the rest** — `quantize_layers = {0..9: None, 10..19: "int8", 20..29: ("nf4", 64)}`, which the loader now accepts directly.
- It reaches **0.1695 nats at 25.70 GB** of expert store, against 42.5 GB for all-bf16 experts and 12.0 GB for all-NF4.
- A user who wants the 0.05 fidelity floor still pays for it: on the uniform curve the floor arrives at 32.35 GB, and the graded configuration that would reach it is not measured here.

**No Gemma-4 position is quoted, and none ships as a silent default. The gate has since been RUN AND NOT PASSED** (`bench/p52/RESULTS-p52.md`: on held-out prompts this map reads 0.1319 nats / top-1 0.874 against a bar of ≤ 0.10 and ≥ 0.93 fixed before the measurement; the bar's own provenance point replicated to within 2 % on those prompts, and this lane's matched-bytes result replicated at 1.39×). The gate the rule originally named — K8 on the served stack — **cannot be built on this family**, for two independent reasons recorded in amendment 3: K8 is unreadable on Gemma-4 (its own NLL moves 0.4 nats with batch shape against a 0.05 budget), and the K8 runner requires the arena path, which refuses per-layer store maps by design. The replacement bar, derived from configurations e4b already ships (≤ 0.10 nats and top-1 ≥ 0.93, the band gpt-oss's licensed NF4 and the other families occupy), is **not cleared**: this map reads 0.1695 / 0.855. So the graded map stays a **documented option**. Two boundaries travel with the recommendation: it applies to the **loader path only**, and its quality is materially below every other family's shipped quantisation.

## Cost

Three draws: run 1 ≈ $0.6 (four arms lost to the expected-count bug), run 2 ≈ $1.0, run 3 ≈ $0.9. ≈ **$2.5** total.
