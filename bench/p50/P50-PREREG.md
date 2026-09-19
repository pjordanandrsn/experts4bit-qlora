# P50 — WHAT THE REMEDY COSTS: the bf16-early-layers curve for Gemma-4 (pre-registered 2026-09-19 ~11:20Z, before any box is rented)

Work item: adertha-agents#110; defect [#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597). Lineage: P47 → P48 → P49. e4b's Gemma-4 modelling is faithful (P48 P3: 29 of 30 expert stacks in bf16 reads 0.0056 nats from the checkpoint); the whole gap is NF4 on the EARLY expert layers (P48: layer 0 alone 0.892 of the all-layers 1.084, monotone to 0.006 at layer 27); and **no store e4b ships rescues layer 0** (P49 P1 refuted: int8 0.693, fp8 0.774, NF4-b32 0.868, NF4 0.892, FP4 1.051). The registered decision rule for a refuted P1 names the remedy: **keep the first k expert layers in bf16**. This lane measures what that costs, in quality and in bytes, before any default is proposed.

## Instrument and arms

`bench/p44/kl_serve.py` (prefill scorer by control (i), the same cached bf16 reference and 200 committed prompts as P44-b / P47 / P48 / P49) on family **`gemma4keep`** (`bench/p44/serve_stack.py`): builder `loader_keep_<k>` = `load_moe_4bit_streaming` with `quantize_layers = {k..29}` — **bf16 experts in layers 0..k−1, NF4 block-64 in the rest** — for **k ∈ {5, 10, 15, 20, 24}**. Every row carries `verify_moe_4bit`'s counts (k bf16 / 30−k quantised stacks, checked by `kl_serve._builder_check`) and a new **expert-bytes census** (`expert_bytes_total_gb`: the quantised stacks' packed bytes + scales, plus the bf16 stacks' parameters), so quality and cost are read from the same row.

Two arms of the curve are already measured and are anchors this lane re-reads rather than assumes: **k = 0 is P47's `loader_nf4` = 1.0837** and **k = 15 is P47's `loader_nf4_hi` = 0.1334** (`quantize_layers = {15..29}`, the identical set).

## Registered predictions (falsifiable; reducer `bench/p50/p50_reduce.py`)

- **P1 (the anchor):** `K15` ∈ [0.11, 0.16] — P47's 0.1334 on a different box and a later e4b head. Refuted → the curve is read against this lane's own rows only, disclosed.
- **P2 (the curve is monotone and k = 15 is not enough):** KL(k) strictly decreases in k, and **`K15` > 0.05** (the other families' NF4 floor, and the threshold any Gemma-4 position must reach before it is quotable). Refuted if `K15` ≤ 0.05 — then the remedy is far cheaper than P48's profile implies.
- **P3 (k = 20 reaches the floor):** **`K20` ≤ 0.05**. Registered alternative: `K20` > 0.05 and `K24` ≤ 0.05 (the floor needs 24 of 30 layers); if even `K24` > 0.05 the remedy fails as specified and the family needs a different lever entirely.
- **P4 (the remedy is expensive):** at the SMALLEST k whose KL ≤ 0.05, the expert store is **≥ 2.5×** the all-NF4 store's bytes (`expert_bytes_total_gb` against the `K00`-equivalent computed from the same census: 30 NF4 layers). Refuted below 2.5× — then the remedy is close to free and the default is easy.
- **P5 (top-1 tracks):** the top-1 agreement at the passing k is ≥ 0.95 (P48's layer-27-only row read 0.97 at 0.0056 nats; a config at the KL floor that still disagrees on 1 token in 10 would mean the KL floor is the wrong gate for this family).

## Decision rules

- P3 ∧ P5 hold → a per-family default is **proposable**, not shipped: `quantize_layers` = {k..29} for Gemma-4 with the measured k, its memory cost quoted beside it, and it ships only after the **K8 two-text gate** (wikitext + c4val1, the registered 0.05-ppl budget) passes on the served stack at that k — a separate lane with its own pre-registration, because K8 is a serving-stack instrument and this one is a loader-model instrument.
- P4 holds (expensive) → the write-up states plainly that Gemma-4 at the quality floor is **not a 4-bit model on this path**; the honest framing for any position is "k layers bf16, the rest NF4, N GB", never "Gemma-4 in 4-bit".
- P3's second alternative (only k = 24 reaches it) → same, with the cost stated at k = 24; the default is likely not worth shipping and the lane says so.
- P3 refuted at every k → the remedy fails; the next candidate is activation-aware scaling of the early layers' experts, which needs P49's redrawn probe (P5 there) first.
- Nothing here changes a default or the register. **Gemma-4 stays unquoted.**

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p50-gemma4keep` | H100 NVL (≥ 80 GB) | 3.10 | 1.5 h | $4.65 |

Timing basis (P48/P49): fetch ~7 min, K0 < 2 min, reference pass 40 s, five loader arms ~90–150 s each (the bf16-heavy arms load more bytes) ≈ 12 min; ≈ 25 min. The largest arm (k = 24) holds ~39 GB of experts, which fits the class with the reference freed. STOP: not the class (15); egress < 20 MB/s (14); free disk < 120 GB (13); K0 failing (16); a fetch alarm → not_run; a refused builder row (wrong layer set) is an honest hole and the other arms still run; a second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p50-gemma4keep/p47/` — `gemma4keep_kl.json` (5 rows with their expert-bytes census), `k0.json`, `versions.txt`, `forensics.txt`, `logs/`; read by `bench/p50/p50_reduce.py`. Results → `RESULTS-p50.md`; #597 updated.

## Amendments

(none yet)

## Read (2026-09-19, run p50-gemma4keep)

`RESULTS-p50.md`: P1 HOLDS (k=15 = 0.1334, P47's row to four digits), P2 HOLDS, **P3 HOLDS (k=20 = 0.0469 ≤ 0.05)**, **P4 HOLDS (32.3 GB vs 12.0 GB all-NF4 = 2.70×, i.e. 76 % of the all-bf16 store)**, **P5 REFUTED and its threshold was miscalibrated by me** (gpt-oss's shipped NF4 reads top-1 0.9366, itself under the 0.95 I registered). Decision: **no NF4 expert default for Gemma-4**; `quantize_layers` plus this curve is the documentation; no position is quoted or proposed.

### Amendment 1 (2026-09-19 ~14:30Z) — the K8 gate this lane's decision rule named is unbuildable on this family

P50's decision rules say a per-family default ships "only after the **K8 two-text gate** passes on the served stack". P51 amendment 3 records why that gate cannot be built on Gemma-4: K8 is unreadable here (the family's own NLL moves 0.4 nats with batch shape against a 0.05 budget — `e4b.parity.gemma4.no-reference`), and the K8 runner requires the arena path, which refuses per-layer store maps by design. The replacement is the KL-from-checkpoint instrument these lanes already use, with a bar derived from shipped configurations (≤ 0.10 nats, top-1 ≥ 0.93). Neither the uniform curve here nor P51's graded map clears it, so the conclusion of both lanes is unchanged: **documented option, no default, no quoted position.**

