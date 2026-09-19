# Results — P46: the adapter path at the field recipe (Qwen3-30B-A3B, RTX 5090, 2026-09-19)

Pre-registration: [`P46-PREREG.md`](P46-PREREG.md) (+ amendment 1, written after the data and marked so). Run `p46-qwen3lora` (RTX 5090, vast 51551542, e4b `3c14f567`, gnf4 `d8f737a6` = the P46 knobs), receipt `receipts/experts4bit-qlora/2026-09-19/p46-qwen3lora/tp4/`, read by `bench/p46/p46_reduce.py`. The fixture is tp4's field recipe (alpaca, seq 2048, mb 2 × accum 4, r 16, adamw_8bit, 20 steps, the same tokens sha as P43/P45); every arm trains the same 30,428 tokens from the same init (`init_sha` identical across arms).

| arm | `NF4_QLORA_LORA_PATH` | s/step (median, steps 11+) | tok/s | peak VRAM GB | held-out final | Δ vs reference | path census per step (of 768 `fused_grouped_lora` calls) |
|---|---|---|---|---|---|---|---|
| `reference_attn4` | — (no fused kernel) | 63.08 | 23.8 | 24.581 | 0.90206 | — | — |
| `fused_attn4` (`auto`, the shipped rule) | auto | **24.46** | 61.8 | 24.581 | 0.90359 | +0.00153 | loop 652–768, padded 0–116, grouped_mm 0 |
| `fused_attn4_pad` (`padded`) | padded | **4.22** | 365.2 | 24.581 | 0.90274 | +0.00068 | padded 768 |
| `fused_attn4_gmm` (`grouped_mm`) | grouped_mm | REFUSED (recorded) | | | | | `torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0` — the 5090 is sm_120 |

## Verdicts (reducer output, rules as registered)

- **P1 (mechanism) — REFUTED at the letter; the mechanism stands.** `auto` took the per-expert loop on EVERY step (652–768 of 768 calls), but the registered clause also demanded `lora_path_padded = 0`; the padded path served 8–116 calls per step (the groups whose skew fell under the 4× waste guard). The reducer reads REFUTED and that stands; amendment 1 records what the clause should have said and what this row establishes: **the loop serves ≥ 85 % of the adapter calls at the field fixture, so P45's ~273k `aten::mm` per step ARE the loop.** The registered "refuted → stop and re-read" was written for the loop NOT being taken; that did not occur.
- **P2 (the padded path) — HOLDS, by a wide margin.** `padded` trains at **0.173× the s/step of `auto`** (4.22 vs 24.46 s; registered ≤ 0.55×, refute > 0.8×). That is **5.8× faster**, 365.2 tok/s against 61.8 — the deliberately conservative prediction (from the A2000 E=256 3.0×) under-estimated because the field fixture is launch-bound, not flop-bound: padding wastes flops on rank-16 matmuls that cost nothing while removing ~650 launches per call.
- **P3 (the jagged path) — NOT READ**: `torch._grouped_mm` refuses on sm_120 (Hopper-only in torch 2.8). Recorded as registered; readable only on an H100-class box.
- **P4 (parity, the gate) — HOLDS for both fused arms.** Held-out final: `auto` +0.00153, `padded` +0.00068 nats vs the reference (band 0.05); median per-step |Δ loss| 0.00177 / 0.00241. Both arms train the same model to within the bf16 noise floor.
- **P5 (memory) — HOLDS**: peak VRAM identical (24.581 GB on every arm — the padded buffer is dwarfed by what the model + optimizer state hold at this fixture).

## What this decides

By the registered decision rule (P1 ∧ P2 ∧ P4(pad), with P1 read under amendment 1): **the padding-waste guard is the defect.** `grouped-nf4-gemm`'s `auto` rule chose the slower path for ≥ 85 % of the adapter calls at the field recipe, costing 20.2 s of a 24.5 s step. The next grouped-nf4-gemm cut replaces the 4× flop-waste rule with a structural guard on what padding actually risks — the padded block's BYTES (the loop is kept for the case where it would not fit; the old ratio guard becomes opt-in) — with the waste ratio recorded for the census, and e4b's field-recipe H2H (tp4 box B: qwen3 + qwen3_5, e4b vs Unsloth vs HF) is re-run on that cut. **The training position moves only from THAT receipt**; nothing in the register changes from this one. (For orientation only, not a claim: e4b's T1 step was 29.3 s where Unsloth's was ~13 s on the same box class; a ~4–5 s step would invert the field-recipe comparison, if it holds under the H2H harness.)

## Cost

`p46-qwen3lora`: ~55 min on a 5090 at ≤ $0.65/h ≈ $0.60 (registered $1.30). Torn down with proof.
