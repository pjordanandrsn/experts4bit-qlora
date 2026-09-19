# Results — tp4 re-run on grouped-nf4-gemm 0.32.1: the adapter path's `auto` rule, measured on five MoE families (2026-09-19)

Pre-registration: [`bench/p46/P46-PREREG.md`](../p46/P46-PREREG.md) (the decision rule that ordered the re-run, and amendment 2 that extended it to box A) + [`TP4-PREREG.md`](TP4-PREREG.md) (the fixture, the validity rules and the quoting rules, unchanged). Runs `tp4-b-p46cut-4` (RTX 5090, qwen3 + qwen3_5) and `tp4-b-p46cut-A` (RTX 5090, granite + olmoe + gpt-oss), both on e4b `47e6f0b` with **grouped-nf4-gemm pinned at the v0.32.1 release commit** `9206352f`. Read by `bench/tp4/tp4_reduce.py` under its own rules; nothing below is hand-transcribed from a log.

**What changed between the old rows and these.** One rule, in the kernel package: `nf4_qlora.lora_delta_grouped`'s `auto` path used to send any adapter call past a 4× padding-waste ratio to a per-expert Python loop. P46 measured that guard choosing the loop for ≥ 85 % of calls on Qwen3's field recipe, at 24.46 s/step against 4.22 for the padded path with identical loss. 0.32.1 makes the guard structural (pad unless the padded block would not fit). Nothing else in either package moved.

## The fixture

tp4's field recipe, unchanged: alpaca, seq 2048, micro-batch 2 × accum 4, r 16 / α 16, lr 2e-4, AdamW-8bit, linear warm-up 5, seed 3407, **N = 20** optimizer steps, s/step as the median over steps 11+. The earlier rows this compares against ran **N = 60** on the same fixture; the median is over steps 11..N in both cases, which is the comparison tp4 registered, but the window differs and that is stated wherever a before/after appears.

## e4b's own step, before and after the kernel change

| family | e4b `fused_attn4`, old cut | on 0.32.1 | change |
|---|---|---|---|
| Granite-3.1-3B-A800M | 2.8531 s/step (N=60) | **2.3660** | ×1.21 faster |
| OLMoE-1B-7B | 2.7389 s/step (N=60) | **1.3949** | ×1.96 faster |
| Qwen3-30B-A3B | **alarmed** at 3600 s without finishing; P43's T1 diagnosis measured 29.30 s/step | **6.4707** | the arm finishes |
| Qwen3.6-35B-A3B | **never produced a row** (box B's earlier draws never reached it) | **6.3344** | first row for this family |

> **Correction, disclosed:** P46 amendment 2 registered OLMoE's threshold as "< 4.6867 s/step". That figure was **mis-transcribed** from another family's row; OLMoE's actual tp4 receipt reads **2.7389**. The prediction holds against either number, but the one I registered was wrong and this is the record of it.

## The cross-family table, as the reducer reads it

| family | e4b `fused_attn4` | e4b `reference_attn4` | Unsloth | HF+PEFT | position quoted | e4b internal parity |
|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | 2.366 s | 18.590 s | 2.445 s — **VOID** | 3.057 s | **hf/e4b = 1.292**, quality COMPARABLE | **PASS** Δfinal 0.00063 |
| OLMoE-1B-7B | 1.395 s | 14.883 s | died before its first step | 2.713 s | hf/e4b = 1.945 — **quality FLAGGED**, not a clean position | **PASS** Δfinal 0.00203 |
| gpt-oss-20b | REFUSED by citation (bare experts) | REFUSED | REFUSED | REFUSED | none by registration | NO-ARM |
| **Qwen3-30B-A3B** | **6.471 s** | 60.256 s | **29.055 s — VALID** | OOM | **unsloth/e4b = 4.490**, quality COMPARABLE | **PASS** Δfinal 0.00140 |
| Qwen3.6-35B-A3B | 6.334 s | 88.009 s | 26.979 s — **VOID** | OOM | none (arms not both VALID) | **PASS** Δfinal 0.00159 |

Secondary pair at micro-batch 1 × accum 8, run because a primary arm OOMed: Qwen3 **unsloth/e4b = 5.024** (51.691 vs 10.290 s), quality COMPARABLE.

## The one clean head-to-head

**Qwen3-30B-A3B, one RTX 5090, one fixture, both frameworks training the same 642,514,944 parameters:**

| | e4b `fused_attn4` | Unsloth |
|---|---|---|
| s/step (median 11+) | **6.4707** | 29.0547 |
| tokens/s | 237.4 | 48.3 |
| peak VRAM | 24.581 GB | 24.728 GB |
| J/step | 1186.42 | 3340.5 |
| held-out final | 0.9042 | 0.93253 |

**e4b is ×4.49 per step at comparable quality** (held-out Δ 0.0283 nats, inside the 0.05 reading band) **and ×2.82 less energy**, at the same peak VRAM. The position is quoted because both arms are VALID under tp4's rules — same trainable-parameter count, both in the 4-bit MoE regime — and because e4b's fused path passes its own parity control on the same box: **0.00140 nats from e4b's dense reference**, which is ×9.31 slower. A fast arm that failed that control would not be a position.

## What is NOT quoted, and why

- **Granite and Qwen3.6 against Unsloth**: Unsloth's arm trains 5.2 M parameters against e4b's 99.6 M on Granite — attention only, not the 4-bit MoE regime — so the reducer marks it VOID and quotes nothing. Same class of mismatch on Qwen3.6.
- **OLMoE against Unsloth**: its arm died before its first step (a known grouped-matmul fault on its transformers pin), recorded as a row.
- **OLMoE against HF**: the ratio is 1.945 but the held-out gap is 0.0550 nats, outside the reading band, so it is FLAGGED and not a clean position.
- **gpt-oss**: e4b's fused path refuses this family by citation (experts built bare, no `ExpertsLoRA`), so there is no primary pair; the `attn_only` secondary row is recorded.
- **Gemma-4 and Mixtral**: not in this re-run's scope.

## The registered predictions, scored

- **A1 (every family whose e4b arm was VALID is VALID again and faster)** — **HOLDS**: Granite 2.853 → 2.366, OLMoE 2.739 → 1.395. (With the transcription correction above.)
- **A2 (the gain is smaller than Qwen3's)** — **HOLDS**: ×1.21 and ×1.96 against Qwen3's ×4.5. The registered alternative (any family ≥ 3× ⇒ the guard was mis-tuned across the board) does not fire. The guard bit hardest where router skew is worst, which is what the rule predicted.
- **A3 (internal parity holds on every family with both arms)** — **HOLDS**: 0.00063, 0.00203, 0.00140, 0.00159, all against a 0.05 band. **A3 was the condition that would have stopped the release; it passes.**

## Cost

`tp4-b-p46cut-4` ≈ 3.6 h and `tp4-b-p46cut-A` ≈ 0.9 h on RTX 5090s, plus three refused or faulted draws earlier in the day (a 2.7 MB/s host, a manifest-validation refusal, and a driver quoting bug of mine) ≈ $0.1 total. Both boxes torn down with proof.
