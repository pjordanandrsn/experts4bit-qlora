# tp2 — e4b vs Unsloth per MoE family, one box, one fixture (/root/tp2)
Rule (tp2/P40-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per P40's validity rules with the registered n_layers (granite 32, olmoe 16, qwen3 48, gemma4 30, mixtral 32, gptoss 24); the position = s/step ratio Unsloth/e4b from the primary pair (median of steps 11..N), quoted only when both arms are VALID, with the quality reading (held-out |Δ| at N ≤ 0.05 nats → COMPARABLE, else the ratio carries the flag); e4b internal parity in tp1's B2/C2 units, informational; Qwen3's ratio vs P38's 1.413 within ±10 % or a stated finding. VOID never enters a ratio. Nothing is licensed; no cross-box number is divided into these.
```
e4b 0.35.1 (PyPI)
gnf4 0.30.2 (PyPI)
torch(e4b) 2.8.0+cu128
triton(e4b) 3.4.0
transformers(e4b) 5.16.1
bitsandbytes(e4b) 0.50.1
usercustomize_hook none
unsloth 2026.9.2
unsloth_zoo 2026.9.1
torch(unsloth) 2.8.0+cu128
triton(unsloth) 3.4.0
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft 0.20.0
torchao None
moe_backend native_torch
```

### Granite-3.1-3B-A800M-instruct (`granite`, registered n_layers 32)
- model `ibm-granite/granite-3.1-3b-a800m-instruct` @ `a02780686e08`; tokens sha `df16efaf2fa0`; N=60; fixture lr 0.0001 accum 1 autocast False seq 512 r 8 α 16; e4b trainable 49807360; box_class pcie-full/launch-fast gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | reference_attn4 | **OK** | VALID | 60 | 3.587 | 28.3 | 2.511 | 315.4 | 3.2951→0.2258 | 3.2750→0.2552 | 0 / 0 | 128 | 49807360 | 105.0 (torch.bfloat16,torch.float32) |  |
| e4b | fused_attn4 | **OK** | VALID | 60 | 0.641 | 152.9 | 2.511 | 57.0 | 3.2568→0.2342 | 3.2674→0.2605 | 32 / 128 | 128 | 49807360 | 105.0 (torch.bfloat16,torch.float32) |  |
| unsloth | ckpt_unsloth | **OK** | VOID | 60 | 0.461 | 154.2 | 6.462 | 60.5 | 3.2319→0.3870 | 3.2490→0.4546 | stacks 0 / fwd 0 / u8 0 | 0 | 2621440 | 10.5 (F32) | trainable 2621440 != e4b's 49807360 (by group: {'attention': 2621440, 'experts': 0, 'other': 0}); Params4bit expert stacks 0 < 2*32; experts forward calls/step min 0 < 32; bnb4bit expert modules (innermost) 0 < 32 (silent fallback?) |
- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal 0.00836, median step |Δ| 0.01914 → **PASS** (band 0.05/0.05); ×5.59 faster per step, peak ×1.000
- **NO POSITION QUOTED** — primary arms not both VALID: e4b fused_attn4 VALID / unsloth ckpt_unsloth VOID

### OLMoE-1B-7B-0924-Instruct (`olmoe`, registered n_layers 16)
- model `allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f440f0`; tokens sha `019e2d3f048f`; N=60; fixture lr 0.0001 accum 1 autocast False seq 512 r 8 α 16; e4b trainable 60817408; box_class pcie-full/launch-fast gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | reference_attn4 | **OK** | VALID | 60 | 2.644 | 31.7 | 4.835 | 262.4 | 4.1872→0.2877 | 4.0323→0.3164 | 0 / 0 | 64 | 60817408 | 125.9 (torch.bfloat16,torch.float32) |  |
| e4b | fused_attn4 | **OK** | VALID | 60 | 0.708 | 116.3 | 4.835 | 76.0 | 4.1813→0.3088 | 4.0220→0.3186 | 16 / 64 | 64 | 60817408 | 125.9 (torch.bfloat16,torch.float32) |  |
| unsloth | ckpt_unsloth | **HARNESS_ERROR** | — | — | — | — | — | — | —→— | —→— | stacks — / fwd — / u8 — | — | — | 0.0 (?) | rc=1 and no receipt (the process died before its first write; attempts [1]) |
- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal 0.02109, median step |Δ| 0.01803 → **PASS** (band 0.05/0.05); ×3.73 faster per step, peak ×1.000
- **NO POSITION QUOTED** — primary arms not both VALID: e4b fused_attn4 VALID / unsloth ckpt_unsloth —

### gpt-oss-20b (`gptoss`, registered n_layers 24)
- model `None` @ ``; tokens sha `9e958bf926de`; N=60; fixture lr None accum 1 autocast None seq 512 r None α None; e4b trainable 3981312; box_class None gpu None
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | reference_attn4 | **REFUSED** | — | 60 | — | — | — | — | —→— | —→— | 0 / — | — | — | 0.0 (?) | TRAIN_ATTN_4BIT would refuse: 96 of 96 attention projections carry a bias (quantize_attention_projections_4bit raises SystemExit on a bias; e.g. ['model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj']) |
| e4b | fused_attn4 | **REFUSED** | — | 60 | — | — | — | — | —→— | —→— | 0 / — | — | — | 0.0 (?) | SKIPPED as REFUSED: tp1 (P36) row cited -- enable_fast_train(dgrad=True) patched 0 modules on gpt-oss (experts built bare, no ExpertsLoRA: 'GPT-OSS-aware training LoRA is a separate change', loader.py); P40 arm 2: not ru |
| e4b | attn_only | **OK** | VALID | 60 | 1.464 | 54.1 | 14.687 | 201.5 | 5.1060→0.3664 | 5.1405→0.3435 | 0 / 0 | 0 | 3981312 | 16.0 (torch.float32) |  |
| unsloth | ckpt_unsloth | **REFUSED** | — | 60 | — | — | — | — | —→— | —→— | stacks — / fwd — / u8 — | — | — | 0.0 (?) | RuntimeError: We encountered some issues during automatic conversion of the weights. For details look at the `CONVERSION` entries of the above report! |
- e4b internal parity: NO-ARM
- **NO POSITION QUOTED** — no primary pair by registration (e4b fused REFUSED on this family; attn_only is a secondary row; no 4-bit ratio either way)

### Qwen3-30B-A3B (`qwen3`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `81dc24c3c195`; N=60; fixture lr 0.0001 accum 1 autocast False seq 512 r 8 α 16; e4b trainable 321257472; box_class pcie-full/launch-fast gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | reference_attn4 | **OK** | VALID | 60 | 11.067 | 7.8 | 21.371 | 1026.3 | 3.6535→0.2642 | 3.7021→0.2909 | 0 / 0 | 192 | 321257472 | 656.1 (torch.bfloat16,torch.float32) |  |
| e4b | fused_attn4 | **OK** | VALID | 60 | 4.108 | 21.0 | 21.371 | 383.9 | 3.6282→0.2664 | 3.7022→0.2935 | 48 / 192 | 192 | 321257472 | 656.1 (torch.bfloat16,torch.float32) |  |
| unsloth | ckpt_unsloth | **OK** | VALID | 60 | 5.986 | 14.1 | 23.141 | 561.4 | 3.6210→0.2695 | 3.6986→0.3087 | stacks 96 / fwd 96 / u8 48 | 0 | 321257472 | 1285.1 (F32) |  |
| unsloth | ckpt_hf | **OK** | VALID | 60 | 5.967 | 14.4 | 23.137 | 501.1 | 3.6210→0.2695 | 3.6986→0.3087 | stacks 96 / fwd 96 / u8 48 | 0 | 321257472 | 1285.1 (F32) |  |
- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal 0.00215, median step |Δ| 0.00962 → **PASS** (band 0.05/0.05); ×2.69 faster per step, peak ×1.000
- **POSITION: s/step ratio Unsloth/e4b = 1.457** (5.986 vs 4.108 s; e4b faster per step); peak VRAM unsloth 23.14 vs e4b 21.37 GB (Δ +1.77); J/step unsloth 561.4 vs e4b 383.9 (×1.462); tok/s unsloth 14.1 vs e4b 21.0
- quality reading at N=60: held-out e4b 0.2935 / unsloth 0.3087 (Δ 0.0152) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE; a reading threshold, not a gate); shared evals: 0: 3.7022/3.6986; 20: 0.3474/0.5622; 40: 0.3225/0.3477; 60: 0.2935/0.3087; step-0 Δ -0.0036 (the two quantisers on the same bytes)
- cross-lane anchor: ratio 1.457 vs P38's 1.413 (+3.1%) → **AGREES (within ±10 %)**

### Gemma-4-26B-A4B-it (`gemma4`, registered n_layers 30)
- model `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7d`; tokens sha `731db49a8ce4`; N=60; fixture lr None accum 1 autocast None seq 512 r None α None; e4b trainable None; box_class None gpu None
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | reference_attn4 | **HARNESS_ERROR** | — | 60 | — | — | — | — | —→— | —→— | — / — | — | — | 0.0 (?) | quantize_attention_projections_4bit converted 100 projections, expected 120 |
| e4b | fused_attn4 | **HARNESS_ERROR** | — | 60 | — | — | — | — | —→— | —→— | — / — | — | — | 0.0 (?) | quantize_attention_projections_4bit converted 100 projections, expected 120 |
| unsloth | ckpt_unsloth | **OK** | VALID | 60 | 3.510 | 23.6 | 20.824 | 403.4 | 10.6219→0.2863 | 9.8817→0.3218 | stacks 60 / fwd 60 / u8 30 | 0 | 247188480 | 988.8 (F32) |  |
- e4b internal parity: NO-ARM
- **NO POSITION QUOTED** — primary arms not both VALID: e4b fused_attn4 — / unsloth ckpt_unsloth VALID

### Mixtral-8x7B-Instruct-v0.1 (`mixtral`, registered n_layers 32)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `2a7505f989f8`; N=60; fixture lr 0.0001 accum 1 autocast False seq 512 r 8 α 16; e4b trainable 111673344; box_class pcie-full/launch-fast gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | reference_attn4 | **OK** | VALID | 60 | 2.929 | 34.1 | 8.570 | 461.5 | 3.0159→0.2232 | 2.9749→0.2495 | 0 / 0 | 128 | 111673344 | 237.1 (torch.bfloat16,torch.float32) |  |
| e4b | fused_attn4 | **OK** | VALID | 60 | 2.377 | 42.0 | 3.223 | 298.3 | 3.0147→0.2185 | 2.9761→0.2590 | 32 / 128 | 128 | 111673344 | 237.1 (torch.bfloat16,torch.float32) |  |
| unsloth | ckpt_unsloth | **OK** | VALID | 60 | 0.858 | 79.1 | 29.163 | 350.2 | 2.9993→0.2203 | 2.9690→0.2502 | stacks 64 / fwd 64 / u8 32 | 0 | 111673344 | 446.8 (F32) |  |
- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal 0.00462, median step |Δ| 0.01184 → **PASS** (band 0.05/0.05); ×1.23 faster per step, peak ×0.376
- **POSITION: s/step ratio Unsloth/e4b = 0.361** (0.858 vs 2.377 s; Unsloth faster per step); peak VRAM unsloth 29.16 vs e4b 3.22 GB (Δ +25.94); J/step unsloth 350.2 vs e4b 298.3 (×1.174); tok/s unsloth 79.1 vs e4b 42.0
- quality reading at N=60: held-out e4b 0.2590 / unsloth 0.2502 (Δ -0.0087) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE; a reading threshold, not a gate); shared evals: 0: 2.9761/2.9690; 20: 0.3199/0.3909; 40: 0.2670/0.3085; 60: 0.2590/0.2502; step-0 Δ -0.0071 (the two quantisers on the same bytes)

## Cross-family summary
| family | e4b primary | unsloth primary | position (Unsloth/e4b s/step) | quality | e4b internal parity | notes |
|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M-instruct | OK VALID | OK VOID | not quoted | — | PASS | e4b/reference_attn4: OK |
| OLMoE-1B-7B-0924-Instruct | OK VALID | HARNESS_ERROR — | not quoted | — | PASS | e4b/reference_attn4: OK |
| gpt-oss-20b | OK VALID (attn_only) | REFUSED — | not quoted | — | NO-ARM | e4b/reference_attn4: REFUSED; e4b/fused_attn4: REFUSED |
| Qwen3-30B-A3B | OK VALID | OK VALID | **1.457** | COMPARABLE | PASS | e4b/reference_attn4: OK; unsloth/ckpt_hf: OK |
| Gemma-4-26B-A4B-it | HARNESS_ERROR — | OK VALID | not quoted | — | NO-ARM | e4b/reference_attn4: HARNESS_ERROR |
| Mixtral-8x7B-Instruct-v0.1 | OK VALID | OK VALID | **0.361** | COMPARABLE | PASS | e4b/reference_attn4: OK |

## Predictions P1–P7 (P40, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | both OK but no VALID ratio: primary arms not both VALID: e4b fused_attn4 VALID / unsloth ckpt_unsloth VOID |
| P1 | olmoe | **FALSIFIED** | both train: e4b OK / unsloth HARNESS_ERROR |
| P2 | qwen3 | **HELD** | ratio 1.457 vs 1.413 (+3.1%); same fixture as P38: True |
| P3 | gemma4 | **UNTESTED** | Unsloth engaged; no ratio: primary arms not both VALID: e4b fused_attn4 — / unsloth ckpt_unsloth VALID |
| P4 | mixtral | **FALSIFIED** | unsloth OK / e4b fused (offload) OK; ratio 0.361 was quoted |
| P5 | gptoss | **HELD** | e4b fused REFUSED; unsloth REFUSED |
| P6 | all | **HELD** | qwen3 COMPARABLE (Δ +0.0152); mixtral COMPARABLE (Δ -0.0087) |
| P7 | five | **PARTIAL** | granite PASS; olmoe PASS; qwen3 PASS; gemma4 NO-ARM; mixtral PASS |
