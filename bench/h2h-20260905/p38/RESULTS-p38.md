# p38 -- e4b vs Unsloth QLoRA end-to-end, one box, one training problem (/root/p38)
Rule: positions in registered units (s/step median of steps 11..N, GB allocator peak, tok/s, J/step, held-out loss in nats); VOID arms never enter a ratio; cross-framework held-out |Δ| ≤ 0.05 reads 'comparable quality' (a reading threshold, not a gate) and outside it time-to-target (0.32) is the headline; the e4b fused-vs-reference band (0.05/0.05) is informational here (tp1 owns it). Nothing is licensed; no cross-box number is divided into these.
tokens sha: 81dc24c3c195667fd7d0829a5f73f401e91708bd911ef21cb89794515cd1e583
```
e4b 0.35.0 @
gnf4 0.30.0
torch(e4b) 2.8.0+cu128
triton(e4b) 3.4.0
transformers(e4b) 5.16.1
bitsandbytes(e4b) 0.50.1
unsloth 2026.9.2
unsloth_zoo 2026.9.1
torch(unsloth) 2.8.0+cu128
triton(unsloth) 3.4.0
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft 0.20.0
moe_backend native_torch
unsloth 2026.9.2
unsloth_zoo 2026.9.1
torch(unsloth) 2.8.0+cu128
triton(unsloth) 3.4.0
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft 0.20.0
moe_backend native_torch
unsloth 2026.9.2
unsloth_zoo 2026.9.1
torch(unsloth) 2.8.0+cu128
triton(unsloth) 3.4.0
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft 0.20.0
moe_backend native_torch
```
| framework | arm | status | validity | N | s/step med(11+) | s/step mean | tok/s | peak GB | J/step | train first→last | held-out 0→final | t→target s | adapter MB (dtype) | trainable | ckpt | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused | ok | **VALID** | 60 | 1.473 | 1.951 | 58.8 | 22.715 | 145.6 | 3.6185→0.2642 | 3.7362→0.2851 | 58.4 | 656.1 (torch.bfloat16,torch.float32) | 321257472 | hf:use_reentrant=False |  |
| e4b | fused_attn4 | ok | **VALID** | 60 | 1.522 | 2.053 | 56.0 | 21.371 | 157.1 | 3.6282→0.2629 | 3.7022→0.2923 | 92.5 | 656.1 (torch.bfloat16,torch.float32) | 321257472 | hf:use_reentrant=False |  |
| e4b | fused_attn4_200 | ok | **VALID** | 200 | 1.515 | 2.035 | 56.6 | 21.372 | 177.7 | 3.6282→0.3017 | 3.7022→0.2881 | 91.5 | 656.1 (torch.bfloat16,torch.float32) | 321257472 | hf:use_reentrant=False |  |
| e4b | fused_attn4_nosamp | ok | **VALID** | 20 | 1.530 | 2.042 | 56.4 | 21.371 | — | 3.6282→0.3375 | 3.7022→0.3459 | not reached by 20 steps | 656.1 (torch.bfloat16,torch.float32) | 321257472 | hf:use_reentrant=False |  |
| e4b | reference_attn4 | ok | **VALID** | 60 | 4.437 | 6.607 | 19.5 | 21.371 | 268.1 | 3.6535→0.2642 | 3.7021→0.2909 | 176.7 | 656.1 (torch.bfloat16,torch.float32) | 321257472 | hf:use_reentrant=False |  |
| unsloth | ckpt_hf | ok | **VALID** | 60 | 2.146 | 3.493 | 39.8 | 23.137 | 221.3 | 3.6550→0.2606 | 3.6968→0.2975 | 130.1 | 1285.1 (F32) | 321257472 | hf:True (via get_peft_model) |  |
| unsloth | ckpt_unsloth | ok | **VALID** | 60 | 2.151 | 3.499 | 39.7 | 23.141 | 224.7 | 3.6550→0.2606 | 3.6968→0.2975 | 130.3 | 1285.1 (F32) | 321257472 | unsloth |  |
| unsloth | ckpt_unsloth_200 | ok | **VALID** | 200 | 2.157 | 3.522 | 39.2 | 23.141 | 258.3 | 3.6550→0.2795 | 3.6968→0.2713 | 135.9 | 1285.1 (F32) | 321257472 | unsloth |  |

- energy control: sampler perturbation 1.61% (steps 11..20: sampled 1554 ms vs unsampled 1530 ms) -> energy reported without caveat
- e4b internal parity (informational; tp1 owns the licence): fused_attn4 vs reference_attn4 Δfinal 0.00131, median step |Δ| 0.01138 -> **PASS** in the B2/C2 band; cost ×2.92 faster per step, peak ×1.000

## Primary pair -- Unsloth (4-bit MoE, 'unsloth' checkpointing) vs e4b (fused dgrad path + NF4 attention), N=60
- **s/step ratio unsloth/e4b = 1.413** (2.151 vs 1.522 s; e4b faster per step at this workload)
- peak VRAM: unsloth 23.14 GB vs e4b 21.37 GB (Δ +1.77 GB)
- tokens/s: unsloth 39.7 vs e4b 56.0; J/step: unsloth 224.7 vs e4b 157.1 (sampler perturbation 1.61% (steps 11..20: sampled 1554 ms vs unsampled 1530 ms) -> energy reported without caveat)
- held-out loss at shared evals: 0: e4b 3.7022 / unsloth 3.6968 (Δ -0.0055 ≤ 0.05); 20: e4b 0.3489 / unsloth 0.5798 (Δ +0.2309 > 0.05); 40: e4b 0.3205 / unsloth 0.3672 (Δ +0.0467 ≤ 0.05); 60: e4b 0.2923 / unsloth 0.2975 (Δ +0.0052 ≤ 0.05)
- reading: COMPARABLE QUALITY at N=60 (|Δ| ≤ 0.05) -- the s/step ratio is the position
- time-to-target (held-out ≤ 0.32, cumulative training wall, evals excluded): e4b 92.5 s (60-step arm); unsloth 130.3 s (60-step arm)
- adapters: e4b 656.1 MB ['torch.bfloat16', 'torch.float32'] / unsloth 1285.1 MB ['F32']; both 321257472 parameters (asserted)
- step-0 held-out (the two quantisers on the same bytes, B=0): e4b 3.7022 vs unsloth 3.6968 (Δ -0.0055)

## Secondary rows (same columns; never the headline)
- e4b tp1 fixture (bf16 attention): VALID s/step 1.473, peak 22.71 GB, held-out 3.7362→0.2851; vs the primary e4b arm: s/step ×0.968, peak +1.34 GB
- Unsloth with HF checkpointing (the semantic match to e4b's mode): VALID s/step 2.146, peak 23.14 GB, held-out 3.6968→0.2975; vs the primary unsloth arm: s/step ×0.998, peak -0.00 GB

## 200-step curves (held-out loss at every eval; cumulative training wall)
- e4b (VALID): 0:3.7022@0s · 20:0.3504@31s · 40:0.3200@61s · 60:0.2848@92s · 80:0.2873@122s · 100:0.3184@152s · 120:0.3004@182s · 140:0.2933@212s · 160:0.2947@243s · 180:0.2900@273s · 200:0.2881@303s
- unsloth (VALID): 0:3.6968@0s · 20:0.5798@49s · 40:0.3672@92s · 60:0.2975@136s · 80:0.2918@179s · 100:0.2894@222s · 120:0.2844@265s · 140:0.2799@309s · 160:0.2768@352s · 180:0.2753@395s · 200:0.2713@438s
