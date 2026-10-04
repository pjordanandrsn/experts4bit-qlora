# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.42.0 @14a9a17d090e89a0a9ea546438013ea66d88c962 (GitHub main)
gnf4 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
unsloth(unsloth-t28) 2026.9.14
unsloth_zoo(unsloth-t28) 2026.9.9
torch(unsloth-t28) 2.8.0+cu128
triton(unsloth-t28) 3.4.0
transformers(unsloth-t28) 5.5.0
bitsandbytes(unsloth-t28) 0.50.2
peft(unsloth-t28) 0.21.2
torchao(unsloth-t28) None
moe_backend(unsloth-t28) native_torch
unsloth(unsloth) 2026.9.14
unsloth_zoo(unsloth) 2026.9.9
torch(unsloth) 2.12.1+cu130
triton(unsloth) 3.7.1
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft(unsloth) 0.21.2
torchao(unsloth) 0.18.0+cu130
moe_backend(unsloth) grouped_mm
e4b(t212) 0.42.0 @14a9a17d090e89a0a9ea546438013ea66d88c962
gnf4(t212) 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6
torch(e4b-t212) 2.12.1+cu130
axolotl 0.20.0
torch(axolotl) 2.14.0+cu130
transformers(axolotl) 5.17.0
peft(axolotl) 0.21.0
bitsandbytes(axolotl) 0.50.2
python(axolotl) 3.12.15
deepspeed(axolotl) None
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1-5090-49",
 "instance_id": "54075516",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "858436212",
 "cgroup_memory_max": "377392988160",
 "disk_root": "overlay         320G   52M  320G   1% /",
 "hostname": "3346d4c50ea4",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (`qwen3`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.944 | 341.8 | 27.210 | 1126.7 | 2.0614→0.8332 | 1.9441→0.8502 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0118) | 20 | 7.943 | 173.2 | 24.269 | 1296.9 | 2.0533→0.8343 | 1.9558→0.8498 | -0.0003 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0067) | 20 | 62.806 | 24.1 | 27.097 | 6275.3 | 2.0669→0.8315 | 1.9508→0.8499 | -0.0002 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.001 | 376.9 | 27.230 | 1048.3 | 2.0614→0.8331 | 1.9441→0.8524 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0118) | 20 | 7.924 | 185.6 | 24.269 | 1239.3 | 2.0533→0.8332 | 1.9558→0.8464 | -0.0037 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 13.06 MiB is free. Including non-PyTorch memory, this process has 31.34 GiB memory in use. Of the allocated memory 30.76 GiB is allocated by PyT |
| axolotl | ckpt_axolotl_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0323) | 20 | 7.863 | 190.6 | 26.882 | 2969.2 | 2.0554→0.8359 | 1.9764→0.8566 | 0.0065 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.373 | 342.6 | 27.210 | 541.9 | 2.0614→0.8317 | 1.9441→0.8508 | 0.0007 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **ALARM** | — | **ALARM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | arm alarm 2400 s (SIGALRM; the process could not write its own stub) |
| e4b | fused_attn4_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 6.840 | 220.3 | 26.055 | 1299.3 | 2.0660→0.8676 | 1.9441→0.8455 | -0.0046 | patched 48 / kcalls 1536 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| hf | hf_peft_m_mb1 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 13.06 MiB is free. Including non-PyTorch memory, this process has 31.34 GiB memory in use. Of the allocated memory 30.76 GiB is allocated by PyT |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0118) | 20 | 14.374 | 103.2 | 24.244 | 2076.3 | 2.1123→0.8683 | 1.9558→0.8436 | -0.0065 | stacks 96 / fwd 768 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **109.3 s** before step 1 (55% of the arm): c1_before 51.6, load_weights 28.6, c1_after 25.6, eval0 10.8; unattributed 1.232; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **114.0 s** before step 1 (39% of the arm): c1_before 45.6, load_weights 35.9, c1_after 23.8, eval0 15.0; unattributed 9.639; budget 1260.0
- prologue `e4b/reference_attn4_m` **122.6 s** before step 1 (9% of the arm): c1_before 51.8, load_weights 28.6, c1_after 26.3, eval0 23.8; unattributed 1.216; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **99.9 s** before step 1 (55% of the arm): c1_before 51.3, load_weights 28.4, c1_after 25.5, attn4 5.8; unattributed 1.21; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **105.1 s** before step 1 (38% of the arm): c1_before 45.9, load_weights 32.4, c1_after 23.1, eval0 9.0; unattributed 9.764; budget 1260.0
- prologue `axolotl/ckpt_axolotl_m` **103.7 s** before step 1 (39% of the arm): c1_before 46.5, load_weights 35.3, c1_after 25.8, eval0 7.6; unattributed 5.581; budget 945.0
- prologue `e4b/fused_attn4_m_prof` **93.6 s** before step 1 (29% of the arm): c1_before 52.6, load_weights 21.6, c1_after 16.3, attn4 5.3; unattributed 1.211; budget 840.0
- prologue `e4b/fused_attn4_m_mb1` **102.9 s** before step 1 (42% of the arm): c1_before 53.8, load_weights 28.1, c1_after 25.2, attn4 5.6; unattributed 1.217; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **101.1 s** before step 1 (25% of the arm): c1_before 49.0, load_weights 26.2, c1_after 24.2, eval0 9.3; unattributed 9.442; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (3.944/4.001 s, |Δ|/mean 1.4% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (7.943/7.924 s, |Δ|/mean 0.2% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `axolotl/ckpt_axolotl_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_mb1` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00172, median step |Δ| 0.00175 → **PASS** (band 0.05/0.05); ×15.92 faster per step, peak ×1.004
- **MATCHED POSITION: s/step ratio unsloth/e4b = 1.997** [1.980, 2.014 over 4 cross-draw ratios] (7.933 vs 3.973 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 27.22 GB (Δ -2.95); J/step unsloth 1268.1 vs e4b 1087.5 (×1.166); tok/s unsloth 179.4 vs e4b 359.4; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8502 / unsloth 0.8498 (Δ -0.0003) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0118
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is OOM
- **MATCHED POSITION: s/step ratio axolotl/e4b = 1.979** [1.965, 1.994 over 2 cross-draw ratios] (7.863 vs 3.973 s, medians over 1/2 draws; e4b faster per step); peak VRAM axolotl 26.88 vs e4b 27.22 GB (Δ -0.34); J/step axolotl 2969.2 vs e4b 1087.5 (×2.730); tok/s axolotl 190.6 vs e4b 359.4; axolotl regime: 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384)
- quality reading at N=20 (axolotl): held-out e4b 0.8502 / axolotl 0.8566 (Δ 0.0065) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0323
- **SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed): s/step ratio unsloth (mb1) / e4b = 2.101** [2.101, 2.101 over 1 cross-draw ratios] (14.374 vs 6.840 s, medians over 1/1 draws; e4b faster per step); peak VRAM unsloth (mb1) 24.24 vs e4b 26.05 GB (Δ -1.81); J/step unsloth (mb1) 2076.3 vs e4b 1299.3 (×1.598); tok/s unsloth (mb1) 103.2 vs e4b 220.3
- quality reading at N=20 (unsloth (mb1)): held-out e4b 0.8455 / unsloth (mb1) 0.8436 (Δ -0.0019) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0118
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (hf (mb1))** — not quoted: e4b single draw (no second draw registered) / hf (mb1) hf/hf_peft_m_mb1 is OOM
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (axolotl (mb1))** — not quoted: e4b single draw (no second draw registered) / axolotl (mb1) no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0052 / held-out 0.0050; COMPARABLE ≤ 0.05; draw-noise floor 0.0034): `unsloth/ckpt_unsloth_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0003, step-0 0.0118 NEAR, |Δ loss at step 2| 0.0010, paired rows mean -0.0003 ± 0.0025 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0003 / 0.0020 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement; `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0002, step-0 0.0067 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0045, paired rows mean -0.0003 ± 0.0028 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0002 / 0.0017 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0022, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0019, paired rows mean +0.0022 ± 0.0022 SE over 8, favouring arm 3 / anchor 4) — |delta| 0.0022 / 0.0010 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0018, |Δ held-out at N| 0.0037, step-0 0.0118 NEAR, |Δ loss at step 2| 0.0141, paired rows mean -0.0037 ± 0.0023 SE over 8, favouring arm 6 / anchor 2); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **COMPARABLE** (median step |Δ| 0.0076, |Δ held-out at N| 0.0065, step-0 0.0323 NEAR, |Δ loss at step 2| 0.0060, paired rows mean +0.0065 ± 0.0026 SE over 8, favouring arm 2 / anchor 6); `e4b/fused_attn4_m_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0007, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0042, paired rows mean +0.0007 ± 0.0030 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0007 / 0.0015 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_prof` **—** — no OK receipt; `e4b/fused_attn4_m_mb1` **COMPARABLE** (median step |Δ| 0.0159, |Δ held-out at N| 0.0046, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0012, paired rows mean -0.0047 ± 0.0034 SE over 8, favouring arm 6 / anchor 2); `hf/hf_peft_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_mb1` **COMPARABLE** (median step |Δ| 0.0196, |Δ held-out at N| 0.0065, step-0 0.0118 NEAR, |Δ loss at step 2| 0.0073, paired rows mean -0.0065 ± 0.0043 SE over 8, favouring arm 5 / anchor 3)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `axolotl/ckpt_axolotl_m` same; `e4b/fused_attn4_m_prof` same; `e4b/fused_attn4_m_mb1` same; `unsloth/ckpt_unsloth_m_mb1` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_mb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_mb1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0003 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | -0.0002 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0022 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0037 |  |
| qwen3 | hf/hf_peft_m | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 13.06 MiB is free. Including non-PyTorch mem |
| qwen3 | axolotl/ckpt_axolotl_m | **VALID** | VALID | 0.0065 |  |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0007 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **ALARM** | — | — | arm alarm 2400 s (SIGALRM; the process could not write its own stub) |
| qwen3 | e4b/fused_attn4_m_mb1 | **VALID** | VALID | -0.0046 |  |
| qwen3 | hf/hf_peft_m_mb1 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 13.06 MiB is free. Including non-PyTorch mem |
| qwen3 | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | -0.0065 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 1.997 vs [2.0, 5.0]; outside the band |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 1.4% vs 5% -> STABLE; unsloth |d1-d2|/mean 0.2% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m INSIDE-DRAW-NOISE (median step |Δ| 0.0017, |Δ held-out at N| 0.0002, band {'train': 0.005249999999999921, 'heldout': 0.005}); unsloth/ckpt_unsloth_m INSIDE-DRAW-NOISE (median step |Δ| 0.0020, |Δ held-out at N| 0.0003, band {'train': 0.005249999999999921, 'heldout': 0.005}); draw-noise floor 0.0034 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **HELD** | hf_peft_m OOM and hf_peft_m_mb1 OOM |
| P6 | qwen3 | **HELD** | axolotl trained; axolotl/e4b 1.979 vs [1.5, 6] |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **UNTESTED** | ckpt_unsloth_prof ALARM carries no profile summary |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00172, median 0.00175) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
