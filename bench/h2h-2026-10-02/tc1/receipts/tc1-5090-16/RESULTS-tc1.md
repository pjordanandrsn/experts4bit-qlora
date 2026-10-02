# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.38.0 @079a4220b4b334903b0e77725eb94a0f58848018 (GitHub main)
gnf4 0.34.0 @846b512b905468c08f5748943d08769b572affa2 (GitHub main)
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
e4b(t212) 0.38.0 @079a4220b4b334903b0e77725eb94a0f58848018
gnf4(t212) 0.34.0 @846b512b905468c08f5748943d08769b572affa2
torch(e4b-t212) 2.12.1+cu130
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1-5090-16",
 "instance_id": "53773622",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.178.04",
 "cpu": "AMD Ryzen Threadripper PRO 3975WX 32-Cores",
 "nproc": 64,
 "mem_total_kb": "263745776",
 "cgroup_memory_max": "183333027840",
 "disk_root": "overlay         320G  1.7M  320G   1% /",
 "hostname": "ad355743cef9",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md"
}
```

### Qwen3-30B-A3B (`qwen3`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.692 | 261.8 | 27.849 | 935.4 | 2.0705→0.8318 | 1.9505→0.8516 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0078) | 20 | 8.160 | 171.5 | 24.269 | 636.4 | 2.0705→0.8319 | 1.9583→0.8482 | -0.0034 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0003) | 20 | 59.171 | 25.4 | 27.097 | 3210.9 | 2.0669→0.8314 | 1.9508→0.8475 | -0.0041 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.683 | 268.7 | 27.819 | 907.2 | 2.0705→0.8320 | 1.9505→0.8487 | -0.0030 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0078) | 20 | 8.183 | 181.3 | 24.269 | 684.9 | 2.0705→0.8305 | 1.9583→0.8476 | -0.0040 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 11.75 MiB is free. Including non-PyTorch memory, this process has 31.34 GiB memory in use. Of the allocated memory 30.76 GiB is allocated by PyT |
| axolotl | ckpt_axolotl_m | **INSTALL_FAILED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | axolotl venv install failed rc=1 (logs/pip_axolotl.log):          And because you require axolotl==0.20.0, we can conclude that your requirements are unsatisfiable.  hint: `packaging` was found on https://download.pytorch.org/whl/cu130, but not at the requeste |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 6.169 | 242.6 | 27.822 | 432.1 | 2.0705→0.8335 | 1.9505→0.8518 | 0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0078) | 20 | 9.868 | 134.4 | 24.269 | -198.1 | 2.0705→0.8307 | 1.9583→0.8502 | -0.0014 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 8.706 | 176.4 | 26.418 | 947.9 | 2.0727→0.8688 | 1.9505→0.8485 | -0.0031 | patched 48 / kcalls 1536 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_prof_profile | **HARNESS_ERROR** | — | **HARNESS_ERROR** | native | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — |  |
| hf | hf_peft_m_mb1 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 11.75 MiB is free. Including non-PyTorch memory, this process has 31.34 GiB memory in use. Of the allocated memory 30.76 GiB is allocated by PyT |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0078) | 20 | 15.021 | 99.3 | 24.244 | 1120.0 | 2.0605→0.8750 | 1.9583→0.8431 | -0.0085 | stacks 96 / fwd 768 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_prof_profile | **HARNESS_ERROR** | — | **HARNESS_ERROR** | native | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — |  |
- prologue `e4b/fused_attn4_m` **149.7 s** before step 1 (56% of the arm): load_weights 63.2, c1_before 56.1, c1_after 27.9, eval0 14.4; unattributed 1.225; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **104.7 s** before step 1 (37% of the arm): c1_before 49.8, c1_after 25.2, load_weights 23.6, eval0 15.2; unattributed 9.672; budget 1260.0
- prologue `e4b/reference_attn4_m` **110.1 s** before step 1 (8% of the arm): c1_before 53.5, c1_after 27.8, eval0 21.5, load_weights 19.5; unattributed 1.08; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **94.3 s** before step 1 (45% of the arm): c1_before 55.3, c1_after 27.8, load_weights 19.8, attn4 5.5; unattributed 1.09; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **97.0 s** before step 1 (36% of the arm): c1_before 49.4, c1_after 25.0, load_weights 23.4, eval0 9.1; unattributed 8.781; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **92.9 s** before step 1 (23% of the arm): c1_before 54.9, load_weights 19.8, c1_after 10.6, attn4 5.8; unattributed 1.116; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **98.0 s** before step 1 (4% of the arm): c1_before 50.3, load_weights 23.4, c1_after 9.1, eval0 9.1; unattributed 8.691; budget 840.0
- prologue `e4b/fused_attn4_m_mb1` **96.6 s** before step 1 (36% of the arm): c1_before 54.8, c1_after 27.1, load_weights 21.3, attn4 6.1; unattributed 1.223; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **97.0 s** before step 1 (24% of the arm): c1_before 49.5, c1_after 25.0, load_weights 23.5, eval0 9.1; unattributed 8.608; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (5.692/5.683 s, |Δ|/mean 0.2% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (8.160/8.183 s, |Δ|/mean 0.3% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_mb1` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00041, median step |Δ| 0.00141 → **PASS** (band 0.05/0.05); ×10.39 faster per step, peak ×1.028
- **MATCHED POSITION: s/step ratio unsloth/e4b = 1.437** [1.434, 1.440 over 4 cross-draw ratios] (8.171 vs 5.688 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 27.83 GB (Δ -3.57); J/step unsloth 660.7 vs e4b 921.3 (×0.717); tok/s unsloth 176.4 vs e4b 265.2; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8516 / unsloth 0.8482 (Δ -0.0034) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0078
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is OOM
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is UNSUPPORTED
- **SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed): s/step ratio unsloth (mb1)/e4b = 1.725** [1.725, 1.725 over 1 cross-draw ratios] (15.021 vs 8.706 s, medians over 1/1 draws; e4b faster per step); peak VRAM unsloth (mb1) 24.24 vs e4b 26.42 GB (Δ -2.17); J/step unsloth (mb1) 1120.0 vs e4b 947.9 (×1.182); tok/s unsloth (mb1) 99.3 vs e4b 176.4
- quality reading at N=20 (unsloth (mb1)): held-out e4b 0.8485 / unsloth (mb1) 0.8431 (Δ -0.0054) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0078
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (hf (mb1))** — not quoted: e4b single draw (no second draw registered) / hf (mb1) hf/hf_peft_m_mb1 is OOM
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (axolotl (mb1))** — not quoted: e4b single draw (no second draw registered) / axolotl (mb1) no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0050 / held-out 0.0123; COMPARABLE ≤ 0.05; draw-noise floor 0.0030): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0011, |Δ held-out at N| 0.0034, step-0 0.0078 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0113, paired rows mean -0.0034 ± 0.0028 SE over 8, favouring arm 3 / anchor 5); `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0014, |Δ held-out at N| 0.0041, step-0 0.0003 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0065, paired rows mean -0.0041 ± 0.0019 SE over 8, favouring arm 7 / anchor 1); `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0030, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0001, paired rows mean -0.0029 ± 0.0034 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0030 / 0.0009 are narrower than the draw-noise floor 0.0030: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0011, |Δ held-out at N| 0.0040, step-0 0.0078 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0191, paired rows mean -0.0040 ± 0.0025 SE over 8, favouring arm 5 / anchor 3); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0006, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0117, paired rows mean +0.0002 ± 0.0032 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0002 / 0.0006 are narrower than the draw-noise floor 0.0030: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0014, step-0 0.0078 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0054, paired rows mean -0.0014 ± 0.0026 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0014 / 0.0017 are narrower than the draw-noise floor 0.0030: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_mb1` **COMPARABLE** (median step |Δ| 0.0179, |Δ held-out at N| 0.0031, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0156, paired rows mean -0.0032 ± 0.0048 SE over 8, favouring arm 4 / anchor 4); `hf/hf_peft_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_mb1` **COMPARABLE** (median step |Δ| 0.0180, |Δ held-out at N| 0.0085, step-0 0.0078 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0007, paired rows mean -0.0085 ± 0.0033 SE over 8, favouring arm 6 / anchor 2)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same; `e4b/fused_attn4_m_mb1` same; `unsloth/ckpt_unsloth_m_mb1` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_mb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_mb1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.234, device events/step 612179, CPU ops/step 7430931, CPU self by family {'other': 0.2907, 'matmul': 0.2565, 'norm_act': 0.1707, 'fused_kernel': 0.1576, 'autograd': 0.0665, 'memcpy': 0.0509, 'routing': 0.0054, 'optimizer': 0.0017}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0034 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | -0.0041 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0030 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0040 |  |
| qwen3 | hf/hf_peft_m | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 11.75 MiB is free. Including non-PyTorch mem |
| qwen3 | axolotl/ckpt_axolotl_m | **UNSUPPORTED** | — | — | axolotl venv install failed rc=1 (logs/pip_axolotl.log):          And because you require axolotl==0.20.0, we can conclude that your requirements are unsatisfia |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0002 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | -0.0014 |  |
| qwen3 | e4b/fused_attn4_m_mb1 | **VALID** | VALID | -0.0031 |  |
| qwen3 | e4b/fused_attn4_m_prof_profile | **HARNESS_ERROR** | — | — |  |
| qwen3 | hf/hf_peft_m_mb1 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 11.75 MiB is free. Including non-PyTorch mem |
| qwen3 | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | -0.0085 |  |
| qwen3 | unsloth/ckpt_unsloth_prof_profile | **HARNESS_ERROR** | — | — |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 1.437 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 0.2% vs 5% -> STABLE; unsloth |d1-d2|/mean 0.3% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m EQUIVALENT (median step |Δ| 0.0014, |Δ held-out at N| 0.0041, band {'train': 0.005, 'heldout': 0.012299999999999978}); unsloth/ckpt_unsloth_m EQUIVALENT (median step |Δ| 0.0011, |Δ held-out at N| 0.0034, band {'train': 0.005, 'heldout': 0.012299999999999978}); draw-noise floor 0.0030 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **HELD** | hf_peft_m OOM and hf_peft_m_mb1 OOM |
| P6 | qwen3 | **HELD** | axolotl UNSUPPORTED |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.234 (>= 0.5 predicted on the grouped_mm arm); device events/step 612179 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00041, median 0.00141) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
