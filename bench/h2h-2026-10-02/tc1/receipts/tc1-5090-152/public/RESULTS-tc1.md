# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.49.0 @737f02ff45319f06116f75c6d4e5ed29970e5c5e (GitHub main)
gnf4 0.43.0 @8371c38dc6a4ebfbf54c850bf9cbb53a4120a258 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
huggingface_hub(e4b/hf) 2.2.0
httpx2(e4b/hf) 2.16.0
brotli(e4b/hf) 1.2.0
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
e4b(t212) 0.49.0 @737f02ff45319f06116f75c6d4e5ed29970e5c5e
gnf4(t212) 0.43.0 @8371c38dc6a4ebfbf54c850bf9cbb53a4120a258
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
 "run_id": "tc1-5090-152",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.119.02",
 "cpu": "AMD EPYC 9655 96-Core Processor",
 "nproc": 48,
 "mem_total_kb": "113302948",
 "cgroup_memory_max": "111380791296",
 "disk_root": "overlay         220G  2.8M  220G   1% /",
 "cgroup_cpu_max": "4608000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 73: amendment 69's field recipe with the memory census -- e4b defaults vs Unsloth at the claim's pins, one draw each; no speed read) (`qwen3pos69mc`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_fd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.159 | 617.8 | 26.156 | 952.9 | 2.0722→0.8159 | 1.9592→0.7566 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_fv | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0086) | 60 | 5.438 | 274.1 | 24.273 | 874.7 | 2.0593→0.8176 | 1.9506→0.7574 | 0.0008 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_fd` **73.0 s** before step 1 (32% of the arm): c1_before 27.6, load_weights 26.2, c1_after 15.5, eval0 8.1; unattributed 1.602; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_fv` **80.6 s** before step 1 (18% of the arm): load_weights 31.1, c1_before 26.5, c1_after 14.0, eval0 10.6; unattributed 7.183; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_fd` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_fv` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: the memory census slows the step (amendment 73, as amendment 23), so no speed is read on this token
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: the memory census slows the step (amendment 73, as amendment 23), so no speed is read on this token
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: the memory census slows the step (amendment 73, as amendment 23), so no speed is read on this token
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `unsloth/ckpt_unsloth_m_fv` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0008, step-0 0.0086 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0145, paired rows mean +0.0008 ± 0.0013 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_fd` same; `unsloth/ckpt_unsloth_m_fv` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m_fv`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos69mc | e4b/fused_attn4_m_fd | **VALID** | VALID | 0.0000 |  |
| qwen3pos69mc | unsloth/ckpt_unsloth_m_fv | **VALID** | VALID | 0.0008 |  |

## Amendment 73: the field recipe's memory gap, attributed (descriptive: no prediction, no verdict; GB = 1e9 bytes)

| | e4b defaults (fd) | Unsloth (fv) | e4b − Unsloth |
|---|---|---|---|
| peak_vram_gb vs the claim's two draws | 26.156 (claim 26.156–26.165): reproduced | 24.273 (claim 24.273–24.273): reproduced | |
| end of setup: allocated (GB) | 19.297 | 19.302 | -0.004 |
| end of setup: reserved (GB) | 20.160 | 19.447 | +0.713 |
| training window: peak allocated (GB) | 26.156 | 24.273 | +1.883 |
| the peak sits at | s22.mb2.backward, attributed 0.9987 | s18.mb3.backward, attributed 1.0 | |

The end-of-setup rows answer the load-phase device gap; the training-window row answers the claim. Neither is read as evidence about the other. CUDA context and non-PyTorch device memory are not in the census and are not read here.

## Amendment 73: the census tables (Qwen3-30B-A3B, TC1's field recipe, one draw per arm; GB = 1e9 bytes; no speed is read)

| | e4b defaults (fd) | Unsloth (fv) |
|---|---|---|
| peak allocated / reserved (GB) | 26.156 / 26.697 | 24.273 / 24.606 |
| attributed fraction at the peak | 0.9987 | 1.0000 |
| peak at (checkpoint / phase) | s22.mb2 / s22.mb2.backward | s18.mb3 / s18.mb3.backward |
| window holds the run's peak / events / ring full | True / 1000000 / True | True / 1000000 / True |
| snapshots / census s / torch | 8 / 20.33 / 2.12.1+cu130 | 6 / 11.9 / 2.12.1+cu130 |

**Static census by class (GB: end of setup / end of training)**

| | e4b defaults (fd) | Unsloth (fv) |
|---|---|---|
| frozen_expert_weights | 14.496 / 14.496 | 14.496 / 14.496 |
| expert_absmax | 0.460 / 0.460 | 0.460 / 0.460 |
| expert_absmax.bnb.absmax | — / — | 0.453 / 0.453 |
| expert_absmax.bnb.code | — / — | 0.000 / 0.000 |
| expert_absmax.bnb.offset | — / — | 0.000 / 0.000 |
| expert_absmax.bnb.state2.absmax | — / — | 0.007 / 0.007 |
| expert_absmax.bnb.state2.code | — / — | 0.000 / 0.000 |
| expert_absmax.nested_code | 0.000 / 0.000 | — / — |
| expert_absmax.nested_off | 0.000 / 0.000 | — / — |
| expert_absmax.nested_q | 0.453 / 0.453 | — / — |
| expert_absmax.nested_s | 0.007 / 0.007 | — / — |
| other_frozen | 1.738 / 1.738 | 1.738 / 1.738 |
| other_frozen.bf16 | 1.270 / 1.270 | 1.270 / 1.270 |
| other_frozen.nf4_linear4bit | 0.468 / 0.468 | 0.468 / 0.468 |
| trainable_adapters | 2.570 / 2.570 | 2.570 / 2.570 |
| adapter_grads | 0.000 / 0.000 | 0.000 / 0.000 |
| optimizer_state | 0.000 / 1.305 | 0.000 / 1.305 |
| other_buffers | 0.000 / 0.000 | 0.000 / 0.000 |
| other | 0.034 / 0.116 | 0.038 / 0.078 |
| allocated_bytes | 19.297 / 20.685 | 19.302 / 20.647 |

**Live bytes at the peak by class (GB; transient = every live block that holds no static tensor)**

| | e4b defaults (fd) | Unsloth (fv) |
|---|---|---|
| frozen_expert_weights | 14.496 | 14.496 |
| expert_absmax | 0.460 | 0.460 |
| other_frozen | 1.738 | 1.738 |
| trainable_adapters | 2.570 | 2.570 |
| adapter_grads | 2.570 | 2.570 |
| optimizer_state | 1.305 | 1.305 |
| other_buffers | 0.000 | 0.000 |
| transient | 3.017 | 1.130 |

**Top live-at-peak groups (GB ×count; 20 of the receipts' 40 shown)**

| rank | e4b defaults (fd) | Unsloth (fv) |
|---|---|---|
| 1 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 |
| 2 | static:adapter_grads 2.570 ×576 | static:adapter_grads 2.570 ×576 |
| 3 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 |
| 4 | static:optimizer_state 1.305 ×2306 | static:optimizer_state 1.305 ×2306 |
| 5 | static:other_frozen.bf16 1.270 ×243 | static:other_frozen.bf16 1.270 ×243 |
| 6 | site:nf4_qlora.py:582 _lora_delta_padded 1.117 ×2 | site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 ×1 |
| 7 | site:nf4_qlora.py:584 _lora_delta_padded 0.825 ×3 | static:other_frozen.nf4_linear4bit 0.468 ×1152 |
| 8 | static:other_frozen.nf4_linear4bit 0.468 ×1152 | static:expert_absmax 0.460 ×480 |
| 9 | static:expert_absmax 0.460 ×384 | site:torch/autograd/graph.py:882 _engine_run_backward 0.116 ×7 |
| 10 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 | site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.036 ×1 |
| 11 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.181 ×39 | site:bitsandbytes/backends/cuda/ops.py:916 _dequant_linear_fallback 0.034 ×1 |
| 12 | site:nf4_qlora.py:386 forward 0.124 ×5 | site:unsloth_zoo/temporary_patches/moe_utils.py:920 _probe_torch_grouped_mm_supported 0.034 ×1 |
| 13 | site:nf4_qlora.py:583 _lora_delta_padded 0.102 ×2 | site:Linear4bit_peft_forward.py:49 lora_forward 0.023 ×12 |
| 14 | site:experts4bit_qlora/lora.py:787 forward 0.047 ×8 | site:transformers/integrations/sdpa_attention.py:25 repeat_kv 0.018 ×2 |
| 15 | site:experts4bit_qlora/engines/fast.py:752 fused_experts_train_forward 0.037 ×1 | site:transformers/integrations/sdpa_attention.py:92 sdpa_attention_forward 0.010 ×3 |
| 16 | site:nf4_grouped.py:1830 gemm_4bit_grouped 0.037 ×1 | site:Linear4bit_peft_forward.py:52 lora_forward 0.010 ×2 |
| 17 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.009 ×2 |
| 18 | unattributed:allocated before the census window, no frame 0.034 ×1 | site:unsloth_compiled_module_qwen3_moe.py:239 apply_rotary_pos_emb 0.009 ×1 |
| 19 | site:nf4_qlora.py:978 fused_grouped_lora 0.028 ×1 | site:torch/nn/functional.py:2567 embedding 0.005 ×1 |
| 20 | site:transformers/integrations/sdpa_attention.py:25 repeat_kv 0.019 ×2 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:346 forward 0.005 ×1 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos69mc/e4b/fused_attn4_m_fd attempt 0 load1_median 0.99 gate 6.0 status ok over 0`
- `qwen3pos69mc/unsloth/ckpt_unsloth_m_fv attempt 0 load1_median 1.14 gate 6.0 status ok over 0`
