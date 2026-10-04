# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @a35a597d526a597aaa178ce12912ad50a8d42823 (GitHub main)
gnf4 0.38.0 @bb56b42c574d043df9428b85b17066f5bf6c092a (GitHub main)
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
e4b(t212) 0.45.0 @a35a597d526a597aaa178ce12912ad50a8d42823
gnf4(t212) 0.38.0 @bb56b42c574d043df9428b85b17066f5bf6c092a
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
 "run_id": "tc1-5090-65",
 "instance_id": "54195008",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9454P 48-Core Emb Processor",
 "nproc": 96,
 "mem_total_kb": "527772888",
 "cgroup_memory_max": "518820724736",
 "disk_root": "overlay         320G   56M  320G   1% /",
 "hostname": "16b54e775e08",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 23: the memory census -- e4b fp32 absmax, e4b double-quantized absmax, Unsloth; micro-batch 1 × accum 8; no speed read) (`qwen3memcensus`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 1 × accum 8 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.331 | 250.9 | 26.022 | 1222.6 | 2.0660→0.8681 | 1.9441→0.8454 | 0.0000 | patched 48 / kcalls 1536 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_mb1_dq | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0190) | 20 | 5.464 | 250.6 | 24.676 | 1208.3 | 2.0716→0.8704 | 1.9251→0.8478 | 0.0024 | patched 48 / kcalls 1536 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0083) | 20 | 11.933 | 116.9 | 24.244 | 1946.5 | 2.0805→0.8680 | 1.9523→0.8513 | 0.0059 | stacks 96 / fwd 768 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_mb1` **85.1 s** before step 1 (41% of the arm): c1_before 45.1, load_weights 17.3, c1_after 10.6, eval0 8.6; unattributed 0.998; budget 1260.0
- prologue `e4b/fused_attn4_m_mb1_dq` **78.2 s** before step 1 (39% of the arm): c1_before 43.0, load_weights 19.7, c1_after 9.8, attn4 4.5; unattributed 0.974; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **89.3 s** before step 1 (25% of the arm): c1_before 42.6, c1_after 21.9, load_weights 20.5, eval0 12.1; unattributed 7.796; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_mb1` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_mb1_dq` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (unsloth (mb1))** — not quoted: the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (hf (mb1))** — not quoted: the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (axolotl (mb1))** — not quoted: the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_mb1_dq` **COMPARABLE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0024, step-0 0.0190 NEAR, |Δ loss at step 2| 0.0010, paired rows mean +0.0024 ± 0.0032 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_mb1` **COMPARABLE** (median step |Δ| 0.0024, |Δ held-out at N| 0.0059, step-0 0.0083 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0004, paired rows mean +0.0059 ± 0.0038 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_mb1` same; `e4b/fused_attn4_m_mb1_dq` same; `unsloth/ckpt_unsloth_m_mb1` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_mb1_dq`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_mb1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3memcensus | e4b/fused_attn4_m_mb1 | **VALID** | VALID | 0.0000 |  |
| qwen3memcensus | e4b/fused_attn4_m_mb1_dq | **VALID** | VALID | 0.0024 |  |
| qwen3memcensus | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | 0.0059 |  |

## Amendment 23: the memory census (Qwen3-30B-A3B, micro-batch 1 × accum 8, one draw per arm; GB = 1e9 bytes; no speed is read)

| | e4b fp32 absmax | e4b dq absmax | Unsloth |
|---|---|---|---|
| peak allocated / reserved (GB) | 26.022 / 26.313 | 24.676 / 24.998 | 24.244 / 24.480 |
| attributed fraction at the peak | 0.9996 | 0.9996 | 1.0000 |
| peak at (checkpoint / phase) | s18.mb6 / s18.mb6.backward | s18.mb6 / s18.mb6.backward | s2.mb3 / s2.mb3.backward |
| window holds the run's peak / events / ring full | True / 1000000 / True | True / 1000000 / True | True / 720830 / False |
| snapshots / census s / torch | 9 / 13.13 / 2.8.0+cu128 | 9 / 13.24 / 2.8.0+cu128 | 6 / 12.64 / 2.12.1+cu130 |

**Static census by class (GB: end of setup / end of training)**

| | e4b fp32 absmax | e4b dq absmax | Unsloth |
|---|---|---|---|
| frozen_expert_weights | 14.496 / 14.496 | 14.496 / 14.496 | 14.496 / 14.496 |
| expert_absmax | 1.812 / 1.812 | 0.460 / 0.460 | 0.460 / 0.460 |
| expert_absmax.bnb.absmax | — / — | — / — | 0.453 / 0.453 |
| expert_absmax.bnb.code | — / — | — / — | 0.000 / 0.000 |
| expert_absmax.bnb.offset | — / — | — / — | 0.000 / 0.000 |
| expert_absmax.bnb.state2.absmax | — / — | — / — | 0.007 / 0.007 |
| expert_absmax.bnb.state2.code | — / — | — / — | 0.000 / 0.000 |
| expert_absmax.fp32 | 1.812 / 1.812 | — / — | — / — |
| expert_absmax.nested_code | — / — | 0.000 / 0.000 | — / — |
| expert_absmax.nested_off | — / — | 0.000 / 0.000 | — / — |
| expert_absmax.nested_q | — / — | 0.453 / 0.453 | — / — |
| expert_absmax.nested_s | — / — | 0.007 / 0.007 | — / — |
| other_frozen | 1.738 / 1.738 | 1.738 / 1.738 | 1.738 / 1.738 |
| other_frozen.bf16 | 1.270 / 1.270 | 1.270 / 1.270 | 1.270 / 1.270 |
| other_frozen.nf4_linear4bit | 0.468 / 0.468 | 0.468 / 0.468 | 0.468 / 0.468 |
| trainable_adapters | 2.570 / 2.570 | 2.570 / 2.570 | 2.570 / 2.570 |
| adapter_grads | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| optimizer_state | 0.000 / 1.305 | 0.000 / 1.305 | 0.000 / 1.305 |
| other_buffers | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| other | 0.009 / 0.035 | 0.009 / 0.036 | 0.038 / 0.073 |
| allocated_bytes | 20.624 / 21.956 | 19.272 / 20.604 | 19.302 / 20.641 |

**Live bytes at the peak by class (GB; transient = every live block that holds no static tensor)**

| | e4b fp32 absmax | e4b dq absmax | Unsloth |
|---|---|---|---|
| frozen_expert_weights | 14.496 | 14.496 | 14.496 |
| expert_absmax | 1.812 | 0.460 | 0.460 |
| other_frozen | 1.738 | 1.738 | 1.738 |
| trainable_adapters | 2.570 | 2.570 | 2.570 |
| adapter_grads | 2.570 | 2.570 | 2.570 |
| optimizer_state | 1.305 | 1.305 | 1.305 |
| other_buffers | 0.000 | 0.000 | 0.000 |
| transient | 1.532 | 1.536 | 1.095 |

**Top live-at-peak groups (GB ×count; 20 of the receipts' 40 shown)**

| rank | e4b fp32 absmax | e4b dq absmax | Unsloth |
|---|---|---|---|
| 1 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 |
| 2 | static:adapter_grads 2.570 ×576 | static:adapter_grads 2.570 ×576 | static:adapter_grads 2.570 ×576 |
| 3 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 |
| 4 | static:expert_absmax 1.812 ×96 | static:optimizer_state 1.305 ×2306 | static:optimizer_state 1.305 ×2306 |
| 5 | static:optimizer_state 1.305 ×2306 | static:other_frozen.bf16 1.270 ×243 | static:other_frozen.bf16 1.270 ×243 |
| 6 | static:other_frozen.bf16 1.270 ×243 | site:nf4_qlora.py:444 _lora_delta_padded 0.615 ×2 | site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 ×1 |
| 7 | site:nf4_qlora.py:444 _lora_delta_padded 0.629 ×2 | static:other_frozen.nf4_linear4bit 0.468 ×1152 | static:other_frozen.nf4_linear4bit 0.468 ×1152 |
| 8 | static:other_frozen.nf4_linear4bit 0.468 ×1152 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×480 |
| 9 | site:nf4_qlora.py:446 _lora_delta_padded 0.465 ×3 | site:nf4_qlora.py:446 _lora_delta_padded 0.454 ×3 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.089 ×48 |
| 10 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:654 forward 0.169 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:654 forward 0.169 ×1 | site:torch/autograd/graph.py:882 _engine_run_backward 0.078 ×7 |
| 11 | site:nf4_qlora.py:267 forward 0.083 ×5 | site:nf4_qlora.py:267 forward 0.080 ×5 | site:bitsandbytes/backends/cuda/ops.py:916 _dequant_linear_fallback 0.034 ×1 |
| 12 | site:nf4_qlora.py:445 _lora_delta_padded 0.050 ×2 | site:nf4_qlora.py:445 _lora_delta_padded 0.050 ×2 | site:unsloth_zoo/temporary_patches/moe_utils.py:920 _probe_torch_grouped_mm_supported 0.034 ×1 |
| 13 | site:experts4bit_qlora/lora.py:787 forward 0.023 ×8 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:347 forward 0.027 ×12 | site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.015 ×1 |
| 14 | site:experts4bit_qlora/engines/fast.py:685 fused_experts_train_forward 0.018 ×1 | site:experts4bit_qlora/lora.py:787 forward 0.023 ×8 | site:Linear4bit_peft_forward.py:49 lora_forward 0.010 ×12 |
| 15 | site:nf4_grouped.py:1401 gemm_4bit_grouped 0.018 ×1 | site:experts4bit_qlora/engines/fast.py:685 fused_experts_train_forward 0.018 ×1 | site:transformers/integrations/sdpa_attention.py:25 repeat_kv 0.007 ×2 |
| 16 | site:experts4bit_qlora/lora.py:200 _epilogue 0.014 ×2 | site:nf4_grouped.py:1401 gemm_4bit_grouped 0.018 ×1 | site:transformers/integrations/sdpa_attention.py:92 sdpa_attention_forward 0.004 ×3 |
| 17 | site:nf4_qlora.py:599 fused_grouped_lora 0.014 ×1 | site:experts4bit_qlora/lora.py:200 _epilogue 0.014 ×2 | site:Linear4bit_peft_forward.py:52 lora_forward 0.004 ×2 |
| 18 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:430 forward 0.009 ×1 | site:nf4_qlora.py:599 fused_grouped_lora 0.014 ×1 | site:unsloth_compiled_module_qwen3_moe.py:239 apply_rotary_pos_emb 0.004 ×1 |
| 19 | unattributed:allocated before the census window, no frame 0.009 ×1 | site:bitsandbytes/backends/cuda/ops.py:337 _ 0.013 ×1 | site:unsloth_zoo/gradient_checkpointing.py:791 <listcomp> 0.002 ×1 |
| 20 | site:experts4bit_qlora/engines/fast.py:427 backward 0.007 ×11 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:430 forward 0.009 ×1 | site:unsloth_zoo/gradient_checkpointing.py:801 <listcomp> 0.002 ×1 |

## Predictions P41 / P42 / P43 (TC1-PREREG amendment 23: the memory census, e4b against Unsloth at micro-batch 1; scored mechanically from the receipts)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P41 | qwen3memcensus | **HELD** | analytic 1.8119 GB = expert_params 28,991,029,248 / 64 × 4 bytes; e4b fp32 absmax 1.8119 GB vs the analytic value 1.8119 GB (Δ +0.00 %, within 2 %: yes); e4b dq absmax 0.4602 GB vs the analytic value / 3.94 0.4599 GB (Δ +0.06 %, within 2 %: yes); fp32 / dq bytes 3.938 |
| P42 | qwen3memcensus | **HELD** | attributed fraction at the peak: e4b fp32 absmax 0.9996 (>= 0.9; largest unattributed: unattributed:allocated before the census window, no frame 0.009 GB); e4b dq absmax 0.9996 (>= 0.9; largest unattributed: unattributed:allocated before the census window, no frame 0.009 GB); Unsloth 1.0000 (>= 0.9; largest unattributed: none) |
| P43 | qwen3memcensus | **FALSIFIED** | excess_after_absmax = e4b dq peak 24.676 - Unsloth peak 24.244 = +0.431 GB vs [0.5, 2.5] GB (e4b fp32-absmax peak 26.022 GB); the excess's largest group at the peak: transient +0.441 GB; by class at the peak, dq - Unsloth (GB): transient +0.441, other_buffers +0.000, adapter_grads +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000, expert_absmax -0.000; e4b dq's largest non-static groups: site:nf4_qlora.py:444 _lora_delta_padded 0.615 GB ×2, site:nf4_qlora.py:446 _lora_delta_padded 0.454 GB ×3, site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:654 forward 0.169 GB ×1; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.089 GB ×48, site:torch/autograd/graph.py:882 _engine_run_backward 0.078 GB ×7 |
