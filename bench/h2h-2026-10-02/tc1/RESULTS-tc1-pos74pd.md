# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.56.1 @cedddf8452c06c8d24f52d7954b55a8b55cd1fa6 (GitHub main)
gnf4 0.49.0 @a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7 (GitHub main)
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
e4b(t212) 0.56.1 @cedddf8452c06c8d24f52d7954b55a8b55cd1fa6
gnf4(t212) 0.49.0 @a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7
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
 "run_id": "tc1-5090-154",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7402P 24-Core Processor",
 "nproc": 48,
 "mem_total_kb": "263782596",
 "cgroup_memory_max": "183358193664",
 "disk_root": "overlay         220G   30M  220G   1% /",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 74: amendment 73's census at today's pins -- e4b's defaults against one grouped-nf4-gemm switch per arm, one draw each; speed recorded, never gated) (`qwen3pos74pd`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_fd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.673 | 475.0 | 26.500 | 996.5 | 2.0463→0.8183 | 1.9468→0.7557 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fd_sl0 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 3.438 | 408.9 | 26.149 | 925.4 | 2.0463→0.8179 | 1.9468→0.7581 | 0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fd_cd1 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 3.434 | 407.8 | 25.861 | 962.5 | 2.0463→0.8178 | 1.9468→0.7571 | 0.0014 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fd_pb1 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 3.870 | 369.7 | 25.861 | 849.6 | 2.0463→0.8186 | 1.9468→0.7560 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_fd` **115.3 s** before step 1 (36% of the arm): c1_before 61.1, c1_after 30.8, load_weights 20.6, eval0 15.2; unattributed 2.184; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_sl0` **102.3 s** before step 1 (30% of the arm): c1_before 61.2, c1_after 31.0, load_weights 20.5, attn4 6.0; unattributed 2.195; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_cd1` **102.4 s** before step 1 (30% of the arm): c1_before 61.2, c1_after 31.1, load_weights 20.5, attn4 6.0; unattributed 2.183; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_pb1` **102.6 s** before step 1 (28% of the arm): c1_before 61.2, c1_after 31.2, load_weights 20.5, attn4 6.0; unattributed 2.229; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_fd` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_fd_sl0` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_fd_cd1` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_fd_pb1` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b only: no position against another framework; each arm's s/step is recorded, never gated (amendment 74)
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b only: no position against another framework; each arm's s/step is recorded, never gated (amendment 74)
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b only: no position against another framework; each arm's s/step is recorded, never gated (amendment 74)
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64 sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_fd_sl0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects
- frozen base `e4b/fused_attn4_m_fd_cd1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects
- frozen base `e4b/fused_attn4_m_fd_pb1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos74pd | e4b/fused_attn4_m_fd | **VALID** | VALID | 0.0000 |  |
| qwen3pos74pd | e4b/fused_attn4_m_fd_sl0 | **VALID** | VALID | 0.0024 |  |
| qwen3pos74pd | e4b/fused_attn4_m_fd_cd1 | **VALID** | VALID | 0.0014 |  |
| qwen3pos74pd | e4b/fused_attn4_m_fd_pb1 | **VALID** | VALID | 0.0003 |  |

## Amendment 74: the padded LoRA delta's switches at today's pins (one draw per arm; GB = 1e9 bytes; speed recorded, never gated)

| arm | verdict | peak allocated (Δ vs fd) | padded-delta live at the peak | removed vs fd | held-out at N (Δ vs fd) | end of setup alloc / reserved | s/step |
|---|---|---|---|---|---|---|---|
| fd: today's defaults | VALID | 26.500 | 2.348 | — | 0.7557 | 19.321 / 19.986 | 2.673 |
| sl0: + NF4_QLORA_SINGLE_LADDER=0 | VALID | 26.149 (-0.351) | 2.007 | 14.5% | 0.7581 (+0.0024) | 19.321 / 19.986 | 3.438 |
| cd1: + NF4_QLORA_COMPACT_DELTA=1 | VALID | 25.861 (-0.639) | 0.000 | 100.0% | 0.7571 (+0.0014) | 19.321 / 19.986 | 3.434 |
| pb1: + NF4_QLORA_PAD_BUCKETS=1 | VALID | 25.861 (-0.639) | 0.000 | 100.0% | 0.7560 (+0.0003) | 19.321 / 19.986 | 3.870 |

Amendment 73 measured 2.044 GB of padded-delta live bytes at the claim's commits; this box's fd re-measures them at today's pins, and the rule reads against this box's fd. s/step is recorded, never gated: TC1's field draws have been unstable on rented hosts. CUDA context and non-PyTorch device memory are not in the census and are not read.
- **No candidate:** no arm removes 50% of fd's padded-delta bytes, lowers its peak by 50% of them and holds held-out within 0.005. The fix is new code in grouped-nf4-gemm (a group-chunked single block), with this box as its baseline.
- **Moved, not saved:** cd1: + NF4_QLORA_COMPACT_DELTA=1 removes 2.348 GB at the padded-delta sites but its peak moves only -0.639 GB, under 50% of what it removed: never a candidate.
- **Moved, not saved:** pb1: + NF4_QLORA_PAD_BUCKETS=1 removes 2.348 GB at the padded-delta sites but its peak moves only -0.639 GB, under 50% of what it removed: never a candidate.

## Amendment 74: the census tables (Qwen3-30B-A3B, TC1's field recipe at today's pins, one draw per arm; GB = 1e9 bytes)

| | fd: today's defaults | sl0: + NF4_QLORA_SINGLE_LADDER=0 | cd1: + NF4_QLORA_COMPACT_DELTA=1 | pb1: + NF4_QLORA_PAD_BUCKETS=1 |
|---|---|---|---|---|
| peak allocated / reserved (GB) | 26.500 / 26.814 | 26.149 / 26.730 | 25.861 / 26.835 | 25.861 / 26.751 |
| attributed fraction at the peak | 0.9987 | 0.9987 | 0.9456 | 0.9456 |
| peak at (checkpoint / phase) | s22.mb2 / s22.mb2.backward | s22.mb2 / s22.mb2.backward | s22.mb2 / s22.mb2.backward | s22.mb2 / s22.mb2.backward |
| window holds the run's peak / events / ring full | True / 1000000 / True | True / 1000000 / True | True / 1000000 / True | True / 1000000 / True |
| snapshots / census s / torch | 8 / 27.92 / 2.12.1+cu130 | 8 / 27.38 / 2.12.1+cu130 | 8 / 27.72 / 2.12.1+cu130 | 8 / 28.08 / 2.12.1+cu130 |

**Static census by class (GB: end of setup / end of training)**

| | fd: today's defaults | sl0: + NF4_QLORA_SINGLE_LADDER=0 | cd1: + NF4_QLORA_COMPACT_DELTA=1 | pb1: + NF4_QLORA_PAD_BUCKETS=1 |
|---|---|---|---|---|
| frozen_expert_weights | 14.496 / 14.496 | 14.496 / 14.496 | 14.496 / 14.496 | 14.496 / 14.496 |
| expert_absmax | 0.460 / 0.460 | 0.460 / 0.460 | 0.460 / 0.460 | 0.460 / 0.460 |
| expert_absmax.nested_code | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| expert_absmax.nested_off | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| expert_absmax.nested_q | 0.453 / 0.453 | 0.453 / 0.453 | 0.453 / 0.453 | 0.453 / 0.453 |
| expert_absmax.nested_s | 0.007 / 0.007 | 0.007 / 0.007 | 0.007 / 0.007 | 0.007 / 0.007 |
| other_frozen | 1.761 / 1.761 | 1.761 / 1.761 | 1.761 / 1.761 | 1.761 / 1.761 |
| other_frozen.bf16 | 1.270 / 1.270 | 1.270 / 1.270 | 1.270 / 1.270 | 1.270 / 1.270 |
| other_frozen.nf4_linear4bit | 0.491 / 0.491 | 0.491 / 0.491 | 0.491 / 0.491 | 0.491 / 0.491 |
| trainable_adapters | 2.570 / 2.570 | 2.570 / 2.570 | 2.570 / 2.570 | 2.570 / 2.570 |
| adapter_grads | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| optimizer_state | 0.000 / 1.305 | 0.000 / 1.305 | 0.000 / 1.305 | 0.000 / 1.305 |
| other_buffers | 0.252 / 0.252 | 0.252 / 0.252 | 0.252 / 0.252 | 0.252 / 0.252 |
| other | -0.218 / -0.136 | -0.218 / -0.136 | -0.218 / -0.136 | -0.218 / -0.136 |
| allocated_bytes | 19.321 / 20.708 | 19.321 / 20.708 | 19.321 / 20.708 | 19.321 / 20.708 |

**Live bytes at the peak by class (GB; transient = every live block that holds no static tensor)**

| | fd: today's defaults | sl0: + NF4_QLORA_SINGLE_LADDER=0 | cd1: + NF4_QLORA_COMPACT_DELTA=1 | pb1: + NF4_QLORA_PAD_BUCKETS=1 |
|---|---|---|---|---|
| frozen_expert_weights | 14.496 | 14.496 | 14.496 | 14.496 |
| expert_absmax | 0.460 | 0.460 | 0.460 | 0.460 |
| other_frozen | 1.509 | 1.509 | 1.509 | 1.509 |
| trainable_adapters | 2.570 | 2.570 | 2.570 | 2.570 |
| adapter_grads | 2.570 | 2.570 | 2.570 | 2.570 |
| optimizer_state | 1.305 | 1.305 | 1.305 | 1.305 |
| other_buffers | 0.252 | 0.252 | 0.252 | 0.252 |
| transient | 3.338 | 2.986 | 2.699 | 2.699 |

**Top live-at-peak groups (GB ×count; 20 of the receipts' 40 shown)**

| rank | fd: today's defaults | sl0: + NF4_QLORA_SINGLE_LADDER=0 | cd1: + NF4_QLORA_COMPACT_DELTA=1 | pb1: + NF4_QLORA_PAD_BUCKETS=1 |
|---|---|---|---|---|
| 1 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 |
| 2 | static:adapter_grads 2.570 ×480 | static:adapter_grads 2.570 ×480 | static:adapter_grads 2.570 ×480 | static:adapter_grads 2.570 ×480 |
| 3 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 |
| 4 | static:optimizer_state 1.305 ×2306 | static:optimizer_state 1.305 ×2306 | unattributed:backward, no Python frame (an autograd C++ op or allocator-internal) 1.374 ×2 | unattributed:backward, no Python frame (an autograd C++ op or allocator-internal) 1.374 ×2 |
| 5 | site:nf4_qlora.py:652 _lora_delta_padded_ladder 1.292 ×2 | static:other_frozen.bf16 1.270 ×243 | static:optimizer_state 1.305 ×2306 | static:optimizer_state 1.305 ×2306 |
| 6 | static:other_frozen.bf16 1.270 ×243 | site:nf4_qlora.py:624 _lora_delta_padded 1.096 ×2 | static:other_frozen.bf16 1.270 ×243 | static:other_frozen.bf16 1.270 ×243 |
| 7 | site:nf4_qlora.py:654 _lora_delta_padded_ladder 0.954 ×3 | site:nf4_qlora.py:626 _lora_delta_padded 0.809 ×3 | site:torch/nn/functional.py:3507 cross_entropy 0.687 ×2 | site:torch/nn/functional.py:3507 cross_entropy 0.687 ×2 |
| 8 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×384 |
| 9 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 |
| 10 | static:other_buffers 0.252 ×98 | static:other_buffers 0.252 ×98 | static:other_buffers 0.252 ×98 | static:other_buffers 0.252 ×98 |
| 11 | static:other_frozen.nf4_linear4bit 0.239 ×480 | static:other_frozen.nf4_linear4bit 0.239 ×480 | static:other_frozen.nf4_linear4bit 0.239 ×480 | static:other_frozen.nf4_linear4bit 0.239 ×480 |
| 12 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.213 ×46 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.208 ×45 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.222 ×48 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.222 ×48 |
| 13 | site:nf4_qlora.py:653 _lora_delta_padded_ladder 0.102 ×2 | site:nf4_qlora.py:418 forward 0.120 ×5 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 |
| 14 | site:nf4_qlora.py:418 forward 0.074 ×1 | site:nf4_qlora.py:625 _lora_delta_padded 0.102 ×2 | unattributed:allocated before the census window, no frame 0.034 ×1 | unattributed:allocated before the census window, no frame 0.034 ×1 |
| 15 | site:torch/nn/functional.py:5462 pad 0.052 ×4 | site:experts4bit_qlora/engines/fast.py:752 fused_experts_train_forward 0.037 ×1 | site:torch/nn/functional.py:2567 embedding 0.005 ×1 | site:torch/nn/functional.py:2567 embedding 0.005 ×1 |
| 16 | site:experts4bit_qlora/engines/fast.py:752 fused_experts_train_forward 0.037 ×1 | site:nf4_grouped.py:1944 gemm_4bit_grouped 0.037 ×1 | site:transformers/masking_utils.py:54 and_mask 0.001 ×1 | site:transformers/masking_utils.py:54 and_mask 0.001 ×1 |
| 17 | site:nf4_grouped.py:1944 gemm_4bit_grouped 0.037 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 forward 0.000 ×2 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 forward 0.000 ×2 |
| 18 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | unattributed:allocated before the census window, no frame 0.034 ×1 | site:nf4_qlora.py:591 lora_delta_grouped 0.000 ×1 | site:nf4_qlora.py:697 _lora_delta_grouped_bucketed 0.000 ×1 |
| 19 | unattributed:allocated before the census window, no frame 0.034 ×1 | site:nf4_qlora.py:1042 fused_grouped_lora 0.028 ×1 | site:tc1_arm.py:1527 collate 0.000 ×3 | site:tc1_arm.py:1527 collate 0.000 ×3 |
| 20 | site:nf4_qlora.py:1042 fused_grouped_lora 0.028 ×1 | site:experts4bit_qlora/lora.py:787 forward 0.019 ×2 | site:nf4_grouped.py:543 stage 0.000 ×8 | site:nf4_grouped.py:543 stage 0.000 ×8 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos74pd/e4b/fused_attn4_m_fd attempt 0 load1_median 2.87 gate 6.0 status ok over 0`
- `qwen3pos74pd/e4b/fused_attn4_m_fd_sl0 attempt 0 load1_median 3.11 gate 6.0 status ok over 0`
- `qwen3pos74pd/e4b/fused_attn4_m_fd_cd1 attempt 0 load1_median 3.13 gate 6.0 status ok over 0`
- `qwen3pos74pd/e4b/fused_attn4_m_fd_pb1 attempt 0 load1_median 2.91 gate 6.0 status ok over 0`
