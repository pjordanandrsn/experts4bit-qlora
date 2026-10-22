# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.56.1 @970d2078caebf7f9806d8df774d8d82a6fb17d31 (GitHub main)
gnf4 0.49.0 @a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
huggingface_hub(e4b/hf) 2.2.0
httpx2(e4b/hf) 2.16.0
brotli(e4b/hf) 1.2.1
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
e4b(t212) 0.56.1 @970d2078caebf7f9806d8df774d8d82a6fb17d31
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
 "run_id": "tc1-5090-155",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.119.02",
 "cpu": "AMD EPYC 9474F 48-Core Processor",
 "nproc": 48,
 "mem_total_kb": "113302944",
 "cgroup_memory_max": "111380791296",
 "disk_root": "overlay         220G  2.9M  220G   1% /",
 "cgroup_cpu_max": "4608000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 75: the 4090-fit read -- e4b's defaults, the compact delta, the chunked LM loss and both, at today's pins, one draw each; reserved and device peaks read first) (`qwen3pos75cl`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_fd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.327 | 574.6 | 26.500 | 949.6 | 2.0463→0.8175 | 1.9468→0.7567 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fd_cd1 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 2.520 | 526.2 | 25.861 | 947.0 | 2.0463→0.8174 | 1.9468→0.7579 | 0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fd_cl1 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 2.324 | 564.9 | 26.157 | 951.0 | 2.0463→0.8172 | 1.9468→0.7578 | 0.0010 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fd_cd1cl1 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 2.502 | 515.3 | 25.291 | 971.8 | 2.0463→0.8201 | 1.9468→0.7549 | -0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_fd` **151.7 s** before step 1 (47% of the arm): load_weights 77.4, c1_before 48.4, c1_after 24.4, eval0 10.4; unattributed 2.075; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_cd1` **114.0 s** before step 1 (38% of the arm): c1_before 56.1, load_weights 40.9, c1_after 24.2, attn4 4.6; unattributed 1.592; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_cl1` **102.7 s** before step 1 (37% of the arm): c1_before 56.5, load_weights 28.4, c1_after 24.2, attn4 4.7; unattributed 1.915; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_cd1cl1` **88.2 s** before step 1 (32% of the arm): c1_before 47.8, load_weights 24.5, c1_after 24.2, attn4 4.4; unattributed 1.564; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_fd` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_fd_cd1` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_fd_cl1` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_fd_cd1cl1` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b only: no position against another framework; each arm's s/step is recorded, never gated (amendment 75)
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b only: no position against another framework; each arm's s/step is recorded, never gated (amendment 75)
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b only: no position against another framework; each arm's s/step is recorded, never gated (amendment 75)
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64 sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_fd_cd1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects
- frozen base `e4b/fused_attn4_m_fd_cl1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects
- frozen base `e4b/fused_attn4_m_fd_cd1cl1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos75cl | e4b/fused_attn4_m_fd | **VALID** | VALID | 0.0000 |  |
| qwen3pos75cl | e4b/fused_attn4_m_fd_cd1 | **VALID** | VALID | 0.0012 |  |
| qwen3pos75cl | e4b/fused_attn4_m_fd_cl1 | **VALID** | VALID | 0.0010 |  |
| qwen3pos75cl | e4b/fused_attn4_m_fd_cd1cl1 | **VALID** | VALID | -0.0018 |  |

## Amendment 75: the 4090-fit read (one draw per arm; GB = 1e9 bytes; reserved and device peaks first; speed recorded, never gated)

| arm | verdict | reserved peak | nvidia-smi peak (vs a 4090's 24,564 MiB) | allocated peak | padded delta at peak | LM-loss backward at peak | other unattributed backward at peak | largest transient at peak | held-out at N (Δ vs fd) | host RSS GB | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fd: today's defaults | VALID | 26.814 | 27.459 (26,187 MiB, -1,623 MiB to a 4090's total) | 26.500 | 2.348 | 0.000 | 0.000 | site:nf4_qlora.py:652 _lora_delta_padded_ladder 1.292 | 0.7567 | 63.832 | 2.327 |
| cd1: + NF4_QLORA_COMPACT_DELTA=1 | VALID | 26.667 | 27.312 (26,047 MiB, -1,483 MiB to a 4090's total) | 25.861 | 0.000 | 2.060 | 0.000 | unattributed:backward, no Python frame (an autograd C++ op or allocator-internal) 1.374 | 0.7579 (+0.0012) | 64.524 | 2.520 |
| cl1: + E4B_CHUNKED_LM_LOSS=1 | VALID | 26.458 | 27.103 (25,847 MiB, -1,283 MiB to a 4090's total) | 26.157 | 2.348 | 0.000 | 0.000 | site:nf4_qlora.py:652 _lora_delta_padded_ladder 1.292 | 0.7578 (+0.0010) | 64.525 | 2.324 |
| cd1cl1: both | VALID | 26.332 | 26.977 (25,727 MiB, -1,163 MiB to a 4090's total) | 25.291 | 1.663 | 0.000 | 0.000 | site:nf4_qlora.py:940 backward 0.811 | 0.7549 (-0.0018) | 64.525 | 2.502 |

| comparison | G | G at the parent's peak / the arm's | live sets | drop: reserved / device / allocated | candidate |
|---|---|---|---|---|---|
| cd1: + NF4_QLORA_COMPACT_DELTA=1 vs fd: today's defaults | delta | 2.348 / 0.000 | RELOCATED: G is gone from the live set at the peak, the whole drop is credited | 0.147 / 0.147 / 0.639 | **CANDIDATE** |
| cl1: + E4B_CHUNKED_LM_LOSS=1 vs fd: today's defaults | loss | 0.000 / 0.000 | G is not live at the parent's peak: nothing to credit | 0.357 / 0.357 / 0.343 | no: the drop is not credited |
| cd1cl1: both vs cd1: + NF4_QLORA_COMPACT_DELTA=1 | loss | 2.060 / 0.000 | RELOCATED: G is gone from the live set at the peak, the whole drop is credited | 0.336 / 0.336 / 0.570 | **CANDIDATE** |
| cd1cl1: both vs fd: today's defaults (reported beside, never a verdict) | both | — | — | 0.482 / 0.482 / 1.209 | — |

- **CANDIDATE:** cd1: + NF4_QLORA_COMPACT_DELTA=1. It goes to the pre-registered RTX 4090 acceptance box at the same recipe: 60 steps, no OOM, free headroom recorded. No default flips before data and the quality and speed gates.
- **CANDIDATE:** cd1cl1: both. It goes to the pre-registered RTX 4090 acceptance box at the same recipe: 60 steps, no OOM, free headroom recorded. No default flips before data and the quality and speed gates.
- CUDA context and non-PyTorch device memory sit in the nvidia-smi peak only; the census reads PyTorch's allocator. The device peak is read from the 1 Hz sampler.

## Amendment 75: the census tables (Qwen3-30B-A3B, TC1's field recipe at today's pins, one draw per arm; GB = 1e9 bytes)

| | fd: today's defaults | cd1: + NF4_QLORA_COMPACT_DELTA=1 | cl1: + E4B_CHUNKED_LM_LOSS=1 | cd1cl1: both |
|---|---|---|---|---|
| peak allocated / reserved (GB) | 26.500 / 26.814 | 25.861 / 26.667 | 26.157 / 26.458 | 25.291 / 26.332 |
| attributed fraction at the peak | 0.9987 | 0.9456 | 1.0000 | 1.0000 |
| peak at (checkpoint / phase) | s22.mb2 / s22.mb2.backward | s22.mb2 / s22.mb2.backward | s22.mb2 / s22.mb2.backward | s22.mb2 / s22.mb2.backward |
| window holds the run's peak / events / ring full | True / 1000000 / True | True / 1000000 / True | True / 1000000 / True | True / 1000000 / True |
| snapshots / census s / torch | 8 / 21.85 / 2.12.1+cu130 | 8 / 28.31 / 2.12.1+cu130 | 8 / 29.62 / 2.12.1+cu130 | 9 / 35.51 / 2.12.1+cu130 |

**Static census by class (GB: end of setup / end of training)**

| | fd: today's defaults | cd1: + NF4_QLORA_COMPACT_DELTA=1 | cl1: + E4B_CHUNKED_LM_LOSS=1 | cd1cl1: both |
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
| other | -0.218 / -0.136 | -0.218 / -0.136 | -0.218 / -0.183 | -0.218 / -0.183 |
| allocated_bytes | 19.321 / 20.708 | 19.321 / 20.708 | 19.321 / 20.661 | 19.321 / 20.661 |

**Live bytes at the peak by class (GB; transient = every live block that holds no static tensor)**

| | fd: today's defaults | cd1: + NF4_QLORA_COMPACT_DELTA=1 | cl1: + E4B_CHUNKED_LM_LOSS=1 | cd1cl1: both |
|---|---|---|---|---|
| frozen_expert_weights | 14.496 | 14.496 | 14.496 | 14.496 |
| expert_absmax | 0.460 | 0.460 | 0.460 | 0.460 |
| other_frozen | 1.509 | 1.509 | 1.509 | 1.509 |
| trainable_adapters | 2.570 | 2.570 | 2.570 | 2.570 |
| adapter_grads | 2.570 | 2.570 | 2.570 | 2.570 |
| optimizer_state | 1.305 | 1.305 | 1.305 | 1.305 |
| other_buffers | 0.252 | 0.252 | 0.252 | 0.252 |
| transient | 3.338 | 2.699 | 2.994 | 2.129 |

**Top live-at-peak groups (GB ×count; 20 of the receipts' 40 shown)**

| rank | fd: today's defaults | cd1: + NF4_QLORA_COMPACT_DELTA=1 | cl1: + E4B_CHUNKED_LM_LOSS=1 | cd1cl1: both |
|---|---|---|---|---|
| 1 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 | static:frozen_expert_weights 14.496 ×96 |
| 2 | static:adapter_grads 2.570 ×480 | static:adapter_grads 2.570 ×480 | static:adapter_grads 2.570 ×480 | static:adapter_grads 2.570 ×480 |
| 3 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 | static:trainable_adapters 2.570 ×576 |
| 4 | static:optimizer_state 1.305 ×2306 | unattributed:backward, no Python frame (an autograd C++ op or allocator-internal) 1.374 ×2 | static:optimizer_state 1.305 ×2306 | static:optimizer_state 1.305 ×2306 |
| 5 | site:nf4_qlora.py:652 _lora_delta_padded_ladder 1.292 ×2 | static:optimizer_state 1.305 ×2306 | site:nf4_qlora.py:652 _lora_delta_padded_ladder 1.292 ×2 | static:other_frozen.bf16 1.270 ×243 |
| 6 | static:other_frozen.bf16 1.270 ×243 | static:other_frozen.bf16 1.270 ×243 | static:other_frozen.bf16 1.270 ×243 | site:nf4_qlora.py:940 backward 0.811 ×1 |
| 7 | site:nf4_qlora.py:654 _lora_delta_padded_ladder 0.954 ×3 | site:torch/nn/functional.py:3507 cross_entropy 0.687 ×2 | site:nf4_qlora.py:654 _lora_delta_padded_ladder 0.954 ×3 | site:nf4_qlora.py:942 backward 0.811 ×1 |
| 8 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×384 | static:expert_absmax 0.460 ×384 |
| 9 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 | site:torch/nn/modules/linear.py:134 forward 0.343 ×1 | static:other_buffers 0.252 ×98 | static:other_buffers 0.252 ×98 |
| 10 | static:other_buffers 0.252 ×98 | static:other_buffers 0.252 ×98 | static:other_frozen.nf4_linear4bit 0.239 ×480 | static:other_frozen.nf4_linear4bit 0.239 ×480 |
| 11 | static:other_frozen.nf4_linear4bit 0.239 ×480 | static:other_frozen.nf4_linear4bit 0.239 ×480 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.213 ×46 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.185 ×40 |
| 12 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.213 ×46 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.222 ×48 | site:nf4_qlora.py:653 _lora_delta_padded_ladder 0.102 ×2 | site:torch/autograd/graph.py:882 _engine_run_backward 0.088 ×3 |
| 13 | site:nf4_qlora.py:653 _lora_delta_padded_ladder 0.102 ×2 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | site:nf4_qlora.py:418 forward 0.074 ×1 | site:experts4bit_qlora/engines/fast.py:752 fused_experts_train_forward 0.037 ×1 |
| 14 | site:nf4_qlora.py:418 forward 0.074 ×1 | unattributed:allocated before the census window, no frame 0.034 ×1 | site:torch/nn/functional.py:5462 pad 0.052 ×4 | site:torch/nn/modules/linear.py:134 forward 0.034 ×1 |
| 15 | site:torch/nn/functional.py:5462 pad 0.052 ×4 | site:torch/nn/functional.py:2567 embedding 0.005 ×1 | site:experts4bit_qlora/engines/fast.py:752 fused_experts_train_forward 0.037 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 |
| 16 | site:experts4bit_qlora/engines/fast.py:752 fused_experts_train_forward 0.037 ×1 | site:transformers/masking_utils.py:54 and_mask 0.001 ×1 | site:nf4_grouped.py:1944 gemm_4bit_grouped 0.037 ×1 | site:experts4bit_qlora/lora.py:787 forward 0.019 ×2 |
| 17 | site:nf4_grouped.py:1944 gemm_4bit_grouped 0.037 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 forward 0.000 ×2 | site:torch/nn/modules/linear.py:134 forward 0.034 ×1 | site:transformers/integrations/sdpa_attention.py:25 repeat_kv 0.019 ×2 |
| 18 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | site:nf4_qlora.py:591 lora_delta_grouped 0.000 ×1 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 forward 0.034 ×1 | site:nf4_qlora.py:944 backward 0.016 ×1 |
| 19 | unattributed:allocated before the census window, no frame 0.034 ×1 | site:tc1_arm.py:1527 collate 0.000 ×3 | site:nf4_qlora.py:1042 fused_grouped_lora 0.028 ×1 | site:nf4_qlora.py:938 backward 0.012 ×1 |
| 20 | site:nf4_qlora.py:1042 fused_grouped_lora 0.028 ×1 | site:nf4_grouped.py:543 stage 0.000 ×8 | site:experts4bit_qlora/lora.py:787 forward 0.019 ×2 | site:transformers/integrations/sdpa_attention.py:92 sdpa_attention_forward 0.011 ×3 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos75cl/e4b/fused_attn4_m_fd attempt 0 load1_median 1.38 gate 6.0 status ok over 0`
- `qwen3pos75cl/e4b/fused_attn4_m_fd_cd1 attempt 0 load1_median 1.23 gate 6.0 status ok over 0`
- `qwen3pos75cl/e4b/fused_attn4_m_fd_cl1 attempt 0 load1_median 1.21 gate 6.0 status ok over 0`
- `qwen3pos75cl/e4b/fused_attn4_m_fd_cd1cl1 attempt 0 load1_median 1.33 gate 6.0 status ok over 0`
