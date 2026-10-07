# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @e4c88425123af2125af92909423e4bb00b382148 (GitHub main)
gnf4 0.42.0 @5a60c37dbd0756040052b603c9b0ee680f06d444 (GitHub main)
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
e4b(t212) 0.48.0 @e4c88425123af2125af92909423e4bb00b382148
gnf4(t212) 0.42.0 @5a60c37dbd0756040052b603c9b0ee680f06d444
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
 "run_id": "tc1-5090-130",
 "instance_id": "54697226",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "129453600",
 "cgroup_memory_max": "127257280512",
 "disk_root": "overlay         320G   31M  320G   1% /",
 "hostname": "d7cf8684e75c",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 65: the training-phase census on packed rows at the new defaults -- e4b defaults, e4b offload, Unsloth; no speed read) (`qwen3memc4kr`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=20; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_p4r | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 9.318 | 1575.7 | 26.581 | 4883.9 | 1.2578→0.9401 | 1.2885→0.9857 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_p4r_off | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 9.472 | 1610.2 | 25.826 | 4507.9 | 1.2578→0.9404 | 1.2885→0.9860 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_p4r | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0018) | 20 | 11.144 | 1398.6 | 24.864 | 3942.2 | 1.2601→0.9406 | 1.2903→0.9857 | 0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_p4r` **60.0 s** before step 1 (22% of the arm): c1_before 29.3, c1_after 14.9, load_weights 10.2, eval0 9.3; unattributed 1.315; budget 1890.0
- prologue `e4b/fused_attn4_m_p4r_off` **56.4 s** before step 1 (22% of the arm): c1_before 29.2, c1_after 14.9, load_weights 10.2, eval0 5.9; unattributed 1.309; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_p4r` **70.2 s** before step 1 (23% of the arm): c1_before 28.7, load_weights 15.6, c1_after 14.5, eval0 13.8; unattributed 6.366; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_p4r` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_p4r_off` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_p4r` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: the memory census slows the step (amendment 65, as amendment 47), so no speed is read on this token
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: the memory census slows the step (amendment 65, as amendment 47), so no speed is read on this token
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: the memory census slows the step (amendment 65, as amendment 47), so no speed is read on this token
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_p4r_off` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0026, paired rows mean +0.0003 ± 0.0004 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_p4r` **COMPARABLE** (median step |Δ| 0.0006, |Δ held-out at N| 0.0001, step-0 0.0018 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0049, paired rows mean +0.0001 ± 0.0003 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_p4r` same; `e4b/fused_attn4_m_p4r_off` same; `unsloth/ckpt_unsloth_m_p4r` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_p4r_off`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_p4r`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3memc4kr | e4b/fused_attn4_m_p4r | **VALID** | VALID | 0.0000 |  |
| qwen3memc4kr | e4b/fused_attn4_m_p4r_off | **VALID** | VALID | 0.0003 |  |
| qwen3memc4kr | unsloth/ckpt_unsloth_m_p4r | **VALID** | VALID | 0.0001 |  |

## Predictions P184-P188 (TC1-PREREG amendment 65: the training-phase census at the new defaults, e4b against Unsloth; scored mechanically from the receipts)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P184 | qwen3memc4kr | **HELD** | attributed fraction at the peak: e4b defaults 1.0000 (>= 0.9); e4b offload 1.0000 (>= 0.9); Unsloth 1.0000 (>= 0.9) |
| P185 | qwen3memc4kr | **HELD** | e4b defaults peak 26.581 - Unsloth peak 24.864 = +1.717 GB vs [1.3, 2.1] GB; the excess's largest class at the peak: transient +1.685 GB; by class, e4b - Unsloth (GB): transient +1.685, other_buffers +0.000, adapter_grads +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000, expert_absmax -0.000; e4b's largest non-static groups: site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.772 GB ×46, site:nf4_qlora.py:672 _lora_delta_bucketed 0.534 GB ×2, site:nf4_qlora.py:679 <listcomp> 0.395 GB ×24; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:torch/autograd/graph.py:882 _engine_run_backward 0.287 GB ×7, site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.134 GB ×1 |
| P186 | qwen3memc4kr | **HELD** | e4b offload peak 25.826 - Unsloth peak 24.864 = +0.962 GB vs <= 1.2 GB; the excess's largest class at the peak: transient +0.949 GB; by class, e4b - Unsloth (GB): transient +0.949, other_buffers +0.000, adapter_grads +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000, expert_absmax -0.000; e4b's largest non-static groups: site:nf4_qlora.py:672 _lora_delta_bucketed 0.541 GB ×2, site:nf4_qlora.py:679 <listcomp> 0.399 GB ×27, site:nf4_qlora.py:679 _lora_delta_bucketed 0.393 GB ×1; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:torch/autograd/graph.py:882 _engine_run_backward 0.287 GB ×7, site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.134 GB ×1 |
| P187 | qwen3memc4kr | **HELD** | census peak phase per arm: e4b defaults s12.mb2.backward; e4b offload s2.mb3.backward; Unsloth s2.mb2.backward |
| P188 | qwen3memc4kr | **HELD** | e4b offload's largest non-static group at the peak: site:nf4_qlora.py:672 _lora_delta_bucketed 0.541 GB x2 (registered: a site in nf4_qlora.py); next: site:nf4_qlora.py:672 _lora_delta_bucketed 0.541 GB ×2, site:nf4_qlora.py:679 <listcomp> 0.399 GB ×27, site:nf4_qlora.py:679 _lora_delta_bucketed 0.393 GB ×1, site:nf4_qlora.py:673 _lora_delta_bucketed 0.369 GB ×2 |
