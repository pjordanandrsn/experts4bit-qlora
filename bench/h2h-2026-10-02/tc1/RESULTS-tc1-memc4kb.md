# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1-5090-112)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @dde06a9263dc1c9057c141233cdea5a51bb61d8e (GitHub main)
gnf4 0.42.0 @71847e5fc1ab17af083bcf5d9992793f1ffd573c (GitHub main)
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
e4b(t212) 0.48.0 @dde06a9263dc1c9057c141233cdea5a51bb61d8e
gnf4(t212) 0.42.0 @71847e5fc1ab17af083bcf5d9992793f1ffd573c
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
 "run_id": "tc1-5090-112",
 "instance_id": "54549887",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994548",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   48M  320G   1% /",
 "hostname": "e13904e43afa",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 96,
 "cgroup_cpuset_effective": "0-95",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 55: the memory census on packed 4,096-token rows at the current defaults -- e4b defaults, e4b absmax-dq, Unsloth; no speed read) (`qwen3memc4kb`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=20; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_p4d | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 10.409 | 1379.0 | 28.228 | 4945.4 | 1.2644→0.9405 | 1.2893→0.9854 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_p4d_dq | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0008) | 20 | 10.351 | 1450.4 | 26.877 | 4813.8 | 1.2578→0.9409 | 1.2885→0.9855 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_p4d | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0010) | 20 | 16.296 | 939.4 | 24.864 | 4240.9 | 1.2601→0.9408 | 1.2903→0.9858 | 0.0004 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_p4d` **126.9 s** before step 1 (34% of the arm): c1_before 71.0, c1_after 36.0, load_weights 21.0, eval0 13.7; unattributed 2.601; budget 1890.0
- prologue `e4b/fused_attn4_m_p4d_dq` **117.9 s** before step 1 (33% of the arm): c1_before 69.5, c1_after 34.9, load_weights 21.0, eval0 6.7; unattributed 2.633; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_p4d` **146.5 s** before step 1 (29% of the arm): c1_before 67.1, c1_after 34.3, load_weights 33.3, eval0 24.5; unattributed 12.883; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_p4d` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_p4d_dq` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_p4d` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: the memory census slows the step (amendment 55, as amendment 47), so no speed is read on this token
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: the memory census slows the step (amendment 55, as amendment 47), so no speed is read on this token
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: the memory census slows the step (amendment 55, as amendment 47), so no speed is read on this token
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_p4d_dq` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0001, step-0 0.0008 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0034, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_p4d` **COMPARABLE** (median step |Δ| 0.0005, |Δ held-out at N| 0.0004, step-0 0.0010 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0045, paired rows mean +0.0004 ± 0.0003 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_p4d` same; `e4b/fused_attn4_m_p4d_dq` same; `unsloth/ckpt_unsloth_m_p4d` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_p4d_dq`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_p4d`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3memc4kb | e4b/fused_attn4_m_p4d | **VALID** | VALID | 0.0000 |  |
| qwen3memc4kb | e4b/fused_attn4_m_p4d_dq | **VALID** | VALID | 0.0001 |  |
| qwen3memc4kb | unsloth/ckpt_unsloth_m_p4d | **VALID** | VALID | 0.0004 |  |

## Predictions P141 / P142 / P143 (TC1-PREREG amendment 55: amendment 47's census at the current defaults, e4b against Unsloth; scored mechanically from the receipts)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P141 | qwen3memc4kb | **HELD** | attributed fraction at the peak: e4b defaults 1.0000 (>= 0.9); e4b absmax-dq 1.0000 (>= 0.9); Unsloth 1.0000 (>= 0.9) |
| P142 | qwen3memc4kb | **HELD** | e4b defaults peak 28.228 - Unsloth peak 24.864 = +3.364 GB vs [2.0, 5.0] GB; the excess's largest class at the peak: transient +4.583 GB; by class, e4b - Unsloth (GB): transient +4.583, expert_absmax +1.352, other_buffers +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000, adapter_grads -2.570; e4b's largest non-static groups: site:torch/nn/functional.py:3507 cross_entropy 2.489 GB ×3, site:transformers/loss/loss_utils.py:55 ForCausalLMLoss 2.489 GB ×1, site:torch/nn/modules/linear.py:134 forward 1.278 GB ×2; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:torch/autograd/graph.py:882 _engine_run_backward 0.287 GB ×7, site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.134 GB ×1 |
| P143 | qwen3memc4kb | **HELD** | e4b absmax-dq peak 26.877 - Unsloth peak 24.864 = +2.013 GB vs <= 2.5 GB; the excess's largest class at the peak: transient +4.583 GB; by class, e4b - Unsloth (GB): transient +4.583, other_buffers +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000, expert_absmax -0.000, adapter_grads -2.570; e4b's largest non-static groups: site:torch/nn/functional.py:3507 cross_entropy 2.489 GB ×3, site:transformers/loss/loss_utils.py:55 ForCausalLMLoss 2.489 GB ×1, site:torch/nn/modules/linear.py:134 forward 1.278 GB ×2; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:torch/autograd/graph.py:882 _engine_run_backward 0.287 GB ×7, site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.134 GB ×1 |
