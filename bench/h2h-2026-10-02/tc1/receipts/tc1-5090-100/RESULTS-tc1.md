# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @dfc5bdafd19ea659365eafe0663686ca238eac55 (GitHub main)
gnf4 0.41.0 @a93b80c5112e6f3639f9287d2f290ce1619d26f4 (GitHub main)
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
e4b(t212) 0.48.0 @dfc5bdafd19ea659365eafe0663686ca238eac55
gnf4(t212) 0.41.0 @a93b80c5112e6f3639f9287d2f290ce1619d26f4
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
 "run_id": "tc1-5090-100",
 "instance_id": "54415375",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "a17f9b6d231c",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 256,
 "cgroup_cpuset_effective": "0-255",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 47: the memory census on packed 4,096-token rows -- e4b defaults, e4b absmax-dq + compact delta, Unsloth; no speed read) (`qwen3memc4k`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=20; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_p4 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 11.378 | 1295.9 | 32.337 | 5934.8 | 1.2644→0.9406 | 1.2893→0.9858 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_p4_lev | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0008) | 20 | 11.734 | 1348.7 | 29.544 | 5821.1 | 1.2578→0.9405 | 1.2885→0.9858 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_p4 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0010) | 20 | 14.194 | 1091.1 | 24.864 | 4661.4 | 1.2601→0.9409 | 1.2903→0.9855 | -0.0003 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_p4` **108.6 s** before step 1 (30% of the arm): c1_before 52.9, c1_after 26.3, load_weights 25.2, eval0 12.6; unattributed 2.136; budget 1890.0
- prologue `e4b/fused_attn4_m_p4_lev` **101.8 s** before step 1 (29% of the arm): c1_before 54.1, c1_after 26.4, load_weights 22.7, eval0 7.2; unattributed 2.171; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_p4` **121.6 s** before step 1 (28% of the arm): c1_before 50.7, load_weights 32.1, c1_after 25.6, eval0 20.7; unattributed 9.769; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_p4` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_p4_lev` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_p4` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: the memory census slows the step (amendment 47), so no speed is read on this token; positions stay with the boxes that read them
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: the memory census slows the step (amendment 47), so no speed is read on this token; positions stay with the boxes that read them
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: the memory census slows the step (amendment 47), so no speed is read on this token; positions stay with the boxes that read them
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_p4_lev` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0001, step-0 0.0008 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0006, paired rows mean -0.0001 ± 0.0003 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_p4` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0010 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0035, paired rows mean -0.0003 ± 0.0003 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_p4` same; `e4b/fused_attn4_m_p4_lev` same; `unsloth/ckpt_unsloth_m_p4` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_p4_lev`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_p4`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3memc4k | e4b/fused_attn4_m_p4 | **VALID** | VALID | 0.0000 |  |
| qwen3memc4k | e4b/fused_attn4_m_p4_lev | **VALID** | VALID | -0.0001 |  |
| qwen3memc4k | unsloth/ckpt_unsloth_m_p4 | **VALID** | VALID | -0.0003 |  |

## Predictions P112 / P113 / P114 (TC1-PREREG amendment 47: the memory census on packed 4,096-token rows, e4b against Unsloth; scored mechanically from the receipts)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P112 | qwen3memc4k | **HELD** | attributed fraction at the peak: e4b defaults 0.9995 (>= 0.9); e4b absmax-dq + compact delta 0.9994 (>= 0.9); Unsloth 1.0000 (>= 0.9) |
| P113 | qwen3memc4k | **HELD** | e4b defaults peak 32.337 - Unsloth peak 24.864 = +7.473 GB vs [5.0, 10.0] GB; the excess's largest class at the peak: transient +6.095 GB; by class, e4b - Unsloth (GB): transient +6.095, expert_absmax +1.352, other_buffers +0.000, adapter_grads +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000; e4b's largest non-static groups: site:nf4_qlora.py:451 _lora_delta_padded 3.349 GB ×2, site:nf4_qlora.py:453 _lora_delta_padded 2.474 GB ×3, site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.419 GB ×25; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:torch/autograd/graph.py:882 _engine_run_backward 0.287 GB ×7, site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.134 GB ×1 |
| P114 | qwen3memc4k | **FALSIFIED** | e4b absmax-dq + compact delta peak 29.544 - Unsloth peak 24.864 = +4.680 GB vs <= 3.0 GB; the excess's largest class at the peak: transient +4.676 GB; by class, e4b - Unsloth (GB): transient +4.676, other_buffers +0.000, adapter_grads +0.000, frozen_expert_weights +0.000, optimizer_state +0.000, other_frozen +0.000, trainable_adapters +0.000, expert_absmax -0.000; e4b's largest non-static groups: site:nf4_qlora.py:493 forward 3.108 GB ×1, site:nf4_qlora.py:490 forward 1.166 GB ×1, site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.638 GB ×38; Unsloth's: site:unsloth_zoo/temporary_patches/moe_triton_kernels.py:253 nf4_dequant_triton 0.805 GB ×1, site:torch/autograd/graph.py:882 _engine_run_backward 0.287 GB ×7, site:unsloth_zoo/temporary_patches/moe_utils.py:383 _grouped_mm_with_backward_fix 0.134 GB ×1 |
