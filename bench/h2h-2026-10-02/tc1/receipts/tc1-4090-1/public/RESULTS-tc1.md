# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.57.0 @576d99e9fef92c406642e995d9945fc5e4cf3292 (GitHub main)
gnf4 0.50.0 @efdda943d5a310343ed7fef2e23d1cbf2df27e51 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
huggingface_hub(e4b/hf) 2.3.0
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
e4b(t212) 0.57.0 @576d99e9fef92c406642e995d9945fc5e4cf3292
gnf4(t212) 0.50.0 @efdda943d5a310343ed7fef2e23d1cbf2df27e51
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
 "run_id": "tc1-4090-1",
 "gpu": "NVIDIA GeForce RTX 4090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7702P 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527914168",
 "cgroup_memory_max": "259479568384",
 "disk_root": "overlay         220G   13M  220G   1% /",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "0-127",
 "registered_gpu_class": "4090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 78: the field recipe on a 24 GiB RTX 4090 -- pb1cl256, every LoRA delta width-bucketed through the compact bucketed node and 256-token LM-loss chunks, 60 steps; two boxes on two machines) (`qwen3pos78fit`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 4090 gpu NVIDIA GeForce RTX 4090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_fd_pb1cl256 | **OK** | VALID | **VALID** | native | matched:3407 (complete 12480/12480) / float32 | — | 60 | 3.520 | 411.7 | 23.926 | 850.7 | 2.0490→0.8167 | 1.9760→0.7544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_fd_pb1cl256` **110.1 s** before step 1 (32% of the arm): c1_before 61.4, c1_after 30.3, load_weights 25.0, attn4 6.2; unattributed 2.413; budget 765.1
- draws (R1): `e4b/fused_attn4_m_fd_pb1cl256` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b only, one arm a box; s/step is recorded, never gated (amendment 78)
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b only, one arm a box; s/step is recorded, never gated (amendment 78)
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b only, one arm a box; s/step is recorded, never gated (amendment 78)
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64 sha 5c5e402ce51d control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos78fit | e4b/fused_attn4_m_fd_pb1cl256 | **VALID** | VALID | 0.0000 |  |

## Amendment 78: the field recipe on a 24 GiB RTX 4090 (pb1cl256; this box is ONE draw -- the verdict reads every started draw, `--pos78-lane`)

| draw state | memory.total | machine id read | steps trained | usable - max allocated | usable - max reserved | total - device used | held-out at 60 |
|---|---|---|---|---|---|---|---|
| VOID | 24564 MiB | no | 60 | 1,260 MiB (step 32) | 586 MiB (step 29) | 586 MiB (step 29) | 0.75444 (tc1-5090-155's cd1cl1 0.75489; reported, not like-for-like) |
- **VOID:** the machine id is unread (the launcher's receipt beside the tc1 dir)

## Amendment 78: the census table (Qwen3-30B-A3B, pb1cl256, this draw; GB = 1e9 bytes)

| | pb1cl256 |
|---|---|
| peak allocated / reserved (GB) | 23.926 / 24.633 |
| attributed fraction at the peak | 1.0000 |
| peak at (checkpoint / phase) | s18.mb3 / s18.mb3.backward |
| window holds the run's peak / events / ring full | True / 1000000 / True |
| snapshots / census s / torch | 7 / 21.49 / 2.12.1+cu130 |

**Static census by class (GB: end of setup / end of training)**

| | pb1cl256 |
|---|---|
| frozen_expert_weights | 14.496 / 14.496 |
| expert_absmax | 0.460 / 0.460 |
| expert_absmax.nested_code | 0.000 / 0.000 |
| expert_absmax.nested_off | 0.000 / 0.000 |
| expert_absmax.nested_q | 0.453 / 0.453 |
| expert_absmax.nested_s | 0.007 / 0.007 |
| other_frozen | 1.761 / 1.761 |
| other_frozen.bf16 | 1.270 / 1.270 |
| other_frozen.nf4_linear4bit | 0.491 / 0.491 |
| trainable_adapters | 2.570 / 2.570 |
| adapter_grads | 0.000 / 0.000 |
| optimizer_state | 0.000 / 1.305 |
| other_buffers | 0.252 / 0.252 |
| other | -0.243 / -0.233 |
| allocated_bytes | 19.296 / 20.611 |

**Live bytes at the peak by class (GB; transient = every live block that holds no static tensor)**

| | pb1cl256 |
|---|---|
| frozen_expert_weights | 14.496 |
| expert_absmax | 0.460 |
| other_frozen | 1.509 |
| trainable_adapters | 2.570 |
| adapter_grads | 2.570 |
| optimizer_state | 1.305 |
| other_buffers | 0.252 |
| transient | 0.753 |

**Top live-at-peak groups (GB ×count; 20 of the receipts' 40 shown)**

| rank | pb1cl256 |
|---|---|
| 1 | static:frozen_expert_weights 14.496 ×96 |
| 2 | static:adapter_grads 2.570 ×480 |
| 3 | static:trainable_adapters 2.570 ×576 |
| 4 | static:optimizer_state 1.305 ×2306 |
| 5 | static:other_frozen.bf16 1.270 ×243 |
| 6 | static:expert_absmax 0.460 ×384 |
| 7 | static:other_buffers 0.252 ×98 |
| 8 | static:other_frozen.nf4_linear4bit 0.239 ×480 |
| 9 | site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 forward 0.218 ×48 |
| 10 | site:nf4_qlora.py:845 backward 0.097 ×1 |
| 11 | site:nf4_qlora.py:847 backward 0.097 ×1 |
| 12 | site:torch/autograd/graph.py:882 _engine_run_backward 0.082 ×2 |
| 13 | site:nf4_qlora.py:834 backward 0.073 ×1 |
| 14 | site:experts4bit_qlora/engines/fast.py:773 fused_experts_train_forward 0.036 ×1 |
| 15 | site:experts4bit_qlora/lora.py:858 forward 0.018 ×2 |
| 16 | site:transformers/integrations/sdpa_attention.py:25 repeat_kv 0.018 ×2 |
| 17 | site:nf4_qlora.py:830 backward 0.014 ×1 |
| 18 | site:nf4_qlora.py:853 backward 0.014 ×8 |
| 19 | site:transformers/integrations/sdpa_attention.py:92 sdpa_attention_forward 0.010 ×3 |
| 20 | site:nf4_qlora.py:842 backward 0.010 ×8 |

## Load gate (TC1-PREREG amendment 33): 2 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos78fit/e4b/fused_attn4_m_fd_pb1cl256 attempt 0 load1_median 17.84 gate 6.0 status ok over 1`
- `qwen3pos78fit/e4b/fused_attn4_m_fd_pb1cl256 attempt 0 VOID (host load1 median 17.84 > 6.0): re-run 1 of 2`
- `qwen3pos78fit/e4b/fused_attn4_m_fd_pb1cl256 attempt 1 load1_median 18.09 gate 6.0 status ok over 1`
- `qwen3pos78fit/e4b/fused_attn4_m_fd_pb1cl256 attempt 1 VOID (host load1 median 18.09 > 6.0): re-run 2 of 2`
- `qwen3pos78fit/e4b/fused_attn4_m_fd_pb1cl256 attempt 2 load1_median 18.22 gate 6.0 status ok over 1`
