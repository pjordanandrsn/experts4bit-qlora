# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.41.0 @635b66ba74a6cf459cab51c43f323c8705dd756e (GitHub main)
gnf4 0.34.1 @133ad9d4f35937a5fbd929703059b4f3b08010e9 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
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
 "run_id": "tc1-5090-41",
 "instance_id": "54000851",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "Intel(R) Core(TM) Ultra 9 285K",
 "nproc": 24,
 "mem_total_kb": "197216440",
 "cgroup_memory_max": "193871216640",
 "disk_root": "overlay         320G  1.6M  320G   1% /",
 "hostname": "831bb798366a",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 12: where e4b's step goes after #945, profiled; the legacy path as the before-picture) (`qwen3prof945`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_prof | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.207 | 650.2 | 24.581 | 452.9 | 2.0705→0.7982 | 1.9505→0.8147 | -0.0363 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.119 | 478.8 | 27.824 | 461.0 | 2.0705→0.8318 | 1.9505→0.8504 | -0.0007 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_prof_legacy | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.924 | 385.3 | 27.849 | 517.7 | 2.0705→0.8358 | 1.9505→0.8510 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_prof` **60.7 s** before step 1 (32% of the arm): c1_before 28.3, load_weights 15.1, c1_after 8.3, eval0 8.2; unattributed 0.614; budget 1231.3
- prologue `e4b/fused_attn4_m_prof` **56.2 s** before step 1 (24% of the arm): c1_before 26.1, load_weights 18.9, c1_after 6.1, preamble 3.1; unattributed 0.751; budget 1155.7
- prologue `e4b/fused_attn4_m_prof_legacy` **57.1 s** before step 1 (23% of the arm): c1_before 28.1, load_weights 17.1, c1_after 8.6, attn4 3.2; unattributed 0.748; budget 1066.8
- draws (R1): `e4b/fused_attn4_shipped_prof` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof_legacy` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b single draw (no second draw registered) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b single draw (no second draw registered) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b single draw (no second draw registered) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_prof` **COMPARABLE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0007, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0066, paired rows mean -0.0007 ± 0.0017 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_prof` same; `e4b/fused_attn4_m_prof_legacy` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3prof945 | e4b/fused_attn4_shipped_prof | **VALID** | VALID | -0.0363 |  |
| qwen3prof945 | e4b/fused_attn4_m_prof | **VALID** | VALID | -0.0007 |  |
| qwen3prof945 | e4b/fused_attn4_m_prof_legacy | **VALID** | VALID | 0.0000 |  |

## Amendment 12 (#945): the profile after the syncs are gone (descriptive) and P19
| arm | verdict | s/step (profiled run) | device busy | device events/step | CPU ops/step | CPU self by family |
|---|---|---|---|---|---|---|
| fused_attn4_shipped_prof | **VALID** | 2.207 | 0.830 | 133898 | 733073 | {"autograd": 0.1858, "fused_kernel": 0.0287, "matmul": 0.076, "memcpy": 0.2042, "norm_act": 0.1733, "optimizer": 0.0054, "other": 0.286, "routing": 0.0405} |
| fused_attn4_m_prof | **VALID** | 3.119 | 0.740 | 136969 | 750409 | {"autograd": 0.168, "fused_kernel": 0.0217, "matmul": 0.2438, "memcpy": 0.09, "norm_act": 0.1706, "optimizer": 0.0058, "other": 0.2656, "routing": 0.0346} |
| fused_attn4_m_prof_legacy | **VALID** | 3.924 | 0.595 | 140809 | 753087 | {"autograd": 0.1279, "fused_kernel": 0.0168, "matmul": 0.199, "memcpy": 0.0481, "norm_act": 0.1297, "optimizer": 0.0045, "other": 0.4453, "routing": 0.0286} |

| prediction | family | verdict | evidence |
|---|---|---|---|
| P19 | qwen3prof945 | **HELD** | device busy fraction matched new 0.740 vs legacy 0.595 (+0.145; >= +0.05 predicted); device events/step 136969 vs 140809; CPU ops/step 750409 vs 753087 |
