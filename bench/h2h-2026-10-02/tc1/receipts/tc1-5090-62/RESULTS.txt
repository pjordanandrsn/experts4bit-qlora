# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @5c74564e8753b66dc73399d524944a8f4ebede37 (GitHub main)
gnf4 0.37.0 @d6df825a5571506c31078bdbec5c7f4d7c189a00 (GitHub main)
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
 "run_id": "tc1-5090-62",
 "instance_id": "54179030",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "Intel(R) Core(TM) Ultra 9 285K",
 "nproc": 24,
 "mem_total_kb": "197216440",
 "cgroup_memory_max": "193871216640",
 "disk_root": "overlay         320G  9.4M  320G   1% /",
 "hostname": "5a1bc65e5ada",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Mixtral-8x7B-Instruct-v0.1 (amendment 22: grouped-nf4-gemm's fused 4-bit kernels vs its dense route, matched arm, resident, E4B_ABSMAX_DQ=1) (`mixtraldenseab`, registered n_layers 32, attention census 128)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dense0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.491 | 329.2 | 29.030 | 2121.6 | 1.4111→0.6955 | 1.4266→0.7156 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dense1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0059) | 20 | 3.592 | 500.9 | 29.020 | 1461.0 | 1.4104→0.6944 | 1.4207→0.7127 | -0.0029 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dense1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0059) | 20 | 3.589 | 501.1 | 29.023 | 1468.5 | 1.4104→0.6941 | 1.4207→0.7123 | -0.0033 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dense0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.541 | 333.1 | 29.025 | 1992.5 | 1.4111→0.6954 | 1.4266→0.7137 | -0.0019 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dense0` **102.2 s** before step 1 (48% of the arm): load_weights 46.4, c1_before 39.5, c1_after 20.3, eval0 5.7; unattributed 0.765; budget 1260.0
- prologue `e4b/fused_attn4_m_dense1` **74.4 s** before step 1 (51% of the arm): c1_before 37.7, load_weights 23.7, c1_after 19.1, attn4 4.7; unattributed 0.876; budget 1260.0
- prologue `e4b/fused_attn4_m_dense1_d2` **65.4 s** before step 1 (47% of the arm): c1_before 38.0, c1_after 19.2, load_weights 15.1, attn4 4.8; unattributed 0.744; budget 1260.0
- prologue `e4b/fused_attn4_m_dense0_d2` **66.3 s** before step 1 (38% of the arm): c1_before 39.2, c1_after 19.2, load_weights 14.9, attn4 4.8; unattributed 0.751; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_dense0` STABLE (5.491/5.541 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_m_dense1` STABLE (3.592/3.589 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0019): `e4b/fused_attn4_m_dense1` **COMPARABLE** (median step |Δ| 0.0022, |Δ held-out at N| 0.0029, step-0 0.0059 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0030, paired rows mean -0.0029 ± 0.0018 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dense1_d2` **COMPARABLE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0033, step-0 0.0059 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0018, paired rows mean -0.0033 ± 0.0018 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dense0_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0019, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0003, paired rows mean -0.0019 ± 0.0015 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m_dense0` same; `e4b/fused_attn4_m_dense1` same; `e4b/fused_attn4_m_dense1_d2` same; `e4b/fused_attn4_m_dense0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e809c37f09fd control detects, down nf4/64+dq sha 682dd5320a04 control detects, q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `e4b/fused_attn4_m_dense1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dense1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dense0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| mixtraldenseab | e4b/fused_attn4_m_dense0 | **VALID** | VALID | 0.0000 |  |
| mixtraldenseab | e4b/fused_attn4_m_dense1 | **VALID** | VALID | -0.0029 |  |
| mixtraldenseab | e4b/fused_attn4_m_dense1_d2 | **VALID** | VALID | -0.0033 |  |
| mixtraldenseab | e4b/fused_attn4_m_dense0_d2 | **VALID** | VALID | -0.0019 |  |

## Predictions P38 / P39 / P40 (TC1-PREREG amendment 22: grouped-nf4-gemm's dense route vs its fused kernels, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P38 | mixtraldenseab | **HELD** | Mixtral-8x7B: dense1 / dense0 0.651 [0.648, 0.654 over 4 cross-draw ratios] vs [0.55, 0.9]; s/step dense0 5.491 / 5.541 (within 0.9%), dense1 3.592 / 3.589 (within 0.1%); peak dense0 29.03 / dense1 29.02 GB; dense1 route counts (process) {"dense_dgrad": 5120, "dense_fwd": 11264, "dgrad": 0, "fwd": 0} |
| P39 | qwen3denseab | **UNTESTED** | Qwen3-30B-A3B: no qwen3denseab receipts in this directory |
| P40 | denseab | **UNTESTED** | mixtraldenseab: mean held-out dense1 - dense0 -0.0021 (|.| <= 0.01); held-out at N dense0 [0.7156, 0.7137] dense1 [0.7127, 0.7123]; qwen3denseab: no receipts |
