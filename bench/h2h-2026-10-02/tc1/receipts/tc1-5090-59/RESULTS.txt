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
 "run_id": "tc1-5090-59",
 "instance_id": "54177463",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9454P 48-Core Emb Processor",
 "nproc": 96,
 "mem_total_kb": "527772888",
 "cgroup_memory_max": "518820724736",
 "disk_root": "overlay         320G   56M  320G   1% /",
 "hostname": "c014519455cd",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 22: grouped-nf4-gemm's fused 4-bit kernels vs its dense route, matched arm) (`qwen3denseab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dense0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.949 | 450.9 | 27.157 | 979.1 | 2.0614→0.8332 | 1.9441→0.8497 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dense1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0231) | 20 | 8.674 | 173.0 | 27.157 | 1343.3 | 2.0682→0.8326 | 1.9672→0.8482 | -0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dense1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0231) | 20 | 8.784 | 171.4 | 27.120 | 1297.6 | 2.0682→0.8340 | 1.9672→0.8502 | 0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dense0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.976 | 505.6 | 27.157 | 921.3 | 2.0614→0.8325 | 1.9441→0.8489 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dense0` **86.2 s** before step 1 (56% of the arm): c1_before 46.7, c1_after 23.4, load_weights 17.2, eval0 8.7; unattributed 1.006; budget 1260.0
- prologue `e4b/fused_attn4_m_dense1` **85.9 s** before step 1 (32% of the arm): c1_before 47.1, c1_after 23.8, load_weights 19.6, eval0 5.6; unattributed 0.96; budget 1260.0
- prologue `e4b/fused_attn4_m_dense1_d2` **83.0 s** before step 1 (31% of the arm): c1_before 47.1, c1_after 23.6, load_weights 17.2, eval0 5.0; unattributed 0.966; budget 1260.0
- prologue `e4b/fused_attn4_m_dense0_d2` **79.7 s** before step 1 (56% of the arm): c1_before 47.2, c1_after 23.6, load_weights 17.2, attn4 4.5; unattributed 0.977; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_dense0` STABLE (2.949/2.976 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_m_dense1` STABLE (8.674/8.784 s, |Δ|/mean 1.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0020): `e4b/fused_attn4_m_dense1` **COMPARABLE** (median step |Δ| 0.0035, |Δ held-out at N| 0.0015, step-0 0.0231 NEAR, |Δ loss at step 2| 0.0122, paired rows mean -0.0015 ± 0.0021 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dense1_d2` **COMPARABLE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0005, step-0 0.0231 NEAR, |Δ loss at step 2| 0.0208, paired rows mean +0.0005 ± 0.0027 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dense0_d2` **COMPARABLE** (median step |Δ| 0.0007, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0065, paired rows mean -0.0008 ± 0.0026 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_dense0` same; `e4b/fused_attn4_m_dense1` same; `e4b/fused_attn4_m_dense1_d2` same; `e4b/fused_attn4_m_dense0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_dense1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dense1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dense0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3denseab | e4b/fused_attn4_m_dense0 | **VALID** | VALID | 0.0000 |  |
| qwen3denseab | e4b/fused_attn4_m_dense1 | **VALID** | VALID | -0.0015 |  |
| qwen3denseab | e4b/fused_attn4_m_dense1_d2 | **VALID** | VALID | 0.0005 |  |
| qwen3denseab | e4b/fused_attn4_m_dense0_d2 | **VALID** | VALID | -0.0008 |  |

## Predictions P38 / P39 / P40 (TC1-PREREG amendment 22: grouped-nf4-gemm's dense route vs its fused kernels, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P38 | mixtraldenseab | **UNTESTED** | Mixtral-8x7B: no mixtraldenseab receipts in this directory |
| P39 | qwen3denseab | **FALSIFIED** | Qwen3-30B-A3B: dense1 / dense0 2.947 [2.915, 2.979 over 4 cross-draw ratios] vs [0.85, 1.15]; above 0.95 (fused stays at every group count on this family); s/step dense0 2.949 / 2.976 (within 0.9%), dense1 8.674 / 8.784 (within 1.3%); peak dense0 27.16 / dense1 27.14 GB; dense1 route counts (process) {"dense_dgrad": 7680, "dense_fwd": 16896, "dgrad": 0, "fwd": 0} |
| P40 | denseab | **UNTESTED** | mixtraldenseab: no receipts; qwen3denseab: mean held-out dense1 - dense0 -0.0001 (|.| <= 0.01); held-out at N dense0 [0.8497, 0.8489] dense1 [0.8482, 0.8502] |
