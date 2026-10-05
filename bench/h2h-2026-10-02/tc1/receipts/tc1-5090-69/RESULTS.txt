# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.46.0 @6f99fefd4294b1ca0d4f7fb710c3c094ca9039e2 (GitHub main)
gnf4 0.39.0 @f0c1ece95902a7b7d004b0f2af331dcd9f06e7c7 (GitHub main)
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
 "run_id": "tc1-5090-69",
 "instance_id": "54223080",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "Intel(R) Core(TM) Ultra 9 285K",
 "nproc": 24,
 "mem_total_kb": "197216440",
 "cgroup_memory_max": "193871216640",
 "disk_root": "overlay         320G  9.5M  320G   1% /",
 "hostname": "27c8d701c24c",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 26: Triton launches prebound off vs on, e4b + grouped-nf4-gemm, venv-e4b) (`qwen3prebindab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_pb0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 1.925 | 714.4 | 24.576 | 699.0 | 2.0614→0.7973 | 1.9441→0.8135 | -0.0389 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 1.972 | 802.5 | 24.576 | 614.7 | 2.0614→0.7981 | 1.9441→0.8113 | -0.0410 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.555 | 601.4 | 27.137 | 765.6 | 2.0614→0.8351 | 1.9441→0.8524 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.447 | 632.7 | 27.157 | 795.6 | 2.0614→0.8317 | 1.9441→0.8513 | -0.0011 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.476 | 629.9 | 27.157 | 775.1 | 2.0614→0.8323 | 1.9441→0.8480 | -0.0044 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.505 | 618.3 | 27.155 | 790.4 | 2.0614→0.8327 | 1.9441→0.8493 | -0.0031 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 1.970 | 802.5 | 24.576 | 622.0 | 2.0614→0.7979 | 1.9441→0.8115 | -0.0408 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 1.772 | 879.5 | 24.576 | 613.9 | 2.0614→0.7958 | 1.9441→0.8149 | -0.0374 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_pb0` **58.8 s** before step 1 (58% of the arm): c1_before 25.6, load_weights 16.9, c1_after 13.0, eval0 6.8; unattributed 0.758; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb1` **53.8 s** before step 1 (58% of the arm): c1_before 26.5, load_weights 17.3, c1_after 13.4, preamble 3.1; unattributed 0.748; budget 1260.0
- prologue `e4b/fused_attn4_m_pb0` **54.3 s** before step 1 (51% of the arm): c1_before 26.4, load_weights 16.1, c1_after 14.7, preamble 3.1; unattributed 0.742; budget 1260.0
- prologue `e4b/fused_attn4_m_pb1` **55.9 s** before step 1 (53% of the arm): c1_before 28.0, load_weights 16.2, c1_after 14.2, preamble 3.3; unattributed 0.748; budget 1260.0
- prologue `e4b/fused_attn4_m_pb1_d2` **55.9 s** before step 1 (53% of the arm): c1_before 28.0, load_weights 16.4, c1_after 14.1, attn4 3.3; unattributed 0.749; budget 1260.0
- prologue `e4b/fused_attn4_m_pb0_d2` **55.9 s** before step 1 (53% of the arm): c1_before 28.3, load_weights 16.1, c1_after 14.2, attn4 3.3; unattributed 0.748; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb1_d2` **54.3 s** before step 1 (58% of the arm): c1_before 27.9, load_weights 16.1, c1_after 14.1, attn4 3.2; unattributed 0.75; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb0_d2` **50.7 s** before step 1 (59% of the arm): c1_before 28.5, c1_after 14.3, load_weights 12.7, preamble 3.2; unattributed 0.601; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_pb0` UNSTABLE (1.925/1.772 s, |Δ|/mean 8.3% vs 5%); `e4b/fused_attn4_shipped_pb1` STABLE (1.972/1.970 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_pb0` STABLE (2.555/2.505 s, |Δ|/mean 2.0% vs 5%); `e4b/fused_attn4_m_pb1` STABLE (2.447/2.476 s, |Δ|/mean 1.2% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0033): `e4b/fused_attn4_m_pb1` **COMPARABLE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0011, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0168, paired rows mean -0.0011 ± 0.0012 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pb1_d2` **COMPARABLE** (median step |Δ| 0.0023, |Δ held-out at N| 0.0044, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0153, paired rows mean -0.0044 ± 0.0021 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pb0_d2` **COMPARABLE** (median step |Δ| 0.0021, |Δ held-out at N| 0.0031, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0152, paired rows mean -0.0031 ± 0.0023 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_pb0` same; `e4b/fused_attn4_m_pb1` same; `e4b/fused_attn4_m_pb1_d2` same; `e4b/fused_attn4_m_pb0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_pb0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pb1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pb0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pb1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pb0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3prebindab | e4b/fused_attn4_shipped_pb0 | **VALID** | VALID | -0.0389 |  |
| qwen3prebindab | e4b/fused_attn4_shipped_pb1 | **VALID** | VALID | -0.0410 |  |
| qwen3prebindab | e4b/fused_attn4_m_pb0 | **VALID** | VALID | 0.0000 |  |
| qwen3prebindab | e4b/fused_attn4_m_pb1 | **VALID** | VALID | -0.0011 |  |
| qwen3prebindab | e4b/fused_attn4_m_pb1_d2 | **VALID** | VALID | -0.0044 |  |
| qwen3prebindab | e4b/fused_attn4_m_pb0_d2 | **VALID** | VALID | -0.0031 |  |
| qwen3prebindab | e4b/fused_attn4_shipped_pb1_d2 | **VALID** | VALID | -0.0408 |  |
| qwen3prebindab | e4b/fused_attn4_shipped_pb0_d2 | **VALID** | VALID | -0.0374 |  |

## Predictions P53 / P54 / P55 (TC1-PREREG amendment 26: prebound Triton launches off vs on, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P53 | qwen3prebindab | **UNTESTED** | shipped: two stable VALID draws a side are registered -- pb0 UNSTABLE: draws 1.925 / 1.772 s/step differ by 8.3% > 5% (UNSTABLE: reported, not quoted); pb1 STABLE: |
| P54 | qwen3prebindab | **HELD** | matched: pb1 / pb0 0.973 [0.958, 0.988 over 4 cross-draw ratios] vs [0.92, 0.99]; s/step pb0 2.555 / 2.505 (within 2.0%), pb1 2.447 / 2.476 (within 1.2%); pb1 launch counts (process) {"e4b": {"prebound": 73591, "triton": 313}, "gnf4": {"prebound": 24546, "triton": 30}}; triton 3.4.0 |
| P55 | qwen3prebindab | **UNTESTED** | shipped: pb0 UNSTABLE: draws 1.925 / 1.772 s/step differ by 8.3% > 5% (UNSTABLE: reported, not quoted); pb1 STABLE:; matched: mean held-out pb1 - pb0 -0.0012 (|.| <= 0.005); held-out at N pb0 [0.8524, 0.8493] pb1 [0.8513, 0.848] |
