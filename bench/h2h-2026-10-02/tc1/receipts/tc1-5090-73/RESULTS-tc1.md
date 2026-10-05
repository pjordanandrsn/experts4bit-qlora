# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.46.0 @306dfa9c890b22c9227672cfa6725001d233fd97 (GitHub main)
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
 "run_id": "tc1-5090-73",
 "instance_id": "54227619",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "9325308abaa7",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 26: Triton launches prebound off vs on, e4b + grouped-nf4-gemm, venv-e4b) (`qwen3prebindab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable None; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_pb0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.092 | 490.6 | 24.623 | 908.8 | 2.0614→0.8172 | 1.9441→0.7575 | N-A (anchor missing) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.056 | 519.6 | 24.623 | 874.5 | 2.0614→0.8174 | 1.9441→0.7589 | N-A (anchor missing) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb0 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_pb1 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_pb1_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_pb0_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped_pb1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.960 | 535.2 | 24.623 | 857.5 | 2.0614→0.8186 | 1.9441→0.7565 | N-A (anchor missing) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.047 | 520.0 | 24.623 | 873.7 | 2.0614→0.8159 | 1.9441→0.7556 | N-A (anchor missing) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_pb0` **113.7 s** before step 1 (36% of the arm): c1_before 53.8, load_weights 28.2, c1_after 26.8, eval0 11.4; unattributed 1.267; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb1` **116.7 s** before step 1 (38% of the arm): c1_before 62.9, c1_after 32.3, load_weights 30.9, attn4 6.2; unattributed 1.349; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb1_d2` **109.4 s** before step 1 (37% of the arm): c1_before 55.9, load_weights 30.0, c1_after 27.1, attn4 6.8; unattributed 1.268; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb0_d2` **106.9 s** before step 1 (36% of the arm): c1_before 54.5, load_weights 29.9, c1_after 27.9, trainable_sha 6.2; unattributed 1.242; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_pb0` STABLE (3.092/3.047 s, |Δ|/mean 1.5% vs 5%); `e4b/fused_attn4_shipped_pb1` STABLE (3.056/2.960 s, |Δ|/mean 3.2% vs 5%); `e4b/fused_attn4_m_pb0` — (e4b/fused_attn4_m_pb0 is NOT_RUN); `e4b/fused_attn4_m_pb1` — (e4b/fused_attn4_m_pb1 is NOT_RUN)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m_pb0 is NOT_RUN / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m_pb0 is NOT_RUN / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m_pb0 is NOT_RUN / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0024): `e4b/fused_attn4_m_pb1` **—** — no OK receipt; `e4b/fused_attn4_m_pb1_d2` **—** — no OK receipt; `e4b/fused_attn4_m_pb0_d2` **—** — no OK receipt
- frozen base `e4b/fused_attn4_shipped_pb0`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `e4b/fused_attn4_shipped_pb1`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `e4b/fused_attn4_shipped_pb1_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `e4b/fused_attn4_shipped_pb0_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3prebindab | e4b/fused_attn4_shipped_pb0 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3prebindab | e4b/fused_attn4_shipped_pb1 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3prebindab | e4b/fused_attn4_m_pb0 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3prebindab | e4b/fused_attn4_m_pb1 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3prebindab | e4b/fused_attn4_m_pb1_d2 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3prebindab | e4b/fused_attn4_m_pb0_d2 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3prebindab | e4b/fused_attn4_shipped_pb1_d2 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3prebindab | e4b/fused_attn4_shipped_pb0_d2 | **VALID** | VALID | N-A (anchor missing) |  |

## Predictions P53 / P54 / P55 (TC1-PREREG amendment 26: prebound Triton launches off vs on, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P53 | qwen3prebindab | **HELD** | shipped: pb1 / pb0 0.980 [0.957, 1.003 over 4 cross-draw ratios] vs [0.9, 0.98]; s/step pb0 3.092 / 3.047 (within 1.5%), pb1 3.056 / 2.960 (within 3.2%); pb1 launch counts (process) {"e4b": {"prebound": 216335, "triton": 753}, "gnf4": {"prebound": 72162, "triton": 30}}; triton 3.4.0 |
| P54 | qwen3prebindab | **UNTESTED** | matched: two stable VALID draws a side are registered -- pb0 —: e4b/fused_attn4_m_pb0 is NOT_RUN; pb1 —: e4b/fused_attn4_m_pb1 is NOT_RUN |
| P55 | qwen3prebindab | **UNTESTED** | shipped: mean held-out pb1 - pb0 +0.0011 (|.| <= 0.005); held-out at N pb0 [0.7575, 0.7556] pb1 [0.7589, 0.7565]; matched: pb0 —: e4b/fused_attn4_m_pb0 is NOT_RUN; pb1 —: e4b/fused_attn4_m_pb1 is NOT_RUN |
