# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.41.0 @43f36c79f6f2b0c028744a5e8f9acc43acb15c58 (GitHub main)
gnf4 0.34.1 @c8f0adcc67f4cbd954b4c12e50dd48bf8d024efd (GitHub main)
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
 "run_id": "tc1-5090-43",
 "instance_id": "54011443",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.159.03",
 "cpu": "AMD Ryzen 9 7950X 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "130979720",
 "cgroup_memory_max": "128756744192",
 "disk_root": "overlay         320G   35M  320G   1% /",
 "hostname": "ff64d9f56550",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 14: gnf4's max-keyed prefill M-tile vs the cost rule, on the trimmed delta and the post-#945 sync path) (`qwen3tileab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_tilemax | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.232 | 666.7 | 24.581 | 763.0 | 2.0705→0.7988 | 1.9505→0.8105 | -0.0394 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tilecost | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.077 | 756.7 | 24.581 | 656.7 | 2.0705→0.7986 | 1.9505→0.8095 | -0.0404 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tilemax | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.618 | 579.8 | 27.241 | 849.1 | 2.0705→0.8336 | 1.9505→0.8499 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tilecost | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.533 | 614.8 | 27.221 | 792.6 | 2.0705→0.8321 | 1.9505→0.8470 | -0.0029 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tilecost_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.543 | 611.5 | 27.221 | 804.1 | 2.0705→0.8325 | 1.9505→0.8484 | -0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tilemax_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.628 | 579.3 | 27.258 | 891.5 | 2.0705→0.8298 | 1.9505→0.8533 | 0.0034 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tilecost_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.049 | 764.4 | 24.581 | 644.8 | 2.0705→0.7966 | 1.9505→0.8105 | -0.0394 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tilemax_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.232 | 687.6 | 24.581 | 700.0 | 2.0705→0.7996 | 1.9505→0.8115 | -0.0383 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_tilemax` **66.2 s** before step 1 (59% of the arm): c1_before 33.5, c1_after 17.0, load_weights 15.0, eval0 7.7; unattributed 0.738; budget 1005.2
- prologue `e4b/fused_attn4_shipped_tilecost` **63.5 s** before step 1 (61% of the arm): c1_before 34.9, c1_after 17.4, load_weights 16.6, attn4 3.3; unattributed 0.72; budget 958.3
- prologue `e4b/fused_attn4_m_tilemax` **62.9 s** before step 1 (54% of the arm): c1_before 34.7, c1_after 17.0, load_weights 15.1, attn4 3.3; unattributed 0.738; budget 914.2
- prologue `e4b/fused_attn4_m_tilecost` **61.9 s** before step 1 (55% of the arm): c1_before 34.9, c1_after 17.4, load_weights 15.1, attn4 3.3; unattributed 0.739; budget 865.5
- prologue `e4b/fused_attn4_m_tilecost_d2` **62.1 s** before step 1 (55% of the arm): c1_before 34.8, c1_after 17.3, load_weights 15.1, attn4 3.4; unattributed 0.745; budget 818.3
- prologue `e4b/fused_attn4_m_tilemax_d2` **62.1 s** before step 1 (54% of the arm): c1_before 34.9, c1_after 17.2, load_weights 15.1, attn4 3.3; unattributed 0.738; budget 771.0
- prologue `e4b/fused_attn4_shipped_tilecost_d2` **61.1 s** before step 1 (60% of the arm): c1_before 35.0, c1_after 17.1, load_weights 15.2, attn4 3.4; unattributed 0.746; budget 722.8
- prologue `e4b/fused_attn4_shipped_tilemax_d2` **60.2 s** before step 1 (57% of the arm): c1_before 34.0, c1_after 17.5, load_weights 15.1, attn4 3.3; unattributed 0.735; budget 679.7
- draws (R1): `e4b/fused_attn4_shipped_tilemax` STABLE (2.232/2.232 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_shipped_tilecost` STABLE (2.077/2.049 s, |Δ|/mean 1.4% vs 5%); `e4b/fused_attn4_m_tilemax` STABLE (2.618/2.628 s, |Δ|/mean 0.4% vs 5%); `e4b/fused_attn4_m_tilecost` STABLE (2.533/2.543 s, |Δ|/mean 0.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0034): `e4b/fused_attn4_m_tilecost` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0029, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0094, paired rows mean -0.0029 ± 0.0028 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_tilecost_d2` **COMPARABLE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0015, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0143, paired rows mean -0.0015 ± 0.0017 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_tilemax_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0034, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0010, paired rows mean +0.0034 ± 0.0031 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_tilemax` same; `e4b/fused_attn4_m_tilecost` same; `e4b/fused_attn4_m_tilecost_d2` same; `e4b/fused_attn4_m_tilemax_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_tilemax`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tilecost`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tilecost`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tilecost_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tilemax_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tilecost_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tilemax_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3tileab | e4b/fused_attn4_shipped_tilemax | **VALID** | VALID | -0.0394 |  |
| qwen3tileab | e4b/fused_attn4_shipped_tilecost | **VALID** | VALID | -0.0404 |  |
| qwen3tileab | e4b/fused_attn4_m_tilemax | **VALID** | VALID | 0.0000 |  |
| qwen3tileab | e4b/fused_attn4_m_tilecost | **VALID** | VALID | -0.0029 |  |
| qwen3tileab | e4b/fused_attn4_m_tilecost_d2 | **VALID** | VALID | -0.0015 |  |
| qwen3tileab | e4b/fused_attn4_m_tilemax_d2 | **VALID** | VALID | 0.0034 |  |
| qwen3tileab | e4b/fused_attn4_shipped_tilecost_d2 | **VALID** | VALID | -0.0394 |  |
| qwen3tileab | e4b/fused_attn4_shipped_tilemax_d2 | **VALID** | VALID | -0.0383 |  |

## Predictions P22 / P23 (TC1-PREREG amendment 14, #945: gnf4's cost tile rule vs the max-keyed M-tile, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P22 | qwen3tileab | **HELD** | shipped: tilecost / tilemax 0.924 [0.918, 0.930 over 4 cross-draw ratios] vs [0.85, 0.97]; decision reading: flip-eligible; s/step tilemax 2.232 / 2.232 (within 0.0%), tilecost 2.077 / 2.049 (within 1.4%); held-out at N tilemax 0.8105 / tilecost 0.8095; tile heights launched (process) tilemax {"128": 15592, "16": 0, "32": 114, "64": 1190} tilecost {"128": 1256, "16": 754, "32": 4096, "64": 10790} |
| P23 | qwen3tileab | **HELD** | matched: tilecost / tilemax 0.968 [0.964, 0.971 over 4 cross-draw ratios] vs [0.88, 0.98]; decision reading: flip-eligible; s/step tilemax 2.618 / 2.628 (within 0.4%), tilecost 2.533 / 2.543 (within 0.4%); held-out at N tilemax 0.8499 / tilecost 0.8470; tile heights launched (process) tilemax {"128": 15562, "16": 0, "32": 110, "64": 1224} tilecost {"128": 1320, "16": 756, "32": 4052, "64": 10768} |
