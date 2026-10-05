# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @e1faf47fd68bacf5f3a5bca9747ca3dc12832402 (GitHub main)
gnf4 0.41.0 @0e393f0886a8bf5cf78e2213dd61c9695cb21a49 (GitHub main)
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
e4b(t212) 0.48.0 @e1faf47fd68bacf5f3a5bca9747ca3dc12832402
gnf4(t212) 0.41.0 @0e393f0886a8bf5cf78e2213dd61c9695cb21a49
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
 "run_id": "tc1-5090-85",
 "instance_id": "54292473",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.119.02",
 "cpu": "AMD EPYC 9655 96-Core Processor",
 "nproc": 48,
 "mem_total_kb": "113303348",
 "cgroup_memory_max": "111380791296",
 "disk_root": "overlay         320G  2.3M  320G   1% /",
 "hostname": "2e44b764cc91",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 38: the compact padded LoRA delta's default decision, a third host, venv-unsloth) (`qwen3compactab3`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_cd0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.716 | 878.9 | 24.673 | 683.6 | 2.0554→0.8196 | 1.9478→0.7583 | 0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.742 | 926.1 | 24.673 | 630.2 | 2.0554→0.8152 | 1.9478→0.7557 | -0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.167 | 731.2 | 27.495 | 821.6 | 2.0554→0.8197 | 1.9478→0.7574 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.215 | 727.5 | 27.189 | 813.6 | 2.0554→0.8186 | 1.9478→0.7571 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.199 | 730.3 | 27.189 | 822.1 | 2.0554→0.8195 | 1.9478→0.7565 | -0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.178 | 734.9 | 27.507 | 776.1 | 2.0554→0.8165 | 1.9478→0.7546 | -0.0028 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.740 | 928.5 | 24.673 | 616.8 | 2.0554→0.8169 | 1.9478→0.7589 | 0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.720 | 928.1 | 24.673 | 627.5 | 2.0554→0.8170 | 1.9478→0.7564 | -0.0010 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_cd0` **276.2 s** before step 1 (71% of the arm): load_weights 226.4, c1_before 30.0, c1_after 15.0, eval0 8.5; unattributed 1.974; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd1` **63.3 s** before step 1 (38% of the arm): c1_before 30.0, load_weights 22.8, c1_after 15.1, preamble 2.8; unattributed 1.097; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0` **56.0 s** before step 1 (30% of the arm): c1_before 28.9, load_weights 14.6, c1_after 14.5, preamble 3.2; unattributed 1.899; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1` **53.7 s** before step 1 (29% of the arm): c1_before 30.6, c1_after 15.3, load_weights 11.7, preamble 2.7; unattributed 1.096; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1_d2` **53.4 s** before step 1 (29% of the arm): c1_before 30.5, c1_after 15.3, load_weights 11.6, preamble 2.7; unattributed 1.088; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0_d2` **53.5 s** before step 1 (29% of the arm): c1_before 30.5, c1_after 15.3, load_weights 11.7, preamble 2.7; unattributed 1.079; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd1_d2` **52.6 s** before step 1 (33% of the arm): c1_before 30.5, c1_after 15.4, load_weights 11.6, preamble 2.7; unattributed 1.082; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd0_d2` **53.0 s** before step 1 (34% of the arm): c1_before 30.8, c1_after 15.4, load_weights 11.7, preamble 2.8; unattributed 1.091; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_cd0` STABLE (1.716/1.720 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_shipped_cd1` STABLE (1.742/1.740 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_cd0` STABLE (2.167/2.178 s, |Δ|/mean 0.5% vs 5%); `e4b/fused_attn4_m_cd1` STABLE (2.215/2.199 s, |Δ|/mean 0.7% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0032): `e4b/fused_attn4_m_cd1` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0042, paired rows mean -0.0003 ± 0.0018 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd1_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0009, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0109, paired rows mean -0.0009 ± 0.0019 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0028, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0064, paired rows mean -0.0028 ± 0.0019 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_cd0` same; `e4b/fused_attn4_m_cd1` same; `e4b/fused_attn4_m_cd1_d2` same; `e4b/fused_attn4_m_cd0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_cd0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_cd1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_cd1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_cd0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

### Mixtral-8x7B-Instruct-v0.1 (amendment 38: the compact padded LoRA delta's default decision, matched arm, resident, venv-unsloth) (`mixtralcompactab`, registered n_layers 32, attention census 128)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_cd0 | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped_cd1 | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_cd0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.326 | 567.1 | 31.236 | 1485.4 | 1.4134→0.7131 | 1.4268→0.6317 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.374 | 560.3 | 31.207 | 1368.2 | 1.4134→0.7123 | 1.4268→0.6318 | 0.0001 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.371 | 559.9 | 31.188 | 1386.3 | 1.4134→0.7126 | 1.4268→0.6318 | 0.0001 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.330 | 566.0 | 31.232 | 1351.3 | 1.4134→0.7110 | 1.4268→0.6303 | -0.0014 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1_d2 | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped_cd0_d2 | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
- prologue `e4b/fused_attn4_m_cd0` **259.3 s** before step 1 (56% of the arm): load_weights 196.4, c1_before 47.0, c1_after 23.3, eval0 5.6; unattributed 1.388; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1` **72.6 s** before step 1 (26% of the arm): c1_before 46.4, c1_after 23.1, load_weights 14.6, attn4 3.8; unattributed 1.111; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1_d2` **70.3 s** before step 1 (26% of the arm): c1_before 46.5, c1_after 23.3, load_weights 12.3, attn4 3.8; unattributed 1.108; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0_d2` **70.2 s** before step 1 (26% of the arm): c1_before 46.6, c1_after 23.3, load_weights 12.1, attn4 3.8; unattributed 1.087; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_cd0` — (e4b/fused_attn4_shipped_cd0 is NOT_RUN); `e4b/fused_attn4_shipped_cd1` — (e4b/fused_attn4_shipped_cd1 is NOT_RUN); `e4b/fused_attn4_m_cd0` STABLE (3.326/3.330 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_cd1` STABLE (3.374/3.371 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0014): `e4b/fused_attn4_m_cd1` **COMPARABLE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0067, paired rows mean +0.0001 ± 0.0009 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd1_d2` **COMPARABLE** (median step |Δ| 0.0006, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0021, paired rows mean +0.0001 ± 0.0008 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd0_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0014, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0015, paired rows mean -0.0014 ± 0.0016 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m_cd0` same; `e4b/fused_attn4_m_cd1` same; `e4b/fused_attn4_m_cd1_d2` same; `e4b/fused_attn4_m_cd0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha c56e6fc0fd22 control detects, down nf4/64 sha 7e63cab94f9c control detects, q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `e4b/fused_attn4_m_cd1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3compactab3 | e4b/fused_attn4_shipped_cd0 | **VALID** | VALID | 0.0009 |  |
| qwen3compactab3 | e4b/fused_attn4_shipped_cd1 | **VALID** | VALID | -0.0017 |  |
| qwen3compactab3 | e4b/fused_attn4_m_cd0 | **VALID** | VALID | 0.0000 |  |
| qwen3compactab3 | e4b/fused_attn4_m_cd1 | **VALID** | VALID | -0.0003 |  |
| qwen3compactab3 | e4b/fused_attn4_m_cd1_d2 | **VALID** | VALID | -0.0009 |  |
| qwen3compactab3 | e4b/fused_attn4_m_cd0_d2 | **VALID** | VALID | -0.0028 |  |
| qwen3compactab3 | e4b/fused_attn4_shipped_cd1_d2 | **VALID** | VALID | 0.0015 |  |
| qwen3compactab3 | e4b/fused_attn4_shipped_cd0_d2 | **VALID** | VALID | -0.0010 |  |
| mixtralcompactab | e4b/fused_attn4_shipped_cd0 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtralcompactab | e4b/fused_attn4_shipped_cd1 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtralcompactab | e4b/fused_attn4_m_cd0 | **VALID** | VALID | 0.0000 |  |
| mixtralcompactab | e4b/fused_attn4_m_cd1 | **VALID** | VALID | 0.0001 |  |
| mixtralcompactab | e4b/fused_attn4_m_cd1_d2 | **VALID** | VALID | 0.0001 |  |
| mixtralcompactab | e4b/fused_attn4_m_cd0_d2 | **VALID** | VALID | -0.0014 |  |
| mixtralcompactab | e4b/fused_attn4_shipped_cd1_d2 | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtralcompactab | e4b/fused_attn4_shipped_cd0_d2 | **NOT_RUN** | — | — | skipped by TC1_SKIP |

## Predictions P77 / P78 / P79 / P82 (TC1-PREREG amendment 38: the compact padded LoRA delta's default decision, Qwen3-30B-A3B; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P79 | qwen3compactab3 | **HELD** | matched: peak cd0 27.501 / cd1 27.189 GB, drop 0.312 vs [-0.05, 99.0] |
| P77 | qwen3compactab3 | **FALSIFIED** | matched: cd1 / cd0 1.016 [1.010, 1.022 over 4 cross-draw ratios] vs [0.0, 0.99]; s/step cd0 2.167 / 2.178 (within 0.5%), cd1 2.215 / 2.199 (within 0.7%); peak cd0 27.50 / cd1 27.19 GB |
| P78 | qwen3compactab3 | **FALSIFIED** | shipped: cd1 / cd0 1.013 [1.011, 1.015 over 4 cross-draw ratios] vs [0.0, 0.99]; s/step cd0 1.716 / 1.720 (within 0.2%), cd1 1.742 / 1.740 (within 0.1%); peak cd0 24.67 / cd1 24.67 GB |
| P82 | qwen3compactab3 | **HELD** | matched: mean held-out cd1 - cd0 +0.0008 (|.| <= 0.005); held-out at N cd0 [0.7574, 0.7546] cd1 [0.7571, 0.7565]; shipped: mean held-out cd1 - cd0 -0.0000 (|.| <= 0.005); held-out at N cd0 [0.7583, 0.7564] cd1 [0.7557, 0.7589] |

## Predictions P80 / P81 / P83 (TC1-PREREG amendment 38: the compact padded LoRA delta's default decision, Mixtral-8x7B-Instruct-v0.1; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P81 | mixtralcompactab | **HELD** | matched: peak cd0 31.234 / cd1 31.197 GB, drop 0.037 vs [-0.05, 99.0] |
| P80 | mixtralcompactab | **FALSIFIED** | matched: cd1 / cd0 1.013 [1.012, 1.014 over 4 cross-draw ratios] vs [0.0, 1.01]; s/step cd0 3.326 / 3.330 (within 0.1%), cd1 3.374 / 3.371 (within 0.1%); peak cd0 31.23 / cd1 31.20 GB |
| P83 | mixtralcompactab | **HELD** | matched: mean held-out cd1 - cd0 +0.0008 (|.| <= 0.005); held-out at N cd0 [0.6317, 0.6303] cd1 [0.6318, 0.6318] |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3compactab3/e4b/fused_attn4_shipped_cd0 attempt 0 load1_median 1.1 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_shipped_cd1 attempt 0 load1_median 1.94 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_m_cd0 attempt 0 load1_median 1.1 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_m_cd1 attempt 0 load1_median 1.07 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_m_cd1_d2 attempt 0 load1_median 1.17 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_m_cd0_d2 attempt 0 load1_median 1.04 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_shipped_cd1_d2 attempt 0 load1_median 1.02 gate 6.0 status ok over 0`
- `qwen3compactab3/e4b/fused_attn4_shipped_cd0_d2 attempt 0 load1_median 1.13 gate 6.0 status ok over 0`
- `mixtralcompactab/e4b/fused_attn4_m_cd0 attempt 0 load1_median 1.04 gate 6.0 status ok over 0`
- `mixtralcompactab/e4b/fused_attn4_m_cd1 attempt 0 load1_median 1.16 gate 6.0 status ok over 0`
- `mixtralcompactab/e4b/fused_attn4_m_cd1_d2 attempt 0 load1_median 1.42 gate 6.0 status ok over 0`
- `mixtralcompactab/e4b/fused_attn4_m_cd0_d2 attempt 0 load1_median 1.1 gate 6.0 status ok over 0`
