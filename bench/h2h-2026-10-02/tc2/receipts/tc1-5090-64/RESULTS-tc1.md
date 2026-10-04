# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @ed08029d9fb8d210ac78e9371f64bd06098ed07d (GitHub main)
gnf4 0.38.0 @bb56b42c574d043df9428b85b17066f5bf6c092a (GitHub main)
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
e4b(t212) 0.45.0 @ed08029d9fb8d210ac78e9371f64bd06098ed07d
gnf4(t212) 0.38.0 @bb56b42c574d043df9428b85b17066f5bf6c092a
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
 "box": "B",
 "run_id": "tc1-5090-64",
 "instance_id": "54187395",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "03467dfb3121",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC2 (`tc2small` / `tc2big` tokens, TC2-PREREG.md, drafted in TC2-PREREG-draft): TC1's readings per family with the registered n_layers {'granite': 32, 'olmoe': 16, 'gptoss': 24, 'qwen3_5': 40, 'mixtral': 32} and attention census {'granite': 128, 'olmoe': 64, 'gptoss': None, 'qwen3_5': None, 'mixtral': 128} (None = the receipt's own structural census governs); the HF position is quoted as HF (bf16 experts) / e4b (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID (attention-only when it adapted no expert parameter), as tp4; gpt-oss carries a NO COMMON ADAPTER SET line (both s/step values, both trainable counts), never a ratio; mixtral's FOOTPRINT line (e4b under expert offload vs Unsloth resident: peak VRAM and s/step) leads its block; `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert parameters of a recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime; an HF t214 arm whose dispatch did not reach grouped_mm is recorded, never VOID; P1–P7 of the draft scored HELD / FALSIFIED / UNTESTED (P1 HF band (1.1, 1.6); P2 Unsloth (1.2, 3.0) / HF (1.5, 2.5); P4 Unsloth (2.0, 6.0), e4b within 15% of tp4's 6.3344 s/step; P5 e4b/Unsloth (0.3, 0.5) at a >= 8x lower e4b peak, tp2 0.361).

### Qwen3.6-35B-A3B (lane TC2, box B) (`qwen3_5`, registered n_layers 40)
- model `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd9`; tokens sha `0d94e9b88936`; N=20; fixture template alpaca seq 2048 micro-batch 1 × accum 8 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 926187520; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 9.325 | 168.3 | 31.345 | 1624.3 | 1.1830→0.7363 | 1.1770→0.6885 | 0.0000 | patched 40 / kcalls 1280 | 926187520 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | NEAR (0.0191) | 20 | 18.985 | 74.2 | 30.410 | 2849.4 | 1.1919→0.7339 | 1.1962→0.6875 | -0.0010 | stacks 80 / fwd 640 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 9.170 | 175.1 | 31.345 | 1595.5 | 1.1830→0.7342 | 1.1770→0.6879 | -0.0006 | patched 40 / kcalls 1280 | 926187520 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | NEAR (0.0191) | 20 | 18.914 | 74.7 | 30.410 | 2842.4 | 1.1919→0.7338 | 1.1962→0.6888 | 0.0003 | stacks 80 / fwd 640 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| unsloth | ckpt_unsloth_m_experts | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
- prologue `e4b/fused_attn4_m` **115.5 s** before step 1 (38% of the arm): c1_before 55.6, load_weights 28.6, c1_after 26.9, eval0 10.3; unattributed 1.208; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **161.6 s** before step 1 (27% of the arm): c1_before 55.4, eval0 47.7, load_weights 36.6, c1_after 26.6; unattributed 9.565; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **109.6 s** before step 1 (38% of the arm): c1_before 55.3, load_weights 30.1, c1_after 29.1, attn4 7.6; unattributed 1.239; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **150.7 s** before step 1 (26% of the arm): c1_before 54.4, eval0 38.9, load_weights 34.1, c1_after 27.8; unattributed 10.231; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (9.325/9.170 s, |Δ|/mean 1.7% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (18.985/18.914 s, |Δ|/mean 0.4% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **MATCHED POSITION (micro-batch 1 × accum 8): s/step ratio unsloth/e4b = 2.049** [2.028, 2.070 over 4 cross-draw ratios] (18.949 vs 9.248 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 30.41 vs e4b 31.34 GB (Δ -0.93); J/step unsloth 2845.9 vs e4b 1609.9 (×1.768); tok/s unsloth 74.5 vs e4b 171.7; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498)
- quality reading at N=20 (unsloth): held-out e4b 0.6885 / unsloth 0.6875 (Δ -0.0010) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0191
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b STABLE / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b STABLE / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0013): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0010, step-0 0.0191 NEAR, |Δ loss at step 2| 0.0066, paired rows mean -0.0010 ± 0.0023 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0006, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0051, paired rows mean -0.0006 ± 0.0015 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0003, step-0 0.0191 NEAR, |Δ loss at step 2| 0.0103, paired rows mean +0.0003 ± 0.0026 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **—** — no OK receipt
- matched_init_sha (B, name-free, canonical slot order): anchor `4b90bf05bd02baf3`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha c798a2390c7a control detects, down nf4/64+dq sha 20146c64832c control detects, q_proj nf4/64+dq sha b44f6f975053 control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3_5 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3_5 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0010 |  |
| qwen3_5 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0006 |  |
| qwen3_5 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0003 |  |
| qwen3_5 | unsloth/ckpt_unsloth_m_experts | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3_5 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **UNTESTED** | e4b's anchor ran micro-batch 1: P4's e4b leg reads the field recipe's step (tp4's 6.3344 s/step at micro-batch 2); TC2 amendment 8's box Q quotes its pair on the position line |
| P5 | mixtral | **UNTESTED** | no receipts |
| P6 | tc2 | **UNTESTED** | no family with both e4b arms OK |
| P7 | tc2 | **FALSIFIED** | qwen3_5 unsloth/ckpt_unsloth_m COMPARABLE; qwen3_5 unsloth/ckpt_unsloth_m_d2 COMPARABLE |
