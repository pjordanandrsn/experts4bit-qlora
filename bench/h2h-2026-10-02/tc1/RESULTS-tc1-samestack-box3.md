# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.46.0 @42280377c2dc2f4877a921aad751c2aec0582db6 (GitHub main)
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
e4b(t212) 0.46.0 @42280377c2dc2f4877a921aad751c2aec0582db6
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
 "box": "A",
 "run_id": "tc1-5090-72",
 "instance_id": "54227048",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "1056596104",
 "cgroup_memory_max": "367227043840",
 "disk_root": "overlay         320G   56M  320G   1% /",
 "hostname": "130d43f41b0c",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 25: the matched set with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0) (`qwen3samestack`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.979 | 376.4 | 27.500 | 923.0 | 2.0554→0.8170 | 1.9478→0.7576 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0027) | 60 | 8.759 | 179.1 | 24.273 | 1282.7 | 2.0405→0.8194 | 1.9506→0.7596 | 0.0019 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 3.834 | 394.1 | 27.448 | 955.8 | 2.0614→0.8184 | 1.9441→0.7582 | 0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 4.087 | 381.0 | 27.441 | 900.0 | 2.0614→0.8173 | 1.9441→0.7574 | -0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0027) | 60 | 8.500 | 182.5 | 24.273 | 1284.9 | 2.0405→0.8172 | 1.9506→0.7564 | -0.0012 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.490 | 443.6 | 27.493 | 878.5 | 2.0554→0.8185 | 1.9478→0.7568 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **184.5 s** before step 1 (42% of the arm): load_weights 73.3, c1_before 72.6, c1_after 28.7, adapter_save 25.9; unattributed 2.732; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **146.2 s** before step 1 (21% of the arm): adapter_save 54.2, c1_before 53.8, load_weights 52.2, c1_after 27.9; unattributed 12.733; budget 1260.0
- prologue `e4b/fused_attn4_m_t28` **144.5 s** before step 1 (37% of the arm): c1_before 59.1, load_weights 53.6, c1_after 31.2, adapter_save 21.8; unattributed 1.516; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **157.1 s** before step 1 (38% of the arm): c1_before 81.6, load_weights 52.4, c1_after 33.9, adapter_save 29.2; unattributed 1.661; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **139.2 s** before step 1 (21% of the arm): c1_before 55.1, load_weights 49.5, c1_after 28.4, adapter_save 22.2; unattributed 12.631; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **139.2 s** before step 1 (39% of the arm): c1_before 59.7, load_weights 55.1, c1_after 30.9, adapter_save 14.0; unattributed 2.582; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (3.979/3.490 s, |Δ|/mean 13.1% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (8.759/8.500 s, |Δ|/mean 3.0% vs 5%); `e4b/fused_attn4_m_t28` UNSTABLE (3.834/4.087 s, |Δ|/mean 6.4% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 3.979 / 3.490 s/step differ by 13.1% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 3.979 / 3.490 s/step differ by 13.1% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 3.979 / 3.490 s/step differ by 13.1% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0032): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0019, step-0 0.0027 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0067, paired rows mean +0.0019 ± 0.0020 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0006, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0026, paired rows mean +0.0006 ± 0.0018 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0002, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0059, paired rows mean -0.0002 ± 0.0011 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0012, step-0 0.0027 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0084, paired rows mean -0.0012 ± 0.0009 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0012, paired rows mean -0.0008 ± 0.0015 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_t28` same; `e4b/fused_attn4_m_t28_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3samestack | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m | **VALID** | VALID | 0.0019 |  |
| qwen3samestack | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestack | e4b/fused_attn4_m_t28 | **VALID** | VALID | 0.0006 |  |
| qwen3samestack | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | -0.0002 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0012 |  |
| qwen3samestack | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0008 |  |

## Predictions P50 / P51 / P52 (TC1-PREREG amendment 25: the matched position with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P50 | qwen3samestack | **UNTESTED** | two stable VALID draws a side are registered -- not quoted: e4b draws 3.979 / 3.490 s/step differ by 13.1% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE |
| P51 | qwen3samestack | **UNTESTED** | two stable VALID draws a side are registered -- venv-unsloth UNSTABLE: draws 3.979 / 3.490 s/step differ by 13.1% > 5% (UNSTABLE: reported, not quoted); venv-e4b UNSTABLE: draws 3.834 / 4.087 s/step differ by 6.4% > 5% (UNSTABLE: reported, not quoted) |
| P52 | qwen3samestack | **UNTESTED** | `e4b/reference_attn4_m` —; `unsloth/ckpt_unsloth_m` COMPARABLE; e4b parity NO-REF |
