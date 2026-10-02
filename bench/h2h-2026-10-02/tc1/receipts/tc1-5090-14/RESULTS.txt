# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.38.0 @079a4220b4b334903b0e77725eb94a0f58848018 (GitHub main)
gnf4 0.34.0 @846b512b905468c08f5748943d08769b572affa2 (GitHub main)
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
e4b(t212) 0.38.0 @079a4220b4b334903b0e77725eb94a0f58848018
gnf4(t212) 0.34.0 @846b512b905468c08f5748943d08769b572affa2
torch(e4b-t212) 2.12.1+cu130
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1-5090-14",
 "instance_id": "53773863",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz",
 "nproc": 40,
 "mem_total_kb": "263751300",
 "cgroup_memory_max": "183336173568",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "ec7b8f70d5d8",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md"
}
```

### Qwen3-30B-A3B (the labelled / native-best box) (`qwen3native`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 9.336 | 162.3 | 27.849 | 1255.0 | 2.0705→0.8339 | 1.9505→0.8481 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_best | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 15.245 | 91.6 | 24.835 | 1199.8 | 2.0705→0.8320 | 1.9558→0.8467 | -0.0014 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0093) | 20 | 61.295 | 23.9 | 24.728 | 3885.0 | 2.0428→0.8296 | 1.9598→0.8471 | -0.0010 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_triton | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0054) | 20 | 14.754 | 99.9 | 24.269 | 1214.3 | 2.0705→0.8334 | 1.9558→0.8479 | -0.0002 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) | requested backend unsloth_triton: unsloth_triton engaged 0 < 48*accum 4 per step (moe_backend_selected grouped_mm) |
| e4b | fused_attn4_shipped | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 6.267 | 244.0 | 24.581 | 843.0 | 2.0705→0.7985 | 1.9505→0.8108 | -0.0373 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_nodgrad | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 9.248 | 164.7 | 27.849 | 1199.5 | 2.0705→0.8317 | 1.9505→0.8512 | 0.0032 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t212 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0006) | 20 | 8.069 | 186.9 | 27.899 | 1119.3 | 2.0685→0.8330 | 1.9498→0.8506 | 0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_best | **INSTALL_FAILED** | — | **UNSUPPORTED** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | axolotl venv install failed rc=1 (logs/pip_axolotl.log):          And because you require axolotl==0.20.0, we can conclude that your requirements are unsatisfiable.  hint: `packaging` was found on https://download.pytorch.org/whl/cu130, but not at the requeste |
| hf | hf_peft_m_mb1_t214 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | runs only when the judged family's hf/hf_peft_m on this box OOMed; its status here is 'missing' |
- prologue `e4b/fused_attn4_m` **194.6 s** before step 1 (50% of the arm): c1_before 112.5, c1_after 56.6, load_weights 34.9, eval0 20.4; unattributed 1.636; budget 1260.0
- prologue `unsloth/ckpt_unsloth_best` **179.7 s** before step 1 (35% of the arm): c1_before 102.2, c1_after 54.3, load_weights 28.8, eval0 24.3; unattributed 11.815; budget 1260.0
- prologue `unsloth/ckpt_unsloth_t28` **244.8 s** before step 1 (16% of the arm): c1_before 108.9, eval0 77.9, c1_after 52.7, load_weights 29.5; unattributed 12.789; budget 1260.0
- prologue `unsloth/ckpt_unsloth_triton` **179.6 s** before step 1 (37% of the arm): c1_before 107.6, c1_after 53.2, load_weights 29.9, eval0 14.8; unattributed 11.907; budget 1260.0
- prologue `e4b/fused_attn4_shipped` **173.5 s** before step 1 (58% of the arm): c1_before 118.7, c1_after 57.2, load_weights 26.0, attn4 7.2; unattributed 1.64; budget 1260.0
- prologue `e4b/fused_attn4_m_nodgrad` **171.5 s** before step 1 (48% of the arm): c1_before 113.7, c1_after 58.0, load_weights 25.3, attn4 7.1; unattributed 1.734; budget 1260.0
- prologue `e4b/fused_attn4_m_t212` **186.2 s** before step 1 (53% of the arm): c1_before 114.1, c1_after 58.7, load_weights 23.1, eval0 21.8; unattributed 2.459; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_best` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_t28` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_shipped` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_nodgrad` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_t212` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b single draw (no second draw registered) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b single draw (no second draw registered) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b single draw (no second draw registered) / axolotl no receipt
- **NATIVE-BEST (reported beside, never instead of, the matched position): s/step ratio unsloth native-best vs e4b shipped/e4b = 2.433** [2.433, 2.433 over 1 cross-draw ratios] (15.245 vs 6.267 s, medians over 1/1 draws; e4b faster per step); peak VRAM unsloth native-best vs e4b shipped 24.84 vs e4b 24.58 GB (Δ +0.25); J/step unsloth native-best vs e4b shipped 1199.8 vs e4b 843.0 (×1.423); tok/s unsloth native-best vs e4b shipped 91.6 vs e4b 244.0; unsloth native-best vs e4b shipped regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth native-best vs e4b shipped): held-out e4b 0.8108 / unsloth native-best vs e4b shipped 0.8467 (Δ 0.0359) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0054
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b single draw (no second draw registered) / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is UNSUPPORTED
- **LABELLED ROW ckpt_unsloth_t28 vs e4b/fused_attn4_m (never the quoted position): s/step ratio unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend)/e4b = 6.565** [6.565, 6.565 over 1 cross-draw ratios] (61.295 vs 9.336 s, medians over 1/1 draws; e4b faster per step); peak VRAM unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend) 24.73 vs e4b 27.85 GB (Δ -3.12); J/step unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend) 3885.0 vs e4b 1255.0 (×3.096); tok/s unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend) 23.9 vs e4b 162.3; unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend) regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend)): held-out e4b 0.8481 / unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend) 0.8471 (Δ -0.0010) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0093
- **NO LABELLED ROW ckpt_unsloth_triton vs e4b/fused_attn4_m (never the quoted position) QUOTED (unsloth triton backend)** — not quoted: e4b single draw (no second draw registered) / unsloth triton backend unsloth/ckpt_unsloth_triton is VOID
- **LABELLED ROW ckpt_unsloth_best vs e4b/fused_attn4_m (never the quoted position): s/step ratio unsloth native-best (grouped_mm + speed tilt, native init)/e4b = 1.633** [1.633, 1.633 over 1 cross-draw ratios] (15.245 vs 9.336 s, medians over 1/1 draws; e4b faster per step); peak VRAM unsloth native-best (grouped_mm + speed tilt, native init) 24.84 vs e4b 27.85 GB (Δ -3.01); J/step unsloth native-best (grouped_mm + speed tilt, native init) 1199.8 vs e4b 1255.0 (×0.956); tok/s unsloth native-best (grouped_mm + speed tilt, native init) 91.6 vs e4b 162.3; unsloth native-best (grouped_mm + speed tilt, native init) regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth native-best (grouped_mm + speed tilt, native init)): held-out e4b 0.8481 / unsloth native-best (grouped_mm + speed tilt, native init) 0.8467 (Δ -0.0014) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0054
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b single draw (no second draw registered) / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is UNSUPPORTED
- **LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position): s/step ratio e4b shipped (bf16 expert adapters, N(0,1/r) init)/e4b = 0.671** [0.671, 0.671 over 1 cross-draw ratios] (6.267 vs 9.336 s, medians over 1/1 draws; e4b shipped (bf16 expert adapters, N(0,1/r) init) faster per step); peak VRAM e4b shipped (bf16 expert adapters, N(0,1/r) init) 24.58 vs e4b 27.85 GB (Δ -3.27); J/step e4b shipped (bf16 expert adapters, N(0,1/r) init) 843.0 vs e4b 1255.0 (×0.672); tok/s e4b shipped (bf16 expert adapters, N(0,1/r) init) 244.0 vs e4b 162.3; e4b shipped (bf16 expert adapters, N(0,1/r) init) regime: 4-bit experts (e4b NF4) + NF4 attention
- quality reading at N=20 (e4b shipped (bf16 expert adapters, N(0,1/r) init)): held-out e4b 0.8481 / e4b shipped (bf16 expert adapters, N(0,1/r) init) 0.8108 (Δ -0.0373) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0000
- **LABELLED ROW fused_attn4_m_nodgrad vs e4b/fused_attn4_m (never the quoted position): s/step ratio e4b fused dgrad=False (enable_fast_train's default)/e4b = 0.991** [0.991, 0.991 over 1 cross-draw ratios] (9.248 vs 9.336 s, medians over 1/1 draws; e4b fused dgrad=False (enable_fast_train's default) faster per step); peak VRAM e4b fused dgrad=False (enable_fast_train's default) 27.85 vs e4b 27.85 GB (Δ +0.00); J/step e4b fused dgrad=False (enable_fast_train's default) 1199.5 vs e4b 1255.0 (×0.956); tok/s e4b fused dgrad=False (enable_fast_train's default) 164.7 vs e4b 162.3; e4b fused dgrad=False (enable_fast_train's default) regime: 4-bit experts (e4b NF4) + NF4 attention
- quality reading at N=20 (e4b fused dgrad=False (enable_fast_train's default)): held-out e4b 0.8481 / e4b fused dgrad=False (enable_fast_train's default) 0.8512 (Δ 0.0032) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0000
- **LABELLED ROW fused_attn4_m_t212 vs e4b/fused_attn4_m (never the quoted position): s/step ratio e4b fused on torch 2.12.1+cu130 (venv-unsloth)/e4b = 0.864** [0.864, 0.864 over 1 cross-draw ratios] (8.069 vs 9.336 s, medians over 1/1 draws; e4b fused on torch 2.12.1+cu130 (venv-unsloth) faster per step); peak VRAM e4b fused on torch 2.12.1+cu130 (venv-unsloth) 27.90 vs e4b 27.85 GB (Δ +0.05); J/step e4b fused on torch 2.12.1+cu130 (venv-unsloth) 1119.3 vs e4b 1255.0 (×0.892); tok/s e4b fused on torch 2.12.1+cu130 (venv-unsloth) 186.9 vs e4b 162.3; e4b fused on torch 2.12.1+cu130 (venv-unsloth) regime: 4-bit experts (e4b NF4) + NF4 attention
- quality reading at N=20 (e4b fused on torch 2.12.1+cu130 (venv-unsloth)): held-out e4b 0.8481 / e4b fused on torch 2.12.1+cu130 (venv-unsloth) 0.8506 (Δ 0.0025) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0006
- **NO LABELLED ROW hf_peft_m_mb1_t214 vs e4b/fused_attn4_m (never the quoted position) QUOTED (hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm)** — not quoted: e4b single draw (no second draw registered) / hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm hf/hf_peft_m_mb1_t214 is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `unsloth/ckpt_unsloth_t28` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0010, step-0 0.0093 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0004, paired rows mean -0.0010 ± 0.0023 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_triton` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_nodgrad` **COMPARABLE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0032, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0012, paired rows mean +0.0032 ± 0.0031 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t212` **COMPARABLE** (median step |Δ| 0.0021, |Δ held-out at N| 0.0025, step-0 0.0006 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0040, paired rows mean +0.0025 ± 0.0025 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `hf/hf_peft_m_mb1_t214` **—** — no OK receipt
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_t28` same; `unsloth/ckpt_unsloth_triton` same; `e4b/fused_attn4_m_nodgrad` same; `e4b/fused_attn4_m_t212` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_best`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_t28`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_triton`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_nodgrad`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t212`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3native | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3native | unsloth/ckpt_unsloth_best | **VALID** | VALID | -0.0014 |  |
| qwen3native | unsloth/ckpt_unsloth_t28 | **VALID** | VALID | -0.0010 |  |
| qwen3native | unsloth/ckpt_unsloth_triton | **VOID** | VOID | -0.0002 | requested backend unsloth_triton: unsloth_triton engaged 0 < 48*accum 4 per step (moe_backend_selected grouped_mm) |
| qwen3native | e4b/fused_attn4_shipped | **VALID** | VALID | -0.0373 |  |
| qwen3native | e4b/fused_attn4_m_nodgrad | **VALID** | VALID | 0.0032 |  |
| qwen3native | e4b/fused_attn4_m_t212 | **VALID** | VALID | 0.0025 |  |
| qwen3native | axolotl/ckpt_axolotl_best | **UNSUPPORTED** | — | — | axolotl venv install failed rc=1 (logs/pip_axolotl.log):          And because you require axolotl==0.20.0, we can conclude that your requirements are unsatisfia |
| qwen3native | hf/hf_peft_m_mb1_t214 | **NOT_RUN** | — | — | runs only when the judged family's hf/hf_peft_m on this box OOMed; its status here is 'missing' |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **UNTESTED** | no receipts |
| P2 | qwen3 | **UNTESTED** | no receipts |
| P3 | qwen3 | **UNTESTED** | no receipts |
| P4 | qwen3 | **UNTESTED** | no receipts |
| P5 | qwen3 | **UNTESTED** | no receipts |
| P6 | qwen3 | **UNTESTED** | no receipts |
| P7 | qwen3 | **UNTESTED** | no receipts |
| P8 | qwen3 | **UNTESTED** | no receipts |
| P9 | qwen3 | **UNTESTED** | no receipts |
| P10 | qwen3 | **UNTESTED** | no receipts |
