# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.42.0 @14a9a17d090e89a0a9ea546438013ea66d88c962 (GitHub main)
gnf4 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6 (GitHub main)
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
 "run_id": "tc1-5090-50",
 "instance_id": "54075633",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "610.57.04",
 "cpu": "AMD Ryzen 9 9950X3D 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "129218432",
 "cgroup_memory_max": "127025545216",
 "disk_root": "overlay         320G  9.3M  320G   1% /",
 "hostname": "43bf9be57d0f",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 3: the axolotl box) (`qwen3axolotl`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.129 | 641.1 | 27.230 | 929.5 | 2.0614→0.8313 | 1.9441→0.8511 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0323) | 20 | 5.993 | 258.7 | 26.882 | 2635.6 | 2.0554→0.8359 | 1.9764→0.8587 | 0.0076 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.164 | 708.4 | 27.210 | 829.1 | 2.0614→0.8339 | 1.9441→0.8535 | 0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0323) | 20 | 5.920 | 259.2 | 26.882 | 2592.2 | 2.0554→0.8340 | 1.9764→0.8541 | 0.0031 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| axolotl | ckpt_axolotl_best | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 3.363 | 93.5 | 25.841 | 2650.7 | 2.0621→0.8235 | 1.9462→0.8330 | -0.0181 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| hf | hf_peft_m_mb1_t214 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.39 GiB of which 6.12 MiB is free. Including non-PyTorch memory, this process has 31.37 GiB memory in use. Of the allocated memory 30.76 GiB is allocated by PyTo |
- prologue `e4b/fused_attn4_m` **60.6 s** before step 1 (56% of the arm): c1_before 24.2, load_weights 22.0, c1_after 11.9, eval0 5.2; unattributed 0.636; budget 1260.0
- prologue `axolotl/ckpt_axolotl_m` **47.4 s** before step 1 (28% of the arm): c1_before 22.3, load_weights 11.8, c1_after 11.2, eval0 5.4; unattributed 2.873; budget 945.0
- prologue `e4b/fused_attn4_m_d2` **43.9 s** before step 1 (50% of the arm): c1_before 24.5, c1_after 12.0, load_weights 9.4, preamble 2.9; unattributed 0.673; budget 1260.0
- prologue `axolotl/ckpt_axolotl_m_d2` **46.3 s** before step 1 (28% of the arm): c1_before 22.4, load_weights 11.3, c1_after 11.3, eval0 4.9; unattributed 2.664; budget 945.0
- prologue `axolotl/ckpt_axolotl_best` **138.7 s** before step 1 (30% of the arm): eval0 95.6, c1_before 22.4, load_weights 13.6, c1_after 11.1; unattributed 2.626; budget 945.0
- draws (R1): `e4b/fused_attn4_m` STABLE (2.129/2.164 s, |Δ|/mean 1.6% vs 5%); `axolotl/ckpt_axolotl_m` STABLE (5.993/5.920 s, |Δ|/mean 1.2% vs 5%); `axolotl/ckpt_axolotl_best` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **MATCHED POSITION: s/step ratio axolotl/e4b = 2.775** [2.735, 2.814 over 4 cross-draw ratios] (5.956 vs 2.147 s, medians over 2/2 draws; e4b faster per step); peak VRAM axolotl 26.88 vs e4b 27.22 GB (Δ -0.34); J/step axolotl 2613.9 vs e4b 879.3 (×2.973); tok/s axolotl 258.9 vs e4b 674.8; axolotl regime: 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384)
- quality reading at N=20 (axolotl): held-out e4b 0.8511 / axolotl 0.8587 (Δ 0.0076) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0323
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b no receipt / axolotl native-best vs e4b shipped single draw (no second draw registered)
- **LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position): s/step ratio axolotl native-best (KernelsPlugin scattermoe) / e4b = 1.567** [1.554, 1.579 over 2 cross-draw ratios] (3.363 vs 2.147 s, medians over 1/2 draws; e4b faster per step); peak VRAM axolotl native-best (KernelsPlugin scattermoe) 25.84 vs e4b 27.22 GB (Δ -1.38); J/step axolotl native-best (KernelsPlugin scattermoe) 2650.7 vs e4b 879.3 (×3.015); tok/s axolotl native-best (KernelsPlugin scattermoe) 93.5 vs e4b 674.8; axolotl native-best (KernelsPlugin scattermoe) regime: 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384)
- quality reading at N=20 (axolotl native-best (KernelsPlugin scattermoe)): held-out e4b 0.8511 / axolotl native-best (KernelsPlugin scattermoe) 0.8330 (Δ -0.0181) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0021
- **NO LABELLED ROW hf_peft_m_mb1_t214 vs e4b/fused_attn4_m (never the quoted position) QUOTED (hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm)** — not quoted: e4b STABLE / hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm hf/hf_peft_m_mb1_t214 is OOM
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0045): `axolotl/ckpt_axolotl_m` **COMPARABLE** (median step |Δ| 0.0090, |Δ held-out at N| 0.0076, step-0 0.0323 NEAR, |Δ loss at step 2| 0.0130, paired rows mean +0.0076 ± 0.0029 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0024, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0139, paired rows mean +0.0024 ± 0.0018 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `axolotl/ckpt_axolotl_m_d2` **COMPARABLE** (median step |Δ| 0.0100, |Δ held-out at N| 0.0031, step-0 0.0323 NEAR, |Δ loss at step 2| 0.0110, paired rows mean +0.0031 ± 0.0010 SE over 8, favouring arm 0 / anchor 8) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `hf/hf_peft_m_mb1_t214` **—** — no OK receipt
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `axolotl/ckpt_axolotl_m` same; `e4b/fused_attn4_m_d2` same; `axolotl/ckpt_axolotl_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `axolotl/ckpt_axolotl_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `axolotl/ckpt_axolotl_best`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3axolotl | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_m | **VALID** | VALID | 0.0076 |  |
| qwen3axolotl | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0024 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_m_d2 | **VALID** | VALID | 0.0031 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_best | **VALID** | VALID | -0.0181 |  |
| qwen3axolotl | hf/hf_peft_m_mb1_t214 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.39 GiB of which 6.12 MiB is free. Including non-PyTorch memo |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **UNTESTED** | no receipts |
| P2 | qwen3 | **UNTESTED** | no receipts |
| P3 | qwen3 | **UNTESTED** | no receipts |
| P4 | qwen3 | **UNTESTED** | no receipts |
| P5 | qwen3 | **UNTESTED** | no receipts |
| P6 | qwen3axolotl | **HELD** | axolotl trained; axolotl/e4b 2.775 vs [1.5, 6] |
| P7 | qwen3 | **UNTESTED** | no receipts |
| P8 | qwen3 | **UNTESTED** | no receipts |
| P9 | qwen3 | **UNTESTED** | no receipts |
| P10 | qwen3 | **UNTESTED** | no receipts |
