# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.40.0 @3e41f0f2d1e6e20bcbf25711dde087deec97a563 (GitHub main)
gnf4 0.34.0 @846b512b905468c08f5748943d08769b572affa2 (GitHub main)
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
 "run_id": "tc1-5090-30",
 "instance_id": "53898131",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "a70019ba1776",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.408 | 275.8 | 27.797 | 1307.7 | 2.0705→0.8334 | 1.9505→0.8472 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0259) | 20 | 7.759 | 192.6 | 26.882 | 2947.8 | 2.0554→0.8362 | 1.9764→0.8567 | 0.0095 | peft mods 192 / params 96 / fwd 384 | 642514944 | bf16 experts (NOT the 4-bit MoE regime) + bnb-4bit attention (Params4bit stacks 0, Linear4bit 384); experts_implementation None, torch grouped_mm 768/step (reached) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.464 | 286.3 | 27.822 | 1232.6 | 2.0705→0.8297 | 1.9505→0.8523 | 0.0051 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0259) | 20 | 7.642 | 197.8 | 26.882 | 2924.7 | 2.0554→0.8364 | 1.9764→0.8575 | 0.0103 | peft mods 192 / params 96 / fwd 384 | 642514944 | bf16 experts (NOT the 4-bit MoE regime) + bnb-4bit attention (Params4bit stacks 0, Linear4bit 384); experts_implementation None, torch grouped_mm 768/step (reached) |  |
| axolotl | ckpt_axolotl_best | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 5.049 | 52.1 | 25.841 | 3266.6 | 2.0621→0.8236 | 1.9462→0.8280 | -0.0191 | peft mods 192 / params 96 / fwd 384 | 642514944 | bf16 experts (NOT the 4-bit MoE regime) + bnb-4bit attention (Params4bit stacks 0, Linear4bit 384); experts_implementation None, torch grouped_mm 0/step (NOT reached) |  |
| hf | hf_peft_m_mb1_t214 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 13.06 MiB is free. Including non-PyTorch memory, this process has 31.34 GiB memory in use. Of the allocated memory 30.76 GiB is allocated by PyT |
- prologue `e4b/fused_attn4_m` **134.2 s** before step 1 (54% of the arm): c1_before 67.8, c1_after 35.6, load_weights 32.4, eval0 14.9; unattributed 1.37; budget 1260.0
- prologue `axolotl/ckpt_axolotl_m` **122.8 s** before step 1 (43% of the arm): c1_before 58.8, load_weights 40.8, c1_after 31.8, eval0 7.2; unattributed 5.857; budget 945.0
- prologue `e4b/fused_attn4_m_d2` **125.4 s** before step 1 (54% of the arm): c1_before 74.8, c1_after 32.3, load_weights 28.1, attn4 6.7; unattributed 1.512; budget 1260.0
- prologue `axolotl/ckpt_axolotl_m_d2` **115.0 s** before step 1 (42% of the arm): c1_before 53.9, load_weights 40.9, c1_after 29.3, eval0 6.2; unattributed 5.54; budget 945.0
- prologue `axolotl/ckpt_axolotl_best` **292.4 s** before step 1 (33% of the arm): eval0 174.8, c1_before 55.3, load_weights 49.1, c1_after 30.7; unattributed 5.728; budget 945.0
- draws (R1): `e4b/fused_attn4_m` STABLE (5.408/5.464 s, |Δ|/mean 1.0% vs 5%); `axolotl/ckpt_axolotl_m` STABLE (7.759/7.642 s, |Δ|/mean 1.5% vs 5%); `axolotl/ckpt_axolotl_best` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **MATCHED POSITION: s/step ratio axolotl/e4b = 1.416** [1.398, 1.435 over 4 cross-draw ratios] (7.700 vs 5.436 s, medians over 2/2 draws; e4b faster per step); peak VRAM axolotl 26.88 vs e4b 27.81 GB (Δ -0.93); J/step axolotl 2936.3 vs e4b 1270.1 (×2.312); tok/s axolotl 195.2 vs e4b 281.1; axolotl regime: bf16 experts (NOT the 4-bit MoE regime) + bnb-4bit attention (Params4bit stacks 0, Linear4bit 384); experts_implementation None, torch grouped_mm 768/step (reached)
- quality reading at N=20 (axolotl): held-out e4b 0.8472 / axolotl 0.8567 (Δ 0.0095) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0259
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b no receipt / axolotl native-best vs e4b shipped single draw (no second draw registered)
- **LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position): s/step ratio axolotl native-best (KernelsPlugin scattermoe) / e4b = 0.929** [0.924, 0.934 over 2 cross-draw ratios] (5.049 vs 5.436 s, medians over 1/2 draws; axolotl native-best (KernelsPlugin scattermoe) faster per step); peak VRAM axolotl native-best (KernelsPlugin scattermoe) 25.84 vs e4b 27.81 GB (Δ -1.97); J/step axolotl native-best (KernelsPlugin scattermoe) 3266.6 vs e4b 1270.1 (×2.572); tok/s axolotl native-best (KernelsPlugin scattermoe) 52.1 vs e4b 281.1; axolotl native-best (KernelsPlugin scattermoe) regime: bf16 experts (NOT the 4-bit MoE regime) + bnb-4bit attention (Params4bit stacks 0, Linear4bit 384); experts_implementation None, torch grouped_mm 0/step (NOT reached)
- quality reading at N=20 (axolotl native-best (KernelsPlugin scattermoe)): held-out e4b 0.8472 / axolotl native-best (KernelsPlugin scattermoe) 0.8280 (Δ -0.0191) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0043
- **NO LABELLED ROW hf_peft_m_mb1_t214 vs e4b/fused_attn4_m (never the quoted position) QUOTED (hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm)** — not quoted: e4b STABLE / hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm hf/hf_peft_m_mb1_t214 is OOM
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0051): `axolotl/ckpt_axolotl_m` **COMPARABLE** (median step |Δ| 0.0126, |Δ held-out at N| 0.0095, step-0 0.0259 NEAR, |Δ loss at step 2| 0.0149, paired rows mean +0.0095 ± 0.0040 SE over 8, favouring arm 0 / anchor 8) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0019, |Δ held-out at N| 0.0051, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0039, paired rows mean +0.0051 ± 0.0033 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `axolotl/ckpt_axolotl_m_d2` **COMPARABLE** (median step |Δ| 0.0099, |Δ held-out at N| 0.0103, step-0 0.0259 NEAR, |Δ loss at step 2| 0.0098, paired rows mean +0.0104 ± 0.0037 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `hf/hf_peft_m_mb1_t214` **—** — no OK receipt
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
| qwen3axolotl | axolotl/ckpt_axolotl_m | **VALID** | VALID | 0.0095 |  |
| qwen3axolotl | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0051 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_m_d2 | **VALID** | VALID | 0.0103 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_best | **VALID** | VALID | -0.0191 |  |
| qwen3axolotl | hf/hf_peft_m_mb1_t214 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 13.06 MiB is free. Including non-PyTorch mem |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **UNTESTED** | no receipts |
| P2 | qwen3 | **UNTESTED** | no receipts |
| P3 | qwen3 | **UNTESTED** | no receipts |
| P4 | qwen3 | **UNTESTED** | no receipts |
| P5 | qwen3 | **UNTESTED** | no receipts |
| P6 | qwen3axolotl | **FALSIFIED** | axolotl trained; axolotl/e4b 1.416 vs [1.5, 6] |
| P7 | qwen3 | **UNTESTED** | no receipts |
| P8 | qwen3 | **UNTESTED** | no receipts |
| P9 | qwen3 | **UNTESTED** | no receipts |
| P10 | qwen3 | **UNTESTED** | no receipts |
