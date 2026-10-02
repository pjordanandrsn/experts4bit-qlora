# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.40.0 @2ad31872d1ced9218e5f701544fcb0785a17c46a (GitHub main)
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
e4b(t212) 0.40.0 @2ad31872d1ced9218e5f701544fcb0785a17c46a
gnf4(t212) 0.34.0 @846b512b905468c08f5748943d08769b572affa2
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
 "run_id": "tc1-5090-34",
 "instance_id": "53927504",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "590.48.01",
 "cpu": "AMD EPYC 9334 32-Core Processor",
 "nproc": 64,
 "mem_total_kb": "659622416",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   35M  320G   1% /",
 "hostname": "aab2b9184093",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 5: native-best against native-best, two draws each) (`qwen3nativebest`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.277 | 454.8 | 24.581 | 917.2 | 2.0705→0.7986 | 1.9505→0.8137 | -0.0342 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_best | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 4.574 | 63.5 | 25.839 | 2836.3 | 2.0621→0.8234 | 1.9462→0.8304 | -0.0175 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| unsloth | ckpt_unsloth_best | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 5.869 | 233.3 | 24.835 | 951.0 | 2.0533→0.8308 | 1.9558→0.8442 | -0.0037 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_shipped_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.279 | 467.6 | 24.581 | 869.4 | 2.0705→0.7987 | 1.9505→0.8153 | -0.0326 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_best_d2 | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 4.107 | 64.5 | 25.839 | 2652.0 | 2.0621→0.8249 | 1.9462→0.8331 | -0.0148 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| unsloth | ckpt_unsloth_best_d2 | **OK** | VALID | **VALID** | native | native / float32 | — | 20 | 5.889 | 249.5 | 24.835 | 928.6 | 2.0533→0.8311 | 1.9558→0.8466 | -0.0013 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.225 | 366.7 | 27.819 | 1116.5 | 2.0705→0.8332 | 1.9505→0.8479 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped` **89.8 s** before step 1 (57% of the arm): c1_before 50.4, c1_after 25.3, load_weights 15.7, eval0 11.1; unattributed 1.005; budget 1260.0
- prologue `axolotl/ckpt_axolotl_best` **219.4 s** before step 1 (31% of the arm): eval0 141.8, c1_before 45.9, c1_after 22.9, load_weights 21.7; unattributed 4.406; budget 945.0
- prologue `unsloth/ckpt_unsloth_best` **88.4 s** before step 1 (40% of the arm): c1_before 45.7, c1_after 23.0, load_weights 18.5, eval0 11.6; unattributed 7.783; budget 1260.0
- prologue `e4b/fused_attn4_shipped_d2` **80.5 s** before step 1 (55% of the arm): c1_before 51.2, c1_after 26.0, load_weights 15.4, attn4 4.6; unattributed 0.995; budget 1260.0
- prologue `axolotl/ckpt_axolotl_best_d2` **216.7 s** before step 1 (31% of the arm): eval0 141.9, c1_before 45.5, c1_after 22.8, load_weights 19.5; unattributed 4.333; budget 945.0
- prologue `unsloth/ckpt_unsloth_best_d2` **83.9 s** before step 1 (40% of the arm): c1_before 45.6, c1_after 22.9, load_weights 18.4, eval0 7.3; unattributed 7.779; budget 1260.0
- prologue `e4b/fused_attn4_m` **82.0 s** before step 1 (49% of the arm): c1_before 51.1, c1_after 25.2, load_weights 15.4, attn4 4.6; unattributed 0.96; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped` STABLE (3.277/3.279 s, |Δ|/mean 0.1% vs 5%); `axolotl/ckpt_axolotl_best` UNSTABLE (4.574/4.107 s, |Δ|/mean 10.8% vs 5%); `unsloth/ckpt_unsloth_best` STABLE (5.869/5.889 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b single draw (no second draw registered) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b single draw (no second draw registered) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b single draw (no second draw registered) / axolotl no receipt
- **NATIVE-BEST (reported beside, never instead of, the matched position): s/step ratio unsloth native-best vs e4b shipped/e4b = 1.794** [1.790, 1.797 over 4 cross-draw ratios] (5.879 vs 3.278 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth native-best vs e4b shipped 24.84 vs e4b 24.58 GB (Δ +0.25); J/step unsloth native-best vs e4b shipped 939.8 vs e4b 893.3 (×1.052); tok/s unsloth native-best vs e4b shipped 241.4 vs e4b 461.2; unsloth native-best vs e4b shipped regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth native-best vs e4b shipped): held-out e4b 0.8137 / unsloth native-best vs e4b shipped 0.8442 (Δ 0.0306) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0054
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b STABLE / axolotl native-best vs e4b shipped draws 4.574 / 4.107 s/step differ by 10.8% > 5% (UNSTABLE: reported, not quoted)
- **LABELLED ROW ckpt_unsloth_best vs e4b/fused_attn4_m (never the quoted position): s/step ratio unsloth native-best (grouped_mm + speed tilt, native init) / e4b = 1.391** [1.389, 1.394 over 2 cross-draw ratios] (5.879 vs 4.225 s, medians over 2/1 draws; e4b faster per step); peak VRAM unsloth native-best (grouped_mm + speed tilt, native init) 24.84 vs e4b 27.82 GB (Δ -2.98); J/step unsloth native-best (grouped_mm + speed tilt, native init) 939.8 vs e4b 1116.5 (×0.842); tok/s unsloth native-best (grouped_mm + speed tilt, native init) 241.4 vs e4b 366.7; unsloth native-best (grouped_mm + speed tilt, native init) regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth native-best (grouped_mm + speed tilt, native init)): held-out e4b 0.8479 / unsloth native-best (grouped_mm + speed tilt, native init) 0.8442 (Δ -0.0037) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0054
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b single draw (no second draw registered) / axolotl native-best (KernelsPlugin scattermoe) draws 4.574 / 4.107 s/step differ by 10.8% > 5% (UNSTABLE: reported, not quoted)
- **LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position): s/step ratio e4b shipped (bf16 expert adapters, N(0,1/r) init) / e4b = 0.776** [0.776, 0.776 over 2 cross-draw ratios] (3.278 vs 4.225 s, medians over 2/1 draws; e4b shipped (bf16 expert adapters, N(0,1/r) init) faster per step); peak VRAM e4b shipped (bf16 expert adapters, N(0,1/r) init) 24.58 vs e4b 27.82 GB (Δ -3.24); J/step e4b shipped (bf16 expert adapters, N(0,1/r) init) 893.3 vs e4b 1116.5 (×0.800); tok/s e4b shipped (bf16 expert adapters, N(0,1/r) init) 461.2 vs e4b 366.7; e4b shipped (bf16 expert adapters, N(0,1/r) init) regime: 4-bit experts (e4b NF4) + NF4 attention
- quality reading at N=20 (e4b shipped (bf16 expert adapters, N(0,1/r) init)): held-out e4b 0.8479 / e4b shipped (bf16 expert adapters, N(0,1/r) init) 0.8137 (Δ -0.0342) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0000
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_best`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `unsloth/ckpt_unsloth_best`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_best_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `unsloth/ckpt_unsloth_best_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3nativebest | e4b/fused_attn4_shipped | **VALID** | VALID | -0.0342 |  |
| qwen3nativebest | axolotl/ckpt_axolotl_best | **VALID** | VALID | -0.0175 |  |
| qwen3nativebest | unsloth/ckpt_unsloth_best | **VALID** | VALID | -0.0037 |  |
| qwen3nativebest | e4b/fused_attn4_shipped_d2 | **VALID** | VALID | -0.0326 |  |
| qwen3nativebest | axolotl/ckpt_axolotl_best_d2 | **VALID** | VALID | -0.0148 |  |
| qwen3nativebest | unsloth/ckpt_unsloth_best_d2 | **VALID** | VALID | -0.0013 |  |
| qwen3nativebest | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |

## Prediction P13 (TC1-PREREG amendment 5: native-best against native-best, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P13 (axolotl half) | qwen3nativebest | **UNTESTED** | not quoted: e4b STABLE / axolotl native-best vs e4b shipped draws 4.574 / 4.107 s/step differ by 10.8% > 5% (UNSTABLE: reported, not quoted) |
| P13 (Unsloth half) | qwen3nativebest | **HELD** | Unsloth native-best / e4b shipped 1.794 [1.790, 1.797 over 4 cross-draw ratios] (> 1.0 predicted); s/step e4b 3.278 (draws within 0.1%) vs Unsloth 5.879 (within 0.3%); held-out at N e4b shipped 0.8137 / Unsloth 0.8442 (reported, not judged: each framework's own init) |
| P13 | qwen3nativebest | **UNTESTED** | axolotl half UNTESTED; Unsloth half HELD |
