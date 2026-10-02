# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.40.0 @bb32ea83f99985494d8731318eec0f8a261398eb (GitHub main)
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
e4b(t212) 0.40.0 @bb32ea83f99985494d8731318eec0f8a261398eb
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
 "run_id": "tc1-5090-33",
 "instance_id": "53916526",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "c0bbcf5ef381",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### qwen3nativebest (`qwen3nativebest`, registered n_layers ?)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.320 | 284.7 | 27.822 | 1251.0 | 2.0705→0.8322 | 1.9505→0.8498 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | no registered n_layers for family qwen3nativebest |
| unsloth | ckpt_unsloth_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | no receipt and no attempt line |
| e4b | fused_attn4_m_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | no receipt and no attempt line |
| unsloth | ckpt_unsloth_m_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | no receipt and no attempt line |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | no receipt and no attempt line |
| e4b | fused_attn4_m_prof | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | no receipt and no attempt line |
| unsloth | ckpt_unsloth_prof | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| axolotl | ckpt_axolotl_best | **OK** | VOID | **VOID** | native | native / float32 | — | 20 | 5.128 | 52.4 | 25.839 | 3330.8 | 2.0621→0.8262 | 1.9462→0.8330 | N-A (anchor VOID) | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) | no registered n_layers for family qwen3nativebest |
| axolotl | ckpt_axolotl_best_d2 | **REFUSED** | — | **UNSUPPORTED** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | ValueError: Version 2 of 'kernels-community/rotary' is not available in the local cache and Hugging Face Hub is in offline mode. Download the kernel while online first, or pass an explicit `revision=<commit>`. |
| e4b | fused_attn4_shipped | **OK** | VOID | **VOID** | native | native / bfloat16,float32 | — | 20 | 4.306 | 350.7 | 24.581 | 1011.9 | 2.0705→0.7990 | 1.9505→0.8140 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | no registered n_layers for family qwen3nativebest |
| e4b | fused_attn4_shipped_d2 | **OK** | VOID | **VOID** | native | native / bfloat16,float32 | — | 20 | 4.578 | 338.7 | 24.581 | 996.8 | 2.0705→0.7997 | 1.9505→0.8135 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | no registered n_layers for family qwen3nativebest |
| unsloth | ckpt_unsloth_best | **OK** | VOID | **VOID** | native | native / float32 | — | 20 | 7.938 | 173.3 | 24.835 | 1188.2 | 2.0705→0.8332 | 1.9583→0.8425 | N-A (anchor VOID) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) | no registered n_layers for family qwen3nativebest |
| unsloth | ckpt_unsloth_best_d2 | **OK** | VOID | **VOID** | native | native / float32 | — | 20 | 7.897 | 190.0 | 24.835 | 1180.8 | 2.0705→0.8313 | 1.9583→0.8471 | N-A (anchor VOID) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) | no registered n_layers for family qwen3nativebest |
- prologue `e4b/fused_attn4_m` **116.1 s** before step 1 (52% of the arm): c1_before 67.4, c1_after 31.1, load_weights 26.9, attn4 5.8; unattributed 1.296; budget 1260.0
- prologue `axolotl/ckpt_axolotl_best` **296.4 s** before step 1 (33% of the arm): eval0 178.3, c1_before 54.9, load_weights 49.3, c1_after 30.9; unattributed 6.285; budget 945.0
- prologue `e4b/fused_attn4_shipped` **123.8 s** before step 1 (58% of the arm): c1_before 60.0, c1_after 30.4, load_weights 29.5, eval0 14.6; unattributed 1.257; budget 1260.0
- prologue `e4b/fused_attn4_shipped_d2` **117.2 s** before step 1 (56% of the arm): c1_before 65.8, c1_after 33.8, load_weights 28.7, attn4 6.3; unattributed 1.283; budget 1260.0
- prologue `unsloth/ckpt_unsloth_best` **113.9 s** before step 1 (39% of the arm): c1_before 51.6, load_weights 30.1, c1_after 26.4, eval0 15.5; unattributed 10.251; budget 1260.0
- prologue `unsloth/ckpt_unsloth_best_d2` **105.8 s** before step 1 (39% of the arm): c1_before 52.1, load_weights 28.6, c1_after 24.7, eval0 9.2; unattributed 10.182; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is VOID); `unsloth/ckpt_unsloth_m` — (unsloth/ckpt_unsloth_m is NOT_RUN)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m is VOID / unsloth unsloth/ckpt_unsloth_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m is VOID / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m is VOID / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is VOID / unsloth native-best vs e4b shipped unsloth/ckpt_unsloth_best is VOID
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is VOID / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is VOID
- **NO LABELLED ROW ckpt_unsloth_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (unsloth native-best (grouped_mm + speed tilt, native init))** — not quoted: e4b e4b/fused_attn4_m is VOID / unsloth native-best (grouped_mm + speed tilt, native init) unsloth/ckpt_unsloth_best is VOID
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b e4b/fused_attn4_m is VOID / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is VOID
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b e4b/fused_attn4_m is VOID / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is VOID
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `axolotl/ckpt_axolotl_best`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `e4b/fused_attn4_shipped`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_best`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_best_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3nativebest | e4b/fused_attn4_m | **VOID** | VOID | N-A (anchor VOID) | no registered n_layers for family qwen3nativebest |
| qwen3nativebest | unsloth/ckpt_unsloth_m | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | e4b/reference_attn4_m | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | e4b/fused_attn4_m_d2 | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | unsloth/ckpt_unsloth_m_d2 | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | hf/hf_peft_m | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | e4b/fused_attn4_m_prof | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | unsloth/ckpt_unsloth_prof | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3nativebest | axolotl/ckpt_axolotl_best | **VOID** | VOID | N-A (anchor VOID) | no registered n_layers for family qwen3nativebest |
| qwen3nativebest | axolotl/ckpt_axolotl_best_d2 | **UNSUPPORTED** | — | — | ValueError: Version 2 of 'kernels-community/rotary' is not available in the local cache and Hugging Face Hub is in offline mode. Download the kernel while onlin |
| qwen3nativebest | e4b/fused_attn4_shipped | **VOID** | VOID | N-A (anchor VOID) | no registered n_layers for family qwen3nativebest |
| qwen3nativebest | e4b/fused_attn4_shipped_d2 | **VOID** | VOID | N-A (anchor VOID) | no registered n_layers for family qwen3nativebest |
| qwen3nativebest | unsloth/ckpt_unsloth_best | **VOID** | VOID | N-A (anchor VOID) | no registered n_layers for family qwen3nativebest |
| qwen3nativebest | unsloth/ckpt_unsloth_best_d2 | **VOID** | VOID | N-A (anchor VOID) | no registered n_layers for family qwen3nativebest |
