# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.38.1 @09b0f6a6586d755dccad3550f68ae1b2e57773ad (GitHub main)
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
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1-5090-20",
 "instance_id": "53812706",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 112,
 "mem_total_kb": "263686464",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G  1.8M  320G   1% /",
 "hostname": "d52acd8b45e0",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md"
}
```

### Qwen3-30B-A3B (amendment 3: the axolotl box) (`qwen3axolotl`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.460 | 273.5 | 27.849 | 1175.8 | 2.0705→0.8332 | 1.9505→0.8508 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_m | **HARNESS_ERROR** | — | **HARNESS_ERROR** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.916 | 263.8 | 27.795 | 1183.8 | 2.0705→0.8342 | 1.9505→0.8517 | 0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| axolotl | ckpt_axolotl_m_d2 | **HARNESS_ERROR** | — | **HARNESS_ERROR** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| axolotl | ckpt_axolotl_best | **REFUSED** | — | **UNSUPPORTED** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | ValueError: Version 2 of 'kernels-community/rotary' is not available in the local cache and Hugging Face Hub is in offline mode. Download the kernel while online first, or pass an explicit `revision=<commit>`. |
| hf | hf_peft_m_mb1_t214 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 3.75 MiB is free. Including non-PyTorch memory, this process has 31.35 GiB memory in use. Of the allocated memory 30.74 GiB is allocated by PyTo |
- prologue `e4b/fused_attn4_m` **122.1 s** before step 1 (52% of the arm): c1_before 66.5, c1_after 33.6, load_weights 22.0, eval0 14.9; unattributed 1.354; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **109.0 s** before step 1 (48% of the arm): c1_before 66.9, c1_after 33.5, load_weights 21.5, attn4 6.3; unattributed 1.248; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (5.460/5.916 s, |Δ|/mean 8.0% vs 5%); `axolotl/ckpt_axolotl_m` — (axolotl/ckpt_axolotl_m is HARNESS_ERROR)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 5.460 / 5.916 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 5.460 / 5.916 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 5.460 / 5.916 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted) / axolotl axolotl/ckpt_axolotl_m is HARNESS_ERROR
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b no receipt / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is UNSUPPORTED
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b draws 5.460 / 5.916 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted) / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is UNSUPPORTED
- **NO LABELLED ROW hf_peft_m_mb1_t214 vs e4b/fused_attn4_m (never the quoted position) QUOTED (hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm)** — not quoted: e4b draws 5.460 / 5.916 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted) / hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm hf/hf_peft_m_mb1_t214 is OOM
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0009, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0070, paired rows mean +0.0009 ± 0.0019 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `hf/hf_peft_m_mb1_t214` **—** — no OK receipt
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `e4b/fused_attn4_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3axolotl | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_m | **HARNESS_ERROR** | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| qwen3axolotl | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0009 |  |
| qwen3axolotl | axolotl/ckpt_axolotl_m_d2 | **HARNESS_ERROR** | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| qwen3axolotl | axolotl/ckpt_axolotl_best | **UNSUPPORTED** | — | — | ValueError: Version 2 of 'kernels-community/rotary' is not available in the local cache and Hugging Face Hub is in offline mode. Download the kernel while onlin |
| qwen3axolotl | hf/hf_peft_m_mb1_t214 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 3.75 MiB is free. Including non-PyTorch memo |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **UNTESTED** | no receipts |
| P2 | qwen3 | **UNTESTED** | no receipts |
| P3 | qwen3 | **UNTESTED** | no receipts |
| P4 | qwen3 | **UNTESTED** | no receipts |
| P5 | qwen3 | **UNTESTED** | no receipts |
| P6 | qwen3axolotl | **UNTESTED** | axolotl HARNESS_ERROR (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | no receipts |
| P8 | qwen3 | **UNTESTED** | no receipts |
| P9 | qwen3 | **UNTESTED** | no receipts |
| P10 | qwen3 | **UNTESTED** | no receipts |
