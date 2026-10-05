# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @c12111d1f3062d0df0d6e13674a28c9e5bfe63ab (GitHub main)
gnf4 0.41.0 @0d18e13489034d5748c491d38855e76c73e1dbe1 (GitHub main)
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
e4b(t212) 0.48.0 @c12111d1f3062d0df0d6e13674a28c9e5bfe63ab
gnf4(t212) 0.41.0 @0d18e13489034d5748c491d38855e76c73e1dbe1
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
 "run_id": "tc1-5090-86",
 "instance_id": "54297512",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "76499fe21b67",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 39: the packed 4,096-token regime with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0) (`qwen3samestack4k`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=30; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable None; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 30 | — | — | 30.317 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.85 GiB is free. Including non-PyTorch memory, this process has 29.50 GiB memory in use. Of the alloca |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | — | 30 | 15.164 | 1073.6 | 24.864 | 4797.4 | 1.2587→0.9929 | 1.2871→0.9672 | N-A (anchor missing) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 30 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 30 | — | — | 30.267 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.01 GiB is free. Including non-PyTorch memory, this process has 29.34 GiB memory in use. Of the alloca |
| e4b | fused_attn4_m_t28_d2 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 30 | — | — | 30.267 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.01 GiB is free. Including non-PyTorch memory, this process has 29.34 GiB memory in use. Of the alloca |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | — | 30 | 14.040 | 1153.6 | 24.864 | 4543.6 | 1.2587→0.9933 | 1.2871→0.9674 | N-A (anchor missing) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 30 | — | — | 30.317 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.85 GiB is free. Including non-PyTorch memory, this process has 29.50 GiB memory in use. Of the alloca |
- prologue `unsloth/ckpt_unsloth_m` **128.5 s** before step 1 (22% of the arm): c1_before 61.1, load_weights 33.3, c1_after 27.9, eval0 14.4; unattributed 10.963; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_d2` **109.4 s** before step 1 (20% of the arm): c1_before 47.9, load_weights 31.2, c1_after 24.1, eval0 12.7; unattributed 10.086; budget 1890.0
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is OOM); `unsloth/ckpt_unsloth_m` UNSTABLE (15.164/14.040 s, |Δ|/mean 7.7% vs 5%); `e4b/fused_attn4_m_t28` — (e4b/fused_attn4_m_t28 is OOM)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m is OOM / unsloth draws 15.164 / 14.040 s/step differ by 7.7% > 5% (UNSTABLE: reported, not quoted)
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m is OOM / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m is OOM / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `unsloth/ckpt_unsloth_m` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **—** — no OK receipt; `e4b/fused_attn4_m_t28_d2` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_d2` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `e4b/fused_attn4_m_d2` **—** — no OK receipt
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3samestack4k | e4b/fused_attn4_m | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.85 GiB is free. Including non-PyTorch memory,  |
| qwen3samestack4k | unsloth/ckpt_unsloth_m | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3samestack4k | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestack4k | e4b/fused_attn4_m_t28 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.01 GiB is free. Including non-PyTorch memory,  |
| qwen3samestack4k | e4b/fused_attn4_m_t28_d2 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.01 GiB is free. Including non-PyTorch memory,  |
| qwen3samestack4k | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3samestack4k | e4b/fused_attn4_m_d2 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.85 GiB is free. Including non-PyTorch memory,  |

## Predictions P84 / P85 / P86 (TC1-PREREG amendment 39: the packed 4,096-token regime with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P84 | qwen3samestack4k | **UNTESTED** | two stable VALID draws a side are registered -- not quoted: e4b e4b/fused_attn4_m is OOM / unsloth draws 15.164 / 14.040 s/step differ by 7.7% > 5% (UNSTABLE: reported, not quoted) |
| P85 | qwen3samestack4k | **UNTESTED** | two stable VALID draws a side are registered -- venv-unsloth —: e4b/fused_attn4_m is OOM; venv-e4b —: e4b/fused_attn4_m_t28 is OOM |
| P86 | qwen3samestack4k | **FALSIFIED** | e4b OOM on fused_attn4_m (OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total), fused_attn4_m_t28 (OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total), fused_attn4_m_t28_d2 (OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total), fused_attn4_m_d2 (OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total); 4 e4b arm(s) ran: fused_attn4_m OOM peak 30.32 GB; fused_attn4_m_t28 OOM peak 30.27 GB; fused_attn4_m_t28_d2 OOM peak 30.27 GB; fused_attn4_m_d2 OOM peak 30.32 GB |

## Load gate (TC1-PREREG amendment 33): 2 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestack4k/e4b/fused_attn4_m attempt 0 load1_median 9.32 gate 6.0 status oom over 1`
- `qwen3samestack4k/unsloth/ckpt_unsloth_m attempt 0 load1_median 6.81 gate 6.0 status ok over 1`
- `qwen3samestack4k/unsloth/ckpt_unsloth_m attempt 0 VOID (host load1 median 6.81 > 6.0): re-run 1 of 2`
- `qwen3samestack4k/unsloth/ckpt_unsloth_m attempt 1 load1_median 22.49 gate 6.0 status ok over 1`
- `qwen3samestack4k/unsloth/ckpt_unsloth_m attempt 1 VOID (host load1 median 22.49 > 6.0): re-run 2 of 2`
- `qwen3samestack4k/unsloth/ckpt_unsloth_m attempt 2 load1_median 16.19 gate 6.0 status ok over 1`
- `qwen3samestack4k/e4b/fused_attn4_m_t28 attempt 0 load1_median 16.95 gate 6.0 status oom over 1`
- `qwen3samestack4k/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 8.83 gate 6.0 status oom over 1`
- `qwen3samestack4k/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 4.61 gate 6.0 status ok over 0`
- `qwen3samestack4k/e4b/fused_attn4_m_d2 attempt 0 load1_median 8.73 gate 6.0 status oom over 1`
