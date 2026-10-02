# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (bench/h2h-2026-10-02/tc3/receipts/tc3-4090-8)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.39.0 @39834a7cff18e4eb0830a65e6d8b54806960722c (GitHub main)
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
e4b(t212) 0.39.0 @39834a7cff18e4eb0830a65e6d8b54806960722c
gnf4(t212) 0.34.0 @846b512b905468c08f5748943d08769b572affa2
torch(e4b-t212) 2.12.1+cu130
axolotl 0.20.0
torch(axolotl) 2.14.0+cu130
transformers(axolotl) 5.17.0
peft(axolotl) 0.21.0
bitsandbytes(axolotl) 0.50.2
python(axolotl) 3.12.15
deepspeed(axolotl) 0.19.7
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc3-4090-8",
 "instance_id": "53830054",
 "gpu": "NVIDIA GeForce RTX 4090",
 "driver": "595.58.03",
 "cpu": "AMD EPYC 7642 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527991836",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   76M  320G   1% /",
 "hostname": "dc1a9dac7875",
 "registered_gpu_class": "4090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC3 (`qwen3frontier` / `qwen3frontier12`, TC3-PREREG.md): one box per token; the box's anchor is `e4b/fused_attn4_m_offload` (the resident e4b arm is expected to OOM); (a) the FIT TABLE per framework; (b) in-box equivalence under TC1's R4 bands (EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05) with the fused/reference offload pair as the control, and with `--tc1-dir` the matched trajectory against TC1's resident `e4b/fused_attn4_m` (median per-step |Δ| ≤ 0.02 reads EQUIVALENT-TO-RESIDENT); (c) no cross-box ratio: P1's ratios are within the 24 GB box, P4 is two measurements; (d) P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED.

### Qwen3-30B-A3B (lane TC3: the 24 GB RTX 4090 memory frontier, every framework with its own lever) (`qwen3frontier`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 r 16 α 16; e4b trainable 642514944; box_class RTX 4090 gpu NVIDIA GeForce RTX 4090; host RAM total 540.664 GB (cgroup limit 259.518); the box's anchor `e4b/fused_attn4_m_offload`
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 24.449 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 105.69 MiB is free. Process 2370149 has 23.40 GiB memory in use. Of the allocated memory 22.67 GiB is |
| e4b | fused_attn4_m_offload | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 10.521 | 144.7 | 11.881 | 1325.4 | 2.0509→0.8307 | 1.9542→0.8548 | the box's anchor | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_mb1 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 24.530 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 2: CUDA out of memory. Tried to allocate 206.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 143.69 MiB is free. Process 2384322 has 23.37 GiB memory in use. Of the allocated memory 22.68 GiB is |
| e4b | fused_attn4_shipped | **OOM** | — | **OOM** | native | native / bfloat16,float32 | — | 20 | — | — | 24.408 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 18: CUDA out of memory. Tried to allocate 644.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 557.69 MiB is free. Process 2387951 has 22.96 GiB memory in use. Of the allocated memory 22.26 GiB is |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0028) | 20 | 10.032 | 138.5 | 24.219 | 1145.3 | 2.0448→0.8323 | 1.9515→0.8445 | -0.0103 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0028) | 20 | 18.366 | 80.7 | 24.192 | 1922.0 | 2.0753→0.8712 | 1.9515→0.8418 | -0.0130 | stacks 96 / fwd 768 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 9.69 MiB is free. Process 2421420 has 23.50 GiB memory in use. Of the allocated memory 23.05 GiB is allocated by PyTorch, and 12.74 MiB is reser |
| hf | hf_peft_m_offload | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | RuntimeError: Tensor.item() cannot be called on meta tensors |
| axolotl | ckpt_axolotl_m | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | RuntimeError in phase after-prologue: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float |
| axolotl | ckpt_axolotl_m_layeroffload | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | RuntimeError: invalid argument to getCurrentStream |
| axolotl | ckpt_axolotl_m_zero3 | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning (loaders/model.py:1152-1189, gated on the launcher's ACCELERATE_DEEPSPEED_ZERO_STAGE; modeling_utils.py:1440)  |
| e4b | reference_attn4_m_offload | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0155) | 20 | 77.387 | 19.3 | 13.890 | 4603.8 | 2.0673→0.8339 | 1.9697→0.8503 | -0.0045 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_offload` **117.5 s** before step 1 (35% of the arm): load_weights 52.2, c1_before 40.5, c1_after 20.4, eval0 7.9; unattributed 1.412; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **126.9 s** before step 1 (36% of the arm): c1_before 63.5, c1_after 31.9, load_weights 27.5, eval0 17.3; unattributed 10.992; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **120.7 s** before step 1 (24% of the arm): c1_before 63.8, c1_after 31.8, load_weights 27.8, eval0 10.6; unattributed 10.873; budget 1260.0
- prologue `e4b/reference_attn4_m_offload` **140.7 s** before step 1 (8% of the arm): load_weights 52.4, c1_before 40.5, eval0 31.0, c1_after 21.0; unattributed 1.376; budget 1890.0
- **(a) FIT TABLE** (per framework: did any arm complete on this box, with its lever; peak VRAM = torch max_memory_allocated over the window (an OOM row: at the OOM); host RAM high-water = max over the arm of the process peak RSS and the cgroup peak when it rose during the arm; s/step = the median over steps 11..N; J/step = net of idle):
| framework | arm | lever | **VERDICT** | peak VRAM GB | host RAM high-water GB | s/step | J/step | regime | note |
|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | resident | **OOM** | 24.449 | 91.003 | — | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 105.69 MiB is free. Proces |
| e4b | fused_attn4_m_offload | e4b expert offload (--offload 1) | **VALID** | 11.881 | 113.460 | 10.521 | 1325.4 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_mb1 | resident, micro-batch 1 x accum 8 | **OOM** | 24.530 | 63.895 | — | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 206.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 143.69 MiB is free. Proces |
| e4b | fused_attn4_shipped | resident, as shipped (bf16 expert adapters, N(0,1/r) init): a fit row, never a position | **OOM** | 24.408 | 64.256 | — | — | — | OOM at step 18: CUDA out of memory. Tried to allocate 644.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 557.69 MiB is free. Proce |
| e4b | reference_attn4_m_offload | e4b expert offload, the reference path | **VALID** | 13.890 | 86.174 | 77.387 | 4603.8 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | resident (Unsloth's own: use_gradient_checkpointing=unsloth) | **VALID** | 24.219 | 62.887 | 10.032 | 1145.3 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_mb1 | micro-batch 1 x accum 8 | **VALID** | 24.192 | 62.891 | 18.366 | 1922.0 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | resident | **OOM** | — | — | — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 9.69 MiB is free. Proces |
| hf | hf_peft_m_offload | accelerate device_map=auto + max_memory (--hf-offload 1) | **UNSUPPORTED** | — | — | — | — | — | RuntimeError: Tensor.item() cannot be called on meta tensors |
| axolotl | ckpt_axolotl_m | resident (quantize_moe_experts) | **UNSUPPORTED** | — | — | — | — | — | RuntimeError in phase after-prologue: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float |
| axolotl | ckpt_axolotl_m_layeroffload | layer_offloading (--axolotl-layer-offload 1) | **UNSUPPORTED** | — | — | — | — | — | RuntimeError: invalid argument to getCurrentStream |
| axolotl | ckpt_axolotl_m_zero3 | DeepSpeed ZeRO-3 parameter offload, bf16 experts (--axolotl-zero3 1) | **UNSUPPORTED** | — | — | — | — | — | axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() par |
- fit `e4b`: **FITS on this box: `fused_attn4_m_offload` [e4b expert offload (--offload 1)] VALID, `reference_attn4_m_offload` [e4b expert offload, the reference path] VALID**
- fit `unsloth`: **FITS on this box: `ckpt_unsloth_m` [resident (Unsloth's own: use_gradient_checkpointing=unsloth)] VALID, `ckpt_unsloth_m_mb1` [micro-batch 1 x accum 8] VALID**
- fit `hf`: **NO ARM COMPLETED on this box: `hf_peft_m` OOM, `hf_peft_m_offload` UNSUPPORTED**
- fit `axolotl`: **NO ARM COMPLETED on this box: `ckpt_axolotl_m` UNSUPPORTED, `ckpt_axolotl_m_layeroffload` UNSUPPORTED, `ckpt_axolotl_m_zero3` UNSUPPORTED**
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is OOM); `e4b/fused_attn4_m_offload` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered)); `hf/hf_peft_m` — (hf/hf_peft_m is OOM); `axolotl/ckpt_axolotl_m` — (axolotl/ckpt_axolotl_m is UNSUPPORTED); `e4b/reference_attn4_m_offload` SINGLE (single draw (no second draw registered))
- e4b internal parity under offload (tp1's rule): fused_attn4_m_offload vs reference_attn4_m_offload Δfinal 0.00319, median step |Δ| 0.00282 → **PASS** (band 0.05/0.05); ×7.36 faster per step, peak ×0.855
- **(b) in-box equivalence** vs the box's anchor `e4b/fused_attn4_m_offload` (TC1's R4 bands, fixed: EQUIVALENT iff median step |Δ train| ≤ 0.02 and |Δ held-out at N| ≤ 0.02; COMPARABLE ≤ 0.05; the fused/reference pair is the control): `e4b/fused_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0018, |Δ held-out at N| 0.0103, step-0 0.0028 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0095); `unsloth/ckpt_unsloth_m_mb1` **COMPARABLE** (median step |Δ| 0.0219, |Δ held-out at N| 0.0130, step-0 0.0028 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0031); `hf/hf_peft_m` **—** — no OK receipt; `hf/hf_peft_m_offload` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m_layeroffload` **—** — no OK receipt; `axolotl/ckpt_axolotl_m_zero3` **—** — no OK receipt; `e4b/reference_attn4_m_offload` **EQUIVALENT** (median step |Δ| 0.0028, |Δ held-out at N| 0.0045, step-0 0.0155 NEAR, |Δ loss at step 2| 0.0120)
- **RESIDENT COMPARISON: **EQUIVALENT-TO-RESIDENT**** — the offload anchor vs TC1's resident `e4b/fused_attn4_m` (same tokens, init, precision, N asserted): median per-step |Δ| 0.0025 vs band 0.02, |Δ held-out at N| 0.0032, step-0 0.0037 SAME-BYTES-CLASS; measurements beside each other, never a ratio: this box's offload 10.521 s/step at peak 11.881 GB; TC1's resident on RTX 5090 5.688 s/step over 2 draw(s) at peak 27.849 GB
- matched_init_sha (B, name-free): anchor `f7832488926eda91`; `e4b/fused_attn4_m_offload` same; `unsloth/ckpt_unsloth_m` same; `unsloth/ckpt_unsloth_m_mb1` same; `e4b/reference_attn4_m_offload` same
- lever `axolotl/ckpt_axolotl_m_zero3` (ZeRO-3): REFUSED — axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning (loaders/model.py:1152-1189, gated on the launcher's ACCELERATE_DEEPSPEED_ZERO_STAGE; modeling_utils.py:1440)  (deepspeed 0.19.7; the config it would have used is on the receipt)

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3frontier | e4b/fused_attn4_m | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 105.69 MiB is free. Process 2370149 has 23.40  |
| qwen3frontier | e4b/fused_attn4_m_offload | **VALID** | VALID | the box's anchor |  |
| qwen3frontier | e4b/fused_attn4_m_mb1 | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 206.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 143.69 MiB is free. Process 2384322 has 23.37  |
| qwen3frontier | e4b/fused_attn4_shipped | **OOM** | — | — | OOM at step 18: CUDA out of memory. Tried to allocate 644.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 557.69 MiB is free. Process 2387951 has 22.96 |
| qwen3frontier | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0103 |  |
| qwen3frontier | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | -0.0130 |  |
| qwen3frontier | hf/hf_peft_m | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 9.69 MiB is free. Process 2421420 has 23.50  |
| qwen3frontier | hf/hf_peft_m_offload | **UNSUPPORTED** | — | — | RuntimeError: Tensor.item() cannot be called on meta tensors |
| qwen3frontier | axolotl/ckpt_axolotl_m | **UNSUPPORTED** | — | — | RuntimeError in phase after-prologue: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float |
| qwen3frontier | axolotl/ckpt_axolotl_m_layeroffload | **UNSUPPORTED** | — | — | RuntimeError: invalid argument to getCurrentStream |
| qwen3frontier | axolotl/ckpt_axolotl_m_zero3 | **UNSUPPORTED** | — | — | axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning (loaders/m |
| qwen3frontier | e4b/reference_attn4_m_offload | **VALID** | VALID | -0.0045 |  |

## TC3 predictions P1–P4 (TC3-PREREG-draft, scored mechanically; P1 on the 24 GB token, P2 on the 12 GB token, P3 per token, P4 on the 24 GB token with --tc1-dir)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3frontier | **FALSIFIED** | (i) e4b offload VALID at 10.521 s/step; mb1 resident OOM (informational) -> HELD; (ii) Unsloth completed: m VALID, mb1 VALID -> FALSIFIED; (iii-a) HF resident OOM -> HELD; (iii-b) HF offload REFUSED (RuntimeError: Tensor.item() cannot be called on meta tensors) -> HELD; (iv) axolotl zero3 did not run (UNSUPPORTED: axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: Mod) -- a conditional clause, vacuously held -> HELD |
| P3 | qwen3frontier | **HELD** | median per-step |Δ| vs TC1's resident e4b/fused_attn4_m 0.0025 <= 0.02 (|Δ held-out at N| 0.0032, step-0 0.0037 SAME-BYTES-CLASS); in-box control fused_offload vs reference_offload: EQUIVALENT (median step |Δ| 0.0028, |Δ held-out at N| 0.0045) |
| P4 | qwen3frontier | **HELD** | e4b offload on this box (RTX 4090) 10.521 s/step; TC1's resident e4b/fused_attn4_m on RTX 5090 5.688 s/step over 2 draw(s); 2 x 5.688 = 11.376 -- two measurements, no cross-box ratio formed |
