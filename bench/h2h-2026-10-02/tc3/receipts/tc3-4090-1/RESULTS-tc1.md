# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.38.1 @ff1af2dca1ad03814db5513d586aadf5cd6adba2 (GitHub main)
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
e4b(t212) 0.38.1 @ff1af2dca1ad03814db5513d586aadf5cd6adba2
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
 "run_id": "tc3-4090-1",
 "instance_id": "53804549",
 "gpu": "NVIDIA GeForce RTX 4090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527974204",
 "cgroup_memory_max": "259508928512",
 "disk_root": "overlay         320G  1.9M  320G   1% /",
 "hostname": "c4d8dd892048",
 "registered_gpu_class": "4090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC3 (`qwen3frontier` / `qwen3frontier12`, TC3-PREREG.md): one box per token; the box's anchor is `e4b/fused_attn4_m_offload` (the resident e4b arm is expected to OOM); (a) the FIT TABLE per framework; (b) in-box equivalence under TC1's R4 bands (EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05) with the fused/reference offload pair as the control, and with `--tc1-dir` the matched trajectory against TC1's resident `e4b/fused_attn4_m` (median per-step |Δ| ≤ 0.02 reads EQUIVALENT-TO-RESIDENT); (c) no cross-box ratio: P1's ratios are within the 24 GB box, P4 is two measurements; (d) P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED.

### Qwen3-30B-A3B (lane TC3: the 24 GB RTX 4090 memory frontier, every framework with its own lever) (`qwen3frontier`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 r 16 α 16; e4b trainable None; box_class RTX 4090 gpu NVIDIA GeForce RTX 4090; host RAM total 540.646 GB (cgroup limit 259.509); NO ANCHOR: no e4b offload arm completed
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 24.320 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 284.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 271.00 MiB is free. Including non-PyTorch memory, this process has 23.23 GiB memory in use. Of the al |
| e4b | fused_attn4_m_offload | **HARNESS_ERROR** | — | **HARNESS_ERROR** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| e4b | fused_attn4_m_mb1 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 24.302 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 230.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 249.00 MiB is free. Including non-PyTorch memory, this process has 23.25 GiB memory in use. Of the al |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | 8.434 | 163.8 | 24.219 | 921.6 | 2.0595→0.8334 | 1.9515→0.8527 | N-A (anchor missing) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | 15.707 | 94.3 | 24.192 | 1674.9 | 2.0753→0.8691 | 1.9515→0.8423 | N-A (anchor missing) | stacks 96 / fwd 768 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 35.00 MiB is free. Including non-PyTorch memory, this process has 23.46 GiB memory in use. Of the allocated memory 23.01 GiB is allocated by PyT |
| hf | hf_peft_m_offload | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | RuntimeError: Tensor.item() cannot be called on meta tensors |
| axolotl | ckpt_axolotl_m | **HARNESS_ERROR** | — | **HARNESS_ERROR** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| axolotl | ckpt_axolotl_m_layeroffload | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | RuntimeError: invalid argument to getCurrentStream |
| axolotl | ckpt_axolotl_m_zero3 | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning (loaders/model.py:1152-1189, gated on the launcher's ACCELERATE_DEEPSPEED_ZERO_STAGE; modeling_utils.py:1440)  |
| e4b | reference_attn4_m_offload | **HARNESS_ERROR** | — | **HARNESS_ERROR** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
- prologue `unsloth/ckpt_unsloth_m` **107.8 s** before step 1 (36% of the arm): c1_before 50.6, c1_after 25.7, load_weights 25.2, eval0 15.0; unattributed 9.297; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **102.2 s** before step 1 (24% of the arm): c1_before 50.7, c1_after 25.6, load_weights 25.3, eval0 9.3; unattributed 9.291; budget 1260.0
- **(a) FIT TABLE** (per framework: did any arm complete on this box, with its lever; peak VRAM = torch max_memory_allocated over the window (an OOM row: at the OOM); host RAM high-water = max over the arm of the process peak RSS and the cgroup peak when it rose during the arm; s/step = the median over steps 11..N; J/step = net of idle):
| framework | arm | lever | **VERDICT** | peak VRAM GB | host RAM high-water GB | s/step | J/step | regime | note |
|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | resident | **OOM** | 24.320 | 64.282 | — | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 284.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 271.00 MiB is free. Includ |
| e4b | fused_attn4_m_offload | e4b expert offload (--offload 1) | **HARNESS_ERROR** | — | — | — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| e4b | fused_attn4_m_mb1 | resident, micro-batch 1 x accum 8 | **OOM** | 24.302 | 63.926 | — | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 230.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 249.00 MiB is free. Includ |
| e4b | reference_attn4_m_offload | e4b expert offload, the reference path | **HARNESS_ERROR** | — | — | — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| unsloth | ckpt_unsloth_m | resident (Unsloth's own: use_gradient_checkpointing=unsloth) | **VALID** | 24.219 | 62.894 | 8.434 | 921.6 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_mb1 | micro-batch 1 x accum 8 | **VALID** | 24.192 | 62.895 | 15.707 | 1674.9 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | resident | **OOM** | — | — | — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 35.00 MiB is free. Inclu |
| hf | hf_peft_m_offload | accelerate device_map=auto + max_memory (--hf-offload 1) | **UNSUPPORTED** | — | — | — | — | — | RuntimeError: Tensor.item() cannot be called on meta tensors |
| axolotl | ckpt_axolotl_m | resident (quantize_moe_experts) | **HARNESS_ERROR** | — | — | — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| axolotl | ckpt_axolotl_m_layeroffload | layer_offloading (--axolotl-layer-offload 1) | **UNSUPPORTED** | — | — | — | — | — | RuntimeError: invalid argument to getCurrentStream |
| axolotl | ckpt_axolotl_m_zero3 | DeepSpeed ZeRO-3 parameter offload, bf16 experts (--axolotl-zero3 1) | **UNSUPPORTED** | — | — | — | — | — | axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() par |
- fit `e4b`: **NO ARM COMPLETED on this box: `fused_attn4_m` OOM, `fused_attn4_m_offload` HARNESS_ERROR, `fused_attn4_m_mb1` OOM, `reference_attn4_m_offload` HARNESS_ERROR**
- fit `unsloth`: **FITS on this box: `ckpt_unsloth_m` [resident (Unsloth's own: use_gradient_checkpointing=unsloth)] VALID, `ckpt_unsloth_m_mb1` [micro-batch 1 x accum 8] VALID**
- fit `hf`: **NO ARM COMPLETED on this box: `hf_peft_m` OOM, `hf_peft_m_offload` UNSUPPORTED**
- fit `axolotl`: **NO ARM COMPLETED on this box: `ckpt_axolotl_m` HARNESS_ERROR, `ckpt_axolotl_m_layeroffload` UNSUPPORTED, `ckpt_axolotl_m_zero3` UNSUPPORTED**
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is OOM); `e4b/fused_attn4_m_offload` — (e4b/fused_attn4_m_offload is HARNESS_ERROR); `unsloth/ckpt_unsloth_m` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered)); `hf/hf_peft_m` — (hf/hf_peft_m is OOM); `axolotl/ckpt_axolotl_m` — (axolotl/ckpt_axolotl_m is HARNESS_ERROR)
- e4b internal parity under offload: NO-ARM
- **(b) in-box equivalence** vs the box's anchor `e4b/fused_attn4_m_offload` (TC1's R4 bands, fixed: EQUIVALENT iff median step |Δ train| ≤ 0.02 and |Δ held-out at N| ≤ 0.02; COMPARABLE ≤ 0.05; the fused/reference pair is the control): `e4b/fused_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m` **N-A** — the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK; `unsloth/ckpt_unsloth_m_mb1` **N-A** — the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK; `hf/hf_peft_m` **—** — no OK receipt; `hf/hf_peft_m_offload` **—** — no OK receipt; `axolotl/ckpt_axolotl_m_layeroffload` **—** — no OK receipt; `axolotl/ckpt_axolotl_m_zero3` **—** — no OK receipt
- RESIDENT COMPARISON: UNTESTED — no --tc1-dir given
- lever `axolotl/ckpt_axolotl_m_zero3` (ZeRO-3): REFUSED — axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning (loaders/model.py:1152-1189, gated on the launcher's ACCELERATE_DEEPSPEED_ZERO_STAGE; modeling_utils.py:1440)  (deepspeed 0.19.7; the config it would have used is on the receipt)

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3frontier | e4b/fused_attn4_m | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 284.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 271.00 MiB is free. Including non-PyTorch memo |
| qwen3frontier | e4b/fused_attn4_m_offload | **HARNESS_ERROR** | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| qwen3frontier | e4b/fused_attn4_m_mb1 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 230.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 249.00 MiB is free. Including non-PyTorch memo |
| qwen3frontier | unsloth/ckpt_unsloth_m | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3frontier | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3frontier | hf/hf_peft_m | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 35.00 MiB is free. Including non-PyTorch mem |
| qwen3frontier | hf/hf_peft_m_offload | **UNSUPPORTED** | — | — | RuntimeError: Tensor.item() cannot be called on meta tensors |
| qwen3frontier | axolotl/ckpt_axolotl_m | **HARNESS_ERROR** | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| qwen3frontier | axolotl/ckpt_axolotl_m_layeroffload | **UNSUPPORTED** | — | — | RuntimeError: invalid argument to getCurrentStream |
| qwen3frontier | axolotl/ckpt_axolotl_m_zero3 | **UNSUPPORTED** | — | — | axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning (loaders/m |
| qwen3frontier | e4b/reference_attn4_m_offload | **HARNESS_ERROR** | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |

## TC3 predictions P1–P4 (TC3-PREREG-draft, scored mechanically; P1 on the 24 GB token, P2 on the 12 GB token, P3 per token, P4 on the 24 GB token with --tc1-dir)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3frontier | **FALSIFIED** | (i) e4b offload HARNESS_ERROR; mb1 resident OOM (informational) -> UNTESTED; (ii) Unsloth completed: m VALID, mb1 VALID -> FALSIFIED; (iii-a) HF resident OOM -> HELD; (iii-b) HF offload REFUSED (RuntimeError: Tensor.item() cannot be called on meta tensors) -> HELD; (iv) axolotl zero3 did not run (UNSUPPORTED: axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: Mod) -- a conditional clause, vacuously held -> HELD |
| P3 | qwen3frontier | **UNTESTED** | resident comparison UNTESTED: no --tc1-dir given; in-box control fused_offload vs reference_offload: — |
| P4 | qwen3frontier | **UNTESTED** | resident comparison UNTESTED: no --tc1-dir given |
