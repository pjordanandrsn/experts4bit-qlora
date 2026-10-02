# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (bench/h2h-2026-10-02/tc3/receipts/local-20261002T052229Z)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.39.0 @516c87c0aa4253ed8ac6a683ee96fad69be58ba8 (GitHub main)
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
e4b 0.39.0 @516c87c0aa4253ed8ac6a683ee96fad69be58ba8 (GitHub main)
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
```
`box.json`
```
{
 "box": "A",
 "run_id": "local-20261002T052229Z",
 "instance_id": "local:gpu-dev",
 "gpu": "NVIDIA RTX A2000 12GB",
 "driver": "575.64.05",
 "cpu": "Intel(R) Xeon(R) W-1250 CPU @ 3.30GHz",
 "nproc": 12,
 "mem_total_kb": "131757372",
 "cgroup_memory_max": "",
 "disk_root": "overlay         598G  197G  396G  34% /",
 "hostname": "gpu-dev",
 "registered_gpu_class": "RTX A2000",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": "1",
 "local_snapshot": "/models/Qwen3-30B-A3B"
}
```
Lane TC3 (`qwen3frontier` / `qwen3frontier12`, TC3-PREREG.md): one box per token; the box's anchor is `e4b/fused_attn4_m_offload` (the resident e4b arm is expected to OOM); (a) the FIT TABLE per framework; (b) in-box equivalence under TC1's R4 bands (EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05) with the fused/reference offload pair as the control, and with `--tc1-dir` the matched trajectory against TC1's resident `e4b/fused_attn4_m` (median per-step |Δ| ≤ 0.02 reads EQUIVALENT-TO-RESIDENT); (c) no cross-box ratio: P1's ratios are within the 24 GB box, P4 is two measurements; (d) P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED.

### Qwen3-30B-A3B (lane TC3: the owned 12 GB RTX A2000, e4b's offload territory) (`qwen3frontier12`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 1 × accum 8 r 16 α 16; e4b trainable 642514944; box_class RTX A2000 gpu NVIDIA RTX A2000 12GB; host RAM total 134.920 GB (cgroup limit —); the box's anchor `e4b/fused_attn4_m_offload_mb1`
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_offload | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 10.465 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 2: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 65.50 MiB is free. Including non-PyTorch memory, this process has 10.17 GiB memory in use. Of the all |
| e4b | fused_attn4_m_offload_d2 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 10.465 | — | —→— | —→— | — | patched 48 / kcalls — | 642514944 | — | OOM at step 2: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 65.50 MiB is free. Including non-PyTorch memory, this process has 10.17 GiB memory in use. Of the all |
| e4b | reference_attn4_m_offload | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 10.789 | — | —→— | —→— | — | patched 0 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 12.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 15.50 MiB is free. Including non-PyTorch memory, this process has 10.22 GiB memory in use. Of the allo |
| e4b | fused_attn4_m | **ALARM** | — | **ALARM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | prologue phase 'load_weights' has run 413.2s and the prologue is 420.2s in, past its 420.0s budget; refusing now so this row can name the phase (#548) |
| unsloth | ckpt_unsloth_m_mb1 | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules in 32-bit, you need to set `llm_int8_enable_fp32_ |
| hf | hf_peft_m_mb1 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 5.50 MiB is free. Including non-PyTorch memory, this process has 11.04 GiB memory in use. Of the allocated memory 10.93 GiB is allocated by PyTo |
| axolotl | ckpt_axolotl_m | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | cu130 wheels need driver >= 580; host has 575.64.05 |
| e4b | fused_attn4_m_offload_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 69.919 | 22.0 | 10.463 | 3273.4 | 2.0882→0.8717 | 1.9533→0.8483 | the box's anchor | patched 48 / kcalls 1536 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_offload | **ALARM** | — | **ALARM** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | prologue phase 'load_weights' has run 2511.2s and the prologue is 2520.0s in, past its 2520.0s budget; refusing now so this row can name the phase (#548) |
| e4b | reference_attn4_m_offload_mb1 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 10.779 | — | —→— | —→— | — | patched 0 / kcalls — | 642514944 | — | OOM at step 1: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 3.50 MiB is free. Including non-PyTorch memory, this process has 10.23 GiB memory in use. Of the alloc |
- prologue `e4b/fused_attn4_m` **420.2 s** before step 1 (—): load_weights (in flight) 413.2, preamble 5.3; unattributed —; budget 420.0
- prologue `e4b/fused_attn4_m_offload_mb1` **2013.9 s** before step 1 (59% of the arm): load_weights 1871.8, c1_before 84.2, c1_after 47.4, eval0 31.6; unattributed 2.099; budget 2520.0
- prologue `e4b/fused_attn4_shipped_offload` **2520.0 s** before step 1 (—): load_weights (in flight) 2511.2, preamble 6.5; unattributed —; budget 2520.0
- **(a) FIT TABLE** (per framework: did any arm complete on this box, with its lever; peak VRAM = torch max_memory_allocated over the window (an OOM row: at the OOM); host RAM high-water = max over the arm of the process peak RSS and the cgroup peak when it rose during the arm; s/step = the median over steps 11..N; J/step = net of idle):
| framework | arm | lever | **VERDICT** | peak VRAM GB | host RAM high-water GB | s/step | J/step | regime | note |
|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_offload | e4b expert offload (--offload 1) | **OOM** | 10.465 | 42.074 | — | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 65.50 MiB is free. Includi |
| e4b | fused_attn4_m_offload_d2 | e4b expert offload (--offload 1), draw 2 | **OOM** | 10.465 | 39.527 | — | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 65.50 MiB is free. Includi |
| e4b | reference_attn4_m_offload | e4b expert offload, the reference path | **OOM** | 10.789 | 36.772 | — | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 12.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 15.50 MiB is free. Includin |
| e4b | fused_attn4_m | resident | **ALARM** | — | — | — | — | — | prologue phase 'load_weights' has run 413.2s and the prologue is 420.2s in, past its 420.0s budget; refusing now so this row can name the ph |
| e4b | fused_attn4_m_offload_mb1 | e4b expert offload, micro-batch 1 x accum 8 (the 12 GB secondary) | **VALID** | 10.463 | 25.345 | 69.919 | 3273.4 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_offload | e4b expert offload, as shipped (bf16 expert adapters, N(0,1/r) init): a fit row, never a position | **ALARM** | — | — | — | — | — | prologue phase 'load_weights' has run 2511.2s and the prologue is 2520.0s in, past its 2520.0s budget; refusing now so this row can name the |
| e4b | reference_attn4_m_offload_mb1 | e4b expert offload, the reference path, micro-batch 1 x accum 8 | **OOM** | 10.779 | 33.061 | — | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 3.50 MiB is free. Including |
| unsloth | ckpt_unsloth_m_mb1 | micro-batch 1 x accum 8 | **UNSUPPORTED** | — | — | — | — | — | ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to |
| hf | hf_peft_m_mb1 | micro-batch 1 x accum 8 | **OOM** | — | — | — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 5.50 MiB is free. Includ |
| axolotl | ckpt_axolotl_m | resident (quantize_moe_experts) | **UNSUPPORTED** | — | — | — | — | — | cu130 wheels need driver >= 580; host has 575.64.05 |
- fit `e4b`: **FITS on this box: `fused_attn4_m_offload_mb1` [e4b expert offload, micro-batch 1 x accum 8 (the 12 GB secondary)] VALID**
- fit `unsloth`: **NO ARM COMPLETED on this box: `ckpt_unsloth_m_mb1` UNSUPPORTED**
- fit `hf`: **NO ARM COMPLETED on this box: `hf_peft_m_mb1` OOM**
- fit `axolotl`: **NO ARM COMPLETED on this box: `ckpt_axolotl_m` UNSUPPORTED**
- draws (R1): `e4b/fused_attn4_m_offload` — (e4b/fused_attn4_m_offload is OOM); `e4b/fused_attn4_m` — (e4b/fused_attn4_m is ALARM); `axolotl/ckpt_axolotl_m` — (axolotl/ckpt_axolotl_m is UNSUPPORTED); `e4b/fused_attn4_m_offload_mb1` SINGLE (single draw (no second draw registered))
- e4b internal parity under offload: NO-ARM
- **(b) in-box equivalence** vs the box's anchor `e4b/fused_attn4_m_offload` (TC1's R4 bands, fixed: EQUIVALENT iff median step |Δ train| ≤ 0.02 and |Δ held-out at N| ≤ 0.02; COMPARABLE ≤ 0.05; the fused/reference pair is the control): `e4b/fused_attn4_m_offload_d2` **—** — no OK receipt; `e4b/reference_attn4_m_offload` **—** — no OK receipt; `e4b/fused_attn4_m` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_mb1` **—** — no OK receipt; `hf/hf_peft_m_mb1` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_offload_mb1` **N-A** — the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK; `e4b/reference_attn4_m_offload_mb1` **—** — no OK receipt
- RESIDENT COMPARISON: N-A — the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK
- matched_init_sha (B, name-free): anchor `f7832488926eda91`; `e4b/fused_attn4_m_offload_mb1` same

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3frontier12 | e4b/fused_attn4_m_offload | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 65.50 MiB is free. Including non-PyTorch memor |
| qwen3frontier12 | e4b/fused_attn4_m_offload_d2 | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 65.50 MiB is free. Including non-PyTorch memor |
| qwen3frontier12 | e4b/reference_attn4_m_offload | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 12.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 15.50 MiB is free. Including non-PyTorch memory |
| qwen3frontier12 | e4b/fused_attn4_m | **ALARM** | — | — | prologue phase 'load_weights' has run 413.2s and the prologue is 420.2s in, past its 420.0s budget; refusing now so this row can name the phase (#548) |
| qwen3frontier12 | unsloth/ckpt_unsloth_m_mb1 | **UNSUPPORTED** | — | — | ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to dispatch the model  |
| qwen3frontier12 | hf/hf_peft_m_mb1 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 5.50 MiB is free. Including non-PyTorch memo |
| qwen3frontier12 | axolotl/ckpt_axolotl_m | **UNSUPPORTED** | — | — | cu130 wheels need driver >= 580; host has 575.64.05 |
| qwen3frontier12 | e4b/fused_attn4_m_offload_mb1 | **VALID** | VALID | the box's anchor |  |
| qwen3frontier12 | e4b/fused_attn4_shipped_offload | **ALARM** | — | — | prologue phase 'load_weights' has run 2511.2s and the prologue is 2520.0s in, past its 2520.0s budget; refusing now so this row can name the phase (#548) |
| qwen3frontier12 | e4b/reference_attn4_m_offload_mb1 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.62 GiB of which 3.50 MiB is free. Including non-PyTorch memory, |

## TC3 predictions P1–P4 (TC3-PREREG-draft, scored mechanically; P1 on the 24 GB token, P2 on the 12 GB token, P3 per token, P4 on the 24 GB token with --tc1-dir)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P2 | qwen3frontier12 | **HELD** | e4b offload arms fused_attn4_m_offload OOM, fused_attn4_m_offload_d2 OOM, reference_attn4_m_offload OOM ; mb1 secondary fused_attn4_m_offload_mb1 VALID, reference_attn4_m_offload_mb1 OOM; other frameworks unsloth/ckpt_unsloth_m_mb1 UNSUPPORTED (not an OOM reading: ValueError: Some modules are dispatched on the CPU or the di), hf/hf_peft_m_mb1 OOM, axolotl/ckpt_axolotl_m UNSUPPORTED (not an OOM reading: cu130 wheels need driver >= 580; host has 575.64.05); e4b resident ALARM; e4b as-shipped offload ALARM (a fit row, outside P2); the e4b fused offload arm completed at its mb1 secondary (the field recipe OOMed) |
| P3 | qwen3frontier12 | **UNTESTED** | resident comparison N-A: the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK; in-box control fused_offload vs reference_offload: — (no OK receipt) |
