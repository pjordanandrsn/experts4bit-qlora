# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.40.0 @0933d4e20b3adff6fea9c4f9ac0e5c37d25b338e (GitHub main)
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
e4b(t212) 0.40.0 @0933d4e20b3adff6fea9c4f9ac0e5c37d25b338e
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
 "box": "B",
 "run_id": "tc1-5090-27",
 "instance_id": "53859323",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "64c505c2c79b",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC2 (`tc2small` / `tc2big` tokens, TC2-PREREG.md, drafted in TC2-PREREG-draft): TC1's readings per family with the registered n_layers {'granite': 32, 'olmoe': 16, 'gptoss': 24, 'qwen3_5': 40, 'mixtral': 32} and attention census {'granite': 128, 'olmoe': 64, 'gptoss': None, 'qwen3_5': None, 'mixtral': 128} (None = the receipt's own structural census governs); the HF position is quoted as HF (bf16 experts) / e4b (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID (attention-only when it adapted no expert parameter), as tp4; gpt-oss carries a NO COMMON ADAPTER SET line (both s/step values, both trainable counts), never a ratio; mixtral's FOOTPRINT line (e4b under expert offload vs Unsloth resident: peak VRAM and s/step) leads its block; `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert parameters of a recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime; an HF t214 arm whose dispatch did not reach grouped_mm is recorded, never VOID; P1–P7 of the draft scored HELD / FALSIFIED / UNTESTED (P1 HF band (1.1, 1.6); P2 Unsloth (1.2, 3.0) / HF (1.5, 2.5); P4 Unsloth (2.0, 6.0), e4b within 15% of tp4's 6.3344 s/step; P5 e4b/Unsloth (0.3, 0.5) at a >= 8x lower e4b peak, tp2 0.361).

### Qwen3.6-35B-A3B (lane TC2, box B) (`qwen3_5`, registered n_layers 40)
- model `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd9`; tokens sha `0d94e9b88936`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable None; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.520 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 1: CUDA out of memory. Tried to allocate 606.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 495.00 MiB is free. Including non-PyTorch memory, this process has 30.86 GiB memory in use. Of the al |
| unsloth | ckpt_unsloth_m | **OK** | VOID | **VOID** | yes | matched:3407 (INCOMPLETE 40/20520) / float32 | — | 20 | 4.866 | 209.2 | 21.083 | 1215.6 | 1.1905→0.7754 | 1.1940→0.8355 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 3440640 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) | matched_init.complete is False (set 40 of 20520; unmapped []) |
| e4b | fused_attn4_m_d2 | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.520 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 1: CUDA out of memory. Tried to allocate 606.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 495.06 MiB is free. Including non-PyTorch memory, this process has 30.86 GiB memory in use. Of the al |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VOID | **VOID** | yes | matched:3407 (INCOMPLETE 40/20520) / float32 | — | 20 | 4.890 | 228.2 | 21.083 | 1173.2 | 1.1905→0.7776 | 1.1940→0.8373 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 3440640 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) | matched_init.complete is False (set 40 of 20520; unmapped []) |
| unsloth | ckpt_unsloth_m_experts | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 10.593 | 143.1 | 30.469 | 1603.5 | 1.1905→0.6968 | 1.1940→0.6880 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| hf | hf_peft_m | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 15.06 MiB is free. Including non-PyTorch memory, this process has 31.33 GiB memory in use. Of the allocated memory 30.74 GiB is allocated by Py |
| axolotl | ckpt_axolotl_m | **REFUSED** | — | **UNSUPPORTED** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | NoMatchingPeftModuleError: Target modules {'model.layers.35.self_attn.q_proj', 'model.layers.39.self_attn.v_proj', 'model.layers.27.self_attn.o_proj', 'model.layers.23.self_attn.q_proj', 'model.layers.19.self_attn.o_proj', 'model.layers.23.self_attn.v_proj', ' |
| axolotl | ckpt_axolotl_best | **REFUSED** | — | **UNSUPPORTED** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | ValueError: Version 1 of 'kernels-community/activation' is not available in the local cache and Hugging Face Hub is in offline mode. Download the kernel while online first, or pass an explicit `revision=<commit>`. |
| e4b | fused_attn4_shipped | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 5.088 | 314.6 | 32.484 | 1200.0 | 1.1365→0.6909 | 1.1433→0.6729 | N-A (anchor missing) | patched 40 / kcalls 640 | 926187520 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | reference_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.837 | — | —→— | —→— | — | patched 0 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 440.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 367.00 MiB is free. Including non-PyTorch memory, this process has 30.99 GiB memory in use. Of the al |
| e4b | fused_attn4_m_mb1 | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.542 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 366.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 301.00 MiB is free. Including non-PyTorch memory, this process has 31.05 GiB memory in use. Of the al |
| hf | hf_peft_m_mb1 | **OOM** | — | **OOM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 15.06 MiB is free. Including non-PyTorch memory, this process has 31.33 GiB memory in use. Of the allocated memory 30.74 GiB is allocated by Py |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VOID | **VOID** | yes | matched:3407 (INCOMPLETE 40/20520) / float32 | — | 20 | 8.855 | 164.1 | 21.045 | 1659.7 | 1.1995→0.8571 | 1.1940→0.8402 | N-A (anchor missing) | stacks 80 / fwd 640 / u8 40 | 3440640 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) | matched_init.complete is False (set 40 of 20520; unmapped []) |
- prologue `unsloth/ckpt_unsloth_m` **147.1 s** before step 1 (49% of the arm): c1_before 60.6, eval0 43.9, c1_after 29.8, load_weights 29.4; unattributed 10.149; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **138.3 s** before step 1 (50% of the arm): c1_before 62.8, eval0 35.9, c1_after 29.7, load_weights 26.5; unattributed 10.147; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_experts` **136.7 s** before step 1 (38% of the arm): c1_before 57.6, load_weights 41.7, c1_after 29.0, eval0 14.2; unattributed 10.42; budget 1260.0
- prologue `e4b/fused_attn4_shipped` **115.2 s** before step 1 (53% of the arm): c1_before 61.6, load_weights 33.6, c1_after 31.6, trainable_sha 7.0; unattributed 1.235; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **112.0 s** before step 1 (36% of the arm): c1_before 59.1, c1_after 30.1, load_weights 28.9, eval0 11.2; unattributed 9.825; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is OOM); `unsloth/ckpt_unsloth_m` — (unsloth/ckpt_unsloth_m is VOID); `unsloth/ckpt_unsloth_m_experts` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_shipped` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m is OOM / unsloth unsloth/ckpt_unsloth_m is VOID
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m is OOM / hf hf/hf_peft_m is OOM
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m is OOM / axolotl axolotl/ckpt_axolotl_m is UNSUPPORTED
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (unsloth (mb1))** — not quoted: e4b e4b/fused_attn4_m_mb1 is OOM / unsloth (mb1) unsloth/ckpt_unsloth_m_mb1 is VOID
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (hf (mb1))** — not quoted: e4b e4b/fused_attn4_m_mb1 is OOM / hf (mb1) hf/hf_peft_m_mb1 is OOM
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (axolotl (mb1))** — not quoted: e4b e4b/fused_attn4_m_mb1 is OOM / axolotl (mb1) no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b single draw (no second draw registered) / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b single draw (no second draw registered) / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is UNSUPPORTED
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b e4b/fused_attn4_m is OOM / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is UNSUPPORTED
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b e4b/fused_attn4_m is OOM / e4b shipped (bf16 expert adapters, N(0,1/r) init) single draw (no second draw registered)
- **NO LABELLED ROW ckpt_unsloth_m_experts vs e4b/fused_attn4_m (never the quoted position) QUOTED (unsloth with the family's own expert names as targets (tp4 amendment 4's second arm))** — not quoted: e4b e4b/fused_attn4_m is OOM / unsloth with the family's own expert names as targets (tp4 amendment 4's second arm) single draw (no second draw registered)
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `unsloth/ckpt_unsloth_m` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `e4b/fused_attn4_m_d2` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_d2` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `unsloth/ckpt_unsloth_m_experts` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_mb1` **—** — no OK receipt; `hf/hf_peft_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_mb1` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_experts`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `e4b/fused_attn4_shipped`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_mb1`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects

### Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b under expert offload, Unsloth resident) (`mixtral`, registered n_layers 32, attention census 128)
- **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): not readable** — e4b: NOT_RUN: no peak to read; Unsloth: NOT_RUN: no peak to read
- model `None` @ ``; tokens sha ``; N=20; fixture template None seq 2048 micro-batch 2 × accum 4 lr None r None α None optimizer None autocast None; e4b trainable None; box_class None gpu None
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | family skipped by TC1_SKIP |
| unsloth | ckpt_unsloth_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | family skipped by TC1_SKIP |
| e4b | fused_attn4_m_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | family skipped by TC1_SKIP |
| unsloth | ckpt_unsloth_m_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | family skipped by TC1_SKIP |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | family skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | family skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | family skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | family skipped by TC1_SKIP |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | family skipped by TC1_SKIP |
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is NOT_RUN); `unsloth/ckpt_unsloth_m` — (unsloth/ckpt_unsloth_m is NOT_RUN)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m is NOT_RUN / unsloth unsloth/ckpt_unsloth_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m is NOT_RUN / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m is NOT_RUN / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b e4b/fused_attn4_m is NOT_RUN / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b e4b/fused_attn4_m is NOT_RUN / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `unsloth/ckpt_unsloth_m` **—** — no OK receipt; `e4b/fused_attn4_m_d2` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_d2` **—** — no OK receipt; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **—** — no OK receipt

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3_5 | e4b/fused_attn4_m | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 606.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 495.00 MiB is free. Including non-PyTorch memo |
| qwen3_5 | unsloth/ckpt_unsloth_m | **VOID** | VOID | N-A (anchor missing) | matched_init.complete is False (set 40 of 20520; unmapped []) |
| qwen3_5 | e4b/fused_attn4_m_d2 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 606.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 495.06 MiB is free. Including non-PyTorch memo |
| qwen3_5 | unsloth/ckpt_unsloth_m_d2 | **VOID** | VOID | N-A (anchor missing) | matched_init.complete is False (set 40 of 20520; unmapped []) |
| qwen3_5 | unsloth/ckpt_unsloth_m_experts | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3_5 | hf/hf_peft_m | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 15.06 MiB is free. Including non-PyTorch me |
| qwen3_5 | axolotl/ckpt_axolotl_m | **UNSUPPORTED** | — | — | NoMatchingPeftModuleError: Target modules {'model.layers.35.self_attn.q_proj', 'model.layers.39.self_attn.v_proj', 'model.layers.27.self_attn.o_proj', 'model.la |
| qwen3_5 | axolotl/ckpt_axolotl_best | **UNSUPPORTED** | — | — | ValueError: Version 1 of 'kernels-community/activation' is not available in the local cache and Hugging Face Hub is in offline mode. Download the kernel while o |
| qwen3_5 | e4b/fused_attn4_shipped | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3_5 | e4b/reference_attn4_m | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 440.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 367.00 MiB is free. Including non-PyTorch memo |
| qwen3_5 | e4b/fused_attn4_m_mb1 | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 366.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 301.00 MiB is free. Including non-PyTorch memo |
| qwen3_5 | hf/hf_peft_m_mb1 | **OOM** | — | — | OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 15.06 MiB is free. Including non-PyTorch me |
| qwen3_5 | unsloth/ckpt_unsloth_m_mb1 | **VOID** | VOID | N-A (anchor missing) | matched_init.complete is False (set 40 of 20520; unmapped []) |
| mixtral | e4b/fused_attn4_m | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | unsloth/ckpt_unsloth_m | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | e4b/fused_attn4_m_d2 | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | unsloth/ckpt_unsloth_m_d2 | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | hf/hf_peft_m | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | family skipped by TC1_SKIP |
| mixtral | e4b/reference_attn4_m | **NOT_RUN** | — | — | family skipped by TC1_SKIP |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **FALSIFIED** | HF OOM; unsloth/ckpt_unsloth_m_experts VALID (neither an engaged ratio nor a trainable-count VOID): not quoted: e4b e4b/fused_attn4_m is OOM / unsloth with the family's own expert names as targets (tp4 amendment 4's second arm) single draw (no second draw registered); e4b fused_m not usable: e4b/fused_attn4_m is OOM |
| P5 | mixtral | **UNTESTED** | mixtral position not quoted: not quoted: e4b e4b/fused_attn4_m is NOT_RUN / unsloth unsloth/ckpt_unsloth_m is NOT_RUN (footprint: **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): not readable** — e4b: NOT_R); hf/hf_peft_m NOT_RUN; axolotl/ckpt_axolotl_m NOT_RUN |
| P6 | tc2 | **UNTESTED** | no family with both e4b arms OK |
| P7 | tc2 | **UNTESTED** | no VALID matched pair sharing e4b's adapter set |
