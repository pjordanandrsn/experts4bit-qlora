# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @7145425a0719dde7cb7e0c5cbc405134ccac4287 (GitHub main)
gnf4 0.37.0 @71185d6b52f6fc4d09a881d04e0be2ac196d15fe (GitHub main)
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
e4b(t212) 0.45.0 @7145425a0719dde7cb7e0c5cbc405134ccac4287
gnf4(t212) 0.37.0 @71185d6b52f6fc4d09a881d04e0be2ac196d15fe
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
 "run_id": "tc1-5090-55",
 "instance_id": "54159409",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "Intel(R) Core(TM) Ultra 9 285K",
 "nproc": 24,
 "mem_total_kb": "197216440",
 "cgroup_memory_max": "193871216640",
 "disk_root": "overlay         320G  9.4M  320G   1% /",
 "hostname": "97f2b223cf2f",
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
| e4b | fused_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.722 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 704.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 493.69 MiB is free. Including non-PyTorch memory, this process has 30.83 GiB memory in use. Of the al |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 6.504 | 204.8 | 30.469 | 1220.7 | 1.1853→0.6963 | 1.1962→0.6879 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| e4b | fused_attn4_m_d2 | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.722 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 704.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 493.69 MiB is free. Including non-PyTorch memory, this process has 30.83 GiB memory in use. Of the al |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 6.604 | 202.8 | 30.469 | 1167.6 | 1.1853→0.6963 | 1.1962→0.6878 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| unsloth | ckpt_unsloth_m_experts | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.358 | — | —→— | —→— | — | patched 0 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 878.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 417.69 MiB is free. Including non-PyTorch memory, this process has 30.91 GiB memory in use. Of the al |
| e4b | fused_attn4_m_mb1 | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.667 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 18: CUDA out of memory. Tried to allocate 690.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 665.69 MiB is free. Including non-PyTorch memory, this process has 30.67 GiB memory in use. Of the al |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 13.675 | 112.5 | 30.410 | 1841.0 | 1.1919→0.7325 | 1.1962→0.6870 | N-A (anchor missing) | stacks 80 / fwd 640 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
- prologue `unsloth/ckpt_unsloth_m` **86.7 s** before step 1 (35% of the arm): c1_before 27.6, eval0 26.6, load_weights 19.6, c1_after 14.1; unattributed 5.91; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **86.9 s** before step 1 (35% of the arm): c1_before 27.8, eval0 27.6, load_weights 18.5, c1_after 13.8; unattributed 5.929; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **67.2 s** before step 1 (19% of the arm): c1_before 27.4, load_weights 18.7, c1_after 14.2, eval0 7.8; unattributed 5.94; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is OOM); `unsloth/ckpt_unsloth_m` STABLE (6.504/6.604 s, |Δ|/mean 1.5% vs 5%); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m is OOM / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m is OOM / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m is OOM / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (unsloth (mb1))** — not quoted: e4b e4b/fused_attn4_m_mb1 is OOM / unsloth (mb1) single draw (no second draw registered)
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (hf (mb1))** — not quoted: e4b e4b/fused_attn4_m_mb1 is OOM / hf (mb1) no receipt
- **NO SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed) QUOTED (axolotl (mb1))** — not quoted: e4b e4b/fused_attn4_m_mb1 is OOM / axolotl (mb1) no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b e4b/fused_attn4_m is OOM / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b e4b/fused_attn4_m is OOM / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0001): `unsloth/ckpt_unsloth_m` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `e4b/fused_attn4_m_d2` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_d2` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_mb1` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_mb1`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects

### Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b's regime per arm, the footprint line when it ran under offload) (`mixtral`, registered n_layers 32, attention census 128)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.494 | 329.0 | 29.026 | 2130.7 | 1.4111→0.6950 | 1.4266→0.7138 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0026) | 20 | 3.017 | 447.7 | 29.115 | 1262.5 | 1.4108→0.6946 | 1.4291→0.7094 | -0.0043 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.543 | 332.9 | 29.028 | 1988.9 | 1.4111→0.6953 | 1.4266→0.7155 | 0.0018 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0026) | 20 | 2.960 | 548.7 | 29.142 | 1167.7 | 1.4108→0.6952 | 1.4291→0.7083 | -0.0054 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0051) | 20 | 4.694 | 380.1 | 28.205 | 1522.7 | 1.4148→0.6956 | 1.4215→0.7105 | -0.0032 | patched 0 / kcalls 0 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **103.2 s** before step 1 (49% of the arm): load_weights 46.9, c1_before 40.8, c1_after 20.8, eval0 5.6; unattributed 0.638; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **85.8 s** before step 1 (51% of the arm): c1_before 38.8, load_weights 24.7, c1_after 20.6, eval0 12.0; unattributed 6.411; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **65.7 s** before step 1 (38% of the arm): c1_before 38.3, c1_after 19.1, load_weights 15.0, attn4 4.7; unattributed 0.878; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **70.8 s** before step 1 (52% of the arm): c1_before 37.6, c1_after 18.5, load_weights 17.1, eval0 6.2; unattributed 6.145; budget 1260.0
- prologue `e4b/reference_attn4_m` **66.8 s** before step 1 (41% of the arm): c1_before 38.9, c1_after 19.6, load_weights 14.9, attn4 4.7; unattributed 0.748; budget 1890.0
- draws (R1): `e4b/fused_attn4_m` STABLE (5.494/5.543 s, |Δ|/mean 0.9% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (3.017/2.960 s, |Δ|/mean 1.9% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00066, median step |Δ| 0.00130 → **PASS** (band 0.05/0.05); ×0.85 faster per step, peak ×1.029
- **MATCHED POSITION: s/step ratio unsloth/e4b = 0.542** [0.534, 0.549 over 4 cross-draw ratios] (2.989 vs 5.519 s, medians over 2/2 draws; unsloth faster per step); peak VRAM unsloth 29.13 vs e4b 29.03 GB (Δ +0.10); J/step unsloth 1215.1 vs e4b 2059.8 (×0.590); tok/s unsloth 498.2 vs e4b 330.9; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256)
- quality reading at N=20 (unsloth): held-out e4b 0.7138 / unsloth 0.7094 (Δ -0.0043) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0026
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b STABLE / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b STABLE / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0050 / held-out 0.0097; COMPARABLE ≤ 0.05; draw-noise floor 0.0018): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0068, |Δ held-out at N| 0.0043, step-0 0.0026 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0733, paired rows mean -0.0043 ± 0.0027 SE over 8, favouring arm 5 / anchor 3); `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0018, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0001, paired rows mean +0.0018 ± 0.0012 SE over 8, favouring arm 2 / anchor 6) — |delta| 0.0018 / 0.0004 are narrower than the draw-noise floor 0.0018: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0063, |Δ held-out at N| 0.0054, step-0 0.0026 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0745, paired rows mean -0.0054 ± 0.0029 SE over 8, favouring arm 6 / anchor 2); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0013, |Δ held-out at N| 0.0032, step-0 0.0051 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0140, paired rows mean -0.0032 ± 0.0024 SE over 8, favouring arm 5 / anchor 3)
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/reference_attn4_m` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e809c37f09fd control detects, down nf4/64+dq sha 682dd5320a04 control detects, q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3_5 | e4b/fused_attn4_m | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 704.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 493.69 MiB is free. Including non-PyTorch memo |
| qwen3_5 | unsloth/ckpt_unsloth_m | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3_5 | e4b/fused_attn4_m_d2 | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 704.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 493.69 MiB is free. Including non-PyTorch memo |
| qwen3_5 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3_5 | unsloth/ckpt_unsloth_m_experts | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3_5 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/reference_attn4_m | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 878.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 417.69 MiB is free. Including non-PyTorch memo |
| qwen3_5 | e4b/fused_attn4_m_mb1 | **OOM** | — | — | OOM at step 18: CUDA out of memory. Tried to allocate 690.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 665.69 MiB is free. Including non-PyTorch mem |
| qwen3_5 | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | N-A (anchor missing) |  |
| mixtral | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| mixtral | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0043 |  |
| mixtral | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0018 |  |
| mixtral | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0054 |  |
| mixtral | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/reference_attn4_m | **VALID** | VALID | -0.0032 |  |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **UNTESTED** | unsloth/ckpt_unsloth_m_experts NOT_RUN; hf/hf_peft_m NOT_RUN; e4b fused_m not usable: e4b/fused_attn4_m is OOM |
| P5 | mixtral | **UNTESTED** | e4b's anchor ran RESIDENT: P5 is the offload pair's prediction (TC2 amendment 6 scores a resident box) |
| P6 | tc2 | **HELD** | mixtral PASS (Δfinal 0.00066) |
| P7 | tc2 | **FALSIFIED** | mixtral unsloth/ckpt_unsloth_m COMPARABLE; mixtral unsloth/ckpt_unsloth_m_d2 COMPARABLE |
