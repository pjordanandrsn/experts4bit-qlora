# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.44.0 @4ea0b483114ff0b216ffa8c1a4e5af4d22f4cdd0 (GitHub main)
gnf4 0.36.0 @34a088122346cc8fedadd55461a8eb6d646fb481 (GitHub main)
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
e4b(t212) 0.44.0 @4ea0b483114ff0b216ffa8c1a4e5af4d22f4cdd0
gnf4(t212) 0.36.0 @34a088122346cc8fedadd55461a8eb6d646fb481
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
 "run_id": "tc1-5090-53",
 "instance_id": "54140024",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "1056596104",
 "cgroup_memory_max": "367227043840",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "dfc446d50277",
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
| e4b | fused_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.545 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 17.69 MiB is free. Including non-PyTorch memory, this process has 31.33 GiB memory in use. Of the allo |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 11.658 | 106.2 | 30.469 | 1614.2 | 1.1853→0.6979 | 1.1962→0.6893 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| e4b | fused_attn4_m_d2 | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.545 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 17.69 MiB is free. Including non-PyTorch memory, this process has 31.33 GiB memory in use. Of the allo |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 11.888 | 113.8 | 30.469 | 1551.3 | 1.1853→0.6961 | 1.1962→0.6897 | N-A (anchor missing) | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
| unsloth | ckpt_unsloth_m_experts | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.837 | — | —→— | —→— | — | patched 0 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 440.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 357.69 MiB is free. Including non-PyTorch memory, this process has 31.00 GiB memory in use. Of the al |
| e4b | fused_attn4_m_mb1 | **OOM** | — | **OOM** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | — | — | 32.680 | — | —→— | —→— | — | patched 40 / kcalls — | 926187520 | — | OOM at step 2: CUDA out of memory. Tried to allocate 26.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 11.62 MiB is free. Including non-PyTorch memory, this process has 31.34 GiB memory in use. Of the allo |
| unsloth | ckpt_unsloth_m_mb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | — | 20 | 19.014 | 78.2 | 30.410 | 2652.5 | 1.1919→0.7326 | 1.1962→0.6892 | N-A (anchor missing) | stacks 80 / fwd 640 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) |  |
- prologue `unsloth/ckpt_unsloth_m` **198.2 s** before step 1 (40% of the arm): c1_before 65.9, load_weights 56.9, eval0 49.4, c1_after 32.0; unattributed 12.267; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **171.9 s** before step 1 (38% of the arm): c1_before 65.5, eval0 43.4, load_weights 37.4, c1_after 33.2; unattributed 12.055; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_mb1` **136.9 s** before step 1 (25% of the arm): c1_before 63.6, c1_after 31.5, load_weights 30.9, eval0 14.7; unattributed 10.39; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is OOM); `unsloth/ckpt_unsloth_m` STABLE (11.658/11.888 s, |Δ|/mean 2.0% vs 5%); `unsloth/ckpt_unsloth_m_mb1` SINGLE (single draw (no second draw registered))
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
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0004): `unsloth/ckpt_unsloth_m` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `e4b/fused_attn4_m_d2` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_d2` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_mb1` **—** — no OK receipt; `unsloth/ckpt_unsloth_m_mb1` **N-A** — the anchor e4b/fused_attn4_m is missing or not OK
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects
- frozen base `unsloth/ckpt_unsloth_m_mb1`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **N-A** (slot missing on the anchor) control detects

### Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b's regime per arm, the footprint line when it ran under offload) (`mixtral`, registered n_layers 32, attention census 128)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.885 | 294.9 | 31.066 | 2055.4 | 1.4112→0.6948 | 1.4313→0.7113 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0022) | 20 | 4.024 | 314.3 | 29.115 | 1366.8 | 1.4108→0.6935 | 1.4291→0.7097 | -0.0016 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.923 | 299.7 | 31.075 | 2031.7 | 1.4112→0.6964 | 1.4313→0.7145 | 0.0032 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0022) | 20 | 4.203 | 375.0 | 29.142 | 1284.3 | 1.4108→0.6947 | 1.4291→0.7092 | -0.0021 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0054) | 20 | 6.719 | 264.8 | 30.307 | 1640.5 | 1.4199→0.6950 | 1.4259→0.7135 | 0.0022 | patched 0 / kcalls 0 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **184.3 s** before step 1 (60% of the arm): c1_before 108.1, c1_after 52.0, load_weights 45.3, adapter_save 15.8; unattributed 1.478; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **196.7 s** before step 1 (63% of the arm): c1_before 104.6, load_weights 50.7, c1_after 50.6, eval0 21.8; unattributed 12.04; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **169.8 s** before step 1 (59% of the arm): c1_before 103.3, c1_after 55.3, load_weights 39.7, attn4 12.5; unattributed 1.657; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **154.9 s** before step 1 (62% of the arm): c1_before 90.2, c1_after 49.5, load_weights 36.5, adapter_save 17.7; unattributed 12.031; budget 1260.0
- prologue `e4b/reference_attn4_m` **182.3 s** before step 1 (57% of the arm): c1_before 104.8, c1_after 55.5, load_weights 51.7, attn4 10.3; unattributed 1.921; budget 1890.0
- draws (R1): `e4b/fused_attn4_m` STABLE (5.885/5.923 s, |Δ|/mean 0.6% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (4.024/4.203 s, |Δ|/mean 4.3% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00027, median step |Δ| 0.00194 → **PASS** (band 0.05/0.05); ×1.14 faster per step, peak ×1.025
- **MATCHED POSITION: s/step ratio unsloth/e4b = 0.697** [0.679, 0.714 over 4 cross-draw ratios] (4.114 vs 5.904 s, medians over 2/2 draws; unsloth faster per step); peak VRAM unsloth 29.13 vs e4b 31.07 GB (Δ -1.94); J/step unsloth 1325.6 vs e4b 2043.6 (×0.649); tok/s unsloth 344.6 vs e4b 297.3; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256)
- quality reading at N=20 (unsloth): held-out e4b 0.7113 / unsloth 0.7097 (Δ -0.0016) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0022
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b STABLE / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b STABLE / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0058 / held-out 0.0067; COMPARABLE ≤ 0.05; draw-noise floor 0.0032): `unsloth/ckpt_unsloth_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0027, |Δ held-out at N| 0.0016, step-0 0.0022 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0708, paired rows mean -0.0016 ± 0.0038 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0016 / 0.0027 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0032, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0029, paired rows mean +0.0032 ± 0.0015 SE over 8, favouring arm 2 / anchor 6) — |delta| 0.0032 / 0.0017 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0027, |Δ held-out at N| 0.0021, step-0 0.0022 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0686, paired rows mean -0.0021 ± 0.0033 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0021 / 0.0027 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0019, |Δ held-out at N| 0.0022, step-0 0.0054 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0050, paired rows mean +0.0022 ± 0.0026 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0022 / 0.0019 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/reference_attn4_m` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha c56e6fc0fd22 control detects, down nf4/64 sha 7e63cab94f9c control detects, q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3_5 | e4b/fused_attn4_m | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 17.69 MiB is free. Including non-PyTorch memory |
| qwen3_5 | unsloth/ckpt_unsloth_m | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3_5 | e4b/fused_attn4_m_d2 | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 17.69 MiB is free. Including non-PyTorch memory |
| qwen3_5 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | N-A (anchor missing) |  |
| qwen3_5 | unsloth/ckpt_unsloth_m_experts | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3_5 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/reference_attn4_m | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 440.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 357.69 MiB is free. Including non-PyTorch memo |
| qwen3_5 | e4b/fused_attn4_m_mb1 | **OOM** | — | — | OOM at step 2: CUDA out of memory. Tried to allocate 26.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 11.62 MiB is free. Including non-PyTorch memory |
| qwen3_5 | unsloth/ckpt_unsloth_m_mb1 | **VALID** | VALID | N-A (anchor missing) |  |
| mixtral | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| mixtral | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0016 |  |
| mixtral | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0032 |  |
| mixtral | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0021 |  |
| mixtral | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/reference_attn4_m | **VALID** | VALID | 0.0022 |  |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **UNTESTED** | unsloth/ckpt_unsloth_m_experts NOT_RUN; hf/hf_peft_m NOT_RUN; e4b fused_m not usable: e4b/fused_attn4_m is OOM |
| P5 | mixtral | **UNTESTED** | e4b's anchor ran RESIDENT: P5 is the offload pair's prediction (TC2 amendment 6 scores a resident box) |
| P6 | tc2 | **HELD** | mixtral PASS (Δfinal 0.00027) |
| P7 | tc2 | **HELD** | mixtral unsloth/ckpt_unsloth_m INSIDE-DRAW-NOISE; mixtral unsloth/ckpt_unsloth_m_d2 INSIDE-DRAW-NOISE |
