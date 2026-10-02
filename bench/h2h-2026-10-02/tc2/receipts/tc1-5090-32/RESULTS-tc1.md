# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.40.0 @3e41f0f2d1e6e20bcbf25711dde087deec97a563 (GitHub main)
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
e4b(t212) 0.40.0 @3e41f0f2d1e6e20bcbf25711dde087deec97a563
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
 "run_id": "tc1-5090-32",
 "instance_id": "53908726",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "fd982f00013f",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC2 (`tc2small` / `tc2big` tokens, TC2-PREREG.md, drafted in TC2-PREREG-draft): TC1's readings per family with the registered n_layers {'granite': 32, 'olmoe': 16, 'gptoss': 24, 'qwen3_5': 40, 'mixtral': 32} and attention census {'granite': 128, 'olmoe': 64, 'gptoss': None, 'qwen3_5': None, 'mixtral': 128} (None = the receipt's own structural census governs); the HF position is quoted as HF (bf16 experts) / e4b (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID (attention-only when it adapted no expert parameter), as tp4; gpt-oss carries a NO COMMON ADAPTER SET line (both s/step values, both trainable counts), never a ratio; mixtral's FOOTPRINT line (e4b under expert offload vs Unsloth resident: peak VRAM and s/step) leads its block; `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert parameters of a recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime; an HF t214 arm whose dispatch did not reach grouped_mm is recorded, never VOID; P1–P7 of the draft scored HELD / FALSIFIED / UNTESTED (P1 HF band (1.1, 1.6); P2 Unsloth (1.2, 3.0) / HF (1.5, 2.5); P4 Unsloth (2.0, 6.0), e4b within 15% of tp4's 6.3344 s/step; P5 e4b/Unsloth (0.3, 0.5) at a >= 8x lower e4b peak, tp2 0.361).

### Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b under expert offload, Unsloth resident) (`mixtral`, registered n_layers 32, attention census 128)
- **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): e4b `fused_attn4_m` peak VRAM 7.16 GB under expert offload vs Unsloth `ckpt_unsloth_m` 29.13 GB resident (×4.07 lower on e4b); s/step e4b 19.751 vs Unsloth 3.741 (e4b/Unsloth 5.279)** — e4b: single receipt, verdict VALID (not a quoted draw); Unsloth: quoted draws (STABLE)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 19.751 | 88.0 | 7.164 | 3495.3 | 1.4102→0.6959 | 1.4297→0.7147 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0020) | 20 | 3.735 | 338.4 | 29.115 | 1632.8 | 1.4108→0.6938 | 1.4277→0.7086 | -0.0060 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 21.181 | 84.5 | 7.145 | 3465.7 | 1.4102→0.6944 | 1.4297→0.7138 | -0.0009 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0020) | 20 | 3.748 | 408.5 | 29.142 | 1546.2 | 1.4108→0.6936 | 1.4277→0.7095 | -0.0052 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0037) | 20 | 20.618 | 89.0 | 10.595 | 3043.4 | 1.4199→0.6958 | 1.4259→0.7135 | -0.0012 | patched 0 / kcalls 0 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **176.4 s** before step 1 (30% of the arm): c1_before 74.1, load_weights 64.8, c1_after 37.3, eval0 20.5; unattributed 1.376; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m` **158.1 s** before step 1 (60% of the arm): c1_before 88.0, c1_after 41.7, load_weights 34.9, eval0 19.1; unattributed 9.751; budget 2520.0
- prologue `e4b/fused_attn4_m_d2` **180.1 s** before step 1 (29% of the arm): c1_before 83.3, load_weights 63.7, c1_after 38.9, eval0 16.2; unattributed 1.259; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_d2` **150.8 s** before step 1 (63% of the arm): c1_before 95.8, c1_after 46.1, load_weights 28.1, eval0 10.5; unattributed 9.923; budget 2520.0
- prologue `e4b/reference_attn4_m` **181.8 s** before step 1 (30% of the arm): c1_before 81.9, load_weights 65.2, c1_after 38.7, eval0 17.2; unattributed 1.451; budget 2100.0
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (19.751/21.181 s, |Δ|/mean 7.0% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (3.735/3.748 s, |Δ|/mean 0.3% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00010, median step |Δ| 0.00155 → **PASS** (band 0.05/0.05); ×1.04 faster per step, peak ×0.676
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 19.751 / 21.181 s/step differ by 7.0% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 19.751 / 21.181 s/step differ by 7.0% > 5% (UNSTABLE: reported, not quoted) / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 19.751 / 21.181 s/step differ by 7.0% > 5% (UNSTABLE: reported, not quoted) / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b draws 19.751 / 21.181 s/step differ by 7.0% > 5% (UNSTABLE: reported, not quoted) / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b draws 19.751 / 21.181 s/step differ by 7.0% > 5% (UNSTABLE: reported, not quoted) / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0050 / held-out 0.0050; COMPARABLE ≤ 0.05; draw-noise floor 0.0009): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0054, |Δ held-out at N| 0.0060, step-0 0.0020 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0670, paired rows mean -0.0060 ± 0.0021 SE over 8, favouring arm 7 / anchor 1); `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0007, |Δ held-out at N| 0.0009, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0020, paired rows mean -0.0009 ± 0.0012 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0009 / 0.0007 are narrower than the draw-noise floor 0.0009: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0064, |Δ held-out at N| 0.0052, step-0 0.0020 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0603, paired rows mean -0.0052 ± 0.0029 SE over 8, favouring arm 5 / anchor 3); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0015, |Δ held-out at N| 0.0012, step-0 0.0037 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0082, paired rows mean -0.0012 ± 0.0016 SE over 8, favouring arm 5 / anchor 3)
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/reference_attn4_m` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **N-A** (slot missing on both) control FAILS, down **N-A** (slot missing on both) control FAILS, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **N-A** (slot missing on both) control FAILS, down **N-A** (slot missing on both) control FAILS, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| mixtral | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| mixtral | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0060 |  |
| mixtral | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0009 |  |
| mixtral | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0052 |  |
| mixtral | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/reference_attn4_m | **VALID** | VALID | -0.0012 |  |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **UNTESTED** | no receipts |
| P5 | mixtral | **UNTESTED** | mixtral position not quoted: not quoted: e4b draws 19.751 / 21.181 s/step differ by 7.0% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE (footprint: **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): e4b `fused_attn4_m` peak VR); hf/hf_peft_m NOT_RUN; axolotl/ckpt_axolotl_m NOT_RUN |
| P6 | tc2 | **HELD** | mixtral PASS (Δfinal 0.00010) |
| P7 | tc2 | **FALSIFIED** | mixtral unsloth/ckpt_unsloth_m COMPARABLE; mixtral unsloth/ckpt_unsloth_m_d2 COMPARABLE |
