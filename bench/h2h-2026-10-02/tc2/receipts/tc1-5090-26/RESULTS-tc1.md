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
 "run_id": "tc1-5090-26",
 "instance_id": "53859501",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "1056596464",
 "cgroup_memory_max": "367227043840",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "e55939d24343",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC2 (`tc2small` / `tc2big` tokens, TC2-PREREG.md, drafted in TC2-PREREG-draft): TC1's readings per family with the registered n_layers {'granite': 32, 'olmoe': 16, 'gptoss': 24, 'qwen3_5': 40, 'mixtral': 32} and attention census {'granite': 128, 'olmoe': 64, 'gptoss': None, 'qwen3_5': None, 'mixtral': 128} (None = the receipt's own structural census governs); the HF position is quoted as HF (bf16 experts) / e4b (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID (attention-only when it adapted no expert parameter), as tp4; gpt-oss carries a NO COMMON ADAPTER SET line (both s/step values, both trainable counts), never a ratio; mixtral's FOOTPRINT line (e4b under expert offload vs Unsloth resident: peak VRAM and s/step) leads its block; `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert parameters of a recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime; an HF t214 arm whose dispatch did not reach grouped_mm is recorded, never VOID; P1–P7 of the draft scored HELD / FALSIFIED / UNTESTED (P1 HF band (1.1, 1.6); P2 Unsloth (1.2, 3.0) / HF (1.5, 2.5); P4 Unsloth (2.0, 6.0), e4b within 15% of tp4's 6.3344 s/step; P5 e4b/Unsloth (0.3, 0.5) at a >= 8x lower e4b peak, tp2 0.361).

### Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b under expert offload, Unsloth resident) (`mixtral`, registered n_layers 32, attention census 128)
- **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): e4b `fused_attn4_m` peak VRAM 7.15 GB under expert offload vs Unsloth `ckpt_unsloth_m` 31.95 GB resident (×4.47 lower on e4b); s/step e4b 15.805 vs Unsloth 6.213 (e4b/Unsloth 2.544)** — e4b: quoted draws (STABLE); Unsloth: single receipt, verdict VALID (not a quoted draw)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 15.891 | 105.2 | 7.151 | 2984.0 | 1.4102→0.6961 | 1.4297→0.7142 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0029) | 20 | 6.213 | 6.6 | 31.951 | 11563.7 | 1.4166→0.6957 | 1.4326→0.7107 | -0.0035 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 15.719 | 107.8 | 7.143 | 2968.1 | 1.4102→0.6960 | 1.4297→0.7132 | -0.0010 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0029) | 20 | 7.011 | 7.6 | 32.239 | 10218.4 | 1.4166→0.6962 | 1.4326→0.7089 | -0.0053 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0037) | 20 | 16.079 | 109.4 | 10.595 | 2414.4 | 1.4199→0.6941 | 1.4259→0.7134 | -0.0008 | patched 0 / kcalls 0 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **323.6 s** before step 1 (48% of the arm): c1_before 157.0, load_weights 119.4, c1_after 56.3, eval0 20.3; unattributed 1.559; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m` **698.3 s** before step 1 (11% of the arm): eval0 555.0, c1_before 94.1, c1_after 49.9, load_weights 28.5; unattributed 11.603; budget 2520.0
- prologue `e4b/fused_attn4_m_d2` **213.4 s** before step 1 (39% of the arm): c1_before 100.5, load_weights 76.6, c1_after 46.7, attn4 13.8; unattributed 1.762; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_d2` **480.7 s** before step 1 (9% of the arm): eval0 336.0, c1_before 93.7, c1_after 49.5, load_weights 32.3; unattributed 10.851; budget 2520.0
- prologue `e4b/reference_attn4_m` **212.4 s** before step 1 (39% of the arm): c1_before 102.4, load_weights 77.3, c1_after 50.6, eval0 13.0; unattributed 1.581; budget 2100.0
- draws (R1): `e4b/fused_attn4_m` STABLE (15.891/15.719 s, |Δ|/mean 1.1% vs 5%); `unsloth/ckpt_unsloth_m` UNSTABLE (6.213/7.011 s, |Δ|/mean 12.1% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00195, median step |Δ| 0.00166 → **PASS** (band 0.05/0.05); ×1.01 faster per step, peak ×0.675
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth draws 6.213 / 7.011 s/step differ by 12.1% > 5% (UNSTABLE: reported, not quoted)
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b STABLE / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b STABLE / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0050 / held-out 0.0050; COMPARABLE ≤ 0.05; draw-noise floor 0.0010): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0036, |Δ held-out at N| 0.0035, step-0 0.0029 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0588, paired rows mean -0.0035 ± 0.0030 SE over 8, favouring arm 6 / anchor 2); `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0005, |Δ held-out at N| 0.0010, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0051, paired rows mean -0.0010 ± 0.0008 SE over 8, favouring arm 6 / anchor 2) — |delta| 0.0010 / 0.0005 are narrower than the draw-noise floor 0.0010: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0051, |Δ held-out at N| 0.0053, step-0 0.0029 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0671, paired rows mean -0.0053 ± 0.0030 SE over 8, favouring arm 6 / anchor 2); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0017, |Δ held-out at N| 0.0008, step-0 0.0037 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0067, paired rows mean -0.0007 ± 0.0015 SE over 8, favouring arm 6 / anchor 2)
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
| mixtral | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0035 |  |
| mixtral | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0010 |  |
| mixtral | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0053 |  |
| mixtral | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtral | e4b/reference_attn4_m | **VALID** | VALID | -0.0008 |  |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **UNTESTED** | no receipts |
| P5 | mixtral | **UNTESTED** | mixtral position not quoted: not quoted: e4b STABLE / unsloth draws 6.213 / 7.011 s/step differ by 12.1% > 5% (UNSTABLE: reported, not quoted) (footprint: **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): e4b `fused_attn4_m` peak VR); hf/hf_peft_m NOT_RUN; axolotl/ckpt_axolotl_m NOT_RUN |
| P6 | tc2 | **HELD** | mixtral PASS (Δfinal 0.00195) |
| P7 | tc2 | **FALSIFIED** | mixtral unsloth/ckpt_unsloth_m EQUIVALENT; mixtral unsloth/ckpt_unsloth_m_d2 COMPARABLE |
