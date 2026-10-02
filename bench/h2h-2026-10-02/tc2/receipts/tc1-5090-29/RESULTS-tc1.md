# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.40.0 @baa536bb6b980881b0a5b9425a20b464f27acaeb (GitHub main)
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
e4b(t212) 0.40.0 @baa536bb6b980881b0a5b9425a20b464f27acaeb
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
 "run_id": "tc1-5090-29",
 "instance_id": "53896301",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "d30a46bcc4ec",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```
Lane TC2 (`tc2small` / `tc2big` tokens, TC2-PREREG.md, drafted in TC2-PREREG-draft): TC1's readings per family with the registered n_layers {'granite': 32, 'olmoe': 16, 'gptoss': 24, 'qwen3_5': 40, 'mixtral': 32} and attention census {'granite': 128, 'olmoe': 64, 'gptoss': None, 'qwen3_5': None, 'mixtral': 128} (None = the receipt's own structural census governs); the HF position is quoted as HF (bf16 experts) / e4b (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID (attention-only when it adapted no expert parameter), as tp4; gpt-oss carries a NO COMMON ADAPTER SET line (both s/step values, both trainable counts), never a ratio; mixtral's FOOTPRINT line (e4b under expert offload vs Unsloth resident: peak VRAM and s/step) leads its block; `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert parameters of a recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime; an HF t214 arm whose dispatch did not reach grouped_mm is recorded, never VOID; P1–P7 of the draft scored HELD / FALSIFIED / UNTESTED (P1 HF band (1.1, 1.6); P2 Unsloth (1.2, 3.0) / HF (1.5, 2.5); P4 Unsloth (2.0, 6.0), e4b within 15% of tp4's 6.3344 s/step; P5 e4b/Unsloth (0.3, 0.5) at a >= 8x lower e4b peak, tp2 0.361).

### Qwen3.6-35B-A3B (lane TC2, box B) (`qwen3_5`, registered n_layers 40)
- **FOOTPRINT (e4b (offload) vs Unsloth (resident)): e4b `fused_attn4_m` peak VRAM 19.35 GB under expert offload vs Unsloth `ckpt_unsloth_m` 30.47 GB resident (×1.57 lower on e4b); s/step e4b 12.372 vs Unsloth 10.107 (e4b/Unsloth 1.224)** — e4b: single receipt, verdict VALID (not a quoted draw); Unsloth: single receipt, verdict VOID (not a quoted draw)
- model `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd9`; tokens sha `0d94e9b88936`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 926187520; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 12.372 | 126.3 | 19.347 | 1964.1 | 1.1365→0.6895 | 1.1433→0.6834 | 0.0000 | patched 40 / kcalls 640 | 926187520 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VOID | **VOID** | yes | matched:3407 (complete 20520/20520) / float32 | VOID (0.0502) | 20 | 10.107 | 121.9 | 30.469 | 1717.5 | 1.1925→0.6975 | 1.1935→0.6868 | 0.0034 | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) | step-0 held-out 1.1935 differs from the anchor's 1.1433 by 0.0502 > 0.05 |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 13.846 | 117.5 | 19.327 | 2010.0 | 1.1365→0.6914 | 1.1433→0.6831 | -0.0003 | patched 40 / kcalls 640 | 926187520 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 20520/20520) / float32 | VOID (0.0529) | 20 | 10.618 | 123.7 | 30.469 | 1706.6 | 1.1853→0.6982 | 1.1962→0.6904 | 0.0069 | stacks 80 / fwd 320 / u8 40 | 926187520 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 80, Linear4bit 498) | step-0 held-out 1.1962 differs from the anchor's 1.1433 by 0.0529 > 0.05 |
| unsloth | ckpt_unsloth_m_experts | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | stacks — / fwd — / u8 — | — | — | no receipt and no attempt line |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_best | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_shipped | **NOT_RUN** | — | **NOT_RUN** | native | — / — | — | 20 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 20520/20520) / float32 | SAME-BYTES-CLASS (0.0052) | 20 | 110.570 | 14.6 | 21.178 | 9661.6 | 1.1370→0.6908 | 1.1381→0.6822 | -0.0012 | patched 0 / kcalls 0 | 926187520 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **158.1 s** before step 1 (38% of the arm): load_weights 58.8, c1_before 55.6, c1_after 27.7, eval0 24.1; unattributed 1.197; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m` **167.8 s** before step 1 (39% of the arm): c1_before 58.0, eval0 46.3, load_weights 41.4, c1_after 32.9; unattributed 9.774; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **145.1 s** before step 1 (34% of the arm): load_weights 61.9, c1_before 52.7, c1_after 25.8, eval0 9.9; unattributed 1.469; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_d2` **152.2 s** before step 1 (37% of the arm): c1_before 58.5, eval0 39.3, load_weights 32.1, c1_after 28.9; unattributed 9.836; budget 1260.0
- prologue `e4b/reference_attn4_m` **169.9 s** before step 1 (7% of the arm): load_weights 62.3, c1_before 51.0, eval0 37.9, c1_after 25.2; unattributed 1.247; budget 2520.0
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (12.372/13.846 s, |Δ|/mean 11.2% vs 5%); `unsloth/ckpt_unsloth_m` — (unsloth/ckpt_unsloth_m is VOID); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00133, median step |Δ| 0.00159 → **PASS** (band 0.05/0.05); ×8.94 faster per step, peak ×0.914
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 12.372 / 13.846 s/step differ by 11.2% > 5% (UNSTABLE: reported, not quoted) / unsloth unsloth/ckpt_unsloth_m is VOID
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 12.372 / 13.846 s/step differ by 11.2% > 5% (UNSTABLE: reported, not quoted) / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 12.372 / 13.846 s/step differ by 11.2% > 5% (UNSTABLE: reported, not quoted) / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (unsloth native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / unsloth native-best vs e4b shipped no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped is NOT_RUN / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW ckpt_axolotl_best vs e4b/fused_attn4_m (never the quoted position) QUOTED (axolotl native-best (KernelsPlugin scattermoe))** — not quoted: e4b draws 12.372 / 13.846 s/step differ by 11.2% > 5% (UNSTABLE: reported, not quoted) / axolotl native-best (KernelsPlugin scattermoe) axolotl/ckpt_axolotl_best is NOT_RUN
- **NO LABELLED ROW fused_attn4_shipped vs e4b/fused_attn4_m (never the quoted position) QUOTED (e4b shipped (bf16 expert adapters, N(0,1/r) init))** — not quoted: e4b draws 12.372 / 13.846 s/step differ by 11.2% > 5% (UNSTABLE: reported, not quoted) / e4b shipped (bf16 expert adapters, N(0,1/r) init) e4b/fused_attn4_shipped is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0050 / held-out 0.0050; COMPARABLE ≤ 0.05; draw-noise floor —): `unsloth/ckpt_unsloth_m` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_d2` **EQUIVALENT** (median step |Δ| 0.0007, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0024, paired rows mean -0.0003 ± 0.0027 SE over 8, favouring arm 5 / anchor 3); `unsloth/ckpt_unsloth_m_d2` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0016, |Δ held-out at N| 0.0012, step-0 0.0052 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0002, paired rows mean -0.0012 ± 0.0012 SE over 8, favouring arm 6 / anchor 2)
- matched_init_sha (B, name-free, canonical slot order): anchor `4b90bf05bd02baf3`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/reference_attn4_m` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: q_proj nf4/64+dq sha b44f6f975053 control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **N-A** (slot missing on both) control FAILS, down **N-A** (slot missing on both) control FAILS, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (slot missing on the anchor) control detects, down **N-A** (slot missing on the anchor) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **N-A** (slot missing on both) control FAILS, down **N-A** (slot missing on both) control FAILS, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3_5 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3_5 | unsloth/ckpt_unsloth_m | **VOID** | VOID | 0.0034 | step-0 held-out 1.1935 differs from the anchor's 1.1433 by 0.0502 > 0.05 |
| qwen3_5 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0003 |  |
| qwen3_5 | unsloth/ckpt_unsloth_m_d2 | **VOID** | VOID | 0.0069 | step-0 held-out 1.1962 differs from the anchor's 1.1433 by 0.0529 > 0.05 |
| qwen3_5 | unsloth/ckpt_unsloth_m_experts | **NOT_RUN** | — | — | no receipt and no attempt line |
| qwen3_5 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | axolotl/ckpt_axolotl_best | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/fused_attn4_shipped | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3_5 | e4b/reference_attn4_m | **VALID** | VALID | -0.0012 |  |

## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P2 | olmoe | **UNTESTED** | no receipts |
| P3 | gptoss | **UNTESTED** | no receipts |
| P4 | qwen3_5 | **UNTESTED** | unsloth/ckpt_unsloth_m_experts NOT_RUN; hf/hf_peft_m NOT_RUN; e4b fused_m not usable: draws 12.372 / 13.846 s/step differ by 11.2% > 5% (UNSTABLE: reported, not quoted) |
| P5 | mixtral | **UNTESTED** | no receipts |
| P6 | tc2 | **HELD** | qwen3_5 PASS (Δfinal 0.00133) |
| P7 | tc2 | **UNTESTED** | no VALID matched pair sharing e4b's adapter set |
