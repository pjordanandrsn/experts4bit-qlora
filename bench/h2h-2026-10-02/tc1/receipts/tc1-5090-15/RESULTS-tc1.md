# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.38.0 @079a4220b4b334903b0e77725eb94a0f58848018 (GitHub main)
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
e4b(t212) 0.38.0 @079a4220b4b334903b0e77725eb94a0f58848018
gnf4(t212) 0.34.0 @846b512b905468c08f5748943d08769b572affa2
torch(e4b-t212) 2.12.1+cu130
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1-5090-15",
 "instance_id": "53772765",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "13th Gen Intel(R) Core(TM) i9-13900",
 "nproc": 32,
 "mem_total_kb": "131726324",
 "cgroup_memory_max": "129490747392",
 "disk_root": "overlay         320G  1.7M  320G   1% /",
 "hostname": "ce06413975ef",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md"
}
```
Lane TC1b (`qwen3curve`, TC1B-PREREG.md): validity per sub-fixture against its own e4b arm; the curve reading EQUIVALENT-AT-EVERY-EVAL iff every paired held-out |Δ| <= 0.02 (the draft's band, kept fixed by the registration (0.02 nats at every eval: this lane has no in-draw reference arm); TC1's floor is quoted beside it), else DIVERGENT with the first divergent step; plateau REPRODUCES-P38 iff >= +0.01 at 200 and <= 0 at 40; time to target = the matched pair's step-200 held-out + 0.02; s/step 11..200 vs TC1's 11..20 within 10% = TRAVELS; anchor within ±10% of tp2 1.457 / P38 1.413; the t1 and r64 pairs are SCALING POINTS, never positions; P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED.

### Qwen3-30B-A3B (lane TC1b: the 200-step curve box, the anchor pair, the t1 and r64 scaling pairs) (`qwen3curve`, registered n_layers 48)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; sub-fixtures (each with its OWN e4b arm as the trainable / tokens / step-0 / sha reference): `200` anchor `e4b/fused_attn4_m_200` OK tokens `f07accfe6fcf` trainable 642514944; `p38` anchor `e4b/fused_attn4_p38` OK tokens `cb4f505d61ef` trainable 321257472; `t1` anchor `e4b/fused_attn4_m_t1` OK tokens `f07accfe6fcf` trainable 642514944; `r64` anchor `e4b/fused_attn4_m_r64` not OK tokens `` trainable None; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_200 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 200 | 3.162 | 493.6 | 29.116 | 1163.6 | 2.0705→0.7379 | 1.9607→0.7687 | anchor of its sub-fixture | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_200 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0004) | 200 | 4.505 | 346.2 | 24.344 | 861.9 | 2.0705→0.7330 | 1.9604→0.7688 | 0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_shipped_200 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 200 | 2.527 | 618.2 | 25.514 | 796.2 | 2.0705→0.6063 | 1.9607→0.7945 | 0.0258 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_p38 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 0.451 | 188.8 | 21.371 | 101.6 | 3.6282→0.2624 | 3.7468→0.2847 | anchor of its sub-fixture | patched 48 / kcalls 192 | 321257472 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_p38 | **OK** | VALID | **VALID** | native | native / float32 | — | 60 | 1.095 | 74.0 | 23.191 | 185.6 | 3.6115→0.2665 | 3.7606→0.3053 | 0.0206 | stacks 96 / fwd 96 / u8 48 | 321257472 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_p38_t28 | **OK** | VALID | **VALID** | native | native / float32 | — | 60 | 2.322 | 35.5 | 23.141 | 203.4 | 3.6550→0.2575 | 3.7502→0.2900 | 0.0053 | stacks 96 / fwd 96 / u8 48 | 321257472 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_t1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 0.559 | 267.2 | 25.755 | 121.6 | 3.3024→0.9749 | 1.9505→0.9242 | anchor of its sub-fixture | patched 48 / kcalls 192 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_t1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0054) | 20 | 1.073 | 132.0 | 24.093 | 184.7 | 3.4177→0.9640 | 1.9558→0.9275 | 0.0033 | stacks 96 / fwd 96 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_r64 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 31.876 | — | —→— | —→— | — | patched 48 / kcalls — | 2570059776 | — | OOM at step 1: CUDA out of memory. Tried to allocate 52.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 31.69 MiB is free. Including non-PyTorch memory, this process has 31.32 GiB memory in use. Of the allo |
| unsloth | ckpt_unsloth_m_r64 | **OOM** | — | **OOM** | yes | matched:3407 (complete 12480/12480) / float32 | — | 20 | — | — | 32.384 | — | —→— | —→— | — | stacks 96 / fwd — / u8 — | 2570059776 | — | OOM at step 1: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 753.69 MiB is free. Including non-PyTorch memory, this process has 30.61 GiB memory in use. Of the al |
- prologue `e4b/fused_attn4_m_200` **60.8 s** before step 1 (9% of the arm): c1_before 29.2, c1_after 14.6, load_weights 12.6, eval0 9.3; unattributed 0.604; budget 1680.0
- prologue `unsloth/ckpt_unsloth_m_200` **60.1 s** before step 1 (6% of the arm): c1_before 26.3, c1_after 13.2, load_weights 12.9, eval0 11.1; unattributed 4.93; budget 3150.0
- prologue `e4b/fused_attn4_shipped_200` **52.9 s** before step 1 (9% of the arm): c1_before 28.9, c1_after 14.5, load_weights 12.8, attn4 2.9; unattributed 0.613; budget 1680.0
- prologue `e4b/fused_attn4_p38` **48.0 s** before step 1 (61% of the arm): c1_before 28.3, c1_after 14.6, load_weights 10.9, preamble 2.9; unattributed 0.615; budget 630.0
- prologue `unsloth/ckpt_unsloth_p38` **52.2 s** before step 1 (40% of the arm): c1_before 26.6, c1_after 13.2, load_weights 11.8, eval0 5.7; unattributed 4.877; budget 630.0
- prologue `unsloth/ckpt_unsloth_p38_t28` **72.8 s** before step 1 (31% of the arm): c1_before 26.6, eval0 26.3, c1_after 13.1, load_weights 11.6; unattributed 5.072; budget 630.0
- prologue `e4b/fused_attn4_m_t1` **53.0 s** before step 1 (81% of the arm): c1_before 29.2, c1_after 14.5, load_weights 12.8, attn4 2.9; unattributed 0.618; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_t1` **54.9 s** before step 1 (68% of the arm): c1_before 26.5, c1_after 13.2, load_weights 13.1, eval0 5.4; unattributed 4.954; budget 1260.0
- **(a) curve table** (paired held-out mean ± SE over the rows at every eval; Δ(2−1) = `ckpt_unsloth_m_200` − `fused_attn4_m_200`, Δ(3−1) = `fused_attn4_shipped_200` − `fused_attn4_m_200`; a non-VALID arm's column reads —):
| eval step | e4b `fused_attn4_m_200` | unsloth `ckpt_unsloth_m_200` | e4b `fused_attn4_shipped_200` | paired Δ(2−1) ± SE | paired Δ(3−1) ± SE |
|---|---|---|---|---|---|
| 0 | 1.9607 ± 0.1113 | 1.9604 ± 0.1074 | 1.9607 ± 0.1113 | -0.0004 ± 0.0104 (|Δ| 0.0004; rows favouring 7/9) | +0.0000 ± 0.0000 (|Δ| 0.0000; rows favouring 0/0) |
| 40 | 0.7880 ± 0.0683 | 0.7885 ± 0.0682 | 0.7894 ± 0.0677 | +0.0005 ± 0.0016 (|Δ| 0.0005; rows favouring 8/8) | +0.0014 ± 0.0039 (|Δ| 0.0014; rows favouring 8/8) |
| 80 | 0.7709 ± 0.0668 | 0.7724 ± 0.0666 | 0.7741 ± 0.0667 | +0.0015 ± 0.0011 (|Δ| 0.0015; rows favouring 4/12) | +0.0032 ± 0.0029 (|Δ| 0.0032; rows favouring 6/10) |
| 120 | 0.7665 ± 0.0671 | 0.7669 ± 0.0672 | 0.7674 ± 0.0674 | +0.0004 ± 0.0010 (|Δ| 0.0004; rows favouring 8/8) | +0.0009 ± 0.0031 (|Δ| 0.0009; rows favouring 7/9) |
| 160 | 0.7626 ± 0.0666 | 0.7646 ± 0.0671 | 0.7671 ± 0.0688 | +0.0020 ± 0.0018 (|Δ| 0.0020; rows favouring 6/10) | +0.0045 ± 0.0040 (|Δ| 0.0045; rows favouring 6/10) |
| 200 | 0.7687 ± 0.0676 | 0.7688 ± 0.0676 | 0.7945 ± 0.0733 | +0.0001 ± 0.0016 (|Δ| 0.0001; rows favouring 6/10) | +0.0258 ± 0.0069 (|Δ| 0.0258; rows favouring 3/13) |
- **CURVE READING (arms 1 vs 2): EQUIVALENT-AT-EVERY-EVAL** — largest paired |Δ(1,2)| 0.0020 at step 160 over 6 evals; Δ at step 200 +0.0001 (sign +; + = Unsloth's held-out above e4b's); band 0.02 = the draft's band, kept fixed by the registration (0.02 nats at every eval: this lane has no in-draw reference arm); TC1's floor is quoted beside it: TC1's floor max(0.005, 3 × fused-vs-reference |Δ|) (no --tc1-dir: the measured band is not to hand)
- **(b) PLATEAU TEST (arm 3 − arm 1): NOT-P38-SHAPE** — held-out gap at 200 +0.0258 (paired +0.0258 ± 0.0069 (|Δ| 0.0258; rows favouring 3/13)), at 40 +0.0014 (paired +0.0014 ± 0.0039 (|Δ| 0.0014; rows favouring 8/8)); REPRODUCES-P38 iff >= +0.01 at 200 and <= 0 at 40 — as-shipped is +0.0014 ABOVE the matched arm at step 40 too: no early lead, not P38's shape
- **(c) time to target** (target 0.7888 = mean of the matched pair's step-200 held-out + 0.02; the first eval at or below it, informational): `e4b/fused_attn4_m_200` step 40 at 132.27 s train wall (held-out 0.7880); `unsloth/ckpt_unsloth_m_200` step 40 at 189.07 s train wall (held-out 0.7885); `e4b/fused_attn4_shipped_200` step 80 at 204.86 s train wall (held-out 0.7741)
- **(d) s/step medians over steps 11..200** (not a position: TC1 owns positions; TRAVELS iff within 10% of TC1's 11..20 median when `--tc1-dir` is given, NOT given: UNTESTED; the in-receipt 11..20 window is a same-draw check): `e4b/fused_attn4_m_200` 3.162 s/step (in-receipt 11..20 3.277, -3.5%) **UNTESTED** (no --tc1-dir given); `unsloth/ckpt_unsloth_m_200` 4.505 s/step (in-receipt 11..20 4.496, +0.2%) **UNTESTED** (no --tc1-dir given); `e4b/fused_attn4_shipped_200` 2.527 s/step (in-receipt 11..20 2.532, -0.2%) **UNTESTED** (no --tc1-dir given)
- **(e) ANCHOR grouped_mm: s/step ratio unsloth/e4b = 2.428 → FINDING** (unsloth p38 (venv-unsloth cu130 torch 2.12.1, grouped_mm: tp4's arm EXCEPT the venv); 1.095 vs 0.451 s; tp2 1.457 +66.6% outside, P38 1.413 +71.8% outside, tp4's anchor row not in this tree (UNVERIFIED); ±10%); quality Δ 0.0206 COMPARABLE; peak Δ +1.82 GB
- **(e) ANCHOR t28: s/step ratio unsloth/e4b = 5.147 → FINDING** (unsloth p38_t28 (venv-unsloth-t28, loader-default backend: tp4's arm byte-for-byte); 2.322 vs 0.451 s; tp2 1.457 +253.2% outside, P38 1.413 +264.2% outside, tp4's anchor row not in this tree (UNVERIFIED); ±10%); quality Δ 0.0053 COMPARABLE; peak Δ +1.77 GB
- **(f) SCALING POINT `t1` (never a position): s/step ratio unsloth/e4b = 1.918** (tokens-per-step scaling point (seq 2048, micro-batch 1 x accum 1, N 20); 1.073 vs 0.559 s, single draws; peak Δ -1.66 GB; held-out Δ 0.0033 COMPARABLE); predicates: both arms VALID ok (e4b VALID / unsloth VALID); same trainable count within the pair ok (642514944 / 642514944); lora_path_loop == 0 on every e4b step ok (loop steps []); Unsloth backend counters: torch._grouped_mm >= 6*L*A per step, manual fallback 0 ok (grouped_mm min 672 (>= 288), manual max 0)
- **(f) SCALING POINT `r64`: NOT QUOTED** — not quoted: e4b e4b/fused_attn4_m_r64 is OOM / rank scaling point (r 64 / alpha 64, N 20) unsloth/ckpt_unsloth_m_r64 is OOM; predicates failing: both arms VALID, lora_path_loop == 0 on every e4b step, Unsloth backend counters: torch._grouped_mm >= 6*L*A per step, manual fallback 0; predicates: both arms VALID FAILS (e4b — / unsloth —); same trainable count within the pair ok (2570059776 / 2570059776); lora_path_loop == 0 on every e4b step FAILS (loop steps None); Unsloth backend counters: torch._grouped_mm >= 6*L*A per step, manual fallback 0 FAILS (grouped_mm min None (>= 1152), manual max None)
- TC1's equivalence reading at N per matched pair (informational; no reference arm in this family, so COMPARABLE is the ceiling): `unsloth/ckpt_unsloth_m_200` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0001, step-0 0.0004 SAME-BYTES-CLASS); `unsloth/ckpt_unsloth_m_t1` **COMPARABLE** (median step |Δ| 0.0152, |Δ held-out at N| 0.0033, step-0 0.0054 SAME-BYTES-CLASS); `unsloth/ckpt_unsloth_m_r64` **—** (no OK receipt)
- matched_init_sha per sub-fixture (B, name-free): `200` anchor `f7832488926eda91`: unsloth/ckpt_unsloth_m_200 same; `t1` anchor `f7832488926eda91`: unsloth/ckpt_unsloth_m_t1 same

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3curve | e4b/fused_attn4_m_200 | **VALID** | VALID | anchor of its sub-fixture |  |
| qwen3curve | unsloth/ckpt_unsloth_m_200 | **VALID** | VALID | 0.0001 |  |
| qwen3curve | e4b/fused_attn4_shipped_200 | **VALID** | VALID | 0.0258 |  |
| qwen3curve | e4b/fused_attn4_p38 | **VALID** | VALID | anchor of its sub-fixture |  |
| qwen3curve | unsloth/ckpt_unsloth_p38 | **VALID** | VALID | 0.0206 |  |
| qwen3curve | unsloth/ckpt_unsloth_p38_t28 | **VALID** | VALID | 0.0053 |  |
| qwen3curve | e4b/fused_attn4_m_t1 | **VALID** | VALID | anchor of its sub-fixture |  |
| qwen3curve | unsloth/ckpt_unsloth_m_t1 | **VALID** | VALID | 0.0033 |  |
| qwen3curve | e4b/fused_attn4_m_r64 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 52.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 31.69 MiB is free. Including non-PyTorch memory |
| qwen3curve | unsloth/ckpt_unsloth_m_r64 | **OOM** | — | — | OOM at step 1: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 753.69 MiB is free. Including non-PyTorch memo |

## TC1b predictions P1–P4 (TC1b-PREREG-draft, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3curve | **HELD** | max paired |Δ(1,2)| 0.0020 at step 160 over 6 evals (<= 0.02: the draft's band, kept fixed by the registration (0.02 nats at every eval: this lane has no in-draw reference arm); TC1's floor is quoted beside it); Δ at 200 +0.0001 (sign +, + = Unsloth above) |
| P2 | qwen3curve | **FALSIFIED** | gap at 200 +0.0258, at 40 +0.0014: as-shipped is +0.0014 ABOVE the matched arm at step 40 too: no early lead, not P38's shape |
| P3 | qwen3curve | **UNTESTED** | e4b/fused_attn4_m_200: no --tc1-dir given; unsloth/ckpt_unsloth_m_200: no --tc1-dir given; e4b/fused_attn4_shipped_200: no --tc1-dir given |
| P4 | qwen3curve | **FALSIFIED** | p38 unsloth/e4b 2.428 vs tp2 1.457 (+66.6%) and P38 1.413 (+71.8%), +-10%; tp4's own anchor row is NOT in this tree (TP4_ANCHOR_RATIO unset, UNVERIFIED) -- scored against tp2's 1.457, the number tp4 registered against; t28 variant 5.147 (+253.2% vs tp2) |
