# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, matched init and matched adapter precision (lanes TC1 and TC1b of #835; 2026-10-01/02)

Pre-registrations, read first: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md) (with its three dated amendments) and
[`../../tc1/TC1B-PREREG.md`](../../tc1/TC1B-PREREG.md). The harness is `bench/tc1/` at e4b `079a422` (0.38.0; the boxes'
`versions.txt`), grouped-nf4-gemm `846b512` (0.34.0). Upstream facts read from source: [`../../tc1/UPSTREAM-NOTES.md`](../../tc1/UPSTREAM-NOTES.md).
Design review and its disposition: [`../../tc1/DESIGN-REVIEW.md`](../../tc1/DESIGN-REVIEW.md). Receipts landed privately first (the
adertha receipt store, `receipts/experts4bit-qlora/2026-10-0{1,2}/<run id>/tc1/`) and are copied here under `receipts/<run id>/`
(the per-arm JSONs, `box.json`, `summary.txt`, `versions.txt`, the box-side `RESULTS-tc1.md`, `outer.log`, `logs/`; the tokenised
fixtures and the Unsloth compile cache are left in the private store).

| box | token | host (Vast verified-secure) | what | cost |
|---|---|---|---|---|
| `tc1-5090-16` | `qwen3` | instance 53773622, AMD Threadripper PRO 3975WX, driver 580.178.04 | the matched set, two interleaved draws, HF, the profiled pair, the mb1 secondary | $1.27 |
| `tc1-5090-14` | `qwen3native` | instance 53773863, Xeon E5-2698 v4, driver 595.71.05 | the labelled / native-best rows against the box's own e4b matched arm | $0.93 |
| `tc1-5090-15` | `qwen3curve` (TC1b) | instance 53772765, i9-13900, driver 595.91.07 | the 200-step matched curves, the as-shipped curve, the tp2/P38 anchor pair, the t1 and r64 scaling pairs | $0.56 |
| `tc1-5090-20` | `qwen3axolotl` (amendment 3) | instance 53812706, AMD EPYC 7663 56-Core Processor, driver 595.71.05 | the axolotl rows re-asked on their own box with a working venv, plus the HF torch-2.14 grouped_mm row at micro-batch 1 | $0.32 |
| `tc1-5090-33` | `qwen3nativebest` (amendment 5) | instance 53916526, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | each framework's native-best, two draws each, e4b's matched arm as the anchor; P13 UNTESTED (a refused scattermoe second draw, e4b's pair 6.1 % apart) | $0.51 |
| `tc1-5090-34` | `qwen3nativebest` (amendment 7) | instance 53927504, AMD EPYC 9334 32-Core Processor, driver 590.48.01 | the same token again, with clocks recorded; P13 scored here | $0.44 |
| `tc1-5090-35` | `qwen3nativebest200` (amendment 8) | instance 53935033, Intel Xeon E5-2698 v4, driver 595.71.05 | e4b shipped vs axolotl scattermoe over 200 steps, two draws each, the 200-step matched anchor; P14 HELD, axolotl faster at steady state | $1.76 |
| `tc1-5090-36` | `qwen3nativebest200` (amendment 9) | instance 53971676, AMD Ryzen 9 3900X 12-Core Processor, driver 595.71.05 (machine 45501) | P14's box again on a second host, e4b pinned at 7c31b88; P15 HELD (0.901 [0.892, 0.910]) | $0.93 |
| `tc1-5090-38` | `qwen3syncab` (amendment 10, #945) | instance 53991731, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | e4b against itself: 13 host syncs per MoE layer pass vs 1; P16 0.866, P17 0.847, both HELD | $0.42 |
| `tc1-5090-40` | `qwen3nativebest200` (amendment 11, #945) | instance 53999231, Intel Xeon E5-2696 v4 @ 2.20GHz, driver 595.91.07 | e4b shipped vs axolotl scattermoe over 200 steps after the sync fix; P18 UNTESTED (both pairs unstable, 12.0 % / 9.1 %); [read](RESULTS-tc1-steady945.md) | $1.50 |
| `tc1-5090-41` | `qwen3prof945` (amendment 12, #945) | instance 54000851, Intel Core Ultra 9 285K, driver 595.91.07 | e4b profiled after the sync fix, the legacy path as the before-picture; P19 HELD (matched device busy 0.595 → 0.740); [read](RESULTS-tc1-prof945.md) | $0.27 |
| `tc1-5090-42` | `qwen3leanab` (amendment 13, #945) | instance 54006898, AMD Ryzen 9 7900 12-Core Processor, driver 580.178.04 | e4b against itself: gnf4's previous padded LoRA delta vs its trimmed body (#440); P20 0.939, P21 0.911, both HELD; [read](RESULTS-tc1-leanab.md) | $0.33 |
| `tc1-5090-43` | `qwen3tileab` (amendment 14, #945) | instance 54011443, AMD Ryzen 9 7950X 16-Core Processor, driver 580.159.03 | e4b against itself: gnf4's max-keyed prefill M-tile vs the cost rule (#441); P22 0.924, P23 0.968, both HELD; [read](RESULTS-tc1-tileab.md) | $0.41 |
| `tc1-5090-44` | `qwen3rmsab` (amendment 15, #945) | instance 54018557 | HARNESS_ERROR before the command ran: the guard's Vast auth probe got one HTTP 429 (adertha-agents#138) | $0.00 |
| `tc1-5090-45` | `qwen3rmsab` (amendment 15, #945) | instance 54018669, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | e4b against itself: the HF RMSNorm composite vs e4b's fused training RMSNorm (#961); P24 0.924, P25 0.959, P26 held-out within 0.0021, all HELD; [read](RESULTS-tc1-rmsab.md) | $0.39 |
| `tc1-5090-46` | `qwen3nativebest200` (amendment 16) | instance 54024744, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | e4b shipped (after amendments 10-15) vs axolotl scattermoe over 200 steps; P27 HELD, axolotl / e4b 1.238 [1.231, 1.246], e4b faster at steady state; [read](RESULTS-tc1-steady16.md) | $0.99 |
| `tc1-5090-47` | `qwen3nativebest200` (amendment 17) | instance 54037643, machine 142284 again (AMD EPYC 7663) | the same box; a same-machine repeat, P28 UNTESTED by rule; reads 1.253 [1.248, 1.258]; [read](RESULTS-tc1-steady18.md) | $0.98 |
| `tc1-5090-48` | `qwen3nativebest200` (amendment 18) | instance 54051454, machine 150527, AMD EPYC 7C13 64-Core Processor, driver 580.95.05 | the same box with machine 142284 excluded by evidence; P29 HELD, 1.146 [1.129, 1.164]: e4b faster at steady state on a second host; [read](RESULTS-tc1-steady18.md) | $1.23 |
| `tc1-5090-49` | `qwen3` (amendment 19) | instance 54075516, AMD EPYC 7C13 64-Core Processor, driver 580.95.05 | the matched set again with e4b after amendments 10-15: Unsloth/e4b 1.997 (P30 HELD), matched set inside the draw noise (P31 HELD); [read](RESULTS-tc1-matched19.md) | $1.15 |
| `tc1-5090-50` | `qwen3axolotl` (amendment 19) | instance 54075633, AMD Ryzen 9 9950X3D 16-Core Processor, driver 610.57.04 | the axolotl matched rows again: axolotl/e4b 2.775 (P32 HELD); [read](RESULTS-tc1-matched19.md) | $0.33 |
| `tc1-5090-59` | `qwen3denseab` (amendment 22) | instance 54177463, AMD EPYC 9454P | e4b against itself: grouped-nf4-gemm's fused kernels vs its dense route: dense/fused 2.947, far slower (P39 FALSIFIED); [read](RESULTS-tc1-denseab.md) | $0.45 |
| `tc1-5090-62` | `mixtraldenseab` (amendment 22) | instance 54179030, Intel Core Ultra 9 285K | the same A/B on Mixtral-8x7B resident (`E4B_ABSMAX_DQ=1` both sides): dense/fused 0.651 (P38 HELD); held-out within 0.01 on both families (P40 HELD); [read](RESULTS-tc1-denseab.md) | $0.51 |
| `tc1-5090-65` | `qwen3memcensus` (amendment 23) | instance 54195008, AMD EPYC 9454P | a memory census, no speed read: e4b fp32 absmax, e4b `E4B_ABSMAX_DQ=1`, Unsloth at micro-batch 1; e4b dq − Unsloth +0.43 GB, all transient (P41, P42 HELD; P43 FALSIFIED); [read](RESULTS-tc1-memcensus.md) | $0.38 |
| `tc1-5090-66` | `qwen3bmmab` (amendment 24) | instance 54201179, AMD EPYC 7B13 | a bmm replay (no model), then e4b in its field environment vs Unsloth's: the matched arm 0.882× in torch 2.12.1+cu130 (P47 HELD); the fp32 bmm's cost is per new shape and the same on both torch versions (P44, P46 FALSIFIED; P45 HELD); [read](RESULTS-tc1-bmmab.md) | $1.26 |
| `tc1-5090-67` | `qwen3samestack` (amendment 25) | instance 54209084, Intel Core Ultra 9 285K | both frameworks on one stack: the matched set EQUIVALENT, parity PASS (P52 HELD); Unsloth's draws 18.2 % apart and e4b's field-image draws 8.4 % apart, so no ratio is read (P50, P51 UNTESTED); one re-draw registered (amendment 27); [read](RESULTS-tc1-samestack-box1.md) | $0.61 |
| `tc1-5090-68` | `qwen3samestack` (amendment 27, the re-draw) | instance 54214853, AMD EPYC 7663 | the matched set INSIDE-DRAW-NOISE, parity PASS (P52 HELD again); e4b's same-stack draws 7.8 % apart, so no ratio is read (P50, P51 UNTESTED, final); amendment 29 re-asks them over 60 steps; [read](RESULTS-tc1-samestack-box2.md) | $1.86 |
| `tc1-5090-69` | `qwen3prebindab` (amendment 26) | instance 54223080, Intel Core Ultra 9 285K | e4b against itself, prebound Triton launches off vs on: matched arm 0.973 (P54 HELD); the shipped arm's flags-off draws 8.3 % apart (P53, P55 UNTESTED); the flags stay opt-in; amendment 30 re-asks the shipped pair over 60 steps; [read](RESULTS-tc1-prebindab.md) | $0.43 |
| `tc1-5090-70` | `qwen3dqab` (amendment 28) | instance 54223173, AMD EPYC 7C13 (Vast machine 45511) | e4b against itself, the expert absmax fp32 vs double-quantized: peak 27.15 → 25.82 GB and held-out +0.0026, but both speed pairs unstable under host load (P56 UNTESTED); amendment 31 re-asks it over 60 steps; [read](RESULTS-tc1-dqab.md) | $1.13 |
| `tc1-5090-71` | `mixtraldqab` (amendment 28) | instance 54223342, AMD EPYC 7B13 (Vast machine 145701) | the same on Mixtral-8x7B resident: dq1/dq0 1.023, peak 31.07 → 29.03 GB, held-out −0.0028 (P57 HELD); [read](RESULTS-tc1-dqab.md) | $1.27 |
| `tc1-5090-72` | `qwen3samestack` (amendment 29, 60 steps) | instance 54227048, AMD EPYC 7C13 (Vast machine 45511) | the same-stack pair over 60 steps: e4b's same-stack draws 13.1 % apart and its field-image draws 6.4 %, host load1 6–23, so no ratio is read (P50, P51 UNTESTED, final); amendment 33 re-asks with load-gated draws; [read](RESULTS-tc1-samestack-box3.md) | $1.75 |
| `tc1-5090-73` | `qwen3prebindab` (amendment 30, 60 steps) | instance 54227619, AMD EPYC 7B13 (Vast machine 145701) | the shipped arm's prebind pair over 60 steps: `_pb1`/`_pb0` 0.980 [0.957, 1.003] (P53 HELD), held-out +0.0011; with amendment 26's P54 the prebound launches become the default |
| `tc1-5090-75` | `qwen3dqab` (amendment 31, 60 steps) | instance 54238515, AMD EPYC 7B13 (Vast machine 145701) | Qwen3-30B-A3B's absmax pair over 60 steps on a quiet host: dq1/dq0 1.014, peak 27.44 → 26.10 GB, held-out −0.0021 (P56 HELD; with P57 and P58, amendment 28's rule makes the double-quantized absmax the default); [read](RESULTS-tc1-dqab-qwen3-60.md) | $1.03 |
| `tc1-5090-74` | `qwen3tritonab` (amendment 32, 60 steps) | instance 54237146, AMD EPYC 7B13 (Vast machine 145701) | one variable, triton 3.4 vs 3.7.1 in venv-e4b: matched arm 0.992 (P59 FALSIFIED), shipped arm 0.971 (P60 HELD), held-out within 0.001 (P61 HELD); the environment gain is not triton's on this host-bound step; amendment 34 splits it; [read](RESULTS-tc1-tritonab.md) | $1.28 |
| `tc1-5090-76` | `qwen3samestack` (amendment 33, 60 steps, load-gated) | instance 54239673, AMD EPYC 7B13 (Vast machine 145701) | both frameworks on one stack, every pair stable: Unsloth/e4b 2.352 (P50 HELD), e4b same-stack / field-image 0.900 (P51 HELD); three draws voided for host load and run again; by amendment 25's rule 2.352 becomes the quoted Qwen3-30B-A3B position; [read](RESULTS-tc1-samestack-box4.md) | $1.76 |
| `tc1-5090-78` | `qwen3envsplit` (amendment 34, 60 steps, load-gated) | instance 54248510, AMD EPYC 7B13 (Vast machine 145701) | e4b's matched arm in three environments: transformers 5.5 vs 5.18 on torch 2.8 1.005 (P62 FALSIFIED), torch 2.12 + triton 3.7 vs torch 2.8 + triton 3.4 0.905 (P63 HELD), the whole environment 0.909 (P64 HELD); the environment gain is torch's; [read](RESULTS-tc1-envsplit.md) | $1.73 |
| `tc1-5090-79` | `qwen3prebind37` (amendment 35, 60 steps, load-gated) | instance 54255834, AMD EPYC 7B13 (Vast machine 145701) | amendment 26's prebind A/B under triton 3.7.1 in venv-unsloth: shipped 0.996 (P66 HELD), matched 0.986 (P67 HELD), held-out within 0.003 (P68 HELD); triton 3.7 stays in the prebound path's supported versions; [read](RESULTS-tc1-prebind37.md) | $1.68 |
| `tc1-5090-80` | `qwen3compactab` (amendment 36, 60 steps, load-gated) | instance 54259219, AMD EPYC 7B13 (Vast machine 145701) | grouped-nf4-gemm's compact padded LoRA delta off vs on in venv-unsloth: the matched peak ROSE 0.23 GB (P69 FALSIFIED), and the step got faster than registered, matched 0.969 and shipped 0.970 (P70, P71 FALSIFIED); held-out within 0.001 (P72 HELD); it stays opt-in; [read](RESULTS-tc1-compactab.md) | $1.48 |
| `tc1-5090-83` | `qwen3compactab2` (amendment 37, 60 steps, load-gated) | instance 54275103, AMD EPYC 7702P (Vast machine 45379) | the compact delta again with grouped-nf4-gemm#473's backward, on another host: matched peak −0.288 GB (P73 HELD), matched 0.967 (P74 HELD), shipped 0.948, faster than its band (P75 FALSIFIED), held-out within 0.003 (P76 HELD); by the rule it stays opt-in pending its own registration; [read](RESULTS-tc1-compactab2.md) | $0.78 |
| `tc1-5090-85` | `qwen3compactab3` + `mixtralcompactab` (amendment 38, 60 steps, load-gated) | instance 54292473, AMD EPYC 9655 (Vast machine 150700) | the compact delta's default decision on a fast host: slower on every arm, Qwen3 matched 1.016 and shipped 1.013, Mixtral 1.013 (P77, P78, P80 FALSIFIED), while the peaks held (Qwen3 −0.312 GB, Mixtral −0.037; P79, P81 HELD); it stays opt-in; [read](RESULTS-tc1-compact-default.md) | $2.64 |
| `tc1-5090-86` | `qwen3samestack4k` (amendment 39, packed 4,096-token rows, 30 steps, load-gated) | instance 54297512, AMD EPYC 7B13 (Vast machine 145701) | the packed regime on one stack: every e4b arm OOMed at step 1 allocating 2.32 GiB, the fp32 copy of the full-vocabulary logits (P86 FALSIFIED), while Unsloth trained at 24.86 GB (its draws 7.7 % apart; P84, P85 UNTESTED); an e4b loss in that regime; [read](RESULTS-tc1-packed4k.md) | $1.37 |
| `tc1-5090-94` | `qwen3samestackh2` (amendment 42, 60 steps, load-gated) | instance 54328898, AMD EPYC 7K62 (Vast machine 152440) | amendment 33's same-stack box on a second host, the current code: Unsloth/e4b 2.468 (P94 HELD) and the environment 0.870 (P95 HELD); 2.352 stays the position to quote, now read on two hosts; [read](RESULTS-tc1-samestack-host2.md) | $1.25 |

Thirteen earlier draws were refused or stopped before producing a row (driver floor, pre-flight bandwidth, a controller-slot
race, the cu130 pip resolver — TC1 amendments 1 and 2) for about $0.57 in total, and the first axolotl box (`tc1-5090-19`) was
lost at 12 minutes when its controller was killed on the operator's side (the launcher's guard destroyed the instance on heartbeat
loss; about $0.14, no receipt); every one is a receipt or a guard record in the private store and none is quoted. Verdicts, validity, positions, equivalence bands and the prediction scoring are the reducer's
([`../../tc1/tc1_reduce.py`](../../tc1/tc1_reduce.py)) and are reproduced here from the receipts:
[`RESULTS-tc1-combined.md`](RESULTS-tc1-combined.md) (the four boxes in one pass, the amendment-3 reducer) and
[`RESULTS-tc1b-vs-tc1.md`](RESULTS-tc1b-vs-tc1.md) (TC1b read against the matched box with `--tc1-dir`).

## Amendment 42 (2026-10-05): the same-stack position replicates on a second host, 2.468

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 42. One RTX 5090 (`tc1-5090-94`, AMD EPYC 7K62, Vast
machine 152440; earlier launches cost $0.006 and $0: a host below the driver floor, and a receipts-store race): amendment 33's box on the
current code (e4b `5c90886`, grouped-nf4-gemm `ccf4de9`), off amendment 33's machine. Read:
[`RESULTS-tc1-samestack-host2.md`](RESULTS-tc1-samestack-host2.md).

| arm | s/step (two draws, 60 steps) | peak |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 4.143 / 4.073 (1.7 % apart) | 27.50 GB |
| Unsloth `ckpt_unsloth_m`, venv-unsloth | 10.155 / 10.120 (0.3 % apart) | 24.27 GB |
| e4b `fused_attn4_m_t28`, venv-e4b | 4.734 / 4.709 (0.5 % apart) | 27.45 GB |

- **P94 HELD.** Unsloth/e4b on one stack reads **2.468** [2.443, 2.493], against amendment 33's 2.352 on an EPYC 7B13: 5 % apart, both inside
  [1.9, 2.9]. Held-out COMPARABLE (Δ −0.0012); Unsloth peaks 3.22 GB lower and spends ×1.41 e4b's energy per step.
- **P95 HELD.** The environment reads **0.870** [0.860, 0.880]. The prebound launches now cover triton 3.7, so both of e4b's sides take them;
  amendment 33's 0.900 had them on one side only, and its read bounded the symmetric value at about 0.876–0.900.
- **By amendment 42's rule 2.352 stays the position to quote**, now read on two hosts (2.352 and 2.468). Both steps ran longer on this host
  (e4b 4.11 s, Unsloth 10.14 s, against 3.49 and 8.22 s) and the ratio moved 5 %.
- **The host was shared** with this campaign's `tc1-5090-91` (amendment 40) until 15:30Z; the load gate voided no draw.

## Amendment 39 (2026-10-05): on packed 4,096-token rows e4b at its defaults runs out of memory where Unsloth trains

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 39. One RTX 5090 (`tc1-5090-86`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3samestack4k`, amendment 25's same-stack family on packed rows of exactly 4,096 real tokens (tokens sha
`d2a501eba57d`, the same as the harness PR's local build), micro-batch 1 × accum 4, 30 steps, `TC1_FREE_OUTPUTS=1`, load-gated draws.
Read: [`RESULTS-tc1-packed4k.md`](RESULTS-tc1-packed4k.md).

| arm | status | s/step (two draws) | peak |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth, defaults | **OOM at step 1** (both draws) | — | 30.32 GB at the failure |
| e4b `fused_attn4_m_t28`, venv-e4b, defaults | **OOM at step 1** (both draws) | — | 30.27 GB at the failure |
| Unsloth `ckpt_unsloth_m` | OK, VALID | 15.164 / 14.040 (7.7 % apart) | 24.86 GB |

- **P86 FALSIFIED.** All four e4b arms failed at step 1 allocating **2.32 GiB**: 4,096 × 151,936 × 4 bytes, the fp32 copy of the
  full-vocabulary logits that Hugging Face's causal-LM loss makes. At that moment 29.5 GiB of the card's 31.36 GiB were in use.
- **P84 and P85 UNTESTED.** No e4b arm ran. Unsloth's two draws are 7.7 % apart; its first draw stood at a host load of 16.2 after the
  gate's two re-runs (the voided attempts read 14.19 and 14.55 s/step).
- **By amendment 39's rule this is an e4b loss in the packed regime**, recorded with the peak where it failed (row
  `e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.packed-4k`). Unsloth trains the same rows resident at 24.86 GB: it computes its loss in
  chunks and never materialises the full logits.
- **The lever is a chunked loss for e4b**, its own registration. experts4bit-qlora#1142 adds one, opt-in: on an RTX A2000, a 4-layer
  slice of the same checkpoint hits the stock path's exact 2.32 GiB failure at 4,096 tokens and trains through it with the loss chunked.
- **The host was busy** (load1 6.8–22.5 during Unsloth's draws). That costs the speed reading here, not the memory one.

## Amendment 38 (2026-10-05): on a fast host the compact delta is 1.3–1.6 % slower; it stays opt-in

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 38. One RTX 5090 (`tc1-5090-85`, AMD EPYC 9655, Vast
machine 150700), every arm in venv-unsloth with e4b `e1faf47` and grouped-nf4-gemm `0e393f0` (after #473), 60 steps, load-gated draws
(none voided). Read: [`RESULTS-tc1-compact-default.md`](RESULTS-tc1-compact-default.md).

| family, arm | `_cd0` s/step | `_cd1` s/step | `_cd1` / `_cd0` | peak `_cd0` → `_cd1` | prediction |
|---|---|---|---|---|---|
| Qwen3-30B-A3B, matched | 2.167 / 2.178 | 2.215 / 2.199 | **1.016** [1.010, 1.022] | 27.501 → 27.189 GB | P77 FALSIFIED (≤ 0.99); P79 HELD |
| Qwen3-30B-A3B, shipped | 1.716 / 1.720 | 1.742 / 1.740 | **1.013** [1.011, 1.015] | 24.673 → 24.673 GB | P78 FALSIFIED (≤ 0.99) |
| Mixtral-8x7B, matched (resident, defaults) | 3.326 / 3.330 | 3.374 / 3.371 | **1.013** [1.012, 1.014] | 31.234 → 31.197 GB | P80 FALSIFIED (≤ 1.01); P81 HELD |

- **Held-out** moves +0.0008 / 0.0000 (Qwen3) and +0.0008 (Mixtral): P82 and P83 HELD.
- **By amendment 38's rule the flag stays opt-in**: P77, P78 and P80 fell on the slow side. No further box is registered for it under
  these rules.
- **The peak held everywhere; the speed depends on the host.** With #473's backward the flag lowers the fp32 arm's peak by 0.29–0.31 GB
  on both hosts that measured it, and never raises a peak. Its speed tracks how host-bound the step is:

  | box | host | e4b's Qwen3 matched step | `_cd1` / `_cd0` matched / shipped |
  |---|---|---|---|
  | `tc1-5090-80` (amendment 36) | EPYC 7B13 | 3.43 s | 0.969 / 0.970 |
  | `tc1-5090-83` (amendment 37) | EPYC 7702P | 3.89 s | 0.967 / 0.948 |
  | `tc1-5090-85` (this box) | EPYC 9655 | 2.17 s | 1.016 / 1.013 |

  The compact node trades host work (one autograd node instead of about ten per projection) for device work (the padded block rebuilt in
  backward: +0.7 ms per layer's backward on an RTX A2000, grouped-nf4-gemm#445). Where the host is slow the first wins; on this host the
  same step is 37–44 % shorter, less of it is host time, and the second wins.
- **Recorded for later, not read here.** e4b's Qwen3 step on this host is 2.17 s against 3.4–3.9 s on the other two, the same code and
  card. Positions are within-box readings of their host, and this box ran no Unsloth arm.

## Amendment 37 (2026-10-05): with its backward releasing early, the compact delta lowers the matched peak 0.29 GB and is 3–5 % faster on another host

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 37. One RTX 5090 (`tc1-5090-83`, AMD EPYC 7702P, Vast
machine 45379; two earlier launches cost $0.007 and $0: a host below the harness's driver floor, then an avoid receipt the launcher does not
accept): amendment 36's box, `NF4_QLORA_COMPACT_DELTA` 0 vs 1, every arm in venv-unsloth with e4b `64afc6d` and grouped-nf4-gemm `9622144`
(after #473), 60 steps, load-gated draws (none voided). Read: [`RESULTS-tc1-compactab2.md`](RESULTS-tc1-compactab2.md).

| arm | `_cd0` s/step | `_cd1` s/step | `_cd1` / `_cd0` | peak `_cd0` → `_cd1` | held-out Δ | prediction |
|---|---|---|---|---|---|---|
| matched (fp32 adapters) | 3.914 / 3.860 | 3.752 / 3.763 | **0.967** [0.959, 0.975] | 27.477 → **27.189 GB** | +0.0029 | P73, P74 HELD |
| shipped (bf16 adapters) | 3.026 / 2.970 | 2.824 / 2.864 | **0.948** [0.933, 0.964] | 24.673 → 24.673 GB | −0.0003 | P75 FALSIFIED (below [0.95, 0.99]) |

- **The peak moved the way the A2000 said.** Amendment 36's box read the matched peak +0.229 GB with the flag on; with #473's backward it
  reads −0.288 GB (P73 HELD, band [−0.05, 0.50]). The shipped arm's peak is elsewhere in its step and does not move.
- **The speed replicated, and the shipped arm more than registered.** Matched 0.967 against amendment 36's 0.969 (P74 HELD); shipped 0.948
  against 0.970, just below the band (P75 FALSIFIED on the fast side).
- **P76 HELD.** Held-out at N moves +0.0029 (matched) and −0.0003 (shipped).
- **By amendment 37's rule the flag stays opt-in pending its own registration**: a ratio below 0.95 is the "otherwise" branch. The bands
  were two-sided, and the reading fell outside on the side the decision wanted. A default needs a registration whose bands allow that.
- **Two hosts now agree on the direction** (EPYC 7B13 and EPYC 7702P): 3–5 % off the step on both arms, with outputs and gradients
  bit-identical by construction.

## Amendment 36 (2026-10-05): the compact padded LoRA delta is 3 % faster, not lighter — the matched peak rose 0.23 GB

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 36. One RTX 5090 (`tc1-5090-80`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3compactab`, `NF4_QLORA_COMPACT_DELTA=0` (`_cd0`) against `=1` (`_cd1`) on the shipped and the matched
arm, two draws a side in ABBA order, every arm in venv-unsloth with e4b `23edeff` and grouped-nf4-gemm `c4a683b`, 60 steps, load-gated
draws. Read: [`RESULTS-tc1-compactab.md`](RESULTS-tc1-compactab.md).

| arm | `_cd0` s/step | `_cd1` s/step | `_cd1` / `_cd0` | peak `_cd0` → `_cd1` | held-out Δ |
|---|---|---|---|---|---|
| matched (fp32 adapters) | 3.431 / 3.428 | 3.348 / 3.299 | **0.969** [0.962, 0.977] | 27.490 → **27.719 GB** | +0.0004 |
| shipped (bf16 adapters) | 2.919 / 2.858 | 2.793 / 2.810 | **0.970** [0.957, 0.983] | 24.673 → 24.673 GB | −0.0006 |

- **Engagement.** Every arm resolved the flag its tag names, kept no layer's MoE activations and ran the padded LoRA path (49,152 padded
  calls each).
- **P69 FALSIFIED.** The matched arm's peak rose by 0.229 GB instead of falling by [0.3, 2.0] GB. The shipped arm's did not move.
- **P70 and P71 FALSIFIED, on the fast side.** The step got faster than the registered band allowed, 0.969 (matched) and 0.970 (shipped,
  0.9699 before rounding) against [0.97, 1.02]. One autograd node in place of about ten per projection is less host work, and this step is
  host-bound.
- **P72 HELD.** Held-out at N moves +0.0004 and −0.0006.
- **By amendment 36's rule the compact delta stays opt-in.** The registered memory reason did not hold, and a speed default needs its own
  registration.
- **Why the peak rose (from the code, not yet measured).** `_CompactPaddedDelta.backward` keeps the padded output gradient `[G, widest, N]`
  referenced until it returns, while it rebuilds the input block and computes the input's gradient. Autograd's separate nodes release that
  gradient after the second product's backward. With whole-layer checkpointing, the matched arm's peak falls inside a MoE layer's backward,
  where the extra block sits. The shipped arm's peak did not move, so it is elsewhere in that step. A grouped-nf4-gemm change that
  releases each intermediate at its last use is being measured on an RTX A2000.
- **The gate.** The shipped arm's first draws were voided (load 8.2) and run again; the first attempts read 0.966 for the shipped
  arm, the same verdict.
- **The host was shared with this campaign's own boxes** (`tc1-5090-79` until 06:13Z).

## Amendment 35 (2026-10-05): under triton 3.7.1 the prebound launches take 1.4 % off the matched arm and 0.4 % off the shipped arm

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 35. One RTX 5090 (`tc1-5090-79`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3prebind37`, amendment 26's eight arms in its ABBA order, every one in venv-unsloth (torch 2.12.1+cu130,
transformers 5.5.0, triton 3.7.1) with e4b `3cc7f10` and grouped-nf4-gemm `c4a683b` (both after #1108 / #471), 60 steps, load-gated
draws. Read: [`RESULTS-tc1-prebind37.md`](RESULTS-tc1-prebind37.md).

| arm | `_pb0` s/step | `_pb1` s/step | `_pb1` / `_pb0` | held-out Δ | prediction |
|---|---|---|---|---|---|
| shipped (bf16 adapters) | 2.876 / 2.902 | 2.839 / 2.913 | **0.996** [0.978, 1.013] | −0.0026 | P66 HELD ([0.97, 1.00]) |
| matched (fp32 adapters) | 3.444 / 3.418 | 3.408 / 3.360 | **0.986** [0.975, 0.997] | +0.0009 | P67 HELD ([0.97, 1.00]) |

- **Engagement.** Every `_pb1` arm counted 216,335 prebound e4b launches and 72,162 grouped-nf4-gemm ones per process, every `_pb0` arm
  none, all under triton 3.7.1 and torch 2.12.1.
- **P68 HELD.** Held-out at N moves −0.0026 (shipped) and +0.0009 (matched), inside 0.005.
- **By amendment 35's rule** triton 3.7 stays in the prebound path's supported versions in both repositories; row
  `e4b.train.prebind.triton37.qwen3.5090.2026-10-05`.
- **What it is worth.** The matched interval excludes 1.0; the shipped interval does not. Under triton 3.4 the same flags read 0.973
  (matched) and 0.980 (shipped); triton 3.7.1's own launch is cheaper, so there is less to take off.
- **The gate.** Five draws were voided for host load and run again; their files are in
  [`receipts/tc1-5090-79/loadvoid/`](receipts/tc1-5090-79/loadvoid/). The shipped arm's first draws exhausted the retries and stood at a
  median load1 of 8.17 (`_pb0`) and 8.24 (`_pb1`). Reading the first attempts instead gives 0.980 (shipped) and 0.995 (matched), inside
  the same bands.
- **The host was shared with this campaign's own boxes.** `tc1-5090-78` (until 05:23Z) and `tc1-5090-80` (from 05:11Z) ran on the same
  machine, so part of the load the gate saw was the campaign's.

## Amendment 34 (2026-10-05): the 5090's environment gain is torch 2.12's, not transformers 5.5's

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 34. One RTX 5090 (`tc1-5090-78`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3envsplit`, the matched arm in three environments, two draws each in ABC CBA order, 60 steps, load-gated
draws, the prebound launches off everywhere. Read: [`RESULTS-tc1-envsplit.md`](RESULTS-tc1-envsplit.md).

| side | torch | transformers | triton | s/step (two draws) | stable |
|---|---|---|---|---|---|
| `_e0` venv-e4b | 2.8.0+cu128 | 5.18.0 | 3.4.0 | 3.840 / 3.816 | yes (0.6 %) |
| `_e1` venv-e4b-tf55 (built on the box) | 2.8.0+cu128 | 5.5.0 | 3.4.0 | 3.821 / 3.871 | yes (1.3 %) |
| `_e2` venv-unsloth + e4b | 2.12.1+cu130 | 5.5.0 | 3.7.1 | 3.507 / 3.452 | yes (1.6 %) |

- **P62 FALSIFIED.** transformers 5.5 over 5.18 on torch 2.8 reads **1.005** [0.995, 1.014], outside [0.90, 1.00]. The router casts the
  A2000 profile pointed at do not show in the 5090's step. By amendment 34's rule nothing follows from P62.
- **P63 HELD.** torch 2.12 + triton 3.7 over torch 2.8 + triton 3.4, at transformers 5.5, reads **0.905** [0.892, 0.918]. Amendment 32
  read triton 3.7.1 alone (on torch 2.8) at 0.992 on this arm, so nearly all of it is torch 2.12's.
- **P64 HELD.** The whole environment reads **0.909** [0.899, 0.919]; amendment 24 read 0.882, amendment 33 0.900.
- **P65 HELD.** Held-out at N moves +0.0013 (`_e1`) and −0.0010 (`_e2`).
- **By amendment 34's rule (P63 HELD)** the install section now says that torch 2.12 runs e4b's host-bound training step faster on an
  RTX 5090. Row `e4b.train.env-split.qwen3.5090.2026-10-05`.
- **The gate.** Six draws were voided for host load and run again; their files are in
  [`receipts/tc1-5090-78/loadvoid/`](receipts/tc1-5090-78/loadvoid/). Reading the first attempts instead gives 1.004, 0.907 and 0.911:
  no verdict changes. `_e0`'s second draw stood on its third attempt at a median load1 of 9.15 (the last attempt stands whatever its
  load); it read 3.816 against its first draw's 3.840 at 4.2.
- **The host was shared with this campaign's own boxes.** From 04:45Z and 05:11Z, `tc1-5090-79` and `tc1-5090-80` ran on the same machine
  (145701). Their steps are host-bound too, so part of the load the gate saw was this campaign's. The ABC CBA order puts each side's draws
  on both halves of the run.
- **Not registered, recorded.** Step-0 held-out on `_e1` (torch 2.8, transformers 5.5) is 1.9733 against `_e0`'s 1.9441 and `_e2`'s
  1.9478 (NEAR, 0.029). The same transformers on torch 2.12 does not move it, so it is the combination, not transformers 5.5 alone. At N=60 the
  three sides are within 0.0014.

## Amendment 30 (2026-10-05): the shipped prebind pair over 60 steps (P53 HELD); with P54 and P55 HELD the prebound launches become the default

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 26 and 30.
- **The box.** One RTX 5090 (`tc1-5090-73`, AMD EPYC 7B13, Vast machine 145701, not machine 151350 as the amendment
  requires; $0.97 invoiced): the token `qwen3prebindab` with `TC1_STEPS=60`, the shipped arm only, two draws a side in ABBA
  order; e4b `306dfa9c`, grouped-nf4-gemm `f0c1ece`, triton 3.4.0.
- **Read:** [`RESULTS-tc1-prebindab-60.md`](RESULTS-tc1-prebindab-60.md), the box's own mechanically scored file.

| `fused_attn4_shipped` | `_pb0` (flags off) | `_pb1` (flags on) |
|---|---|---|
| s/step, two draws (60 steps) | 3.092 / 3.047 (1.5 % apart) | 3.056 / 2.960 (3.2 % apart) |
| held-out at N | 0.7575, 0.7556 | 0.7589, 0.7565 |

- **P53 HELD.** `_pb1`/`_pb0` reads 0.980 [0.957, 1.003 over four cross-draw ratios], inside [0.90, 0.98], at its upper
  edge, as the reducer scores it. On the `_pb1` side, 216,335 e4b launches and 72,162 grouped-nf4-gemm launches were
  prebound.
- **P55 HELD.** The shipped half moves held-out by +0.0011, inside 0.005. The matched half is amendment 26's −0.0012
  (`tc1-5090-69`). This box ran no matched arm, so its own file scores P54 and P55 UNTESTED, and the combined verdict is
  amendment 30's registered rule.
- **P54 HELD** on amendment 26's box (matched 0.973 [0.958, 0.988]).
- **By the rule, both flags become defaults.** They already have: e4b #1099 and grouped-nf4-gemm #470 flipped them at
  ~01:50Z, citing this box before its read was on `main`. This read was committed afterwards by the maintainer session,
  from the run's output in the private receipts store (`tc1-5090-73`, committed there as `b43ec238`), under the standing
  permission to commit another lane's finished receipts. The gain is small (2–3 %), and the shipped interval reaches 1.0.

## Amendment 33 (2026-10-05): on one stack, with load-gated draws, Unsloth/e4b 2.352 and the environment gain 0.900 (P50, P51 HELD)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 25, 29 and 33. One RTX 5090 (`tc1-5090-76`, AMD EPYC
7B13, Vast machine 145701): the token `qwen3samestack` with `TC1_STEPS=60`, no reference arm, `TC1_LOAD_GATE=6.0` and
`TC1_LOAD_RETRIES=2`. Read: [`RESULTS-tc1-samestack-box4.md`](RESULTS-tc1-samestack-box4.md).

| arm | s/step (two draws, 60 steps) | stable | host load1 median per standing draw |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth (torch 2.12.1+cu130, transformers 5.5.0, triton 3.7.1) | 3.496 / 3.492 | yes (0.1 %) | 4.41 / 5.83 |
| e4b `fused_attn4_m_t28`, venv-e4b (torch 2.8.0+cu128, transformers 5.18.0, triton 3.4.0) | 3.855 / 3.913 | yes (1.5 %) | 2.94 / 5.85 |
| Unsloth `ckpt_unsloth_m`, venv-unsloth | 8.208 / 8.228 | yes (0.3 %) | 4.08 / 5.22 |

- **P50 HELD.** On one stack Unsloth/e4b reads **2.352** [2.348, 2.356 over 4 cross-draw ratios], inside [1.9, 2.9]. Unsloth peaks
  3.22 GB lower (24.27 vs 27.49 GB); its energy per step is ×1.37 e4b's. Held-out at N=60: e4b 0.7569, Unsloth 0.7557, COMPARABLE
  (|Δ| 0.0012; no reference arm, so EQUIVALENT cannot be read).
- **P51 HELD.** e4b's matched arm in venv-unsloth over venv-e4b reads **0.900** [0.893, 0.907], inside [0.80, 0.95]. Amendment 24 read
  0.882 on another host; the environment gain replicates.
- **By amendment 25's rule** (P50 HELD here, P52 HELD on amendment 25's two boxes), 2.352 becomes the Qwen3-30B-A3B 5090 position to
  quote, register `e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.same-stack`. Amendment 19's 1.997 stays as the reading with e4b in the
  field image's environment.
- **The gate.** Three draws ran over a median load1 of 6.0 and were run again; their files are in
  [`receipts/tc1-5090-76/loadvoid/`](receipts/tc1-5090-76/loadvoid/). The voided draws read 3.899 and 4.119 s/step (e4b's field-image
  second draw, load 6.26 and 8.44) and 7.928 (Unsloth's second draw, load 6.33): the voided Unsloth draw was the faster one. Reading the
  first attempts instead gives 2.309 and 0.901, inside the same bands, so the gate changed no verdict.
- **A difference between the sides that the registration did not name.** The box ran e4b `c8925bb`, after the prebound Triton launches
  became the default (#1099) for triton 3.4 and 3.6. They engaged on the venv-e4b arm (216,335 prebound launches) and not on the
  venv-unsloth arm (triton 3.7.1, not covered at that commit; 0). grouped-nf4-gemm's prebound launches were in neither (its pin predates
  them). Amendment 26 read e4b's and grouped-nf4-gemm's flags together at 0.973 on the matched arm, so with e4b's flag off on both sides
  P51 would sit between about 0.876 and 0.900: inside the band either way. That is arithmetic, not a measurement. Amendment 34's box runs
  every arm with both flags off.
- P52 is not re-asked (no reference arm); it held on amendment 25's two boxes.

## Amendment 32 (2026-10-05): triton 3.7.1 alone is not the 5090's environment gain on the matched arm (0.992), and is 2.9 % on the shipped arm

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 32 and 34. One RTX 5090 (`tc1-5090-74`, AMD EPYC 7B13,
Vast machine 145701, host load1 3.0–9.4). venv-e4b (torch 2.8.0) with its own triton 3.4 (`_tr0`) against triton 3.7.1 put first on the
arm's `PYTHONPATH` (`_tr1`), 60-step runs, prebound launches off. Read: [`RESULTS-tc1-tritonab.md`](RESULTS-tc1-tritonab.md).

| arm | `_tr0` s/step | `_tr1` s/step | `_tr1`/`_tr0` | prediction |
|---|---|---|---|---|
| matched (fp32 adapters) | 3.827 / 3.871 | 3.894 / 3.746 | 0.992 [0.968, 1.017] | **P59 FALSIFIED** ([0.82, 0.95]) |
| shipped (bf16 adapters) | 3.117 / 3.023 | 2.976 / 2.986 | **0.971** [0.955, 0.988] | **P60 HELD** ([0.85, 0.98]) |

- **P61 HELD.** Held-out moves −0.0008 on the matched arm and +0.0005 on the shipped arm.
- **By the registered rule the A2000 decomposition does not transfer.** On the A2000 the slice ran device-bound, and triton 3.7.1's faster
  code for grouped-nf4-gemm's kernels was the whole gain. On the 5090's host-bound step that device time is not what binds, and the
  matched arm does not move. The shipped arm gains 2.9 %.
- **What is left.** Amendment 24's 0.882 must come from torch 2.12 and/or transformers 5.5. Both removed host work on the A2000:
  attention-mask handling in torch, and the router's dtype round trip in transformers. Amendment 34 splits them on the 5090. Row
  `e4b.train.triton37.qwen3.5090.2026-10-05`.


## Amendment 31 (2026-10-05): Qwen3-30B-A3B's absmax pair, read over 60 steps; the double-quantized absmax becomes the default

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 28 and 31. One RTX 5090 (`tc1-5090-75`, AMD EPYC 7B13,
Vast machine 145701; host load1 3.8–4.9 throughout): the token `qwen3dqab` with `TC1_STEPS=60`. Read:
[`RESULTS-tc1-dqab-qwen3-60.md`](RESULTS-tc1-dqab-qwen3-60.md).

| | `_dq0` (fp32 absmax) | `_dq1` (double-quantized) |
|---|---|---|
| s/step, two draws (60 steps) | 3.751 / 3.727 (0.6 % apart) | 3.777 / 3.805 (0.7 % apart) |
| peak | 27.44 GB | 26.10 GB |
| held-out at N | 0.7597, 0.7583 | 0.7567, 0.7572 |

- **P56 HELD.** `_dq1`/`_dq0` reads 1.014 [1.007, 1.021], inside [0.97, 1.03], and the peak falls 1.339 GB, inside [1.25, 1.45]. The
  double-quantized absmax costs 1.4 % of the step on Qwen3-30B-A3B and 2.3 % on Mixtral.
- **P58 HELD.** Held-out moves −0.0021 on Qwen3 and −0.0028 on Mixtral, both inside 0.005.
- **By amendment 28's rule, with P57 already HELD, the double-quantized absmax becomes the default for resident training.** That is
  e4b's own PR, citing these boxes. The TC1 harness keeps `E4B_ABSMAX_DQ` explicit, because its boxes are registered instruments.
- On a quiet host the 60-step pairs were stable. Amendment 28's Qwen3 box lost both pairs on a busy one.

## Amendment 29 (2026-10-05): the same-stack pair over 60 steps, unstable again on a busy host; amendment 33 gates the draws on host load

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 29 and 33. One RTX 5090 (`tc1-5090-72`, AMD EPYC 7C13,
Vast machine 45511): the token `qwen3samestack` with `TC1_STEPS=60` and no reference arm. Read:
[`RESULTS-tc1-samestack-box3.md`](RESULTS-tc1-samestack-box3.md).

| arm | s/step (two draws, 60 steps) | stable | host load1 median per draw |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 3.979 / 3.490 | **no** (13.1 %) | 22.9 / 11.0 |
| e4b `fused_attn4_m_t28`, venv-e4b | 3.834 / 4.087 | **no** (6.4 %) | 8.8 / 6.4 |
| Unsloth `ckpt_unsloth_m` | 8.759 / 8.500 | yes (3.0 %) | 13.6 / 14.8 |

- **P50 and P51 UNTESTED, final under amendment 29.** Unsloth and e4b's field-image arm are COMPARABLE to e4b's same-stack arm at N=60.
  There is no reference arm, so EQUIVALENT cannot be read.
- **The box ran on the machine whose load had already spoiled amendment 28's Qwen3 box.** That was found only after this box was queued.
  The 60-step window alone does not survive a busy host: the slow e4b draw ran at twice the other's load.
- **Not readings.** Across the three same-stack boxes, the medians sit near 2.3–2.45 for Unsloth/e4b and 0.90–0.94 for the environment.
- **Amendment 33** adds load-gated draws: an arm whose median host load1 exceeds 6.0 is set aside and run again, at most twice. It re-asks
  P50 and P51 once more, off the busy machines. Its reading is final.

## Amendment 28 (2026-10-05): the double-quantized expert absmax costs Mixtral 2.3 % of its step for 2.04 GB; Qwen3's speed pair was unstable

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 28 and 31. Two RTX 5090s, e4b against itself on the
matched arm, resident, `E4B_ABSMAX_DQ=0` (`_dq0`) against `=1` (`_dq1`), two draws a side in ABBA order. Read (both boxes in one pass):
[`RESULTS-tc1-dqab.md`](RESULTS-tc1-dqab.md).

| family | `_dq0` s/step | `_dq1` s/step | `_dq1`/`_dq0` | peak `_dq0` → `_dq1` | held-out Δ | prediction |
|---|---|---|---|---|---|---|
| Mixtral-8x7B (`tc1-5090-71`) | 3.807 / 3.917 | 3.932 / 3.970 | **1.023** [1.004, 1.043] | 31.07 → **29.03** GB (−2.04) | −0.0028 | **P57 HELD** |
| Qwen3-30B-A3B (`tc1-5090-70`) | 3.801 / 4.019 (5.6 % apart) | 3.947 / 5.014 (23.8 % apart) | not read | 27.15 → 25.82 GB (−1.33) | +0.0026 | **P56 UNTESTED** |

- **Mixtral (P57 HELD).** Double-quantizing the absmax costs 2.3 % of the step and saves 2.04 GB. The registered arithmetic predicted
  2.10 GB.
- **Qwen3 (P56 UNTESTED).** Both speed pairs were unstable. The box's host load average tracked the draws: 6.6 and 7.3 on the `_dq0`
  draws, 12.1 and 19.4 on the `_dq1` draws. Its peak and held-out readings do not depend on the host and sit inside their bands, but
  P56 is one prediction, so it stays UNTESTED.
- **P58 UNTESTED** for the same reason: its Mixtral half reads −0.0028, its Qwen3 half is unread.
- **By the registered rule `E4B_ABSMAX_DQ` stays opt-in for now.** Amendment 31 re-asks Qwen3's pair over 60 steps on a quieter machine.
- **Why boxes go unstable.** The step is host-bound, and on multi-tenant hosts the other tenants' load shows up directly in it:
  - this box (load 6.6–19.4);
  - `tc1-5090-66`'s shipped pair (load 4 vs 19);
  - `tc1-5090-68` (load 30–37 throughout).

  Machine 151350 is the exception: it lost pairs at a load near 1.

## Amendment 26 (2026-10-05): prebound Triton launches take 2.7 % off the matched arm's step; the shipped pair was unstable, so the flags stay opt-in

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 26 and 30. One RTX 5090 (`tc1-5090-69`, Intel Core
Ultra 9 285K, Vast machine 151350). The box ran e4b against itself in venv-e4b (triton 3.4), comparing `E4B_TRITON_PREBIND` and
`GNF4_TRITON_PREBIND` both 0 (`_pb0`) against both 1 (`_pb1`), two draws a side in ABBA order. Read: [`RESULTS-tc1-prebindab.md`](RESULTS-tc1-prebindab.md).

| arm | `_pb0` s/step | `_pb1` s/step | `_pb1`/`_pb0` | held-out at N, `_pb0` / `_pb1` |
|---|---|---|---|---|
| matched (fp32 adapters) | 2.555 / 2.505 | 2.447 / 2.476 | **0.973** [0.958, 0.988] | 0.8524, 0.8493 / 0.8513, 0.8480 |
| shipped (bf16 adapters) | 1.926 / 1.772 (8.3 % apart) | 1.972 / 1.970 | not read | 0.8135, 0.8149 / 0.8113, 0.8116 |

- **P54 HELD.** On the matched arm the prebound launches take 2.7 % off the step. The launches were prebound as registered: the `_pb1`
  arm counted 73,591 prebound e4b launches and 24,546 prebound grouped-nf4-gemm launches, against 313 and 30 through Triton's own path.
  The `_pb0` arm counted none.
- **P53 UNTESTED.** The shipped arm's flags-off draws were 8.3 % apart.
- **The shipped pair, unread, points the other way.** Its `_pb1` draws (1.972, 1.970) are slower than both `_pb0` draws. Had the pair
  been stable, P53 ([0.90, 0.98]) would have read FALSIFIED at about 1.07. Host noise or a real cost on this arm: this box cannot tell.
- **P55 UNTESTED** for the shipped half. The matched half moved 0.0012 nats.
- **By the registered rule the flags stay opt-in.**
- **One machine, several boxes.** This box, `tc1-5090-67` and three earlier stable ones (`-55`, `-62`, `-63`) all ran on one machine.
  The launcher ranks it near the top. Two of its last three boxes lost a pair to instability. The re-asks avoid it.
- **Amendment 30** re-asks the shipped pair over 60 steps (50-step medians) on another machine. Its reading is final.

## Amendment 27 (2026-10-05): the re-draw, also unstable on one pair; amendment 29 re-asks P50 and P51 over 60 steps

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 25, 27 and 29. One RTX 5090 (`tc1-5090-68`, AMD EPYC
7663), the same token on a machine box 1 did not use. Read: [`RESULTS-tc1-samestack-box2.md`](RESULTS-tc1-samestack-box2.md).

| arm | s/step (two draws) | stable |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 4.060 / 3.755 | **no** (7.8 %) |
| Unsloth `ckpt_unsloth_m` | 8.989 / 9.246 | yes (2.8 %) |
| e4b `fused_attn4_m_t28`, venv-e4b | 4.388 / 4.314 | yes (1.7 %) |

- **P52 HELD again.** Unsloth and e4b's reference read INSIDE-DRAW-NOISE against e4b's fused arm, and parity PASSES.
- **P50 and P51 UNTESTED, final under amendment 25.** This time e4b's same-stack pair was the unstable one.
- **The instrument, not a prediction.** Across the two boxes, three different pairs lost stability. Each draw's median is over 10 steps.
- **Amendment 29** re-asks P50 and P51, with their bands unchanged, over 60-step runs (50-step medians) on a third machine; its reading
  is final.

## Amendment 25, first box (2026-10-04): on one stack the matched set holds; the speed pairs were unstable, so one re-draw is registered

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 25 and 27. One RTX 5090 (`tc1-5090-67`, Intel Core
Ultra 9 285K). The box ran:

- e4b's matched set in Unsloth's venv (torch 2.12.1+cu130, transformers 5.5.0);
- Unsloth as in TC1;
- e4b's field-image arm (`_t28`) alongside.

Read: [`RESULTS-tc1-samestack-box1.md`](RESULTS-tc1-samestack-box1.md).

| arm | s/step (two draws) | stable |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 2.220 / 2.188 | yes (1.5 %) |
| Unsloth `ckpt_unsloth_m` | 4.913 / 5.899 | **no** (18.2 %) |
| e4b `fused_attn4_m_t28`, venv-e4b | 2.305 / 2.507 | **no** (8.4 %) |
| e4b `reference_attn4_m`, venv-unsloth | 29.582 (one draw) | — |

- **P52 HELD.** On one stack Unsloth and e4b's reference read EQUIVALENT to e4b's fused arm (held-out Δ 0.0039 and 0.0035 against a
  band of 0.0105), and e4b's parity PASSES (Δ final 0.0038).
- **P50 and P51 UNTESTED.** Two of the three speed pairs were unstable. The GPU logs show steady SM clocks and a host load average near
  1.2, so they do not explain it.
- **No ratio is read.** The medians would read about 2.45 for Unsloth/e4b and 0.92 for the environment, but those are not readings and
  are not quoted.
- **Amendment 27** registers one re-draw on another machine with amendment 25's predictions unchanged. Its reading is final.

## Amendment 24 (2026-10-04): the 5090's fp32 `bmm` costs host time per new shape, not per torch version; in Unsloth's environment e4b's matched arm steps 0.882×

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 24. One RTX 5090 (`tc1-5090-66`, AMD EPYC 7B13). The box
ran two parts:

1. A replay with no model. `bmm_bench.py` timed the padded LoRA delta's batched products at the 96 recorded real-router shapes, with
   the queue drained before each call.
2. e4b against itself in two environments, on TC1's qwen3 tokens and recipe, two draws a side in ABBA order:
   - `_tv0`: the field image's torch 2.8.0+cu128, transformers 5.18.0, triton 3.4.0;
   - `_tv1`: Unsloth's venv, torch 2.12.1+cu130, transformers 5.5.0, triton 3.7.1.

Read: [`RESULTS-tc1-bmmab.md`](RESULTS-tc1-bmmab.md).

**The replay** (median host µs per forward `bmm`):

| torch | fp32, recorded shapes | bf16 | fp32, new shapes | fp32, repeated shapes | fp32, one fixed shape |
|---|---|---|---|---|---|
| 2.8.0+cu128 | 88.9 | 38.3 | **119.4** | **37.9** | 37.4 |
| 2.12.1+cu130 | 93.4 | 38.3 | **120.9** | **37.9** | 35.9 |

- **P44 FALSIFIED.** Outside training, the fp32 call costs 89 µs at the recorded shapes, 2.3× bf16's. The profiles' 197–324 µs
  (under the profiler, mid-training) does not reproduce.
- **P45 HELD.** What does reproduce is a per-shape cost: a shape the process has not seen costs 119 µs, and a repeated one 38 µs,
  the same as bf16. The routing makes nearly every call's padded shape new.
- **P46 FALSIFIED.** torch 2.12.1+cu130 pays the same cost (120.9 µs).
- The cublasLt and TF32 rows are descriptive and cost more on new shapes (223–269 µs). The bucket rows are not a clean first
  exposure: the cold passes had already used most of the bucketed widths.

**The environment A/B:**

| arm | `_tv0` s/step | `_tv1` s/step | `_tv1`/`_tv0` | held-out at N, `_tv0` / `_tv1` |
|---|---|---|---|---|
| matched (fp32 adapters) | 3.915 / 3.895 | 3.495 / 3.394 | **0.882** [0.867, 0.897] | 0.8522, 0.8531 / 0.8467, 0.8483 |
| shipped (bf16 adapters) | 3.009 / 3.171 (5.3 % apart) | 2.950 / 2.897 | not quoted | 0.8160, 0.8104 / 0.8124, 0.8133 |

- **P47 HELD.** In Unsloth's environment e4b's matched arm steps 0.882× as long, with held-out 0.005 lower. The peak is unchanged at 27.2 GB.
- **P48 and P49 UNTESTED.** The shipped arm's torch 2.8 draws were 5.3 % apart, so its pair cannot be quoted. Its medians read 0.946.
- The replay shows the `bmm` is not the cause: torch 2.12 pays the same per call. The environment changes torch, transformers and
  triton together, so this box does not say which one gives the gain.

**What follows, as registered.**

- P44 failed, so grouped-nf4-gemm does not change.
- P46 failed, so the docs do not change.
- The rule that would have named an environment asymmetry in STATUS needed P48 as well, and P48 went untested.
- What stands is P47's reading. Every 5090 position so far ran e4b in the field image's environment and Unsloth in torch 2.12.1's.
  e4b requires only `torch>=2.2` and `transformers>=5.0`, so both are e4b environments.
- A position with both frameworks on one stack needs its own registration. Rows: `e4b.train.env-ab.qwen3.5090.2026-10-04` and
  `e4b.train.bmm-host-replay.5090.2026-10-04`.

## Amendment 23 (2026-10-04): a memory census of e4b against Unsloth — with the absmax double-quantized, e4b's peak is 0.43 GB above Unsloth's, all of it transient

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 23. One RTX 5090 (`tc1-5090-65`), Qwen3-30B-A3B's matched
set at micro-batch 1 × accum 8, one draw per arm. `tc1_arm.py --mem-census 1` recorded the allocator's history from before the load,
and reduced its snapshots on the box. The census slows the step, so no speed is read. Read: [`RESULTS-tc1-memcensus.md`](RESULTS-tc1-memcensus.md).

| live at the peak (GB) | e4b, fp32 absmax (default) | e4b, `E4B_ABSMAX_DQ=1` | Unsloth |
|---|---|---|---|
| frozen expert weights (NF4) | 14.496 | 14.496 | 14.496 |
| expert absmax | **1.812** | 0.460 | 0.460 |
| other frozen (bf16 1.270, NF4 attention 0.468) | 1.738 | 1.738 | 1.738 |
| adapters / their grads / optimizer state | 2.570 / 2.570 / 1.305 | 2.570 / 2.570 / 1.305 | 2.570 / 2.570 / 1.305 |
| transient | **1.532** | **1.536** | 1.095 |
| **peak allocated** | **26.022** | **24.676** | **24.244** |

- **The census measures what it should (P41 HELD).** It finds e4b's fp32 absmax at exactly the analytic 1.8119 GB (29.0 B expert
  parameters / 64 × 4 bytes), and the double-quantized absmax at 0.4602 GB, 3.94× smaller.
- **It accounts for nearly everything (P42 HELD).** At least 99.96 % of each arm's peak goes to named groups.
- **The gap is smaller than registered (P43 FALSIFIED, below [0.5, 2.5] GB).** Every static class is byte-for-byte the same in both
  frameworks except the absmax. With it double-quantized, e4b's peak sits **0.43 GB** above Unsloth's, and the whole gap is transient.
- **What the transients are.** e4b's largest are grouped-nf4-gemm's padded LoRA delta (`kernel/nf4_qlora.py:_lora_delta_padded`):
  - the zero-padded input block it allocates in the adapters' dtype, fp32 on this arm (0.62 GB, 2 live);
  - the batched products' outputs (0.45 GB, 3 live).

  Unsloth's largest transient is its NF4 dequant (0.81 GB).
- **At e4b's defaults the gap is 1.78 GB, and 1.35 GB of it is the fp32 absmax.** Those defaults are what this lane has quoted;
  `E4B_ABSMAX_DQ=1` removes the absmax part.
- **As registered, this is a measurement.** Each fix the read names gets its own A/B. The candidates are the absmax default and the
  padded block's dtype and size on fp32 adapters. Row `e4b.train.memory-census.qwen3.5090.2026-10-04`.

## Amendment 22 (2026-10-04): grouped-nf4-gemm's dense route against its fused kernels — far faster on Mixtral, far slower on Qwen3-30B-A3B

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 22. Each box ran e4b against itself on the matched arm,
`GNF4_TRAIN_GEMM=fused` against `=dense` (grouped-nf4-gemm#459: one present expert dequantized at a time, its GEMM through `torch.mm`).
There were two draws a side in ABBA order, with grouped-nf4-gemm at #459's merge. Read: [`RESULTS-tc1-denseab.md`](RESULTS-tc1-denseab.md),
both boxes in one pass.

| family | box, host | fused s/step | dense s/step | **dense/fused** | held-out Δ (dense − fused) | prediction |
|---|---|---|---|---|---|---|
| Mixtral-8x7B (2 of 8 experts), resident, `E4B_ABSMAX_DQ=1` both sides | `tc1-5090-62`, Core Ultra 9 285K | 5.491 / 5.541 | 3.592 / 3.589 | **0.651** [0.648, 0.654] | −0.0021 | **P38 HELD** ([0.55, 0.90]) |
| Qwen3-30B-A3B (up to 128 present experts) | `tc1-5090-59`, EPYC 9454P | 2.949 / 2.976 | 8.674 / 8.784 | **2.947** [2.915, 2.979] | −0.0001 | **P39 FALSIFIED** ([0.85, 1.15]) |

**P40 HELD:** both families' held-out stay within 0.01, and the peaks do not move (Mixtral 29.03 / 29.02 GB).

**Why the two families split.** The dense route trades the fused kernels' device time for launches. Each present expert costs a dequant and
a GEMM, in every projection call. On Mixtral that is 8 experts per call, and the expert GEMMs are large (about 1,000 rows each), so the
device-side gain dominates. An RTX A2000 replay put the dense forward at about 0.3× and its dgrad at about 0.13× of the fused kernels on
those shapes. On Qwen3-30B-A3B a step makes about 1,150 projection calls with up to 128 experts each, roughly 295,000 extra launches. On a
5090, where this step is already launch-bound, that cost dominates.

**Step-0 held-out on Qwen3 moves with every GEMM implementation**, not only this route: the fused kernels read 1.9714 on an H100 and
1.9441 on a 5090, the reference loop 1.9551 / 1.9508, the grouped_mm route 1.9614, and the dense route 1.9672. All of them train to the
same held-out (here Δ −0.0001).

**Decision, as registered.** P38 and P40 held, so grouped-nf4-gemm's `auto` takes the dense route on cards other than sm_90 for calls
with at most 16 present groups (grouped-nf4-gemm#463). Qwen3-30B-A3B's calls stay on the fused kernels. Rows
`e4b.train.dense-route.mixtral.5090.2026-10-04` and `e4b.train.dense-route.qwen3.5090.2026-10-04`. No position against another framework is
read here; the Mixtral position with the new default needs its own box.

## The position (register `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02`)

On one RTX 5090 at the tp4 field recipe — alpaca, seq 2048, micro-batch 2 × accum 4, r 16 / alpha 16, lr 2e-4, AdamW-8bit, linear
warm-up 5, N = 20 — with **the same 642,514,944 trainable parameters, the same per-slot LoRA A init (`matched:3407`, name-free
sha `f7832488926eda91` in every matched arm, 12,480 / 12,480 slots), fp32 adapters in both frameworks, the same tokens
(sha `bfc742f67e37`)**, and Unsloth 2026.9.14 on the route its installer names for Blackwell (torch 2.12.1+cu130, the `grouped_mm`
backend engaged on every step: 384 `torch._grouped_mm` calls per step, 0 loop calls):

| | e4b `fused_attn4_m` | Unsloth `ckpt_unsloth_m` | reading |
|---|---|---|---|
| s/step, median of steps 11..20, two draws | 5.692 / 5.683 (**5.688**) | 8.160 / 8.183 (**8.171**) | **Unsloth/e4b 1.437 [1.434, 1.440]** over the four cross-draw ratios; both STABLE (0.2 % / 0.3 %) |
| tokens/s | 265.2 | 176.4 | |
| peak VRAM | 27.83 GB | **24.27 GB** | Unsloth lower by 3.57 GB |
| energy per step (dmon) | 921.3 J | **660.7 J** | Unsloth ×0.72 |
| held-out loss at N = 20 (8 rows) | 1.9505 → 0.8516 / 0.8487 | 1.9583 → 0.8482 / 0.8476 | Δ −0.0034 ± 0.0028 SE: **EQUIVALENT** (band 0.0123 = 3 × the in-draw fused-vs-reference Δ; draw-noise floor 0.0030); step-0 Δ 0.0078 SAME-BYTES-CLASS |
| frozen base | C1 bit-exact, the real-storage flip control detects | C1 bit-exact, control detects | attention projections SAME-BYTES (nf4/64 + double-quant); expert stacks N-A (e4b nf4/64, Unsloth nf4/64 + dq: its loader exposes no double-quant knob) |

e4b's fused path against its own per-expert reference on the same box: Δ final train loss 0.00041, median step |Δ| 0.00141 — PASS;
×10.39 faster per step than the reference (5.692 vs 59.171 s) at ×1.03 its peak. The secondary pair (micro-batch 1 × accum 8, run
because HF OOMed) reads 1.725 the same way (15.021 vs 8.706 s, single draws).

**The standing position is refuted and superseded.** `e4b.train.h2h.unsloth.qwen3.5090.2026-09-19` quoted 4.490 (6.47 vs 29.05 s/step)
at this recipe. That Unsloth arm ran on the field image's torch 2.8.0+cu128, where `torch._grouped_mm` is sm_90-only and Unsloth's
loader silently selects its `native_torch` per-expert loop; the comparison was e4b's fused kernel against Unsloth's fallback. TC1's
registered prediction P1 put the matched ratio in [2.0, 5.0] with "below 1.5 refutes the standing position": it reads **1.437,
FALSIFIED**, and this bundle's row supersedes the 2026-09-19 row (its `.quality-n20`, `.e4b-internal-parity` and `.secondary-mb1`
rows likewise). The loop is still what a torch-2.8 user gets: on the native box the same Unsloth install on torch 2.8 (`ckpt_unsloth_t28`,
backend `native_torch`, 96 loop calls per step) takes 61.3 s/step, 6.565 × e4b's matched arm on that host.

## What the ratio measures, and what it does not

- **Both steps are host-bound on a 5090, and e4b's is the less host-bound of the two.** The profiled matched arms on the same box:
  e4b device-busy 0.480, 141 k device events and 753 k CPU-side ops per step; Unsloth device-busy 0.234, 612 k device events and
  7.43 M CPU-side ops per step (P8, which predicted ≥ 0.5 on the grouped_mm arm, is FALSIFIED). The fused path records 4.3 × fewer
  device events and 9.9 × fewer CPU-side ops per step than the dequant-then-dense-grouped-GEMM path, and that dispatch cost — not the
  GEMM — is where the gap is on this card.
- **Absolute s/step does not travel between hosts; the ratio does.** The same e4b matched arm takes 3.16 s/step on the i9-13900 host,
  5.69 on the Threadripper host and 9.34 on the Xeon E5-2698 v4 host (same card class, same code, 0 recompiles); the Unsloth matched
  arm tracks it (4.50 / 8.17). TC1b's P3 ("the 200-step and 20-step medians agree within 10 %") is FALSIFIED mechanically because the
  reducer compares across those two hosts (−44 %); within one draw the 11..20 and 11..200 windows agree within 3.5 %. Nothing on this
  page divides numbers from different boxes; every ratio is within one box.
- **e4b on the cu130 torch is 14 % faster per step** (`fused_attn4_m_t212`: e4b + grouped-nf4-gemm installed into the comparator's
  torch 2.12.1+cu130 venv, 8.069 vs 9.336 s on the native box; VALID, EQUIVALENT-class quality). The position above is e4b on the
  field image's torch 2.8; a position with both frameworks on cu130 is not quoted here (one draw, one host).
- **Unsloth's footprint rows are in its favour** (peak VRAM −3.57 GB, energy ×0.72 per step at the matched configuration). The
  matched configuration carries fp32 expert adapters on e4b's side by design; e4b as shipped (bf16 expert adapters) peaks at 24.58 GB
  and 843 J/step on the native box — the same footprint as Unsloth's — so the gap is the matched set's precision choice, not the kernel.
  Peak VRAM and J/step are quoted beside the ratio wherever it is quoted.

## The native and labelled rows (box `tc1-5090-14`, each against the box's own e4b matched arm, single draws)

| row | s/step | vs e4b matched (9.336) | held-out at N (e4b matched 0.8481) | note |
|---|---|---|---|---|
| Unsloth native-best (grouped_mm, speed tilt, native init, fp32) | 15.245 | 1.633 | 0.8467 | vs e4b as shipped (6.267): **2.433** — the "both at their own defaults" row, never the position; P7 (≥ 2 vs e4b matched) FALSIFIED |
| Unsloth torch-2.8 (`native_torch` loop, tp4's venv) | 61.295 | 6.565 | 0.8471 | 23.9 tok/s, 3885 J/step; P1b (within 15 % of tp4's 29.05) FALSIFIED on this slower host |
| Unsloth `unsloth_triton` backend | 14.754 | VOID | — | requested backend never engaged (`moe_backend_selected grouped_mm`, 0 triton calls): a row, not a position |
| e4b as shipped (bf16 expert adapters, N(0, 1/r) init) | 6.267 | **0.671** | **0.8108** | 1.49 × faster than the matched e4b arm and 0.037 nats LOWER held-out at N = 20; P4 (1.05–1.3 × and ≥ 0.01 lower) FALSIFIED on the speed half |
| e4b `dgrad=False` (`enable_fast_train`'s default) | 9.248 | 0.991 | 0.8512 | the dgrad kernel is not where the step goes at this shape |
| e4b on torch 2.12.1+cu130 | 8.069 | 0.864 | 0.8506 | see above |
| axolotl native-best (scattermoe) | — | — | — | INSTALL_FAILED here (the harness's uv index strategy, amendment 3); on the re-ask box REFUSED at load: the KernelsPlugin fetches `kernels-community/rotary` and the harness runs arms with the Hub offline — a harness limit, not a reading |
| HF on torch 2.14 + `grouped_mm` at mb1 | — | — | — | NOT_RUN here (its gate reads the matched box's OOM); on the re-ask box OOM at load (31.35 GB in use): bf16 expert stacks do not fit the 32 GB card under any experts implementation |

HF + PEFT (transformers 5.18.0, bf16 expert stacks, `target_parameters`) OOMs on the 32 GB card at micro-batch 2 and at micro-batch 1
(31.34 GB in use at a 20 MiB allocation), and so does the torch-2.14 `experts_implementation="grouped_mm"` variant at micro-batch 1 on the
re-ask box: P5 HELD. No HF position exists on this card at this recipe.

> **Correction (2026-10-02, TC1 amendment 4).** The axolotl failure below was this harness's, not axolotl's. axolotl's loader keeps every
> `*.gate` router in fp32 on purpose (`loaders/model.py:611-616, 1449-1451`) because its own trainer runs the forward under bf16 autocast;
> this harness runs no autocast, so the fp32 router met bf16 activations. The receipts show the router in float32 on every failing axolotl
> arm and in bfloat16 on the HF and e4b arms. `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.axolotl-unsupported` is retired, and the arm is
> re-asked with the routers cast to bf16 after load (what autocast computes per call).

> **The re-run (`tc1-5090-30`, 2026-10-02, $0.43; `RESULTS-tc1-5090-30-axolotl.md`).** With the routers cast, axolotl trains the matched set:
> 7.759 / 7.642 s/step against e4b's 5.408 / 5.464 — **axolotl/e4b 1.416 [1.398, 1.435]**, both pairs stable, e4b faster per step; axolotl's peak
> 26.88 GB against 27.81 and its energy ×2.31; held-out COMPARABLE (about 0.01 nats). Its 96 expert stacks are NF4 (axolotl's parametrized
> `quantize_moe_experts`, double-quantised statistics), and its step-0 loss sits 0.026 above e4b's: the two quantisers' bytes differ. axolotl's
> scattermoe native-best trains too, at **0.929 × e4b's matched step** and 25.84 GB (one draw, its own init — a labelled row). TC1 P6 FALSIFIED:
> axolotl trains, faster than the [1.5, 6] band. Register `e4b.train.h2h.axolotl.qwen3.5090.2026-10-02` and `.scattermoe-native`.

> **Native-best against native-best (`tc1-5090-33` and `tc1-5090-34`, amendments 5-7; `RESULTS-tc1-nativebest.md`).** Each framework as its
> users run it, on one box, two draws each. On `tc1-5090-34` **Unsloth native-best / e4b as shipped is 1.794 [1.790, 1.797]**, both pairs
> stable: P13's Unsloth half HELD (register `.native-vs-native`, a labelled row). axolotl's scattermoe draws were 10.8 % apart, with
> 216, 61-65 and 76-81 s warm-up steps inside the window, so P13 is UNTESTED and no "fastest" statement is made. Amendment 8 re-asks the
> axolotl half over steps 101..200 of a 200-step run (P14).
>
> **P14 (`tc1-5090-35`, amendment 8).** Over steps 101..200 of a 200-step run, **axolotl scattermoe / e4b as shipped is 0.911
> [0.902, 0.921]**: P14 HELD, and axolotl is faster at steady state (register `.native-steady-state`). e4b still finishes the 200-step run
> first, because of axolotl's warm-up, and axolotl's held-out matches the matched curve while e4b as shipped sits 0.024-0.027 above it.
>
> **P15 (`tc1-5090-36`, amendment 9).** The same box on a second host (Ryzen 9 3900X, machine 45501): **0.901 [0.892, 0.910]**, P15 HELD.
> axolotl's steady-state lead over e4b as shipped holds on two hosts with different CPU classes, and e4b again finishes the 200-step run first.

**axolotl 0.20.0 on Qwen3-30B-A3B at its pins (torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0) does not train** (box `tc1-5090-20`,
with the venv installing after amendment 3): both matched draws loaded (`quantize_moe_experts` packed all 96 stacks, PEFT's
`target_parameters` on the expert stacks, the attention projections by module) and died in the first forward inside transformers'
`Qwen3MoeTopKRouter` — `F.linear(hidden_states, self.weight)`: bf16 activations against an fp32 router weight — the same exception on
the 24 GB and H100 boxes (five attempts, three cards). The router was not a target; the HF arm's own loader (transformers 5.18.0, the
same model, the same quantizer) builds a bf16 router and trains; the router code is identical in 5.17.0 and 5.18.0, so the fp32 weight
is a property of the model axolotl's `ModelLoader` hands over — whether its post-quantisation conversion or 5.17.0's bnb path leaves it,
this bundle does not establish. On those boxes the harness wrote no receipt for the crash (it died after load, outside the loader's
own handler — TC3 amendment 4 makes such a crash a row), so the mechanical verdict is HARNESS_ERROR and P6 reads UNTESTED; by the
logs the row is UNSUPPORTED, and no axolotl position is quoted on this family. axolotl trains Granite's 4-bit path in lane TC2 (its
own bundle), so this is the family, not the framework.

## TC1b — the long curve re-asked with matched init and precision (box `tc1-5090-15`)

200 steps, 16 held-out rows every 40 steps, paired by row:

| eval step | e4b matched | Unsloth matched | e4b as shipped | Δ Unsloth − e4b | Δ shipped − matched |
|---|---|---|---|---|---|
| 0 | 1.9607 | 1.9604 | 1.9607 | −0.0004 | 0 |
| 40 | 0.7880 | 0.7885 | 0.7894 | +0.0005 | +0.0014 |
| 80 | 0.7709 | 0.7724 | 0.7741 | +0.0015 | +0.0032 |
| 120 | 0.7665 | 0.7669 | 0.7674 | +0.0004 | +0.0009 |
| 160 | 0.7626 | 0.7646 | 0.7671 | +0.0020 | +0.0045 |
| 200 | 0.7687 | 0.7688 | 0.7945 | +0.0001 | **+0.0258** |

- **EQUIVALENT at every eval** (largest paired |Δ| 0.0020, at step 160; TC1b P1 HELD): with init, precision, tokens and schedule matched,
  the two 4-bit MoE training paths produce the same curve to 200 steps. s/step over steps 11..200: 3.162 vs 4.505 (1.425 within this
  box, consistent with the position's 1.437).
- **The as-shipped e4b curve ends 0.0258 nats above the matched one at step 200** while being faster per step (2.527 s, 1.25 ×) — the
  direction P38 reported against e4b, now reproduced on e4b's own side with the comparator out of the picture: the loader's defaults
  (bf16 expert adapters, N(0, 1/r) init) are the cause, not the kernel. It is NOT P38's shape (TC1b P2 FALSIFIED): the shipped curve
  is above the matched one at step 40 too (+0.0014), with no early lead; on the N = 20 instrument (8 rows) it reads 0.037 LOWER.
  The two instruments disagree at step 20–40 and agree from 80 on; an open item for `load_moe_4bit_streaming`'s adapter defaults.
- **The tp2/P38 anchor pair does not reproduce** (TC1b P4 FALSIFIED): at that fixture (clinical text, seq 512, 1 × 1, r 8, N 60) the ratio
  is 2.428 with Unsloth on cu130 grouped_mm (1.095 vs 0.451 s/step) and 5.147 with Unsloth on torch 2.8 (2.322 s/step) against
  tp2's 1.457 and P38's 1.413 (e4b 0.35.0 / grouped-nf4-gemm 0.30.0 vs Unsloth 2026.9.2). Both frameworks moved between those
  cuts and these; the old anchors stand as dated measurements on their cuts and are not repeated as current.
- Scaling points, never positions: at seq 2048 × micro-batch 1 × accum 1 the matched ratio is 1.918 (1.073 vs 0.559 s); at r 64 /
  alpha 64 both frameworks OOM at step 1 on the 32 GB card (31.9 / 32.4 GB).

## Predictions, scored mechanically

TC1 (`RESULTS-tc1-combined.md`): P1 FALSIFIED (1.437, below 1.5), P1b FALSIFIED (host), P2 HELD (draws), P3 HELD (equivalence), P4
FALSIFIED (shipped 1.49 ×, not 1.05–1.3 ×; the quality half held), P5 HELD (HF OOM, both implementations), P6 **UNTESTED** (on the matched and native boxes the
INSTALL_FAILED was this harness's own install line; on the axolotl box the framework's crash left no receipt — UNSUPPORTED by the logs,
not a mechanical reading), P7 FALSIFIED (1.633 < 2 within its box), P8 FALSIFIED (device-busy 0.234), P9 HELD (parity),
P10 UNTESTED (the expert slots are different regimes by construction). TC1b: P1 HELD, P2 FALSIFIED, P3 FALSIFIED (cross-host),
P4 FALSIFIED.

## Harness defects found by the boxes (fixed in amendment 3, `../../tc1/TC1-PREREG.md`)

1. The axolotl venv never installed: uv's first-index strategy took `packaging` from PyTorch's cu130 index, where axolotl's
   `packaging==26.0` does not exist. Reproduced off-box and fixed with `--index-strategy unsafe-best-match` (dry-resolved: axolotl 0.20.0,
   torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0). The axolotl rows are re-asked on their own box (`qwen3axolotl`); this bundle
   quotes no axolotl row.
2. The reducer read the profiled arms' kernel-table sidecars as HARNESS_ERROR rows (cosmetic; skipped now).
3. The native box's HF torch-2.14 row's gate could not be read on that box (re-asked with the axolotl rows: OOM).
4. Found by the re-ask box and fixed after it (TC3 amendment 4): an exception after load left no receipt; the scattermoe row needs the
   KernelsPlugin's kernels fetched before the Hub goes offline (open).

## Not claimed here

No position on any other family (lane TC2) or under a memory budget (lane TC3) — each has its own registration and bundle; no axolotl
position on this family (it does not train it at its pins, above). No cross-box ratio. **The H100 NVL reading is the opposite sign** — Unsloth/e4b 0.621, Unsloth faster — see
[`../tc1c/README.md`](../tc1c/README.md): the position above is a 5090 position, where both paths are launch-bound and Unsloth's grouped
GEMM is not one launch per call.
