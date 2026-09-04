# Build-out validation lane bo3 — 2026-09-04, one RTX 5090

**Box:** Vast.ai instance 49817195, RTX 5090 (sm_120) on an AMD EPYC 9755 host (see
[`forensics.txt`](forensics.txt)); transformers 5.16.1, torch 2.8 (`pytorch/pytorch:2.8.0-cuda12.8`).
**Protocol:** the Qwen3-30B campaign's — per family an NF4 bake of the released checkpoint, then arms
under `bench/hybrid-g9/step_decomp.py`: K8 teacher-forced NLL (2048 steps, wikitext, sha-matched windows,
`--ppl-source wikitext --b1d-loop eager`), B=1 decode (512-token prompt, 128 generated, graph loop, timed
window, `--no-fuse-qkv`, fp8 paged KV, placement all-vram) and B=16 aggregate (graph loop, 70 steps).
One arm PER FUSION: a refusal is a row, never the end of a family. The lane ran 17 phases (b–q) over
03:20–08:50Z on successively newer cuts; the phase scripts are in [`logs/`](logs/) with their pip pins.

## The table

Every row is one `run_<family>_<kind>_<arm>.log` in [`logs/`](logs/) with its JSON beside this file
(`<family>_<kind>_<arm>.json`), reduced by [`buildout_reduce.py`](buildout_reduce.py). The gate column
applies the registered rule (`experts4bit_qlora.k8_gate`, 0.28.0) in its own units — perplexity, not
nats: uncalibrated |Δppl| ≤ 0.05; a calibrated pack ≤ +0.05 one-sided on ≥ 2 texts (`pass*` = one text).
Where a family has no instrument, the column says so instead of a verdict.


### Qwen3-30B-A3B — arithmetic-order floor 0.0095 nats
| arm | configuration | K8 nll | Δ nats | Δ ppl | gate (k8_gate: uncal |Δppl|≤0.05; calib ≤+0.05 one-sided) | B=1 ms | B=1 tok/s | ×NF4 | B=16 tok/s | ×NF4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline) | 1.86213 | 0.0000 | 0.0000 | baseline | 10.20 | 98.0 | 1.00 | — | — |
| `all` | int4 + calib attn + r1 + r2 + epilogue | 1.85053 | -0.0116 | -0.0742 | pass* | 6.41 | 156.0 | 1.59 | 1089.6 | — |
| `all_rope` | all + round-2 fold on unfused attention (#375) | 1.84944 | -0.0127 | -0.0812 | pass* | 5.62 | 177.9 | 1.81 | — | — |
| `head` | NF4 + calibrated int4 output head only (#373) | 1.87176 | 0.0096 | 0.0623 | FAIL | 9.92 | 100.8 | 1.03 | — | — |
| `allhead` | all + calibrated int4 head | 1.85898 | -0.0031 | -0.0202 | pass* | 6.15 | 162.6 | 1.66 | — | — |

### Granite-3.1-3B-A800M — arithmetic-order floor 0.0033 nats
| arm | configuration | K8 nll | Δ nats | Δ ppl | gate (k8_gate: uncal |Δppl|≤0.05; calib ≤+0.05 one-sided) | B=1 ms | B=1 tok/s | ×NF4 | B=16 tok/s | ×NF4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline) | 1.67407 | 0.0000 | 0.0000 | baseline | 5.27 | 189.8 | 1.00 | 1436.2 | 1.00 |
| `int4exp` | + int4 experts | 1.68587 | 0.0118 | 0.0633 | FAIL | 3.85 | 259.7 | 1.37 | 2201.3 | 1.53 |
| `calib` | int4 experts + C4-calibrated int4 attention | 1.70028 | 0.0262 | 0.1416 | FAIL | 4.62 | 216.5 | 1.14 | — | — |
| `r1` | round-1 norm folds only | 1.67927 | 0.0052 | 0.0278 | pass | 4.63 | 216.0 | 1.14 | — | — |
| `r12` | round-1 + round-2 folds only | 1.67927 | 0.0052 | 0.0278 | pass | 4.51 | 221.7 | 1.17 | — | — |
| `epi` | router epilogue only | 1.67492 | 0.0009 | 0.0045 | pass | 5.04 | 198.4 | 1.05 | — | — |
| `stack` | int4 experts + r1 + epilogue | 1.68449 | 0.0104 | 0.0559 | FAIL | 3.31 | 302.1 | 1.59 | 2581.6 | 1.80 |
| `all` | int4 + calib attn + r1 + r2 + epilogue | 1.70160 | 0.0275 | 0.1489 | FAIL | 3.63 | 275.5 | 1.45 | 2631.2 | 1.83 |
| `stack_old` | stack, old cut (control) | — | — | — | — | 3.31 | 302.1 | 1.59 | — | — |
| `stackr2_old` | stack + r2, old cut | 1.68449 | 0.0104 | 0.0559 | FAIL | 3.19 | 313.5 | 1.65 | 2624.8 | 1.83 |
| `stack_new` | stack, new cut (control) | — | — | — | — | 3.31 | 302.1 | 1.59 | — | — |
| `stackr2_new` | stack + r2 + rope-only fold (#379) | 1.68190 | 0.0078 | 0.0419 | pass | 2.76 | 362.3 | 1.91 | 2868.6 | 2.00 |
| `nf4_r12epi_new` | NF4 experts + r1 + r2 + epilogue, with the rotary-only fold (#379) — Granite's licensed stack | 1.67767 | 0.0036 | 0.0192 | pass | 3.86 | 259.1 | 1.37 | 1689.6 | 1.18 |
| `nf4_r12epi_old` | NF4 experts + r1 + r2 + epilogue, before the rotary-only fold | 1.67566 | 0.0016 | 0.0085 | pass | 4.28 | 233.6 | 1.23 | 1603.6 | 1.12 |
| `nf4_r1epi_new` | NF4 experts + r1 + epilogue (round-2 off, control) | — | — | — | — | 4.40 | 227.3 | 1.20 | — | — |
| `nf4_r1epi_old` | NF4 experts + r1 + epilogue (round-2 off, control) | — | — | — | — | 4.40 | 227.3 | 1.20 | — | — |

### Gemma-4-26B-A4B-it — no stable floor (see SERVING-PARITY)
| arm | configuration | K8 nll | Δ nats | Δ ppl | gate (k8_gate: uncal |Δppl|≤0.05; calib ≤+0.05 one-sided) | B=1 ms | B=1 tok/s | ×NF4 | B=16 tok/s | ×NF4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline) | 4.81029 | 0.0000 | 0.0000 | no instrument (chaos band) | 13.96 | 71.6 | 1.00 | 574.1 | 1.00 |
| `int4exp` | + int4 experts | 4.83107 | 0.0208 | 2.5778 | no instrument (chaos band) | 11.63 | 86.0 | 1.20 | 797.2 | 1.39 |
| `calib` | int4 experts + C4-calibrated int4 attention | 4.72623 | -0.0841 | -9.8980 | no instrument (chaos band) | 11.07 | 90.3 | 1.26 | — | — |
| `r1` | round-1 norm folds only | 4.72056 | -0.0897 | -10.5361 | no instrument (chaos band) | 10.72 | 93.3 | 1.30 | — | — |
| `r12` | round-1 + round-2 folds only REFUSED(b1,ppl) | — | — | — | — | — | — | — | — | — |
| `epi` | router epilogue only | 4.71993 | -0.0904 | -10.6068 | no instrument (chaos band) | 13.82 | 72.4 | 1.01 | — | — |
| `stack` | int4 experts + r1 + epilogue | 4.66820 | -0.1421 | -16.2614 | no instrument (chaos band) | 8.26 | 121.1 | 1.69 | 962.8 | 1.68 |
| `all` | int4 + calib attn + r1 + r2 + epilogue REFUSED(b16,b1,ppl) | — | — | — | — | — | — | — | — | — |
| `head` | NF4 + calibrated int4 output head only (#373) | 4.81643 | 0.0061 | 0.7561 | no instrument (chaos band) | 13.42 | 74.5 | 1.04 | — | — |
| `allhead` | all + calibrated int4 head REFUSED(b1,ppl) | — | — | — | — | — | — | — | — | — |
| `best` | int4 + calib attn + r1 + epilogue | 4.90318 | 0.0929 | 11.9503 | no instrument (chaos band) | 7.68 | 130.2 | 1.82 | 960.7 | 1.67 |
| `bestdense` | best + calibrated int4 dense MLP (#378) | 4.69438 | -0.1159 | -13.4362 | no instrument (chaos band) | 7.60 | 131.6 | 1.84 | — | — |
| `bestdensehead` | bestdense + int4 head | 4.69199 | -0.1183 | -13.6972 | no instrument (chaos band) | 7.05 | 141.8 | 1.98 | — | — |

### Mixtral-8x7B-Instruct — no stable floor (see SERVING-PARITY)
| arm | configuration | K8 nll | Δ nats | Δ ppl | gate (k8_gate: uncal |Δppl|≤0.05; calib ≤+0.05 one-sided) | B=1 ms | B=1 tok/s | ×NF4 | B=16 tok/s | ×NF4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline) | 1.18048 | 0.0000 | 0.0000 | baseline | 20.83 | 48.0 | 1.00 | 186.7 | 1.00 |
| `int4exp` | + int4 experts | 1.16898 | -0.0115 | -0.0372 | pass | 10.05 | 99.5 | 2.07 | 367.2 | 1.97 |
| `calib` | int4 experts + C4-calibrated int4 attention | 1.19159 | 0.0111 | 0.0364 | pass* | 9.32 | 107.3 | 2.23 | — | — |
| `r1` | round-1 norm folds only | 1.18076 | 0.0003 | 0.0009 | pass | 20.25 | 49.4 | 1.03 | — | — |
| `r12` | round-1 + round-2 folds only | 1.18076 | 0.0003 | 0.0009 | pass | 20.22 | 49.5 | 1.03 | — | — |
| `epi` | router epilogue only | 1.17782 | -0.0027 | -0.0086 | pass | 20.61 | 48.5 | 1.01 | — | — |
| `stack` | int4 experts + r1 + epilogue | 1.17356 | -0.0069 | -0.0225 | pass | 9.73 | 102.8 | 2.14 | 372.0 | 1.99 |
| `all` | int4 + calib attn + r1 + r2 + epilogue | 1.19408 | 0.0136 | 0.0446 | pass* | 9.09 | 110.0 | 2.29 | 373.4 | 2.00 |

### gpt-oss-20b — arithmetic-order floor 0.0176 nats
| arm | configuration | K8 nll | Δ nats | Δ ppl | gate (k8_gate: uncal |Δppl|≤0.05; calib ≤+0.05 one-sided) | B=1 ms | B=1 tok/s | ×NF4 | B=16 tok/s | ×NF4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline) | 6.33544 | 0.0000 | 0.0000 | no instrument (OOD regime, ppl ≈ 560) | 8.12 | 123.2 | 1.00 | 726.2 | 1.00 |
| `int4exp` | + int4 experts | 6.96110 | 0.6257 | 490.5756 | no instrument (OOD regime, ppl ≈ 560) | 6.55 | 152.7 | 1.24 | 795.9 | 1.10 |
| `calib` | int4 experts + C4-calibrated int4 attention REFUSED(b1,ppl) | — | — | — | — | — | — | — | — | — |
| `r1` | round-1 norm folds only | 6.38437 | 0.0489 | 28.2937 | no instrument (OOD regime, ppl ≈ 560) | 7.53 | 132.8 | 1.08 | — | — |
| `r12` | round-1 + round-2 folds only | 6.38437 | 0.0489 | 28.2937 | no instrument (OOD regime, ppl ≈ 560) | 7.50 | 133.3 | 1.08 | — | — |
| `epi` | router epilogue only | 6.31817 | -0.0173 | -9.6604 | no instrument (OOD regime, ppl ≈ 560) | 8.04 | 124.4 | 1.01 | — | — |
| `stack` | int4 experts + r1 + epilogue | 6.95135 | 0.6159 | 480.3413 | no instrument (OOD regime, ppl ≈ 560) | 5.90 | 169.5 | 1.38 | 833.9 | 1.15 |
| `all` | int4 + calib attn + r1 + r2 + epilogue REFUSED(b16,b1,ppl) | — | — | — | — | — | — | — | — | — |
| `calibbias` | int4 + calib attn with biases (#377) | 6.16328 | -0.1722 | -89.2341 | no instrument (OOD regime, ppl ≈ 560) | 8.08 | 123.8 | 1.00 | — | — |
| `r1epicalib` | int4 + calib(bias) + r1 + epi | 6.19571 | -0.1397 | -73.5779 | no instrument (OOD regime, ppl ≈ 560) | 7.42 | 134.8 | 1.09 | — | — |
| `r1epicalibhead` | r1epicalib + int4 head | 6.20544 | -0.1300 | -68.7807 | no instrument (OOD regime, ppl ≈ 560) | 7.28 | 137.4 | 1.12 | — | — |
| `mx_int4exp` | native MXFP4 store (#372) | 6.68565 | 0.3502 | 236.6134 | no instrument (OOD regime, ppl ≈ 560) | 8.70 | 114.9 | 0.93 | 295.7 | 0.41 |
| `mx_stack` | MXFP4 store + r1 + epi | 6.70909 | 0.3736 | 255.6067 | no instrument (OOD regime, ppl ≈ 560) | 8.05 | 124.2 | 1.01 | 298.5 | 0.41 |

`pass*` = one-sided calibrated rule on ONE text; the second, out-of-domain text has not been scored on these lanes.


## Reading it

- **Qwen3-30B-A3B (the reference).** The round-2 fold had never engaged on the campaign's own best stack
  (the calibrated int4 attention lane packs q/k/v/o separately and is exclusive with qkv fusion; the fold
  licensed only the fused module). #375 licenses the separate-projection shape: `all` → `all_rope`,
  6.41 → 5.62 ms, **156.0 → 177.9 tok/s (×1.14)**, K8 −0.0011 nats. The calibrated output head (#373)
  passes only INSIDE the calibrated stack (`allhead` −0.020 ppl vs NF4); alone (`head`) it is +0.062 ppl,
  over the budget — the store is licensed as part of the stack, not standalone.
- **Granite-3.1-3B-A800M — RETRACTION.** Its int4 experts cost +0.063 ppl (`int4exp`) and the "302 tok/s,
  ×1.59" stack +0.056, both over the 0.05 budget; the 0.33.0 notes quoted that row as parity because the
  lane table carried the family's 0.0033-nat noise floor and no budget verdict (`docs/STATUS.md`, #381).
  The licensed stack keeps NF4 experts: `nf4_r12epi_new` (NF4 + round-1 + round-2 + epilogue, with the
  rotary-only fold #379) = **259.1 tok/s B=1 (×1.37), 1689.6 B=16 (×1.18)**, +0.019 ppl. The
  `stackr2_old/new` pair measures the fold itself on the int4 stack (×1.156 / ×1.093, control flat).
- **Gemma-4-26B-A4B.** No K8 instrument at any resolution (SERVING-PARITY: the same tokens move ±0.1–0.27
  nats under arithmetically equivalent forwards). Quoted best stays `stack` (int4 + r1 + epilogue):
  **121.1 / 962.8 (×1.69 / ×1.68)**. Calibrated attention (`best`) is ×1.075 at B=1 and the calibrated
  262k-vocabulary head (`bestdensehead`) ×1.078 on top — 141.8 tok/s, with a K8 nobody can read; the
  dense MLP target (#378) is ×1.010, not a lever.
- **Mixtral-8x7B.** int4 experts pass (−0.037 ppl); `all` (+ calibrated attention) +0.045 ppl, one-sided
  pass on one text: **110.0 / 373.4 (×2.29 / ×2.00)**; `stack` without calibrated attention 102.8 / 372.0.
  The rotary-only fold applies to Mixtral's norm-less attention too and was not measured on this lane
  (no arena left on the box).
- **gpt-oss-20b.** No ppl instrument on raw text (ppl ≈ 560, the OOD-flattery regime; a noisier arm scores
  BETTER — `calibbias` −0.17 nats). Uniform int4 experts fail on a grid mismatch (+0.63 nats; e2m1 levels
  are not representable). The native MXFP4 store (#372) is exact against its own bytes on every row of
  every call and ~10%/row from the NF4 path (NF4's own error on e2m1 weights); its `+0.35 nats` on this
  window is the same regime, not a defect. Speed: NF4 123.2; round-1/2 folds 133.3; calibrated attention
  + head (`r1epicalibhead`) 137.4; the MXFP4 store through `gemv_mxfp4_b32` (grouped-nf4-gemm 0.28.0,
  measured on the bo3n phase, JSON `gptoss_mx2_b1_gemv.json`) **151.1 tok/s at B=1 (×1.22)** and 589.7 at
  B=16 (×0.81 — the decode GEMV re-streams weights per row; batched rows keep NF4).

## What is NOT in the table

The probe phases (bo3h/j/o/q: `*probe*`, `mxh_*`, `gptoss_ppl64_*`, the `ROUTEPROBE` lines in
`logs/bo3j.log`, `logs/bo3o.log`, `logs/bo3q.log`) scored 16- or 64-step windows for a route question and
are not comparable to the 2048-step rows; the kernel censuses (`census_*.txt`) and op censuses
(`opcensus*_*.txt`, phases bo3f/l/p) are the per-kernel and per-call-site step breakdowns behind the
optimisation pass. `pip_*.log` records each phase's installed cut.
