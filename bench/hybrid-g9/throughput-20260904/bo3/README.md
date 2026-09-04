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
nats: uncalibrated |Δppl| ≤ 0.05; a calibrated pack ≤ +0.05 one-sided AND the same sign on ≥ 2 texts, one
outside the calibration domain. These lanes scored ONE text, so no calibrated arm is licensed here —
the column says `one text (needs 2)`; the second text is on the next lane.
Where a family has no instrument, the column says so instead of a verdict.


`one text (needs 2)` = a calibrated pack within +0.05 ppl on the one text these lanes scored; the registered rule (k8_gate) licenses it only with the same sign on a second text, so it is NOT licensed here.

## Reading it

- **Qwen3-30B-A3B (the reference).** The round-2 fold had never engaged on the campaign's own best stack
  (the calibrated int4 attention lane packs q/k/v/o separately and is exclusive with qkv fusion; the fold
  licensed only the fused module). #375 licenses the separate-projection shape: `all` → `all_rope`,
  6.41 → 5.62 ms, **156.0 → 177.9 tok/s (×1.14)**, K8 −0.0011 nats vs `all`. The stack carries a
  calibrated pack (the int4 attention), so the registered gate needs a second text before it is licensed;
  the one text scored says −0.081 ppl. The calibrated output head (#373) is within budget only INSIDE the
  stack (`allhead` −0.020 ppl vs NF4); alone (`head`) it is +0.062 ppl, over the budget.
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
- **Mixtral-8x7B.** int4 experts pass (−0.037 ppl, uncalibrated rule); `all` (+ calibrated attention)
  +0.045 ppl on one text — within budget, not yet licensed (second text pending): **110.0 / 373.4 (×2.29 / ×2.00)**; `stack` without calibrated attention 102.8 / 372.0.
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
