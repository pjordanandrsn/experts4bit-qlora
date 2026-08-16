# RESULTS — G3 formal: hybrid three-tier vs pure streaming (Qwen3-235B)

The gate run. Qwen3-235B-A22B (94 MoE layers × 128 experts, top-8), NF4
quantize-bake arena of 127 GB (`row_stride` 10,616,832), rented Zen 5 EPYC
9655 + RTX 5090 box, destroyed after. Both arms: same box, same artifact,
same greedy decode of 64 new tokens (transformers generate, sync-bracketed),
placement solved from this box's calibration blob (`calib-235b.json`, tag
`g3formal`) plus a routing profile taken in the same run (arm A's decode).

| arm | config | tok/s |
|---|---|---|
| A — pure GPU streaming | `enable_nvme_residency`, empty hot sets (the pre-hybrid engine) | 0.292 |
| B — hybrid three-tier | solver placement, 40-worker pool | **6.211** |

**ratio = 21.28× — gate G3 (≥4× on a 235B-class artifact, same box, same
artifact) PASSES.**

- Placement (solver, measured blob + fresh routing profile): 44.8% of
  routing mass VRAM / 55.2% DRAM / 0% NVMe under a 10.85 GB VRAM budget
  (`G3_DRAM_GB=100`; the full 127 GB arena minus the VRAM residents fits
  in host DRAM, so the solver correctly leaves NVMe empty and cold misses
  are pure tail: 12 tier misses in the timed decode).
- Manifest sha `a57642fbc0ce72da…` (`placement_dram100.json`); report
  `g3_formal_dram100.json`; run log `run_g3.log`.
- The bandwidth-ratio target was NOT hit (achieved GPU/CPU work ratio
  0.812 vs target 3.187): the VRAM budget on a 32 GB card next to 94
  layers of non-expert weights caps the GPU bus's share of mass well
  below what the bandwidth ratio alone would assign. The solver's
  capacity constraint dominates its balance objective here — expected at
  this VRAM:model-size ratio, and the 21.28× stands on the measured arms.

## The bus audit (nsys 2026.1.3, whole-run trace, decode window)

nsys 2024.4.2 (the CUDA-repo default) records NO GPU-side rows on this
sm_120 card — its CUPTI predates Blackwell and the stats reports come back
"does not contain GPU memory data" while the workload runs fine. The audit
below is from nsight-systems **2026.1.3** (same signed CUDA apt repo), and
the API-gated capture (`-c cudaProfilerApi`) produced no report on this
build either, so the trace is whole-run and the decode window is isolated
after the fact: **everything after the last >5 MB H2D copy + 200 ms** (the
load/VRAM-fill phase is the only source of multi-MB copies at dram100).
Window accounting (`g3_nsys_window.txt`, 5.38 s ≈ the 32-token profiled
decode):

| flow | total in window |
|---|---|
| PCIe H2D | **194.7 MB** (8,409 copies, max single copy **131.1 KB**) |
| PCIe D2H | 97.6 MB (25,322 copies, max 65.5 KB) |
| D2D | 0.26 MB |

One expert row is 10.6 MB; the largest copy in the window is 131 KB — two
orders of magnitude below a single expert row, five below the ~255 GB of
expert mass a streaming path moves for 32 tokens. The link carried
activation-class traffic only (DRAM-bus contribution landings + router
staging + logits). That is the disjoint-bus law at gate scale. Trace
committed (`g3_nsys_235b.nsys-rep`).

## Honest caveats

- **Run-to-run nondeterminism observed at this scale** (found by the G4
  A/B/A sandwich, same code path as this run): two identical
  prefetch-off arms produced different output bits mid-decode. At
  batch=1 all top-8 expert contributions `index_add_` into one token
  row, so CUDA atomic ordering collides every token. This violates the
  determinism invariant as stated ("same seed + same placement ⇒
  bit-identical logits per backend") and is filed as an open defect with
  a known fix (pre-combine per-token contributions before scatter); it
  does not affect the ratio (both arms measured identically) and was not
  patched mid-gate.
- The routing profile is a 64-token greedy decode of one prompt, and the
  timed decode replays the same prompt — placement is therefore
  profile-matched, the directive's stated methodology (profile → solve →
  run). Cross-prompt generalization of a placement is Stage-2 material.
- The warm (DRAM) branch still host-syncs per layer (the
  cold-engine-inherited pattern); composing Phase 1's async CPU router
  into the hybrid decode remains the named next lever, not a
  prerequisite. 21.28× is with the syncs in.
- Arm A is the directive's stated baseline (pure streaming, same
  artifact). It leaves the GPU idle waiting on NVMe for ~47 GB/token of
  expert mass; nothing about 0.292 tok/s is surprising, and the ratio's
  size is mostly the baseline's smallness. The absolute 6.2 tok/s hybrid
  number is the load-bearing one for Stage 2.
