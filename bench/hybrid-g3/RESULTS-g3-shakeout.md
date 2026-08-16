# RESULTS — G3 shakeout: hybrid three-tier vs pure streaming (30B-class)

Pipeline shakeout for gate G3 on Qwen3-30B-A3B (48 MoE layers × 128
experts, NF4 quantize-bake arena, 16.3 GB), rented Zen 5 EPYC + RTX 5090
box, destroyed after. Both arms: same box, same artifact, same greedy
decode of 64 new tokens (transformers generate, sync-bracketed).

| arm | config | tok/s |
|---|---|---|
| A — pure GPU streaming | `enable_nvme_residency`, empty hot sets (the pre-hybrid engine) | 1.539 |
| B — hybrid three-tier | solver placement, 40-worker pool | **12.071** |

**ratio = 7.84× (G3 bar for the full gate: ≥4×).**

- Placement (solver, measured blob + fresh routing profile): 76.9% of
  routing mass VRAM / 23.1% DRAM / 0% NVMe under a 23.7 GB VRAM budget.
  The completion-time greedy hit its bandwidth-ratio target to five
  digits: achieved 3.32778 vs target 3.32773.
- Manifest sha `5241708758…`; receipts in this directory (placement,
  profile, per-box calibration blob, tier stats in `g3_shakeout.json`).

## The bus audit (nsys, 32 steady-state decode tokens, capture-range)

| flow | total over 32 tokens |
|---|---|
| PCIe H2D | **33.4 MB** (4,547 copies, max single copy 0.4 MB) |
| PCIe D2H | 16.9 MB |
| memset/D2D | negligible |

No expert-row-sized transfers appear in the captured window — the link
carried activation-class traffic only, versus the ~1 GB/token the
streaming baseline moves by design. That is the disjoint-bus law,
observed. Trace committed (`g3_nsys.nsys-rep` + stats extracts).

## Honest caveats

- **This is the shakeout, not the gate.** G3 is defined on a
  Qwen3-235B-class artifact; that leg (bf16 download + bake ≈ 600 GB of
  disk, hours) is scheduled as its own run. The 30B result validates the
  full pipeline (bake → profile → solve → both arms → nsys) end to end.
- The warm (DRAM) branch still host-syncs (~17.8K `cudaStreamSynchronize`
  in the window — the cold_engine-inherited `.cpu()`/`.to("cpu")` pattern).
  The ratio is 7.84× anyway; composing Phase 1's async CPU router into the
  hybrid decode is the named next lever, not a prerequisite.
- Arm A is the directive's stated baseline (pure streaming, same
  artifact). The strongest pre-hybrid config (informed hot sets on the
  pipelined engine) sits between the two arms; the flagship-vs-flagship
  comparison belongs to the 235B gate run.
