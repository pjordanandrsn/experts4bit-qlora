# #171 — cold experts routed to the GPU: what the arms were measuring

Issue #171 reports that a model with NVMe-placed experts routed to the GPU
generates a different token sequence than the "resident" control, while the
same manifest at `cold_dest="cpu"` matches exactly, and names
`_NvmeResidency._cold_contrib` / `_TieredStack` as the likely defect.

The cold path is not what differs between those arms.

## What the arms actually change

`placement.force_cold_mass` moves experts out of the **DRAM** tier by default
(`source="dram"`, [placement.py](../../../experts4bit_qlora/engines/placement.py)),
and a DRAM expert **executes on the CPU** — `cpu_grouped`'s locked fp32 tree.
So the two cold arms are not two versions of one change:

* `cold_dest="cpu"` keeps every moved expert on the engine it was already
  running on. Same kernel, same bytes, same summation tree.
* `cold_dest="gpu"` moves it to the fused Triton kernel, which lands each
  grouped GEMM's output in the compute dtype (`out = torch.empty(...,
  dtype=torch.bfloat16)`) and runs the SwiGLU epilogue there.

That is the cross-placement rounding law `engines/hybrid.py` states in its own
first docstring, reached through a destination switch instead of a tier move.

## Measurement

`destination_gap.py` — four placements of the SAME experts through the SAME
engine, one MoE layer at OLMoE-1B-7B geometry (H=2048, I=1024, 8 of 16 experts
moved), identical inputs and routing. Every routed row lands on an expert under
test, so the arms differ in nothing but where those experts execute.

Box: RTX A2000 12GB (sm_86), torch 2.8.0+cu128, triton 3.4.0, gnf4 native CPU
kernels AVX2 + OpenMP.

| pair | T=1 | T=8 | T=64 | bitwise |
|---|---|---|---|---|
| `control_dram` vs `cold_cpu` | 0.000e+00 | 0.000e+00 | 0.000e+00 | **yes** |
| `control_vram` vs `cold_gpu` | 0.000e+00 | 0.000e+00 | 0.000e+00 | **yes** |
| `control_dram` vs `cold_gpu` | 4.590e-03 | 4.622e-03 | 4.627e-03 | no |
| **`control_dram` vs `control_vram`** | **4.590e-03** | **4.622e-03** | **4.627e-03** | no |

(relative RMS on the layer output.)

The last row is the finding. DRAM against VRAM, with **no cold path anywhere in
it**, reproduces the cold arm's divergence to the digit. T=1 and T=8 take the
decode GEMV, T=64 takes the M-tile prefill path, and the number barely moves —
so it is the epilogue and output rounding, not a launch-config artifact.

Both destinations are **exact** against their matched control, at every shape.

## Consequences

1. An arm comparison has to hold the execution destination fixed: `"gpu"`
   against the same experts in `vram`, `"cpu"` against them in `dram`. Both of
   those are bitwise. Comparing across destinations measures the CPU/GPU
   rounding path and reports it as a cold-path defect, which is #171.
2. A gate-1 sweep that wants the cold-GPU arm's equivalence clause to be
   meaningful should either dial cold mass with `force_cold_mass(...,
   source="vram")` or carry a per-destination reference. Under `source="dram"`
   the clause fails for a reason that has nothing to do with the cold path, and
   under `source="vram"` the arms swap — cold-GPU becomes the one that matches.
3. `cold_dest` belongs in run identity next to the manifest, the way
   `prefill_gpu_only` already documents for the DRAM tier.

## What is now pinned

* `test_cold_on_gpu_is_bit_identical_to_the_same_expert_placed_in_vram` — the
  mirror of the CPU destination's existing bitwise test, and the arm that was
  missing when #171 was filed. A mis-indexed cold gather or a misapplied router
  weight lands here as an O(1) difference; rounding cannot, because a matched
  destination cannot round differently from itself.
* `test_the_two_destinations_are_not_interchangeable_and_the_gap_is_bf16_scale`
  — the law as a measurement, floored (a zero means a destination stopped being
  the engine it claims to be) and ceilinged at 2**-5 relative RMS (past bf16
  mantissa scale is a defect, not a rounding path).
* `test_cold_gather_lands_the_right_expert_on_a_hybrid_partition` — the cheap
  half, no CUDA and no kernel: the cold branch's id algebra on a three-tier
  manifest, where NVMe rows sit at cold-local ids with DRAM-shaped holes
  between them. Mutation-checked — replacing the `cold_ids` remap with a
  positional index fails it on the first step.

Suite on the box: 962 passed / 45 skipped against 959 / 45 on `origin/main`,
i.e. exactly the three new tests, no new skips. The one failure
(`test_version_matches_distribution_metadata`) is present on both and is an
artifact of running from `PYTHONPATH` rather than an installed distribution.
