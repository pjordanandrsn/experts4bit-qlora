# Hybrid CPU/GPU execution tier — architecture notes (e4b side)

Pre-work map for the hybrid-tier program (Stage 1 Phases 0–5, gated G0–G5):
where each phase hooks into this repo. Kernel-side notes live in
`grouped-nf4-gemm/docs/cold-engine/ARCHITECTURE-NOTES.md`. Placement rule:
CPU kernels + calibration → gnf4; router, placement solver, executor,
prefetcher, manifest, training → here.

## What already exists that the tier builds on

- `engines/cold_engine.py` is the proof-of-regime: `_ColdEngine` overrides
  exactly one hook (`_cold_contrib`) of `_HotResidency` and computes the cold
  tail on host — activation-sized traffic only, values rounded through
  `compute_dtype` so host and fused-kernel results agree. Its docstring
  reserves the native-kernel slot this program fills. Today it is a
  single-threaded python loop over experts with per-expert torch dequant:
  correctness path, ~100–200× off on small-expert models (config matrix,
  2026-08-11), yet already 1.85× over streaming on Mixtral-wide experts.
  Phase 2/3 replace the loop body with gnf4's AVX-512 grouped kernels and the
  scheduling around it.
- `engines/nvme_experts.py`'s `_TieredStack` proves a tier can be added
  without touching `forward()` — but its contract (`.index_select(0, ids)`
  then `.to(dev)`) is *supply-bytes*. A tier that COMPUTES does not fit it;
  the `_cold_contrib` override is the correct seam for the CPU-compute side.
- `_ColdEngine` and `_NvmeResidency` are sibling subclasses of
  `_HotResidency` (one overrides `_cold_contrib`, the other
  `_build_hot`/`_build_cold`) and are NOT composable today. The hybrid
  executor needs "CPU compute over a DRAM tier + NVMe cold path" in one
  object — the Phase 3 PR either refactors these hooks into explicit tier
  objects or introduces the hybrid engine as a third sibling that owns both.
  Stage-2 invariant 8 (KV must reuse the tier abstraction) argues for the
  refactor: tier objects should hold generic blocks, not expert rows.
- Residency economics law (issue #108, measured): residency/tiering pays iff
  Δgpu_saved > Δhost_added ON THAT BOX; no ratio predicts it. This is why the
  Phase 3 solver consumes a measured calibration blob, never constants.

## Phase 1 — router on CPU

**The hook is upstream of this package.** Every residency engine receives
`(hidden_states, top_k_index, top_k_weights)` — router logits and top-k are
computed by the parent MoE block in transformers code. So Phase 1 patches at
the MoE-block level, per architecture, the way gnf4's `arena_moe_patch` does
for K3: a new `engines/cpu_router.py` wraps the block's forward to (a) issue
an async D2H of the post-attention hidden state on a dedicated stream into
pinned, write-combined, double-buffered per-layer staging (persistent
allocation), (b) compute router logits + top-k on host against an FP32
router-weight copy (~1M params — the documented, deliberate exemption to the
one-artifact invariant), (c) dispatch CPU-resident experts immediately and
push the index vector H2D for GPU-resident ones, while the GPU overlaps
shared-expert/residual work. One round trip per layer.

- Deterministic top-k: fixed reduction order, ties break to the lower expert
  index. Debug builds cross-check CPU routing against the GPU reference every
  N steps (stop condition if they diverge beyond tie-break cases).
- The pipelined engine's zero-sync contract
  (`tests/test_pipelined.py` enforces `torch.cuda.set_sync_debug_mode("error")`)
  is untouched: `cpu_router` is part of the hybrid decode path only, never
  patched over `_pipelined` modules (mutual-exclusion markers, below).
- G1 (nsys-verified): zero blocking D2H syncs in decode; round-trip p50
  ≤35 µs / p99 ≤100 µs at batch 1. This kills the ~5-syncs-per-layer-step
  stall class found in the #105/#108 campaign, and pays even if no later
  phase ships.

## Phase 3 — placement solver + disjoint-bus executor

- `engines/placement.py` — static solver. Inputs: gnf4 calibration blob +
  routing-frequency profile (`engines/expert_profile.py` JSONL is the
  existing profiling pass; `hot_sets_from_profile` / `coverage_from_profile`
  are the precedent ranking functions; ship a default trace). Objective
  order: (1) minimize NVMe-resident routing mass, (2) approach GPU:CPU
  routing-mass ratio of `B_vram_effective : B_dram_grouped` subject to VRAM
  capacity after attention/KV reservation. Stage-2 note: the balance
  objective later moves to unique-expert counts weighted by bandwidth
  (batched dispatch); keep the objective function swappable.
- `engines/hybrid.py` — the executor. Three disjoint flows enforced in code:
  VRAM bus = attention/KV/shared/hot experts; DRAM bus = warm experts
  computed in place via gnf4 `cpu_grouped`; PCIe = activations + cold NVMe
  experts streamed NVMe→GPU (GDS when available, pinned bounce buffers
  otherwise). Cold misses never touch the DRAM bus. Joins the engine
  conventions: mutual-exclusion marker checks in BOTH directions against
  `_e4b_fast_ref`/`_e4b_pipe_ref`/`_e4b_hot_ref`/`_e4b_cold_ref`/
  `_e4b_mxfp4_ref`, a `hybrid_available()` probe, `__all__` +
  `scripts/wheel_smoke.py` + README decision-table entries, and a correct
  `disable_*` that removes every stamped attribute (the
  `enable_nvme_residency` teardown gap is the counterexample to copy-fix).
- Manifest: this repo has **no runtime manifest module today** — `verify.py`
  is an is-it-quantized checker (no SHA-256, no manifest, no CLI); per-expert
  SHA-256 lives in bench drivers (`bench/flagship-matrix/drivers/n17_cell.py`
  `expert_hashes`, with the empty-hash guard + flip control); job provenance
  lives in `scripts/validate_job_provenance.py` + `docs/provenance_contract.md`;
  arena manifests belong to gnf4's `nvme_arena`. Phase 3 adds, additively: a
  placement manifest (expert→device map, calibration blob, routing-profile
  hash, backend reduction-order IDs) and grows `verify.py` a `--manifest`
  mode (bytes + placement + calibration ⇒ reproducible run) reusing the
  n17-style per-expert hashing. Existing outputs keep their schemas.

## Phase 4 — speculative prefetch

`_PipelinedResidency.hint()` is the intended hook (mechanism landed,
explicitly untuned, nothing calls it) — but the #105 lesson stands:
same-stream prefetch is not prefetch, and the existing speculative
staging (`engines/offload.py` router-ahead path) syncs on `.tolist()`.
Phase 4's predictor routes layer L+1 from layer L's hidden on the CPU
(cheap once Phase 1's router copy exists), issues NVMe→GPU reads on the cold
stream, LRU over a small VRAM prefetch pool, mispredicts fall through to the
demand path. Gate G4 requires the feature be free when NVMe mass = 0.

## Phase 5 — hybrid backward (training)

All residency engines are inference-only today (every one gates on grad and
routes to the saved reference forward), so the backward is net-new. Frozen
base ⇒ no expert weight grads; `grad_in = grad_out @ Wᵀ` is a second CPU
pass with transposed access, dequant-transpose into L2-resident tiles inside
the kernel — never a stored transposed copy (invariant 2). Precedents:
`_FrozenLinearRecomputeBackward` (re-dequantize in backward rather than save)
and gnf4's `dgrad_4bit_grouped` (the GPU dgrad twin). Gradient checkpointing
default-on for hybrid training (`engines/offload.py` states the invariant:
non-checkpointed offload training fails loud); LoRA math stays GPU-side.
Parity gate: hybrid QLoRA grads vs full-GPU reference within documented FP32
tolerance + loss-curve overlay.

## Bench + reporting conventions to reuse

House tok/s = `infer.timed_decode` (manual KV-cache greedy loop,
sync-bracketed) with the grep-able `BENCH ...` line. A/B discipline =
`bench/bench_hotsets_ab.py` (interleaved arms, Mann-Whitney U inline).
The honest external baseline is banked: llama.cpp `--n-cpu-moe 24` at
11.80 tok/s vs ours-hybrid 1.12–1.33 on the same box
(`bench/RESULTS-gptoss-hybrid-ab.md`) — the gap this program exists to
close, and the number G3 is measured against alongside the pure-streaming
baseline. Every `enable_*` return count is captured (a 0-patch config is
`not-engaged`, never a datapoint). Docs and results must pass the
private-marker guard; run its grep locally before committing.

## Stop conditions (verbatim from the directive)

G0 <50% · any invariant requires violation · determinism unachievable in a
phase · a dependency forces a weight-format change · CPU router disagrees
with GPU reference beyond tie-break cases. Halt and report; do not improvise.
