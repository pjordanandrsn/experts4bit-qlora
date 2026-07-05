# ExpertsNbit finish plan

Scope note for the v0.2.x completion pass. Goal: make the fused-expert storage contract
explicit, tested, documented, and reproducible — not to add capability. If a model should fit,
the stack must not silently make it impossible; if a path is supported it is tested under stated
conditions; anything else fails loudly or is labeled.

## Current state (baseline `2503de4`)

`ExpertsNbit` already stores fused expert stacks (`gate_up_proj [E, 2*I|I, H]`,
`down_proj [E, H, I]`) in six schemes — `nf4`/`fp4` (4-bit packed), `int8`/`fp8` (8-bit
blockwise codebook), `bf16`/`fp16` (passthrough) — quantized per expert (blocks never cross an
expert boundary; dims must divide `blocksize`). `Experts4bit` is the 4-bit-only subclass kept
for the original API. Forward and backward share one path: dequantize-per-expert +
`F.linear`, re-dequantizing in backward (`_FrozenLinearRecomputeBackward`) so no
full-precision weight is saved. Offload, decode fast-path, and probed GEMV ride on top.

What was missing, verified against the tree:

- `quant_type` is exact-string matched. `"NF4"`, `"bfloat16"` → `ValueError`. The loader
  validates nothing before streaming shards, and its progress log says "NF4" for every scheme.
- `state_dict` carries no construction metadata. An `fp4` checkpoint loads byte-compatibly
  into an `nf4`-constructed module and silently mis-decodes — same shapes, wrong codebook.
- Forward parity vs a float reference is tested for `nf4`/`fp4` only; `int8`/`fp8`/`bf16`/`fp16`
  had build/isfinite smoke. The fidelity-ordering test covered three of six schemes.
- `bf16`/`fp16` offload crashes (`_build_homes` calls `.detach()` on absmax buffers that are
  `None` for passthrough schemes). Offload math-identity was tested on `nf4` only.
- `offload_model_experts()` silently returns `[]` on a model with zero `ExpertsLoRA` modules;
  the loader silently returns an unquantized model when a supported `model_type` has no expert
  tensors. Both are the exact silent-skip failure class this package exists to prevent.
- No support-matrix / warranty language in the README; no single validation entrypoint.
- Known suite defect from the v0.2.0 cross-arch runs: the Level-2 parity tests hard-fail on
  sm_120 (Blackwell) because the *transformers reference* routes MoE through
  `torch._grouped_mm` (hard `cc == 9.0` gate) — an oracle limitation, not a library defect.

## Mode contract after this pass

| mode | status |
|---|---|
| `nf4` | supported + benchmarked (the headline path) |
| `fp4` | supported — parity-tested, not perf-benchmarked |
| `int8`, `fp8`, `bf16`, `fp16` | supported for the tested contract (below), not perf-benchmarked |

Tested contract per mode: build, forward parity vs float reference, `state_dict` round-trip
with validated metadata, LoRA-over-frozen-base training step, offload math-identity.
Canonical names are the six above; accepted aliases are exactly `bfloat16`→`bf16` and
`float16`→`fp16` (case/whitespace-insensitive). Anything else raises listing the six. The
`QUANT_TYPE` env knob and the `quant_type=` kwarg are the same path, validated before any
shard is read. `fp8` is bitsandbytes' e4m3 *codebook* (256-entry blockwise), not torch-native
float8 — coarser than `int8`'s dynamic map.

## state_dict metadata design

`get_extra_state()` serializes `{schema: 1, quant_type, blocksize, has_gate, num_experts,
hidden_dim, intermediate_dim}` through torch's standard `_extra_state` key;
`set_extra_state()` validates each field against the constructed module and raises naming
checkpoint-vs-module values on mismatch (the load-bearing case: `fp4` vs `nf4` are
byte-compatible). Backward compatibility, verified against torch 2.2/2.7 source:

- old checkpoint → new code: a `_load_from_state_dict` override injects a tolerated `None`
  (`state_dict.setdefault(prefix + "_extra_state", None)`), so legacy checkpoints load under
  **both** strict modes, bit-identically to before.
- new checkpoint → old (≤0.2.0) code: `strict=False` loads correctly (`_extra_state` reported
  in `unexpected_keys`); `strict=True` fails loudly with a self-describing unexpected-key error.
- adapter-only saves/loads (`"lora" in k` filter) never carry the key — unaffected.
- known limitation: `safetensors` refuses dict-valued entries, so a *full-module* safetensors
  save needs `{k: v for k, v in sd.items() if not k.endswith("_extra_state")}` — which degrades
  to legacy (unvalidated) behavior. If that flow ever matters, the drop-in alternative is
  encoding the metadata as JSON bytes in a uint8 tensor; not done now for debuggability.

## Parity tolerances

Per-mode ceilings on forward relerr vs the float reference and on dequantized-weight relerr
(same dict serves both), calibrated to ≥3× the observed error on CPU and on the A2000 before
committing; the self-calibrating structural-bug margins (0.33, `fp4` 0.5) stay:

| mode | expected relerr | ceiling | anchor |
|---|---|---|---|
| `nf4` | ~0.09 fwd | 0.15 | measured (existing test comment) |
| `fp4` | ~0.20 | 0.30 | measured |
| `int8` | ~1e-2 | 0.03 | dynamic-map estimate → calibrate |
| `fp8` | ~3-4e-2 | 0.08 | e4m3 4-significand-bit RMS → calibrate |
| `bf16` | ~2e-3 | 8e-3 | 2⁻⁸ rounding |
| `fp16` | ~3e-4 | 1e-3 | 2⁻¹¹ rounding |

Fidelity chain (single seed, mean-abs): `fp16 < bf16 < int8 < fp8 < nf4 < fp4`. The
`int8 < fp8` link is pinned by measurement, not construction — if a bitsandbytes codebook
change ever flips it, demote that link to documentation rather than forcing it.

**Calibration outcome (2026-07-04, CPU + A2000, bnb 0.49.2):** the nf4 anchor above was wrong —
the "~0.09" in the old test comment did not match what the test's shapes/seed actually produce
(measured forward relerr 0.154 CPU / 0.169 CUDA; fp4 0.220, int8 0.0175, fp8 0.0449, bf16
0.0031, fp16 0.0004). Shipped ceilings moved accordingly and split into forward vs weight dicts:
`TOL_FWD = {nf4 0.25, fp4 0.30, int8 0.03, fp8 0.08, bf16 8e-3, fp16 1e-3}`,
`TOL_WEIGHT = {nf4 0.15, fp4 0.20, int8 0.02, fp8 0.04, bf16 3e-3, fp16 5e-4}` — ~1.4–2.6x
headroom over the worst observed kernel, not the blanket ≥3x this plan first proposed (for the
4-bit modes that would exceed the structural-bug-control level and make the ceiling meaningless).
`tests/test_reference_parity.py` is the source of truth. The fidelity chain held as predicted,
int8 < fp8 included (7.4e-4 vs 1.7e-3 mean-abs).

## Changes

- `_vendor/experts.py` — `normalize_quant_type()` + the two aliases; extra-state trio
  (`get_extra_state` / `set_extra_state` / `_load_from_state_dict` shim).
- `__init__.py` — export `normalize_quant_type`; adopt upstream classes only if
  `issubclass(Experts4bit, ExpertsNbit)` and they carry the extra-state surface (else vendored).
- `loader.py` — validate `quant_type` before any I/O; raise when zero expert stacks were
  quantized; log the actual scheme instead of "NF4".
- `offload.py` — `offload_model_experts()` raises on zero `ExpertsLoRA` found; passthrough
  schemes offloadable (skip `None` absmax homes).
- Tests — six-mode parity with the ceilings above; square-dims + many-experts cases; six-mode
  fidelity chain; strict-mode round-trip pin; extra-state accept/reject; loader pre-I/O
  validation + no-experts rejection; offload identity over `nf4`/`int8`/`bf16`; optimizer-step
  sanity (assert `lora_B` moved — `lora_A`'s grad is exactly zero at `B=0` init); recompute
  Function on the tape for all six modes; sm_120 oracle-skip guard on the Level-2 tests.
- `README.md` — storage-mode support matrix + warranty legend + limitations (commit 6).
- `scripts/validate_expertsnbit.py` — one-command report: versions/GPU header, per
  mode×check `[PASS|FAIL|SKIP]` lines with reasons, peak CUDA memory, nonzero exit on FAIL.
  No model downloads; big-model validation stays manual (`bench/`, `infer.py`).

## Verification record (2026-07-04, this pass)

Full suite (`pytest tests/ -q`) + `scripts/validate_expertsnbit.py` (37 checks), every host on
stock bitsandbytes 0.49.2:

| host | arch | torch | suite | validate |
|---|---|---|---|---|
| RTX A2000 12GB (reference) | sm_86 | 2.6.0+cu124 | 115 passed / 1 skipped | 37/0/0 |
| same host, CPU | — | 2.6.0 | 109 passed / 7 skipped | 37/0/0 |
| RTX 4090 (RunPod) | sm_89 | 2.8.0+cu128 | 115 passed / 1 skipped | 37/0/0 |
| RTX 5090 (RunPod) | sm_120 | 2.8.0+cu128 | 111 passed / 5 skipped, **0 failed** | 37/0/0 |
| H100 PCIe (RunPod) | sm_90 | 2.8.0+cu128 | 115 passed / 1 skipped | 37/0/0 |

Baseline at `2503de4` reproduced first (GPU 77/1, CPU 71/7 — the PROVENANCE v0.2.0 counts). The
sm_120 row is the oracle-skip guard's positive verification: at v0.2.0 that machine hard-failed 4
tests inside the *transformers reference's* `torch._grouped_mm`; those four are now skips whose
reason names the oracle. The sm_90 row is the guard's negative control — the one arch where the
oracle runs `_grouped_mm` natively; its only skip is the CPU-only branch, i.e. the Level-2 tests
PASSED there rather than skipping. Forward relerrs were bit-identical across every CUDA arch
tested (sm_86/89/90/120).

## Non-goals

No grouped GEMM. No double quantization. No Transformers-wide walker integration. No
multi-GPU/FSDP. No new architecture campaign. No public-API breaks (aliases only). No
version bump or release in this pass. `PROVENANCE.md` is append-only (OTS-stamped) and is
not touched; it states the v0.2.0 counts, and the README points at the validation script
instead of promising counts.
