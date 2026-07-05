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

## Non-goals

No grouped GEMM. No double quantization. No Transformers-wide walker integration. No
multi-GPU/FSDP. No new architecture campaign. No public-API breaks (aliases only). No
version bump or release in this pass. `PROVENANCE.md` is append-only (OTS-stamped) and is
not touched; it states the v0.2.0 counts, and the README points at the validation script
instead of promising counts.
