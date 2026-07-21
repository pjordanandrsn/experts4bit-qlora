# Storage modes: the support matrix

*(Moved out of the top-level README for length; linked from it.)* [← back to README](../README.md)

One knob selects the frozen experts' storage: `quant_type=` in code, `QUANT_TYPE=` in the
train/infer scripts — the same validation path, checked **before any checkpoint I/O**. Canonical
names are the six below; `bfloat16`/`float16` are accepted aliases (case/whitespace-insensitive);
anything else raises listing the valid set. There is no per-expert mode mixing: one module, one
scheme.

**What the words mean.** *Supported* means tested under the stated conditions — no more: an
exposed code path is not a warranty. *Experimental* means the path exists but may change or break
(the `ExpertsNbit` primitive as a whole carries upstream's experimental tag until
bitsandbytes#1965 settles). *Unsupported* means it fails loudly by design — never a silent no-op.

| Mode | Status | Intended use | Memory | Quality risk¹ | Training | Inference | Offload | Notes |
|---|---|---|---|---|---|---|---|---|
| `nf4` | supported + benchmarked | the QLoRA default | 4x smaller | ~0.17 (cap 0.25) | end-to-end (20-step test + convergence run) | fast-path + probed GEMV + prefetch | tested + benchmarked | the headline path |
| `fp4` | supported | nf4 alternative codebook | 4x | ~0.22 (0.30) | recompute-tested | dequantize + probed GEMV | same code path as nf4 | coarser than nf4 on ~Gaussian weights |
| `int8` | supported (tested contract) | higher-fidelity frozen base | 2x | ~0.017 (0.03) | LoRA step tested | dequantize (no GEMV) | identity-tested | blockwise dynamic map — **not** LLM.int8() |
| `fp8` | supported (tested contract) | int8 alternative | 2x | ~0.045 (0.08) | LoRA step tested | dequantize | same code path as int8 | bnb e4m3 **codebook**, not torch float8; coarser than int8 (test-pinned) |
| `bf16` | supported (tested contract) | reference baseline / per-layer opt-out | 1x (none) | ~0.003 (8e-3) | LoRA step tested | dequantize | identity-tested | passthrough; no absmax buffers |
| `fp16` | supported (tested contract) | as bf16 | 1x | ~0.0004 (1e-3) | LoRA step tested | dequantize | same code path as bf16 | passthrough |

¹ Forward relerr vs a float reference on synthetic ~Gaussian expert weights, measured on CPU and
A2000 kernels (bnb 0.49.2); the parenthesized cap is the calibrated test ceiling
(`tests/test_reference_parity.py`). Not an end-task quality claim.

**"Tested contract"** = build, forward parity vs a float reference (per-scheme ceilings), a
state_dict round-trip with validated metadata, a LoRA-over-frozen-base training step with the
recompute Function on the autograd tape, and offload math-identity. Offload is identity-tested
directly on `nf4`/`int8`/`bf16`; `fp4`/`fp8`/`fp16` ride the same code paths byte-for-byte. Only
`nf4` is performance-benchmarked end-to-end — the other five are correctness-tested, not measured
for speed or end-task quality.

### What ExpertsNbit is / is not

**Is:** frozen quantized *storage* for fused expert stacks (`[num_experts, out, in]`) — a
per-expert-loop forward, quantization blocks that never cross an expert boundary, and a
recompute-in-backward projection so training holds no dequantized-expert activations.

**Is not:** grouped-GEMM (per-expert loop only, intentionally), a Transformers-wide quantization
walker, double quantization, multi-GPU/FSDP, or a speed play — on a card that already fits the
model it is strictly a memory trade (see the energy caveat above).

### Experts4bit compatibility

`Experts4bit` is the 4-bit-restricted subclass (`nf4`/`fp4` only — it rejects the 8/16-bit names
*and their aliases*) and keeps its pre-0.2 API: same constructor, same `from_float`, same
state_dict tensor keys. The loader still instantiates `Experts4bit` for 4-bit runs, so existing
`isinstance(m, Experts4bit)` checks keep working.

### Known limitations & unsupported paths

- **Checkpoint metadata:** state_dicts now embed construction metadata (scheme, blocksize, dims)
  and loads validate it — loading an `fp4` checkpoint into an `nf4`-built module raises instead
  of silently decoding against the wrong codebook (the packed bytes are shape-identical).
  Pre-metadata checkpoints load unvalidated, under both `strict` modes. *New* checkpoints into
  ≤0.2.0 code: `strict=False` works (`_extra_state` lands in `unexpected_keys`); `strict=True`
  fails loudly on the unexpected key.
- **safetensors full-module saves:** the `_extra_state` entry is a dict, which safetensors
  refuses (loudly). Filter it — `{k: v for k, v in sd.items() if not k.endswith("_extra_state")}`
  — and the save loads as a legacy (unvalidated) checkpoint. Adapter-only saves never carry it.
- **Non-checkpointed offload *training* is unsupported** and fails loudly naming the invariant
  (the shipped trainer always enables gradient checkpointing).
- **`offload_model_experts` raises when it finds no `ExpertsLoRA` modules** (changed this
  version: it used to return `[]` silently). The streaming loader likewise refuses to return a
  model on which it quantized zero expert layers.
- **GEMV is 4-bit-only** and probe-gated per configuration; the 8/16-bit schemes always decode
  via the dequantize path.
- **Loader scope** is the four architecture families under [Scope](#scope); the `ExpertsNbit`
  primitive itself is model-agnostic.

### Reading the headline memory numbers

The **7.16 GB** for Qwen3-30B-A3B (and 8.47 GB for Gemma-4-26B-A4B) is **peak GPU allocation
during a QLoRA training step with `OFFLOAD_EXPERTS=1`** on the reference A2000: roughly one
layer's experts resident plus activations/adapters, while the other ~13–15 GB of packed experts
sit in pinned CPU RAM. It is a *capability* number — fits vs doesn't fit — not a throughput
claim: the same mechanism costs ~+11 % s/step at OLMoE scale and is PCIe-bound at 26–30B scale
(0.22–0.43 tok/s decode). Method and grids: [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/METHODOLOGY.md) §11–§12;
environment and commit pins: [`PROVENANCE.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/PROVENANCE.md).

### How to reproduce validation

One command, no model downloads, nonzero exit on any FAIL:

```bash
python scripts/validate_expertsnbit.py
```

```
experts4bit-qlora validate | v0.2.0 | commit <sha> | torch 2.6.0+cu124 | bnb 0.49.2 | cuda yes | NVIDIA RTX A2000 12GB
[PASS] nf4   build            0.15s
[PASS] nf4   forward_parity   relerr=0.1691 (tol 0.25)
...
[PASS] -     metadata_guard   raised ValueError: checkpoint/module config mismatch...
SUMMARY pass=37 fail=0 skip=0 -> exit 0
```

It runs the tested contract per scheme (build, forward parity, state round-trip, LoRA step,
synthetic decode sanity, offload identity) plus the checkpoint-metadata guard; SKIP lines always
say why (e.g. a host whose bitsandbytes can't quantize a scheme). The full suite is
`pip install -e ".[test]" && pytest tests/ -q`; big-model numbers reproduce via the manual
[Benchmarks](#benchmarks) scripts, not this report.

### Validation grids

experts4bit-qlora does not name a winning quantization mode — it produces a measured decision
surface (fit / fidelity / speed / portability / residency budget) with per-cell provenance. An
OLMoE-1B-7B validation grid (bundle `olmoe-qlora-grid-20260705-1351`, 3 seeds) shows a storage/
offload asymmetry — resident training exposes the memory cost of wider storage while offload
collapses the 4-bit-vs-int8 gap to ~2.4–2.7 GB — and finds int8-offload a low-VRAM/high-fidelity
training candidate *for OLMoE* (best eval 3/3 seeds). Repeating also corrected a single-run
artifact: fp4 decode is **not** faster than nf4 once sampled. Expert-streaming profiling found the
offload wall **diffuse** (no hot-static pinning justified). Qwen3-30B-A3B is a separate
scale-transfer probe: nf4 resident fits a 24 GB card, int8 resident is impractical, offload is
blocked by the pod's RAM cap. Start with
[`docs/results_summary.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/results_summary.md) and
[`docs/support_matrix.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/support_matrix.md); details in
[`OLMOE_EXPERTSNBIT_GRID`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/OLMOE_EXPERTSNBIT_GRID.md),
[`OLMOE_REPEAT_VALIDATION_PLAN`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/OLMOE_REPEAT_VALIDATION_PLAN.md),
[`MODE_DECOUPLED_ADAPTERS`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/MODE_DECOUPLED_ADAPTERS.md),
[`EXPERT_STREAMING_PROFILE`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/EXPERT_STREAMING_PROFILE.md),
[`QWEN3_30B_EXPERTSNBIT_GRID`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/QWEN3_30B_EXPERTSNBIT_GRID.md); apparatus in
[`RUNPOD_DISTRIBUTED_VALIDATION`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/RUNPOD_DISTRIBUTED_VALIDATION.md) and
[`provenance_contract`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/provenance_contract.md). An external review pass —
[`MEASUREMENT_AUDIT`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/MEASUREMENT_AUDIT.md) — recomputed every number, computed the ∅/G
quality yardstick that was latent in the bundle, and downgraded the int8-offload "best eval"
claims to confounded (a precision×placement interaction the bf16 control exposes); read it
alongside the results.
