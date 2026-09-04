# AGENTS.md — working in experts4bit-qlora

Operational notes for coding agents and contributors. Not a second README:
[`README.md`](README.md) argues the case, [`docs/STATUS.md`](docs/STATUS.md)
states the position, [`docs/claims.json`](docs/claims.json) holds the numbers.

## Purpose

`experts4bit-qlora` (import `experts4bit_qlora`) trains and serves fused
Mixture-of-Experts models in 4-bit on hardware that cannot hold them in
bf16. It owns model loading, quantisation orchestration (`Experts4bit` /
`ExpertsNbit` over the fused 3-D expert stacks bitsandbytes' walker skips),
per-expert LoRA (`ExpertsLoRA`), training, residency (host RAM, NVMe arena),
and the paged serving engine. The expert kernels come from
[`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm)
through the `[fast]` extra; that package owns the grouped NF4/MXFP4 GEMMs,
the int4 decode GEMV, the FP8 paged attention, the decode glue and the
host/NVMe primitives. When a change needs a new kernel, the kernel lands
there first and this package floors on the release that ships it.

## Repository map

| path | what |
|---|---|
| `experts4bit_qlora/` | the package: `loader.py` (streaming quantising loader), `lora.py`, `verify.py`, `arch/` (per-family conventions and load plans), `engines/` (fast, batched, offload, dense_offload, hot/pipelined/nvme residency, paged serving, int4 lanes, capture), `formats/`, `_vendor/experts.py` (vendored bitsandbytes class) |
| `tests/` | CPU-runnable suites (`pytest tests/ -q`); GPU-only tests skip with a reason |
| `docs/` | `STATUS.md`, `claims.json` + `claims-schema.md`, `CHOOSING.md`, `SOLUTIONS.md` + `solutions/`, `capabilities.json`, `INDEX.md` (which documents are current), anchored research records |
| `bench/` | harnesses and receipts (`bench/hybrid-g9/step_decomp.py` is the serving lane harness) |
| `scripts/`, `tools/` | link checker, wheel smoke, capability/discovery/llms checks, analysis tools |
| `audits/` | falsification work |

## Canonical public API

Exported from `experts4bit_qlora` (see `__all__` in `__init__.py`):
`load_moe_4bit_streaming` (lazy; needs `[train]`), `verify_moe_4bit`,
`Experts4bit`, `ExpertsNbit`, `ExpertsLoRA`, `enable_fast`,
`enable_fast_train`, `enable_batched_train`, `enable_dense_offload`,
`enable_nvme_residency`, `enable_mxfp4_nvme_residency`,
`enable_nvme_train_residency`, `enable_pipelined_residency`,
`hot_sets_from_profile`, `capture_decode` / `probe_capture`. Serving lanes live in
`engines/int4_experts.py` (`enable_serve_experts_int4`),
`engines/int4_attn_calib.py` (`calibrate_attention_hessians`,
`enable_serve_attn_int4_calib`), `engines/glue_fuse.py`, `engines/glue_r2.py`
and `engines/router_epilogue.py`; the fusion functions read
`E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2` and `E4B_FUSE_ROUTER_EPI`, and
`enable_from_env` reads `E4B_SERVE_ATTN_INT4_CALIB`. `E4B_SERVE_EXP_INT4` is
consumed by the bench harness hook, not by the package. CLIs are module mains, configured by
environment variables: `python -m experts4bit_qlora.{train,infer,serve,verify}`.
Names with a leading underscore are internal. The machine-readable list is
`docs/capabilities.json` (`entrypoints`).

## Sources of truth

- **Whether a number is current: `docs/claims.json`**, not CHANGELOG prose,
  not a README sentence, not a release note. A claim with `status`
  `retired` or `superseded` is never repeated as current; the register
  keeps it so the retraction is findable.
- **The current position: `docs/STATUS.md`.** It distinguishes measured,
  measured-private (a real run whose receipt is not in this repository),
  retired, superseded and open, and it is edited when the position moves.
- **Which documents are current: `docs/INDEX.md`.** Anchored documents
  (an `ots-attestation-footer` line, or a sibling `.ots`) are never edited
  in place — corrections go in a sibling file (`grep -l
  ots-attestation-footer docs/*.md` before touching anything under `docs/`).

## Build and test

```bash
pip install -e ".[test]"                      # CPU torch is enough for tests/
ruff check experts4bit_qlora tests scripts tools
pytest tests/ -q                              # GPU tests skip with a reason
python -m build && python -m twine check dist/*
python scripts/wheel_smoke.py                 # from outside the tree, against the wheel
python scripts/check_readme_links.py          # README links are absolute; self-refs = v<version> or main
python scripts/check_capabilities.py          # docs/capabilities.json vs schema, pyproject, source, claims
python scripts/check_discovery_contract.py    # docs/discovery-queries.json vs docs/solutions/
python scripts/build_llms_bundle.py --check   # llms-full.txt is current
```

Anything that needs a GPU, the network, a model download or a large disk is
run on a rented lane (`bench/`) and reported with a receipt; CI is CPU-only.
**A green skip is not evidence that a path was exercised**: the CI pins
`grouped-nf4-gemm` by commit before `.[test]` and has an import tripwire so
the int4 suites cannot silently skip. When you add a suite that can skip,
add its tripwire.

## Rules that have bitten

- Examples must not silently fall back. Every `enable_*` returns a count or
  raises; a documented example asserts the count (`0` and "still on the
  per-expert loop" look identical otherwise). `verify_moe_4bit(model,
  strict=True)` is the load-time assertion.
- Fusion and licensing decisions are made on module STRUCTURE (exact
  children, nothing of the module's own), never on class names.
- A K8 quality gate is applied in its registered units (perplexity budget
  from `experts4bit_qlora.k8_gate`), with a verdict column per arm; a noise
  floor is not a budget, and a calibrated pack needs the same sign on two
  texts before it is licensed.
- Do not publish a number before its receipt exists; if the receipt is
  private, the claim says `evidence_private`.

## When you change something

- **A public API** (signature, return, refusal condition): update its
  docstring (when to use, expected layout, what it returns or asserts,
  refusal conditions, platform needs, the solution page), the affected
  `docs/solutions/*.md`, `docs/capabilities.json` (`entrypoints`,
  `limitations`), this file, and `CHANGELOG.md`.
- **An extra or a dependency floor** (`[fast]`, `[train]`, `[serve]`,
  `grouped-nf4-gemm>=…`): update the README install block, the install
  commands in `docs/capabilities.json`, the pyproject comment that records
  why the floor moved, the CI pin, and the related package if the contract
  changed.
- **The README opening, `docs/SOLUTIONS.md`, `docs/STATUS.md`,
  `docs/capabilities.json` or a listed document:** regenerate
  `llms-full.txt` (`python scripts/build_llms_bundle.py`); `--check` is a
  CI gate.
- **A measured position:** the claim entry first, then `docs/STATUS.md`,
  then prose that quotes the claim ID.

## Platform caveats

Linux + CUDA is the tested environment; CI runs Python 3.11 (pyproject's
`>=3.9` floor is not exercised). The kernels need Triton on sm_80+; the
fp8 paged kernel's f32 compute modes fail on triton 3.4 (fp8 modes are the
default). Gemma-4 fails to load on some rented hosts (#344) and has no
quality instrument at 512-token resolution (#359); gpt-oss's raw-text
perplexity cannot rank an exact arm against a noisy one. B=1 decode is
host-bound: absolutes do not travel between hosts, ratios do.

## Release notes

First paragraph in ordinary language — which problem changed, who is
affected (families, hardware, environments), whether to upgrade — then the
mechanism, measurements with receipts and tiers, corrections and caveats.
See [`docs/RELEASE_NOTES_GUIDE.md`](docs/RELEASE_NOTES_GUIDE.md). Releases
are cut from `main` by the maintainer; do not tag or publish from a branch.
