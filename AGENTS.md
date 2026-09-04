# AGENTS.md — working in experts4bit-qlora

- Starting from a model, training, serving, or fit problem? → experts4bit-qlora (this repository).
- Starting from a packed layout, kernel, GEMM/GEMV, attention, or arena primitive? → grouped-nf4-gemm.
- Need a current number? → docs/claims.json.
- Need current support/status? → docs/STATUS.md + docs/capabilities.json.
- Changing public behaviour? → AGENTS.md: 'When you change something'.

## 2. Purpose and ownership boundary

`experts4bit-qlora` (import `experts4bit_qlora`) trains and serves fused
Mixture-of-Experts models in 4-bit on hardware that cannot hold them in
bf16: model loading, quantisation orchestration (`Experts4bit` / `ExpertsNbit`
over the fused 3-D expert stacks bitsandbytes' walker skips), per-expert LoRA
(`ExpertsLoRA`), training, residency (host RAM, NVMe arena) and the paged
serving engine. The kernels (GEMMs, decode GEMV, FP8 paged
attention, decode glue, host/NVMe primitives) come from
[`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm)
through the `[fast]` extra; a kernel lands there first and this package
floors on the release that ships it. The machine-readable boundary — what
each package owns, which version of this package needs which kernel release
and why (`compatibility`), the evidence vocabulary, the invariants — is
[`docs/system-manifest.json`](docs/system-manifest.json), byte-identical in
both repositories and validated by `scripts/check_system_manifest.py`.

## 3. Repository map

| path | what |
|---|---|
| `experts4bit_qlora/` | the package: `loader.py` (streaming quantising loader), `lora.py`, `verify.py`, `k8_gate.py` (the registered quality gate), `arch/` (per-family conventions and load plans), `engines/` (fast, batched, offload, dense_offload, hot/pipelined/nvme residency, paged serving, int4 lanes, capture), `formats/`, `_vendor/experts.py` (vendored bitsandbytes class) |
| `tests/` | CPU-runnable suites (`pytest tests/ -q`); GPU-only tests skip with a reason |
| `docs/` | `STATUS.md`, `claims.json` + `claims-schema.md`, `capabilities.json`, `system-manifest.json`, `change-impact.json`, `discovery-queries.json`, `CHOOSING.md`, `SOLUTIONS.md` + `solutions/`, `INDEX.md` (which documents are current), anchored research records |
| `bench/` | harnesses and receipts (`bench/hybrid-g9/step_decomp.py` is the serving lane harness) |
| `scripts/`, `tools/` | the checks in section 6, wheel smoke, link checker, analysis tools |
| `audits/` | falsification work |

## 4. Sources of truth

- **Numbers: `docs/claims.json` wins** over CHANGELOG prose, README
  sentences and release notes; a `retired` or `superseded` claim is never
  repeated as current and never deleted.
- **Position: `docs/STATUS.md` wins**; it is edited when the position moves.
- **Historical and anchored records are never rewritten.** `docs/INDEX.md`
  says which documents are current; a sibling `.ots` or an
  `<!-- ots-attestation-footer -->` line means corrections go in a sibling file.
- **A green skipped test is not evidence** that a path was exercised.
- **A private measurement is not publicly reproducible**: `measured-private`
  is real, labelled, and not checkable from this repository.
- **A noise floor is not a budget**: the registered gate is applied in its
  own units, with a verdict column per arm.
- **Failed gates stay failed**; a gate is never retuned to fit a result.
- **Dependency floors: `pyproject.toml` wins**; the manifest's `compatibility`
  records say which version needs which kernel release and why, and
  `scripts/check_dependency_floor.py` holds every current document to it.

Evidence words: the manifest's `evidence_vocabulary`; these rules: its `invariants`. Point there.

## 5. Public API and capability map

Exported from `experts4bit_qlora` (`__all__` in `__init__.py`; a leading
underscore means internal): `load_moe_4bit_streaming` (lazy; needs `[train]`),
`verify_moe_4bit`, `Experts4bit`, `ExpertsNbit`, `ExpertsLoRA`, `enable_fast`,
`enable_fast_train`, `enable_batched_train`, `enable_dense_offload`,
`enable_nvme_residency`, `enable_mxfp4_nvme_residency`,
`enable_nvme_train_residency`, `enable_pipelined_residency`,
`hot_sets_from_profile`, `capture_decode` / `probe_capture`. The serving lanes
(`engines/int4_experts.py`, `int4_attn_calib.py`, `glue_fuse.py`, `glue_r2.py`,
`router_epilogue.py`), the `E4B_*` flag each reads and which surfaces consult
it are the lever table in [`docs/solutions/serve-large-moe-on-a-consumer-gpu.md`](docs/solutions/serve-large-moe-on-a-consumer-gpu.md).
CLIs are module mains configured by environment variables
(`python -m experts4bit_qlora.{train,infer,serve}`; `verify --manifest` checks
a placement manifest, not a model — `verify_moe_4bit` does that). The
capability map — six ids, each with entry points, environment, limitations,
claim IDs and its solution page — is [`docs/capabilities.json`](docs/capabilities.json),
validated by `scripts/check_capabilities.py`; the human index is [`docs/SOLUTIONS.md`](docs/SOLUTIONS.md).

## 6. Build and test

```bash
pip install -e ".[test]"                         # CPU torch is enough for tests/
ruff check experts4bit_qlora tests scripts tools
pytest tests/ -q                                 # GPU tests skip with a reason
python -m build && python -m twine check dist/*
python scripts/wheel_smoke.py                    # from outside the tree, against the wheel
python scripts/check_readme_links.py             # README links are absolute; self-refs = v<version> or main
python scripts/check_capabilities.py             # docs/capabilities.json vs schema, pyproject, source, claims
python scripts/check_system_manifest.py          # docs/system-manifest.json vs pyproject, claims, capabilities; --sibling <kernel checkout>
python scripts/check_dependency_floor.py         # every current document states pyproject's [fast] floor
python scripts/check_change_impact.py --base origin/main    # docs/change-impact.json: what must move together
python scripts/check_discovery_contract.py --bm25 --bm25-min-top1 30   # queries -> pages; the BM25 floor is a local proxy
python scripts/check_docs_examples.py --root .   # doc code blocks parse; local links resolve
python scripts/build_llms_bundle.py --check      # llms-full.txt is current
```

Anything that needs a GPU, the network, a model download or a large disk runs
on a rented lane (`bench/`) with a receipt; CI is CPU-only, pins
`grouped-nf4-gemm` by commit before `.[test]` and has an import tripwire so the
int4 suites cannot silently skip (a suite that can skip gets one). A BM25
regression, or gain, is never evidence of LLM discoverability; the floor only
catches a corpus that stopped routing its own queries.

## 7. Rules that have bitten

- Examples must not silently fall back: every `enable_*` returns a count or a
  non-empty handle list, or raises, and a documented example asserts it (`0`
  and "still on the per-expert loop" look identical otherwise).
  `verify_moe_4bit(model, strict=True)` is the load-time assertion.
- Fusion and licensing decisions are made on module STRUCTURE (exact
  children, nothing of the module's own), never on class names.
- A K8 quality gate is applied in its registered units (perplexity budget
  from `experts4bit_qlora.k8_gate`), with a verdict column per arm; a
  calibrated pack needs the same sign on two texts before it is licensed.
- Do not publish a number before its receipt exists; if the receipt is
  private, the claim says `evidence_private`.

## 8. When you change something

The contract is [`docs/change-impact.json`](docs/change-impact.json);
`scripts/check_change_impact.py --base <ref>` reads the diff, detects the
mechanical triggers and names the missing companions (CI runs it on pull
requests). **public-api-change** (a symbol enters or leaves `__all__`; a
signature, return or refusal condition moves): docstring,
`docs/capabilities.json` or the solution page, and `CHANGELOG.md` — warns,
`--strict` fails. **new-kernel-capability** (this package starts using
something the kernel package released): the version guard, and whether the
`[fast]` floor must rise. **measured-result** (a claim added, or its
`status`/value moved): `docs/STATUS.md` in the same diff, then prose that
quotes the claim ID. **dependency-floor** (`pyproject.toml`'s `[fast]`
floor moved): `docs/system-manifest.json`, `docs/capabilities.json`, the CI
`--requires` assertion and every current solution document in the same
diff; the pyproject comment ladder says why. Regenerate `llms-full.txt`
(`python scripts/build_llms_bundle.py`) when a bundled document changed.
Releases are cut from `main` by the maintainer ([`docs/RELEASE_NOTES_GUIDE.md`](docs/RELEASE_NOTES_GUIDE.md)).

## 9. Platform caveats

Linux + CUDA is the tested environment; CI runs Python 3.11 (pyproject's
`>=3.9` floor is not exercised). The kernels need Triton on sm_80+; the
fp8 paged kernel's f32 compute modes fail on triton 3.4 (fp8 modes are the
default). Gemma-4 fails to load on some rented hosts (#344) and has no
quality instrument at 512-token resolution (#359); gpt-oss's raw-text
perplexity cannot rank an exact arm against a noisy one. B=1 decode is
host-bound: absolutes do not travel between hosts, ratios do.
