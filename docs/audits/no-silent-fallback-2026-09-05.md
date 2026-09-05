# No-silent-fallback audit — 2026-09-05

**Scope.** The public enable/load/train/serve entry points of `experts4bit_qlora`, read for
one class of defect: *unsupported behaviour → plausible output → no explicit refusal, status
or count*. The invariant under test is `no-silent-fallback` in
[`docs/system-manifest.json`](../system-manifest.json): an accelerated path never falls back
silently; every `enable_*` returns a count or a handle to assert on, and a documented example
asserts it. The trigger was issue #397 (read, not run, while pre-registering the training
parity lane): with `arena_train=True` the streaming loader wrapped gpt-oss's meta
`GptOssExperts4bit` in `ExpertsLoRA`, whose epilogue silently substituted `silu(gate) * up` for
the biased, clamped GLU.

**Method.** Each entry point was read for (a) what it does when the module in front of it is
outside what it can reproduce, (b) whether that decision is made on module STRUCTURE (buffers,
parameters, attributes, the class's own `forward`, the names its code references) or on a
class/family name, and (c) whether the caller can tell from the return value or the log. Only
concrete structural ambiguity in this branch is fixed; everything else is listed with its
reason. The fix for #397 is one contract in one place — `assert_stock_epilogue` in
`experts4bit_qlora/lora.py` — applied at every seam that wraps, trains over or attaches to an
`ExpertsLoRA`, and the CPU tests are `tests/test_epilogue_contract.py`.

**Verdict column.** `FIXED` — a change on this branch (`fix/397-structural-refusal`);
`already refuses` — the entry point raised, skipped-with-reason or reported before this
branch, and the decision is structural; `accepted-with-reason` — the behaviour is documented
and the fallback is faithful or logged, so it is not a silent wrong answer; `OPEN` — a real
finding this branch does not fix, with the recommended fix.

## The contract

`assert_stock_epilogue(module) -> None | raises EpilogueContractError` (a `TypeError`, so the
test guards that skip on an absent quantizer can never report it as a green skip). It reads:

| structure inspected | passes | refused |
|---|---|---|
| registered buffers / parameters / plain tensor attributes whose name contains `bias` | none | `gate_up_bias`, `down_bias`, `gate_up_proj_bias`, `down_proj_bias`, any `*bias*` |
| `type(module).forward` identity vs the vendored and resolved `ExpertsNbit`/`Experts4bit` forwards | stock, or non-stock with a callable `_apply_gate` | non-stock and no hook |
| epilogue scalars `alpha`, `limit`, `swiglu_alpha`, `swiglu_limit` | present only with a hook (V4's `limit`) | present without a hook (the stock epilogue ignores them; `hot_residency` reads `limit` — two answers for one module) |
| names in the forward's code object (`co_names`) | none of `clamp`, `clamp_`, `sigmoid`, `*bias*` on a non-stock forward | a forward that clamps, applies the sigmoid GLU or adds a bias in its own body rather than in the hook |
| interleave markers `interleaved`, `gate_up_interleaved`, `_e4b_interleaved` | absent (`from_gptoss` de-interleaves at load) | any set |
| `act_fn` | one required positional (SiLU, Gemma-4's gelu_tanh, an `nn.Module`) | none, or a callable that needs two (a GLU in disguise) |
| declared `_gate_up_shape` / `_down_shape` vs `intermediate_dim` / `hidden_dim` | `[2I, H]` (or `[I, H]` non-gated) and `[H, I]` | anything else |

Enumerated on the shipped classes (`describe_epilogue_structure`): `Experts4bit` /
`ExpertsNbit` — pass; `DeepseekV4Experts4bit` / `DeepseekV4ExpertsNbit` — non-stock forward
**with** `_apply_gate`, `limit` present, forward body references nothing of the epilogue —
pass; `GptOssExperts4bit` / `GptOssExpertsNbit` — `gate_up_bias` + `down_bias`, `alpha` +
`limit`, non-stock forward, no hook, forward body references `clamp`, `sigmoid`,
`gate_up_bias`, `down_bias` — refused on four independent grounds. The convention classes for
Gemma-4, Granite and Mixtral are storage conventions (`arch/moe_conventions.py`); their
experts are stock `Experts4bit` with the family's `act_fn` and pass.

## Entry points

| entry point | what could be unsupported | behaviour before this branch | verdict |
|---|---|---|---|
| `ExpertsLoRA(base)` (`lora.py`) | a base whose forward is not `down(act_fn(gate)*up)`: biases, clamp, interleaved rows, a family-specific forward with no hook | wrapped anything; `_epilogue` fell back to `act_fn(gate)*up` — plausible, wrong, no error | **FIXED**: `assert_stock_epilogue(base)` in `__init__`, `EpilogueContractError` names the attributes and the faithful route (`mxfp4_qlora.ExpertsMxfp4LoRA`) |
| `load_moe_4bit_streaming(..., arena=, arena_train=True)` (`loader.py`) | gpt-oss's meta stack (biases registered, `alpha`/`limit` set) wrapped for training | wrapped it (#397) | **FIXED**: the same check, re-raised with the layer and `model_type`; an end-to-end CPU test on a gpt-oss-shaped checkpoint and NF4 arena |
| `load_moe_4bit_streaming` resident gpt-oss branch | `r`/`alpha` are required positionals; the stack is built bare with no adapter | documented in the docstring; nothing in the log | **FIXED** (log): a one-time NOTE names the bare family and the faithful route; the bare build itself is correct (`GptOssExperts4bit.forward` is the faithful epilogue) |
| `load_moe_4bit_streaming`, `hidden_act` not in transformers' `ACT2FN` | the module keeps its default `silu` (Kimi K3 declares `situ`) | `NOTE: activation ... will use the module default. Verify numerics` in the log; no refusal | **OPEN** (listed, not fixed): a resident (non-arena) build of such a family computes the wrong activation with a log line as the only signal; the K3 serve engine (`Mxfp4NvmeResidencyK3`) carries SiTU itself, so the arena route is faithful. Recommended: refuse the resident build when `arena_index is None` and the activation is unknown, adding the message to `tests/quant_guard.LOADER_REFUSALS`. Not done here: it changes a loader refusal condition outside #397's scope. |
| `load_moe_4bit_streaming` family dispatch (`model_type == "deepseek_v4"` selects `DeepseekV4Experts4bit`; `gptoss_arena` selects `GptOssExperts4bit` by the presence of `gate_up_proj_bias` keys) | a clamped family under a model_type not in the allowlist | `_declares_clamped_swiglu(config)` refuses any convention-admitted family that declares `swiglu_alpha`/`swiglu_limit`; V4's selection is by name over an adjudicated allowlist | accepted-with-reason: family dispatch over `SUPPORTED_ARCHITECTURES` is the loader's adjudication step, and the config gate refuses the clamped epilogue outside it |
| `enable_nvme_train_residency` (`engines/nvme_train.py`) | attaches to any `ExpertsLoRA`, whatever its base computes | attached; the wrapper's unfaithful forward then trained against the arena bytes | **FIXED**: `assert_stock_epilogue(base)` per module in the pre-flight, before the tier opens; a refusal stamps nothing |
| `enable_fast_train`, `enable_batched_train` | an `ExpertsLoRA` whose base has a non-stock forward and no hook | skipped, not counted (verbose reason) — but the wrapper's reference forward is the same unfaithful re-implementation, so "skip" left it training the wrong function | **FIXED**: refused (`EpilogueContractError`) through the shared contract; the constructor already makes such a wrapper unreachable except by swapping the base afterwards, which is the case the tests construct |
| `enable_fast` wrapper loop | same | same skip | **FIXED**: refused, before the storage-eligibility test |
| `enable_fast` bare loop | a bare module with a forward the grouped kernel cannot reproduce | skipped by class-forward identity + `_apply_gate` presence | **FIXED** (unified): the skip predicate is the contract; a bare module keeps its own faithful forward, so skipping is correct and the count excludes it |
| `enable_batched_train`'s per-call fallback (`_PAD_WASTE_LIMIT`, evicted storage, empty batch) | the arm claims "batched" while some layers or calls ran the reference | the fallback is correct and invisible from the output; nothing counted it (tp1 read OLMoE's batched arm as VOID) | **FIXED**: `mod._e4b_batched_stats` (`calls`, `batched`, `fallback_calls`, `by_reason`) and `batched_fallback_stats(model)`; the module docstring's example asserts `fallback_calls == 0` |
| `enable_hybrid_train` (`engines/hybrid_train.py`) | gpt-oss's biased epilogue (`_glu` reproduces stock + the V4 clamp only) | refused on the tier state's `gptoss` flag (itself structural: `alpha is not None and hasattr(mod, "gate_up_bias")`) | **FIXED** (unified): refused through the contract with the module index; any future biased or hook-less family is caught by the same check |
| `enable_mxfp4_nvme_residency` (`engines/nvme_experts.py`) | a bias-carrying module (gpt-oss) under an engine that receives no biases and defaults to `Mxfp4NvmeResidencyV4` (bias-free clamped SwiGLU) | bound; the released bytes were served through the wrong epilogue with the biases dropped, `limit` taken from the module | **FIXED**: refused on `describe_epilogue_structure(mod)["bias_tensors"]` before any engine or tier is built; the message names `enable_nvme_residency` (NF4 arena; the residency state carries the biases) and the paged engine's native MXFP4 store |
| `mxfp4_experts_forward` fused lane (an MXFP4-arena module under `build_meta_experts`) | `_epilogue(mod, proj)` on a module with no hook | would substitute `act_fn(gate)*up`; unreachable today for gpt-oss (its 4-segment arena does not fuse to the V4 view, so `_redeclare_for_mxfp4_arena` returns False) | **FIXED** (guard): `_e4b_mxfp4_fused_ok` decided by the contract at attach; a refused module keeps its own forward |
| `enable_nvme_residency` (NF4 arena serving) | gpt-oss biases on a meta module | `_NvmeResidency(_HotResidency)` derives the epilogue structurally (`alpha` + `gate_up_bias` → biases + clamped GLU; `limit` alone → V4 clamp); bias device pinned by `tests/test_gptoss_arena_bias.py` | already handles |
| `enable_hot_residency`, `enable_cold_engine`, `enable_pipelined_residency`, `enable_hybrid_tier` | a module whose forward the engine does not reproduce | allowlisted by class-forward IDENTITY (`stock_forwards` ∪ the gpt-oss and V4 forwards, never a name string); anything else skipped with a verbose reason, its `hot_sets` entry consumed, the count excluding it; the epilogue itself derived from structure (`gptoss`, `clamp_limit`, `act_fn`); pipelined additionally warns when a patched wrapped base is unreachable | already refuses |
| `enable_serve_experts_int4[_calibrated]` (`engines/int4_experts.py`) | gpt-oss's interleaved MXFP4 stacks and biased epilogue on the int4 grid | `mt == "gpt_oss"` selects the NATIVE MXFP4 store (a name selects the store kind); the epilogue is the wrapper's structural `gptoss` flag, and `_check_gptoss_wrapper` raises when the flag or the bias widths are missing; tiered layers, unknown model_type, calibrated enable without Hessians all raise | accepted-with-reason: the name is a store-kind selector over a family the loader already adjudicated; the epilogue decision is structural and refuses loudly |
| `enable_serve_attn_int4`, `enable_serve_attn_int4_calib`, `enable_from_env` | attention modules of another shape | structural (`type(child) is nn.Linear` under `*Attention`; the dense MLP and output head opt-ins refuse a missing structure); raise when the flag is set and nothing matched | already refuses |
| `fuse_t1_glue` (`E4B_FUSE_T1_GLUE`), `fuse_t1_glue_r2` (`E4B_FUSE_T1_GLUE_R2`), `fuse_router_epilogue` (`E4B_FUSE_ROUTER_EPI`) | a norm, layer body or router of another structure (centred RMSNorm, residual-scaled Granite body, norm-less attention, gpt-oss/Gemma-4 routers) | structure checked, then the module's OWN forward probed against the kernel (licensing on structure, never a name); raise when the flag is set and nothing was patched; return 0 when the flag is off | already refuses |
| `fuse_qkv` (`engines/qkv_fuse.py`) | attention of another class; biased q/k/v | selects `Qwen3MoeAttention` by NAME, then verifies the children structurally and raises on a missing one or a bias; returns 0 on other families | accepted-with-reason: a name-narrowed selector that can only under-engage; the bench harness (`bench/hybrid-g9/step_decomp.py`) refuses a vacuous `--fuse-qkv` arm and prints the count |
| `E4B_FUSE_SWIGLU` / `E4B_FUSE_COMBINE` (hot-residency decode glue) | a non-SiLU `act_fn` (Gemma-4's gelu_tanh), non-bf16 rows | `_is_silu(act_fn)` (identity of `F.silu`, or the activation CLASS name `SiLU`/`SiLUActivation`) gates the kernel; otherwise the three-launch torch chain `act_fn(gate) * up`, which is the faithful epilogue | accepted-with-reason: the fallback is faithful, not wrong; engagement is census PRESENCE of the kernel op (the op-level census instrument), and the flag is the A/B arm. The class-name test admits only a known SiLU class, so it can only under-engage. |
| paged attention (`engines/paged_attention.py`, `paged_runner.py`) | sliding windows, sinks, per-head q/k norms, the scale | window from the module or the interface kwargs, sinks from `module.sinks`, `sm_scale` forwarded (#336); regime mismatches raise (`ValueError`); norms run in the module before the kernel | already handles (by structure); the OPEN quality items are Gemma-4's batch-shape chaos (#359) and the load fault (#344), which are instrument questions, not silent substitutions |
| `capture_decode` / `probe_capture` (`engines/capture.py`) | a model whose decode step does not replay | `probe_capture` classifies the divergence (teacher-forced delta) and reports it; capture is one try per process | already reports |
| `enable_dense_offload` | a model without `...layers.<i>` blocks | raises `ValueError`; touches nothing inside an expert module | already refuses |
| `verify_moe_4bit(strict=True)` | a stack left in high precision | raises naming the first offender | already refuses — note it proves storage, never adapter presence |
| `python -m experts4bit_qlora.train` with `TRAIN_EXPERTS=1` (the default) on a bare-expert family | no expert adapter exists; attention LoRA keeps the trainable count nonzero | trained attention/router only and reported a loss curve | **FIXED**: `SystemExit` naming `TRAIN_EXPERTS=0` and the MXFP4 route |
| `python -m experts4bit_qlora.serve` | an adapter spec over a bare-expert family (no `ExpertsLoRA` to receive expert keys) | nothing said about the bare stack; `serve`'s own warning covers a different case — a non-zero adapter turning pipelined residency off (the wrapper stops delegating) | accepted-with-reason: the loader's NOTE now says the stack is bare at load; expert adapter keys for such a family have no module to land on, and the loader/`verify_moe_4bit` route stays correct for the frozen stack |

## What changed on this branch

- `experts4bit_qlora/lora.py`: `EpilogueContractError`, `STOCK_EPILOGUE_ROUTE`,
  `describe_epilogue_structure`, `assert_stock_epilogue`; `ExpertsLoRA.__init__` calls it.
- `experts4bit_qlora/__init__.py`: exports `EpilogueContractError`, `assert_stock_epilogue`,
  `batched_fallback_stats`.
- `experts4bit_qlora/loader.py`: the `arena_train=True` branch asserts the contract (layer +
  family in the message); the resident gpt-oss branch logs the bare build once.
- `experts4bit_qlora/engines/nvme_train.py`: pre-flight contract per module.
- `experts4bit_qlora/engines/fast.py`: `_refuse_wrapped`; the wrapper loops of `enable_fast`
  and `enable_fast_train` refuse; the bare loop skips through the contract.
- `experts4bit_qlora/engines/batched.py`: refuses through `_refuse_wrapped`; per-call counters
  and `batched_fallback_stats`.
- `experts4bit_qlora/engines/hybrid_train.py`: refusal through the contract.
- `experts4bit_qlora/engines/nvme_experts.py`: `enable_mxfp4_nvme_residency` refuses
  bias-carrying modules; the MXFP4 fused lane is gated by `_e4b_mxfp4_fused_ok`.
- `experts4bit_qlora/train.py`: `TRAIN_EXPERTS=1` with no expert adapter refuses.
- `tests/test_epilogue_contract.py` (new), `tests/test_batched_train.py` (refusal + counters).

Not touched, by design of the parallel release bundle: `docs/claims.json`, `docs/STATUS.md`,
`docs/capabilities.json`, `README.md`, the version. No claim value moves; the contract is a
refusal, not a measurement.
