# UPSTREAM-NOTES — what each competitor actually does at the versions TC1 runs (read from source, 2026-10-01)

Scope: the facts the TC1 arms depend on. Every line below was read in the released source (versions and the
inspection notes' provenance in the private campaign tree, `campaign-2026-10-01/upstream/`), not assumed. Anything not
confirmed from a file is marked UNVERIFIED.

## Hugging Face stack — transformers 5.18.0, bitsandbytes 0.50.2, peft 0.21.2

- Every MoE in scope stores experts as fused 3-D `nn.Parameter` stacks on an `*Experts` module: `gate_up_proj`
  [E, 2I, H] and `down_proj` [E, H, I] (F.linear, [out, in]) for qwen3_moe / olmoe / mixtral / qwen3_5_moe;
  granitemoe under `block_sparse_moe.experts`; gemma4's experts hang off the decoder layer; gpt_oss is transposed,
  interleaved and biased ([E, H, 2I], [E, I, H], `x @ W[e] + b`). Verified on CPU for qwen3_moe (tiny config).
- transformers' bitsandbytes quantizer converts `nn.Linear` (and Conv1D) only
  (`quantizers/quantizer_bnb_4bit.py:91-95`, `integrations/bitsandbytes.py:189`): under `load_in_4bit` the expert
  stacks and the routers stay bf16. bitsandbytes 0.50.2 has no `Experts4bit`-like class; its only arbitrary-shape
  facility is `nn/parametrize.py` (`replace_parameter_4bit`, double-quant default OFF, blocksize 64).
- The experts implementation defaults to `grouped_mm` with no train/eval switch (`modeling_utils.py:1890`); on
  torch <= 2.8 `torch._grouped_mm` is Hopper-only (`integrations/moe.py:303-312`), so on an RTX 5090 (sm_120) with the
  image's torch 2.8.0 the per-expert Python fallback runs silently (`:185-200`). The receipt records
  `config._experts_implementation` and the dispatch outcome.
- PEFT `target_parameters` on a 3-D stack: one A_e [r, H] / B_e [2I, r] per expert, stored as `lora_A.default.weight`
  [E*r, H] (rows e*r:(e+1)*r) and `lora_B.default.weight` [2I, E*r] (`peft/tuners/lora/layer.py:2455-2456, 2534-2535`);
  two stacks on one module nest two wrappers (the outer's `parameter_name` says which); init kaiming-uniform
  (bound 1/sqrt(fan_in)) on the full A, B zeros, scaling alpha/r (`:338-343, 2459-2462`). The delta is applied
  WEIGHT-SIDE: `W_eff = W + s * B_e A_e` materialised for the whole stack every forward through a parametrisation
  (`:2319-2323, 2581-2599`), not `x A^T B^T`. Dropout, DoRA and regex targets are rejected for parameters.
- trl's SFTTrainer inherits the checkpoint's `router_aux_loss_coef` and force-enables router logits
  (`trainer/sft_trainer.py:1431-1443`) — irrelevant to this harness (its own loop, plain LM loss), stated so a
  reader does not expect trl's loss.

## axolotl 0.20.0 (released 2026-09-30)

- Pins (wheel METADATA): python >= 3.12; torch >= 2.13.0, <= 2.14.0; transformers == 5.17.0; peft == 0.21.0;
  trl == 1.13.0; bitsandbytes == 0.50.2; triton >= 3.4; torchao == 0.18.0; liger-kernel == 0.8.1. It does not install
  on the image's python 3.11 / torch 2.8.0; TC1 builds it its own venv (uv, CPython 3.12, torch 2.14.1+cu130 — torch >= 2.13 Linux wheels exist only under
  cu130, so the driver >= 580 gate applies) and records that torch beside its row.
- Plain `load_in_4bit: true` + `adapter: qlora`: experts bf16 (transformers' quantizer, above);
  `lora_target_parameters` maps 1:1 to PEFT `target_parameters` (`loaders/adapter.py:195`); `lora_dropout` must be 0.
- Its 4-bit expert path is `quantize_moe_experts: true` (`utils/schemas/config.py:955-964`): a patch on
  `transformers.core_model_loading.set_param_for_module` quantises every CUDA parameter with `ndim >= 3` whose dotted
  name contains "expert" (`monkeypatch/moe_quant.py:150-181`) with bitsandbytes' `replace_parameter_4bit`
  (NF4, blocksize 64, **double-quant forced True**, `:134-136, 170-175`); dequantise-on-access, the whole stack per
  forward with transformers' built-in experts forward (`chunked_bnb.py:3-6`). Its PEFT patches
  (`moe_quant.py:192-429`: definition-order injection, parametrize-cache eviction in `_activate_lora`, the
  `_remove_parametrizations` fix) are REQUIRED — peft 0.21.x over a bnb parametrisation otherwise leaks the proxy and
  "the delta compounds on every forward". The parametrize-cache gate hooks (PR #3915, merged 2026-09-16) are inside
  bitsandbytes 0.50.2 for the 4-bit path.
- Its native-best expert kernel is ScatterMoE (`integrations/kernels/libs/scattermoe_lora/`): a Triton grouped GEMM
  with the LoRA factors fused in (`parallel_linear_lora.py:5-24`) that keeps bnb-4-bit experts packed and dequantises
  selectively (`experts.py:684-738`, `moe_bnb_fast` default True), "any CUDA" GPU; enabled by
  `plugins: [axolotl.integrations.kernels.KernelsPlugin]`, `expert_backend: scattermoe`. SonicMoE needs sm_90+
  (`integrations/kernels/plugin.py:27-33`; consumer Blackwell only with a special build) and has no bnb branch
  (UNVERIFIED) — not a TC arm. DeepGEMM has no backward (`args.py:392-397`); MegaBlocks was "tested but not
  integrated"; the Liger integration has no experts kernel.
- Native NVFP4 LoRA (0.20.0 highlight) is for ModelOpt NVFP4 checkpoints / TorchAO NVFP4 dense weights, with
  merge-aware training needing SonicMoE (sm_90+): a different base than the pinned bf16 checkpoints — outside TC1's
  matched regime; a native-best NVFP4 row would need the NVFP4 checkpoint and is out of scope here.
- Memory levers (TC3): `layer_offloading` (whole decoder layers, shipped); DeepSpeed ZeRO-3 parameter offload only
  for bf16 experts (bnb-quantised models skip `zero.init()`, `modeling_utils.py:1440`); the expert-granular CPU
  offload is PR #3797 (OPEN, CHANGES_REQUESTED, not in 0.20.0).

## Stacks with no single-GPU 4-bit MoE expert LoRA path (read, not assumed)

LLaMA-Factory 0.9.5 (bnb Linear-only, LoRA targets by Linear class, no target_parameters); Liger-Kernel 0.8.4 (a
Triton fused MoE for bf16 full fine-tuning, no 4-bit); torchao 0.18.0 (NF4Tensor asserts dim <= 2; its MoE training
prototype is fp8/mxfp8/nvfp4 grouped-mm); torchtune 0.6.1 (no MoE); DeepSpeed 0.19.7 (no bitsandbytes awareness;
ZeRO++ int8 comm quant); AnswerDotAI/fsdp_qlora (Linear-only). None is an arm; each is a stated absence.

## Unsloth 2026.9.14 / unsloth_zoo 2026.9.9 (released 2026-10-01; wheel == commit 1b58838 / zoo 92638fb)

- Its 4-bit MoE path (transformers 5.x only, `common.py:80`): bitsandbytes NF4 `Params4bit` on the fused 3-D expert
  stacks (`moe_utils_bnb4bit.py:506 replace_expert_params_with_bnb_params`, packed 2-D uint8 + `_original_shape`),
  then **dequantise to bf16 + a dense grouped GEMM every forward** — there is no 4-bit expert GEMM kernel. The
  dequant is a Triton NF4/64 kernel bit-identical to bitsandbytes' (`moe_triton_kernels.py:212`) or `dequantize_4bit`.
- **The backend decides the speed, and torch decides the backend.** `select_moe_backend()` (`moe_utils.py:1062`):
  `UNSLOTH_MOE_BACKEND` in {`grouped_mm`, `unsloth_triton`, `native_torch`}, default preference grouped_mm ->
  unsloth_triton -> native_torch. `grouped_mm` = `torch._grouped_mm`, which **on torch 2.8 runs only on sm_90**
  (code comment `:374-378`: "a LoRA MoE died on every card but an H100; 2.9 falls back internally"); the Triton
  grouped GEMM is bf16-dense only; `native_torch` is a compiler-disabled per-expert Python loop. So on the RTX 5090
  with the torch-2.8 image every earlier lane (P38, tp2, tp4: `moe_backend native_torch` in their receipts) measured
  Unsloth on its slowest path. Its own code names the fast route: "For max throughput use the grouped_mm backend"
  (`moe_bnb.py:68-70`); its installer names the Blackwell route torch 2.12.x+cu130 (`_auto_install.py:44`;
  extra `unsloth[cu130-torch2121]`); the wheel caps `torch<2.13`, `transformers<=5.5.0`.
- The LoRA delta on experts is **separated** from the base GEMM (default; `UNSLOTH_MOE_LORA_MERGED=1` folds): PEFT's
  `ParamWrapper.forward` is replaced and `(A_e, B_e, scaling)` are applied as two more grouped GEMMs on the
  expert-sorted rows (`moe_utils.py:3836-3880, 4091-4116`) — an x-side delta like e4b's, not PEFT's weight-side fold.
  A/B come from PEFT (kaiming-uniform A over the whole [E*r, in], B zeros, scaling alpha/r; lora_B's E*r axis is
  expert-fastest: column j belongs to expert j % E); fp32 factors cast to the activation dtype per forward.
- Memory defaults: bnb stacks are re-dequantised in backward (`UNSLOTH_MOE_RECOMPUTE`, `UNSLOTH_MOE_GC_REPLAY_PIN`
  tilt it); `use_gradient_checkpointing="unsloth"` forces reentrant checkpointing and offloads only each layer's
  input hidden state to pinned CPU, and only at `max_seq_length >= 512`; MoE layers sit inside the checkpointed
  region (expert forward runs twice). The gate-gradient identity (`UNSLOTH_MOE_GATEGRAD=1`) is on by default.
- Per family: qwen3_moe, qwen3_5_moe (shared expert = a dense sigmoid-gated MLP with ordinary Linear4bit LoRA),
  mixtral, gemma4 are patched and take the dispatcher; **OLMoE** is "a stacked-expert family Unsloth does not patch"
  (`moe_utils.py:2480, 2876`) -> PEFT's dense fold or a generic route (UNVERIFIED); **Granite** `ParallelExperts`
  (`input_linear`/`output_linear`) are never discovered by `get_moe_target_parameters` and never quantised by the
  bnb patch (`_is_expert_module` needs `gate_up_proj` + `down_proj`): its experts stay bf16, so no 4-bit expert LoRA
  exists for that family; **gpt-oss** 4-bit uses per-expert `Linear4bit` ModuleLists and trains LoRA through a
  per-expert loop (`gpt_oss.py:1214-1300`), while the 16-bit load keeps the MXFP4 experts packed
  (`mxfp4.py:141`, `Mxfp4ExpertParam` + fused MXFP4 grouped GEMM, dX only) — needs the grouped_mm backend and no
  offload; DeepSeek-V4 FP4 and Kimi-K3 MXFP4 expert stacks stay packed; NVFP4 and compressed-tensors INT4 "packed"
  training applies to 2-D Linears only, not expert stacks.
- Pre-quantised `unsloth/Qwen3-30B-A3B-*bnb-4bit` repos are **redirected to the bf16 repo** and quantised on the fly
  (`loader_utils.py:55-61`): there is no distinct Unsloth checkpoint arm. Which `BitsAndBytesConfig` (double-quant)
  the MoE loader builds is UNVERIFIED from source; the receipt records the quant state it finds.
- Claims in its README ("Train MoE LLMs 12x faster with 35% less VRAM") have no in-repo benchmark; the only in-repo
  numbers are kernel microbenchmarks on B200/H100 and one Qwen3-30B-A3B step time (0.664 -> 0.622 s/step, hardware
  and fixture unstated).

## TC3 addendum (2026-10-02, read, not run): what the memory levers are at the versions TC3 drives

- axolotl `layer_offloading` (`utils/schemas/config.py:653`, `bool | None`, default False) is a TRAINER lever: `core/builders/base.py:623` copies it into
  the TrainingArguments and `core/trainers/mixins/layer_offloading.py:280-305` (`LayerOffloadingMixin.__init__`) builds `LayerOffloadManager(model, num_prefetch=1)`
  + `setup_hooks()` and wraps each `training_step` in `_LayerOffloadContext(manager)` (`pre_step` / `post_step`). `ModelLoader` never reads the key. The manager
  (`:52-116`) finds the decoder `ModuleList`, moves every layer's frozen (`requires_grad=False`) params to pinned CPU buffers (`param.data = cpu_buf`, `:130-147`)
  and installs forward/backward pre/post hooks per layer (`:177-235`) that load layer N and prefetch N+1 on a transfer stream; trainable params stay on the GPU.
  Both classes are plain and trainer-free, so `tc1_arm.py --axolotl-layer-offload 1` drives them exactly as the mixin does, around every micro-batch.
- axolotl + DeepSpeed ZeRO-3: `ModelLoader` reads `cfg.deepspeed` only for `HfTrainerDeepSpeedConfig` under the launcher's `ACCELERATE_DEEPSPEED_ZERO_STAGE == "3"`
  (`loaders/model.py:1152-1189`; it makes `from_pretrained` partition under `zero.init()`, `modeling_utils.py:1440`, never for a bnb-quantised load),
  `set_z3_leaf_modules` (`:1380-1393`) and a kbit-prepare skip (`:1405-1416`). The engine that gathers partitioned parameters per forward/backward and performs the
  CPU parameter/optimizer offload is `deepspeed.initialize`, created by transformers' `Trainer` (`deepspeed_init`, `trainer.py:1701`, then `accelerator.prepare`
  `:1721-1730`) inside `axolotl.train.train`. It cannot be driven through `ModelLoader` alone: TC3's `ckpt_axolotl_m_zero3` is a refused row.
- HF offload: transformers 5.18.0's bnb-4bit quantizer raises on a device_map dict with CPU/disk entries unless `llm_int8_enable_fp32_cpu_offload=True`
  (`quantizers/quantizer_bnb_4bit.py:70-80`); with it the CPU-resident modules are left unquantised (`:133-136`) and `max_memory` is scaled by 0.90 (`:97-100`).
  Whether PEFT `target_parameters` on an accelerate-offloaded bf16 expert stack can train is UNVERIFIED: the arm records the exception text if it cannot.
- axolotl 0.20.0 pins `packaging==26.0` (wheel METADATA line 12) and `torch<=2.14.0,>=2.13.0` (line 11); the `deepspeed` extra is `deepspeed>=0.18.6,<0.20.0` +
  `deepspeed-kernels` (lines 80-81). The PyTorch cu130 index mirrors `packaging` at an older version, so uv's default first-index strategy with that extra index
  makes axolotl unsatisfiable (both TC1 boxes, 2026-10-01); `tc1_run.sh` now installs torch from that index alone and axolotl under `--index-strategy unsafe-best-match`.
