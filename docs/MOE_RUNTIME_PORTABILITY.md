# How much of the Qwen training stack is a general MoE runtime?

*Session `session/moe-generalize`, 2026-10-04. Code is read at e4b 0.44.0 / grouped-nf4-gemm 0.37.0 plus this branch.
Measurements are one RTX A2000 12 GB (sm_86, the owned seat card, shared with house services). A2000 numbers are
**within-box ratios**. They are not positions, and none of them is a 5090 or H100 reading.*

Qwen3-30B-A3B's fused training step got fast through about a dozen individually measured changes (TC1 amendments 10–21,
TC1c amendments 2–6). This page sorts each one by whether another MoE family inherits it, and records what this branch
changed so that more of them carry over.

**The short answer: a new MoE inherits nearly all of the expert-side work automatically.** Every candidate family calls its
experts through the same `experts(hidden, top_k_index, top_k_weights)` contract, and `ExpertsLoRA` wraps all of them by
structure. The gaps were in the *glue around* the experts: norm fusion, rotary fusion, activation retention, attention
discovery and trainable selection. Those were matched on Qwen's names or formulas, and this branch makes them structural.

## A. Structurally universal (inherited by every family whose experts `ExpertsLoRA` wraps)

| optimisation | where | gate | status after this branch |
|---|---|---|---|
| one host read per grouping (`E4B_GROUPING`) | e4b `engines/fast.py` `_group_by_expert` | none | universal (it already was) |
| pinned host/index staging ring (`GNF4_PINNED_RING`) | gnf4 `nf4_grouped._PinnedRing` | CUDA, not capturing, call ≤ one slot | universal; **the overflow is now counted** (`ring.overflow`): a call larger than `GNF4_PINNED_RING_SLOT_INTS` (16384 ints = tokens·k) silently took the syncing copy |
| per-pass upload/plan reuse (`GNF4_HOST_REUSE`) | gnf4 `_UPLOAD_MEMO`, `lora_delta_grouped` | CUDA, not capturing | universal |
| lean padded LoRA delta / compact delta | gnf4 `nf4_qlora` | padded route | universal |
| router-weight gather with scatter backward, bf16 combine save | e4b `_PermGather`, `_ScatterCombine` | none | universal |
| dgrad single-launch backward | gnf4 `dgrad_4bit_grouped` | bf16 grad, eligible shape, resident storage | universal; **the loop fallback is now counted with its reason** (`nf4_qlora.DGRAD_STATS`). It used to be silent, and an offloaded run always takes it |
| fused frozen RMSNorm (`E4B_FUSED_RMSNORM`) | e4b `engines/rmsnorm_train.py` | the norm's own forward must match the kernel's formula | **was Qwen/Llama-formula only**; now probes four formulas: `w·round(n)` (Qwen3, OLMoE, Mixtral, Granite-3.1, Granite-4.0-H, LFM2, ERNIE: unchanged), `w·n` in fp32 (Gemma-4, Nemotron-H: fused with the Llama rounding before, now their own formula), `(1+w)·n` in fp32 (Qwen3.5/3.6, whose norms were all skipped), and `(1+w)·round(n)`; fallbacks counted |
| fused RoPE (`E4B_FUSED_ROPE`) | e4b `engines/rope_train.py` | signature + per-call shape/dtype | **was signature-only: a correctness hazard on ERNIE-4.5**, whose `apply_rotary_pos_emb` has the HF signature but rotates interleaved pairs; only its fp32 cos/sin kept it off the kernel. Now a patch-time semantics probe must reproduce the composite exactly; refusals are listed |
| MoE-activation retention (`E4B_MOE_KEEP_LAYERS`) | e4b `engines/moe_keep.py` | layer must hold an expert stack | **was `self_attn` + `mlp` by name**, which matched zero layers on Granite, LFM2, Jamba and Nemotron-H and a quarter of Qwen3.5's. Now structural: every checkpointed layer holding experts; every other weighted child (attention, Gated DeltaNet, short conv, Mamba, dense MLP) stays checkpointed on its own. On a Qwen3-MoE layer the result is identical to the old form (test-pinned) |
| storage eligibility | e4b `enable_fast_train` | — | **had no storage gate**: an fp4 store at blocksize 64 would have been decoded through the NF4 table. Now a positively known non-NF4 / non-64 / K%64 base is skipped to the reference path and counted (`FAST_TRAIN_STATS["skipped"]`): a partially engaged arm says so |
| attention LoRA / 4-bit attention discovery | e4b `lora._attention_projection_names` | `q_proj` + `o_proj` | **LFM2's attention (`out_proj`) got neither.** `out_proj` is now admitted beside bias-free `q_proj` and `k_proj`; short-conv, Mamba, GDN and biased CLIP-style towers stay excluded |
| trainable / router selection in the CLI trainer | e4b `train.py` | parameter names | **was `"experts"` / `"self_attn"` / `mlp.gate.weight`.** That froze Nemotron-H's attention adapters (under `mixer.`) and found no router on Granite, Gemma-4, Nemotron-H or LFM2. Now by owning module (`ExpertsLoRA`, `LoRALinear`) and router shape `[E, hidden]` beside the experts |

## B. Geometry-dependent (should be measured dispatch, per shape and per card)

| decision | current rule | where it binds |
|---|---|---|
| LoRA delta route: padded / loop / grouped_mm | `auto` pads unless the padded block exceeds 2 GiB | hot-expert skew and few experts (Mixtral E=8, top-2) make `G·max(rows)` large. Note: the byte estimate uses the activations' itemsize while the block is allocated in the adapter dtype, so it undercounts 2× on fp32 adapters. Recorded now (`LORA_PAD_WASTE["last_bytes_alloc"]`); the rule is unchanged pending a full-step reading |
| prefill M-tile | `cost` rule, `D=96` fitted at Qwen3-30B shapes on an A2000 | small expert widths (OLMoE I=1024, Granite I=512, LFM2 I=1792) |
| packed fused kernel vs dequant + `torch._grouped_mm` | `GNF4_TRAIN_GEMM=auto`: grouped_mm on sm_90 only | a hardware rule, measured only on Qwen3-30B; the crossover depends on rows per expert (tokens·k/E) |
| activation retention | opt-in | worth it where the recomputed MoE forward is a large share of the step *and* the memory exists |

## C. Architecture-specific (kept explicit)

* Qwen3.5/3.6 Gated DeltaNet, Nemotron-H Mamba-2, LFM2 short conv, Granite-H Mamba: outside the expert runtime. They are
  bf16, not LoRA'd, and not quantised, deliberately (the recurrent-state quantisation hypothesis is untested). This
  branch does not change that.
* Shared experts (Qwen3.5 `shared_expert`, Nemotron-H `shared_experts`, ERNIE `shared_experts`, Granite `shared_mlp`) run
  every token on the plain composite in bf16, unchanged.
* Gemma-4's parallel dense MLP, `per_expert_scale` and its `(x, cos, sin)` rotary sit on its own path.
* Partial rotary (Qwen3.5) falls back per call to the composite; it is correct, just not fused.
* Router variants (softmax / sigmoid / score-correction bias / routed scaling) all sit outside the experts module, so the
  fused path never sees them.

## Per-family ladders

See [`../bench/moegen/RESULTS-moegen-ladders.md`](../bench/moegen/RESULTS-moegen-ladders.md).

## Licence reading on an RTX 5090

Lane MG1 ([`../bench/moegen/mg1/mg1-5090-2/RESULTS-mg1.md`](../bench/moegen/mg1/mg1-5090-2/RESULTS-mg1.md)): every family passed tp1's parity verdict on this structural stack. The two regression anchors, OLMoE and Gemma-4, re-read clean, and LFM2, Granite-4.0-H, ERNIE-4.5 and Nemotron-H entered `fast_train = supported`. Qwen3.6 passed and waits on P2.
