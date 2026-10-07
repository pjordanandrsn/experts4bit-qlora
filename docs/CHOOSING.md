# Which door? Start from what does not fit

The README's 'Which door? Start from what does not fit' table carries the decision; this
page carries the reasoning and the caveats.

Every mode below exists because something ran out: VRAM, host RAM, or disk. Find
your constraint, not your model.

**Nothing fits — I just want to train a fused-MoE at all.**
`load_moe_4bit_streaming(model_id, ...)` and train. The reference `ExpertsNbit`
forward is the default and needs no flag, works on any host and any storage
scheme, and is the convergence-tested path.

**It trains, but each step is slow.**
`enable_fast_train(model, dgrad=True)` (needs `[fast]`). Routes the *differentiable* expert path
through the fused grouped kernel; `dgrad=True` additionally routes the backward through
`grouped-nf4-gemm`'s single-launch dgrad kernel, and without it the backward stays on the
per-expert decode loop. The `dgrad=True` form is the one the README's door and results tables,
`docs/STATUS.md` and `docs/capabilities.json` quote and the one the tp1 training-parity
receipts measured — see `e4b.train.flagship-matrix` in [claims.json](claims.json) and the tp1
rows in [ARCHITECTURE_SUPPORT.md](ARCHITECTURE_SUPPORT.md#training-on-real-weights-tp1-2026-09-05)
for the measured cost. Both are opt-in on purpose: the fused forward changes the expert
summation order (group-sorted vs ascending expert id), an ulp-level difference that should be
a deliberate choice in a training run, and `dgrad` is a second numerics change (the backward
accumulates fp32 in a different order; at real width it adds no composed gradient error over
the lane it extends, `e4b.train.fast-train-dgrad`). **Returns the number of modules patched —
check it.** A zero means `grouped-nf4-gemm` is missing and you are silently on the reference
path; `dgrad=True` on a kernel cut too old for it is turned off with a `RuntimeWarning`, not
an error.

**It trains with `enable_fast_train`, there is spare VRAM, and the step should be faster.**
Set `E4B_MOE_KEEP_LAYERS=n` (or `all`) and grouped-nf4-gemm's `NF4_QLORA_COMPACT_DELTA=1` before `enable_fast_train`.
Hugging Face gradient checkpointing recomputes each decoder layer whole, so every MoE forward runs twice. In the last n checkpointed
layers this checkpoints attention only and keeps the MoE activations: the same gradients (`torch.equal`), less device and host work,
more memory. On one RTX 5090 at TC1's field recipe on Qwen3-30B-A3B (`e4b.train.moe-keep.qwen3.5090.2026-10-04`):

- 32 of 48 layers kept on the shipped arm stepped at **0.835** of the default, at +4.5 GB of peak (24.6 → 29.1 GB);
- 16 kept on the matched arm stepped at **0.926**, at +2.3 GB (27.1 → 29.4 GB).

That is about 141 MB of peak per kept layer at that recipe's largest micro-batch. Size n to the headroom you have, and assert
`enable_fast_train`'s count: the setting rides it, and `disable_fast_train` unwinds it. Without the compact delta a kept layer saves
its padded LoRA blocks too, up to 4x the memory. With gradient checkpointing off, there is nothing to keep, and it changes nothing.

**It trains, but long rows run out of memory in the loss.**
On the current code they should not: `enable_fast_train` and the CLI trainer compute the loss over token chunks wherever a training
forward's fp32 logits would reach 1 GiB (`E4B_CHUNKED_LM_LOSS=auto`, the default since TC1 amendment 44). Set `E4B_CHUNKED_LM_LOSS=1`
(or a chunk size in tokens) to chunk every training forward, `0` for the stock loss everywhere, or call
`experts4bit_qlora.engines.chunked_lm_loss.enable_chunked_lm_loss(model)` directly. Hugging Face's causal-LM loss materialises the
`[tokens, vocab]` logits, upcasts them to fp32 and keeps the fp32 log-probabilities for backward. At Qwen3's 151,936-token vocabulary
and 4,096 tokens that is 8.1 GiB for the head and loss alone. This computes the same loss over chunks, recomputing each chunk's logits
in backward: 0.9 GiB at 512-token chunks, for one extra head matmul and cross-entropy forward per chunk in backward (the chunked LM loss entries in CHANGELOG). The same loss
to fp32 rounding. The same gradients, up to cuBLAS's shape-dependent bf16 reduction (`torch.equal` with
`allow_bf16_reduced_precision_reduction` off). It covers the dense Qwen3, Qwen3-MoE, Qwen3.5/3.6-MoE, Mixtral, OLMoE, gpt-oss,
ERNIE-4.5-MoE, Granite-MoE, -MoE-Shared and -Hybrid, LFM2-MoE and Nemotron-H causal LMs. Anything else keeps the stock loss: silently under the default, with a
warning when the variable is set. Only training forwards with
labels take it: evaluation under `torch.no_grad` and generation stay stock, and `.logits` is `None` on the forwards that do.
Evaluation can then be the run's peak. A `torch.no_grad` forward with labels on one 4,096-token Qwen3 row holds the bf16 logits, an
fp32 copy and its log-softmax: 4.64 GiB above the logits on an RTX A2000. That made the held-out evaluation e4b's run peak on TC1's
packed rows (amendment 58). So by default (`E4B_CHUNKED_EVAL_LOSS`, since TC1 amendment 60; `0` turns it off) such a forward, where
its fp32 logits would reach 1 GiB, runs without labels, returns the stock logits unchanged, and takes the loss from them in 512-token
fp32 chunks: 0.58 GiB above the logits on the A2000, and on an RTX 5090 an evaluation-phase peak of 22.50 GB instead of 26.88, below the
training phase (`e4b.train.chunked-eval-loss.packed-4k.5090.2026-10-07`; Qwen3-30B-A3B, torch 2.12). Only the fp32 summation order of
the loss can differ, so a held-out loss above the gate compared across the default is not byte-identical (amendment 60: identical at
step 0, within 0.00005 at the end). Shorter rows are untouched.
Why the gate: on short rows chunking is a cost, not a saving. At TC1's field recipe (Alpaca rows, 0.3–0.6 GiB of logits per
micro-batch) `1` cost a host-bound RTX 5090 4.9 % of the shipped arm's step (TC1 amendment 41), while `auto` never fired there and
stepped 0.992–0.999 of the stock loss (amendment 44). A packed 4,096-token Qwen3 row is 2.32 GiB and chunks, and trains resident on a
5090 where the stock loss ran out of memory (amendments 39 and 43).
Scope: that evidence is Qwen3-30B-A3B. The gate counts bytes, so on a large vocabulary it fires at ordinary micro-batches: about
1,767 positions per forward at Qwen3's 151,936, about 1,335 at gpt-oss's 201,088, about 1,081 at Qwen3.5-MoE's 248,320. There
chunking's step cost was not measured (amendment 41's 4.9 % is Qwen3 at 0.3–0.6 GiB). `E4B_CHUNKED_LM_LOSS=0` restores the stock loss.

**It trains, but each step is slow — and `[fast]` will not build.**
`enable_batched_train(model)` (no extra: stock torch + bitsandbytes). The kernel-free lane:
one whole-stack dequant in place of the per-expert loop. It is the fallback for an arch
`grouped-nf4-gemm` will not build on, not a faster `enable_fast_train`: at real width it barely
beats the loop and costs the most peak memory of any lane (a whole decoded stack rather than
one decoded expert), so default to `enable_fast_train(model, dgrad=True)` wherever `[fast]`
builds. **The count says which modules are patched, not which calls ran batched**: a call
whose padding waste exceeds `_PAD_WASTE_LIMIT` falls back to the reference loop with the count
still positive, so read a batched training result only with `batched_fallback_stats(model)`
(assert `fallback_calls == 0`); tp1's OLMoE row is VOID on exactly this
(`e4b.train.parity.tp1.olmoe.batched.2026-09-05`; [SOLUTIONS.md](SOLUTIONS.md),
"Limitations that apply to every page").

**The experts do not fit in VRAM.**
`load_moe_4bit_streaming(..., offload=True)` or `OFFLOAD_EXPERTS=1`. Frozen
experts live in pinned CPU RAM and stream one layer at a time. This is what makes
a 30B-class MoE QLoRA-trainable on a 12 GB card (`e4b.offload.fits-30b-class`:
Qwen3-30B-A3B peaks at 7.16 GB and Gemma-4-26B-A4B at 8.47 GB, both of which OOM
without offload). Requires gradient checkpointing (either kind), which the
shipped trainer always enables; without it, training fails loudly rather than
mis-training.

**The experts do not fit in host RAM either — and I am serving.**
`enable_nvme_residency(...)` — serves the cold expert tail from an NVMe arena
instead of RAM. For models where even the pinned host copy is too large. It binds
over the *frozen* stack and replaces the module's forward, so it refuses an
adapter-wrapped module rather than silently discarding the delta.

**…and the experts are native MXFP4 (gpt-oss, DeepSeek-V4).**
`enable_mxfp4_nvme_residency(...)` (needs `[fast]` + an arena). The difference from
`enable_nvme_residency` is provenance, not just format: an NF4 arena is baked by
re-quantising, while `grouped-nf4-gemm`'s `nvme_arena.bake_expert_tensors` relocates the
released blocks and scales verbatim, so this path computes on the checkpoint's own expert
bytes. Same seam as the NF4 lane — frozen experts only; an `ExpertsLoRA`-wrapped module is
refused. A bias-carrying module (gpt-oss) is refused on its structure rather than served
through the wrong epilogue: serve that family from an NF4 arena with `enable_nvme_residency`,
or from VRAM through `enable_serve_experts_int4`. DeepSeek-V4 after a relocation bake is the
experimental route. Which model takes which route, and what is refused, is
[solutions/mxfp4-moe-training-and-residency.md](solutions/mxfp4-moe-training-and-residency.md).

**The experts do not fit in host RAM either — and I am TRAINING.**
`enable_nvme_train_residency(model, arena_path, hot_rows=...)`. Same arena, other
side of the seam: the adapter's forward is left exactly alone and only the frozen
base's *home* moves, from pinned host RAM to the arena. A stage reads just the
routed rows off the device, so the host-RAM floor becomes `hot_rows × row_stride`
rather than the whole expert set.

Three things to know before using it:

* **Gradient checkpointing is required**, and this is enforced. The evict hook
  fires when a forward returns, so the checkpoint recompute is what re-stages a
  layer for its own backward. Without it the read is refused with a message
  saying so, rather than returning uninitialised memory.
* **VRAM is unchanged.** The staged stack keeps its full `[E, ...]` shape so every
  consumer still indexes by global expert id. One layer is device-resident, same
  as ordinary offload — this lifts the *host RAM* ceiling, not the VRAM one.
* **`hot_rows` has a hard floor**: at least the number of unique experts one
  forward routes, which for a training batch of `T` tokens at top-`k` approaches
  `min(T*k, num_experts)` — much larger than decode's `k`. Undersizing raises.

**The DENSE side does not fit.**
`enable_dense_offload(model, "cuda")` keeps the non-expert weights in pinned host
RAM; `DenseDiskSource(path)` serves them from the checkpoint's own safetensors
when host RAM cannot hold them either — 114.4 GB for a K3-class model (Kimi K3's
non-expert tensors summed across its 96 shard headers, 2026-07-30: a checkpoint size,
not a register entry; the breakdown is in the `engines/dense_offload.py` docstring).
**Nothing is transformed**: the bytes handed to the GPU are the bytes in the
checkpoint. The alternative way to fit a 114 GB dense side on a small card is to
quantise it, which changes the model.

Freeze first (`model.requires_grad_(False)`, then add any adapters). A trainable matrix beside frozen ones (a
LoRA adapter) stays resident, since an optimizer cannot step a streamed parameter. An unfrozen model streams as
before, but warns.

Training on CUDA streams with `train_prefetch` by default: each layer's frozen weights are copied on a side stream
while the previous layer computes, forward and backward (checkpoint recompute included, reentrant or not), with at most
two layers resident. `train_prefetch=False` falls back to the synchronous single-slot path. Measured on an RTX 5090,
Qwen3-32B architecture, 2048 tokens, LoRA r16 (bitwise identical to resident in every case):

| link | streamed / resident step time | synchronous / resident |
|---|---|---|
| PCIe gen 5 x16 (DQ3) | 1.0023 | 1.18 |
| PCIe gen 4 x16 (DQ5) | 1.0050 | 1.51 |

Streaming also raises the longest sequence that trains. Same subject, chunked LM loss, micro-batch 1:

| card | resident | streamed | ratio | with `expandable_segments` |
|---|---|---|---|---|
| RTX 5090, 32 GB (DQ4) | 7168 tokens | 14336 tokens | 2.00× | 2.375× (8192 → 19456) |
| RTX 4090, 24 GB (DQ6) | 2048 tokens | 9728 tokens | **4.75×** | 5.20× (2560 → 13312) |

The factor is larger on the smaller card: the resident weights (14.5 GiB here) take most of a 24 GB card, while
the streamed model keeps only two layers of them. Two cards are measured; this is not a fitted law.
**Allocator fragmentation is the next limit:** under the default CUDA allocator the streamed run's longest fitting
sequence left 5.69 GiB (4090) and 6.22 GiB (5090) reserved but unallocated.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set before CUDA initialises) recovers most of it. It is a
process-wide setting, so the library leaves it to you. Not measured: gen 3 or x8 links, other cards, real weights,
micro-batches above 1.

**I am serving, not training, and want it faster.**
`enable_fast(model)` (needs `[fast]`) routes the frozen experts through the grouped kernel
(eval, `no_grad`) instead of the per-expert loop. Inference only; for training use
`enable_fast_train`. **Assert the count**: `0` means no eligible module, not a missing
kernel — a missing `grouped-nf4-gemm` surfaces as `ImportError` at the first forward that
reaches the fused path, so call `fast_available()` first. The register has no row for this
call's own decode speedup (the multiplier this page used to quote was the 0.5.0 perf smoke,
#25, and never entered it). The serving position is the census (`e4b.serve.census.bo7.*`),
whose arms reach the kernel through the paged runner and the residency engines — see
[serve-large-moe-on-a-consumer-gpu](solutions/serve-large-moe-on-a-consumer-gpu.md).

**I am serving and have some spare VRAM to trade.**
`enable_pipelined_residency(model, hot_sets, k_slots=k)` (needs `[fast]`). Keeps K hot experts
per layer resident and streams the cold tail; K=0 is pure streaming, K=all is fully resident,
the middle is the dial. **Pick the hot sets from a routing histogram, not by index.**
An `ExpertsLoRA` wrapper (what `load_moe_4bit_streaming` always produces) is a
valid target: the engine patches its base, and the wrapper delegates to it in
eval mode under `no_grad` when the adapter provably contributes nothing (the
serve shim relies on this). Outside those conditions the patch installs and
never runs — assert the returned count and check the served path. (Earlier
notes said this raised `NotImplementedError`; the code no longer does.)

**My GPU is small but my CPU is strong.**
`enable_cold_engine(model, hot_sets, dequant="auto")` computes the cold experts on
the host instead of streaming them. Bit-exact host decode and CPU-complete tests;
**performance-experimental** until the AVX2 kernel lands.

**Deprecated:** `enable_hot_residency` is superseded by
`enable_pipelined_residency` (same capability, K is config). It still ships and
warns at call; the removal *in* 0.7 that an earlier note (and the warning text
itself) promised did not happen. It is kept only so the v0 receipts stay
reproducible; do not build on it.
