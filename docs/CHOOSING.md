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
without offload). Requires gradient checkpointing
(`use_reentrant=False`), which the shipped trainer always enables; the
unsupported non-checkpointed combination fails loudly rather than mis-training.

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
