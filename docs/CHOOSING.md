# Which door? Start from what does not fit

Moved off the README (2026-08-01) so the landing page carries the decision *table* and this
page carries the reasoning. Nothing here is new; the wording is as it was.

Every mode below exists because something ran out: VRAM, host RAM, or disk. Find
your constraint, not your model.

**Nothing fits — I just want to train a fused-MoE at all.**
`load_moe_4bit_streaming(model_id, ...)` and train. The reference `ExpertsNbit`
forward is the default and needs no flag, works on any host and any storage
scheme, and is the convergence-tested path.

**It trains, but each step is slow.**
`enable_fast_train(model)` (needs `[fast]`). Routes the *differentiable* expert path through
the fused grouped kernel — see [Training + expert offload](#training--expert-offload) for the
measured cost. Opt-in on purpose: it changes the expert summation order (group-sorted vs
ascending expert id), an ulp-level difference that should be a deliberate choice in a training
run. **Returns the number of modules patched — check it.** A zero means `grouped-nf4-gemm` is
missing and you are silently on the reference path.

**The experts do not fit in VRAM.**
`load_moe_4bit_streaming(..., offload=True)` or `OFFLOAD_EXPERTS=1`. Frozen
experts live in pinned CPU RAM and stream one layer at a time. This is what makes
a 30B-class MoE trainable on a 24 GB card. Requires gradient checkpointing
(`use_reentrant=False`), which the shipped trainer always enables; the
unsupported non-checkpointed combination fails loudly rather than mis-training.

**The experts do not fit in host RAM either — and I am serving.**
`enable_nvme_residency(...)` — serves the cold expert tail from an NVMe arena
instead of RAM. For models where even the pinned host copy is too large. It binds
over the *frozen* stack and replaces the module's forward, so it refuses an
adapter-wrapped module rather than silently discarding the delta.

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
  saying so, rather than returning uninitialized memory.
* **VRAM is unchanged.** The staged stack keeps its full `[E, ...]` shape so every
  consumer still indexes by global expert id. One layer is device-resident, same
  as ordinary offload — this lifts the *host RAM* ceiling, not the VRAM one.
* **`hot_rows` has a hard floor**: at least the number of unique experts one
  forward routes, which for a training batch of `T` tokens at top-`k` approaches
  `min(T*k, num_experts)` — much larger than decode's `k`. Undersizing raises.

**The DENSE side does not fit.**
`enable_dense_offload(model, "cuda")` keeps the non-expert weights in pinned host
RAM; `DenseDiskSource(path)` serves them from the checkpoint's own safetensors
when host RAM cannot hold them either — 114.4 GB for a K3-class model. **Nothing
is transformed**: the bytes handed to the GPU are the bytes in the checkpoint. The
alternative way to fit a 114 GB dense side on a small card is to quantize it,
which changes the model.

**I am serving, not training, and want it faster.**
`enable_fast(model)` (needs `[fast]`) — 3.65× at bs=1 decode on OLMoE geometry.
Inference only; for training use `enable_fast_train`.

**I am serving and have some spare VRAM to trade.**
`enable_pipelined_residency(model, hot_sets, k_slots=k)` (needs `[fast]`). Keeps K hot experts
per layer resident and streams the cold tail; K=0 is pure streaming, K=all is fully resident,
the middle is the dial. **Pick the hot sets from a routing histogram, not by index.**
⚠️ Requires standalone `Experts4bit`/`ExpertsNbit` modules and raises `NotImplementedError`
on the `ExpertsLoRA` wrapper that `load_moe_4bit_streaming` always produces.

**My GPU is small but my CPU is strong.**
`enable_cold_engine(model, hot_sets, dequant="auto")` computes the cold experts on
the host instead of streaming them. Bit-exact host decode and CPU-complete tests;
**performance-experimental** until the AVX2 kernel lands.

**Deprecated:** `enable_hot_residency` is superseded by
`enable_pipelined_residency` (same capability, K is config). It still ships in
0.8.0 — an earlier note promised removal *in* 0.7 and that was wrong — and warns
at call. It is kept only so the v0 receipts stay reproducible; do not build on it.
