# DQ3 design: overlapped training prefetch in `engines/dense_offload.py` (opt-in)

This note covers what the prototype changes, what it must not change, and the open design points the stage-2 tests
settle. It reads `experts4bit_qlora/engines/dense_offload.py` at `origin/main` `f9800eb0`.

## Today's grad-mode path (correct, no overlap)

| hook | grad enabled (training, or checkpoint recompute) |
|---|---|
| forward pre-hook | `stage()`: evict every other staged handle on this device, then a **synchronous** copy of this layer's homes into fresh device blocks, then bind |
| forward hook | `evict()`: point the slots back at the 0-element placeholder |
| full-backward pre-hook | `stage()` again, synchronous |

The docstring says it plainly: a training step under it is "correct but saves nothing". Every frozen byte crosses PCIe
twice a step, each time on the critical path.

Under HF non-reentrant checkpointing, backward recomputes each layer's forward in reverse order. So the forward
pre-hook fires again during backward, and the layer is staged, synchronously, at the start of its own backward.

## The prototype (`enable_dense_offload(..., train_prefetch=True)`; default `False`)

- **Forward phase.** When layer i's pre-hook stages i (consuming i's ready event if a prefetch is in flight), it starts
  the copy of layer **i+1** on the device's prefetch stream and records a ready event on it. That gives two residents at
  most.
- **Backward phase.** When layer i is staged for its recompute, it starts the copy of layer **i−1**. The backward walks
  the chain in reverse.
- **Phase detection** is the open design point. There are two candidates, and the schedule tests choose:
  - (a) An autograd-engine signal that a recompute is running. This depends on internal torch APIs.
  - (b) The order of pre-hook calls: a pre-hook for layer i that arrives right after layer i+1 means the walk is going
    backward.

  (b) is version-proof but must handle the turnaround: the last layer is used by the forward and immediately by the
  backward recompute, so it should simply stay resident.
- **Eviction.** The forward hook still evicts layer i after its forward (a checkpointed forward saves no weights). Under
  backward, eviction happens after the layer's backward completes (a full-backward hook), not after the recompute's
  forward. The recompute's forward hook must not evict a layer whose backward has not run. This is the case
  `test_grad_enabled_forward_stays_resident_for_backward` already exercises in its own form.

## The fence (write-after-read)

Two device-side designs are possible:

1. **Fresh blocks per copy (today's allocation pattern).** `torch.empty_like` on the prefetch stream, then
   `record_stream(compute)` at bind. The caching allocator will not hand a block to a new allocation while a stream
   that recorded it still has pending work. That is the fence, and it already exists at `_bind`.
2. **A preallocated two-slot ring.** This avoids allocator churn and fragmentation over 64 layers × 2 phases. It needs
   an explicit fence: before the copy into slot k, the prefetch stream waits on an event recorded on the compute
   stream after the last kernel that read slot k. This is grouped-nf4-gemm #467's shape for the pinned ColdTier, and the
   mutation-armed race test mirrors that one.

The prototype starts with (1), the smallest change, with its fence pinned by the race test. (2) is adopted only if the
DQ3 read shows allocator churn costs step time. Either way, the race test's mutant must fail.

## What must not change

- **With the opt-in off,** behaviour is byte-identical to `origin/main`: the same hooks, no side stream, no fence event,
  every existing test passing unchanged.
- **Inference prefetch** (`stage_for_inference`) is untouched.
- **Multi-device chains stay per device.** A cross-device link is refused, as today.
- **`quant_state` and small tensors stay resident.** Only Parameters and buffers of ≥ `min_bytes` 2-D tensors stream, as
  today.

## Counters (for the receipt, and the coverage gate)

Per phase:
- prefetches issued;
- prefetches overlapped (the ready event had already completed when the consumer reached `_consume_ready_event`,
  checked with `event.query()`);
- blocking fetches (a synchronous `stage` copy, or a consumer that had to wait);
- residency high-water.

These are exposed through `dense_offload_report`.

## Host memory

Homes are pinned per tensor, and torch's pinned allocator rounds each to the next power of two
(finding_torch_pinned_alloc_rounds_to_pow2). For Qwen3-32B's seven projections that is 1.135× the packed bytes:
276.8 MB pinned per 243.8 MB layer, ~17.7 GB across 64 layers. The receipt reads the host allocator's reserved bytes and
does not compute them.
