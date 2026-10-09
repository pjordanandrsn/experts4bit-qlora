# P128 — e4b's host work at TC1's field recipe, and what CUDA graphs or `torch.compile` of the dense parts removes (registered before any run)

Issue: experts4bit-qlora#835 (the training-systems campaign). The lane number was claimed by `prereg/p128` (2026-10-09). The registration
is staged:
- **Phase 1** is engineering, checked on an RTX A2000 by counts only.
- **Phase 2** is the rented A/B. It is registered by an amendment to this page only after Phase 1's gate passes, and no box runs before then.

## The question

At TC1's field recipe (Qwen3-30B-A3B, alpaca, seq 2048, micro-batch 2 × accum 4, one RTX 5090), e4b's GPU is busy for under
half of its step on most hosts. The matched arm's device time per profiled step over its timed s/step:

| host | busy |
|---|---|
| AMD EPYC 7713 (TC1 amendment 69) | 0.486 |
| AMD EPYC 7K62 (amendment 70) | 0.417 |
| AMD EPYC 7763 (amendment 71) | 0.475 |
| GPU-bound host (amendment 72) | 0.884 |

Amendment 70's profile shows where the host time goes, per profiled step:
- about 0.86 s inside the checkpoint functions: the decoder layer's Python forward, run once and again in the recompute;
- about 54–59 k `cudaLaunchKernel` calls;
- about 9,000 `aten::mm` calls.

Host syncs are not the lever: about 47–70 ms a step blocked in copies and synchronizations.

**How much of that host work can CUDA graphs or `torch.compile` of the dense parts remove, and what does removing it do to the step on a
host-bound box?**

## What the premise has shown (RTX A2000, 2026-10-09; correctness and counts only)

The test model is a random two-layer Qwen3-MoE at Qwen3-30B-A3B's per-layer dimensions:
- hidden 2048, 128 experts, top 8, expert width 768;
- 32 query and 4 key/value heads of 128.

It is loaded and set up as TC1's e4b arm:
- `load_moe_4bit_streaming`, NF4 experts;
- NF4 attention projections with fp32 LoRA;
- `enable_fast_train` (the reentrant checkpoint).

It trains on two rows of 233–311 tokens (the field recipe's rows vary per forward); the counts below are read on the 270-token step. e4b and grouped-nf4-gemm are at their main heads, on
torch 2.11.

1. **Eager, per two-layer step:**
   - 16,241 Python calls (cProfile, autograd on the calling thread);
   - 769 kernel launches (`cudaLaunchKernel` 693, `cuLaunchKernelEx` 50, `cuLaunchKernel` 26);
   - 841 device kernels;
   - 6 host syncs (4 of them the MoE grouping's one read per layer pass).

   The kernels are a long tail:
   - copies and dtype casts (61), memsets (60), elementwise kernels (204);
   - bitsandbytes' blockwise dequantize (60), cuBLASLt split-K reductions (36), gathers (35), radix sorts (32), fills (41);
   - the grouped NF4 GEMMs themselves are 8.
2. **`torch.compile(dynamic=True)` of each layer's attention block** traces only with `E4B_TRITON_PREBIND=0`. With it on, Dynamo fails
   inside the prebound Triton launcher. With it off:
   - the step trains, with loss equal bit for bit, and the worst gradient differs by 0.49 % of its largest entry (an expert adapter, downstream);
   - it removes **no** launch (769 → 769) and adds Python calls (16,299 → 16,703).

   e4b already runs the block's elementwise work in its own Triton kernels (`_rms_fwd`, `_rope_fwd`). What remains is matmuls and
   bitsandbytes ops.
3. **Compiling the whole decoder layer, or the MoE block alone,** fails before it runs. grouped-nf4-gemm keys its plan memo on the raw
   stream handle (`torch.Stream.cuda_stream`), which Dynamo cannot read.

So no remedy ready today removes the host work. The launches live in the MoE glue (routing sort, gathers, the LoRA delta's casts and
fills, the combine) and in bitsandbytes' attention path. To reach them, Phase 1 has to make that code traceable, or capturable in
segments.

## Phase 1 — a candidate that removes host work (engineering; no rented box)

The work, in the order the premise found the blockers:
1. **Traceable Triton launches.** e4b's prebound kernels (`_rms_*`, `_rope_*`) and grouped-nf4-gemm's become `torch.library.triton_op`s or
   custom ops, so Dynamo can trace them without falling back to the unbound launcher's per-launch cost.
2. **A traceable plan memo.** grouped-nf4-gemm keys it on a stream identity Dynamo can guard, not `cuda_stream`.
3. **The compile region.** The decoder layer with the MoE grouping's host read left eager as a graph break: that read is the kernel
   contract's floor, one per layer pass. So two compiled segments a layer: up to the router, and from the grouped GEMMs to the residual.
4. **CUDA graphs, if the compiled segments are capturable.** `mode="reduce-overhead"` per segment, with the field recipe's varying row
   lengths bucketed only if bucketing leaves values exact (padding rows masked out of attention and routing). Otherwise graphs stay out
   of this lane.

**Phase 1's gate** is read on the A2000, against eager on the same model, rows and seeds, by counts only:
- **Pass:** the candidate cuts kernel launches per step by at least **30 %** and Python calls by at least **30 %**.
- **Quality:** the loss and every gradient bit for bit, or within the TC1 rounding bar: each tensor within `2**-6` of its largest entry for
  bf16 and `2**-16` for fp32.
- **On a fail:** P128 reads **NO_CANDIDATE**, names the blockers it met, and stops. No box runs.

**Constraints on every Phase 1 change (from the review, 2026-10-09):**
- **Opt-in.** The compile wrapper and any change to the prebound launches are off by default. With the knob unset, training and serving
  run exactly today's code. Tests assert that bit for bit: the same op sequence, values and gradients with the knob unset as on the
  commit before the change.
- **grouped-nf4-gemm first.** Its part (`triton_op` launches, a guardable plan-memo key) lands as its own grouped-nf4-gemm PRs, with
  bitwise-against-current tests, in the release after 0.44.0. e4b's part pins that release.
- **Serving untouched.** The wrapper targets the training path (`enable_fast_train`). If any forward body it wraps is one that the
  serving path's fallback observer fingerprints (`bench/ra/fallback-adapters.json`), its fingerprint is coordinated with that lane before
  the PR is marked ready.

## Phase 2 — the rented A/B (registered by amendment once Phase 1 passes; the frame is fixed now)

- **The box:** TC1's field recipe and tokens over 60 load-gated steps, venv-unsloth (torch 2.12), e4b at its defaults, with
  `NF4_QLORA_SINGLE_LADDER=auto` (the default).
- **The arms:** eager (`c0`) against the Phase 1 candidate (`c1`), on the shipped arm (bf16 adapters) and the matched arm (fp32
  adapters). Two draws a side in ABBA order, every arm profiled. One RTX 5090.
- **The recount gate.** Launch and Python-call counts can depend on the card: grouped-nf4-gemm and e4b dispatch on SM count and capability
  (for example the bandwidth GEMV's 160-SM default). So every Phase 2 box recounts both, with and without the candidate, on its own card,
  over a counted step outside the timed window. The box is **VOID** if Phase 1's reduction does not reproduce there: each count at least
  0.8 of Phase 1's relative cut.
- **The premise gate:** the box is **VOID** for speed unless the matched `c0`'s GPU busy share (device ms per profiled step over the timed
  s/step, medians of its draws) is at most **0.85**. A GPU-bound box hides a host remedy. On such a box the read reports the device-time
  ratio and the cost of any compile.
- **Two ratios, never one:**
  - the wall ratio `c1 / c0` (s/step, medians, two stable draws a side);
  - the device-time ratio `c1 / c0`.
  
  A host remedy is judged by the wall ratio on a host-bound box. The device ratio only bounds what it costs.
- **Quality:** step-0 held-out within **0.0005** per draw pair and held-out at N within **0.005**, on each arm, both stated before the read.
  If Phase 1 reaches bit-for-bit values, the bar becomes `torch.equal` on step-0 held-out.
- **The rule**, the first rung that applies:
  1. **VOID:** a record missing or not ok; a commit, revision or default off; a draw without its profile or the compile census (graphs,
     breaks, recompiles); the recount gate unmet; or the premise gate unmet.
  2. **NOISY:** any side's two draws differ by more than 5 %.
  3. **QUALITY_FAIL:** a held-out bar missed.
  4. **NO_GAIN:** wall `c1 / c0` above 0.97 on either arm.
  5. **GAIN:** both arms at most 0.97. The candidate stays opt-in.
  6. **DEFAULT_ON:** GAIN, plus a second box on another host reading GAIN or (GPU-bound there) a wall ratio at most 1.02, plus the
     compile's one-time cost (the first step's extra seconds) at most 2 % of a 60-step run. Then it becomes e4b's default.
- **Predictions** are set at the amendment from Phase 1's counts, with the model stated here. On a host-bound box the step is the host's
  time, so the wall ratio should be about `1 − (host work removed) / (host time)`. The host work removed is estimated from Phase 1's
  launch and Python-call cuts, priced at amendment 70's per-launch and per-op CPU self times.

## Budget

- **Phase 1:** no rental (the A2000).
- **Phase 2:** one RTX 5090 box at the policy rate, about $1.5–2 with the download, and at most one replication box for DEFAULT_ON.
- The lane stays under **$15**, so it needs no ask. Anything above that is asked first.

## Phase 1 read (2026-10-09): NO_CANDIDATE for compile and CUDA graphs; the lane closes with no box

On the RTX A2000, with the same model, rows and setup as the premise, every route to a compiled or captured MoE path stops on the grouped
kernels' contract. Each knob below clears one blocker and exposes the next:

| step | blocker |
|---|---|
| `torch.compile` of the MoE block, defaults | grouped-nf4-gemm's plan memo keys on `torch.Stream.cuda_stream`, which Dynamo cannot read |
| + `E4B_TRITON_PREBIND=0` | e4b's prebound Triton launcher (symbolic grid) |
| + `GNF4_TRITON_PREBIND=0`, `GNF4_HOST_REUSE=0`, `GNF4_PINNED_RING=0` | grouped-nf4-gemm's raw grouped-GEMM launch. Its grid is sized from the routing counts: data-dependent host integers from the MoE grouping's one read per layer pass |

That last blocker is the kernel contract (host-sized grids), not a wrapper:
- **An opaque custom op around the MoE path** would let the layer compile, but every MoE-glue launch inside it stays eager.
- **The compile-reachable rest** (attention, norms) already runs e4b's fused kernels. Compiling it removed **0** of 769 launches.
- **CUDA graphs** fail on the same host-sized grids.

Reaching the glue needs a sync-free MoE path (device offsets, upper-bound grids, no host rows), a kernel-contract change outside this
lane. With all four knobs off, eager runs 901 launches and 998 device kernels per two-layer step (default: 769 and 841).

**Where the launches are** (one training step of the two-layer model at the defaults, 799 launches). Forward and recompute launches are
attributed by profiler ranges on the layer's modules and on e4b's and grouped-nf4-gemm's MoE entry points. Backward launches are
attributed by the autograd node they run under.

| component | forward | recompute | backward | total |
|---|---|---|---|---|
| attention (NF4 projections with fp32 LoRA, norms, rope, flash) | 90 | 90 | 50 | 230 |
| expert LoRA delta | 60 | 46 | 40 | 146 |
| MoE glue (sort, index, gathers) | 44 | 44 | 0 | 88 |
| dense `mm` backward (attention LoRA, router, head) | | | 55 | 55 |
| combine | 14 | 14 | 20 | 48 |
| dtype casts (backward) | | | 39 | 39 |
| router | 16 | 16 | 0 | 32 |
| grouped expert GEMMs | 4 | 4 | 16 | 24 |
| norms, epilogue, grouping, optimizer, elementwise, loss, other | | | | 137 |

The serving path already runs some of this sync-free on the device: the router epilogue, the tile tables, the MoE glue, the combine, the
grouped GEMMs and the norms. That share is about 185 launches (23 %), forward and recompute only, and its backward has no serving
counterpart. The rest, about 70 %, is training-only:
- **The attention projections' LoRA on NF4.** Per layer pass: 16 `mm`, 8 casts, 8 bitsandbytes dequantizes and 8 adds; about 27 of the
  45 are q, k and v.
- **The expert LoRA delta.** About 30 per pass: gathers, casts, `bmm`, plan building, fill, `index_copy`.

A launch cut, if registered, belongs to its own lane with the same counts-only A2000 gate.
