# AMENDMENT-b1d-capture — the grouping sync, named and treated

Registered after stage A's capture failures and BEFORE the treatment
was built (the prereg's STOP-AND-AMEND clause). Receipts:
`receipts-b1d/` diag logs.

## What stage A found (three failures, then a name)

1. First capture attempt: `cudaErrorStreamCaptureInvalidated` at
   `capture_end` with no Python-level site. The attention timing
   wrapper (`torch.cuda.Event(enable_timing=True)` per call) was
   removed for both arms — necessary (timing records are
   capture-illegal) but not sufficient.
2. Side-stream warmup (the documented cuBLAS-workspace recipe) and
   `empty_cache()` + `thread_local` capture mode — also necessary
   hygiene, also not sufficient.
3. `CUDA_LOG_FILE` + the Python traceback finally named it:
   **`torch.unique_consecutive(..., return_counts=True)` inside
   `_fused_over_stack` raises `operation not permitted when stream is
   capturing`** — the expert-grouping step must SYNC to produce
   host-side group sizes for the grouped-GEMM launch. Data-dependent
   grid shapes are structurally incompatible with graph capture.

## The treatment (engine change beyond the original inventory)

`_fused_over_stack` gains `singleton_groups=False`: when set, the
grouping is skipped entirely — `sizes = [1] * R` (a host CONSTANT,
not data), `eids = local_ids` passed as a DEVICE tensor (the kernel
wrapper already accepts one), no argsort, no unique, no unsort. The
collapsed forward sets it **only at T == 1**, where it is exact by
construction:

- a single token's routed top-k ids are DISTINCT (torch.topk), so
  grouping is a no-op there — dedup buys nothing;
- each row's GEMV is computed independently in both paths (the grouped
  path sorts rows and unsorts after), so per-row arithmetic is
  identical ⇒ bitwise-equal outputs — pinned by a CPU dispatch-algebra
  test (mocked GEMM) and held by the on-box position-aligned hash gate;
- at T > 1 the grouped path is untouched (singleton grouping would
  re-read shared experts' weights; B=16's dedup stays).

The `sizes=[1]*R` decode shape takes the kernel's GEVM path with a
grid that depends only on (R, N) — static across replays.

## Bars unchanged

G1 (position-aligned hashes + tokens, both arms), H-G/H-D, GS — all as
registered. This amendment changes the capture-cleanliness inventory,
never a bar.
