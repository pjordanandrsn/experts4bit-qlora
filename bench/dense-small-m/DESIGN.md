# K1: a pinned cuBLASLt GEMM for the dense projections at 2 to 8 decode tokens (design note, zero rental)

**Status:** the serving roadmap's lever K, step K1. This is a design for review only: no package code and no rental
until it is reviewed. A registration comes before any reading.

## 1. Why

A decode step of 2 to 8 tokens costs much more than a step of one. Part of the extra cost is in the expert GEMMs (lever
B, `bench/p227/`). Part of it is in the dense projections, which run through `torch.nn.functional.linear`, so cuBLAS's
own heuristic picks the kernel at every row count. An unregistered exploration on one RTX 5090 found three things at
Qwen3-30B-A3B's dense shapes. Its numbers are design inputs only and are not quoted here.

- **`o_proj` and the router gate:** cuBLAS's default choice at 2 to 8 rows runs slower than its choice at one row.
- **The fix exists already:** a cuBLASLt algorithm from cuBLASLt's own heuristic list, with split-K, does not slow
  down. It runs these shapes at 2 to 8 rows at about the one-row path's time.
- **`qkv` and the bf16 head show no such kernel cliff.** The head's extra cost above one token is bytes: one token
  serves the int8 head (K36), while more tokens serve bf16. That is K2's, a multi-row int8 GEMV, and not this note's.

So K1 is not a kernel. It is a **pinned cuBLASLt algorithm** for named projections at 2 to 8 rows, chosen and verified
on the card, keyed and guarded so that it never runs anything else silently.

## 2. What changes (opt-in; nothing by default)

- **The knob:** `E4B_DENSE_LT_PIN`, a comma-separated list of projection names from `{o_proj, router}`. Unset or empty
  means off, which is the default. An unknown name is refused.
  - `o_proj` and `router` are separate names so that each can be read, and shipped, on its own.
  - The router changes which experts a token picks, so it is the riskier of the two.
- **The rows it takes:** a call of 2 to 8 rows (`2 <= M <= 8`, bf16, contiguous, CUDA) on a named projection goes
  through the pinned cuBLASLt matmul. Every other call keeps its route unchanged:
  - one row: today's cuBLAS, or K35 under `E4B_DENSE_GEMV_BW`;
  - more than 8 rows: prefill and the wider buckets.
- **The call sites:**
  - **`o_proj`:** an `nn.Linear` swap, like `dense_gemv_bw.apply_dense_gemv_bw`. Exact `nn.Linear` only: an int4 or
    int8 attention module is refused when named.
  - **The router:** it is reached two ways, and the pin must cover both or refuse:
    - the HF module path `self.gate(hidden)`, an `nn.Linear` swap;
    - the router epilogue's own `F.linear(rows, mod.weight, bias)` (`engines/router_epilogue._topk_softmax_logits`,
      under `E4B_FUSE_ROUTER_EPI`), a call routed through the same pinned function.
  - A router with a bias is refused for the pin's first version, because the epilogue adds the bias inside the GEMM
    and the pinned call would have to match that exactly.
- **Reported:** `/health` gains `dense_lt_pin`: the resolved table, each projection's state, and which rows routed or
  passed during the build's captures and warm-ups. That is the engagement record a served read checks, P151's form.

## 3. The pin and its key

A pin is one cuBLASLt algorithm configuration: the algorithm id, tile, stages, split-K, reduction scheme, CTA
swizzling and custom option, read back with `cublasLtMatmulAlgoConfigGetAttribute`.
- **The key:** (the cuBLASLt version actually loaded, the device's compute capability, the projection's (N, K), M).
  M is the bucket's row count, 2 to 8.
- **Where the table lives:** in code, the way `_K25_PLANS` does. It is filled by a registered selection on the reading
  box (section 6), never at serve time.
- **At engine build, with the knob set**, every (projection, M in the captured buckets) must find its key:
  - a missing key, another cuBLASLt version, another cc, or another shape is **refused by name**. The pin never falls
    back silently to a different algorithm, and never to cuBLAS.
  - `cublasLtMatmulAlgoCheck` must accept each pinned configuration for the exact descriptors. Otherwise it is refused.
- **The library:** ctypes against the `libcublasLt` that torch has already loaded, the same instance with no second
  copy and no version skew.
  - On Linux it is opened with `RTLD_NOLOAD`. If it is not already loaded, that is a refusal, not a load.
  - Its `cublasLtGetVersion()` is asserted against the key.
  - No new dependency is added. The exploration used nvmath-python; serving does not.

## 4. The call

`cublasLtMatmul` with:
- `CUBLAS_COMPUTE_32F` and fp32 scale;
- A = the weight `[N, K]` row-major, read as the transposed column-major operand, the way `F.linear` presents it to
  cuBLAS;
- B = the `[M, K]` rows, and D = a bf16 `[M, N]` output;
- the pinned algorithm;
- the current torch stream.
- **The workspace:** ONE persistent device buffer per process, sized at build to the largest `workspaceSize` that
  `cublasLtMatmulAlgoCheck` reports over the pinned configurations. Nothing is allocated in the forward, so a CUDA
  graph capture is legal. The output buffer comes from the caller's allocation, which a capture takes from the graph
  pool, as for every other route.
- **The descriptors** (matmul, layouts) are created once per (projection, M) at build and reused.

## 5. Determinism and numerics

- **The reduction scheme.**
  - `CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE` (value 4) stores split-K partial results in the output type in the
    workspace and reduces them in a separate kernel. There are no atomics, so the order is fixed.
  - `COMPUTE_TYPE` (2) does the same with fp32 partials.
  - `INPLACE` (1) reduces in the output buffer using counters. The pin **excludes INPLACE**, and the selection
    considers only schemes 0, 2 and 4.
- **Asserted on the card:** the pinned call is bitwise equal run to run (two calls, same inputs), and a CUDA-graph
  capture of it replays bitwise equal to eager. Both are GPU tests that run on the registration's box.
- **Numerics:**
  - split-K changes the summation order, and output-type partials add one rounding per split, so the pinned
    projection's output differs from `F.linear`'s within bf16 tolerance;
  - **so the served tokens change at 2 to 8 tokens**, like lever B's;
  - the router can also flip which experts are picked.
  - So a served default needs the same quality bar as B3. The router is read as its own arm, with the expert-pick flip
    rate against the default at the same T reported. If that rate is material, K1 ships `o_proj` alone.
  - The selection may prefer scheme 2 (fp32 partials, one fewer rounding) wherever its time is within noise of
    scheme 4's. That is a registered tie rule, never a choice made after data.

## 6. Selection and verification (the registration, later)

- **On the reading box** (one RTX 5090, 575 W floor, pinned host), for each (projection, M in 2, 4, 8):
  1. `cublasLtMatmulAlgoGetHeuristic` with a requested count of 64;
  2. drop INPLACE reduction and anything `AlgoCheck` rejects;
  3. time each candidate under an L2 flush;
  4. check that it is bitwise run to run;
  5. pick the fastest by the registered rule.
- **The ids from the exploration's 400 W host are candidates only.** The box re-selects them and records the full
  configuration. That configuration becomes the pin table through a code PR, exactly as P222's selected plan became
  `_K25_PLANS["m1"]`. Then comes a served read.
- **The microbench reading's bar** is a registration question. The proposal: the pinned path at M = 2, 4 and 8, against
  the one-row served path and against `F.linear` at the same M, with each shape's byte floor reported.

## 7. Interactions and refusals

- **`E4B_DENSE_GEMV_BW` (K35, one row):** disjoint rows, so the two compose.
- **`E4B_SERVE_ATTN_INT8` / int4 attention:** `o_proj` is no longer an exact `nn.Linear`, so naming it is refused.
- **`E4B_PAGED_FUSE_QKV`:** it touches `qkv` only, which K1 does not name.
- **Under an adapter:** the merged attention weights are bf16 `nn.Linear` weights. The pin reads them and changes no
  adapted weight, so it is classified in the adapter lever census with its reason (a numerics change, no
  re-quantization). The census test requires that classification.
- **A device or cuBLASLt version not in the table:** refused at build when the knob is set.
- **ROCm:** refused, since this is cuBLASLt only.

## 8. Tests (with the code PR)

- **CPU:**
  - the knob's parsing and refusals;
  - the key lookup and its refusals: version, cc, shape, M, a missing bucket;
  - rows 1 and 9 pass through untouched;
  - the router's two call sites both route or both refuse;
  - a bias refused;
  - the adapter census classification.
- **GPU** (on the registration's box, never the A2000):
  - bitwise run to run;
  - graph capture and replay bitwise;
  - within tolerance of `F.linear`;
  - `AlgoCheck` refusal of a deliberately unsupported configuration;
  - the version assertion against the loaded library.

## 9. What this note does not decide

- K2: the head's multi-row int8 GEMV, a separate note after K1.
- Any served default: that needs a served read with the quality bar.
- `qkv`: the exploration showed no kernel cliff there. If a registered reading shows one, it joins by an amendment.
