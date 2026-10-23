# K2: the int8 lm_head at 2 to 8 decode rows, a multi-row K36 GEMV (design note, zero rental)

**Status:** the serving roadmap's lever K, step K2, after K1 (`DESIGN.md`, e4b#2019). This is a design for review
only: no package code and no rental until it is reviewed. A registration comes before any reading.

## 1. Why

On `qwen3_moe` the lm_head is served on its per-row int8 grid by default (`E4B_INT8_HEAD_BW`). Two reads license that
default:
- P150 (`bench/p150/RESULTS-p150.md`) ruled the grid GO on quality at both of P125's gates, t1 and k16;
- P153 read it FLIP served, at one request, with quality inside its registered bars
  (`e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19`, `bench/p153/RESULTS-p153.md`). `engines/int8_head_bw.py` rounds the bf16 weight in place to that grid, then:
- **one row:** goes to grouped-nf4-gemm K36's `dense_gemv.gemv_i8_bw`, which reads the **int8 codes** (one byte an
  element) plus a per-row fp32 scale;
- **every other row count:** goes to `F.linear` on the **rounded bf16 weight** (two bytes an element).

At 2 to 8 decode rows, the head therefore reads twice the bytes it reads at one row. On Qwen3-30B-A3B's head
(151936 x 2048) that is about 622 MB a step instead of about 311 MB. The head is a pure stream at these row counts:
cuBLAS's bf16 GEMM there already runs at the byte floor (lever K's exploration, not quoted). So no kernel choice closes
it, and no K1 pin applies. The fix is to keep the int8 bytes at 2 to 8 rows.

## 2. The kernel change (grouped-nf4-gemm, `kernel/dense_gemv.py`)

`_gemv_i8_kernel` today loads, per program and K-step, one `[BLOCK_N, KC]` int8 tile of `q` and one `[KC]` slice of
`x`. It accumulates `acc[n] += sum_k q[n, k] * x[k]` in fp32, then stores `bf16(acc * s)`.

- **The change:** a constexpr `M_ROWS` (1 to 8) and the row stride of `x`.
  - Per K-step the `q` tile is loaded **once**.
  - Then, for each row `m` in `tl.static_range(M_ROWS)`, that row's `x` slice is loaded and `acc_m += sum(q * x_m)`.
  - Each `acc_m` is stored to row `m` of `out`.
- **The arithmetic per row is the M = 1 kernel's, in the same order:** the same tile shape, the same `tl.sum` over the
  same axis, the same scale multiply and the same bf16 store.
- **The contract:** each row of an M-row call is **bitwise** that row's own call through the one-row kernel (K36 at
  M = 1), at the same plan. This is the per-row identity lever B uses (B0's rows at T = 2..8 against T == 1, and B2's
  multi-row GEMV against M = 1). The M = 1 launch is the `M_ROWS = 1` instance of the same kernel, so today's one-row
  results do not move by a bit (a test holds the old kernel's output against it).
- **The contract is REQUIRED AND TESTED, never assumed** (review 1).
  - In the source it holds: K36 never splits K, uses no atomics, runs a 1-D grid over N with a sequential K-loop, and
    each row has its own accumulator.
  - But the bits of `tl.sum` are a compiler property (the layout-driven reduction tree, FMA contraction), and the
    `M_ROWS = 8` instance is a different compiled kernel from `M_ROWS = 1`.
  - So the bitwise test is a GPU test at every M in 1..8, on sm_86 and on sm_120. The interpreter proves the indexing,
    not the bits.
  - At engine build, e4b also runs a bitwise probe on the device at the head's own shape, for every M up to the knob's
    value (an M-row call against M one-row calls on random rows). A failing (M, arch) is refused by name, never
    served.
- **The plan:** the one-row plan (`PLANS_I8`, P148's) at every M. A plan per M could read faster at M = 8, but it would
  break the row identity across M, which is the point (section 4). A per-M plan is a later lane's choice, never this
  one's.
- **Registers:** the `q` tile is converted to fp32 once per K-step (BLOCK_N x KC values across the program's threads,
  as today). The rows add one `[KC]` x slice and one `[BLOCK_N]` accumulator each, at up to 8 rows. A compile-time
  check (registers, spills) is part of the kernel PR's GPU tests, at every M_ROWS on sm_86 and sm_120.
- **The wrapper:** `gemv_i8_bw(x, q, s, out=None)` accepts `x` of `[M, K]` with `1 <= M <= 8` (contiguous rows),
  refuses `M > 8` by name, and exports `GEMV_I8_MAX_ROWS = 8` so a caller can detect it. Every other refusal is
  unchanged.

### Prior art: K37's `dense_i8.linear_i8` (released in grouped-nf4-gemm 0.49.0)

grouped-nf4-gemm already serves int8 weights at 2 to 16 rows: K37's `linear_i8`, on the same `quantize_rows_int8`
grid. e4b uses it for Mixtral's int8 attention projections (`engines/int8_attn.py`). It would keep the head's int8 bytes
at 2 to 8 rows with no kernel change and no release. K2 does not take it as written, for two reasons:
- **No row identity.** Its 2..16-row kernel (`_gemm_i8_small_kernel`, P149's) accumulates through `tl.dot` on a
  16-row MMA tile. That is a different reduction from the one-row GEMV's `tl.sum`, so a row at M rows is not bitwise its
  one-row output, and the head would stay batch-variant (section 4). K37 ships under a tolerance contract for exactly
  this reason.
- **No head plan.** Its plans were read at 4096-wide attention shapes (`p149k-5090-8`). The head's 151936 x 2048 would
  run on `PLAN_SMALL_DEFAULT`, untimed.

So the choice is between two routes:
- **K2 as designed:** the bitwise identity, at the cost of a kernel PR and a release.
- **The head on `linear_i8`:** the bytes now, at a tolerance.

The kernel microbench (section 5, reading 1) reports `linear_i8` at the head's shape as a third arm, so the cost of the
identity is measured, not assumed.

## 3. The e4b route (`engines/int8_head_bw.py`)

- **The knob:** `E4B_INT8_HEAD_ROWS=8`, an opt-in upper bound on the rows the head routes to K36. Unset or `1` is
  today's behaviour, the default. Values 2 to 8 widen the route. Anything else is refused.
  - It needs a grouped-nf4-gemm that exports `GEMV_I8_MAX_ROWS >= value`. Otherwise it is refused at build by name,
    never dropped.
  - **Unset (or `1`) never probes:** it reads no constant and runs no probe, so an older package is untouched.
  - **The route gates on BOTH the knob and the probed constant.** Today's `gemv_i8_bw` raises on M > 1, so a
    multi-row call must never reach a package that lacks the constant.
  - **Safe to merge before the grouped-nf4-gemm release:** CI pins an older grouped-nf4-gemm, so stub-old-package CPU
    tests are what CI runs. One stub without the constant must refuse the knob and leave unset untouched; one stub with
    it must route.
- **The forward:** `Int8HeadLinear.forward` routes `1 <= rows <= value` bf16 CUDA rows (copied contiguous if they are
  not) to `gemv_i8_bw`. More rows keep `F.linear` on the grid.
- **The engagement record:** gains the routed row counts. Under CUDA graphs, buckets up to the value capture the
  kernel, and larger ones capture cuBLAS on the grid.
- **No memory change:** the codes and scales already exist when `E4B_INT8_HEAD_BW` is on. The knob is refused unless
  the head is on its int8 grid with K36 (it widens K36's rows; it switches nothing on by itself).

## 4. Numerics: what changes, and what becomes invariant

- **What changes:** today a 2..8-row head computes `F.linear(x, bf16(q * s))`: each weight rounded to bf16 per element,
  cuBLAS accumulation. With the knob it computes `bf16(s * fp32_sum(q * x))`, K36's one-row function, row by row. These
  are different functions within bf16's output rounding, both on P150's grid. **The served tokens can change at 2 to 8
  requests**, so a default needs a served read with a quality bar, read per bucket like K1 and B.
- **The bar is the one the one-row head met:** K2's served read reuses P153's registered quality bars, unchanged (SANE:
  |bias| <= 0.02, argmax >= 0.95, on P153's windows and gates). It applies them at T = 2, 4 and 8, each bucket on its
  own against the same server with the knob unset. K2 is held to exactly the standard that licensed the T == 1 head,
  never a looser one.
- **What becomes invariant:** every row's head output at 2..8 rows equals that row's output at one row, **bitwise**, by
  the kernel contract.
  - Today the head is batch-variant: one row runs K36, two rows run cuBLAS on the bf16 rounding.
  - So K2 removes the head as a source of bucket-dependent tokens below 9 rows (P149's finding, for this layer).
  - **The served read REGISTERS that identity, not just reports it** (review 1). Hidden states are saved at the head's
    input, and each is fed to the head once in a bucket of T rows (T = 2, 4, 8) and once alone (bucket 1). Every row's
    logits must be bitwise equal; a failure is a contract finding.
  - The hidden states are saved, not regenerated per bucket, because the attention's split plan already varies with the
    bucket (P149). That makes the hidden state itself bucket-dependent, and the identity is the head's to keep, not the
    whole step's.
- **P153's quality bars transfer unchanged, but do not discriminate here** (review 1). The knob against unset is a far
  smaller change than int8 against bf16: both are on the same grid. They stay registered as a floor. The head-logit
  identity above is the test that can fail.

## 5. Readings (registrations, later)

1. **The kernel's microbench** (a grouped-nf4-gemm K-lane, one RTX 5090, 575 W). At the head's shape and M = 2, 4 and 8:
   K36 multi-row against cuBLAS bf16 on the grid (the served path at those rows), against K36 at one row, and against
   K37's `linear_i8` (the released multi-row int8 path, section 2's prior art), with the byte floor reported.
   - The proposed bar: the multi-row call at M = 2 and 4 within 1.25x of K36's one-row time (the bytes are the same)
     and at most 0.70x of cuBLAS's.
   - The contract tests (bitwise per row, the old M = 1 unchanged, registers) are its premise.
2. **The served read**, on its own registration: W2 and W4 ABBA on the default Qwen3 server, the knob against unset,
   the quality bar per bucket as a floor, and the head-logit identity across buckets registered (section 4).

## 6. The release dependency

K2's kernel lives in grouped-nf4-gemm, so it reaches serving in three steps:
1. a grouped-nf4-gemm release carrying the multi-row K36;
2. an e4b PR raising its pin (the `[fast]` floor, or a feature probe that resolves the knob off, recorded, on an older
   package);
3. the e4b release.

Releases go one wave per Chicago day, with Jordan's yes. This note schedules none of them. The kernel's microbench can
read a grouped-nf4-gemm commit before its release. The e4b route can merge before the bump, refusing by name when the
installed package lacks `GEMV_I8_MAX_ROWS`.

## 7. Tests (with the code PRs)

- **grouped-nf4-gemm (GPU, on sm_86 AND sm_120, which the bits require; the interpreter checks the indexing only):**
  - each row of an M-row call is bitwise its one-row call, at every M in 1..8 and on ragged N and K;
  - the M = 1 output equals the pre-change kernel's (a frozen copy in the test);
  - M = 9 is refused;
  - registers and spills are read on sm_86 and sm_120.
- **e4b (CPU, with a stub kernel; GPU on a box):**
  - the knob's parsing and refusals: the value, a kernel package without `GEMV_I8_MAX_ROWS`, the head not on its grid;
  - a stub OLD package (no constant): unset untouched and never probed, the knob refused;
  - a stub NEW package: routed;
  - a failing build probe (a stub whose M-row call differs) refused by name for that M;
  - rows 1..value routed and more passed;
  - the engagement counts;
  - under CUDA graphs, bucket-by-bucket capture equals eager.

## 8. What this note does not decide

- A per-M plan for the head: a later lane, after the identity-preserving plan reads.
- Any served default: the served read with the quality bar comes first.
- Other int8 dense projections (lever C's), which take K2's multi-row interface when they come.
