# K2: the int8 lm_head at 2 to 8 decode rows (design note, zero rental)

**Status:** the serving roadmap's lever K, step K2, after K1 (`DESIGN.md`, e4b#2019). This is a design for review
only: no code and no rental until it is reviewed. A registration comes before any reading.

**The ruling (after review):** K2b goes first. It routes the head's 2 to 8 rows to grouped-nf4-gemm's released K37
`dense_i8.linear_i8`. K2a, a new multi-row K36 kernel with a bitwise row identity, is DEFERRED. Its full design and its
review-1 conditions are kept in section 8, so nothing is lost.

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

## 2. Why K2b first

- **What K2 must deliver is the bytes:** int8 at T >= 2, about 311 MB a step instead of 622 MB.
- **grouped-nf4-gemm already ships an int8 path for 2 to 16 rows:** K37's `dense_i8.linear_i8` (0.49.0), on the same
  `quantize_rows_int8` grid the head uses. e4b already serves it, for Mixtral's int8 attention projections
  (`engines/int8_attn.py`, P149). So K2b needs no kernel change and no release wave.
- **The bitwise bucket identity K2a would add buys no batch invariance at the system level.** K1 (o_proj and the
  router, split-K per M) and lever B (K33 against K25) already make the decode step batch-variant by design. An
  invariant head alone does not make the tokens bucket-independent.
- **K2a comes back only if K2b misses at this shape** (section 5, the fallback rule), or if a later lane needs head
  batch-invariance AND the rest of the step is invariant too.

## 3. K2b: the e4b route (`engines/int8_head_bw.py`)

- **The knob:** `E4B_INT8_HEAD_ROWS`, an opt-in upper bound on the rows the head routes to an int8 kernel.
  - Unset or `1` is today's behaviour, and the default. Values 2 to 8 widen the route. Anything else is refused.
  - It is refused unless `E4B_INT8_HEAD_BW` has put the head on its int8 grid. It widens that route and switches
    nothing on by itself.
- **The kernels:**
  - one row keeps K36's `dense_gemv.gemv_i8_bw`, unchanged;
  - 2 to `value` rows go to K37's `dense_i8.linear_i8(x, q, s)`, over the codes and scales the head already holds;
  - more rows keep `F.linear` on the rounded grid.
- **The gates:**
  - **Unset (or `1`) never imports `dense_i8`**, so an older package is untouched.
  - **With the knob, a grouped-nf4-gemm without `dense_i8.linear_i8` (older than 0.49.0) is refused at build by
    name**, never dropped. e4b's floor is `>=0.43.0`, so that package can be installed.
- **The plan:** `linear_i8` picks its plan from K37's `PLANS_SMALL`, whose shapes were read at Mixtral's 4096-wide
  attention projections (`p149k-5090-8`). The head's 151936 x 2048 runs on `PLAN_SMALL_DEFAULT`, untimed until
  section 5's reads.
  - If another plan reads better at the head's shape, adding `(151936, 2048)` to `PLANS_SMALL` is a grouped-nf4-gemm
    PR and a release, with no kernel change. The e4b route then needs that version for the knob, refused by name below
    it.
- **The forward:** `Int8HeadLinear.forward` routes `2 <= rows <= value` bf16 CUDA rows to `linear_i8`. Rows that are
  not contiguous are copied (at most 8 rows of K elements), never passed to `F.linear`.
- **The engagement record:** `routed` and `passed` per row count, as today, plus which kernel each routed row count
  took. Under CUDA graphs, bucket 1 captures K36, buckets 2 to `value` capture `linear_i8`, and larger buckets capture
  cuBLAS on the grid.
- **No memory change:** the codes and scales already exist when `E4B_INT8_HEAD_BW` is on.

## 4. K2b numerics: a tolerance, not an identity

- **What changes:** today a 2..8-row head computes `F.linear(x, bf16(q * s))`, cuBLAS on the rounded grid. With the
  knob it computes `bf16(s * acc)`, where `acc` is `x . q` accumulated in fp32 through `tl.dot`.
  - Both are on P150's grid, and the codes are exact in bf16 (|q| <= 127).
  - They differ in accumulation order and in where the scale is applied, within bf16's output rounding.
  - **The served tokens can change at 2 to 8 requests.** So a default needs a served read with a quality bar, read
    per bucket like K1 and B.
- **The bar is the one the one-row head met:** P153's registered quality bars, unchanged (SANE: |bias| <= 0.02,
  argmax >= 0.95, on P153's windows and gates).
  - They apply at T = 2, 4 and 8, each bucket on its own, against the same server with the knob unset.
  - K2b is held to exactly the standard that licensed the T == 1 head, never a looser one.
- **What is NOT claimed:** a bucket identity. A row at 2..8 rows is not bitwise its one-row (K36) output, and K37 ships
  under a tolerance contract for exactly that reason.
- **A property to report, not a contract:** `_gemm_i8_small_kernel` always launches with `BLOCK_M = 16` and masks the
  rows past M. So M = 2..16 is one compiled kernel, and a row's output is expected to be the same at every bucket from
  2 to 16. The reads report it (saved hidden states, bucket T against bucket 2); nothing gates on it.

## 5. Readings

1. **The EXPLORE microbench, before any registration** (a cheap box, explore tier, never quoted). It informs the bar;
   it is not a reading.
   - At the head's shape, 151936 x 2048, M = 1..8, CUDA-graph timed.
   - The arms: the bf16 head (`F.linear`); `linear_i8` at `PLAN_SMALL_DEFAULT` and at every plan K37's table offers;
     K36 at M = 1; and the int8 byte floor.
2. **The registered microbench** (one RTX 5090, 575 W): `linear_i8` against `F.linear` on the grid (the served path
   at those rows) and against K36 at one row, at M = 2, 4 and 8, with the byte floor reported.
   - **The proposed bar:** `linear_i8` at M = 2 and 4 within 1.25x of K36's one-row time (the bytes are the same),
     and at most 0.70x of `F.linear`'s.
   - **The fallback rule, fixed before data:** if `linear_i8` misses the bar at the head's shape at every plan K37
     offers, K2b is NO_GO, and K2a (section 8) becomes the candidate. That is a decision of its own, never a silent
     switch.
3. **The served read**, on its own registration: W2 and W4 ABBA on the default Qwen3 server, the knob against unset,
   and P153's quality bars per bucket.

## 6. Tests (with the code PR)

- **e4b (CPU, with stub kernels; GPU on a box):**
  - the knob's parsing and refusals: the value, the head not on its grid, and a package without `dense_i8.linear_i8`
    (refused by name);
  - unset never imports `dense_i8` (a stub whose import raises);
  - one row routed to K36, 2..value rows to `linear_i8`, and more passed to `F.linear`;
  - the engagement counts, by kernel;
  - on the GPU: `linear_i8`'s rows within K37's tolerance contract of `F.linear` on the grid, and, under CUDA graphs,
    bucket-by-bucket capture equal to eager.

## 7. What this note does not decide

- Any served default: the served read with the quality bar comes first.
- A head plan in K37's table: only after a read at the head's shape.
- K2a: section 8 keeps it, deferred.

## 8. Deferred: K2a, a multi-row K36 with a bitwise row identity

This was the note's first design. It was reviewed (review 1: CLEAR with three conditions, all written in below), then
deferred by the ruling in section 2. It is kept so nothing is lost. It returns only under section 5's fallback rule,
or for a lane that needs head batch-invariance with the rest of the step invariant too.

One correction for whoever picks it up: K36's `_gemv_i8_kernel` source is pinned by sha256 as P148's prototype
(`kernel/test_dense_gemv_i8_interp.py`). So K2a would be a sibling kernel for 2..8 rows, leaving the one-row kernel
byte-for-byte, and the identity would be tested against that pinned kernel. It would not be an `M_ROWS` parameter on
the existing kernel.

### 8.1 The kernel change (grouped-nf4-gemm, `kernel/dense_gemv.py`)

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
  break the row identity across M, which is the point (8.3). A per-M plan is a later lane's choice, never this
  one's.
- **Registers:** the `q` tile is converted to fp32 once per K-step (BLOCK_N x KC values across the program's threads,
  as today). The rows add one `[KC]` x slice and one `[BLOCK_N]` accumulator each, at up to 8 rows. A compile-time
  check (registers, spills) is part of the kernel PR's GPU tests, at every M_ROWS on sm_86 and sm_120.
- **The wrapper:** `gemv_i8_bw(x, q, s, out=None)` accepts `x` of `[M, K]` with `1 <= M <= 8` (contiguous rows),
  refuses `M > 8` by name, and exports `GEMV_I8_MAX_ROWS = 8` so a caller can detect it. Every other refusal is
  unchanged.

### 8.2 The e4b route (`engines/int8_head_bw.py`)

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

### 8.3 Numerics: what changes, and what becomes invariant

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

### 8.4 The release dependency

K2's kernel lives in grouped-nf4-gemm, so it reaches serving in three steps:
1. a grouped-nf4-gemm release carrying the multi-row K36;
2. an e4b PR raising its pin (the `[fast]` floor, or a feature probe that resolves the knob off, recorded, on an older
   package);
3. the e4b release.

Releases go one wave per Chicago day, with Jordan's yes. This note schedules none of them. The kernel's microbench can
read a grouped-nf4-gemm commit before its release. The e4b route can merge before the bump, refusing by name when the
installed package lacks `GEMV_I8_MAX_ROWS`.

### 8.5 Tests (with the code PRs)

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

### 8.6 Its readings (as reviewed)

1. **The kernel's microbench** (a grouped-nf4-gemm K-lane, one RTX 5090, 575 W). At the head's shape and M = 2, 4 and 8:
   K36 multi-row against cuBLAS bf16 on the grid (the served path at those rows), against K36 at one row, and against
   K37's `linear_i8` (the released multi-row int8 path, K2b's kernel), with the byte floor reported.
   - The proposed bar: the multi-row call at M = 2 and 4 within 1.25x of K36's one-row time (the bytes are the same)
     and at most 0.70x of cuBLAS's.
   - The contract tests (bitwise per row, the old M = 1 unchanged, registers) are its premise.
   - **The fallback rule, fixed before data:** if K2's multi-row call is slower than `linear_i8` by more than the
     registered bar, K2 is NO_GO. Moving the head onto `linear_i8` is then a separate decision with its own plan check
     at the head's shape, never a silent switch.
2. **The served read**, on its own registration: W2 and W4 ABBA on the default Qwen3 server, the knob against unset,
   the quality bar per bucket as a floor, and the head-logit identity across buckets registered (8.3).
