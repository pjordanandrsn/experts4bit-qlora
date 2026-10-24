# P236: grouped-nf4-gemm 0.51.0's decode GEMV against bitsandbytes 0.50.2, op level, on the same NF4 bytes, on one RTX 5090 (a head-to-head at the released pins; descriptive; registered before any run)

Lane number claimed by `p236/registration`. Issue: #1313. A release-day head-to-head row: the project's own decode GEMV
against the library most NF4 users already run. No reviewed harness races the two (the P133, P134 and P141 harnesses
race K33 against grouped-nf4-gemm's own earlier GEMV, and bitsandbytes appears there only as an installed dependency),
so this is a new, small, op-level harness. It is **descriptive**: it reports ratios beside their self-pair bands, it
licenses nothing, and no default moves on it.

## The question

At decode sizes (M = 1, 2, 4, 8 and 16 rows of one projection), how long does grouped-nf4-gemm 0.51.0's decode GEMV take
against bitsandbytes 0.50.2's `matmul_4bit`, when both read the same NF4 weights, at the expert projections of
Mixtral-8x7B, Qwen3-30B-A3B and OLMoE-1B-7B, on one RTX 5090?

## The arms

| arm | the call | its route |
|---|---|---|
| **gnf4** | `nf4_grouped.gemm_4bit_grouped(x, B[None], absmax[None], [1] * M, eids)`: one singleton group per row, every grouped-nf4-gemm knob at its default | K33, the bandwidth GEMV: experts4bit-qlora's all-resident collapse makes this call at one row by default and at 2 to 8 rows under `E4B_NF4_SMALLM_BW`; at all six shapes on a >= 160-SM card, `auto` routes it to K33, one launch per call |
| **bnb** | `bitsandbytes.matmul_4bit(x, qW.t(), quant_state=qs)`, no grad, every bitsandbytes knob at its default | `bitsandbytes::gemm_4bit`, whose CUDA kernel picks bitsandbytes' fused `cgemm_4bit` kernel (`custom`) or dequantize then `F.linear` (`dequant`) by its own per-architecture rule; the route is counted and reported, never chosen |

**The weights are the same bytes.** For each projection a seeded random bf16 matrix (scale 0.02) is quantized once by
`bitsandbytes.functional.quantize_4bit` (NF4, block 64, no double quantization). grouped-nf4-gemm reads that packed
tensor as `[N, K/2]` and its fp32 absmax as `[N, K/64]`; bitsandbytes reads its own `QuantState`. Timing rotates over
enough distinct copies (each its own seeded matrix) to span four times the 5090's 96 MB L2 (at most 512 copies; Qwen3's
down projection needs 456), through ONE cursor per cell that persists across the warm-up and every visit of both arms.
Consecutive calls walk all the copies, so a weight is read again only after every other copy has been (at least four
L2s of other weights), and no arm reads a cached weight.

| projection | (N, K) | copies | reuse distance | x L2 |
|---|---|---:|---:|---:|
| Mixtral-8x7B gate_up | (28672, 4096) | 7 | 441 MiB | 4.6x |
| Mixtral-8x7B down | (4096, 14336) | 13 | 410 MiB | 4.3x |
| Qwen3-30B-A3B gate_up | (1536, 2048) | 228 | 385 MiB | 4.0x |
| Qwen3-30B-A3B down | (2048, 768) | 456 | 385 MiB | 4.0x |
| OLMoE-1B-7B gate_up | (2048, 2048) | 171 | 385 MiB | 4.0x |
| OLMoE-1B-7B down | (2048, 1024) | 342 | 385 MiB | 4.0x |

The reuse distance is the bytes (packed weight plus fp32 absmax) of the other copies read between two reads of the
same weight, with the cursor walking every copy. At the 5090's 96 MB L2 it is at least 4x at every projection.

## The read, per cell (projection x M; 30 cells)

1. **The premise:** bitsandbytes' `dequantize_4bit` on the bytes equals grouped-nf4-gemm's `dequant_ref` rounded to
   the weight dtype (bf16), bit for bit. (On CPU, bitsandbytes 0.50.2 and grouped-nf4-gemm's CI pin agree, and
   quantizing from fp32 gives the same bytes.)
2. **Correctness:** each arm's output against the fp32 reference `x.float() @ dequant_ref(W).t()`, as
   `max |y - ref| / max |ref|`, at most **0.015** (the served tolerance of `tests/test_nf4_smallm_bw_gpu.py`).
3. **Engagement**, in one eager pass of three calls per arm: grouped-nf4-gemm's dispatch tally shows exactly one
   `bw_prmt32` launch per call and no other decode-GEMV route; bitsandbytes makes one `gemm_4bit` dispatch per call, all
   to one route, counted at `bitsandbytes.backends.cuda.ops._gemm_4bit_kernel_impl` (`custom`) and
   `_dequant_linear_fallback` (`dequant`).
4. **Timing**, only for a cell whose premise, correctness and engagement all hold (checked by the reducer's own cell
   rules before any timing; a failed cell is recorded with its reason, never timed, and VOIDs the read): 10 untimed
   warm-up calls per arm, then **20 reps**. Each rep visits the arms in a palindrome,
   **gnf4 bnb bnb gnf4**, and odd reps swap the arms (bnb gnf4 gnf4 bnb), so each arm leads half the reps. A visit is
   **50 eager calls** between two CUDA events, rotating the weight copies; its time is the events' interval over 50.
   - **The ratio** of a rep is the bnb visits' sum over the gnf4 visits' sum: above 1, grouped-nf4-gemm is faster.
   - **The self-pairs** of a rep are each arm's second visit over its first.
   - Eager calls include each library's Python dispatch, as a caller outside a captured graph pays it.

## The rule (`bench/p236/p236_reduce.py`, self-tested on 54 cases)

**VOID** when any of these holds; **READ** otherwise. Each check fails closed: a missing record is a fault.
- the record is missing, a dry run or not `ok`;
- another grouped-nf4-gemm version or commit, another bitsandbytes version, another torch, or another card (an RTX
  5090, compute capability 12.0, at least 160 SMs);
- **the release is not proven** (the rule for every "at the released pins" row, 2026-10-24): each library's installed
  dist must BE its release, read from its `direct_url.json` by `provenance_faults`. An index wheel (no `direct_url.json`,
  or an archive URL on PyPI's hosts) at the registered version is accepted, and so is, for grouped-nf4-gemm only, a git
  install whose `vcs_info.commit_id` is the registered release commit. An editable install, a `file://` install, another
  commit or a git install of bitsandbytes is refused. The pins are constants checked unconditionally, and the box's
  tripwire runs the same check before anything is timed (rc 9);
- **bitsandbytes' bytes are not proven** to be the published release: `release_proof.py`, the shared helper (loggetta
  `bench/release_proof.py` at `d3d84173`, staged byte-identical, sha256
  `341dd130b1cf681a86f3fa1aad2b3e1bf9a2e4e6c22388e0e35b5736fc446713`), downloads the one wheel PyPI publishes for
  0.50.2 with the installed `WHEEL` tags, checks its digest, and compares its `RECORD` file by file with the bytes on
  disk (only pip's bytecode and declared launchers may be extra). The runner calls it after the installs and before
  any timing (a refusal stops the box, rc 9), and the reducer requires its "RECORD equal" in the record;
- a cell missing, or another rep count or ITERS;
- the premise fails on a cell, or either arm misses the correctness tolerance on a cell;
- engagement as registered fails on a cell (gnf4 not one K33 launch per call, or another route; bnb not one dispatch
  per call to one route);
- a timing sample missing, non-positive or not finite.

**Reported, never a bar:** per cell, the ratio's median and range over the 20 reps, both arms' median ms per call,
both arms' self-pair median and band (min to max), and bitsandbytes' route. A cell whose self-pair band is wide is
reported as it is, beside its ratio.

## Predictions

Written before any P236 data. They are expectations, not the sizing basis.

| # | prediction |
|---|---|
| Q1 | the premise and correctness hold in all 30 cells; K33 engages at every cell; bitsandbytes takes `custom` at M <= 4 (its floor) and its rule decides above |
| Q2 | at M = 1, gnf4 is faster at every projection: ratio between 1.2 and 3 (K33 reached 0.58 to 0.74 of the copy floor in P141's phase 1, and bitsandbytes' one-row kernel is not bandwidth-tuned for these shapes) |
| Q3 | the ratio falls as M grows, since K33 reads the weight once per row; at M = 16 bnb is faster on the large Mixtral projections (ratio below 1) |
| Q4 | the self-pair bands lie inside [0.95, 1.05] on the large projections; the small Qwen3 and OLMoE projections, under 20 us a call, show wider bands |
| Q5 | the verdict is READ (about 90 %), VOID (about 10 %) |

## Consequence, registered now

**READ:** the head-to-head row "grouped-nf4-gemm decode GEMV vs bitsandbytes (op level)" in `docs/REPRODUCE.md` carries
this read's ratios at the released pins, beside "none" as the previous reading. The wins and the losses are reported
together, cell by cell. No default moves, and no claim beyond the 30 cells is made. **VOID:** the row reads "not read
today: registered as e4b#N"; the one rerun (`p236-5090-2`) is the only path to a number.

**Scope.** Op level only: one projection, one card, two libraries at their released versions. Nothing about the served
decode step, quality, the other routes of either library, other cards or other shapes.

## Run ids and the stop

`p236-5090-1` is the reading and `p236-5090-2` its one rerun. The retry ids `p236-5090-r1` to `p236-5090-r3` follow a
box that did not start. A box is started at the `STARTED` marker, written once the refusals and the pre-install fit have
passed, immediately before the first install. The first READ ends the lane; at most 2 started boxes; two strikes end it.

| runner exit | before `STARTED` | after it |
|---|---|---|
| 0 | (cannot occur) | an ANSWER: READ |
| 13 free disk, 15 card / compute capability / SM count / VRAM, 16 host RAM, 18 CUDA unusable | **HOST_REFUSED**: no start | (cannot occur) |
| 41 the guard does not fit the deadline | **FIT_REFUSED**: a retry id; two consecutive are one strike | (cannot occur) |
| 9 staging, install or pins, 10 torch does not import, 28 the image's torch, 78 run id or frozen knobs, 130 a signal | **PRE_INSTALL_HALT**: the lane halts until it is fixed | a started box without an answer: a strike |
| 27 (VOID) and any other code | (cannot occur) | a strike |

## Budget

- **One RTX 5090** (Vast verified/secure, cc 12.0, at least 160 SMs, VRAM at least 30,000 MiB, host RAM at least
  16 GiB, disk at least 20 GB), at most USD 1.05/h, download at most USD 0.02/GB. No power floor (the read is a ratio
  inside one box; the limit is recorded). No model is downloaded.
- **The needs:** the installs, the tripwire and the self-tests 600 s; the bench 600 s (about 2 minutes by its call
  count: 30 cells x 20 reps x 4 visits x 50 calls, at 0.01 to 0.6 ms a call, plus the copies' quantization); a 600 s
  reserve. **The fit is 1,800 s**, and with the 25 % headroom the pre-install check needs **2,250 s** left.
- **Guard 0.85 h (3,060 s):** the boot slack, the guard less the fit's need, is **810 s**, at least the launcher's ssh
  wait (420 s) plus a 300 s margin, as the test checks.
- **Worst per box:** 0.85 h x USD 1.05 + 8 GB of installs x USD 0.02 = **USD 1.0525** (compared unrounded).
- **Worst per box that does not start:** 810 s x USD 1.05/h = **USD 0.23625**.
- **Lane worst:** 2 started x USD 1.0525 + 3 retries x USD 0.23625 = **USD 2.81375**, inside the row's USD 3.
- **Expected:** about 15 minutes, about USD 0.30.

## Launch

- **One exact launch commit:** this registration's merge, through pod-launch with `--validate` passing there, the host
  pinned, rent-watch armed; then the maintainer's Approved naming that commit and the reading kit's hash.
- **The pins:** grouped-nf4-gemm **0.51.0** at its release commit `30adb1bf` (`git rev-list -n1 v0.51.0` on the canonical
  repository, a code constant), bitsandbytes **0.50.2** (its latest release), torch **2.8.0+cu128** (the image's).
  The runner's tripwire checks the release proof (above), that K33 engages at all six shapes on the card with
  every knob unset, and that bitsandbytes' two `gemm_4bit` routes are where the bench counts them.
- **STOP-4:** the driver refuses a dirty tree, an `E4B_SHA` override, an unregistered run id or a staged file that
  differs from `staged-p236.sha256`; its fetch passes `rsync --no-links` (bench/common rule 8).

## What was seen before this page (stated, not hidden)

- No GPU number from either library at these cells on this card. P141's phase 1 read K33's per-call times at
  Mixtral's shapes (the basis of Q2), against grouped-nf4-gemm's own earlier GEMV, never bitsandbytes.
- **On CPU (a development host):** bitsandbytes 0.50.2's `matmul_4bit` dispatches `bitsandbytes::gemm_4bit` for every M
  (not `gemv_4bit` or `dequantize_4bit` at the Python level); its `dequantize_4bit` equals `dequant_ref` rounded to
  bf16; the bytes from a bf16 and an fp32 source agree. Its CUDA rule (read from its source) takes `custom` at M <= 4.
- **Locally:** the reducer's self-test (54 cases), the bench's self-test, and a `--dry` run of the bench on CPU through
  every cell into the reducer, which reads it VOID as a dry run with no cell fault other than the dry rep count.

## What this lane cannot say

- Nothing about either library's served decode step, graph capture, quality, or any route not taken here.
- Nothing about other cards, shapes or M.
- Nothing about bitsandbytes' double-quantized (`compress_statistics=True`) store, which this lane does not read.

## Receipts

Fetched to the run directory's `p236/` and committed to `bench/p236/receipts/<run>/`: `bench_p236.json`,
`verdict_p236.json`, `summary.txt`, `versions.txt`, `logs/`, with `SHA256SUMS`; `forensics.txt` and `outer.log` (the
card's identity) are kept only in the private receipt store and named in `STORE-ONLY.sha256`. `RESULTS-p236.md` is
written from those files.
