# P103 — the Gated DeltaNet kernels under the hybrid paged path: with flash-linear-attention and causal-conv1d installed, do Qwen3.6-35B-A3B's bucketed decode graphs still replay exactly through the serving stack, and what do the kernels buy, on one RTX 5090 and one host (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#928. Follows P101 (#926, SUPPORTED on transformers' torch path).

**Why.**
- `docs/SERVING.md` says the hybrid layers "run whatever kernels transformers finds (`fla` / `causal_conv1d`), or its
  torch path". Every GPU reading so far ran the torch path: P97 (#906), P98 (#912), P99 (#925) and P101 (#926). Each
  box's `versions.txt` reads `{'fla': False, 'causal_conv1d': False, 'kernels': False}`.
- **A $0 probe on the NAS RTX A2000** (#928) used the rented boxes' stack: torch 2.8.0, triton 3.4.0, transformers
  5.17.0, e4b `d88a371`.
  - `flash-linear-attention` 0.5.2 and `causal-conv1d` 1.7.0 install with torch held at 2.8.0, and transformers
    resolves its four Gated DeltaNet functions to them at import.
  - An all-linear dense Qwen3.5 under real capture, through e4b's per-slot pool and per-bucket selectors, replays bit
    for bit against the same padded steps run eagerly, with and without the kernels.
  - What the A2000 cannot answer: the fp8 paged stack on Qwen3.6 (sm_89+), and the kernels' cost or gain inside P101's
    12.6 ms one-row graph step.
- **One host, deliberately.** P101's eager arms ran about 2.2× P98's on a different CPU, because eager hybrid decode is
  launch-bound. A kernel comparison across hosts would measure the hosts, so every phase here runs on the same box.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **The measurement is P98's**, at its registered bytes (`bench/p98/staged.sha256`; `tests/test_p103_staged_pin.py`
  holds them equal):
  - `p98_box.py`'s three arms, each a fresh engine in its own process: **g** bucketed decode graphs (1, 2, 4, 8, 16),
    **e** the same padded steps eager (the bitwise oracle), **p** plain eager;
  - the workloads W16 (16 staggered requests, 256-token prompts, 48 + 4i new tokens) and W1 (one request, 96 new);
  - `p98_bake.py`'s NF4 arena;
  - Qwen3.6-35B-A3B @`995ad96`, e4b at the launch commit, grouped-nf4-gemm `34da93d`.
- **Three phases on one box,** in this order, with nothing uninstalled between them:
  - **t:** transformers' torch path. The tripwire asserts no `fla`, `causal_conv1d` or `kernels` module is present.
    This phase repeats P101.
  - **f:** after `pip install flash-linear-attention==0.5.2`, the chunk and fused-recurrent gated delta rules come from
    `fla`.
  - **fc:** after `pip install causal-conv1d==1.7.0` as well, the conv1d fn and update come from `causal_conv1d`.
  - Every install runs under a constraint holding torch at the image's version.
- **Each phase:**
  1. an **engagement record**, `kernels_<phase>.json`: what transformers resolves at import, the installed versions,
     torch unchanged, the `kernels` hub package absent;
  2. the **premise**: the three hybrid GPU test files, 4 passed, none skipped. With the kernels installed, the tests
     exercise them, including `test_linear_state_gpu.py`'s hybrid path against transformers' forward on the real fp8
     kernel;
  3. the three arms.
- **Gating:** phase t's premise gates the lane (rc 25). A kernel phase whose install or probe fails is recorded
  `install_failed`, and one whose premise fails is recorded `premise_failed`; either way its arms are skipped. Phase fc
  runs only after phase f holds.
- **The proving rental** (`P103_PROVE=1`) runs the install with its tripwire, the reducer's self-test, premise t, and
  both kernel installs with their probes and premises, all on the card, with no model. It answers whether the pinned
  kernels install and pass the premise on sm_120 before any checkpoint is fetched.

**The rule** (`p103_reduce.py`; its self-test holds 20 cases):
- **Per phase:**
  - P98's registered rule, unchanged (`p98_reduce.reduce`): every bucket captures and replays with no eager step, and
    arm g's tokens equal arm e's on all 17 requests.
  - A wrong engagement record makes the phase **VOID**.
  - A kernel phase is **UNAVAILABLE** when its install failed, **NOT_RUN** when skipped, and **NOT_SUPPORTED** when its
    premise failed with the kernels engaged.
- **The lane:**
  - **VOID** when phase t is not SUPPORTED (the torch-path baseline did not hold on this box);
  - otherwise phase fc's result when fc reached one, else phase f's, else **UNAVAILABLE**.
- **Reported, not gated:**
  - decode tok/s per arm and phase;
  - each kernel phase's speed over phase t, per arm and workload;
  - ms per step by rows;
  - each kernel phase's graph-arm token agreement with phase t's. The kernels' arithmetic differs from the torch
    path's, so greedy decode may diverge.
- **`recommend`** is true iff the lane is SUPPORTED, phase fc is SUPPORTED, and fc's graph arm decodes at least 1.05×
  phase t's graph arm on W16 or W1.

**The registered consequence:**
- **SUPPORTED, recommend:** `docs/SERVING.md` documents the two kernels as supported under the hybrid paged path and
  graphs, with the measured gain and the install line. A register row.
- **SUPPORTED, not recommend:** the same, minus the recommendation. The speed is reported as read.
- **NOT_SUPPORTED:** the failing phase, bucket or test is recorded on #928 before any change, and SERVING.md warns
  against installing that kernel for hybrid serving.
- **UNAVAILABLE:** SERVING.md records that the pinned kernels did not install or engage on sm_120.
- **VOID:** nothing moves.

**Not measured:**
- quality across kernels (no KL against the torch path or a bf16 reference);
- prefill and TTFT: P98's harness times decode only, and the chunk rule mostly serves prefill;
- any host but this one.

## Predictions (written before the data)

- **The lane is SUPPORTED, and `recommend` is true.** Phases t, f and fc each read SUPPORTED by P98's rule; the A2000
  replayed exactly on both kernels.
- **Phase t reproduces P101's tokens:** all three arms, both workloads, token for token. Reported, not gated. P98's and
  P101's eager arms already matched across hosts.
- **fc's graph arm over t's: W1 between 1.10× and 1.35×, W16 between 1.03× and 1.20×.**
  - At decode (T = 1 per row), the torch path issues roughly 20 small kernels per linear layer for the recurrent rule
    and about 5 for the conv update. Across 30 layers, that is perhaps 2 ms of P101's 12.6 ms one-row graph step.
    Fused, it is a few tenths of a millisecond.
  - The W16 step (28.1 ms) carries the same overhead at a larger total, so its gain is smaller.
- **fc's plain-eager arm over t's: W1 between 1.15× and 1.6×.** Eager is launch-bound, so removing launches shows up
  directly.
- **f alone delivers most of fc's graph-arm gain:** f over t is at least 60 % of fc over t, measured in excess over
  1.0×.
- **Peak GPU memory stays at or below 28 GB** in every arm.
- **The least certain prediction:** that causal-conv1d's prebuilt 1.7.0 wheel carries sm_120 code. The A2000 is sm_86.
  If it doesn't, phase fc reads UNAVAILABLE or NOT_SUPPORTED, and the headline falls back to phase f. The proving
  rental settles this before the reading.

## Box and cost

- **`p103-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P103_PROVE=1`, no model.
- **`p103-5090-<n>`:** one RTX 5090 with ≥ 200 GB of disk. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).**
  - P101's reading took 22 min, 16 of them fetching the snapshot.
  - Two more phases add two installs, two premises and six arms, about 15 min. Expected total about 40 min.
- **Lane ceiling $3.00; hard stop $4.00.** This leaves room for this morning's pattern: two slow hosts that were the
  cheapest verified offers refused seven launches at about $0.02 each.

## Rehearsal

Already done:
- the $0 A2000 probe above (#928);
- the reducer's self-test, 20 cases, including engagement mutants: fla resolved in phase t, conv functions not engaged
  in fc, another causal-conv1d version, torch moved, the hub package present;
- the driver's dry run;
- P98's measurement, bake and premise on Qwen3.6 on a 5090 in P101.

**The new piece is the runner's phase machinery.** Its proving path (install, tripwire, self-test, premise t, both
kernel installs, probes and premises) is rehearsed on the A2000 before the merge, with the class, disk and premise-skip
knobs that mark a REHEARSAL. On sm_86 the fp8 premise tests skip, so the rehearsal proves the machinery, not the
premise.

**Rehearsed 2026-10-03, 04:16–04:19Z, at `933c01e`, on the NAS RTX A2000**, staged exactly as `p103_drive.sh` stages.
`P103_PROVE=1` with the knobs class A2000, disk 10 GB and premise skips allowed. Exit rc 0, with `PROVED` and the
`REHEARSAL` marker.
- **The install:** e4b `933c01e`, gnf4 `34da93d`, torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0. The tripwire
  held: phase t has no `fla`, `causal_conv1d` or `kernels`.
- **The reducer's self-test:** 20 cases OK.
- **Phase t:** all four functions resolve to transformers' module. Premise 2 passed and 2 skipped (the fp8 tests on
  sm_86).
- **Phase f:** `flash-linear-attention` 0.5.2 installed with torch held. The chunk rule resolves to
  `fla.ops.gated_delta_rule.chunk`, the recurrent rule to `fla.ops.gated_delta_rule.fused_recurrent`, and the conv
  functions stay transformers'. Premise 2 passed and 2 skipped. Recorded `phase f ok`.
- **Phase fc:** `causal-conv1d` 1.7.0 installed with torch held, and both conv functions resolve to
  `causal_conv1d.causal_conv1d_interface`. Premise 2 passed and 2 skipped. Recorded `phase fc ok`.
- **The three real engagement records pass the reducer's checks unchanged**, and its state parser reads t, f and fc as
  `ok`.
- **HF CDN probe:** 34.4 MB/s.

Amendments, dated, go below this line before any data is read.
