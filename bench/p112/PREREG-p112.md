# P112 — does `GNF4_PDL=1` decode SC1's int4 serving configuration's tokens exactly, and faster? On one RTX 5090 (registered 2026-10-04, before any run)

Lane number claimed by `prereg/p112` (pushed by 2026-10-04T06:28:04Z). Issue: #1015. Follows grouped-nf4-gemm's K28
(read LEVER, grouped-nf4-gemm#451) and its switch (grouped-nf4-gemm#448).

## Why this lane

**What SC1b found.** Its census (`bench/h2h-2026-10-02/sc1b/README.md`, `sc1d-5090-3`) put e4b's B=1 decode step on an
RTX 5090 at the same summed kernel work as llama.cpp's (4.087 against 4.135 ms). e4b overlapped none of its 1,550
in-graph kernels; llama.cpp overlapped 95.5 % of its consecutive pairs, hiding 1.38 ms a step. SC1b named programmatic
dependent launch (PDL) as the B=1 lever. 913 of e4b's 1,550 kernels were grouped-nf4-gemm's decode-row kernels.

**What K28 read.** grouped-nf4-gemm's `GNF4_PDL=1` (#448, opt-in) launches twelve of those kernels as programmatic
dependents of the kernel before them, each waiting for its predecessor to complete before it touches memory. K28
replayed 48 layers of the served B=1 layer's 19 gnf4 kernels (912) in one CUDA graph on an RTX 5090:
- 2.6576 → 2.3632 ms, ×1.125: **0.323 µs saved per gnf4 kernel**, with every output bitwise identical;
- 0.217 µs per kernel with two ATen kernels a layer interleaved, where the served step leaves gnf4.

K28's registered consequence is this lane: a served read, for tokens and decode tok/s at B=1 and B=16.

**The subject, and a correction to K28's wording.** K28's consequence named "the default decode step". That step does
not run what K28 measured:
- e4b's default server keeps NF4 experts and attention and leaves the fused glue off (`E4B_FUSE_T1_GLUE`,
  `E4B_FUSE_T1_GLUE_R2` and `E4B_FUSE_ROUTER_EPI` all default to `0`);
- so of the twelve switched kernels it reaches only the ones the NF4 path calls (`swiglu_rows` and `combine_rows`,
  which default on).

That is a reading of the code's defaults, not a measurement. The NAS A2000 could not check it: the served decode
graphs need the fp8 KV kernels (sm_89+). So this lane's box measures it. After the four arms, it builds the default
server once more (no levers, `GNF4_PDL=1`), records which switched kernels that build launched, and reports it.

The 913 kernels behind K28's bar are SC1's int4 configuration, the census subject. **P112 reads that configuration.**

## Subject

The default server (`PagedServeConfig.from_env()` + `build_engine`) with SC1's int4_sched levers, verbatim from
`bench/sc1/sc1_run.sh` (`SPEEDENV`, `FOLDS`, `ROUTEENV` and its fused qkv; the runner's `INT4_ENV`):
- `E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4`: RTN int4 experts and
  attention, packed at each arm's start (`E4B_SERVE_EXP_INT4_CALIB` stays unset);
- `E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1`;
- `E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto`;
- `E4B_PAGED_FUSE_QKV=1`.

Every other knob is the default: the server captures bucketed decode graphs (1–16, `max_seqs` 16, all-vram), and
`E4B_KV_STEP_SELECT` is on (0.44.0).

- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, NF4 arena baked on the box by P39's
  `k8_bake.py` (P109's and P111's subject).
- **Stack:** e4b at the launch commit; grouped-nf4-gemm at `951a97f` (0.36.0 with #448, #449 and #451, so the switch
  and K28's lane); transformers 5.17.0.

## Arms and workloads

Four arms in ABBA order, each a fresh process. `bench/p112/p112_box.py` runs each one, importing P109's prompts,
passes and slope at their registered bytes:

| arm | `GNF4_PDL` |
|---|---|
| P0a, P0b | `0`: no PDL (the shipped default) |
| P1a, P1b | `1`: grouped-nf4-gemm's decode-row kernels launch with PDL |

**Workloads** are P109's: W16 is 16 distinct 512-token wikitext prompts at once, and W1 is row 0 alone. Each runs 32
and 160 new tokens, one warm pass and 3 timed passes. Decode throughput is p37's slope.

**Recorded per arm:**
- every pass's token digest, and the last pass's tokens per row;
- `graph_status`, `pdl_active`, and the int4 configuration's counts (int4 expert layers, int4 attention projections,
  the fusion counts);
- **launch accounting:** a Triton launch hook armed for `build_engine` only (its warm passes and the decode-graph
  capture) counts the build's launches of each switched kernel and how many carried `launch_pdl`. It is disarmed
  before anything is timed;
- memory.

## The rule (`bench/p112/p112_reduce.py`, self-tested on 17 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - an arm is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - different prompts or lengths across arms;
   - a void slope;
   - wrong engagement:
     - every arm must report `graph` for every bucket (1–16);
     - SC1's int4 configuration must be in force in every arm (int4 stores on every MoE layer, int4 attention, the
       fused qkv, T1 glue and router epilogue), with the same counts in all four;
     - `pdl_active` must match the arm;
     - every arm's build must launch the same number of switched kernels, more than zero, with all of them carrying
       PDL in P1 and none in P0.
2. **NOISY.** A self-pair, P0b/P0a or P1b/P1a decode tok/s on either workload, falls outside [0.96, 1.04].
3. **FUNCTION_FAIL.** Any of:
   - P1a or P1b emits a token different from P0a's on any row of either workload at either length;
   - P0b differs from P0a;
   - an arm's timed reps do not all digest the same.
4. **SLOWER.** g16 = min(P1a/P0a, P1b/P0b) on W16 is below 1.00, or g1, the same on W1, is below 1.00.
5. **DEFAULT_ON** otherwise.

**Reported:** both pair ratios per workload, their geometric means, ms per step per arm, each arm's launch accounting,
and the default-server census.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | every arm captures every bucket with SC1's int4 configuration in force, and P1's build launches every switched kernel with PDL |
| Q2 | P1 ≡ P0 bitwise on every row of both workloads at both lengths |
| Q3 | g1 lies between 1.02 and 1.07. K28's chain saved 0.20–0.29 ms of the 913-kernel layer stack, 5–7 % of SC1b's 4.075 ms B=1 step; the served step interleaves about 640 kernels that are not switched |
| Q4 | g16 lies between 1.00 and 1.04. At B=16 the experts run grouped GEMMs, which are not switched, so the switched share of the step is smaller |
| Q5 | the self-pairs read within [0.98, 1.02] (P111's read 0.997–1.002) |
| Q6 | the verdict is DEFAULT_ON, at about 60 %; SLOWER at B=16 inside the noise is the likeliest alternative |

## Consequence, registered now

- **DEFAULT_ON.**
  - grouped-nf4-gemm turns `GNF4_PDL` on by default in its next release; it stays inert off NVIDIA and below sm_90.
  - experts4bit-qlora's CI moves to that release.
  - A register row `e4b.serve.p112.gnf4-pdl.qwen3-int4.5090.<date>` carries g1 from the verdict file, with g16 in
    its text.
  - **Scope:** read on Qwen3-30B-A3B with SC1's int4 configuration. The switch is value-identical by construction on
    every path, which is what licenses a package default; the speed was read on this configuration only.
- **SLOWER.** The switch stays opt-in, and the ratios are recorded.
- **FUNCTION_FAIL.** The switch is withdrawn until the token difference is explained, and nothing else moves.
- **NOISY or VOID.** No consequence; an amendment or a rerun.

## The premise, and why there is no proving rental

**The premise.** It runs on the card before anything is fetched (rc 25):
- `test_decode_graph_buckets.py`, **7 passed**, none skipped: a replay decodes exactly as the padded eager step;
- grouped-nf4-gemm's `kernel/test_pdl.py` from a clone at `951a97f`, **23 passed**, none skipped. These are every
  switched wrapper bitwise on and off, eager and graph-captured, and the engagement probe; K28 ran them on its 5090.

**Why no proving rental.** P111 proved its box on `ibm-granite/granite-3.1-3b-a800m-instruct`. That cannot prove this
one: the subject's fused qkv is Qwen3's, needing per-head q/k norms, and Granite refuses it ("matched no attention
module", seen on the NAS A2000). A proof on Qwen3 itself would cost about what the reading costs. The reading runs
under a **1.0 h guard**, about twice its expected time, so the lane rule asks for no proving rental.

## Budget and STOP rules

- **Reading:** one RTX 5090 (Vast verified/secure), **guard 1.0 h** at ≤ $0.75/h, estimate $0.75. There is no proving
  rental (above). The expected time is about 35 minutes:
  - install and premise 3 minutes, fetch 6–10 and bake 2;
  - four arms of about 4 minutes each, since RTN packs the int4 stores at each arm's start (SC1's `load_s` was
    135–148 s);
  - the census, about 2 minutes.
- **Lane ceiling:** $2.00, inside the owner's $15 no-ask tier. A failed reading is amended, not retried blind.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16), premise (25).
- **STOP-2:** every time-left check (fetch, bake, each arm, the census) fits inside the guard, enforced by
  `tests/test_p112_staged_pin.py`. The census is skipped, not forced, when time runs short.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p112/staged.sha256`.

## What was seen before this page (stated, not hidden)

Everything below ran on the NAS RTX A2000 (sm_86), in a throwaway `pytorch:2.8.0-cuda12.8` container at the lane's
work-in-progress commit `a0146ab`. That card is correctness-only and cannot run PDL or the fp8 KV kernels. None of it is
a reading.
- The reducer's self-test (17 cases) and the box's (5) passed. The box's launch accounting is tested in its own
  self-test.
- **Granite cannot run the subject.** With SC1's int4 levers, every arm refused at build: "E4B_PAGED_FUSE_QKV=1 matched
  no attention module -- refusing a vacuous fusion". This is why there is no proving rental.
- **The A2000 cannot run the served decode graphs.** The default server's build reported every bucket `eager:
  CompilationError` in the fp8 KV append, which needs sm_89+, and launched no Triton kernel. The default-server census
  therefore has to run on the 5090 box.
- `tests/test_p112_staged_pin.py` passes locally (10 tests), including the driver's dry run.

## What this lane cannot say

- **Nothing about the NF4 default server**, which reaches only two of the switched kernels.
- **Nothing about e4b's own decode kernels** (the KV append, the paged decode), which the switch does not reach.
- **Nothing about other cards or families.**

## Receipts

Fetched to the run directory's `p112/` and committed to `bench/p112/receipts/<run>/`:
- `arm_{P0a,P1a,P1b,P0b}.json`, `verdict.json`, `census_default.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
  `prompts.json`, `bake.json`;
- the logs and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p112.md` is written from those files.

## Amendment 1 (2026-10-04, after `p112-5090-1`, written after seeing its arms): the launch accounting's compile launches

`p112-5090-1` ran on one RTX 5090 (Vast 54118102, machine 37958, AMD Ryzen 9 7950X), cost $0.3210, and is in adertha-receipts
`a88df20`. Teardown was proven at 07:31:36Z. Everything before the verdict passed:
- the tripwire and both self-tests;
- the premise: 7 + 23 passed, none skipped;
- the fetch, the bake and the prompts;
- all four arms, rc 0;
- the census.

**The verdict was VOID on engagement:** "P1a: 6907 of 6927 switched launches carried PDL", and the same for P1b.

**Why: the instrument, not the switch.** Triton 3.4's JIT computes the launch hook's metadata, which reads
`kernel.function`, before it calls `kernel.run`. The handle stays `None` until `run` initialises it
(`python/triton/compiler/compiler.py`, `CompiledKernel.__getattribute__`). So the launch that compiles a variant reaches
the hook with `function=None`, and the box counted it as a launch without PDL. In both P1 arms each kernel's shortfall is
one launch per variant that configuration compiles:

| kernel | shortfall | variants |
|---|---:|---|
| `_gemv_int4_b32` | 4 | qkv, o, gate_up and down at one row; the multi-row attention projections take the small-M GEMM |
| `_rmsnorm_rows` | 4 | |
| `_quant_x_rows` | 3 | |
| `_reduce_partials` | 3 | |
| `_rope_norm_heads` | 2 | q and k heads |
| `_rmsnorm_resid_rows`, `_router_epilogue`, `_swiglu_rows`, `_combine_rows` | 1 each | |
| **total** | **20** | |

The run did not record the variants, so that match is inferred. The amended box records the variants, and the rerun
checks the match directly.

**What the arms showed.** This is not a reading, because the verdict is VOID; it is stated so that nothing is hidden.
- **Tokens:** identical in every arm, on every row of both workloads at both lengths.
- **The rule's later steps** would have read **SLOWER**:
  - g1 = 1.0366 (pairs 1.0393 / 1.0366): the B=1 step went from 4.44 to 4.28 ms;
  - g16 = 0.9851 (pairs 0.9851 / 0.9854): the B=16 step went from 9.05 to 9.19 ms;
  - the self-pairs were 1.0023–1.0047.
- **The census:** the default NF4 server's build launched only `_swiglu_rows` and `_combine_rows` of the twelve (1,440
  launches), with no int4 store and no fusion. That confirms the correction above.

**The change.**
- **The instrument.** The box's accounting row is now `[launches, with launch_pdl, compile launches (no handle),
  compiled variants, variants with launch_pdl]`.
- **The engagement clause.**
  - **In P1:** every variant of a launched switched kernel carries `launch_pdl`, every launch with a handle carries PDL,
    and compile launches do not outnumber the variants.
  - **In P0:** no variant and no launch carries PDL.
  - The other engagement conditions are unchanged. `p112_reduce.py` self-tests on 21 cases.
- **The fetch.** The driver no longer fetches the box's grouped-nf4-gemm clone. `p112-5090-1`'s store commit had picked
  it up as a gitlink, which adertha-receipts `551651d` removed.
- **The rerun.** `p112-5090-2` runs under the same relay, guard (1.0 h) and ceiling ($2.00; $0.3210 spent).

**Predictions for the rerun.** These were written after seeing `p112-5090-1`'s arms, so they are not blind:
- Q1, Q2 and Q5 are unchanged.
- Q3′: g1 lies between 1.02 and 1.06.
- Q4′: g16 lies between 0.97 and 1.00, so PDL is slower with 16 concurrent requests.
- Q6′: the verdict is SLOWER, at about 80 %. Its registered consequence is that the switch stays opt-in. A switch that
  applies PDL only where it helps would need a lane of its own.

**Unchanged:** the subject, the arms, the workloads, the rule's other steps and its bars, and the consequences.
