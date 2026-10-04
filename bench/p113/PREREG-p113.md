# P113 — does grouped-nf4-gemm's programmatic dependent launch, everywhere or capped to small launches, decode SC1's int4 serving configuration's tokens exactly, and faster, under decode-only timing? On one RTX 5090 (registered 2026-10-04, before any run)

Lane number claimed by `prereg/p113` (pushed by 2026-10-04T08:40:26Z). Issue: #1015. Follows P112 (closed VOID,
#1026), grouped-nf4-gemm K28 (LEVER, grouped-nf4-gemm#451) and the two switches: `GNF4_PDL` (grouped-nf4-gemm#448) and
`GNF4_PDL_MAX_ROWS` (grouped-nf4-gemm#453).

## Why this lane

**What K28 read.** `GNF4_PDL=1` saved 0.323 µs per gnf4 kernel in a CUDA-graph replay of the served B=1 decode
layer's 912 gnf4 kernels on an RTX 5090, bit-identically.

**What P112 saw but could not read.** P112 put the switch on SC1's int4 serving configuration and closed VOID twice:
- once on its own launch accounting;
- once on a decode slope that a shared host's prefill jitter voided.

Its first run's arms, on a quiet host, showed tokens identical in every arm and:
- B=1 decode ×1.039 / ×1.037 (the step 4.44 → 4.28 ms);
- B=16 ×0.985 / ×0.985 (9.05 → 9.19 ms).

PDL helped one request and cost sixteen.

**The cap.** At B=1 every switched launch carries 1 or 8 activation rows; at B=16 every one carries 16 or more.
grouped-nf4-gemm's `GNF4_PDL_MAX_ROWS=8` (#453) keeps PDL for launches of at most 8 rows. On this subject it should
therefore be ALL's speed at B=1 and OFF's at B=16.

**This lane** reads three settings at once, under decode-only timing, with P112 Amendment 1's accounting.

## Subject

P112's subject, unchanged: the default server (`PagedServeConfig.from_env()` + `build_engine`, graphs on, `max_seqs`
16, buckets 1–16, all-vram, `E4B_KV_STEP_SELECT` on) with SC1's int4_sched levers verbatim (the runner's `INT4_ENV`).
- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, NF4 arena baked on the box by P39's
  `k8_bake.py`.
- **Stack:** e4b at the launch commit; grouped-nf4-gemm main at `bc2214c` (0.36.0 with the opt-in switches `GNF4_PDL`,
  #448, and `GNF4_PDL_MAX_ROWS`, #453); transformers 5.17.0.

## Arms, workloads and timing

Six arms in a palindrome, **OFF1 ALL1 CAP1 CAP2 ALL2 OFF2**, each a fresh process (`bench/p113/p113_box.py`):

| setting | environment |
|---|---|
| OFF | `GNF4_PDL=0` (the shipped default) |
| ALL | `GNF4_PDL=1` |
| CAP | `GNF4_PDL=1 GNF4_PDL_MAX_ROWS=8` |

**Workloads** are P109's: W16 is 16 distinct 512-token wikitext prompts at once, and W1 is row 0 alone. Each runs 32
and 160 new tokens, one warm pass and 3 timed passes.

**Decode-only timing** (P112 Amendment 2; SC1's A10). Every pass records each row's `Request.ttft`; the rows arrive
together. The decode-only time is the wall minus the pass's largest ttft, and p37's slope (P109's `slope`) is taken
over those times. The whole-pass slope is reported beside it.

**Recorded per arm:**
- every pass's wall, ttfts and token digest, and the last pass's tokens per row;
- `graph_status`, `pdl_active`, `pdl_cap`, and the int4 configuration's counts;
- P112 Amendment 1's launch accounting of the build: each switched kernel's launches, those with PDL, compile
  launches, compiled variants, and variants with PDL;
- memory.

## The rule (`bench/p113/p113_reduce.py`, self-tested on 22 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - an arm is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - different prompts or lengths across arms;
   - a void decode-only slope;
   - wrong engagement:
     - every arm must report `graph` for every bucket;
     - SC1's int4 configuration must be in force, with the same counts in all six;
     - `pdl_active` must be off in OFF and on in ALL and CAP, and `pdl_cap` must be 8 in CAP and 0 elsewhere;
     - every build must launch the same number of switched kernels, more than zero, with no more compile launches than
       variants. OFF carries no PDL anywhere. ALL carries it on every variant and every handled launch. In CAP the cap
       splits the build: at least one handled launch carries PDL and at least one does not.
2. **NOISY.** A self-pair, OFF2/OFF1, ALL2/ALL1 or CAP2/CAP1 decode tok/s on either workload, falls outside
   [0.96, 1.04].
3. **FUNCTION_FAIL.** Any arm emits a token different from OFF1's on any row of either workload at either length, or
   an arm's timed reps do not all digest the same.
4. **ALL_DEFAULT.** gALL1 = min(ALL1/OFF1, ALL2/OFF2) on W1 is at least **1.02**, and gALL16, the same on W16, is at
   least **1.00**.
5. **CAP_DEFAULT.** Otherwise, gCAP1 is at least **1.02** and gCAP16 at least **0.99**.
6. **NONE** otherwise.

**Reported:** every pair ratio and geometric mean, ms per step, the whole-pass slopes, and the launch accounting.

**Why these bars.**
- **1.02 at B=1** is about the smallest gain a served lane reads above its self-pairs (P111's read 0.997–1.002).
- **ALL may not cost B=16 at all** (1.00, P112's bar).
- **CAP is held to 0.99 at B=16.** At B=16 the cap launches nothing with PDL, so its expected ratio is 1.00, and 0.99
  is the noise allowance around it.

## Predictions

Written before any P113 data, but after seeing P112 run 1's arms; these are not blind.

| # | prediction |
|---|---|
| Q1 | every arm's engagement holds, including CAP's split |
| Q2 | every arm's tokens equal OFF1's on every row |
| Q3 | gALL1 lies between 1.02 and 1.06 |
| Q4 | gALL16 lies between 0.97 and 1.00 (PDL everywhere costs B=16, as in P112 run 1) |
| Q5 | gCAP1 lies between 1.02 and 1.06, within 0.01 of gALL1 |
| Q6 | gCAP16 lies between 0.99 and 1.01 |
| Q7 | under decode-only timing the self-pairs read within [0.98, 1.02] |
| Q8 | the verdict is CAP_DEFAULT (about 60 %), NONE (about 25 %) or ALL_DEFAULT (about 15 %) |

## Consequence, registered now

- **ALL_DEFAULT.**
  - grouped-nf4-gemm turns `GNF4_PDL` on by default, uncapped, in its next release; it stays inert off NVIDIA and below
    sm_90.
  - experts4bit-qlora's CI moves to that release.
  - A register row `e4b.serve.p113.gnf4-pdl.qwen3-int4.5090.<date>` carries gALL1 from the verdict file.
- **CAP_DEFAULT.**
  - grouped-nf4-gemm's next release turns `GNF4_PDL` on by default with `GNF4_PDL_MAX_ROWS` defaulting to 8. Either can
    still be set: `GNF4_PDL=0` turns it off, and `GNF4_PDL_MAX_ROWS=0` removes the cap.
  - experts4bit-qlora's CI moves to that release.
  - The register row carries gCAP1.
- **NONE.** `GNF4_PDL` stays opt-in, and the ratios are recorded.
- **FUNCTION_FAIL.** The switch is withdrawn until the token difference is explained, and nothing else moves.
- **NOISY or VOID.** No consequence; an amendment or a rerun inside the ceiling.

**Scope.** The speed is read on Qwen3-30B-A3B with SC1's int4 configuration. The switches are value-identical by
construction on every path, which is what licenses a package default; the docs will say where it was read.

## The premise, and why there is no proving rental

**The premise** runs on the card before anything is fetched (rc 25):
- `test_decode_graph_buckets.py`, **7 passed**, none skipped;
- grouped-nf4-gemm's `kernel/test_pdl.py` from a clone at the pinned commit, **25 passed**, none skipped. These cover
  every switched wrapper bitwise on and off, eager and graph-captured, the engagement probe, and the row cap.

**No proving rental**, as in P112. The small proof model (Granite) cannot run the subject's Qwen3-only fused qkv, so a
proof would be the reading's cost. The reading's guard is 1.0 h.

## Budget and STOP rules

- **Reading:** one RTX 5090 (Vast verified/secure), **guard 1.0 h** at ≤ $0.75/h. **Each run is priced at about $1.95:**
  the launcher's own estimate, the hourly rate plus the checkpoint download at about $0.011/GB. P112's registration
  omitted the download, and the invoice for its run 2 showed $0.649.
- **Expected time:** about 45 minutes. That is install and premise 3 minutes, fetch 6–10, bake 2, and six arms of
  about 4 minutes each (RTN packs the int4 stores at each arm's start).
- **Lane ceiling:** $4.00, the reading and one rerun, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16), premise (25).
- **STOP-2:** every time-left check (fetch, bake, each arm) fits inside the guard, enforced by
  `tests/test_p113_staged_pin.py`.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p113/staged.sha256`.

## What was seen before this page (stated, not hidden)

- **P112's two runs** (`bench/p112/RESULTS-p112.md`): run 1's arms are quoted above. Run 2's accounting confirmed P112
  Amendment 1's fix.
- **The NAS A2000** cannot run the served decode graphs, because the fp8 KV kernels need sm_89+. So this box has had no
  local rehearsal.
- **Locally:** the box's self-test (8 cases) and the reducer's (22) pass, as does `tests/test_p113_staged_pin.py`.

## What this lane cannot say

- **Nothing about the NF4 default server.** It reaches only two of the switched kernels (P112's census).
- **Nothing about e4b's own decode kernels**, which the switch does not reach.
- **Nothing about other cards, other families, or batch sizes between 1 and 16.**

## Receipts

Fetched to the run directory's `p113/` and committed to `bench/p113/receipts/<run>/`:
- `arm_{OFF1,ALL1,CAP1,CAP2,ALL2,OFF2}.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
  `prompts.json`, `bake.json`;
- the logs and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p113.md` is written from those files.
