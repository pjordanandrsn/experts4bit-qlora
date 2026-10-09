# P127 — the launch-bound glue on the shipped default's decode: bitwise identity and speed, Phases 1 and 2 (registered 2026-10-09, before any run)

Issue: experts4bit-qlora#1313. The lane number was claimed by `prereg/p127`, pushed 2026-10-09T07:12:11Z.

## Why this lane

**What P123 priced.** At B = 1 on the shipped default, the small glue launches (`moe_route` and `norm_elem`) are about
635 launches and 0.88 ms of a 4.75 ms step (P123, `bench/p123/RESULTS-p123.md`).

**What P127 changed.** It removed launches in two phases. Each change is bitwise by construction, and each has its
own test:
- **Phase 1** (#1448, merged `ce3dfb54`) dropped three host casts the kernels already do:
  - the router logits' widening;
  - the combine weights' widening;
  - the NF4 expert ids' second int32 cast.
- **Phase 2** adds five grouped-nf4-gemm options and their e4b wiring. Each is feature-detected by signature or
  capability constant, so e4b on an older kernel package behaves as before:
  - (b1) the router kernel stores the weights in the logits' dtype (gnf4 #526);
  - (d) q's and k's norm + rotary in one launch (#528);
  - (a1) int64 expert ids read as they are (#529);
  - (b2) the singleton decode reads the token row in place at one token (#530);
  - (c) the decoder layer's MoE residual add in the combine's epilogue (#527). It is licensed per row count by a
    probe of the model as served.
  - The e4b wiring is #1472 (b1, d, a1, b2) and the residual PR (c).

**No knob.** None of this has one: the changes ship on. This lane therefore decides the **claim**, not shipping. It
must do three things:
1. **Show identity on the real served model.** The unit tests run on kernel stand-ins and a correctness-only A2000.
2. **Measure the speed** for the claim.
3. **Catch a slowdown.** A SLOWER verdict puts the Phase 2 options behind an E4B knob defaulting OFF in a follow-up
   (the maintainer's rule, 2026-10-09).

## Subject

The shipped server as `PagedServeConfig.from_env()` builds it with `build_engine`, at its defaults, on one RTX 5090:
- **Fixed by the lane:** `E4B_PAGED_MAX_SEQS=16`. `auto` sizes the server from free memory, and the two arms could
  then differ.
- **Graphs:** on (`auto` on sm_120), buckets 1–16, all-vram, chunk 512, 4,096 tokens per sequence, NF4 experts.
- **Fusions:** every fusion knob unset. On `qwen3_moe` they resolve to `auto` (P115's family default): fused q/k/v,
  both glue folds, and the router epilogue.
- **Other routes:** every other route at its default, including the bandwidth GEMV and the lean glue.

**Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, SC1's pin.

**Inputs:** the NF4 arena is baked on the box by P39's `k8_bake.py`, and the host calibration is P39's `calib.json`,
both at SC1's bytes.

**Prompts:** P109's. Row k is tokens `[k·4096, k·4096+512)` of wikitext-2-raw test, joined as P97 joins it, written
once to `prompts.json` and digest-checked by every arm.

## Arms

Each arm is a fresh process. The box switches the editable installs before each arm and records the module paths it
imported.

| arm | e4b | grouped-nf4-gemm | box-side change |
|---|---|---|---|
| A (before P127) | `a8c01d42`, the last main commit before #1448 | `d1f64ba`, v0.44.0 | none |
| B (after P127) | the launch commit: this registration's merge, which carries #1448, #1472 and the residual PR | `d769d502`, the merge commit of gnf4 #527, which carries #526–#530 | none |
| M (blindness check) | B's | B's | the router kernel's bf16 weight store **rounds toward zero** instead of to nearest even: a box-side wrapper of `int4_b32.router_epilogue` keeps its signature |

**Order:** A1, B1, B2, A2, then M1. A and B alternate in a palindrome, so drift lands on both.

**M** runs the identity pass only.

**Workloads** on every engine, as P109's:
- **W16:** the 16 rows added at once;
- **W1:** row 0 alone on the same 16-slot server.

Per workload and length (SHORT 32, LONG 160 new tokens), the timed arms run one untimed warm pass, then 3 timed passes.
Every request runs to `max_new_tokens`. Decode throughput is p37's slope.

**The identity pass.** After the timed passes, each arm runs one untimed W16 pass and one untimed W1 pass at LONG. It
records each row's tokens and, at every decode step, a SHA-256 digest of that row's last-position logits.
- **How the logits are read.** Before the engine is built, the box wraps `PagedModelRunner._padded_step` to keep a
  reference to the step's `out.logits[:, -1]`. That is graph-owned memory, so a replay overwrites it in place, and no
  kernel is added.
- **When they are read.** The box also wraps `run_decode` to read that view for the step's rows after the step's own
  token read has synchronized, and before `run_decode` returns.
  - **Ordering.** Nothing can replay before the read finishes. `tests/test_p127_box.py` asserts this ordering against a
    runner whose next step overwrites the same storage.
  - **Timed passes.** They do no digest work at all; the wrapper's switch is off.
- **Prefill.** Prefill logits are not digested; the first generated token covers them.

**Engagement.** The box wraps, before the build, each kernel entry point P127 changed: `router_epilogue`,
`rope_norm_qk`, `combine_rows` and `gemm_4bit_grouped`. Each wrapper keeps the signature, so feature detection reads
the real one. It counts, over the arm:
- `router_epilogue` calls with a non-fp32 `weights_dtype`;
- `rope_norm_qk` calls;
- `combine_rows` calls with a `residual`;
- `gemm_4bit_grouped` calls with `gather_div > 1`, and with int64 ids.

**How counting works under graphs.** Decode steps replay graphs, which run no Python, so the counts come from the
eager warm-ups, the captures and prefill. A path counted once is in the captured graph.

## The served-path diff audit

Every change to the `experts4bit_qlora` package between `a8c01d42` and the launch commit, and every grouped-nf4-gemm
change between `d1f64ba` and `d769d502`, is either P127's or inert at this subject's settings:

| repo | commit | what | status |
|---|---|---|---|
| e4b | `ce3dfb54` (#1448) | Phase 1: three host casts | **P127** |
| e4b | `1ddcb0ae` (#1472) | Phase 2 wiring (b1, d, a1, b2) | **P127** |
| e4b | the residual PR | Phase 2 (c), plus `serve_paged.build_engine`'s licence step | **P127** |
| e4b | `058f98eb` (#1456) | Release 0.51.0: `__init__.__version__` only | inert: a version string |
| e4b | `f5398962` (#1459) | `E4B_TRAIN_FUSE_QKV` (`engines/fast.py`, `engines/train_qkv_fuse.py`) | inert: training only, nothing without `E4B_TRAIN_FUSE_QKV=1` and the fused training path |
| e4b | `abe7d772` (#1467) | `recipe.py` training memory estimate | inert: not on the serving path |
| gnf4 | `14b1f23` (#524) | `build_group_tiles_fused(..., programs=P)` | inert: e4b passes `programs` only for `E4B_INT4_TILE_PROGRAMS > 1`; at the default, the launch is the unchanged `_tile_table_r1[(1,)]` with the same arguments |
| gnf4 | `e21a712` (#525), `7d4163b` (#531) | K34's prereg, bench and receipts | inert: no served module |
| gnf4 | `18f5bda` (#526), `f69adcc` (#528), `d0a2e56` (#529), `b64a39b` (#530), `d769d502` (#527) | the five options | **P127** |

**The launch-time check.** `p127_run.sh` lists `git log a8c01d42..E4B_SHA -- experts4bit_qlora/` and
`git log d1f64ba..d769d502 -- kernel/ gnf4_native/` on the box. It refuses (VOID, rc 31) a commit not in this table. If a
non-P127 served-path change lands before the launch, it is added here by amendment, with its reason, or the launch
waits.

## The rule (`bench/p127/p127_reduce.py`, self-tested)

The verdict is the first that applies.

1. **VOID**, any of:
   - an arm is missing or not ok;
   - a receipt names another e4b or gnf4 commit or model revision, or an arm imported the other arm's install;
   - the arms read different prompts or ran different lengths;
   - an arm resolved a `max_seqs` other than 16, or the arms resolved different ones (each record carries its own);
   - a slope is void;
   - **not engaged**, any of:
     - B or M counts zero for any of the five P127 call paths: the router `weights_dtype`, `rope_norm_qk`, the combine
       `residual`, `gather_div` and int64 ids;
     - B's build census (`info["moe_residual"]`) does not license every MoE layer at every decode bucket;
     - A counts any of the four new-API paths (`weights_dtype`, `rope_norm_qk`, `residual`, `gather_div`). A's int64
       count is expected: before #1448 the NF4 route handed torch's int64 ids to the wrapper, which cast them;
   - **nondeterministic**: A1 and A2, or B1 and B2, differ in any token or identity-pass logits digest;
   - the diff audit refused;
   - **the instrument is blind**: M's logits digests differ from A1's at fewer than **50 %** of the identity pass's
     decode positions (P126's Q8 bar, here on logits).
2. **FUNCTION_FAIL**, any of:
   - B1 differs from A1 in any token of any row of either workload at either length, timed or identity pass;
   - B1 differs from A1 in any logits digest of the identity pass;
   - an arm's timed passes do not all digest the same.

   With the nondeterminism VOID above, B1 against A1 stands for every B/A pair.
3. **NOISY:** a self-pair (A2/A1 or B2/B1) decode tok/s falls outside **[0.97, 1.03]** on either workload.
4. **Speed.** The block ratios are r1 = B1/A1 and r2 = B2/A2, decode tok/s per workload. Their mean is g1 on W1 and
   g16 on W16, and the block interval is [min(r1, r2), max(r1, r2)].
   - **SLOWER:** g1 ≤ 0.98 or g16 ≤ 0.98.
   - **FASTER:** g1 ≥ 1.03 and both W1 block ratios > 1.
   - **NO_MEASURABLE_GAIN:** otherwise.

**Reported, ungated:**
- g1, g16 and their intervals;
- the median-pair slopes;
- M's first differing position per row, and its token agreement with A;
- each arm's engagement counts, peak memory, load time and graph stats;
- the step-time saving in µs per step, set against the launch arithmetic below.

## Predictions (written before any data; evaluated by the reducer, none gates the verdict)

| # | prediction | basis |
|---|---|---|
| Q1 | g1 in [1.03, 1.12] | about 8 fewer launches a layer × 48 layers ≈ 384 at T == 1. At 1–1.5 µs a node in a replayed graph, that is 0.4–0.6 ms of about 4.6 ms |
| Q2 | g16 in [0.99, 1.04] | at T == 16 only the router store, the q+k rotary and the residual apply. a1 and b2 are T == 1 routes, and the step is longer |
| Q3 | M's logits differ at ≥ 90 % of positions; its tokens agree with A on ≥ 50 % of W16's positions | a rounding change in the routing weights moves every logit slightly, and greedy tokens rarely |
| Q4 | the self-pairs agree within 1 % | P109's and P126's interleaves |
| Q5 | peak memory: B − A within ±64 MiB | no new buffer outlives a step |

## Consequence, registered now

- **FASTER:** the read PR adds a claims row (g1 with its interval, g16 with its interval). STATUS records the T == 1
  gain.
- **NO_MEASURABLE_GAIN:** the claims row records "no measurable gain" with the intervals. The changes stay; they are
  bitwise.
- **SLOWER:** a follow-up e4b PR puts the Phase 2 options behind `E4B_P127_PHASE2` (default `0`) until a lane explains
  the slowdown. Phase 1's host casts are not knob-able and stay. RESULTS names the arm and the step share that moved.
- **FUNCTION_FAIL:** an issue the same day. The reading's first differing (row, step) and the engagement counts go into
  it, and the responsible option is backed out by a revert PR until it is fixed.
- **NOISY:** a rerun with more reps is an amendment.
- **VOID:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed graphs. `tests/test_p127_box.py` covers the box on CPU, with the
engine, the scheduler and the kernels stood in:
- the arm switch;
- the logits wrapper and its reads;
- the engagement wrappers, which keep the signatures;
- M's round-toward-zero store;
- the reducer on fixtures for every rung.

**The proof** runs the whole box end to end on Qwen3-30B-A3B itself at SHORT 8 / LONG 24, 1 rep. Only this family
engages every P127 path: the fusions' family default is `qwen3_moe`. It covers:
- the refusals, the install of both stacks, the tripwire and the diff audit;
- the self-tests and the premise;
- fetch, bake, prompts, the five arms and the reducer.

**PROVED** iff the lane exits 0 with a verdict other than VOID. The proof's speed is not a reading. Its guard is
sized from SC1-era fetch times on this checkpoint, about 15 minutes for the 61 GB.

**The premise, on the card before anything is fetched:**
- `tests/test_decode_graph_buckets.py`: 7 passed, none skipped;
- `tests/test_t1_glue_host_casts.py`: its CUDA test passes on the card at B's gnf4.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 1.25 h at ≤ $0.75/h, estimate $0.94. It is mostly the 61 GB
  fetch and the bake.
- **Reading:** one RTX 5090, guard 2.5 h at ≤ $0.75/h, estimate $1.875. The expected time is about 90 min: install 10,
  fetch 15, bake 15, five arms of 6–10 min each.
- **Lane ceiling:** $4.00, inside the $15 no-ask tier. Each rental goes after this registration merges and the
  maintainer ACKs.
- **STOP-1:** the refusals (card class, disk, RAM, a dud box, the premise, the diff audit) run before the fetch.
- **STOP-2:** an arm that cannot finish 10 minutes before the deadline is skipped, and the reducer VOIDs.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p127/staged.sha256`.

## What this lane cannot say

- Nothing about other families, the int4 stack, `max_seqs` other than 16, or `auto`'s sizing.
- Nothing about prefill or TTFT beyond the tokens. Prefill is in both walls and cancels.
- Nothing about quality beyond identity. B is bitwise A, or the verdict is FUNCTION_FAIL.
- Nothing about which option bought the speed: the arms are all-or-nothing. A per-option split would be its own lane.
