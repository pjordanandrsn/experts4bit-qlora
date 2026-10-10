# P130 — do #1583's prefill knobs make the served 512-token prefill faster, and does P1's arithmetic cost quality? `E4B_FUSE_PREFILL_GLUE` (P1) and `E4B_PREFILL_LEAN_DISPATCH` (P2) on the first-chunk prefill graph, and P1 teacher-forced against R's own neutral perturbations, on one RTX 5090 (registered 2026-10-10, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; its no-ask tier covers a single run under $15). Lane number claimed
by `prereg/p130`. **This lane is not TC1's predictions P130–P133** (TC1-PREREG amendment 52, the LoRA delta's padded block
against buckets, read in `bench/h2h-2026-10-02/tc1/RESULTS-tc1-padbk28.md`), which used the same labels. Here "P1" and
"P2" name #1583's two knobs, never predictions.

Follows #1583 (the two knobs, both off by default, and its A2000 smoke), `bench/prefill-glue/DESIGN.md` (the glue map
and the decided option B) and `bench/decode-census/PREFILL.md` (lane 2's prefill census). Phase B is P117's instrument
(`bench/p117/PREREG-p117.md`) by named substitution. The two-phase shape, a speed read and a quality read that must both
hold before arithmetic-changing knobs move, is P115's (`bench/p115/PREREG-p115.md`).

## Why this lane

**The prefill forward is the stall decoders wait behind.** On one RTX 5090, Qwen3-30B-A3B's 512-token prefill forward on
SC2e's int4 stack is 40.3 ms. About a quarter of it is PyTorch eager glue around the fused kernels
(`bench/decode-census/PREFILL.md`, from P119's profile). At concurrency C, every admitted prompt's forward stalls the
decoders behind it, so each millisecond off it is about (C − 1) / 256 ms of TPOT under SC5's load.

**#1583 added two knobs, both off by default, and nothing licenses turning them on yet:**
- **P1, `E4B_FUSE_PREFILL_GLUE=1`.** The three decode folds' hand-back gates let prefill rows (above 64) through to the
  kernels decode already uses: the input and final norms (`rmsnorm_rows`), the residual add and post-attention norm
  (`rmsnorm_resid_rows`), and the q/k norms with the rotary (`rope_norm_qk`). The kernels round once where HF's chain
  rounds twice, so P1 changes bf16 arithmetic (DESIGN.md section 3, option B). Its licence is a teacher-forced prefill
  read, which is Phase B.
- **P2, `E4B_PREFILL_LEAN_DISPATCH=1`.** K19's prefill rows read their token rows themselves (`gather_div=`) and store
  in the caller's order (`scatter=order`), so the 4,096-row gather and the unsort are gone. It is bit-identical by
  construction.

**#1583's smoke** ran on the project's own RTX A2000. It used Qwen3-30B-A3B @ `ad44e777` truncated to 4 of 48 layers,
with correctness only and nothing timed:
- with both knobs off, the PR head was `torch.equal` to its merge base on every logit and hidden state;
- with P2 on, the output was `torch.equal` to P2 off, with `seen.moe_k19_dispatch` reading `chained|lean|gt256` against
  `chained|gather|gt256`. A blindness arm with one swapped scatter pair differed;
- with P1 on, the prefill folds engaged (norm 20, layer 16, attention 16) and every logit was finite.
That smoke ran the plain forward. It ran neither the served prefill graph nor any timing, and it could not read P1's
quality.

**What the glue map bounds** (DESIGN.md section 2, device ms in P119's 512-token profile). These are **ceilings, not
predictions**: the fused kernels' own time is not subtracted.
- P1 absorbs at most **5.47 ms**: the RMSNorm chains 3.72, the rotary 1.61, half the residual adds 0.27.
- P2 absorbs at most **1.34 ms**: the gather of x and the unsort `index_copy_`.
- Together at most **6.80 ms**.

**The questions:**
- Phase A: on the served first-chunk prefill graph, is P2 no slower, and is P1 faster?
- Phase B: is P1's prefill arithmetic at parity with R's own neutral perturbations on a teacher-forced NLL?

## Subject

**The model is SC2e's served stack, built as P117 built it:**
- `PagedServeConfig.from_env()` + `build_engine` with `bench/sc1/sc1_run.sh`'s `SPEEDENV` (int4 experts and int4 RTN
  attention, repacked from the checkpoint at load; the three T=1 folds), `ROUTEENV`, and `E4B_PAGED_FUSE_QKV=1`, byte for
  byte as P117's box ran them;
- built eager with one slot of 768 tokens and no prefill graph (`E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1
  E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0`): the box makes its own runners;
- every other lever unset, including `E4B_INT4_PREFILL` (whose `auto` resolves to K19 under device grouping),
  `E4B_PAGED_BULK_KV` (default on) and `E4B_PAGED_LAST_LOGITS` (default off);
- the NF4 arena baked on the box by P39's `k8_bake.py`.

**Pins:**
- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- **Stack:** e4b at the launch commit, which carries #1583; grouped-nf4-gemm v0.45.0
  (`724ccc454f006c1a46836e434e997f31f293747f`, e4b CI's pin at registration); transformers 5.17.0; huggingface_hub
  ≥ 1.31, for the fetch watchdog.

**Processes.** The knobs act at different times, so the lane runs **eight processes** in a registered order:
- P1 is read at patch time (`glue_fuse.rows_cap`, captured per fold closure), so it is set per process. The order is
  **0 1 1 0 0 1 1 0**: ABBA, twice. Its pairs are (1, 2), (4, 3), (5, 6) and (8, 7), each a P1-unset process and the
  adjacent P1-set one.
- P2 is read on every call, so every process captures one prefill graph with P2 off and one with P2 on, and times
  them interleaved.
- Process 1 also runs Phase B's OFF passes; process 2 runs its subject.

## Phase A — speed (`bench/p130/p130_box.py`, `speed()`)

**The graphs.** In each process, a one-slot runner is made as `build_engine` makes the server's (its `PagedModelRunner`
call, with `bulk_kv` and `last_logits` from the config) with device grouping on. It captures the **first-chunk prefill
graph at T = 512 twice**, through the server's own `enable_prefill_graph`: two warm-up forwards, the capture, then the
startup check (each replay equal to an eager forward bit for bit, in logits and every layer's staged K/V):
- **p2_off** with `E4B_PREFILL_LEAN_DISPATCH=0`;
- **p2_on** with `E4B_PREFILL_LEAN_DISPATCH=1`.

A refused capture is recorded with its reason, and the rule reads it per knob.

**The windows.** P117's wikitext windows (wikitext-2-raw test, joined as K8 joins it, window k from token k·3072). The
first 512 tokens of the first **16** windows are the speed windows.

**FUNCTION (P2), before any timing.** On every speed window both graphs replay, and the `p2_on` replay must be
`torch.equal` to the `p2_off` replay in the logits and in every pool layer's staged K and V. Each window's `p2_off` output
is digested with sha256 for the cross-process DETERMINISM gate.

**Timing.** 2 untimed rounds, then **12 timed rounds**. In each round every speed window replays both graphs, in the
order off-on or on-off alternating with (round + window). Each replay sits between two CUDA events. Per graph and
process: all 192 replay times and their **median**.

**Eager (reported, never gated).** 4 windows through the eager first-chunk forward, which is the path a later chunk
takes: the runner's `_prefill_forward` in its prefill scope, wall ms with a sync, both P2 settings interleaved, 2 rounds.

**Engagement.** Every eager forward is counted. Per capture, the box records the deltas of three counters: the route
(`hot_residency.ROUTE_SEEN`), K19's dispatch (`hot_residency.K19_DISPATCH_SEEN`) and the prefill folds
(`glue_fuse.PREFILL_FOLD_SEEN`). A replay runs no Python, so it counts nothing. With F forwards per capture (five) and
48 layers, each capture must show:
- route `int4_k19|gt256` × 48F, and no other route;
- K19 `chained|gather|gt256` × 48F on `p2_off`, or `chained|lean|gt256` × 48F on `p2_on`, and nothing else;
- prefill folds `{norm: 49F, layer: 48F, attention: 48F}` with P1 set, and none with P1 unset. Per forward that is the
  48 input norms plus the final norm, one residual-and-norm fold per layer, and one q/k-norm-and-rotary fold per layer.

## Phase B — quality, P1 only (`p130_box.py`, `quality_off()` / `quality_on()`; P117's instrument)

**P117's instrument by named substitution:**
- the subjects W32, W64 and W64pad become one subject, **P1**: R's arithmetic on the P1 model;
- G64 is dropped (no decode graph is under test);
- mutant_scale runs on R's buckets;
- the passes are split over two processes, as P115's quality phase did, because P1 is read at patch time.

**The passes** are `p117_box.paged_pass` at P117's registered bytes. Each runs over **64 wikitext windows decoded
together** (P117's windows), with a fresh `Fp8PagedKV` and `PagedModelRunner`, device-grouped, and every decode step a
padded eager step of its bucket (`capture=False`). Each window has **512 prompt tokens**, prefilled in one chunk, and
**128 teacher-forced positions**. Scoring is on fp32 log-probs: the true token's NLL, the argmax, and KL against R. P2
stays unset in every pass.

| arm | process | buckets | what it is |
|---|---|---|---|
| R | 1 (P1 unset) | 1–16 | the reference, P117's R; its fp32 log-probs are saved on the box with a digest per window |
| rep | 1 | 1–16 | R's repeatability; a floor draw if not bit-identical |
| half | 1 | 1–8 | floor: a different row count |
| chunk | 1 | 1–16, prompts prefilled in 256-token chunks | floor: a different prefill split |
| rev | 1 | 1–16, windows bound and decoded in reverse | floor: a different row order |
| mutant_scale | 1 | 1–16, decode softmax scale halved | must fail the bar (P108's mutant) |
| **P1** | 2 (P1 set) | 1–16 | **the subject**, scored against R's saved log-probs (each file's digest checked on load) |

P1 changes only the prefill's arithmetic, since decode rows here are at most 16. So the subject differs from R through
two things: the K/V its prompt wrote, and its first scored position, which comes from the prefill's own last logits.

**Engagement, per pass:**
- P117's, unchanged: decode attention calls = 127 × 48 × pieces per step; device grouping on; every bucket
  `eager: capture=False`; the bucket statistics as the registered split;
- route `int4_k19|gt256` = 64 × prefill chunks per window × 48;
- prefill folds none on every OFF pass, and `{norm: 64 × 49, layer: 64 × 48, attention: 64 × 48}` on P1.

## The rule (`bench/p130/p130_reduce.py`, self-tested on 40 cases)

**Lane VOID**, any of:
- a registered process record missing; another e4b or grouped-nf4-gemm commit; another model revision;
- a process whose P1 setting, in its record or in glue round 1's fold report (`prefill`), is not the registered one; a
  process off its registered Phase B role;
- the processes read different windows, or sizes other than the mode's;
- wrong speed engagement on a captured graph, as registered above: T, five forwards per capture, the three deltas;
  FUNCTION or digests over other than every speed window; other than 12 × 16 timed replays per captured graph;
- the default's graph refused, that is a P1-unset process's `p2_off` capture.

**Per knob, the first of these that applies:**
1. **REFUSED.** A capture with the knob on was refused: P2 by any `p2_on` capture, P1 by a P1-set process's `p2_off`
   capture. The server's `auto` prefill graph would stand down under the knob, so it is a defect. A process whose
   `p2_on` capture refused still times its `p2_off` graph.
2. **FUNCTION_FAIL** (P2 only). On any speed window of any process, the `p2_on` replay is not `torch.equal` to the
   `p2_off` replay.
3. **HELD_DETERMINISM.** Two processes with the same P1 setting digest a speed window's `p2_off` output differently.
   Phase B scores the subject from another process against R, so this premise failing holds both knobs.
4. **NOISY.** Within either P1 setting, the processes' `p2_off` medians span more than **×1.02**.
5. **P2, the equivalence read.** Per process, r = median(`p2_on`) / median(`p2_off`). The method is a two-sided 95 %
   t-interval of the mean of ln r over the eight processes (7 degrees of freedom, t = 2.3646), in ln space and reported
   back as ratios.
   - **The margin: ×1.01.**
   - **P2: LICENSED** iff the interval's upper end is at most ln 1.01. P2 is then no slower, at a one-sided level of
     97.5 %. Otherwise **HELD_NOT_NO_SLOWER**, whether the point estimate is slower or the interval is too wide.
6. **P1, speed and quality.**
   - **Speed.** Per registered pair, g = median_unset(`p2_off`) / median_set(`p2_off`). FASTER iff the lower end of the
     two-sided 95 % t-interval of the mean of ln g over the four pairs (3 degrees of freedom, t = 3.1824) is at least
     **ln 1.03**. A change to bf16 arithmetic should buy a clear gain; 1.03 is under half the gain predicted below.
   - **Quality (Phase B).** P117's rule, unchanged. Per window w, d_X(w) is X's mean continuation NLL minus R's. The floor
     is half, chunk and rev, plus rep if R did not repeat bit for bit. B_floor is the largest |mean d_f| and S_floor the
     largest mean |d_f|. passes(X) iff mean d_X ≤ B_floor + 0.01 nats and mean |d_X| ≤ 2 × max(S_floor, 0.005).
     - **QUALITY_VOID** on any integrity fault: the engagement above, the window counts, a Phase B process that raised,
       R's saved log-probs failing their digests, or a mutant_scale that passes.
     - **AT_PARITY** iff P1 passes; **COST** otherwise.
   - **P1: LICENSED** iff FASTER and AT_PARITY. Otherwise **HELD**, naming NOT_FASTER and/or COST, or QUALITY_VOID.

A Phase B COST holds P1 whatever Phase A shows.

**Reported, never gated:**
- every ratio and its interval, and every process's medians;
- the savings in ms, set beside the glue map's ceilings;
- the eager forward's medians;
- every Phase B arm's bias, spread, SE, max |d|, mean KL against R and argmax agreement;
- R's repeatability;
- the predictions below, scored mechanically.

## Predictions (written before any data)

The saving is ms per 512-token first-chunk prefill replay. The ceilings are the glue map's and bound these figures. A
prediction is HELD inside its range and MISSED outside it.

| # | prediction | basis |
|---|---|---|
| Q1 | engagement exact in every process and pass; no capture refused | #1583's smoke engaged every fold and K19's lean dispatch eagerly; the census found no host sync in the prefill forward (`bench/prefill-graph-census-2026-10-04`), and P2 adds only device index tensors |
| Q2 | FUNCTION holds on every window of every process, and DETERMINISM holds | P2 is bitwise by construction (the smoke read it on sm_86); separate processes of one build decode bit for bit alike (P115's F0b and F0a, P111's S0b and S0a) |
| Q3 | **P1 saves 2.5–4.5 ms** (ceiling 5.47): g ≈ 1.06–1.12 on a ~41 ms forward | the map's 5.47 ms less the fused kernels' own time. 145 launches (49 `rmsnorm_rows`, 48 `rmsnorm_resid_rows`, 48 `rope_norm_qk`) each move 2–9 MB, about 1–2.5 ms at the card's bandwidth. `rope_norm_qk`'s grid of (512, 36) small programs is the uncertain part |
| Q4 | **P2 saves 0.5–1.3 ms** (ceiling 1.34): r ≈ 0.97–0.99 | the gather and unsort go; K19's indirect row loads and stores cost some of it back |
| Q5 | **P1 and P2 together save 3.0–5.5 ms** (ceiling 6.80) | Q3 + Q4 (reported per pair: unset `p2_off` against set `p2_on`) |
| Q6 | NOISY does not fire; within each P1 setting the medians span ≤ ×1.01 | graph replays of one shape (P111's self-pairs 0.997–1.002) |
| Q7 | Phase B: P1's bias in [−0.004, +0.004] nats and spread ≤ 0.016; R repeats bit for bit; the floor's B ≤ 0.003 and S in [0.008, 0.020]; mutant_scale's bias > +0.3 | P117's floor read B 0.0025 and S 0.0148 on this stack; P1's single rounding sits inside the arithmetic class decode already serves (P115) |
| Q8 | the verdict is **P1_LICENSED P2_LICENSED** | — |
| Q9 | each process ≤ 4 min; the eight ≤ 35 min; peak device memory ≤ 28 GiB | P117's build 91 s and its ten passes 227 s; two prefill-graph pools of about 3.3 GiB each (SC2b) over P117's 21.3 GB |

**What a MISS means (registered now):**
- **Q3 low.** The fused kernels cost more at 512 rows than modelled. If P1 still clears ×1.03 it is licensed on its
  interval anyway; the read names the kernel to look at (likely `rope_norm_qk`'s grid), a grouped-nf4-gemm item.
- **Q3 or Q5 high, above the ceiling.** The glue map left something out (for example scheduling gaps between the removed
  launches under replay). The read says so and does not lean on the map's attribution again without a profile.
- **Q4 low.** K19's indirect loads eat the gather's saving. P2's licence is the equivalence margin, not the saving, so it
  can be licensed while Q4 is MISSED; its PR then claims "no slower", not a gain.
- **Q7 outside the range but inside the bar.** AT_PARITY still holds, and the bias is reported.

## Consequence, registered now

Each flip is its own small PR after the read, citing it, and each states its scope:
- **P2: LICENSED.** `E4B_PREFILL_LEAN_DISPATCH` defaults to `1`, and `0` restores the gather.
  - `docs/SERVING.md`, `docs/STATUS.md` and the changelog say so.
  - Register row `e4b.serve.p130.prefill-lean-dispatch.qwen3.5090.<date>` with the ratio and interval from
    `verdict.json`.
  - **Scope:** the int4 K19 prefill route, read on Qwen3-30B-A3B. Other families on that route ride its bitwise
    contract (the kernel and tile table are the same; FUNCTION is per shape), and the PR says so.
- **P1: LICENSED.** `E4B_FUSE_PREFILL_GLUE` defaults to `1` on `qwen3_moe` only, the family read here; `0` restores HF's
  prefill chain.
  - Register rows `e4b.serve.p130.prefill-glue-speed.qwen3.5090.<date>` (the gain and its interval) and
    `e4b.serve.p130.prefill-glue-quality.qwen3.5090.<date>` (P1's bias).
  - Other families keep `0` until a read of their own. The PR lists, per family, what was read (P115's pattern).
- **HELD (NOT_FASTER):** P1 stays opt-in; the ratios are recorded.
- **HELD (COST):** P1 stays opt-in. DESIGN.md's option A (HF-order kernels: `rope_train`, bitwise on the rotary, and
  `rmsnorm_train`) becomes the fallback lane.
- **HELD_NOT_NO_SLOWER:** P2 stays opt-in; the ratios are recorded.
- **REFUSED or FUNCTION_FAIL:** a defect to fix before the knob moves; it stays opt-in.
- **HELD_DETERMINISM:** a defect to find before anything moves.
- **NOISY, VOID or QUALITY_VOID:** no consequence. A rerun is a new attempt inside the ceiling, or an amendment if the
  cause is the harness.

The speed claims in the flip PRs and the docs are the read's own receipt wording: the ratios, the intervals and the
host. They give no TPOT or TTFT figure under load, which is an SC lane's to read.

## The premise and the proving rental

**The premise** runs on the card before anything is fetched (rc 25). It is `tests/test_prefill_graph_gpu.py`: the
first-chunk prefill graph's capture, startup check, replay and refusals. **9 passed**, none skipped.

**The proof** runs the whole box end to end on Qwen3-30B-A3B itself, at the proof's sizes:
- four processes (P1 0 1 1 0);
- 4 speed windows and 3 timed rounds after 1 warm-up;
- Phase B over 16 windows of 512 prompt tokens and 32 positions.

No other family takes the int4 K19 prefill route that P2 changes, and no local card runs the fp8 paged KV. It is
**PROVED** iff the lane exits 0 with a verdict other than VOID or NO_READING and Phase B is not QUALITY_VOID. The proof's
verdict is not a reading. A REFUSED in the proof goes to the maintainer before any reading.

## Budget, sequencing and STOP rules

**Pricing.** The launcher's policy rate is at most $0.85/h, plus about $0.67 to download the 61 GB checkpoint (P127's
pricing).
- **Proof:** one RTX 5090 (Vast verified/secure), **guard 1.25 h**, about $1.73. It is mostly the fetch and the bake.
- **Reading:** one RTX 5090, **guard 2.0 h**, about $2.37. Expected about 75 minutes: install 10, fetch 15, bake 15,
  eight processes about 25, Phase B about 10.
- **Lane ceiling:** **$5.50**, covering the proof and the reading (about $4.10) with room for one pre-flight retry.

**Sequencing.** Nothing rents until this registration merges and the maintainer explicitly ACKs it, and not before SD2's
`sd2-5090-1` read has torn down: one rental lane at a time. Then, on the controller, **before the rental controller
quotes**, three gates run in order:
1. the driver's dry run (`P130_DRIVE_DRYRUN=1`);
2. **the pre-rental fetch gate** (STOP-0);
3. the time-left fit (STOP-2).

The reading rents only after the proof's receipts show PROVED, and the fetch gate runs again for its launch.

**STOP rules:**
- **STOP-0, the pre-rental fetch gate** (`bench/p130/p130_fetch_gate.py`; DQ11's launch-gate pattern,
  `bench/dq11/DQ11-AMENDMENT-3.md`). Every locked thing the box will fetch is resolved from a CPU host with the box's own
  clients (huggingface_hub in the box's range ≥ 1.31, < 2; git; pip). It refuses on any missing, unauthorized (401/403)
  or mismatched entry, and it never installs, downloads a whole file, quotes or rents:
  - **the checkpoint:** the revision's file list, with every size, blob id and LFS sha256, from the hub's metadata. Every
    file `p130_run.sh`'s allow-list selects must be listed, including the safetensors index and every shard it names. A
    metadata HEAD on each file's resolve URL must answer with the revision's commit, the listed size and the listed
    hash as its etag;
  - **the windows' corpus** (wikitext-2-raw-v1): its README and every file under the config resolve the same way at
    `main`, whose commit is recorded. P117's box does not pin the corpus; the reducer's windows digest carries it;
  - **e4b at the launch commit and grouped-nf4-gemm at its pin:** `git fetch --depth 1` of each by SHA;
  - **every PyPI requirement `p130_run.sh` installs:** pip's own resolver (`install --dry-run --report`, wheels only, no
    dependencies) for the box (CPython 3.11, manylinux x86_64) picks each wheel, which must satisfy its pin and answer a
    one-byte range GET. Their dependencies resolve on the box, as every lane's do.

  There is no other asset: the staged files travel from the controller under `staged.sha256`. The report goes to the
  launch as `P130_FETCH_GATE`. `p130_drive.sh` refuses (78) before any connection unless the report passed, is for the
  launch commit and is under 24 h old, and it copies the report into the run directory. A gate refusal is final for that
  launch: no retry, no alternative URL.
- **STOP-1:** the refusals run before any install: the host floor (torch cannot use the GPU: 18; P115 Amendment 1),
  dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB (16), premise (25).
- **STOP-2:** every time-left check (fetch, bake, each process) fits inside its guard, enforced by
  `tests/test_p130_staged_pin.py`. A process that cannot start 10 minutes before the deadline is skipped, and the reducer
  VOIDs.
- **STOP-3:** a VOID, NOISY or QUALITY_VOID reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p130/staged.sha256`.
- **The fetch on the box** runs under P127's byte-growth watchdog (`bench/common/hf_fetch_watchdog.py`, staged as a byte
  copy, its self-test run before the fetch). It is killed and restarted after 180 s without growth, at most 3 restarts.
  This is in addition to STOP-0, not instead of it.

## What was seen before this page (stated, not hidden)

- **No P130 data exists.**
- The A2000 smoke's results are #1583's, quoted above. They were correctness only, on a 4-layer truncation, and nothing
  was timed.
- The glue map's figures and the 40.3 ms forward are from P119's committed profile, via DESIGN.md and PREFILL.md.
- **On CPU:** the reducer's self-test (40 cases) passes. `tests/test_p130_box.py` runs both phases' bookkeeping with the
  runners stood in, and passes with grouped-nf4-gemm at the pinned commit. It covers the per-capture deltas, FUNCTION
  catching a K/V-only difference, a refused capture, R's saved log-probs and a corrupted one caught on load.
- **The fetch gate's pre-review CPU pass** (2026-10-10, against #1583's merge `7b451ab9`, from the controller with
  huggingface_hub 1.33.0) is `bench/p130/fetch-gate-prereview.json`. It PASSED:
  - the checkpoint: 23 files, 61.1 GB, every size and hash matched at `ad44e777`;
  - the corpus: resolved at `main` `b08601e0`;
  - both commits fetched;
  - eight wheels picked for the box, each answering 206 with one byte.

  It is not the launch's gate, which must pass again on the launch commit.

## What this lane cannot say

- Nothing about other families' speed or quality: P1's flip is scoped to `qwen3_moe`.
- Nothing about later prefill chunks beyond the eager medians, which are reported only.
- Nothing about other prompt lengths, other cards, or TTFT and TPOT under load.
- Nothing about P3 (the router epilogue above 64 rows), which is out of this round (DESIGN.md).
- Nothing about any P1 arithmetic difference smaller than R's own neutral perturbations: by construction, that is the
  floor.

## Receipts

Fetched to the run directory's `p130/` and committed to `bench/p130/receipts/<run>/`:
- `proc_1.json` … `proc_8.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- `p130_fetch_gate.json`, the launch's STOP-0 report;
- the logs, and the teardown proof;
- `SHA256SUMS`.

R's log-probs (`work/ref`, about 5 GB) stay on the box. `RESULTS-p130.md` is written from those files.

## Amendment 1 (2026-10-10, in review of the registration, before any box): the corpus commit the box read

The fetch gate records the commit `main` names for the windows' corpus (wikitext-2-raw-v1). P117's box does not pin
that corpus, so a drift between the gate and the run would otherwise go unseen. This amendment makes it visible:
- **The box** (`corpus_commits()`) records, in every process record, the commit or commits it actually read. `datasets`
  keys its arrow cache of the config by the hub commit it resolved (`Salesforce___wikitext/wikitext-2-raw-v1/<version>/<commit>`),
  and huggingface_hub keys the files' snapshot by the same commit.
- **The driver** passes the gate report's `corpus.main` to the box as `P130_GATE_CORPUS`.
- **The runner** writes it to `summary.txt` (`GATE_CORPUS`) and hands it to the reducer.
- **The reducer** reports `corpus: {gate, box_arrow, box_snapshots, same}` in `verdict.json`.

It is **reported, never gated**: the windows digest already gates that every process read the same windows. A drift is
stated in the read. Its self-test grows from 40 to 42 cases.

Nothing else changes: no arm, gate, bar, prediction, size, budget or sequencing rule. `staged.sha256` is re-pinned for
the runner, the box and the reducer.
