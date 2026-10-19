# P131: does K19 over a split tile table make the served 512-token prefill faster, and is it the same arithmetic? `E4B_K19_PREFILL_BIG_M` at 0, 32 and 64 on the first-chunk prefill graph, with a FUNCTION gate that picks the quality branch, on one RTX 5090 (registered 2026-10-10, before any run)

Issue: experts4bit-qlora#846, the serving campaign. Its no-ask tier covers a single run under $15. The lane number is
claimed by `prereg/p131`. Here "m0", "m32" and "m64" name the knob's three settings, never predictions.

Sources:
- **Code.** e4b#1625 adds the knob, opt-in. grouped-nf4-gemm#542 adds the split builder and K19's `block_m`; both are
  opt-in, and the A2000 read them bit-identical (sm_86, correctness only).
- **Basis.** A $0 shape census of the prefill expert GEMM on SC2's prompts (#846, the row beside SC2f).
- **Instrument.** P130's (`bench/p130/PREREG-p130.md`), by named substitution: its Phase A for a knob read on every
  call, which is how P130 read P2, and its Phase B (P117's teacher-forced instrument) as the tolerance branch.
  `p131_box.py` imports P130's box for the counters, digests, timers, runner and scoring.

## Why this lane

**K19 is half the prefill forward.** In P119's 512-token profile on one RTX 5090, K19 (`_gemm_int4_b32_grouped_smallm`)
is 19.90–19.95 ms of device time over 96 launches (48 layers × gate_up and down). The chained 16-row tile builder around
it is a further 3.64 ms over 2,021 launches (`bench/p119/receipts/p119-5090-1/box.json`; `bench/prefill-glue/DESIGN.md`).
P130 brought the first-chunk replay to 32.99 ms. Every millisecond off it is about (C − 1) / 256 ms of TPOT under SC5's
load.

**The prefill rows are bimodal.** The census replayed Qwen3-30B-A3B bf16 layer by layer on SC2's 64 prompts and checked
the replay against transformers' top-8 on a 4-layer truncation (agreement 1.0). Per 512-token chunk, 4,096 routed rows:
- **one expert takes a median 398 rows** (p10 254, max 509). In many layers one expert sits in the top-8 of 50–94 % of
  all tokens;
- the other ~90 active experts take a median 24 rows, and 30 % of active experts get 8 or fewer.

Each 16-row tile re-reads and re-dequantises its expert's `BLOCK_N × K` weight slice, so on the 16-row table the heavy
expert pays that about 25 times per N block.

**The split** (`E4B_K19_PREFILL_BIG_M`) keeps the 16-row table's padding exactly:
- each expert's full `big_m`-row tiles run in one launch at `block_m = big_m`;
- the 16-row remainders run in a second launch, into the same result, over the same `order`;
- weight passes per N block fall from **307 to 204** at 64 and to 212 at 32. Uniform 64-row tiles would take 124 passes
  but pad 48 %, against 16 %.

**The knob reaches only prefill calls above 512 routed rows** (e4b#1625's `K19_SPLIT_ROWS_MIN`). A 64-row decode step
routes exactly 512 rows on a top-8 model, so decode keeps its table. This lane asserts that on the card (the decode
check, below).

**The question:** on the served first-chunk prefill graph, is either tall tile faster, and is its arithmetic the 16-row
table's?

## Subject

**The model is SC2e's served stack, built as P117 and P130 built it, at the release's prefill defaults:**
- `PagedServeConfig.from_env()` + `build_engine` with `bench/sc1/sc1_run.sh`'s `SPEEDENV`, `ROUTEENV` and
  `E4B_PAGED_FUSE_QKV=1`, byte for byte as P130's box ran them;
- built eager with one slot of 768 tokens and no prefill graph: the box makes its own runners;
- `E4B_FUSE_PREFILL_GLUE` and `E4B_PREFILL_LEAN_DISPATCH` **unset**, so both of P130's defaults are in force: P1 on for
  `qwen3_moe` (e4b#1615) and K19's lean dispatch on prefill rows (e4b#1608);
- `E4B_K19_PREFILL_BIG_M` unset in the environment. The box refuses to start otherwise, and sets the knob per capture
  and per Phase B pass itself.

**Pins:**
- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- **e4b:** the launch commit. It must carry #1608, #1615 (e4b 0.54.0) and #1625.
- **grouped-nf4-gemm:** #542's merge commit `57f53e19fe65eeef4b7ed26f97d7bd8fdd87f0eb`. #542's own files there are
  byte-identical to its reviewed head `4af1b524`; the merge also carries #540 (ROCm capability gates, touching
  `fp8_kv`, `fp8_paged_attn` and `nf4_route`) and #543 (GGUF backward), which landed on main first.
- **Libraries:** transformers 5.17.0; huggingface_hub ≥ 1.31.

**Processes: eight, identical.** The knob is read on every call, so every process captures all three arms and times them
interleaved; no process differs by a setting. **Process 1 also runs Phase B.**

## Phase A — speed (`bench/p131/p131_box.py`, `speed()`)

**The graphs.** In each process, a one-slot runner made as `build_engine` makes the server's captures the **first-chunk
prefill graph at T = 512 three times**, through the server's own `enable_prefill_graph`: warm-up, capture, then the
startup check (bit for bit against an eager forward, in logits and every layer's staged K/V). The captures are:
- **m0**, with the knob `0`: the 16-row table, the control;
- **m32** and **m64**, with the knob `32` and `64`.

**The capture order rotates by process:** m0, m32, m64 rotated by (process − 1), so over the eight processes every arm is
captured first, second and third. A capture-position effect (pool placement, alignment, the cache residency of the
graph's buffers) is then spread over the arms instead of riding one of them; at m32's predicted low end (×1.033 against
the ×1.02 bar) a 1 % placement bias would otherwise decide the verdict. Each process records its order, and the rule
VOIDs a process off it.

A refused capture is recorded with its reason.

**The windows.** P117's wikitext windows. The first 512 tokens of the first **16** windows are the speed windows.

**FUNCTION, before any timing.** On every speed window all three graphs replay. Each tall arm's logits and every pool
layer's staged K and V are compared with m0's by `torch.equal`, and each window's m0 output is digested (sha256) for
DETERMINISM.

**Timing.** 2 untimed rounds, then **12 timed rounds**. In each round every speed window replays the three graphs in the
order m0, m32, m64 rotated by (round + window), each replay between two CUDA events. Per graph and process the box keeps
all 192 replay times and their **median**.

**The decode check.** Per arm, one eager forward of a **64-token** prompt with the knob set. It routes 512 rows, the rows a
64-row decode step routes. Under every arm it must show no `split` table and the same K19 calls above 256 rows as m0's
in the same process; m0's own must be one `lean` call a layer with no split. Which table m0 takes is
`E4B_INT4_WIDE_TILES`'s choice, not this lane's: on Qwen3-30B, 128 experts × 512 rows is exactly lane P122's `auto` bound,
so a 64-row decode step takes the one-launch table (`one_launch|lean|gt256`). #1625's A2000 smoke showed the same at
1,024 rows on 4 experts.

**Eager (reported, never gated).** 4 windows through the eager first-chunk forward, the three arms rotated, 2 rounds.

**Engagement.** With F = 5 forwards per capture and 48 layers, each capture must show:
- route `int4_k19|gt256` × 48F, and no other route;
- K19 `chained|lean|gt256` × 48F on m0, or `split|lean|gt256` × 48F on m32 and m64, and nothing else;
- prefill folds `{norm: 49F, layer: 48F, attention: 48F}`: P1's default is on.

## Phase B — the tolerance branch's instrument (`quality()`, process 1)

P130's Phase B by named substitution, in **one process** because the knob is read per call. The passes are
`p117_box.paged_pass` at P117's registered bytes. Each runs over **64 windows decoded together**, with 512 prompt tokens
prefilled in one chunk and 128 teacher-forced positions, every decode step a padded eager step of its bucket. Scoring is
on fp32 log-probs (NLL, argmax, KL against R).

| arm | knob | buckets | what it is |
|---|---|---|---|
| R | 0 | 1–16 | the reference; its log-probs are held in memory |
| rep | 0 | 1–16 | R's repeatability; a floor draw if not bit-identical |
| half | 0 | 1–8 | floor: a different row count |
| chunk | 0 | 1–16, prompts in 256-token chunks | floor: a different prefill split |
| rev | 0 | 1–16, windows in reverse | floor: a different row order |
| mutant_scale | 0 | 1–16, decode softmax scale halved | must fail the bar |
| **S32** | 32 | 1–16 | subject: R on the m32 table |
| **S64** | 64 | 1–16 | subject: R on the m64 table |

Decode rows here are at most 16 (128 routed rows), so the knob reaches only the prefill. Each subject also records
whether all its windows' log-probs are `torch.equal` to R's.

**Engagement per pass:**
- P117's, unchanged;
- route `int4_k19|gt256` = 64 × chunks × 48;
- K19's dispatch above 256 rows: `split|lean|gt256` on S32 and S64, `chained|lean|gt256` on every other arm, each
  64 × chunks × 48;
- prefill folds 64 × chunks × the fold table.

Phase B always runs, so the tolerance branch never waits on a second rental. The rule reads it only for an arm whose
FUNCTION is not bitwise.

## The rule (`bench/p131/p131_reduce.py`, self-tested on 37 cases)

**Lane VOID**, any of:
- a registered process record missing; another e4b or grouped-nf4-gemm commit; another model revision; a process off its
  Phase B role;
- a process whose build did not resolve P1's release default (`prefill_glue` ≠ {mode 1, default-allowlisted});
- the processes read different windows, or sizes other than the mode's;
- wrong engagement on a captured graph (above), a capture order off the rotation, the decode check missing, other than
  12 × 16 timed replays per graph, or FUNCTION or digests over other than every speed window;
- the control's (m0) graph refused.

**Per tall arm, the first of these that applies:**
1. **REFUSED.** A capture of the arm refused in any process. The server's prefill graph would stand down: a defect.
2. **DECODE_TOUCHED.** The decode check under the arm shows a `split` table, or K19 calls above 256 rows other than m0's,
   in any process. The knob reaches decode: a defect.
3. **HELD_DETERMINISM.** Two processes digest a speed window's m0 replay differently.
4. **NOISY.** The processes' m0 medians span more than ×1.02.
5. **Speed.** Per process, g = median(m0) / median(arm). The two-sided 95 % t-interval of the mean of ln g over the eight
   processes (7 degrees of freedom, t = 2.3646). **FASTER** iff its lower end is at least **ln 1.02**: under half the
   gain the model predicts at its lower f.
6. **Quality: FUNCTION picks the branch.**
   - **BITWISE** iff the arm's replay is `torch.equal` to m0's on every speed window of every process, in the logits and
     every layer's K and V. The arm is then the 16-row table's arithmetic, and no quality read is needed.
   - **Otherwise the TOLERANCE branch:** Phase B's subject for the arm, by P117's rule unchanged.
     - Per window w, d_X(w) is X's mean continuation NLL minus R's.
     - B_floor is the largest |mean d_f| over the floor (half, chunk, rev, plus rep if R did not repeat), and S_floor
       the largest mean |d_f|.
     - X passes iff mean d_X ≤ B_floor + 0.01 nats and mean |d_X| ≤ 2 × max(S_floor, 0.005).
     - **QUALITY_VOID** on any integrity fault, or a mutant_scale that passes. **AT_PARITY** iff the subject passes;
       **COST** otherwise.
7. **LICENSED** iff FASTER and (BITWISE or AT_PARITY). Otherwise **HELD**, naming NOT_FASTER and/or COST, or
   QUALITY_VOID.

**The default arm.**
- One arm LICENSED: that arm.
- Both LICENSED: the paired per-process interval of ln(median(m32) / median(m64)) decides. **m64** iff its lower end is
  above 0; **m32** iff its upper end is below 0.
- Both LICENSED but the interval holds 0: **m32**, the shorter tall tile, with less register pressure on other cards.
- Neither LICENSED: **NONE**.

The verdict reads `M32_<v> M64_<v> DEFAULT_<arm>`.

**Reported, never gated:**
- every ratio and interval, and every process's medians;
- the savings in ms;
- the m32-against-m64 ordering;
- the eager medians;
- Phase B's bias, spread, SE, max |d|, KL and argmax agreement for every arm, and whether each subject's log-probs were
  `torch.equal` to R's;
- the predictions below, scored mechanically.

## The cost model and the predictions (written before any data)

**The model.**
- **The cost.** With uniform 16-row tiles the cost per tile is a weight pass plus 16 rows of MMA, and no existing read
  separates the two (padded rows are always 16 × tiles). So the saving depends on **f, the weight-pass share of a 16-row
  tile's cost**. Over the census, the split saves 0.333 · f of K19 at 64 and 0.308 · f at 32; its padding is unchanged.
- **The basis for f.** P124's K16 read (`bench/p124`) ran a 64-row tile at 1.70× a 32-row tile (1.38 against 0.81 ms a
  step). Fitting cost = a + b·M gives a = 13.7 b, so f ≈ **0.46** for a 16-row tile. That is a different kernel
  (split-K dense K16) at attention shapes: it is stated as a basis, not a measurement, and the predictions span
  **f ∈ [0.3, 0.8]**.
- **The builder, measured.** The split builds both tables in one stacked pass: 48 ops against the chained builder's 39
  (CPU profiler, R 4,096, E 128, top-level aten ops). At P119's 1.8 µs a launch that costs about **0.8 ms** a forward.
- **The net** is the saving on K19's 19.9 ms less the builder's 0.8 ms, against m0 at about 33.0 ms (P130's both-on
  replay): m64 saves 1.2–4.5 ms (2.3 at f = 0.46), m32 saves 1.1–4.1 ms (2.05).
- **The order.** The m64-over-m32 gap is 0.025 · f · 19.9 ms, 0.15–0.4 ms. That is above P130's paired noise (about
  0.07 ms), so both arms register.
- **A risk the model does not see.** Register pressure at BLOCK_M 64 × KC 256 could make m64 slower than the model
  says, and invert the order. The read's ordering is itself a finding.

A prediction is HELD inside its range and MISSED outside it.

| # | prediction | basis |
|---|---|---|
| Q1 | engagement exact in every process and pass; no capture refused; the decode check untouched under every arm (m0's table, no split) | e4b#1625's CPU tests; the split builder is capture-legal by construction (on-device ops, static shapes, no host sync) |
| Q2 | FUNCTION **bitwise** for both arms on every window of every process, and DETERMINISM holds | a row's reduction order does not depend on the tile's other rows; the A2000 read the split `torch.equal` to the 16-row table at K 768 and 2048 (sm_86). The card's codegen at sm_120 is what this reads |
| Q3 | **m64: g ∈ [1.038, 1.159]**, saving 1.2–4.5 ms | the net above, f ∈ [0.3, 0.8] |
| Q4 | **m32: g ∈ [1.033, 1.143]**, saving 1.1–4.1 ms | the same, at 32 |
| Q5 | **m64 over m32: ×1.004–1.013** (m64 faster) | the gap above |
| Q6 | NOISY does not fire; the m0 medians span ≤ ×1.01 | P130's processes |
| Q7 | Phase B: R repeats bit for bit; S32's and S64's log-probs are `torch.equal` to R's | Q2 at Phase B's prefill (one chunk of 512 tokens per window) |
| Q8 | the verdict is **M32_LICENSED M64_LICENSED DEFAULT_m64** | Q2–Q5 |
| Q9 | each process ≤ 4 min except process 1 (≤ 9 min, Phase B's eight passes); the eight ≤ 40 min; peak ≤ 27 GiB | P130's processes took about 2 min each, plus one prefill-graph pool of about 3.3 GiB over P130's 21.3 GB |

**What a MISS means (registered now):**
- **Q2 not bitwise.** The card's codegen differs with BLOCK_M on sm_120, and the arm takes the tolerance branch. The flip
  PR, if any, then claims parity on Qwen3-30B-A3B only and lists the families it does not cover, as P130's P1 did.
- **Q3 or Q4 low.** f is below 0.3, or register pressure costs the tall tiles. If an arm still clears ×1.02 it is
  licensed on its interval. The read names the plan (`block_n`, `kc`, warps) to look at, a grouped-nf4-gemm item.
- **Q3 or Q4 high.** f is above 0.8: K19 is more weight-bound than P124's kernel suggested. The read says so and
  re-fits the model before the next tile lane leans on it.
- **Q5 inverted.** Register pressure or occupancy at 64 rows; m32 becomes the default by the rule. The plan sweep for the
  tall launch is the next item.
- **Q1's decode check fails.** The knob reaches decode: a defect in e4b#1625's threshold, fixed before anything moves.

## Consequence, registered now

The flip is its own small PR after the read, citing it and stating its scope:
- **DEFAULT_m32 or DEFAULT_m64.**
  - `E4B_K19_PREFILL_BIG_M` defaults to `auto`: the chosen tile wherever the installed grouped-nf4-gemm carries the
    split, and the 16-row table otherwise. `0` restores the 16-row table.
  - `docs/SERVING.md`, `docs/STATUS.md` and the changelog say so.
  - Register row `e4b.serve.p131.k19-prefill-split.qwen3.5090.<date>`, with the ratio and interval from `verdict.json`.
  - **Scope:** the int4 K19 prefill route, read on Qwen3-30B-A3B (top-8). If FUNCTION read BITWISE, other top-8 families
    on that route ride its bitwise contract. If the arm was licensed on the tolerance branch, only `qwen3_moe` moves.
- **DEFAULT_NONE.** The knob stays opt-in; the ratios are recorded.
  - NOT_FASTER: the weight-pass model is wrong for K19; the read says what the census and the timings imply instead.
  - COST: the tall tiles change arithmetic beyond R's own perturbations.
- **REFUSED or DECODE_TOUCHED:** a defect to fix before the knob moves.
- **HELD_DETERMINISM, NOISY, VOID or QUALITY_VOID:** no consequence. A rerun is a new attempt inside the ceiling, or an
  amendment if the cause is the harness.

The speed claims are the read's own receipt wording: the ratios, the intervals and the host. No TTFT or TPOT figure under
load is claimed; that is an SC lane's to read (SC2f's successor).

## The premise and the proving rental

**The premise** runs on the card before anything is fetched (rc 25), as P130's did: `tests/test_prefill_graph_gpu.py`,
**9 passed**, none skipped.

**The proof** runs the whole box end to end on Qwen3-30B-A3B itself, at the proof's sizes:
- four processes;
- 4 speed windows and 3 timed rounds after 1 warm-up;
- Phase B over 16 windows of 512 prompt tokens and 32 positions.

No other model takes the int4 K19 prefill route above 512 rows. It is **PROVED** iff the lane exits 0 with a verdict
other than VOID or NO_READING and Phase B ran without an integrity fault. The proof's verdict is not a reading. A REFUSED
or DECODE_TOUCHED in the proof goes to the maintainer before any reading.

## Budget, sequencing and STOP rules

**Pricing** is P130's: at most $0.85/h, plus about $0.67 to download the 61 GB checkpoint.
- **Proof:** one RTX 5090, **guard 1.25 h**, about $1.73.
- **Reading:** one RTX 5090, **guard 2.0 h**, about $2.37. Expected about 85 minutes: install 10, fetch 15, bake 15,
  eight processes about 35, Phase B inside process 1.
- **Lane ceiling: $5.50.**

**Sequencing.**
- Nothing rents until this registration, #1625 and grouped-nf4-gemm#542 are merged and the maintainer ACKs the launch.
  The ACK names the host (machine or offer id; the launch passes `--vast-pin-machine` and/or `--vast-pin-offer`; a host
  that is gone or over the ceiling means a new ACK, never a substitute).
- The rental also waits until DQ11 has closed and SC2f's rental has torn down: one rental lane at a time, SC2f first.
- The launch commit carries e4b 0.54.0's flips and #1625, and runs only from a commit whose CI is green.
- On the controller, before the rental controller quotes, two gates run in order: the pre-rental fetch gate (STOP-0),
  then the time-left fit (STOP-2).
- The reading rents only after the proof's receipts show PROVED. The fetch gate runs again for its launch.
- Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane's rentals pending a root-cause and simplification
  note.

**STOP rules: P130's, by named substitution:**
- **STOP-0**, the pre-rental fetch gate (`bench/p131/p131_fetch_gate.py`). It covers the checkpoint, the corpus, e4b at
  the launch commit, grouped-nf4-gemm at its pin and every PyPI requirement `p131_run.sh` installs. It never installs,
  downloads a whole file, quotes or rents.
- **STOP-1**, the refusals before any install: host floor 18, dud box 10, card class 15, disk < 150 GB 13, host RAM
  < 60 GiB 16, premise 25.
- **STOP-2**, the time-left checks, enforced by `tests/test_p131_staged_pin.py`.
- **STOP-3:** a VOID, NOISY or QUALITY_VOID reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p131/staged.sha256`.
- **The fetch** runs under P127's byte-growth watchdog.

## What was seen before this page (stated, not hidden)

- **No P131 data exists.**
- **The census** (SC2's prompts; bf16 replay checked against transformers on a 4-layer truncation): sha256
  `94584ec8…` of `census.json`, on the bus as a durable artifact on #846.
- **grouped-nf4-gemm#542 on an RTX A2000 (sm_86)**, correctness only, nothing timed: 37 passed compiled, including the
  split `torch.equal` to the 16-row table at K 768 and 2048, big_m 32 and 64. The interpreter suite gives 136 passed and
  6 compiled-only skipped. The stacked builder's op count (48 against 39) was taken with the CPU profiler.
- **e4b#1625 on CPU:** the knob's contract with stand-ins, `torch.equal` to the 16-row route at 32 and 64, decode at 512
  rows chained and 520 split.
- **e4b#1625's GPU serve smoke on the A2000**, correctness only: the scheduler's path took `split|lean|gt256` at 32 and
  64, logits `torch.equal` to knob 0 in-process and across processes in `qwen3_moe`, `granitemoe` and `mixtral`; at
  knob 0 the 1,024-row calls took the wide one-launch table, which is why the decode check reads m0's table as found.
  Its records are committed as `tests/fixtures/smoke1625/pass2_summary.json` (sha256 `1f4960b1…`, byte-identical to the
  receipt).
- **This registration on CPU:**
  - the reducer's self-test (37 cases) passes;
  - `tests/test_p131_real_records.py` runs the decode check on that smoke's REAL K19 records first (32 passed): every
    cell's own-process knob-0 and knob records, untouched at 0 and 0r and touched at 32 and 64; the serve path's
    records the same; the reducer end to end on real decode records. Then synthetic departures from the real m0
    record. The lane's first rule (`chained|lean|gt256` x layers) fails 17 of these tests;
  - `tests/test_p131_box.py` runs both phases' bookkeeping with the runners stood in (10 passed): the three captures'
    deltas, the decode check, FUNCTION catching a K/V-only difference on one arm, the rotated order, refused captures,
    Phase B's eight arms in one process.
- **Cited figures:** P119's profile figures are from its committed receipt; P124's from its RESULTS.

## What this lane cannot say

- Nothing about other families' speed, or other cards' codegen: bit identity is read on this card only.
- Nothing about later prefill chunks beyond the eager medians, which are reported only.
- Nothing about prompts other than SC2's 512-token wikitext windows: the hot-expert share was measured there.
- Nothing about TTFT or TPOT under load, or about the one-launch tile builder above 1,024 rows, which stays parked.

## Receipts

Fetched to the run directory's `p131/` and committed to `bench/p131/receipts/<run>/`:
- `proc_1.json` … `proc_8.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- `p131_fetch_gate.json`, the launch's STOP-0 report;
- the logs, and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p131.md` is written from those files.

## Amendments

### Amendment 1 (registered 2026-10-19, before any P131 run): lanes rent in parallel, and hosts are pinned under the market rule

**Why.** Two sequencing terms date from 2026-10-10, before the program's standing spending rule (2026-10-18) and its
market rule for pinning hosts:
- "The rental also waits until DQ11 has closed and SC2f's rental has torn down: one rental lane at a time, SC2f first."
- "the maintainer ACKs the launch. The ACK names the host."

SC2f's host bar is scarce: in 15.4 h its four listable bar machines were never listed. So the first term held P131 behind
SC2f indefinitely, although P131 has no host-quality bar of its own.

**The change: sequencing and the launch mechanism only** (maintainer ruling, 2026-10-19 03:03Z).
- **Parallel lanes.** The serial order goes. P131 rents in parallel with other lanes under the standing rule (worst
  ≤ $15 per run).
- **Pinning.** As every lane does under the market rule:
  - one pin post names up to three RTX 5090 machines, each present and passing in 6/6 machine-filtered reads at
    ≤ $0.85/h;
  - the create walks them in order on `no_such_ask`;
  - there is no per-run ACK. This replaces "the maintainer ACKs the launch. The ACK names the host" above, and its
    new-ACK clause.
- **No competition with SC2f.** While SC2f's proof loop is live, P131's pins exclude SC2f's ten bar machines
  (SC2f-PREREG Amendment 2). P131 takes any other RTX 5090 at ≤ $0.85/h.
- **The stack.** Launch commit `c7a40682` (e4b 0.54.0, which carries #1608, #1615 and #1625; the same stack as SC2f),
  grouped-nf4-gemm at its pin `57f53e19`.

**Unchanged:** the arms, the rule, the cost model and the predictions; STOP-0 to STOP-4; the proof (guard 1.25 h, about
$1.73); the reading (guard 2.0 h, about $2.37), only after the proof's receipts show PROVED; the lane ceiling of $5.50;
and the two-strikes rule.
