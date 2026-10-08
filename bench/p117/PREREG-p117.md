# P117 — does decoding 32 or 64 rows in one graph cost quality? `E4B_PAGED_BUCKETS=auto`'s wide buckets against decode in pieces of at most 16 rows, teacher-forced, judged against the 16-row arithmetic's own neutral perturbations, on one RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p117` (pushed 2026-10-08T00:09:18Z). Follows SC2e (#1320, read #1333) and the ruling on #1320.
Derived from P110 (`bench/p110/PREREG-p110.md`, the method that licensed decode graphs) by named substitutions.

## Why this lane

**SC2e read `SLOTS_LICENSED(64, auto)`** (`bench/h2h-2026-10-02/sc2e/README.md`), on Qwen3-30B-A3B int4 on one RTX 5090:
- 64 slots with buckets up to 64 (`E4B_PAGED_BUCKETS=auto`) held the SLO to **12 req/s**; 64 slots on the default list,
  whose decode steps run as 16-row pieces, held it to 8; today's 16 slots hold it to 4.
- One 64-row decode graph ran in 18.4 ms, half of four chained 16-row replays (P1b).
- Serial output was byte-identical on every arm (a single request decodes at bucket 1).

**What changes above 16 rows.** One wide decode step takes the paths prefill chunks take today:
- `Int4Linear` above 16 rows serves its cached bf16 weight with cuBLAS (at ≤ 16 rows: the K16 small-M kernel);
- above 256 routed expert rows (64 rows at top-k 8) K19 runs over the chained tile table (at ≤ 256: the one-launch
  lean builder);
- the T=1 folds still apply at 64 rows.

At 8–16 req/s SC2e's wide-bucket arms kept 0.01–0.12 of the 16-slot server's streamed text byte-equal, against 0.46–0.74
for 64 slots on the default list (no bar). Token agreement measures how fast two bf16 orders part; it cannot say whether either is worse.

**The ruling on #1320:** `SLOTS_LICENSED(64, default)` needs no teacher-forced read (its pieces stay ≤ 16 rows), and the
flip to `E4B_PAGED_MAX_SEQS=auto` on the default list is #1334. `E4B_PAGED_BUCKETS=auto` as a default needs P110's
teacher-forced read at buckets 32 and 64 first, registered only if P1b held and s64a was licensed. Both did.

**The question:** is decoding 32 or 64 rows in one graph at least as good as decoding them in pieces of at most 16 rows,
on a teacher-forced NLL, judged against the 16-row arithmetic's own arithmetically neutral perturbations?

## Instrument

**The model is SC2e's served stack:**
- `PagedServeConfig.from_env()` + `build_engine` with `bench/sc1/sc1_run.sh`'s `SPEEDENV` (int4 experts and int4 RTN
  attention, repacked from the checkpoint at load; the three T=1 folds), `ROUTEENV`, and `E4B_PAGED_FUSE_QKV=1`, byte
  for byte as SC2e's servers ran them;
- built eager with one slot of 768 tokens and no prefill graph (`E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1
  E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0`): the box's passes build their own pools;
- the NF4 arena baked on the box by P39's `k8_bake.py`.

**The passes.** `bench/p117/p117_box.py` runs teacher-forced paged passes on that engine's model
(`parts.runner.model`), each with a fresh `Fp8PagedKV` and `PagedModelRunner`. Every pass is device-grouped (the graph
server's grouping) and decodes every step through the bucketed path with `capture=False`: the runner splits the active
set into pieces of the bucket list's largest bucket and runs each as a padded eager step of its bucket, bit-identical to
the step's graph replay (P109 read G ≡ P at 16 rows; G64 below checks it at 64). Logits are read where the model's forward
returns them, one forward per piece.

**The text.**
- **Source:** wikitext-2-raw test, joined as K8 joins it; window k from token k·3072 (64 windows fit the split).
- **Shape:** 512 prompt tokens and 128 teacher-forced positions per window; **64 windows decoded together**.
- **Scoring:** fp32 log-probs (P97's lesson): the true token's NLL, the argmax, and KL against R.

**Arms** (one pass each over the 64 windows):

| arm | buckets | a 64-row decode step runs as | what it is |
|---|---|---|---|
| R | 1–16 | four 16-row pieces | the reference: s64c's arithmetic class, licensed without this read |
| rep | 1–16 | as R | R's repeatability; a floor draw if not bit-identical |
| half | 1–8 | eight 8-row pieces | floor: a different row count |
| chunk | 1–16 | as R, prompts prefilled in 256-token chunks | floor: a different prefill split |
| rev | 1–16 | as R, windows bound to slots and decoded in reverse | floor: a different row order |
| W32 | 1–32 | two 32-row pieces | **subject** (`Int4Linear` above 16 rows; 256 routed expert rows) |
| W64 | 1–64 | one 64-row piece | **subject** (above 256 routed rows: the chained tile table) |
| W64pad | 1–64 | the first 48 windows, each step padded to 64 | **subject** (padding at the wide bucket) |
| mutant_scale | 1–64 | as W64, decode softmax scale halved | must fail the bar (P108's mutant) |
| G64 | 1–64, **captured** | one 64-row graph replay | FUNCTION: its emitted tokens must equal W64's at every step |

**Engagement, counted per pass:** decode attention calls (P108's `Attention` patch; a replay runs none), the grouping
flags in force, `graph_status` per bucket, and the runner's bucket statistics: pieces per step, eager steps or replays,
rows and padding rows, against the split each arm registers.

## The rule (`bench/p117/p117_reduce.py`, self-tested on 22 cases)

Per window w, `d_X(w)` is X's mean continuation NLL minus R's. Positive means X is worse.

**The floor** is `half`, `chunk` and `rev`, plus `rep` if R did not repeat bit for bit:
- `B_floor` = the largest |mean d_f| over the floor draws;
- `S_floor` = the largest mean |d_f|.

**`passes(X)`** holds iff both (P110's bar, unchanged):
- mean d_X ≤ B_floor + 0.01 nats;
- mean |d_X| ≤ 2 × max(S_floor, 0.005).

**The verdict:**
- **NO_READING:** the premise failed, or there is no box record.
- **VOID**, any of:
  - another e4b or grouped-nf4-gemm commit, or another model revision;
  - an arm with other than its registered windows (64; W64pad the first 48);
  - wrong engagement: calls ≠ (128 − 1) × layers × pieces per step (G64: 0), device grouping off, a bucket not as
    registered (`eager: capture=False`; G64 `graph`), or bucket statistics off the arm's registered split;
  - **FUNCTION:** G64's emitted tokens differ from W64's at any of the 64 × 127 decode positions;
  - `mutant_scale` passes.
- **AT_PARITY:** W32, W64 and W64pad all pass.
- **COST:** otherwise, naming every subject that fails.

**Reported, never gated:** every arm's bias, spread, standard error, max |d|, mean KL against R and argmax agreement;
R's repeatability; the function gate's counts.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | R repeats bit for bit |
| Q2 | the floor's \|bias\| ≤ 0.003 nats and S_floor in [0.008, 0.020] nats (P110's floor on Qwen3-30B-A3B NF4 read B_floor 0.0018, S_floor 0.0127; its Q2 had allowed 0.005 and missed). rev is no longer bit-identical: reversing 64 windows changes which share a 16-row piece |
| Q3 | W32's, W64's and W64pad's bias each lie in [−0.003, +0.003] and spread ≤ 0.016 (inside the floor's class, as P110's P at 0.0114): the wide paths read the same int4 values; only the reduction order differs |
| Q4 | FUNCTION holds: G64's tokens equal W64's at every position |
| Q5 | mutant_scale's bias exceeds +0.3 nats |
| Q6 | engagement is exact |
| Q7 | the box runs ≤ 30 minutes, the int4 repack included |
| Q8 | the verdict is AT_PARITY |

## Consequence, registered now

- **AT_PARITY** licenses `E4B_PAGED_BUCKETS=auto` as a default beside `E4B_PAGED_MAX_SEQS=auto` (#1334), in a
  separate PR citing SC2e and this read:
  - **The default.** `E4B_PAGED_BUCKETS` unset reads `auto`; the explicit list `1,2,4,8,16` restores today's buckets.
    `E4B_PAGED_MAX_SEQS=auto` then chooses among 64, 32 and 16 (the widths SC2e read with `auto` buckets).
  - **The docs.** `docs/SERVING.md` states the scope and that greedy outputs under load change at the bf16 level, at
    no measured quality cost.
  - **The register.** A row `e4b.serve.p117.wide-bucket-quality.qwen3.5090.<date>` carries W64's bias, from the verdict
    file.
  - **Scope.** Qwen3-30B-A3B int4 (SC2e's stack), 512-token prompts, buckets 32 and 64. Other families ride the same
    capture code without a reading of their own; the docs say so.
- **COST:** `E4B_PAGED_BUCKETS=auto` stays opt-in; `docs/SERVING.md` quotes the cost per subject; #1334's default
  stays on the default bucket list.
- **VOID or NO_READING:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed path; `tests/test_p117_box.py` covers the box's bookkeeping on CPU
with the runner stood in (pieces, padding, row mapping, the function comparison).

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` (its NF4 store, no int4 levers, as SC2c's and SC2e's proofs), with 40
windows and 32 positions:
- the refusals;
- install and tripwire;
- the reducer self-test;
- the premise: `tests/test_decode_graph_buckets.py` (P110's registered bytes), 7 passed, none skipped;
- fetch, bake, the box (every arm, G64 included) and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID or NO_READING. The proof's verdict is not a reading.
The proof exercises buckets 32 and 64 through Granite's NF4 routes; the int4 paths above 16 rows first run on the
reading box, where engagement and FUNCTION gate them.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.5 h at ≤ $0.75/h, estimate $1.125. Expected about 45 minutes: install, fetch
  ~10, bake ~5, the box ~25 (the int4 repack at load ~8, ten passes ~10).
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals (exit codes) run before any install: dud box (10), card class (15), disk < 150 GB (13), host
  RAM < 60 GiB (16), premise (25).
- **STOP-2:** every time-left check is a per-mode variable that fits its own guard (enforced by
  `tests/test_p117_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p117/staged.sha256`.

## What this lane cannot say

- Nothing about other families, the NF4 store, or other batch shapes than 32, 64 and 48-padded-to-64 rows.
- Nothing about TTFT or speed, which are SC2e's.
- Nothing about any arithmetic difference smaller than the 16-row arithmetic's own neutral perturbations: by
  construction, that is the floor.

## Receipts

Fetched to the run directory's `p117/` and committed to `bench/p117/receipts/<run>/`:
- `box.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the logs, and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p117.md` is written from those files.
