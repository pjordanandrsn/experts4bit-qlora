# P126 — results: **DEFAULT_ON_4** on attempt 2, under Amendment 1. Splitting the one-launch cumsum tile table over 4 programs makes Qwen3-30B-A3B's captured 64-row decode step about 15.5 % faster, with every token identical. P = 8 was ineligible because its blocks disagreed. Attempt 1 read NOISY (one RTX 5090 each, 2026-10-09)

Registration: `bench/p126/PREREG-p126.md` (#1458, merged `26cafb9c`), with Amendment 1 (#1464, merged `f1cfcf2f`) for
attempt 2 on. The code under test on attempt 2:
- grouped-nf4-gemm `e21a712` (0.44.0 with #524's `programs=` and #525);
- experts4bit-qlora `f1cfcf2f` (0.51.0 with #1433's `E4B_INT4_TILE_PROGRAMS` and the lane's box).

**Verdict by `p126_reduce.py` on attempt 2 (`p126-5090-2`): `DEFAULT_ON_4`.** Amendment 1's rule applies: the record
carries `"amendment": 1`. The rule's steps, in order:

| step | result |
|---|---|
| NO_READING | no. The box record is present; the premise passed: `tests/test_decode_graph_buckets.py` 7 passed, and grouped-nf4-gemm's `kernel/test_tile_table_programs_interp.py` compiled on the card, 202 passed, none skipped |
| VOID | no. Every registered gate held, as on attempt 1. The record names Amendment 1 and carries the host at the box's start and end: a 575 W power cap (600 W maximum), 239.7 of 251.4 GiB free, 64 CPUs, one GPU compute process. Not shared and not capped |
| PREMISE_ABSENT | no. P = 1's one-program table is **18.0 %** of the eager 64-row step's device time (bar 5 %) |
| TOKENS_DIFFER | no. Every candidate emitted P = 1's tokens at every position in every block. The mutant changed 94.9 % of its positions |
| NOISY, per candidate (Amendment 1) | P = 8 is **ineligible**: block a 0.8481, block b 0.8183, 0.0364 apart (bound 0.015). P = 4 is eligible: 0.8442 / 0.8456, 0.0017 apart |
| DEFAULT_ON_4 | **yes**. P = 4, the only eligible candidate, has both block ratios at most 0.98 |

## Attempt 2: the reading (`p126-5090-2`)

**Host:** one RTX 5090 (driver 610.57.04, power cap 575 W of a 600 W maximum) on a 64-thread AMD host with 251 GiB RAM.
It was Vast instance 55056239 on machine 13828; the launcher avoided attempt 1's machine (37367) by citing its receipt.
**Cost:** $0.332. Teardown was proven at 17:01:27Z. The blocks' clock logs read 2130–2242 MHz (attempt 1's capped card:
2385–2632), so the absolute step here is slower than attempt 1's (18.5 against 17.0 ms). Every ratio is inside one box.

**Timeline (the box's own log):** install at 16:48:54Z, fetch from 16:51:11Z, bake from 16:55:31Z, the box from 16:56:19Z
(load 112 s, box 162 s), `TP_DONE` at 17:01:01Z.

**Profiles** (8 eager bucket-64 steps each, device time):

| setting | eager step (ms) | table (ms per step) | table launches per step |
|---|---:|---:|---:|
| P = 1 | 17.445 | 3.143 (`_tile_table_r1`) | 48 |
| P = 4 | 15.050 | 0.690 (`_tile_table_cumsum_mp`) | 48 |
| P = 8 | 14.642 | 0.290 (`_tile_table_cumsum_mp`) | 48 |

The split table runs at 0.220× (P = 4) and 0.092× (P = 8) of the one-program table. The eager step runs at 0.863× and
0.839×.

**Served blocks** (median ms of 256 timed captured steps per runner; the ratio is candidate / P = 1):

| candidate | block | order | P = 1 | candidate | ratio | per-pair median |
|---|---|---|---:|---:|---:|---:|
| 4 | a | 1, 4 | 18.497 | 15.616 | **0.8442** | 0.8445 |
| 4 | b | 4, 1 | 18.469 | 15.618 | **0.8456** | 0.8459 |
| 8 | a | 1, 8 | **17.857** | 15.145 | 0.8481 | 0.8484 |
| 8 | b | 8, 1 | 18.518 | 15.153 | 0.8183 | 0.8184 |

- **P = 4 saves 2.87 ms** of the 64-row step.
- **Busy fractions:** every runner's is 0.970–0.976, so the step is device-bound.
- **Memory:** the peak allocation in block a was 25,323 MiB with P = 4 and 25,461 MiB with P = 8.

## The recurring runner offset (an open measurement question, not gated)

P = 8's disagreement has attempt 1's signature:
- P = 1's runner in P = 8's block a sits about 0.65 ms below P = 1 in the other three blocks, in every quarter of its 256
  steps.
- Its partner, the P = 8 runner, matches its block-b counterpart to 0.01 ms.
- Within every block the per-pair ratio drifts smoothly by 0.8–1.1 % (attempt 1: 0.5–0.8 %).

| quarter | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| P = 1 in P = 4 block a | 18.03 | 18.31 | 18.64 | 18.89 |
| P = 1 in P = 4 block b | 18.00 | 18.29 | 18.61 | 18.86 |
| P = 1 in P = 8 block a | **17.39** | **17.69** | **18.01** | **18.25** |
| P = 1 in P = 8 block b | 18.05 | 18.35 | 18.66 | 18.91 |

**Across both attempts:**
- The offset hit a **block-a P = 1 runner**, the first runner in its block's order, **2 times in 4**: attempt 1, P = 4's
  block; attempt 2, P = 8's block.
- It hit **0 of the 8 candidate runners**.
- It appeared on a shared host with a 400 W cap (attempt 1) and on an unshared host at 575 W (attempt 2), so co-tenancy
  and the cap do not explain it.

Its cause is not established here.

**Proposed for future interleaved lanes, not run here.** Either check would show whether the first-built runner in a
block is the one that drifts:
- a discarded warm-up runner per block: one extra runner built and captured first, then thrown away;
- randomized runner order per block, recorded in the box.

## Attempt 2 against Amendment 1's predictions (evaluated by the reducer; none gates the verdict)

| # | prediction | result |
|---|---|---|
| Q1 | P = 1's table share in [0.12, 0.22] | **held** (0.180) |
| Q2 | P = 8's split / one in [0.06, 0.15] | **held** (0.092) |
| Q3 | P = 4's in [0.15, 0.30] | **held** (0.220) |
| Q4 | both P = 8 block ratios in [0.86, 0.93] | **missed**, on the fast side (0.8481, 0.8183) |
| Q5 | both P = 4 block ratios in [0.88, 0.95] | **missed**, on the fast side (0.8442, 0.8456) |
| Q6 | every busy fraction ≥ 0.90 | **held** (minimum 0.970) |
| Q7 | each candidate's block ratios within 1 % | **missed** (P = 8 3.64 %; P = 4 0.17 % held) |
| Q8 | the mutant differs at ≥ 50 % of positions | **held** (94.9 %) |

**Why Q4 and Q5 missed.** They missed on the fast side, and the cause is not established:
- the served saving at P = 4 (2.87 ms) exceeds the eager profile's own table saving (3.143 − 0.690 = 2.45 ms);
- it is also larger than attempt 1's (about 1.55 ms in its clean block);
- why the captured step gains more than the table's eager time is not explained here.

Q7 missed because the runner offset recurred.

## The registered consequence (DEFAULT_ON_4)

- **A separate experts4bit-qlora PR gives `E4B_INT4_TILE_PROGRAMS` an `auto` default.**
  - It takes **P = 4** for the tables the lane read, `next_pow2(E) × next_pow2(R) ≤ 128 × 512`, when the installed
    grouped-nf4-gemm takes `programs=`, detected from its signature.
  - It takes effect with the grouped-nf4-gemm release carrying #524, and is a no-op on 0.44.0.
  - `E4B_INT4_TILE_PROGRAMS=1` restores the one-program table.
- **The register row** is `e4b.serve.p126.tile-programs.qwen3-int4.5090.2026-10-09`.
- **What this licenses:** P = 4, not P = 8. P = 8 read faster in its clean block, but the rule never selects a noisy
  candidate, and this read does not license it.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p126-prove-1` | REFUSED before any instance | $0.000 | the cheapest offer billed $0.9004/h with 320 GB of storage, over the $0.85/h ceiling |
| `p126-prove-2` | **PROVED** | $0.184 | Granite, the whole box; its numbers are not a reading |
| `p126-5090-1` | **OK, NOISY** | $0.874 | attempt 1, below |
| `p126-5090-2` | **OK, DEFAULT_ON_4** | $0.332 | attempt 2, Amendment 1, another host |

**The lane cost $1.390** of its $3.00 ceiling.

**Receipts** are in `receipts/p126-5090-2/` and `receipts/p126-5090-1/`, each with `SHA256SUMS`.

**Reproduce both verdicts** from the committed copies. Each verdict file is byte-identical on Python 3.9, 3.13 and 3.14.
The reducer applies Amendment 1 only to the record that names it:

```
python bench/p126/p126_reduce.py --dir bench/p126/receipts/p126-5090-2 --out /tmp/v2.json --e4b-sha f1cfcf2fe7f64e68e6b4c7242a3df33664d327de
cmp /tmp/v2.json bench/p126/receipts/p126-5090-2/verdict.json
python bench/p126/p126_reduce.py --dir bench/p126/receipts/p126-5090-1 --out /tmp/v1.json --e4b-sha 26cafb9c3b4564e1c7b0c393efafe4d779cd886f
cmp /tmp/v1.json bench/p126/receipts/p126-5090-1/verdict.json
```

---

## Attempt 1 (`p126-5090-1`): **NOISY**

*As read at the time:* P = 4's two blocks disagree by 2.35 % (bound 1.5 %), so no default moves; P = 8's blocks agree at 0.896 / 0.897. The disagreement is one runner's level offset, which longer blocks cannot remove (one RTX 5090, 2026-10-09)

Registration: `bench/p126/PREREG-p126.md` (#1458, merged `26cafb9c`). The code under test:
- grouped-nf4-gemm `e21a712` (0.44.0 with #524's `programs=` and #525);
- experts4bit-qlora `26cafb9c` (0.51.0 with #1433's `E4B_INT4_TILE_PROGRAMS`).

**Verdict by `p126_reduce.py` on attempt 1 (`p126-5090-1`): `NOISY`.** This attempt stays NOISY: it is never
re-reduced under a later rule. The rule's steps, in order:

| step | result |
|---|---|
| NO_READING | no. The box record is present; the premise passed: `tests/test_decode_graph_buckets.py` 7 passed, and grouped-nf4-gemm's `kernel/test_tile_table_programs_interp.py` compiled on the card, 202 passed, none skipped |
| VOID | no. The commits and model revision are the pinned ones, on SC2e's stack with every tile knob unset outside the box. Every runner replayed bucket 64 exactly 293 times with no eager step. Each profile ran 8 eager bucket-64 steps under its setting and engaged its table: P = 1 one `_tile_table_r1` launch per MoE layer (48), P = 4 and P = 8 one `_tile_table_cumsum_mp` per layer (48). Tokens were deterministic across blocks. The mutant changed 94.9 % of its positions |
| PREMISE_ABSENT | no. P = 1's one-program table is **17.7 %** of the eager 64-row step's device time (bar 5 %) |
| TOKENS_DIFFER | no. Every candidate emitted P = 1's tokens at every position in every block |
| NOISY | **yes**. P = 4: block a **0.9304**, block b **0.9090**, 0.0235 apart (bound 0.015). P = 8: 0.8959 / 0.8974, 0.0015 apart |

### The reading (`p126-5090-1`)

**Host:** one RTX 5090 (driver 595.71.05) on an AMD EPYC 7C13 host (256 CPUs, 1007 GiB RAM, 496 GiB of it already in use
when the box started: a shared host). It was Vast instance 55043022 on machine 37367. In every block's clock log, the card's power draw peaked
at 400 W: on this host the card runs under a 400 W cap. **Cost:** $0.874, most of it the 61 GB fetch at this host's
download price. Teardown was proven at 15:28:03Z.

**Timeline (the box's own log):** install at 15:11:12Z, fetch from 15:13:52Z, bake from 15:19:49Z, the box from
15:21:24Z (load 167 s, box 194 s), `TP_DONE` at 15:27:37Z.

**Profiles** (8 eager bucket-64 steps each, device time):

| setting | eager step (ms) | table (ms per step) | table launches per step |
|---|---:|---:|---:|
| P = 1 | 14.830 | 2.629 (`_tile_table_r1`) | 48 |
| P = 4 | 12.740 | 0.592 (`_tile_table_cumsum_mp`) | 48 |
| P = 8 | 12.425 | 0.241 (`_tile_table_cumsum_mp`) | 48 |

The split table runs at 0.225× (P = 4) and 0.092× (P = 8) of the one-program table. The eager step runs at 0.859× and
0.838×.

**Served blocks** (median ms of 256 timed captured steps per runner; the ratio is candidate / P = 1):

| candidate | block | order | P = 1 | candidate | ratio | per-pair median |
|---|---|---|---:|---:|---:|---:|
| 4 | a | 1, 4 | **16.513** | 15.364 | 0.9304 | 0.9300 |
| 4 | b | 4, 1 | 17.021 | 15.473 | 0.9090 | 0.9093 |
| 8 | a | 1, 8 | 17.063 | 15.286 | 0.8959 | 0.8960 |
| 8 | b | 8, 1 | 17.091 | 15.339 | 0.8974 | 0.8970 |

Every runner's busy fraction was at least 0.967.

### Why P = 4's blocks disagree (diagnosis, not gated)

`box.json` keeps every runner's 256 step times. Two facts from them:
- **Within each block the two runners move together.** Both rise about 1 ms over the 256 steps as the context grows.
  The per-pair ratio drifts up by 0.5–0.8 % across each block, smoothly (P = 4 a by quarter: 0.9276, 0.9294, 0.9311,
  0.9319). That is expected: the table's saving is fixed while attention grows with the context. The lockstep
  alternation held.
- **The disagreement is one runner's level.** P = 1's runner in P = 4's block a, the first timed block of the box, sits
  about 0.5 ms below P = 1 in the other three blocks, in every quarter:

  | quarter | 1 | 2 | 3 | 4 |
  |---|---:|---:|---:|---:|
  | P = 1 in P = 4 block a | 15.97 | 16.34 | 16.71 | 17.01 |
  | P = 1 in P = 4 block b | 16.50 | 16.83 | 17.20 | 17.45 |
  | P = 1 in P = 8 block a | 16.57 | 16.87 | 17.23 | 17.53 |
  | P = 1 in P = 8 block b | 16.58 | 16.92 | 17.27 | 17.56 |

  Its partner, the P = 4 runner in the same block, sits only 0.1 ms below its block-b counterpart.

**What follows:**
- **A fresh runner can carry a level offset of about 3 % for its whole life.** P124's attempt 1 read NOISY with the same
  signature.
- **Longer blocks would not remove it:** the offset lasted all 256 steps.
- **The host is not ruled out:** a shared machine, a 400 W cap, and P = 1's step at 17.0 ms against P124's 14.7 ms on
  another box. The ratios are within-box, so the cap shifts levels, not the comparison.
- **The rule did its job.** One runner's offset in one block voided the attempt rather than leaking into a verdict.

### Against the predictions (evaluated by the reducer; none gates the verdict)

| # | prediction | result |
|---|---|---|
| Q1 | P = 1's table share in [0.10, 0.22] | **held** (0.177) |
| Q2 | P = 8's split / one in [0.10, 0.40] | **missed, faster** (0.092) |
| Q3 | P = 4's in [0.20, 0.55] | **held** (0.225) |
| Q4 | both P = 8 block ratios in [0.86, 0.95] | **held** (0.8959, 0.8974) |
| Q5 | both P = 4 block ratios in [0.88, 0.96] | **held** (0.9304, 0.9090) |
| Q6 | every busy fraction ≥ 0.90 | **held** (minimum 0.967) |
| Q7 | each candidate's block ratios within 0.5 % | **missed** (P = 4 2.35 %; P = 8 0.15 % held) |
| Q8 | the mutant differs at ≥ 50 % of positions | **held** (94.9 %) |

### The registered consequence (NOISY)

- **No default moves.** `E4B_INT4_TILE_PROGRAMS` stays opt-in.
- **The rerun is an amendment.** The registered remedy is longer blocks, but the diagnosis above shows a runner level
  offset, which longer blocks cannot remove. Amendment 1 is registered separately and reviewed before attempt 2. It
  makes NOISY a per-candidate condition: a candidate whose blocks disagree is ineligible, and it does not void a clean
  candidate. It also moves attempt 2 to another machine and records the host's power cap and co-tenancy. Attempt 1 is
  not re-reduced under it.

### What it took

| run | status | cost | note |
|---|---|---:|---|
| `p126-prove-1` | REFUSED before any instance | $0.000 | the cheapest offer billed $0.9004/h with 320 GB of storage, over the $0.85/h ceiling |
| `p126-prove-2` | **PROVED** | $0.184 | Granite, the whole box; its numbers are not a reading |
| `p126-5090-1` | **OK, NOISY** | $0.874 | attempt 1, this reading |

**The lane has spent $1.058** of its $3.00 ceiling.

**Receipts** are in `receipts/p126-5090-1/`, with `SHA256SUMS`:
- `box.json` (every runner's step times), `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the eight traced-step files (`trace_p{4,8}_{a,b}_P*.jsonl`);
- `logs/` and the teardown proof.

**Reproduce the verdict** from the committed copy. The verdict file is byte-identical on Python 3.9, 3.13 and 3.14:

```
python bench/p126/p126_reduce.py --dir bench/p126/receipts/p126-5090-1 --out /tmp/v.json --e4b-sha 26cafb9c3b4564e1c7b0c393efafe4d779cd886f
cmp /tmp/v.json bench/p126/receipts/p126-5090-1/verdict.json
```
