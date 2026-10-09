# P126 — results, attempt 1: **NOISY**. P = 4's two blocks disagree by 2.35 % (bound 1.5 %), so no default moves; P = 8's blocks agree at 0.896 / 0.897. The disagreement is one runner's level offset, which longer blocks cannot remove (one RTX 5090, 2026-10-09)

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

## The reading (`p126-5090-1`)

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

## Why P = 4's blocks disagree (diagnosis, not gated)

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

## Against the predictions (evaluated by the reducer; none gates the verdict)

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

## The registered consequence (NOISY)

- **No default moves.** `E4B_INT4_TILE_PROGRAMS` stays opt-in.
- **The rerun is an amendment.** The registered remedy is longer blocks, but the diagnosis above shows a runner level
  offset, which longer blocks cannot remove. Amendment 1 is registered separately and reviewed before attempt 2. It
  makes NOISY a per-candidate condition: a candidate whose blocks disagree is ineligible, and it does not void a clean
  candidate. It also moves attempt 2 to another machine and records the host's power cap and co-tenancy. Attempt 1 is
  not re-reduced under it.

## What it took

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
