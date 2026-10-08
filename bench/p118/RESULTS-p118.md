# P118 — results: **SLOWER**. The decode lookahead hides the whole host gap between decode steps, but on the default `serve_paged` server that gap is about 0.21 ms of a 7.2 ms step. One request decodes 1.0198× as fast against a registered bar of 1.02, with identical tokens. The switch stays opt-in (Qwen3-30B-A3B NF4, one RTX 5090)

Registration: `bench/p118/PREREG-p118.md` (#1340, `971c2043`). The switch: `E4B_PAGED_DECODE_LOOKAHEAD` (#1339, `5a1e36e9`).
Issue: #1313 (lane 4(a): the out-of-graph host step).

Code under test:
- e4b 0.49.0 at `971c2043`: the registration, on a main that carries #1339;
- grouped-nf4-gemm 0.43.0 at `6ee2e10`, where P116's bandwidth GEMV is the default at Qwen3's shapes;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, with the NF4 arena baked on the box.

The subject was the default graph server at 16 slots (`E4B_PAGED_MAX_SEQS=16` named). Every bucket from 1 to 16 was
captured, device grouping was on, and the four fusion knobs were at `0` in every arm (P115 Phase C held the flip).

**Verdict by `p118_reduce.py`: `SLOWER`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. Commits, model, prompts and lengths agree, and every arm captured every bucket. L1 ran the lookahead scheduler: at W1 it issued 760 steps, collected all 760 with 752 overlapped (98.9 %), and discarded none; at W16, 880 / 880 / 872 (99.1 %). L0 made no lookahead call. Both L0 arms traced a W1 host gap. |
| NOISY | no. Self-pairs L0b/L0a 0.9984 (W1) and 0.9963 (W16); L1b/L1a 0.9978 and 0.9954 |
| FUNCTION_FAIL | no. All four arms decode identical tokens on every row of both workloads at 32 and 160 tokens. Every timed rep digests the same, and the bucket statistics are identical across arms |
| UNTESTED | no. The premise holds: L0's traced W1 host gap is **0.2072 ms** (L0a 0.2053, L0b 0.2092; the bar is 0.2) |
| **SLOWER** | **yes. g1 = min(1.0205, 1.0198) = 1.0198** (bar 1.02); g16 = min(1.0094, 1.0085) = 1.0085 (bar 0.99) |

## The reading (`p118-5090-2`)

**Host:** one RTX 5090 (sm_120, driver 595.84, 575 W, 170 SMs) on an AMD EPYC 7B13 host (256 threads, ~2 TB RAM),
320 GB free. It was Vast instance 54804559. **Cost:** $0.914, 37 minutes from launch to teardown.

**Timeline (the box's own log, UTC):**
- the premise passed by 08:01:49: 16 + 18 passed, none skipped;
- the fetch ended at 08:22, then the bake and the prompts;
- four arms ran 08:24–08:30, each with its traced passes after its timed ones;
- TP_DONE at 08:30:48.

**Speed** (P109's workloads, decode-only slopes):

| arm | W1 tok/s | W1 ms/step | W16 tok/s | W16 ms/step |
|---|---:|---:|---:|---:|
| L0a | 140.12 | 7.137 | 795.28 | 20.119 |
| L1a | 142.99 | 6.993 | 802.73 | 19.932 |
| L1b | 142.67 | 7.009 | 799.06 | 20.024 |
| L0b | 139.90 | 7.148 | 792.31 | 20.194 |

**The mechanism** (the traced passes at 160 tokens, decode-only steps, medians):

| workload | L0 step | L0 device | L0 host gap | L1 step | period saved | saved / gap |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 7.202 ms | 6.995 ms | 0.207 ms | 6.983 ms | 0.220 ms | **1.06** |
| W16 | 19.996 ms | 19.696 ms | 0.283 ms | 19.699 ms | 0.297 ms | **1.05** |

L1's step period equals L0's device time at both workloads: the GPU no longer waits on the host between decode steps.
The saving is the whole gap, so the traced gain is the gap's share of the step, about 2.9 % at W1. The timed
decode-rate slopes, which the rule reads, give 2.0 % (pairs 1.0205 and 1.0198).

## Against the predictions

| # | prediction | result |
|---|---|---|
| Q1 | L1 overlaps ≥ 99 % at W1 and ≥ 90 % at W16; no discards; no L0 calls; every bucket captured | W1 **missed narrowly** at 98.9 %. Each of the 8 W1 passes ends with one drain collect, and the 32-token passes have only 31 steps, which the prediction overlooked. The rest held: W16 99.1 %, no discards, no L0 calls, every bucket captured |
| Q2 | identical tokens and bucket statistics | **held** |
| Q3 | L0 W1 host gap 0.25–0.60 ms; g1 in [1.02, 1.10]; L0 W1 step 6–8.5 ms | **missed**: the gap was 0.207 ms and g1 1.0198, both just under their bands. The step was 7.14 ms, inside its band |
| Q4 | g16 in [1.00, 1.04] | **held** (1.0085) |
| Q5 | self-pairs within [0.995, 1.005] at W1 and [0.98, 1.02] at W16 | **held** |
| Q6 | at W1 the period saved is 0.7–1.1× the gap | **held** (1.06) |
| Q7 | DEFAULT_ON about 60 %, UNTESTED about 20 %, SLOWER about 10 % | **SLOWER** |
| Q8 | each arm ≤ 4 min; peak ≤ 23 GiB | **held**: about 1.5 min per arm, 22.17 GiB |

## The registered consequence (SLOWER)

The switch **stays opt-in**: `E4B_PAGED_DECODE_LOOKAHEAD` defaults to `0`. The ratios and the trace are recorded, and
the trace answers the question SLOWER leaves open. The gap was there (the premise held), and the lookahead recovered
all of it. The gain falls short because the gap is small on this server and this host: about 3 % of a one-request step.

Not read here, as the registration says:
- the HTTP server's own host work (dispatch, detokenisation, the engine lock), which can only widen the gap;
- slower host CPUs;
- buckets above 16, other families and the int4 route.

A later lane could ask about any of these under its own registration.

This read registers one row: `e4b.serve.p118.decode-lookahead.qwen3.5090.2026-10-08` (g1, with g16, the gap and the
saving in the claim text).

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p118-prove-1` | OK, PROVED | $0.125 | Granite end to end. Tokens identical; host gap 0.204 ms recovered 1.01×; its verdict (not a reading) DEFAULT_ON at g1 1.0246 |
| `p118-5090-1` | NOT_RUN | $0.071 | pre-flight: HF CDN bandwidth 8.0 MB/s < 20 MB/s; no lane work |
| `p118-5090-2` | **OK, SLOWER** | $0.914 | the reading |

**P118 cost $1.110**, inside its $2.50 ceiling and its $3.00 hard stop.

**Receipts** are in `receipts/p118-5090-2/`, with `SHA256SUMS`:
- the four arm records and `verdict.json`;
- the eight step traces (`traces/`);
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- the logs and the teardown proof.

The launcher's receipts and ledger rows are in the receipt store: the reading's is adertha-receipts `70ec84e8`, the
proof's is `c5e69167`, and `p118-5090-1`'s is `93a2dc5b`.
