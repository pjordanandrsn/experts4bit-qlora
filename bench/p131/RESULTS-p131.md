# P131 results: K19 over a split tile table on the served prefill graph (e4b#846)

**Status:** closed. **M32_HELD M64_HELD DEFAULT_NONE.** Neither tall tile clears the registered x1.02 on one RTX
5090. Both compute the 16-row table's arithmetic bit for bit, so `E4B_K19_PREFILL_BIG_M` stays opt-in and nothing
changes by default. No claims row and no flip.

P131 asked whether K19 over a split tile table (`E4B_K19_PREFILL_BIG_M`, e4b#1625 over grouped-nf4-gemm#542) makes the
served 512-token first-chunk prefill graph faster on Qwen3-30B-A3B, and whether it is the same arithmetic
([registration](PREREG-p131.md), with Amendments 1 and 2). Each process captures the graph once per arm and times the
three interleaved: m0 (the 16-row table, the control), m32 and m64.

| run | what | cost | outcome |
|---|---|---|---|
| [`p131-prove-1`](receipts/p131-prove-1/README.md) | the proof | $0.661 | **PROVED** (its verdict is not a reading) |
| [`p131-5090-1`](receipts/p131-5090-1/README.md) | the reading | $0.027 | NOT_RUN: a host stall at the pre-flight, not a strike |
| [`p131-5090-2`](receipts/p131-5090-2/README.md) | the reading | $1.048 | **M32_HELD M64_HELD DEFAULT_NONE** |

The lane spent $1.736 of its $5.50 ceiling. The maintainer re-derived both the proof and the reading from the private
receipt store.

## The read

Every integrity rung held:
- every capture was on in every process, and none was refused;
- the decode check was untouched under every arm;
- DETERMINISM held over all nine processes;
- the m0 span was x1.0050 over the scored processes;
- FUNCTION read **BITWISE** for m32 and m64 on every speed window of every process;
- Phase B's subjects S32 and S64 were `torch.equal` to R, R repeated bit for bit, and the mutant was caught.

So both arms are the 16-row table's arithmetic, and only speed decides:

| arm | gain over m0 (processes 1-8) | saving | verdict |
|---|---|---|---|
| m32 | **0.9836** [0.9828, 0.9845] | -0.573 ms | HELD: NOT_FASTER (slower in all 8 processes) |
| m64 | **1.0073** [1.0063, 1.0083] | +0.250 ms | HELD: NOT_FASTER (under x1.02) |

m0 replays at 34.36-34.53 ms. m64 over m32 reads x1.0241 [1.0236, 1.0246]. Per-process medians are in the
[reading's README](receipts/p131-5090-2/README.md).

**Amendment 2 answered the proof's anomaly.** The proof read NOISY because its process 1 ran about 4.7 % faster on
every arm. In the reading, the warm-up process 0 read m0 at x0.999 of the scored median, so the effect did not recur.
The GPU state, recorded four times a process, matched across all nine processes: 2505-2520 MHz at 540-546 W around
the timed rounds (after the warm rounds and after the timed rounds), with the software power cap active (clock event
reason 0x4). The ratios therefore sit within one host
at one clock. What made the proof's process 1 fast stays unattributed, because the proof recorded no clocks.

## Predictions (graded, never ruling)

| # | prediction | read | grade |
|---|---|---|---|
| Q1 | engagement exact; no capture refused; the decode check untouched | as predicted | hit |
| Q2 | FUNCTION bitwise for both arms; DETERMINISM holds | as predicted | hit |
| Q3 | m64: g in [1.038, 1.159], saving 1.2-4.5 ms | 1.0073, 0.250 ms | miss (low) |
| Q4 | m32: g in [1.033, 1.143], saving 1.1-4.1 ms | 0.9836, -0.573 ms | miss (low) |
| Q5 | m64 over m32: x1.004-1.013 | x1.0241 | miss (high) |
| Q6 | NOISY does not fire; m0 span <= x1.01 | x1.0050 | hit |
| Q7 | R repeats bit for bit; S32 and S64 `torch.equal` to R | as predicted | hit |
| Q8 | M32_LICENSED M64_LICENSED DEFAULT_m64 | M32_HELD M64_HELD DEFAULT_NONE | miss |
| Q9 | each process <= 4 min except the Phase B process (<= 9 min); the eight <= 40 min; peak <= 27 GiB | scored processes 3.63-3.82 min (the box's log stamps); process 0 13.4 min (Phase B 9.4); the eight about 30 min; peak 21.3 GB | miss (process 0) |

## What the misses mean

The registration's reading of a low Q3 or Q4 applies: f is below its band, or the tall tiles cost more than the
model sees. The model nets the split's weight-pass saving on K19 (0.333 f at 64 and 0.308 f at 32, of K19's 19.9 ms)
against the builder's 0.8 ms. Solving it for f from each arm's measured saving:
- m64: 0.333 x 19.9 x f - 0.8 = +0.250 ms gives **f about 0.16**;
- m32: 0.308 x 19.9 x f - 0.8 = -0.573 ms gives **f about 0.04**.

Both lie below the registered [0.3, 0.8], and they disagree with each other. One f cannot explain both arms, so the
32-row tile carries a cost the model lacks.

**An open question, not a new lane:** what the BLOCK_M-32 plan (`block_n`, `kc`, warps) costs that the model does
not carry. m64's mean is +0.73 % and its interval tops out at +0.83 %, far under the bar, so chasing it is low value unless a $0 structural profile of
grouped-nf4-gemm on the correctness card shows something obvious. Q5 missed on size, not direction: m64 beat m32 by
2.4 %, against the 0.4-1.3 % predicted, and most of that gap is m32's own 1.6 % penalty against m0, not m64's gain.

## Consequence

As registered for **DEFAULT_NONE**: `E4B_K19_PREFILL_BIG_M` stays opt-in, and the ratios are recorded here. The
NOT_FASTER reading: m64's saving is about a fifth of what the weight-pass model predicted at its lower f (0.250
against 1.19 ms at f = 0.3), and m32's is negative. Since FUNCTION read bitwise, the knob remains numerically safe to
set, but it is not faster on the graph, and the eager first-chunk forward is slower with it: x0.85, about 15.5 ms
(reported, never gated). No TTFT or TPOT figure under load is claimed; that is an SC lane's to read.
