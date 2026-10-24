# P233 results: serving at `E4B_PAGED_PLACEMENT=solver` against all-vram, teacher-forced quality (SOLVER_SANE, TRIPWIRE fired)

**Registration:** [PREREG-p233.md](PREREG-p233.md) (e4b#2103, with Amendments 1-4), read once, as registered.
**Reading:** `p233-5090-1`, one RTX 5090 (575 W limit, recorded and never ruled) on an AMD EPYC 7532 host, e4b `2a0a9afc`
(the Amendment 4 merge), grouped-nf4-gemm v0.50.0, $1.45. Records: [receipts/p233-5090-1/](receipts/p233-5090-1/README.md).

## The question

On Qwen3-30B-A3B's server, does serving at `E4B_PAGED_PLACEMENT=solver` keep teacher-forced quality against the same
server at `all-vram`? Arm S (solver at 8.0 GiB VRAM / 3.0 GiB DRAM, all three tiers) is scored against arm A (all-vram,
the reference); arm C (solver at 8.0 / 0: VRAM and NVMe only) is reported. Per group size T in 1 and 8:
|mean NLL(S) - mean NLL(A)| <= 0.02 and argmax agreement >= 0.95 (P153's bars). Every arm sets `E4B_PAGED_GRAPHS=0`,
`E4B_INT8_HEAD_BW=1`, `E4B_INT8_HEAD_ROWS=8` and `E4B_PAGED_HOT_ROWS=128`.

## The answer

| T | NLL gap (S - A) | argmax agreement | KL (S against A) | rules |
|---|---|---|---|---|
| 1 | +0.002077 | 2993 / 3072 (0.974284) | 0.003925 | SANE |
| 8 | -0.000047 | 2993 / 3072 (0.974284) | 0.003729 | SANE |

- **SOLVER_SANE.** S passes both bars at both T, with wide margins: the largest gap is about a tenth of the 0.02 bar, and the
  agreement is 0.024 above the 0.95 bar.
- **The TRIPWIRE fired, at both T, on two of its three conditions:**
  - argmax agreement **0.974284, below the 0.98 look-bar**, at T = 1 and at T = 8;
  - KL **0.003925 (T = 1) and 0.003729 (T = 8), above max(3 x floor, 0.002) = 0.002**. A's repeat pass was bit-identical
    to A at both T, so the floor read 0 and the absolute 0.002 bound applied;
  - the NLL condition did not fire: |gap| 0.002077 and 0.000047, against 0.005.
- **What the wire means here (as registered):** the verdict stands as read, and no doc or lane may propose serving at
  `solver` until the written look below explains the firing, reviewed as usual.
- **The instrument held:**
  - every record ok; the pins (e4b `2a0a9afc`, grouped-nf4-gemm v0.50.0, torch 2.8.0+cu128, the model and its revision)
    and the card (cc 12.0, 170 SMs) as registered; one window set by sha in every phase (24 wikitext windows x 128
    positions after a 512-token prompt per T); the same NF4 arena (sha256 `6c27ae82…`) in every arm;
  - graphs off and every decode bucket eager in every arm; the head the int8 head at rows 8 (`explicit`) in every arm,
    K37 at plan (32, 256, 4); the split cap (16) and the fusions resolving the same in every phase;
  - the NVMe tier's pinned rows resolved to 128 in every phase (Amendment 3); the pass KV at the server's 4096 tokens;
  - **the placement proof per arm**, from the record:

    | arm | experts VRAM / DRAM / NVMe | MoE layers collapse / tiered | DRAM rows on the host CPU (T = 1) |
    |---|---|---|---|
    | A | 6144 / 0 / 0 | 48 / 0 | 0 |
    | S | 3236 / 1213 / 1695 (8.59 GB / 3.22 GB / 4.50 GB) | 25 / 23 | 30,480 calls |
    | C | 3236 / 0 / 2908 | 25 / 23 | 0 |

## The written look the TRIPWIRE asks for

The wire fired because S's per-position distributions sit about 0.004 nats of KL from A's, and 79 of 3,072 teacher-forced
argmax tokens (2.6 %) differ. The argmax agreement is below the 0.98 look-bar at both T. This look does not move it, and
it changes no bar. It asks what produced a disagreement of that size, from the reading's own records, at $0.

**1. On this instrument, the reference placement moves further than S does under its own registered floor arms.** A's
phase scores P115's floor arms on the all-vram server, the same windows and the same arithmetic as R:
- **chunk** (the same prompts prefilled in 256-token pieces instead of 512);
- **half** (the group decoded as two halves, each padded to its own bucket; at T = 1 a group of one cannot be halved, so
  it reads identical to R).

| T | arm, against A's R | NLL gap | argmax agreement | KL | windows where S's KL is lower |
|---|---|---|---|---|---|
| 1 | **S** | +0.002077 | 0.974284 | 0.003925 | |
| 1 | chunk | +0.003813 | 0.963867 | 0.009986 | 24 / 24 |
| 8 | **S** | -0.000047 | 0.974284 | 0.003729 | |
| 8 | chunk | -0.003719 | 0.967448 | 0.009748 | 24 / 24 |
| 8 | half | -0.002743 | 0.959635 | 0.009098 | 24 / 24 |

In every one of the 24 windows at each T, S's KL against A is below the chunk arm's, and below the half arm's at T = 8.
S's agreement is above every floor arm's. Splitting a prompt's prefill at a different boundary moves the reference's
distributions about 2.5 times as far as the solver placement does. Every chunk window is above 0.002 (its smallest is
0.00434 at T = 1, 0.00501 at T = 8), and both floor arms' agreement is below 0.98. The registered floor is A's repeat
pass, which is bit-identical, so the wire's KL bound falls to 0.002. That bound sits below the instrument's own
reduction-order floor, and any change of summation order of this size would fire it. This is a statement about what
the wire measured. It does not unfire the wire.

**2. The disagreement does not need the DRAM tier.** C keeps no expert in DRAM (no host CPU arithmetic) and reads:

| T | NLL gap (C - A) | argmax agreement | KL (C against A) | S's flips that C shares |
|---|---|---|---|---|
| 1 | +0.000513 | 3030 / 3072 (0.986328) | 0.001553 | 26 of 79 (33 %) |
| 8 | -0.000415 | 2986 / 3072 (0.972005) | 0.003676 | 46 of 79 (58 %) |

- **At T = 8, C is as far from A as S is:** KL 0.003676 against 0.003729, with the per-window KL of S and C correlated
  at 0.88. The tiered forward on the GPU (23 MoE layers, shared by S and C) produces a disagreement of S's size without
  any expert on the host.
- **At T = 1, C is closer** (KL 0.001553, 8 of 24 windows above 0.002, agreement 0.986). There, S's host-side rows add
  to the disagreement: S against C reads 0.976888 agreement.
- **S and C are not the same path.** S against C reads 0.976888 (T = 1) and 0.974935 (T = 8) agreement, so the
  positions that flip differ between the two arithmetics.
- **Not explained by this record:** why C's distance from A grows from T = 1 to T = 8 (KL 0.001553 to 0.003676) while
  S's does not (0.003925 to 0.003729). One interpretation, not tested here, is that the tiered forward schedules the
  rows of a group of 8 differently from a group of 1.

**3. No systematic loss.**
- **S's per-window NLL gaps take both signs:** 14 windows worse and 10 better at T = 1 (mean +0.002077), and 8 worse
  and 16 better at T = 8 (mean -0.000047).
- **Its largest single-window gap is 0.024 (T = 1) and 0.027 (T = 8),** against the chunk arm's 0.052 and 0.050.
- **The instrument does see a real defect.** The registered mutant (R with the decode softmax scale halved) reads a gap
  of +1.000 (T = 1) and +1.012 (T = 8) nats, agreement 0.614 and 0.612, and KL 1.118 and 1.135.

**What this look concludes:**
- **What the record shows:** S's disagreement with A is the size of a change of summation order, not a defect. It is
  smaller in every window than the reference server's own chunked-prefill floor. It is unbiased within ±0.002 nats. At
  T = 8, a placement with no host-side experts matches it in size.
- **What it does not exclude:** a defect smaller than that floor, since a defect of that size would look like the floor.
- **The limits of the read:** it covers one host CPU (an AMD EPYC 7532: AVX2, no AVX-512 or AMX, 64 kernel threads)
  and one split (8.0 / 3.0 GiB). The DRAM tier's reduction order on other instruction sets is not read (the PREREG's "No
  other host CPU").
- **What it licenses, with this look reviewed:** exactly what the verdict does. A doc or lane may propose serving at
  `solver` as an **opt-in** placement, citing this read and this look, never as a default. Any speed claim needs a read
  of its own. The argmax agreement stays below the 0.98 look-bar, and a citation says so.

## Predictions

| prediction | read | graded |
|---|---|---|
| S against A at every T: \|NLL gap\| <= 0.002 nats, argmax >= 0.98 | gaps +0.002077, -0.000047; argmax 0.974284 at both T | **MISSED**: the gap by 0.00008 at T = 1, the agreement at both T |
| S against A KL at most 0.001 at every T | 0.003925, 0.003729 | **MISSED**, about 4x: the tiered forward itself (C) moves the distributions more than a few roundings on the DRAM quarter |
| C against A at every T: \|NLL gap\| <= 0.0005, argmax >= 0.995 | gaps +0.000513, -0.000415; argmax 0.986328, 0.972005 | **MISSED**: the explore box's 10 of 10 greedy prompts did not carry over to 3,072 teacher-forced positions |
| T = 1 and T = 8 within 2x of each other on every S statistic | argmax 0.974284 at both (1.00x), KL 0.003925 / 0.003729 (1.05x); NLL gap +0.002077 / -0.000047 | **HELD for argmax and KL; MISSED for the NLL gap**, which changes sign (both inside the bar, the larger 44x the smaller) |
| verdict **SOLVER_SANE**, TRIPWIRE none | SOLVER_SANE, TRIPWIRE fired at both T | **HELD for the verdict; MISSED for the tripwire** |

## Reported, never ruling

- **Greedy token identity** (the teacher-forced argmax, per window): S differs from A at 79 positions at each T, in 23 of
  24 windows (window 3 at T = 1 and window 17 at T = 8 match A at every position). The positions per window are in
  `verdict_p233.json` (`quality.<T>.argmax_diff_positions`).
- **S's KL per window:** 0.00186 to 0.00653 (median 0.00376) at T = 1; 0.00211 to 0.00587 at T = 8.
- **The host:** AMD EPYC 7532 32-Core (64 CPUs, one NUMA node, AVX2 only), torch 8 threads, the CPU kernels 64 threads,
  the same in every phase.
- **Build and wall times** (an explore figure, never a speed claim):

  | arm | load | arena | T = 1 | T = 8 | arm wall | Amendment 4's term | CUDA allocator peak |
  |---|---|---|---|---|---|---|---|
  | A | 51.4 s | 12.9 s | 911.1 s | 229.9 s | 1,217.0 s | 1,026 s | 23.78 GiB |
  | S | 45.1 s | 14.7 s | 1,040.6 s | 546.0 s | 1,657.3 s | 2,400 s | 16.82 GiB |
  | C | 37.0 s | 12.8 s | 1,185.2 s | 653.5 s | 1,899.4 s | 2,400 s | 16.82 GiB |

  - **Phase A overran its re-fit term by 191 s,** inside its 2,400 s alarm. Every arm ran 1.43 to 1.51 times its
    shakedown projection (A 820.7 s, S 1,098.3 s, C 1,326.8 s). The shakedown host was a 96-core EPYC 9654; this one is
    a Zen 2 EPYC 7532. The run ended at 5,910 s against the 11,700 s guard.
- **Before the phases:** the download probe read 75.9 MB/s (floor 40). The guard fit left 11,587 s against the 11,157 s
  needed. The self-tests passed (`p233_reduce.py` 114 cases, `p233_box.py` 69, `p115_quality.py` 11), and so did the
  premise (GPU 37 passed, CPU 6 passed).
- **Re-derivation:** the verdict was re-derived from the private receipt store by `p233_reduce.py` at `2a0a9afc`, and
  it is byte-identical to the box's `verdict_p233.json` (sha256 `5da5967e…`).

## What follows (as registered)

- **A doc or lane may propose `solver` as an opt-in placement,** citing this read and its written look once reviewed.
  It is never a default, and it is cited with the agreement below 0.98.
- **No default changes here.** `all-vram` remains the point every certified serving number was measured at.
- **Any speed claim at `solver` needs a read of its own.** Decode at `solver` is eager, and this lane read quality only.

## What this does not say

- Nothing about speed. The phase times above are the instrument's passes, not the server's decode loop.
- Nothing about another split, model, card, release or host CPU: one budget (8.0 / 3.0 GiB) on one RTX 5090, Qwen3-30B-A3B
  at its pinned revision on the NF4 arena, grouped-nf4-gemm v0.50.0, one AMD EPYC 7532.
- Nothing about `E4B_PAGED_HOT_ROWS` other than 128 (Amendment 3's scope).
- Quality is P153's instrument on 24 wikitext windows x 128 positions per T, not a downstream task.
