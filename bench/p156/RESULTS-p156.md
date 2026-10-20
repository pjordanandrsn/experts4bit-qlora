# P156 results: K39's decode is a W16 lever (GO, 1.1755x)

**The read** (`p156-5090-1`, one RTX 5090, 2026-10-20; receipts under [`receipts/p156-5090-1/`](receipts/p156-5090-1/)).
K39's decode (K25's `lut="prmt"`, grouped-nf4-gemm#569) at today's plan ran both W16 expert GEMMs **1.1755x** faster than
today's select tree, over the 64 held-out steps of S0's real router picks. That is the median of 64 paired, rotated
reps, with an SE (of the median) of 0.0014 and a **lower bound of 1.1728 against the 1.05 bar: GO.** P moved no bit on
any step. This is a microbench premise: it opens the served read and quotes no gain into the served step.

The registration is [`PREREG-p156.md`](PREREG-p156.md) (e4b#1857), and the launch commit is its merge, `101ef5d7`. It ran
grouped-nf4-gemm main at `68bbefc4` (#569's merge, after v0.49.0; K39 unreleased), on torch 2.8.0+cu128 and triton
3.4.0. The verdict was re-derived from the private receipt store, byte-identical to the box's.

## The numbers

| | |
|---|---|
| T, the tree at today's plan, both GEMMs a step | 12.872 ms (P152's census: 11.748; P155's microbench: 12.075) |
| P, prmt at today's plan | 10.962 ms |
| **P's median T/P** (SE, lower bound) | **1.1755** (0.0014, 1.1728) |
| the self-pair T/T2 | 0.9985 (SE 0.0004) |
| P at P152's scale | 9.994 ms, against the 4.94 ms byte floor |
| P's share of P154's 6.81 ms headroom | 25.75 % |
| the models containing P's median | M3 (serial issue, 1.15-1.18) only |

- **The contract on sm_120:** all 144 prmt plans ran in each projection, every one bit-identical to T, and P and every
  carried plan kept T's bits on all 64 reps. P155's tree sweep lost 18 plans a projection to shared memory; K39's
  decode needs 6,400 B at today's plan against the tree's 14,592 B.
- **The compiled metadata:** T 80 registers and P 64, no spills, so 50 % and 67 % theoretical occupancy.

## The predictions, graded

| prediction | read | grade |
|---|---|---|
| T's both-GEMM microbench step is 9.5-14.0 ms | 12.872 ms | HELD |
| **the headline: P's confirmed median T/P is 1.05-1.18** | 1.1755 | **HELD** |
| P and every swept plan keep T's bits | all of them, every step | HELD |
| T 80 and P 64 registers, no spills | as predicted | HELD |
| the self-pair within 0.005 of 1 | 0.9985 | HELD |
| the sweep's curse: its best plan's selection-over-P exceeds its confirmation by 0-0.03 | 1.770 against 1.617, a curse of 0.153 | MISSED |

**The model:**
- **The band:** P's median sits in M3's band, the serial-issue model, in which every instruction the decode saves is
  time saved. The static count predicted that band: 24.1 against 17.0 lane instructions per weight, both at triton
  3.4.0.
- **M1 is out:** the overlap model with a fixed residual (1.02-1.11) does not contain P.
- **M2 is out:** neither does the overlap model with occupancy hiding the residual (1.19-1.28).

## Caveats

- **Clocks.** The GPU was sampled at the bench's start and end only:
  - 2,400 MHz as it started;
  - 2,272 MHz at 480 W as it ended, with the clock-event reason 0x4 (software power cap).

  The 0x4 sample has two readings: a transient over the 575 W cap, or an enforced limit that dropped to about 480 W.
  The second has never been observed inside a rental, and the box read the 575 W limit before any install. P156
  records no per-arm clocks, and a lighter kernel can hold a higher boost, so this record cannot exclude a clock share
  of the 1.1755. **The GO does not depend on it.** Take the two samples as the extremes (2,400 / 2,272 = 1.0563)
  and give the whole ratio to P under 1:1 clock scaling, the conservative (upper) assumption. P's median is then
  1.1128, with a lower bound of 1.1103, still >= 1.05. The bound rests on those two samples, not on a per-arm trace. The
  served read measures end to end, and P157's per-arm clock rule covers it at the microbench level.
- **The self-pair** read 0.9985, 3.3 SE below 1: a small bias between two captures of the same work. The headline is a
  ratio of two arms timed the same way (rotated capture and replay positions, replays interleaved arm by arm), so a bias
  the arms share cancels in it. The 0.15 % is two orders below the effect.
  - **The predecessor never varied:** a cyclic rotation keeps each arm's predecessor in the replay order. P follows T2
    in 629 of its 640 replays and T follows C3 in 629; the other 11 of each are the arm's first replay after the
    captures. So **this record cannot bound a carry-over from the predecessor**, in the self-pair or in T/P.
  - **What it does show, post hoc on the registered record (not a re-read):** position moves T/P only within
    1.1713-1.1808 (the median by rotation; the 11 reps that list P first read 1.1782, the other 53 read 1.1753). The
    rotation balances position, so the GO's lower bound does not depend on it.
  - **The next lane fixes it:** P157's harness replays an untimed spacer graph before every timed replay. Its A2000
    shakedown found the same carry-over, larger under that card's power cap.
- **Microbench only:** the two K25 GEMMs alone, with no attention, router or glue. bf16 teacher-forced picks, not the
  served NF4 picks. One RTX 5090, at triton 3.4.0 (at 3.8.0 the static counts move: grouped-nf4-gemm#570).

## Reported, never ruling: decode x plan

The prmt sweep carried three plan pairs (today's plan excluded, since it is P), and each confirmed far above P:

| arm | plan (gate_up / down) | median T/arm | over P |
|---|---|---|---|
| C1 | 64/256/4/2 / 128/128/8/3 | 1.9019 | 1.617 |
| C2 | 64/256/4/3 / 32/256/4/2 | 1.8313 | 1.557 |
| C3 | 64/256/8/3 / 128/128/4/2 | 1.8652 | 1.585 |

- **The cause is the decode, not the plan alone.** These plans run under the tree too, and there they are slower than
  today's (P155's record, `bench_p155.json`):
  - gate_up 64/256/4/2 read 14.23 ms in P155's selection, against today's 8.32, at 255 registers and 82,944 B of
    shared memory;
  - down 128/128/8/3 read 5.29 ms, against 3.37, at 127 registers.

  The select tree's register cost is what kept the big tiles from paying, and K39's lighter decode lets them pay.
  P155's "a plan is not the lever" was a reading of the tree.
- **Held for its own read:**
  - C1 was selected with a curse of 0.153: the prediction's band was 0-0.03, MISSED.
  - Its 1.90x over T is a reported figure that needs its own held-out read.
  - A served change would move the decode and the plan together, never the plan alone, which the tree runs slower.

## What follows

- **The served FLIP read** for `lut="prmt"` as qwen3_moe's K25 decode at W16, registered as its own lane with its own
  bars (P143's pattern: the microbench, then the served read). It needs a grouped-nf4-gemm release carrying K39, or a
  pinned commit as here. It quotes no gain from this record.
- **Decode x plan** as a later lane: C1-C3's plans with prmt, on their own held-out read.
- **No default changes.** K39 stays opt-in with no speed claim until a served read.
