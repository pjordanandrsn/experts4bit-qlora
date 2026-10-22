# P222 results: K25's decode x plan at Mixtral-8x7B's W16 shapes (GO, M 2.2678x over the served default)

**The read** (`p222-5090-5`, one RTX 5090, 2026-10-22; receipts under
[`receipts/p222-5090-5/`](receipts/p222-5090-5/)). At Mixtral-8x7B's W16 shapes, K39's decode at the plan pair a sweep
selected (M: gate_up 64/256/4/2, down 32/256/4/2) ran both K25 GEMMs **2.2678x** faster than Mixtral's served default,
the tree at today's plan. That is the median of 120 paired reps on routing draws the selection never saw. The lower
bound is **2.2624**, and the clock-adjusted bound is the same, against the 1.10 bar: **GO.** Every arm kept the tree's
bits on every rep.

This is a microbench premise. It opens a served read and quotes no gain into the served step.

The registration is [`PREREG-p222.md`](PREREG-p222.md) with its three amendments. The launch commit is Amendment 3's
merge, `683b85f2`. The run used grouped-nf4-gemm v0.50.0 at `41ef086e`, on torch 2.8.0+cu128 and triton 3.4.0.
The verdict was re-derived from the private receipt store, and it re-derives from the public bench record here
([`REDERIVE.md`](REDERIVE.md)).

The host was found by a power probe. The screen admits only machines with a recorded power cap of at least 575 W, and no
such machine had been listed for hours. An explore-tier probe booted this machine for about a minute and read its cap
(575 W). The box then re-read its own limit before the install (see the receipt's README).

## The numbers

| | |
|---|---|
| T, the tree at today's plan, both GEMMs of 8 layers (block B) | 11.365 ms |
| **M** (block B) | **4.951 ms** |
| **M's median T/M** (SE, lower bound, clock-adjusted) | **2.2678** (0.0027, 2.2624, 2.2624) |
| C1 (e4b's `c1`) over T | 2.0630 |
| P over T, the decode alone at today's plan | 1.0187 |
| M over C1 | 1.0999 |
| self-pair T/T2 | 0.9999 |
| cost | $0.14 (worst registered $1.69) |

- **Where the gain is:** almost all of it is the plan. The decode alone moved today's plan 1.9 %. Under the same
  decode, c1's pair read 2.0630x, and M a further 1.0999x over c1. M takes c1's gate_up; its down plan, 32/256/4/2,
  differs from c1's 128/128/8/3.
- **The served arm is M** by the registered rule: M is neither today's plan nor c1's pair, and M over C1 is above 1.02.
- **The sweep:** all 288 plans (144 a projection) kept the tree's bits, and none faulted. The selection margins were
  1.0154 for gate_up, a near tie, and 1.1293 for down.
- **Clocks:** no arm is labelled. Every arm's median SM clock read 2707 MHz and its memory clock 13801 MHz. The software
  power-cap event is in 0.719-0.723 of every arm's samples, and the enforced limit read 575 W in every sample.

## The predictions, graded

| prediction | read | grade |
|---|---|---|
| the decode: T/P in 1.15-1.35 | 1.0187 | MISSED (below) |
| the headline: T/M in 1.30-1.90 | 2.2678 | MISSED (above) |
| M is not c1's pair | gate_up is c1's, down is not | HELD |
| the self-pair within 0.005 of 1 | 0.9999 | HELD |
| M's median SM clock within 2 % of T's | 1.0000 | HELD |
| every arm keeps T's bits on every rep | all of them | HELD |

- **The decode prediction missed below.** At Mixtral's shapes, K39's decode is worth 1.9 % at today's plan, against
  Qwen3's 17.6 % (P156). The registration named the reading of a low miss as "Mixtral's pair is bound elsewhere". This
  read attributes nothing further: it times a decode and a plan, not a cause.
- **The headline missed above, and not through the clocks:** every arm ran at the same median SM and memory clock. The
  registration's unverified arithmetic put today's pair at about 0.5 TB/s against a ~1.5 TB/s copy rate.
  - On the same arithmetic, this read's T streams the 8 layers' 6.34 GB of stores at about 0.56 TB/s, and M at about
    1.28 TB/s.
  - That uses all 8 experts a layer; B's draw touches 7.918 on average.
  - It is arithmetic from the bench's own bytes and milliseconds, not a measured bandwidth. It is in the range the
    registration's arithmetic allowed. The prediction band was narrower than that arithmetic.

## Caveats

- **Microbench only:** the two K25 GEMMs alone, with no attention, router or glue. The pair is 0.911 of Mixtral's W16
  step (P145), but this read quotes no step figure.
- **Routing:** seeded and uniform, two experts of eight per row. That is not Mixtral's router.
- **Inputs:** random NF4 bytes, compared decode against decode.
- **Scope:** one RTX 5090 at 575 W, 16 rows, 8 layers, triton 3.4.0.
- **The gate_up choice was a near tie** (runner-up 1.5 % slower on block A). M's gate_up equals c1's, so the near tie
  does not change what a code PR would add, which is the down plan.
- **A plan is its own compile:** the ratios price each plan's whole compile. That is what a served change would ship. No
  stage of it is attributed.

## What follows

- **An e4b PR:** add M's pair to `_K25_PLANS`. It is opt-in, carries no speed claim, and its tests are the
  plan-dispatch contract.
- **A served read, its own registration** on P158's and P161's kit: M's pair against Mixtral's served default, at
  Mixtral's real prompts and routing, with its own bars. Its token gate compares identical schedules only.
- **The host screen for that read:** pin by a recorded power cap, and keep this lane's power-probe path available for
  refreshing caps within the 3-day window.
