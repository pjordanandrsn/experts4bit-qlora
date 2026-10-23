# HO3: a second box for HO2's cohorts, and the first host-offload held-out rows

This registers rented boxes that add rows to the held-out side of Loggetta's planner split. It changes no estimate, default
or gate, and it computes no band: its rows go to a new split version (`SPLIT-v3.json`, its own review), which Loggetta's
band reads.

## Why

HO2 (`bench/ho2/`) put two UNSEEN rows into each of four family × card cohorts (LFM2-8B-A1B and ERNIE-4.5-21B-A3B on an
RTX 5090 and an RTX 4090), but each cohort's rows came from ONE box. A band that rests on one machine cannot tell the
planner's error from that machine's. HO3 gives each of those cohorts its second box (Part A). Every held-out row so far
ran with the experts on the device, so HO3 also measures the first rows with the experts on the host (Part B).

## Stack and launch (bench/common rule 1)

- **Launch commit:** every box's kit and driver launch from this registration's merge commit.
- **Stack under test:** HO2's, exactly, so a cohort's two boxes differ only in the machine. The runner pins these as
  literals, separate from the launch commit:
  - experts4bit-qlora `f6e66fd65b28fe493a89062b02aa91d3970ff995` (0.56.1, HO2's registration merge);
  - grouped-nf4-gemm `a83f6bb3` (0.49.0), Loggetta `9348f8b8` (0.6.2);
  - transformers 5.17.0, bitsandbytes 0.50.2, the image's torch 2.8.0+cu128.
- The install, the tripwire and the four `loggetta train` rows are HO2's, unchanged (`tests/test_ho3.py` checks them
  against `bench/ho2/ho2_run.sh`).

## Rows

On each box, HO2's four rows: each family at 1,024 and 2,048 tokens, micro-batch 2, 20 steps, Alpaca's text column at
its pinned revision, the planner's own plan (no `--fix`, no `--observations`). Before any row trains, the box also runs
`loggetta plan` for each row and keeps the plan: the box-side record of what the box planned.

## Part A: the second box (HO2 replication)

- **`ho3-5090-1`:** one RTX 5090 (cc 12.0, ≥ 30,000 MiB), declared at $1.05/h, a 2 h guard (7,200 s).
- **`ho3-4090-1`:** one RTX 4090 (cc 8.9, ≥ 23,000 MiB), declared at $0.75/h, a 2 h guard.
- **Machines:** never HO2's. Its rows ran on machines `124996` (the RTX 5090) and `154091` (the RTX 4090). The driver
  refuses both, and any box whose machine the launcher did not record, before anything is staged.
- **Prediction:** HO2's own plans, which were byte-identical across the two cards: device residency, plan device totals
  lfm2_1024 9,768,170,758, lfm2_2048 9,386,279,379, ernie_1024 21,856,438,198 and ernie_2048 20,886,295,683 bytes.
  All four rows COVERED on each box.

## Part B: host offload on an RTX A4000

- **`ho3-a4000-1` and `ho3-a4000-2`:** two RTX A4000 boxes (cc 8.6, 16 GB, ≥ 16,000 MiB) on two DIFFERENT machines,
  declared at $0.25/h, a 3 h 10 min guard (11,400 s). On a 16 GB card the planner keeps ERNIE's experts on the host.
  Each box is pinned at create to a machine no earlier A4000 box used. The driver also refuses an A4000 box on the
  machine of any earlier A4000 box that ran the runner, and refuses when such a box's machine is unreadable.
- **Cohorts:** ERNIE × A4000 (host offload) and LFM2 × A4000 (device), each two rows per box and two boxes. Each family's
  result is reported on its own, so a fault in one never reads as the other, and Part B never touches Part A's verdict.
- **Prediction (before any measurement):** Loggetta 0.6.2's plan on a 16 GB RTX A4000 profile (16,376 MiB):

  | row | residency | plan device total (bytes) | plan host total (bytes) |
  |---|---|---|---|
  | lfm2_1024 | device | 9,768,170,758 | 3,221,225,472 |
  | lfm2_2048 | device | 9,386,279,379 | 3,221,225,472 |
  | ernie_1024 | host | 9,747,209,139 | 15,451,815,936 |
  | ernie_2048 | host | 8,777,066,624 | 15,451,815,936 |

  These were derived OFF-STACK, on a development workstation: HO2's pins except CPU torch 2.14.1 (Python 3.14;
  torch 2.8 has no build for it). The plan runs no kernels. Every row COVERED, and each ERNIE row's host peak inside its
  host estimate.
- **The box's plan governs.** Each row runs the plan the box makes. When the receipt's plan differs from the table above
  in residency or either total, the reducer records the difference beside the row as a finding. It never changes the
  row's outcome and never refuses the box.

## The rule (per row, HO2's)

From the receipt's `comparison`: **COVERED** if the plan total ≥ the sampled driver peak AND the allocator estimate ≥ the
allocated peak; **UNDER** if the driver peak is above the plan total; **SHORT** if the allocated peak is above the
estimate. A row with the experts on the host is also **HOST_UNDER** when its host peak is above its host estimate. A row
can carry several. A family on a box is COVERED when both its rows are, MISS otherwise. The box is COVERED when both
families are. A row is VOID, and its family on that box is VOID, on any of HO2's VOID conditions: not exactly one receipt,
status not OK, model, revision, workload, data or card differs from this registration, fewer tokens trained, or an
overhead line that is not the inferred default. Every gate fails closed (bench/common rule 7): a row whose receipt
records no expert residency, and a host row without its host estimate and peak, are VOID too. A row with no receipt is
UNREAD only when the runner recorded a deadline skip for it (`SKIPPED train <row>`); without that record it is VOID. A
family with an UNREAD row and no VOID row reads UNREAD; the box reads VOID if either family is VOID, else UNREAD if either
is UNREAD.

`bench/ho3/ho3_reduce.py --self-test` holds HO2's 17 cases and 11 of HO3's; `tests/test_ho3.py` the rest.

## Run ids, outcomes, caps and the stop (bench/common rules 2–4)

- **Reading ids:** `ho3-5090-1`, `ho3-4090-1`, `ho3-a4000-1`, `ho3-a4000-2`.
- **Re-run ids:** `ho3-5090-2`, `ho3-4090-2`, `ho3-a4000-3`, for a box that follows a STARTED box that was not read.
  With the reading ids they cover every start the cap allows.
- **Retry ids:** `ho3-<card>-r1` to `-r8` per card, for a box that follows one that did not start. Ids are labels: no
  code reads a number out of one.
- **Outcome classes** are bench/common's table, decided from the launcher's receipt and the runner's exit, never by hand:
  refused at create (counts toward nothing), NOT_RUN (nothing), refused before the install (a host refusal; a single
  rc 41 is no strike, two consecutive are one), started (counts toward the cap; a strike only when not read).
- **Started-box cap:** 2 RTX 5090 boxes, 2 RTX 4090 boxes, 3 RTX A4000 boxes (two readings, one re-run).
- **The stop:** two strikes in a part end that part. A read box is never a strike.
- **Worst:** each started box at its guard and declared rate plus a 65 GB download allowance at no more than $0.02/GB
  ($1.30): RTX 5090 $3.40, RTX 4090 $2.80, RTX A4000 $2.09. At the caps that is $18.67 for started boxes. A box that
  never starts is billed up to its card's boot slack, where the fit check refuses it (rc 41): 1,988 s on the RTX 5090
  and RTX 4090, 1,088 s on the RTX A4000. Each card has 8 ids beyond its cap (10, 10 and 11 ids against caps of 2, 2
  and 3): RTX 5090 8 x $0.58 = $4.64, RTX 4090 8 x $0.41 = $3.31, RTX A4000 8 x $0.076 = $0.60, $8.56 in all. The lane
  worst is $27.23, under the $35 run cap; each box is under $15. Expected about $2.50.

## Fit and boot slack (bench/common rule 5)

The runner's fit is the install to the gate, the fetch, the box-side plans, four rows, the reduce and the reserve:
4,170 s on the RTX 5090 and RTX 4090 (480 s per row, HO2's) and 8,250 s on the RTX A4000. Its 1,500 s per row is an
ESTIMATE: host offload has never run on this stack, so no measured host stands behind it. A row that would not fit in
the time left is skipped before it starts, and a row that reaches its own 1,800 s alarm is stopped. The runner records
both as a deadline skip, so the row reads UNREAD and the box keeps its finished rows. With the 25 % headroom, each guard leaves at least 540 s for the launcher's ssh wait and the staging:
7,200 − 5,212 = 1,988 s and 11,400 − 10,312 = 1,088 s. `tests/test_ho3.py` asserts it.

## When it closes (bench/common rule 6)

The results record names the frozen commit at which this lane's pins were last asserted. From then on, the tests read
the staged hashes and pins with `git show <frozen>:<path>`, never the live tree.

## What happens to the rows

COVERED, MISS or VOID, every valid row goes to the held-out side of `SPLIT-v3.json` (Loggetta, its own review), with its
box and machine, so the band can require two boxes per cohort. HO3 computes no band.
