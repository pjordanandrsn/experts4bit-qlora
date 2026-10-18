# P138, the verify split (#1313): where SD2's 1 → n row verify increment goes, on one card's own numbers

**The lane is P138**, numbered at the registration's merge (`1026c439`, #1727). Its files keep the `vsplit` names,
and its run ids are `vsplit-5090-N`. TC1's PREREG uses "P138" only as one of its own prediction labels, a separate
namespace.

## Why this lane exists

**P132 asked this question but could not run.**
- P132 (bench/p132/PREREG-p132.md) asked this question against `sd2-5090-3`'s card. Its rule compared absolute
  milliseconds with that card's (verify within 5 %, the increment within 10 % of the desk's target), and Amendment 2
  refused every other card.
- That card (machine 55913, offer 40421555, GPU-38c66e15) did not return: **0 listings in about 4,050 reads over
  34.5 h** (2026-10-16 18:52Z to 10-18 05:20Z). The machine's other card was listed 71 times.
- **P132 closes "not run: the reference card was unavailable".** That is not a strike.

**A cross-card rule cannot be rescued by normalizing.**
- Another RTX 5090 reads 13–15 % faster at B = 1 on a matched workload (P116: `p116-5090-1` vs `p116-5090-2`, the
  same package bytes).
- Normalizing a multi-row step by the B = 1 anchor fails on that same pair: the one-row and multi-row phases scale
  differently (+7.7 to +10.8 % against a 5 % band).

**This lane therefore takes every number from the box it runs on.** It runs on any RTX 5090, and nothing is compared
with another card.

## The question (P132's, unchanged)

**What SD2 found.** SD2's verify of n rows costs I(n) = verify_ms(n) − anchor more than one plain decode step. That
increment rules out every multi-token single-stream lever on this server (SD2: VERIFY_COST_REFUTES).

**The fork.** A kernel inventory found the expert path switching kernels between one row and two or more:
- **one row:** the bandwidth GEMV;
- **two or more rows:** the device tile table plus the small-M grouped GEMM.

**What this lane decides:**
- **H:** how much of the increment beyond the extra experts' bytes lies in the expert path.
- **BUILD:** whether the part a 2–4-row expert kernel could save reaches what SD2's speed bar needs.

## Inputs and outcome

The maintainer's condition 1 is that the outcome is not computable from the inputs.

| | measured | feeds |
|---|---|---|
| **inputs** | two unprofiled timing passes of SD2's V1 terms (anchor A/A, verify_ms(n), full_ms(k)), in separate processes | anchor, I_b(n), need(n) |
| | the n = 1 census's moe_expert share | bytes(n) |
| | SD2's acceptance TAU_SD1 and speed bar G_BAR (`bench/sd2/sd2_reduce.py`, staged); card-independent | need(n) |
| | rho(n), the distinct-expert ratio from the four registered routing traces (P132's desk); card-independent | bytes(n) |
| **outcome** | the n = 2, 3, 4 censuses' class shares (moe_expert + moe_route) | dExp(n), S(n), H, BUILD |

**Definitions:**
- `need(n)` = the second-smallest over SD2's three workloads of full_ms(k) − TAU_SD1[w][k] × anchor / G_BAR, with
  k = n − 1. This is P132's formula with this box's numbers. On `sd2-5090-3`'s own V1 terms it reproduces P132's
  registered thresholds exactly (`tests/test_vsplit.py`).
- `bytes(n)` = (rho(n) − 1) × share_moe_expert(1) × anchor.
- `dExp(n)` = (share_moe_expert + share_moe_route)(n) × verify_ms(n) − (the same)(1) × anchor.
- `S(n)` = dExp(n) − bytes(n).

**The anti-circularity test.** With every input held fixed, changing only the n ≥ 2 shares takes the reading from
SUPPORT / BUILD to REFUTE / NO_BUILD. This is in the reducer's self-test and in `tests/test_vsplit.py`.

**Milliseconds come only from the unprofiled passes** (the maintainer's condition 2). The Nsight census contributes
shares of its own profiled period (terms / P_ms). A class's milliseconds are its share × the unprofiled step of the
same n: the anchor at n = 1, verify_ms(n) at n ≥ 2. Inflating every census term by one factor changes no millisecond
the rule uses (tested). The census-sum vs unprofiled ratio is REPORTED.

## The rule (`bench/vsplit/vsplit_reduce.py`, self-tested)

**VOID (the first rung that applies)** (the maintainer's condition 3):
- a record is missing or not ok, or the GPU tests failed;
- the target, kernel package or model revision is off the registration;
- a bucket of 1..4 was not captured, or the canary failed;
- a census arm is void, or labelled CLASS_MAP_INCOMPLETE, CLASS_MAP_SEGMENT_BROKEN, NSYS_DIAGNOSTIC_ERRORS or
  CLOCK_MISMATCH;
- a census window did not bracket exactly 64 replays;
- in either timing pass, anchor < verify_ms(2) < verify_ms(3) < verify_ms(4) does not hold.

VOID is for record and instrument failures only. An increment that is all bytes is an answer, below.

**NOISY:**
- in either pass, the A/A anchors differ by more than 3 % (SD2's ANCHOR_TOL); or
- the two passes (the self-pair) differ by more than 3 % on the anchor or on any verify_ms(n).

**READ** otherwise, with two readings. Only a READ carries the top-level H and BUILD; a NOISY verdict keeps them under
`outcome`, for reporting only, so no label can be quoted from it.
- **H:** share(n) = S(n) / [I_b(n) − bytes(n)], the part of the increment beyond its bytes that lies in the expert path.
  - **The excess floor:** at an n where I_b(n) − bytes(n) ≤ max(|I_b,A(n) − I_b,B(n)|, 0.05 × I_b(n)) there is nothing
    measurable beyond the bytes: H(n) = NO_EXCESS, and that n is left out of the label. The first term is the two
    timing passes' own spread at n; the second (EXCESS_FRAC = 0.05, named here before any data) keeps a lucky-tight
    self-pair from shrinking the floor to zero. Without it, share = S / (I_b − bytes) can read SUPPORT on noise when
    the excess is small but positive.
  - Over the remaining n: SUPPORT if share ≥ 0.70 at every one; REFUTE if share < 0.50 at every one; INCONCLUSIVE
    otherwise.
  - NO_EXCESS at every n gives **H = NO_EXCESS**: a reading that closes the lane (a 2–4-row kernel has nothing beyond
    the bytes to save), not a failure and not a strike.
- **BUILD** iff S(n) ≥ need(n) for some n that carries an excess; otherwise NO_BUILD. A NO_EXCESS n never counts,
  even if the census puts its dExp above I_b, so BUILD is always NO_BUILD when H is NO_EXCESS.

Real records do not trip NO_EXCESS: sd2-5090-3's increment against the bytes term built from P123's real B = 1
census share carries an excess above the 5 % floor at every n (`tests/test_vsplit.py`).

**REPORTING, never a rung:**
- census-sum vs unprofiled;
- each class's increment as a fraction of I_b(n), beside the prediction;
- the unattributed remainder;
- the card's uuid (`card.json`).

## The registered predictions

These are fractions of I_b(n), registered before any data. They are P132's desk prediction v2 (`bench/p132/
desk_out.json`, accepted as final on 2026-10-11) restated as fractions of that card's I(n) by
`bench/vsplit/desk_fractions.py` into `desk_fractions.json` (tested equal). Each band is lo / central / hi.

| n | bytes / I | route / I | dense / I | attn + norm / I | share under H | S / need under H |
|---|---|---|---|---|---|---|
| 2 | 0.180 | 0.042 / 0.053 / 0.064 | 0.000 / 0.095 / 0.170 | 0.008 | 0.782 / 0.873 / 0.990 | 1.120 / 1.251 / 1.418 |
| 3 | 0.294 | 0.038 / 0.048 / 0.058 | 0.000 / 0.086 / 0.154 | 0.015 | 0.760 / 0.856 / 0.979 | 1.453 / 1.636 / 1.870 |
| 4 | 0.380 | 0.035 / 0.044 / 0.054 | 0.000 / 0.080 / 0.143 | 0.021 | 0.735 / 0.836 / 0.966 | 1.431 / 1.629 / 1.882 |

**The predicted reading is H SUPPORT and BUILD at every n.** The fractions are reported beside the reading; the
reading itself is decided by the rule above.

## The subject and the instrument

**The subject is P132's, unchanged:** SD2's read server.
- e4b 88ae1cfd (SD2's read target, re-proven on the card by SD2's package-diff audit and its GPU tests) and
  grouped-nf4-gemm v0.45.0 724ccc45.
- E4B_PAGED_SPEC=eagle3 at k = 3 with the pinned head, 16 slots, graphs on.
- Qwen3-30B-A3B at ad44e777, on any RTX 5090.

**The instrument is P132's box, byte-identical** (`bench/p132/p132_box.py`, pinned to the same sha256 as in
`staged-p132.sha256`). It runs under `bench/vsplit/vsplit_run.sh`, which is P132's runner with three changes:
1. **No card-uuid refusal:** any RTX 5090; the uuid is recorded.
2. **A second unprofiled timing pass** after the census, in a fresh process: the self-pair.
3. **This reducer.**

The order is:
1. refusals; install and audit; self-tests; GPU tests; fetch; bake;
2. the canary (with P132 Amendment 1's floor: 12 replays, 8 kept);
3. timing pass A;
4. the census at n = 1..4 (64 replays each, graph then node, class map v1.2);
5. timing pass B;
6. reduce.

## Scope

The maintainer's condition 4: **one card's verdict licenses BUILD (a 2–4-row expert kernel's own registration) or
kills the lane.** It ships no default. Any default needs a second card's read.

## Launch and budget

The maintainer's condition 5:
- **Card:** any RTX 5090 that meets the bar (receipts ranking, clean runs ≥ 5, reliability ≥ 0.5), at or below the
  class rate, pinned by machine and offer set under the market rules.
- **Launcher:** the launch tooling at or after its provider-side machine-pin filter, with `--validate` at its HEAD
  on the exact manifest.
- **Host bar:** inet_down ≥ 1000 Mbps (the fetch is about 62 GB: Qwen3-30B-A3B plus the head).
- **Disk:** the container disk ordered (`vast_disk_gb`) ≥ 170 GB, the runner's 150 GB floor + 20. Free space after the
  image and installs is always below the order, so ordering exactly the floor fails the runner's own disk check.
- **Guard:** 2.0 h (`--wallclock-h 2.0`; the runner's GUARD_S = 7200).
- **Cost:** worst = guard × the class-rate cap + the download budget × the $/GB cap = 2.0 h × $0.85/h + 100 GB ×
  $0.011/GB = **$2.80** (the launcher's default download budget, above the about 62 GB fetch); expected about $1.5.
- **Nothing launches before the maintainer's ACK of the merged registration**, after an independent review by a seat
  that did not write it (the maintainer's condition 6).

## The stop rule

- **READ:** the maintainer re-derives the verdict with main's reducer before anything is quoted.
- **NOISY:** one rerun on another host, under a new ACK.
- **A deadline skip** (the runner's STOP-2: exit 40 before the reduce step, with `SKIPPED … host-limited deadline` in
  `summary.txt`): an incomplete attempt, not a strike, although reducing what exists would VOID on the missing records.
  A slow host is not a measurement. It reruns on another host, under a new ACK.
- **VOID, or a second NOISY:** a strike for this lane. The second strike closes it with a simplification note.

## What this lane cannot say

- Nothing about cards other than the one it ran on, and nothing about quality: no tokens are judged.
- Nothing about B > 1 batched decode of independent sequences. Its routing differs from a verify's consecutive-token
  routing.
- A BUILD licenses building the kernel, not shipping speculation: SD2's E4B_PAGED_SPEC stays off until a new read.

## The consequence

- **BUILD:** a 2–4-row expert kernel (the bandwidth GEMV's design past one row) gets its own registration. That
  registration has a bitwise gate against the current K25 path and a re-read of SD2's verify cost.
- **NO_BUILD:** the B = 1 multi-token lane closes for Qwen3-30B-A3B on this stack, with the measured split as its
  record.
