# P228: K1's selection and microbench, a pinned cuBLASLt GEMM for o_proj and the router at 2 to 8 rows (lever K)

**Registered before any reading.** The lane selects, on one RTX 5090, the cuBLASLt configuration that `E4B_DENSE_LT_PIN`
(e4b#2019) should pin for Qwen3-30B-A3B's `o_proj` and router gate at 2, 4 and 8 rows. It then reads whether the pinned
call beats the served default at those rows. **Each site is ruled on its own.**

| the maintainer's condition (2026-10-22) | where |
|---|---|
| the premise runs K1's GPU suite on the 5090 itself before the selection (sm_120's list and kernels differ from sm_86's) | [Budget](#budget-and-the-fit) |
| the exploration's ids are candidates only; the box re-selects | [The selection](#block-a-the-selection) |
| the tolerance as a number | [The rule](#the-rule-p228_reducepy-fixed-before-data) |
| one 575 W 5090, pinned host, rent-watch at create | [Budget](#budget-and-the-fit) |

## Why

The design note (`bench/dense-small-m/DESIGN.md`, e4b#2016) gives the case. At 2 to 8 rows, cuBLAS's own kernel choice
for `o_proj` and the router gate runs slower than its one-row choice, while cuBLASLt's own heuristic list holds a
configuration that does not slow down. K1 (e4b#2019) pins such a configuration per key and refuses everything else. Its
pin table ships empty, and this lane fills it.

**The question, per site:** at M = 2 and M = 4, does the pinned call run in at most **0.85** of the served default's time
(`F.linear` at the same M), within the served tolerance?

## The sites

| site | (N, K) | copies | why that many |
|---|---|---|---|
| `o_proj` | (2048, 4096) | 48 | one per layer: 805 MB per replay, far beyond L2 |
| router gate | (128, 2048) | 384 | 0.5 MB each: 384 copies (201 MB) so a replay streams past L2, as the router does in a step between larger reads |

Each copy is a distinct random bf16 weight, and every arm reads the same copies. A replay times all copies in one CUDA
graph, and the reducer divides by the copy count.

## Block A, the selection

- **The candidates:** per (site, M in 2, 4, 8), every configuration on cuBLASLt's own heuristic list for the exact
  descriptors. That is K1's `LtLib.heuristics`: 64 requested, INPLACE reduction excluded.
- **Each candidate is:**
  - checked (`cublasLtMatmulAlgoCheck`);
  - run twice (**bitwise** equal required);
  - compared with `F.linear` (**within 1.5e-2** required);
  - timed: one graph of every copy's call, 5 replays on each of 8 input draws (seed 2280 + M).
- **`p228_reduce.select`, fixed before data:**
  - among eligible candidates, the lowest mean over the 8 draws of each draw's median replay;
  - a scheme-2 candidate (fp32 partials) within **1.02x** of the fastest is preferred (the fastest such);
  - ties go to the lower heuristic index.
  - The runner-up margin is reported.
- **Candidates only:** the exploration's 400 W ids are not used. The box selects from its own list.

## Block B, the reading

The selected pins go into `dense_lt_pin.LT_PINS` under the loaded key, and e4b's own `apply_dense_lt_pin` installs them
on toy modules holding the same weights. `o_proj` becomes `PinnedLinear`; routers carry the epilogue's mark and are
called through `router_linear`. The bench therefore times e4b's real pinned path.

| arm | what | role |
|---|---|---|
| **D** | `F.linear` at M (cuBLAS's own choice) | the served default: **the control** |
| D2 | D captured again | the self-pair |
| **P** | the pinned call, through e4b's installed path | **the registered question** |
| T1 | `F.linear` at one row | the one-row served path: reported |

- **The reps:** 60 per (site, M), 15 whole Williams squares of four arms.
  - Each rep copies its step's input (seed 22800 + M) into the static buffer the graphs read.
  - It replays in its Williams order: 10 timed replays per arm, each right after an untimed replay of the spacer S
    (D's work).
  - NVML samples clocks, power and the enforced limit, tagged `site|M|arm`.
  - Then D2 is compared with D bit for bit, and P with D.
- **The engagement:** one eager pass per arm before the reps. P's must count exactly `copies` pinned calls in e4b's
  `SEEN`; D's and T1's must count none.

## The rule (`p228_reduce.py`, fixed before data)

- **Per rep:** each arm's median of its 10 replays, divided by the copy count. Ratios are the median over 60 reps, with
  P155's larger-of SE (seed 228). The upper bound is median + 2 SE. The clock-adjusted bound scales the upper bound up by
  P's median SM clock over D's when P's is faster; P is never credited.
- **VOID, the whole record:**
  - not a 5090 at cc 12.0;
  - the arms, spacer or replays not as registered;
  - the cuBLASLt version unread;
  - a poisoned context or an arm fault (including "no eligible candidate");
  - unread clocks;
  - **any enforced power limit under 575 W**.
- **VOID, one site:**
  - its shape or copies not the registered ones;
  - block B timed a pin other than the one `select` names from block A's record;
  - the engagement not as stated;
  - a rep's order not its Williams row, or a rep without its spacer replays;
  - **D2 differs from D by a bit**;
  - D/D2's median outside 0.99-1.01 at any M.
- **TOLERANCE_FAIL, one site:** P against D beyond **max |P - D| / max |D| <= 1.5e-2** in any rep. It is final: read,
  never re-read, not a strike, and no timing is ruled on.
- **PASS, one site:** at **both** M = 2 and M = 4, P/D's upper bound **and** its clock-adjusted upper bound are
  **<= 0.85**.
- **MISS, one site:** otherwise. M = 8 never rules; its pin is still selected and recorded.
- **A site's VOID** leaves the other site's verdict standing. A run with no site read is not a reading (rc 27).
- **Reported, never ruling:**
  - P/T1 and D/T1 at every M;
  - each arm's per-call time;
  - the byte floor (bf16 weight bytes over the copy rate);
  - the selection margin;
  - every candidate's record;
  - the clocks, with "clock-confounded" where P's SM clock is off D's by more than 2 %.
- **The pin table:** the verdict carries `pin_table`, each (site, shape, M)'s selected configuration under the key. It
  is the input to the code PR that fills `LT_PINS`, and it is written only from a reading, never from a shakedown.

## Predictions (graded in the read, never ruling)

| prediction | basis | if HELD | if MISSED |
|---|---|---|---|
| **router: PASS** | the served path switches to a GEMM family with an 8 x 1 grid on a 128-row output at M >= 2 | the router's 2-4 row cost is a kernel choice | the default is less wasteful on sm_120 than assumed |
| **o_proj: PASS** | the same switch at an 8 x 16 grid; split-K configurations on the list fill the card | K1 ships o_proj | the GEMM family's cost here is not the grid |
| the self-pair within 0.005 of 1 | P222's | | a noisier card |

## Budget and the fit

- **One RTX 5090 on Vast**, rented directly.
  - A host with a recorded power cap of at least 575 W, read again on the box (rc 17 below it).
  - The pin and the expected and worst cost go on the bus before the create.
  - rent-watch is armed at the create (`--until`), and teardown is verified.
- **Worst case:** at most $1.05/h, **guard 1.0 h**, and a 10 GB download allowance at <= $0.04/GB:
  1.0 x $1.05 + 10 x $0.04 = **$1.45**.
  - The 10 GB is the download (the e4b install and its wheels), not the disk. The create asks for a 40 GB disk, and the
    runner refuses under 20 GB free (rc 13). Amendment 2.
- **The fit, with at least 25 % headroom.** Arithmetic, every term conservative:
  - **The selection:** 6 cells x up to 64 candidates (the heuristic request's bound; review, Amendment 1), each three
    graph calls of up to 384 ctypes launches (about 0.3 s), plus 40 replays of up to 4 ms (0.16 s). About 177 s.
  - **The reading:** 6 cells x 60 reps x 80 replays of up to 4 ms = 115 s, plus the captures.
  - **Everything else:** the copy rate, the stores, the engagement.
  - **Total:** about 480 s; x2 rounded up: `NEED_BENCH` = **1200 s**. `INSTALL_TO_GATE_S` = 600 s, `RESERVE` = 600 s.
  - **The runner's check:** `FIT` = 600 + 1200 + 600 = 2400 s, and `FIT_NEED` = 2400 x 1.25 = 3000 s. Short of that is
    rc 41.
  - **The static fit:** (boot 300 s + 2400 s) x 1.25 = 3375 s <= 3600 s (1.0 h).
- **The premise, on the 5090 itself:** exactly **37 passed**, else rc 25. That is K1's GPU suite against the loaded
  cuBLASLt (`tests/test_dense_lt_pin_gpu.py`, 10) and its CPU suite (`tests/test_dense_lt_pin.py`, 27), staged copies.
  An RTX A4000 run of the same suite (sm_86) was a kit check, not the premise.
- **Run ids:** `p228-5090-1` to `-3`, and `p228-shake-1` to `-3` for the shakedown.
  - The shakedown is explore tier, may run this PR's head, and passes `--shakedown --reps 4 --copies-o 8 --copies-r 32`.
  - Its record becomes the reducer's real fixture by an amendment before the reading.
  - Two consecutive runs with no site read stop the lane. rc 17 and rc 41 are not strikes, and a shakedown never is.

## Sequencing

1. This registration merges after one review and the maintainer's pass, and after e4b#2016 and e4b#2019 (its code)
   and the 0.57.0 tag.
2. The shakedown may run first.
3. The reading runs at the merge commit.
4. On a site's PASS, a code PR fills `LT_PINS` from the verdict's `pin_table`, with no speed claim.
5. A served read with B3's quality bar, read per bucket and with the router as its own arm, comes before any default.

## What this lane cannot say

- **No step time:** two dense projections alone.
- **Scope:** one RTX 5090 at 575 W, one cuBLASLt version (the key), random bf16 weights and inputs.
- **No quality:** P differs from D within the tolerance, and the tokens are a served read's to judge.

## Amendment 1 (2026-10-23, before any reading): the shakedown and the real fixture

- **`p228-shake-1`:** an RTX A4000 (sm_86), explore tier, about $0.01, teardown verified. It ran this kit at the
  registration's head end to end:
  - the tripwire, including `LT_PINS == {}` in the commit and `router_linear` as the epilogue's call;
  - the premise: **37 passed** on the card;
  - both blocks, with every candidate on the card's own list checked, timed and selected by the rule;
  - the reducer: `o_proj=SHAKEDOWN router=SHAKEDOWN`.
  - No times are quoted, and the sm_86 selections seed nothing for the 5090.
- **The fit (review 1):** the selection's bound is the heuristic request's 64 candidates per cell, not 8. `NEED_BENCH`
  moves 900 -> 1200 s (above). The guard stays 1.0 h and the worst case $1.45.
- **The real fixture:** its bench record is pinned at `bench/p228/fixtures/p228-shake-1/bench_p228.json` (sha256
  `06adab43...`; no host identifiers).
  - `tests/test_p228.py` reduces it to SHAKEDOWN with every structural check passed.
  - Five departures from it each VOID `o_proj` alone.
  - A mutant without the "block B timed the pin block A's rule names" check accepts one of them.

## Amendment 2 (2026-10-23, before any reading): wording only

- **The budget's 10 GB term** is the download allowance, not the disk. The create asks for a 40 GB disk, and the
  runner refuses under 20 GB free (rc 13). The worst case ($1.45), the guard and every constant are unchanged. P229's
  review found the wording; this lane carries the same correction.
