# P228 results: K1's pinned cuBLASLt GEMM for o_proj and the router at 2 to 8 rows (o_proj=PASS router=PASS)

**Registration:** [PREREG-p228.md](PREREG-p228.md) (e4b#2021, with Amendments 1-7), read once, as registered.
**Reading:** `p228-5090-r1`, one RTX 5090 at a 575 W power limit, e4b `32fda64d` (the Amendment 7 merge), cuBLASLt
120804, $0.04. Records: [receipts/p228-5090-r1/](receipts/p228-5090-r1/README.md).

## The question

For each of Qwen3-30B-A3B's two dense sites, `o_proj` (N, K = 2048, 4096; 48 copies) and the router gate (128, 2048;
384 copies), does the pinned cuBLASLt call that block A's rule selects, run through e4b's installed
`E4B_DENSE_LT_PIN` path (arm P), take at most **0.85** of the served default's time (arm D: `F.linear` at the same M)
at **both** M = 2 and M = 4? M = 8 is selected and recorded but never rules.

## The answer

| site | M | P/D median | P/D upper bound | clock-adjusted | rules |
|---|---|---|---|---|---|
| `o_proj` | 2 | 0.5978 | **0.5981** | **0.5981** | PASS |
| `o_proj` | 4 | 0.6008 | **0.6009** | **0.6009** | PASS |
| `o_proj` | 8 | 0.6023 | 0.6025 | 0.6025 | reported, never rules |
| router | 2 | 0.2831 | **0.2832** | **0.2832** | PASS |
| router | 4 | 0.2842 | **0.2843** | **0.2843** | PASS |
| router | 8 | 0.8838 | 0.8841 | 0.8841 | reported, never rules |

- **`o_proj`: PASS.** The pinned call runs in a median 0.60 of the default's time at 2 and 4 rows (upper bounds
  0.5981 and 0.6009 against 0.85).
- **router: PASS.** The pinned call runs in a median 0.28 of the default's time at 2 and 4 rows (upper bounds 0.2832
  and 0.2843 against 0.85).
- **The router at 8 rows reads 0.884 (upper bound 0.8841), above the 0.85 bar.** M = 8 never rules (the registration's
  rule), so this does not touch the verdict, and it is stated here plainly. The pin did not slow down at 8 rows; the
  default sped up (below). Any router claim stays at M <= 4.
- **The instrument held:**
  - The selection named one pin per (site, M), and block B timed exactly that pin.
  - The engagement was as registered: P counted `copies` pinned calls per site and M in e4b's `SEEN` (48 for
    `o_proj`, 384 for the router), and D and T1 counted none.
  - P matched D within the 1.5e-2 tolerance in every rep (largest relative difference 0.0078), so no site is
    TOLERANCE_FAIL.
  - D2 matched D bit for bit in every rep, and every self-pair D/D2 median sits within 0.9994-1.0000.
  - The enforced power limit was 575 W in every clock sample.

## The pin table

Block A selected the same configuration in all six cells: **algo 21, tile 5, split_k 16, reduction scheme 4 (output
type), 20 stages**, with cluster shape, CTA swizzling, custom option and inner shape all 0. The verdict's `pin_table`
holds it per (site, shape, M) under the key cuBLASLt 120804, cc 12.0, 170 SMs. It is the input to the code PR that fills
`LT_PINS`.

| site | M | candidates | chosen (index) | runner-up / chosen time | workspace |
|---|---|---|---|---|---|
| `o_proj` | 2 | 6 | split_k 16 (1) | 1.086 (split_k 9) | 128 KiB |
| `o_proj` | 4 | 6 | split_k 16 (1) | 1.035 (split_k 9) | 256 KiB |
| `o_proj` | 8 | 6 | split_k 16 (1) | 1.029 (split_k 9) | 512 KiB |
| router | 2 | 6 | split_k 16 (2) | 1.151 (split_k 9) | 8 KiB |
| router | 4 | 6 | split_k 16 (2) | 1.151 (split_k 9) | 16 KiB |
| router | 8 | 6 | split_k 16 (1) | 1.130 (split_k 9) | 32 KiB |

- Every candidate on every list passed the check, ran bitwise equal twice, and matched `F.linear` within the
  tolerance.
- No list held a scheme-2 (fp32 partials) candidate, so the 1.02 preference never applied. Each pin is the fastest
  eligible candidate.
- The lists held 6 configurations per cell, not the request's 64. The selection took 10.5 s, well inside the fit.

## Predictions

| prediction | read | graded |
|---|---|---|
| router: PASS | PASS (upper bounds 0.2832, 0.2843) | **HELD** |
| `o_proj`: PASS | PASS (upper bounds 0.5981, 0.6009) | **HELD** |
| the self-pair within 0.005 of 1 | medians 0.9998, 1.0000, 1.0000 (`o_proj`); 1.0000, 1.0000, 0.9994 (router) | **HELD** at every cell, largest departure 0.0006 |

## Reported, never ruling

| site | M | D, D2, P, T1 (µs a call) | D/T1 | P/T1 | P / byte floor |
|---|---|---|---|---|---|
| `o_proj` | 2 | 22.56, 22.57, 13.48, 12.46 | 1.81 | 1.08 | 1.22 |
| `o_proj` | 4 | 22.66, 22.66, 13.61, 12.89 | 1.76 | 1.06 | 1.23 |
| `o_proj` | 8 | 22.74, 22.74, 13.70, 12.54 | 1.81 | 1.09 | 1.24 |
| router | 2 | 12.75, 12.75, 3.61, 2.35 | 5.43 | 1.54 | 10.5 |
| router | 4 | 12.76, 12.76, 3.63, 2.35 | 5.44 | 1.54 | 10.5 |
| router | 8 | 4.20, 4.20, 3.71, 2.35 | 1.78 | 1.58 | 10.8 |

- **Why the router reads 0.88 at 8 rows: the default changes kernel there, not the pin.** P is flat across M (3.61,
  3.63, 3.71 µs), while D drops from 12.75 µs at 2 and 4 rows to 4.20 µs at 8. Block A's record shows the reason.
  - At M = 8 the heuristic list's first entry is algo 21, tile 5, **split_k 9**, and it is bitwise identical to
    `F.linear` (relative difference 0) at 4.195 µs a call, D's own 4.197.
  - At M = 2 and 4, the entries identical to `F.linear` are all split_k 1 configurations, the fastest at 12.0 µs.
    At M = 8 every other entry, split_k 1 included, differs from `F.linear`.
  - So cuBLAS's own choice for the 128-wide router goes split-K at 8 rows, and the pin keeps only a 0.88 edge.
  - `o_proj`'s default stays split_k 1 at all three M (the entries identical to `F.linear` are all split_k 1, the
    fastest at 22.2-22.7 µs; D 22.6-22.7), which is why it reads about 0.60 throughout.
- **The pinned call is close to the one-row path.** Against T1 (`F.linear` at one row), P costs 1.06-1.09x for
  `o_proj` and 1.54-1.58x for the router. The default costs 1.76-1.81x for `o_proj`, and 5.43-5.44x for the router at
  2 and 4 rows.
- **The byte floors:** a site's bf16 weight bytes over the copy rate measured on the box (1,519 GB/s). That is
  11.04 µs for `o_proj`'s 16.8 MB and 0.35 µs for the router's 0.5 MB. `o_proj`'s pinned call runs at 1.22-1.24x its
  floor. The router call is far from its floor at every arm, including the one-row call (6.8x), so at this size it is
  bound by launch and reduction latency, not by bytes.
- **The timer:** each timed replay reads on a 2.048 µs grid. Many reps tie exactly, which is why the SEs are small.
- **The clocks:** SM clock medians 2,932-2,947 MHz, the same for every arm within a cell, so every clock-adjusted
  bound equals its raw one and no cell is clock-confounded. Power medians 122-296 W under the 575 W limit enforced in
  every sample.
- **The boxes before this one:** `p228-5090-1` did not start (a pre-flight teardown, $0.002). `p228-5090-2` and `-3`
  refused at the box's power floor (rc 17: 525 W and 500 W; $0 and $0.01). None of them started, so none counts as a
  strike (Amendments 5 and 6). `p228-5090-r1` is the first and only started box. The lane cost $0.052 in all.

## What follows (as registered)

- **The pin table fills `LT_PINS`.** A code PR fills K1's table from the verdict's `pin_table` for both sites, with no
  speed claim. The knob stays opt-in.
- **A served read comes before any default.** It uses B3's quality bar, reads per bucket, and keeps the router as its
  own arm. Given the 8-row reading above, that read reports the router at bucket 8 on its own.
- **No default changes here.** `E4B_DENSE_LT_PIN` stays opt-in.

## What this does not say

- Nothing about step time or served speed: these are two dense projections alone, in a captured microbench.
- Nothing about quality: P differs from D within the tolerance (split-K reorders the sum), and the tokens are a served
  read's to judge.
- Nothing about another cuBLASLt version, card, SM count or power limit: the pins are keyed to cuBLASLt 120804 on
  cc 12.0 with 170 SMs, and K1 refuses any other key.
- Nothing about the router at more than 4 rows as a saving worth a claim: at 8 rows the default's own split-K choice
  closes most of the gap.
- The weights and inputs are random bf16 at Qwen3-30B-A3B's shapes.
