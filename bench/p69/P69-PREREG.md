# P69 — does grouped-nf4-gemm's new single-copy link probe read the gen-4 gather rate lane P66 implied? (for grouped-nf4-gemm#400)

Registered before any rental. Record: grouped-nf4-gemm#400 (the follow-up P66's decision rule filed); the e4b side of
the lane is this directory. Consumer of the answer: `kernel/cold_deadline.py`'s `link_eff` (grouped-nf4-gemm#402).

## Question

Lane P66 (`bench/p66/RESULTS-p66.md`, claim `e4b.serve.p66.qwen3.pipelined-gather-over-cold-deadline.5090.2026-09-24`)
found the pipelined engine's UVA gather moving cold rows at the box's **single-copy** H2D rate (its census probe: one
synchronized 64 MB pinned copy, 14.72 GB/s; the gather's implied rate 14.44) while `cold_deadline` was given the
calibration's **back-to-back** rate (40 copies in flight, 23.07 GB/s), so the bytes-over-link term under-predicted the
transfer 1.57–2.02× on a PCIe gen 4 × 16 RTX 5090. On the gen 3 × 8 A2000 the two rates agree and the model read
0.97–1.02×. grouped-nf4-gemm#402 moves that single-copy probe into `bench/calibrate.py` (schema
`gnf4-hybrid-calib/2`) and makes `Costs.from_blob` carry the ratio as `link_eff`.

One question: **run by the calibration script itself, on a gen 4 × 16 RTX 5090 of the class the consumer serves, is the
ratio what P66's probe read?** That turns P66's one-host census number into the kernel package's own calibration
receipt, in the schema `from_blob` reads.

## Instrument

`bench/p69/p69_drive.sh`, the launcher's `--command` (controller side, like `bench/p56/p56_prove.sh`): it refuses
without the launcher's `E4B_RENT_*` environment (exit 78), reads the box's card and PCIe state with `nvidia-smi` and
**refuses any card that is not an RTX 5090 (exit 15)**, fetches `bench/calibrate.py` from grouped-nf4-gemm at the
pinned commit `GNF4_SHA` (the #402 merge commit, set in the manifest; the fetched file's sha256 is recorded), stages
it on the box, runs it **twice back to back** with `--skip-cpu` (GPU benches only; the CPU microbench and the NVMe
probe are not this question), and fetches both blobs, the script's stdout, `nvidia-smi` and the torch version into
`$E4B_RENT_RUN_DIR/p69/`. Nothing is installed: the image ships torch 2.8.0+cu128, and `calibrate.py` needs nothing
else for the GPU benches. No e4b or grouped-nf4-gemm package is installed or imported on the box.

Read: `run1/calib.json` and `run2/calib.json` → `b_link.h2d_64mb` (back-to-back), `b_link.h2d_64mb_single` (single),
their ratio capped at 1 (exactly what `Costs.from_blob` computes), the two runs' spread, and `nvidia-smi`'s PCIe
generation and width. The read is a table of those numbers; no reducer beyond arithmetic.

## Predictions (written before the box is rented)

Quantities are the calibration script's own fields, on one RTX 5090 (gen 4 × 16) in the launcher's verified/secure
class, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.

- **P1 — `link_eff` = `h2d_64mb_single` / `h2d_64mb` ∈ [0.55, 0.75]** on both runs. Basis: P66's census probe on one
  such host read 14.72 / 23.07 = 0.638, and the probe moved into `calibrate.py` is the same measurement (12 copies,
  synchronized on each side, median of the last 10).
- **P2 — `h2d_64mb` (back-to-back) ∈ [20, 30] GB/s.** Basis: 23.07 (P66's box), 28.36 and 28.47 in the two
  repositories' test constants from earlier 5090 calibrations.
- **P3 — `h2d_64mb_single` ∈ [12, 18] GB/s.** Basis: P66's 14.72, and P1 × P2.
- **P4 — the two runs agree within 10 % on each of the three numbers** (the probe's own repeatability on one box).

Disclosed, not a prediction: on the NAS A2000 (gen 3 × 8) the same script at the #402 head read single 5.67 against
back-to-back 5.46 GB/s (`link_eff` capped at 1.0; grouped-nf4-gemm `bench/cold-engine/calib-a2000-400/`). That is the
gen-3 control and the schema-/2 round-trip, run before this registration; it is why the prediction is about gen 4.

## Decision rule

- Whatever P1 reads, both blobs are committed to grouped-nf4-gemm as its 5090-class calibration receipt
  (`bench/cold-engine/calib-5090-p69/`, kernel first) and one claim registers the measured factor with the three
  probe values and the PCIe state: `gnf4.calib.link-efficiency.5090.<date>`. `cold_deadline` reads the blob; nothing in
  the model is tuned to the outcome.
- **P1 holds:** `link_eff` is a measured constant of the class the consumer serves, and the RFC comparison P66's rule
  deferred may be made with it (a separate lane).
- **P1 refuted:** recorded as such; the difference between this box and P66's is stated as a difference between
  boxes, not explained. A second box would be the next draw, under its own amendment.
- **P4 refuted** (the probe is not repeatable within 10 % on one box): the claim carries both runs and no single
  factor; `from_blob` gets a follow-up to take the median of several runs.
- Nothing here changes a default in either package.

## Cost, guard, receipts

One RTX 5090 (Vast, verified/secure), **≤ $0.75/h, guard 0.25 h → approval line $0.1875**, expected ~4 min of
runtime. The guard is under one hour, so the compute rule needs no proving rental. Receipts:
`receipts/experts4bit-qlora/<date>/p69-5090-<n>/` (the launcher's `receipt.json`, `teardown-proof.json`, and
`p69/`). Refusals: 15 (card class) and the launcher's own; a refused box is redrawn under a new run id, at most
three attempts (≤ $0.5625 together).

**Exit codes** (`p69_drive.sh`): 78 launcher environment missing; 15 the card is not an RTX 5090; 20 the pinned
`calibrate.py` could not be fetched or its sha256 was not recorded; 21 ssh to the box failed; 22 the run or the
fetch-back failed; 0 both runs completed and both blobs are in the run dir.

Amendments, dated, go below this line before any data is read.
