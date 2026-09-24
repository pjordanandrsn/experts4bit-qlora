# Results — P69: the single-copy link probe on a gen 4 × 16 RTX 5090 (2026-09-24)

Pre-registration: [`P69-PREREG.md`](P69-PREREG.md) (#741, merged before the rental). Record: grouped-nf4-gemm#400;
the receipt and the claim live in grouped-nf4-gemm (`bench/cold-engine/calib-5090-p69/`,
`gnf4.calib.link-efficiency.5090.2026-09-24`, PR grouped-nf4-gemm#403). Run `p69-5090-2` ($0.0146, 93 s, teardown
proven; one launch before it refused pre-rental on a dirty receipts worktree, $0). The e4b side of the lane is the
driver ([`p69_drive.sh`](p69_drive.sh)) and this page.

**Box.** One RTX 5090 (driver 580.95.05, 600 W; PCIe gen 4 × 16, `nvidia-smi` reading gen 1 *current* at the idle
pre-run read) on an AMD EPYC 7663 (224 CPUs, 2 NUMA nodes); image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`,
torch 2.8.0+cu128; grouped-nf4-gemm's `bench/calibrate.py` at `54ac2a44` (#402), sha256 recorded and equal to that
tree's; nothing installed.

## The read

| run | `h2d_64mb` (40 back-to-back) | `h2d_64mb_single` (one synchronized copy) | `link_eff` |
|---|---|---|---|
| 1 | 20.58 GB/s | 17.97 GB/s | **0.873** |
| 2 | 20.72 GB/s | 19.90 GB/s | **0.960** |
| spread | 0.7 % | 9.7 % | 9.1 % |

| prediction | registered | read | verdict |
|---|---|---|---|
| P1 `link_eff` | [0.55, 0.75] on both runs | 0.873, 0.960 | **REFUTED** |
| P2 back-to-back | [20, 30] GB/s | 20.58, 20.72 | HELD |
| P3 single copy | [12, 18] GB/s | 17.97, 19.90 | REFUTED (run 2) |
| P4 repeatability | the two runs within 10 % | 0.7 / 9.7 / 9.1 % | HELD |

## What the decision rule did

- **P1 refuted → recorded as such.** The prediction came from lane P66's census probe on another gen 4 × 16 RTX 5090
  (14.72 / 23.07 = 0.638, host EPYC 7C13). This host reads 0.87–0.96 with a lower back-to-back rate (20.6 against
  23.07). The difference is stated as a difference between two hosts and is not explained; the candidates the
  registration named (the link's idle state, NUMA placement of the pinned buffer, the DMA engine's behaviour) were
  not measured. A second box is a new draw under its own amendment.
- **Both blobs are committed** as grouped-nf4-gemm's 5090 calibration receipt, and one claim registers the measured
  factor with the three probe values and the PCIe state, as registered whatever P1 read.
- **What it means for `cold_deadline`:** `link_eff` is a per-host measurement, not a constant of the card class — which
  is exactly why grouped-nf4-gemm#402 reads it from each box's own blob (`Costs.from_blob`) rather than fixing it.
  On this host the link term is 14 % longer than the bytes-over-`b_link` model; on P66's host it was 57 %. Nothing in
  either model is tuned to the outcome; no default moves.

## What this read does not say

- Why the two hosts differ. Neither the idle link state, the pinned buffer's NUMA node nor the DMA engine's behaviour
  was measured on either box.
- Anything about the gather itself: this is the calibration script's copy probes, not the engine's UVA gather (P66
  measured that one, on the other host).
