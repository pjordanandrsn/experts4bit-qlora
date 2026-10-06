# SC1g box J: e4b's MXFP4 cost on gpt-oss-20b chat text is the weights, not the route; it does not replicate across windows (lane SC1g of #846; 2026-10-05)

Pre-registration: [`../../sc2/SC1g-PREREG.md`](../../sc2/SC1g-PREREG.md). Box J is set by amendment A2 and re-run under A3;
the reads are its "A3 read" section. These are e4b-only diagnostic runs: no comparator was installed, so G1–G5 read
UNREAD by design.

| run (receipt) | e4b | host | outcome | $ |
|---|---|---|---|---|
| `sc1g-diag-1` (adertha-receipts `a6a16350`) | `3e133cf7` | machine 145701 | OK. A2's arms; 15 of 20 ran (the deadline dropped j16–j20) | 0.709 |
| `sc1g-diag-2` (`28cd4d15`) | `a7891300` | machine 145701 | OK. A3's arms; 10 of 19 ran (the deadline dropped k11–k19); attention check INERT | 0.726 |

**What is committed.**
- Per run: `summary.txt`, `versions.txt`, `box.json`, `forensics.txt`, `kernel_classes.json`, `quiesce_arms.json`,
  `logs/`, and `sc1g/`.
- `sc1g/` holds every arm's receipt, the pinned windows (`k8_window_*.json`), the route records (`routes/`), the kernel
  and attention checks, and the reducer's verdict.
- Trimmed: `sc1g-diag-1`'s captured decode activations (`capture_conv1.pt`, 3.7 MB). The kernel check's receipts from
  them are kept (`gemv_*_5090.json`).

**Reproduce** the verdict lines (G1–G6, J1–J6 for diag-1, K1–K5 and the across-window reads for diag-2, the attention
check) from this directory with the reducer at main:

```
python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g
python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-1/sc1g
```

The $0 A2000 correctness checks the reads also rest on are in [`../../sc2/sc1g-a2000/`](../../sc2/sc1g-a2000/): the GEMV
kernel, the fp8 KV pack, the NF4 M-tile at large M, and the served loop's KV write paths.

## Box R (amendment A4): the bf16-dequant reference, `R_NOT_OK`

**Box:** one H100 NVL at a declared $3.50/h, maintainer-approved 2026-10-05 per tc1c's precedent, guard 1.0 h.

**Launches.** R launched from `8a1513a4` (#1198), then from `9dc7b59a` (#1217, which made the egress probe log why it
refused). The registered rule is identical at both: `test_box_r_rule_is_the_registered_one` digest `ee122b74…`.

| attempt | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-r-1` | none written | REFUSED[97]: the CTO daily ceiling ($50, the manifests' base; the $100 figure is the training campaign's) | 0 |
| `sc1g-r-2` | `f6ec2329` | REFUSED: the store HEAD was ahead of its upstream (the mini's https push failed) | 0 |
| `sc1g-r-3` | `f483ad41` | REFUSED: the same window, ~30 s | 0 |
| `sc1g-r-4` | `6859a12d` | REFUSED: the manifest carried RTX 5090 exclusion receipts (an H100 NVL launch accepts only same-class ones) | 0 |
| `sc1g-r-5` | `1dac8166` | REFUSED: no Vast H100 NVL offer with ≥ 320 GB disk and ≥ 98 GB RAM | 0 |
| `sc1g-r-6` | `ae253242` | HARNESS_ERROR: RunPod; the curl egress probe read 0.0 MB/s with its stderr discarded (fixed by #1217) | 0.1348 |
| `sc1g-r-7` | none written | REFUSED[97]: a live peer run's nested `receipt.json` failed the store reconciler | 0 |
| `sc1g-r-8` | `2c49f7d5` | **ran to its verdict: `R_NOT_OK`** (rc 31; the launcher records HARNESS_ERROR); RunPod, from `9dc7b59a` | 0.5556 |

Box R total **$0.6904**; the lane is at **$4.482**.

**Reproduce R's verdict** (no GPU) from the committed receipt:

```
python bench/sc2/sc1g_ref.py --reverdict bench/h2h-2026-10-02/sc1g/receipts/sc1g-r-8/ref --k0 bench/h2h-2026-10-02/sc1g/receipts/sc1g-r-8/k0.json
```

It exits 0 only if the verdict and every check equal R's own `r_verdict.json`, and every `ref_<src>.npz` hashes to the sha R
recorded. The artifacts are kept as receipts only: R is not OK, so they grade nothing.
