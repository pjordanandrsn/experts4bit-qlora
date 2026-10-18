# Q235 Amendment 7: the PRO 6000 box's C, as read

This records the read of `PREDICTION-C.md` (#1680), which is unchanged. The box is `q235-pro6000-3` (Shape B, RTX PRO 6000,
PCIe gen 4 x16). Receipts: `q235-pro6000-3` `box/box.json` (`8b19c4c2`), `hostrate.json` (`c82685f9`) and `instance.json`
(`295f1dc7`).

## The read

Means over the 8 sampled layers × 8 calls, without layer 0's resident forward calls:

| | per micro-batch per layer |
|---|---:|
| staging, forward | 51.40 ms |
| staging, recompute | 50.81 ms |
| experts compute (forward 4.76 + recompute 4.62 + backward 4.68) | 14.07 ms |
| **C, measured** | **7.26** |

**The registered rule, as registered.** `PREDICTION-C.md` refutes the prediction if C, rescaled to 21.76 GB/s by the box's
own measured H2D, falls outside 5.0 to 11.0. The box's `hostrate.json` reads 20.39 GB/s, so the rescaled C is
7.26 × 20.39 / 21.76 = 6.80: inside the interval. **The prediction is not refuted.**

**C = 7.26, measured, is the read of record.** It also lies inside 5.0 to 11.0, below the 8.25 point, in the direction the
prediction's caveat named. C is reporting-only.

## The H2D rescale is retired (a deviation from the registration)

The prediction rescaled C by the box's own `hostrate.json`: C ≈ 8.25 × 21.76 / H2D. On this host that probe under-measures
the link, so the rescale divides by the wrong rate.
- **Staging rate.** It ran at 26.6 GB/s (25.8–27.9 per layer); the probe read 20.39.
- **The bytes are exact.** 1,016,497,963,008 B / 748 copies = 1,358,954,496 B per call, 128 experts × 10,616,832 B.
- **The timer is not short.** Independently of the timer, the timed pass moved 1,016.5 GB in 44.1 s, so H2D ran at no
  less than 23.0 GB/s. With the experts compute taken out, it ran at no less than 26.2 GB/s.
- **The provider's listing agrees.** It gave pcie_bw 26.6 GB/s for the machine at launch (`instance.json`).
- **The probe's fault.** It copies one 1 GiB pinned buffer. Its D2H (28.29 GB/s) is unaffected while its H2D is low,
  which points at that one source buffer's placement. Which mechanism (NUMA or IOMMU) is not in the receipts.

At the prediction's own basis, a staging call costs 1,358,954,496 B / 21.76 GB/s = 62.45 ms. C at that basis is 8.88
against the predicted 8.25: the experts compute came in 7 % below the prediction's 15.13 ms.

**The staging rate of record on this machine is 26.6 GB/s.** The host-rate probe is a lower bound on this host until its
next version records several buffers and their placement.
