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
