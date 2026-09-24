# P66 reading receipts — `p66-5090-2` (RTX 5090, 2026-09-24)

The box's own outputs from the registered reading, copied out of the fetched run directory unchanged, except that
the arm rows and the full read are gzipped and the logs carry a `.txt` suffix (`*.log` is gitignored).

| path | what |
|---|---|
| `RESULTS-p66-generated.md` | the reducer's read, written on the box (unedited) |
| `p66_read.json.gz` | the reducer's full JSON (every window, gate and verdict) |
| `L/rows_<arm>.json.gz` | one census file per Level L arm: per window, the eager API counts by phase, device rows, sync-debug sites, graph nodes and check, timing, routing, traffic check |
| `L/table_<arm>_<window>_<mode>.txt` | the profiler key-averages tables `bench/hybrid-g9/f1/step_budget.py` parses (gate G5) |
| `L/graph_*.dot.head.txt` | the head of each captured token's `debug_dump()` (the dot reader of `graph_nodes`) |
| `L/box.json` | host record and the box's unit host costs (`probes`), and the transfer constants the reducer used (`costs`) |
| `L/capture_probe_{hyb,mnvme}.json` | the first synchronizing op on the two paths that refuse capture |
| `calib.json` | grouped-nf4-gemm `bench/calibrate.py`'s blob (`b_link` = 40 back-to-back 64 MB pinned copies, `b_vram` = device triad) |
| `M/` | Level M: step_decomp's per-arm JSON, kernels tables and sync-attr counts |
| `logs/`, `outer.log.txt`, `summary.txt`, `versions.txt`, `forensics.txt` | the runner's own record |
| `qwen3.bake.json`, `staged.sha256` | the NF4 arena bake record; the staged-file pin the box verified (`sha256sum -c`) |
| `teardown-proof.json` | the launcher's destroy record |

Kept in `adertha-receipts` (`receipts/experts4bit-qlora/2026-09-24/p66-5090-2/`, commit `f9903cc`), not here:
`receipt.json`, the guard log, the run nonce and instance id, and the two arena index/manifest files (bake artifacts).

## Reproduce the read

```sh
mkdir -p /tmp/p66 && cp -r bench/p66/receipts/L bench/p66/receipts/M /tmp/p66/ && gunzip /tmp/p66/L/rows_*.json.gz
python bench/p66/p66_reduce.py /tmp/p66/L --level-m /tmp/p66/M --step-budget bench/hybrid-g9/f1/step_budget.py \
    --md /tmp/p66/RESULTS-p66-generated.md --json /tmp/p66/p66_read.json
```

`p66_reduce.py` reads unzipped `rows_*.json` only. Run over these files with Python 3.14 on a Mac, the generated
markdown matches the box's byte for byte except one float's last digit in the P3 line (`added_kernels_ms_per_token`
…`812` against …`816`, a summation-order difference); the JSON differs in the last digit of 90 float sums.
