## Generated read (p66_reduce.py, unedited)

Box: NVIDIA RTX A2000 12GB, 12282 MiB, 575.64.05, 1, 3, 8, 16, 70.00 W, 2100 MHz (26 SMs) on Intel(R) Xeon(R) W-1250 CPU @ 3.30GHz; torch 2.8.0+cu128, triton 3.4.0. Host unit costs: {"aten_launch_host_us": 8.88226618990302, "triton_launch_host_us": 24.799767369404435, "sync_idle_us": 7.278234465047717, "item_roundtrip_us": 8.342747925780714, "h2d_pinned_64mb_gbs": 6.047941231951027, "d2d_copy_rw_gbs": 252.48314567389514}. Transfer constants: {"source": "calib:calib.json", "b_link_gbs": 6.11, "b_vram_gbs": 257.5, "b_dram_gbs": 18.01}.

### olmoe-1b-7b (reference: `ref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb-0.25 | uniform | 0.734 | eager | 1183.2 | 317.5 | 159.0 | 1500.8 | 77.591 | +1055.2 | +159.0 | 284.886 |
| hyb-0.25 | uniform | 0.734 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 767.5 | 17.938 | +528.0 | +64.0 | 25.924 |
| hyb-0.25 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1519.5 | 51.547 | +1072.0 | +160.0 | 161.779 |
| hyb-0.25 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.5 | 96.623 | +672.0 | +128.0 | 249.644 |
| hyb-0.25 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | uniform | 0.465 | eager | 1199.5 | 320.0 | 160.0 | 1519.5 | 55.923 | +1071.5 | +160.0 | 182.582 |
| hyb-0.50 | uniform | 0.465 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 768.0 | 16.693 | +528.0 | +64.0 | 25.929 |
| hyb-0.50 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1520.0 | 51.301 | +1072.0 | +160.0 | 142.755 |
| hyb-0.50 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.8 | 92.463 | +672.0 | +128.0 | 245.597 |
| hyb-0.50 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| pipe-0.00 | uniform | 1.000 | eager | 240.0 | 32.0 | 0.0 | 272.0 | 87.046 | +112.0 | +0.0 | 85.959 |
| pipe-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 274.0 | 85.622 | +0.0 | +0.0 | 86.094 |
| pipe-0.25 | uniform | 0.734 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 65.377 | +160.0 | +0.0 | 66.028 |
| pipe-0.25 | uniform | 0.734 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 67.243 | +0.0 | +0.0 | 70.266 |
| pipe-0.25 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.087 | +160.0 | +0.0 | 11.822 |
| pipe-0.25 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.596 | +0.0 | +0.0 | 10.745 |
| pipe-0.25 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.076 | +160.0 | +0.0 | 47.462 |
| pipe-0.25 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 46.802 | +0.0 | +0.0 | 46.408 |
| pipe-0.25 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 85.086 | +160.0 | +0.0 | 86.847 |
| pipe-0.25 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 85.818 | +0.0 | +0.0 | 86.360 |
| pipe-0.50 | uniform | 0.465 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 49.387 | +160.0 | +0.0 | 51.363 |
| pipe-0.50 | uniform | 0.465 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 47.829 | +0.0 | +0.0 | 51.396 |
| pipe-0.50 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.323 | +160.0 | +0.0 | 12.882 |
| pipe-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.690 | +0.0 | +0.0 | 10.671 |
| pipe-0.50 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.427 | +160.0 | +0.0 | 47.460 |
| pipe-0.50 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 46.148 | +0.0 | +0.0 | 46.022 |
| pipe-0.50 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 84.483 | +160.0 | +0.0 | 85.392 |
| pipe-0.50 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 82.748 | +0.0 | +0.0 | 83.531 |
| pipe-0.83 | uniform | 0.172 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 23.622 | +160.0 | +0.0 | 22.483 |
| pipe-0.83 | uniform | 0.172 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 24.864 | +0.0 | +0.0 | 21.385 |
| pipe-0.83 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 10.507 | +160.0 | +0.0 | 11.616 |
| pipe-0.83 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.557 | +0.0 | +0.0 | 10.634 |
| pipe-0.83 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 44.433 | +160.0 | +0.0 | 45.913 |
| pipe-0.83 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 45.995 | +0.0 | +0.0 | 45.002 |
| pipe-0.83 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 82.419 | +160.0 | +0.0 | 80.332 |
| pipe-0.83 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 85.952 | +0.0 | +0.0 | 82.484 |
| pipe-1.00 | uniform | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 16.871 | +160.0 | +0.0 | 24.939 |
| pipe-1.00 | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 11.095 | +0.0 | +0.0 | 11.759 |
| ref | uniform | 0.000 | eager | 128.0 | 0.0 | 0.0 | 128.0 | 16.031 | +0.0 | +0.0 | 13.169 |
| ref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 130.0 | 12.316 | +0.0 | +0.0 | 12.397 |

### gpt-oss-20b (reference: `mref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnvme-0.00 | uniform | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.126 | -8.0 | +24.0 | 116.347 |
| mnvme-0.00 | uniform | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | uniform | 0.766 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 59.948 | -8.0 | +24.0 | 90.782 |
| mnvme-0.25 | uniform | 0.766 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.119 | -8.0 | +24.0 | 13.949 |
| mnvme-0.25 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.702 | -8.0 | +24.0 | 67.685 |
| mnvme-0.25 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.469 | -8.0 | +24.0 | 118.476 |
| mnvme-0.25 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | uniform | 0.523 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 44.483 | -8.0 | +24.0 | 62.845 |
| mnvme-0.50 | uniform | 0.523 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.024 | -8.0 | +24.0 | 14.203 |
| mnvme-0.50 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.282 | -8.0 | +24.0 | 62.968 |
| mnvme-0.50 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.527 | -8.0 | +24.0 | 115.966 |
| mnvme-0.50 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-1.00 | uniform | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 12.035 | -8.0 | +24.0 | 15.723 |
| mnvme-1.00 | uniform | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mpin-0.00 | uniform | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 100.578 | +0.0 | +0.0 | 115.131 |
| mpin-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 85.137 | +0.0 | +0.0 | 77.800 |
| mpin-0.50 | uniform | 0.523 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 45.823 | +0.0 | +0.0 | 45.153 |
| mpin-0.50 | uniform | 0.523 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 37.696 | +0.0 | +0.0 | 36.861 |
| mpin-0.50 | ctrl0 | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 10.071 | +0.0 | +0.0 | 12.881 |
| mpin-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 10.037 | +0.0 | +0.0 | 10.031 |
| mpin-0.50 | ctrl2 | 0.500 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 47.707 | +0.0 | +0.0 | 48.600 |
| mpin-0.50 | ctrl2 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 42.235 | +0.0 | +0.0 | 41.965 |
| mpin-0.50 | ctrl4 | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 79.562 | +0.0 | +0.0 | 84.254 |
| mpin-0.50 | ctrl4 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 76.700 | +0.0 | +0.0 | 76.312 |
| mref | uniform | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 12.057 | +0.0 | +0.0 | 18.815 |
| mref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 11.888 | +0.0 | +0.0 | 11.953 |

### Verdicts (pre-registered)

- **P0 (gates): ALL HOLD** -- nesting 39/39, simulator 14/14, graph check 21/21, graph parity 21/21, budget coverage 39/39, record completeness 39/39 (7 dropped device record(s) in total)
- **P1: HOLDS** -- `{"arms": {"pipe-0.00/eager": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.00/graph": {"windows": 1, "fixed": true, "spreads": {"graph_nodes.work": 0.0, "graph_nodes.kernel": 0.0, "graph_nodes.memcpy": 0.0, "graph_nodes.memset": 0.0}}, "pipe-0.25/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.25/graph": {"windows": 4, "fixed": true, "spreads": {"graph_nodes.work": 0.0, "graph_nodes.kernel": 0.0, "graph_nodes.memcpy": 0.0, "graph_nodes.memset`
- **P2: HOLDS** -- `{"arms": {"pipe-0.00": {"delta_per_layer": {"launches": 7.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 7, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 5.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 5, "memcpy": 2, "sync": 0}, "holds": true, "delta_per_token": {"launches": 112.0, "async_copies": 32.0, "syncs": 0.0}}, "pipe-0.25": {"delta_per_layer": {"launches": 10.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 10, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 8.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 8, "`
- **P3: RECORDED (not the registered box)** -- `{"band_ms": [0.5, 1.5], "families": {"olmoe-1b-7b": {"graph": {"device_ms_per_token": -1.2208739999999907, "added_kernels_ms_per_token": 0.48065150000000895, "shared_kernels_ms_per_token": -1.7015254999999998, "shared_by_kernel_us": {"_gemv_nf4_grouped": -1701.5254999999997}, "submissions_per_token": 0.0, "wall_ms_per_token": -0.6378079997375607}, "eager": {"device_ms_per_token": 0.8398557499999606, "added_kernels_ms_per_token": 0.821849499999984, "shared_kernels_ms_per_token": 0.018006249999974897, "shared_by_kernel_us": {"_gemv_nf4_grouped": 18.006249999974898}, "submissions_per_token": 192.`
- **P4: HOLDS** -- `{"arms": {"mnvme-0.00": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.25": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.50": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1,`
- **P5: HOLDS** -- `{"arms": {"mpin-0.00": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}, "mpin-0.50": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}}}`
- **P6: HOLDS** -- `{"arms": {"hyb-0.25": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed", "syncs_per_layer": 10.0, "launches_per_layer": 75.0, "want_syncs": 10}, "ctrl8": {"composition": "cold_only", "syncs_per_layer": 8.0, "launches_per_layer": 50.0, "want_syncs": 8}}, "holds": true}, "hyb-0.50": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed",`
- **P7: RECORDED (not the registered box)** -- `{"band": [0.8, 1.25], "refute_above": 1.5, "windows": {"hyb-0.25/uniform": {"path": "hybrid", "measured_us": 56554.333000000275, "hot_d2d_us": 0.0, "predicted_us": 55737.178661090365, "ratio": 1.0146608486209647}, "hyb-0.25/ctrl4": {"path": "hybrid", "measured_us": 37640.44375000011, "hot_d2d_us": 0.0, "predicted_us": 37948.71738627429, "ratio": 0.9918765729777818}, "hyb-0.25/ctrl8": {"path": "hybrid", "measured_us": 78301.75899999993, "hot_d2d_us": 0.0, "predicted_us": 75897.43477254859, "ratio": 1.0316785967095816}, "hyb-0.50/uniform": {"path": "hybrid", "measured_us": 35261.49924999983, "ho`
- **P8 (informational)** -- `{"status": "INFORMATIONAL", "rows": {"olmoe-1b-7b/pipe-0.00/graph": {"cold_fraction": 1.0, "residency_ms": 73.69721680879593, "fixed_tax_ms": -0.6378079997375607, "tax_share": -0.0086544380826799}, "olmoe-1b-7b/pipe-0.00/eager": {"cold_fraction": 1.0, "residency_ms": 72.78971472987905, "fixed_tax_ms": 11.770031997002661, "tax_share": 0.16169910873646062}, "olmoe-1b-7b/pipe-0.25/graph": {"cold_fraction": 0.75, "residency_ms": 57.8690092661418, "fixed_tax_ms": -0.6378079997375607, "tax_share": -0.011021581461750229}, "olmoe-1b-7b/pipe-0.25/eager": {"cold_fraction": 0.75, "residency_ms": 52.85900051239878, "fixed_tax_ms": 11.770031997002661, "tax_share": 0.22266845537954968}, "olmoe-1b-7b/pipe-0.50/graph": {"cold_fraction": 0.5, "residency_ms": 38.99886575527489, "fixed_tax_ms": -0.6378079997375607, "tax_share": -0.016354526917267905}, "olmoe-1b-7b/pipe-0.50/eager": {"cold_fraction": 0.5, "`
- **P9 (informational)** -- `{"status": "INFORMATIONAL", "cross_level": {"M_pipe-0.00": {"level_m_per_step": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}, "M_pipe-1.00": {"level_m_per_step": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}}, "level_m": {"M_hyb-0.50": {"steps": 12, "coverage": 0.9933051479158812, "kernel_calls_per_step": 1735.0000000000002, "device_ms_per_step": 57.522466666666666, "manifest_counts": {"vram": 512, "dram": 0, "nvme": 512}, "active_steps": 12, "per_step": {"cudaLaunchKernel": 1778.6666666666667, "cudaMemcpyAsync": 128.33333333333334, "cudaStreamSynchronize": 72.33333333333333, `

**Decision rule:** P1 ∧ P2: the pipelined engine's residency cost is a FIXED per-layer count (as registered), independent of cold fraction -> register the measured claim; the cost model gets a per-step fixed term (P3's graph device time, and submissions x the host's unit cost when eager), never a per-cold-expert launch term (grouped-nf4-gemm follow-up on kernel/cold_deadline.py) | P4: the MXFP4 NVMe engine's +3 syncs/layer are structural; a pinned-staging follow-up (two of the three are pageable H2D copies) is licensed only if their host cost >= 10 % of that token | P6: the hybrid tier's cost is layer-COMPOSITION-dependent (hot-only / mixed / cold-only); cold_dest='deadline' prices only bytes, so its GPU side omits the mixed-layer dispatch term | P8: at the RFC's 17.2 % cold the fixed tax is -7% of residency's cost -- in this engine launches cannot explain an RFC-size (2-4x) gap; the gap would be link-side
