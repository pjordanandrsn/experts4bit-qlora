## Generated read (p66_reduce.py, unedited)

Box: NVIDIA RTX A2000 12GB, 12282 MiB, 575.64.05, 3, 3, 8, 16, 70.00 W, 2100 MHz (26 SMs) on Intel(R) Xeon(R) W-1250 CPU @ 3.30GHz; torch 2.8.0+cu128, triton 3.4.0. Host unit costs: {"aten_launch_host_us": 6.522157182916999, "triton_launch_host_us": 15.99130998365581, "sync_idle_us": 4.64158202521503, "item_roundtrip_us": 6.955188466235995, "h2d_pinned_64mb_gbs": 6.234227383164285, "d2d_copy_rw_gbs": 253.89740153262662}. Transfer constants: {"source": "calib:calib.json", "b_link_gbs": 6.27, "b_vram_gbs": 257.7, "b_dram_gbs": 25.5}.

### olmoe-1b-7b (reference: `ref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb-0.25 | uniform | 0.734 | eager | 1183.2 | 317.5 | 159.0 | 1500.5 | 72.352 | +1055.2 | +159.0 | 183.217 |
| hyb-0.25 | uniform | 0.734 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 767.8 | 14.057 | +528.0 | +64.0 | 27.857 |
| hyb-0.25 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1519.8 | 51.589 | +1072.0 | +160.0 | 139.241 |
| hyb-0.25 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.5 | 91.963 | +672.0 | +128.0 | 238.139 |
| hyb-0.25 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | uniform | 0.465 | eager | 1199.5 | 320.0 | 160.0 | 1519.5 | 56.011 | +1071.5 | +160.0 | 140.122 |
| hyb-0.50 | uniform | 0.465 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 768.0 | 15.889 | +528.0 | +64.0 | 27.591 |
| hyb-0.50 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1520.0 | 50.116 | +1072.0 | +160.0 | 134.918 |
| hyb-0.50 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.8 | 89.737 | +672.0 | +128.0 | 212.714 |
| hyb-0.50 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| pipe-0.00 | uniform | 1.000 | eager | 240.0 | 32.0 | 0.0 | 272.0 | 84.926 | +112.0 | +0.0 | 85.772 |
| pipe-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 274.0 | 85.012 | +0.0 | +0.0 | 85.297 |
| pipe-0.25 | uniform | 0.734 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 65.263 | +160.0 | +0.0 | 65.609 |
| pipe-0.25 | uniform | 0.734 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 65.462 | +0.0 | +0.0 | 68.630 |
| pipe-0.25 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.243 | +160.0 | +0.0 | 11.776 |
| pipe-0.25 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.733 | +0.0 | +0.0 | 10.837 |
| pipe-0.25 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.175 | +160.0 | +0.0 | 47.386 |
| pipe-0.25 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 46.746 | +0.0 | +0.0 | 46.420 |
| pipe-0.25 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 85.317 | +160.0 | +0.0 | 85.560 |
| pipe-0.25 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 86.097 | +0.0 | +0.0 | 86.040 |
| pipe-0.50 | uniform | 0.465 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 44.663 | +160.0 | +0.0 | 46.974 |
| pipe-0.50 | uniform | 0.465 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.850 | +0.0 | +0.0 | 47.719 |
| pipe-0.50 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.049 | +160.0 | +0.0 | 11.735 |
| pipe-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.723 | +0.0 | +0.0 | 10.800 |
| pipe-0.50 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.378 | +160.0 | +0.0 | 46.737 |
| pipe-0.50 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 45.510 | +0.0 | +0.0 | 45.744 |
| pipe-0.50 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 84.254 | +160.0 | +0.0 | 85.071 |
| pipe-0.50 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 82.555 | +0.0 | +0.0 | 82.779 |
| pipe-0.83 | uniform | 0.172 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 22.860 | +160.0 | +0.0 | 21.219 |
| pipe-0.83 | uniform | 0.172 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 23.154 | +0.0 | +0.0 | 21.415 |
| pipe-0.83 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 10.998 | +160.0 | +0.0 | 11.668 |
| pipe-0.83 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.492 | +0.0 | +0.0 | 10.727 |
| pipe-0.83 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 44.050 | +160.0 | +0.0 | 45.611 |
| pipe-0.83 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.442 | +0.0 | +0.0 | 45.481 |
| pipe-0.83 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 78.181 | +160.0 | +0.0 | 80.273 |
| pipe-0.83 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 81.392 | +0.0 | +0.0 | 79.363 |
| pipe-1.00 | uniform | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 16.873 | +160.0 | +0.0 | 16.515 |
| pipe-1.00 | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 11.268 | +0.0 | +0.0 | 11.306 |
| ref | uniform | 0.000 | eager | 128.0 | 0.0 | 0.0 | 128.0 | 16.045 | +0.0 | +0.0 | 13.246 |
| ref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 130.0 | 10.216 | +0.0 | +0.0 | 10.150 |

### gpt-oss-20b (reference: `mref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnvme-0.00 | uniform | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.141 | -8.0 | +24.0 | 112.587 |
| mnvme-0.00 | uniform | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | uniform | 0.766 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 59.522 | -8.0 | +24.0 | 84.612 |
| mnvme-0.25 | uniform | 0.766 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.130 | -8.0 | +24.0 | 13.114 |
| mnvme-0.25 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.097 | -8.0 | +24.0 | 64.549 |
| mnvme-0.25 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.163 | -8.0 | +24.0 | 120.709 |
| mnvme-0.25 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | uniform | 0.523 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 42.829 | -8.0 | +24.0 | 56.655 |
| mnvme-0.50 | uniform | 0.523 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.158 | -8.0 | +24.0 | 13.216 |
| mnvme-0.50 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.061 | -8.0 | +24.0 | 60.664 |
| mnvme-0.50 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 75.906 | -8.0 | +24.0 | 107.576 |
| mnvme-0.50 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-1.00 | uniform | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 12.039 | -8.0 | +24.0 | 14.874 |
| mnvme-1.00 | uniform | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mpin-0.00 | uniform | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 76.048 | +0.0 | +0.0 | 81.830 |
| mpin-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 76.123 | +0.0 | +0.0 | 76.101 |
| mpin-0.50 | uniform | 0.523 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 45.112 | +0.0 | +0.0 | 43.251 |
| mpin-0.50 | uniform | 0.523 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 37.470 | +0.0 | +0.0 | 36.581 |
| mpin-0.50 | ctrl0 | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 10.147 | +0.0 | +0.0 | 12.639 |
| mpin-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 10.084 | +0.0 | +0.0 | 10.060 |
| mpin-0.50 | ctrl2 | 0.500 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 41.268 | +0.0 | +0.0 | 45.117 |
| mpin-0.50 | ctrl2 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 41.181 | +0.0 | +0.0 | 41.133 |
| mpin-0.50 | ctrl4 | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 76.028 | +0.0 | +0.0 | 79.881 |
| mpin-0.50 | ctrl4 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 75.791 | +0.0 | +0.0 | 75.858 |
| mref | uniform | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 12.056 | +0.0 | +0.0 | 14.450 |
| mref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 10.211 | +0.0 | +0.0 | 10.245 |

### Verdicts (pre-registered)

- **P0 (gates): ALL HOLD** -- nesting 39/39, simulator 14/14, graph check 21/21, graph parity 21/21, budget coverage 39/39, record completeness 39/39 (6 dropped device record(s) in total)
- **P1: HOLDS** -- `{"arms": {"pipe-0.00/eager": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.00/graph": {"windows": 1, "fixed": true, "spreads": {"graph_nodes.work": 0.0, "graph_nodes.kernel": 0.0, "graph_nodes.memcpy": 0.0, "graph_nodes.memset": 0.0}}, "pipe-0.25/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.25/graph": {"windows": 4, "fixed": true, "spreads": {"graph_nodes.work": 0.0, "graph_nodes.kernel": 0.0, "graph_nodes.memcpy": 0.0, "graph_nodes.memset`
- **P2: HOLDS** -- `{"arms": {"pipe-0.00": {"delta_per_layer": {"launches": 7.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 7, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 5.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 5, "memcpy": 2, "sync": 0}, "holds": true, "delta_per_token": {"launches": 112.0, "async_copies": 32.0, "syncs": 0.0}}, "pipe-0.25": {"delta_per_layer": {"launches": 10.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 10, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 8.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 8, "`
- **P3: RECORDED (not the registered box)** -- `{"band_ms": [0.5, 1.5], "families": {"olmoe-1b-7b": {"graph": {"device_ms_per_token": 1.0516557500000145, "added_kernels_ms_per_token": 0.5200635000000261, "shared_kernels_ms_per_token": 0.5315922499999888, "shared_by_kernel_us": {"_gemv_nf4_grouped": 531.5922499999888}, "submissions_per_token": 0.0, "wall_ms_per_token": 1.1562019935809076}, "eager": {"device_ms_per_token": 0.8286384999999263, "added_kernels_ms_per_token": 0.8214709999999413, "shared_kernels_ms_per_token": 0.007167499999990469, "shared_by_kernel_us": {"_gemv_nf4_grouped": 7.1674999999904685}, "submissions_per_token": 192.0, "w`
- **P4: HOLDS** -- `{"arms": {"mnvme-0.00": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.25": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.50": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1,`
- **P5: HOLDS** -- `{"arms": {"mpin-0.00": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}, "mpin-0.50": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}}}`
- **P6: HOLDS** -- `{"arms": {"hyb-0.25": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed", "syncs_per_layer": 10.0, "launches_per_layer": 75.0, "want_syncs": 10}, "ctrl8": {"composition": "cold_only", "syncs_per_layer": 8.0, "launches_per_layer": 50.0, "want_syncs": 8}}, "holds": true}, "hyb-0.50": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed",`
- **P7: RECORDED (not the registered box)** -- `{"band": [0.8, 1.25], "refute_above": 1.5, "windows": {"hyb-0.25/uniform": {"path": "hybrid", "measured_us": 56197.63100000004, "hot_d2d_us": 0.0, "predicted_us": 54346.8224812428, "ratio": 1.0340555056258538}, "hyb-0.25/ctrl4": {"path": "hybrid", "measured_us": 37933.14724999974, "hot_d2d_us": 0.0, "predicted_us": 37002.09190212276, "ratio": 1.0251622354308991}, "hyb-0.25/ctrl8": {"path": "hybrid", "measured_us": 76810.23449999995, "hot_d2d_us": 0.0, "predicted_us": 74004.18380424552, "ratio": 1.0379174602232888}, "hyb-0.50/uniform": {"path": "hybrid", "measured_us": 35292.3774999999, "hot_d2`
- **P8 (informational)** -- `{"status": "INFORMATIONAL", "rows": {"olmoe-1b-7b/pipe-0.00/graph": {"cold_fraction": 1.0, "residency_ms": 75.14779380289838, "fixed_tax_ms": 1.1562019935809076, "tax_share": 0.015385707750961467}, "olmoe-1b-7b/pipe-0.00/eager": {"cold_fraction": 1.0, "residency_ms": 72.52530253026634, "fixed_tax_ms": 3.2687015482224524, "tax_share": 0.045069809213940944}, "olmoe-1b-7b/pipe-0.25/graph": {"cold_fraction": 0.75, "residency_ms": 58.48028027685359, "fixed_tax_ms": 1.1562019935809076, "tax_share": 0.01977080116762933}, "olmoe-1b-7b/pipe-0.25/eager": {"cold_fraction": 0.75, "residency_ms": 52.362251735758036, "fixed_tax_ms": 3.2687015482224524, "tax_share": 0.06242477051440982}, "olmoe-1b-7b/pipe-0.50/graph": {"cold_fraction": 0.5, "residency_ms": 37.569874548353255, "fixed_tax_ms": 1.1562019935809076, "tax_share": 0.030774709989857704}, "olmoe-1b-7b/pipe-0.50/eager": {"cold_fraction": 0.5, "r`
- **P9 (informational)** -- `{"status": "INFORMATIONAL", "cross_level": {"M_pipe-0.00": {"level_m_per_step": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}, "M_pipe-1.00": {"level_m_per_step": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}}, "level_m": {"M_hyb-0.50": {"steps": 12, "coverage": 0.9923663842383682, "kernel_calls_per_step": 1719.0000000000002, "device_ms_per_step": 61.265144583333324, "manifest_counts": {"vram": 512, "dram": 0, "nvme": 512}, "active_steps": 12, "per_step": {"cudaLaunchKernel": 1778.6666666666667, "cudaMemcpyAsync": 128.33333333333334, "cudaStreamSynchronize": 72.33333333333333, `

**Decision rule:** P1 ∧ P2: the pipelined engine's residency cost is a FIXED per-layer count (as registered), independent of cold fraction -> register the measured claim; the cost model gets a per-step fixed term (P3's graph device time, and submissions x the host's unit cost when eager), never a per-cold-expert launch term (grouped-nf4-gemm follow-up on kernel/cold_deadline.py) | P4: the MXFP4 NVMe engine's +3 syncs/layer are structural; a pinned-staging follow-up (two of the three are pageable H2D copies) is licensed only if their host cost >= 10 % of that token | P6: the hybrid tier's cost is layer-COMPOSITION-dependent (hot-only / mixed / cold-only); cold_dest='deadline' prices only bytes, so its GPU side omits the mixed-layer dispatch term | P8: at the RFC's 17.2 % cold the fixed tax is 10% of residency's cost -- partial; stated as measured, no attribution claimed for the RFC
