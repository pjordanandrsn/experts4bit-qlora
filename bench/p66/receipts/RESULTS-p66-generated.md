## Generated read (p66_reduce.py, unedited)

Box: NVIDIA GeForce RTX 5090, 32607 MiB, 580.95.05, 1, 4, 16, 16, 600.00 W, 3135 MHz (170 SMs) on AMD EPYC 7C13 64-Core Processor; torch 2.8.0+cu128, triton 3.4.0. Host unit costs: {"aten_launch_host_us": 6.422012438997626, "triton_launch_host_us": 13.171569583937526, "sync_idle_us": 4.573493497446179, "item_roundtrip_us": 8.896808489225805, "h2d_pinned_64mb_gbs": 14.716328179649139, "d2d_copy_rw_gbs": 1505.8636659541642}. Transfer constants: {"source": "calib:calib.json", "b_link_gbs": 23.07, "b_vram_gbs": 1568.3, "b_dram_gbs": 87.14}.

### qwen3-30b-a3b (reference: `ref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb-0.25 | uniform | 0.749 | eager | 3465.9 | 935.6 | 470.2 | 4399.4 | 194.454 | +3081.9 | +470.2 | 1550.030 |
| hyb-0.25 | uniform | 0.749 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl0 | 0.000 | eager | 1968.0 | 336.0 | 192.0 | 2301.1 | 5.784 | +1584.0 | +192.0 | 46.800 |
| hyb-0.25 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl4 | 0.500 | eager | 3600.0 | 960.0 | 480.0 | 4560.0 | 206.726 | +3216.0 | +480.0 | 1088.953 |
| hyb-0.25 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl8 | 1.000 | eager | 2400.0 | 720.0 | 384.0 | 3116.4 | 414.169 | +2016.0 | +384.0 | 1536.860 |
| hyb-0.25 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | uniform | 0.494 | eager | 3590.6 | 957.8 | 479.0 | 4548.4 | 185.022 | +3206.6 | +479.0 | 1100.418 |
| hyb-0.50 | uniform | 0.494 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl0 | 0.000 | eager | 1968.0 | 336.0 | 192.0 | 2303.5 | 5.792 | +1584.0 | +192.0 | 46.896 |
| hyb-0.50 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl4 | 0.500 | eager | 3600.0 | 960.0 | 480.0 | 4560.0 | 378.212 | +3216.0 | +480.0 | 1185.785 |
| hyb-0.50 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl8 | 1.000 | eager | 2400.0 | 720.0 | 384.0 | 3118.5 | 398.296 | +2016.0 | +384.0 | 1348.139 |
| hyb-0.50 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| pipe-0.00 | uniform | 1.000 | eager | 720.0 | 96.0 | 0.0 | 815.0 | 73.838 | +336.0 | +0.0 | 61.711 |
| pipe-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 817.0 | 73.527 | +0.0 | +0.0 | 79.292 |
| pipe-0.25 | uniform | 0.749 | eager | 864.0 | 96.0 | 0.0 | 960.0 | 62.014 | +480.0 | +0.0 | 65.579 |
| pipe-0.25 | uniform | 0.749 | graph | 0.0 | 2.0 | 0.0 | 961.1 | 56.594 | +0.0 | +0.0 | 56.638 |
| pipe-0.25 | ctrl0 | 0.000 | eager | 864.0 | 96.0 | 0.0 | 959.2 | 4.532 | +480.0 | +0.0 | 25.033 |
| pipe-0.25 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 961.1 | 4.387 | +0.0 | +0.0 | 4.750 |
| pipe-0.25 | ctrl4 | 0.500 | eager | 864.0 | 96.0 | 0.0 | 959.2 | 36.380 | +480.0 | +0.0 | 44.543 |
| pipe-0.25 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 961.1 | 43.499 | +0.0 | +0.0 | 36.685 |
| pipe-0.25 | ctrl8 | 1.000 | eager | 864.0 | 96.0 | 0.0 | 959.1 | 78.158 | +480.0 | +0.0 | 75.377 |
| pipe-0.25 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 961.0 | 74.420 | +0.0 | +0.0 | 74.595 |
| pipe-0.50 | uniform | 0.494 | eager | 864.0 | 96.0 | 0.0 | 959.5 | 42.731 | +480.0 | +0.0 | 45.082 |
| pipe-0.50 | uniform | 0.494 | graph | 0.0 | 2.0 | 0.0 | 961.4 | 44.561 | +0.0 | +0.0 | 44.802 |
| pipe-0.50 | ctrl0 | 0.000 | eager | 864.0 | 96.0 | 0.0 | 959.5 | 4.531 | +480.0 | +0.0 | 24.992 |
| pipe-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 961.4 | 4.397 | +0.0 | +0.0 | 4.766 |
| pipe-0.50 | ctrl4 | 0.500 | eager | 864.0 | 96.0 | 0.0 | 959.5 | 43.544 | +480.0 | +0.0 | 36.691 |
| pipe-0.50 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 961.4 | 35.384 | +0.0 | +0.0 | 31.948 |
| pipe-0.50 | ctrl8 | 1.000 | eager | 864.0 | 96.0 | 0.0 | 959.4 | 67.856 | +480.0 | +0.0 | 66.899 |
| pipe-0.50 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 961.2 | 58.893 | +0.0 | +0.0 | 66.464 |
| pipe-0.83 | uniform | 0.163 | eager | 864.0 | 96.0 | 0.0 | 959.8 | 18.665 | +480.0 | +0.0 | 25.166 |
| pipe-0.83 | uniform | 0.163 | graph | 0.0 | 2.0 | 0.0 | 961.8 | 19.038 | +0.0 | +0.0 | 19.113 |
| pipe-0.83 | ctrl0 | 0.000 | eager | 864.0 | 96.0 | 0.0 | 959.8 | 4.535 | +480.0 | +0.0 | 27.443 |
| pipe-0.83 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 961.6 | 4.399 | +0.0 | +0.0 | 4.756 |
| pipe-0.83 | ctrl4 | 0.500 | eager | 864.0 | 96.0 | 0.0 | 959.8 | 34.463 | +480.0 | +0.0 | 43.186 |
| pipe-0.83 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 961.6 | 42.694 | +0.0 | +0.0 | 34.808 |
| pipe-0.83 | ctrl8 | 1.000 | eager | 864.0 | 96.0 | 0.0 | 959.6 | 69.917 | +480.0 | +0.0 | 79.888 |
| pipe-0.83 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 961.5 | 72.134 | +0.0 | +0.0 | 72.057 |
| pipe-1.00 | uniform | 0.000 | eager | 864.0 | 96.0 | 0.0 | 960.0 | 4.534 | +480.0 | +0.0 | 26.301 |
| pipe-1.00 | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 962.0 | 4.396 | +0.0 | +0.0 | 4.754 |
| ref | uniform | 0.000 | eager | 384.0 | 0.0 | 0.0 | 384.0 | 3.297 | +0.0 | +0.0 | 18.247 |
| ref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 386.0 | 3.202 | +0.0 | +0.0 | 3.353 |

### gpt-oss-20b (reference: `mref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnvme-0.00 | uniform | 1.000 | eager | 888.0 | 144.0 | 96.0 | 1032.0 | 92.715 | -24.0 | +72.0 | 340.139 |
| mnvme-0.00 | uniform | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | uniform | 0.737 | eager | 888.0 | 144.0 | 96.0 | 1032.0 | 69.883 | -24.0 | +72.0 | 228.578 |
| mnvme-0.25 | uniform | 0.737 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl0 | 0.000 | eager | 888.0 | 144.0 | 96.0 | 1031.4 | 5.364 | -24.0 | +72.0 | 19.625 |
| mnvme-0.25 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl2 | 0.500 | eager | 888.0 | 144.0 | 96.0 | 1031.4 | 48.889 | -24.0 | +72.0 | 172.951 |
| mnvme-0.25 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl4 | 1.000 | eager | 888.0 | 144.0 | 96.0 | 1031.2 | 93.035 | -24.0 | +72.0 | 306.606 |
| mnvme-0.25 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | uniform | 0.533 | eager | 888.0 | 144.0 | 96.0 | 1031.9 | 53.984 | -24.0 | +72.0 | 162.725 |
| mnvme-0.50 | uniform | 0.533 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl0 | 0.000 | eager | 888.0 | 144.0 | 96.0 | 1031.9 | 5.354 | -24.0 | +72.0 | 19.667 |
| mnvme-0.50 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl2 | 0.500 | eager | 888.0 | 144.0 | 96.0 | 1031.8 | 45.133 | -24.0 | +72.0 | 167.455 |
| mnvme-0.50 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl4 | 1.000 | eager | 888.0 | 144.0 | 96.0 | 1031.8 | 94.069 | -24.0 | +72.0 | 286.092 |
| mnvme-0.50 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-1.00 | uniform | 0.000 | eager | 888.0 | 144.0 | 96.0 | 1032.0 | 5.337 | -24.0 | +72.0 | 19.525 |
| mnvme-1.00 | uniform | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mpin-0.00 | uniform | 1.000 | eager | 912.0 | 72.0 | 24.0 | 983.9 | 69.197 | +0.0 | +0.0 | 74.869 |
| mpin-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 913.8 | 86.275 | +0.0 | +0.0 | 75.169 |
| mpin-0.50 | uniform | 0.533 | eager | 912.0 | 72.0 | 24.0 | 984.0 | 50.943 | +0.0 | +0.0 | 52.058 |
| mpin-0.50 | uniform | 0.533 | graph | 0.0 | 2.0 | 0.0 | 914.0 | 44.653 | +0.0 | +0.0 | 52.691 |
| mpin-0.50 | ctrl0 | 0.000 | eager | 912.0 | 72.0 | 24.0 | 984.0 | 5.386 | +0.0 | +0.0 | 17.601 |
| mpin-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 914.0 | 5.232 | +0.0 | +0.0 | 5.589 |
| mpin-0.50 | ctrl2 | 0.500 | eager | 912.0 | 72.0 | 24.0 | 984.0 | 51.865 | +0.0 | +0.0 | 48.790 |
| mpin-0.50 | ctrl2 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 913.9 | 47.662 | +0.0 | +0.0 | 47.514 |
| mpin-0.50 | ctrl4 | 1.000 | eager | 912.0 | 72.0 | 24.0 | 983.9 | 75.999 | +0.0 | +0.0 | 74.791 |
| mpin-0.50 | ctrl4 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 913.9 | 77.229 | +0.0 | +0.0 | 81.336 |
| mref | uniform | 0.000 | eager | 912.0 | 72.0 | 24.0 | 984.0 | 5.371 | +0.0 | +0.0 | 18.349 |
| mref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 914.0 | 5.220 | +0.0 | +0.0 | 5.567 |

### Verdicts (pre-registered)

- **P0 (gates): ALL HOLD** -- nesting 39/39, simulator 14/14, graph check 21/21, graph parity 21/21, budget coverage 39/39, record completeness 39/39 (162 dropped device record(s) in total)
- **P1: HOLDS** -- `{"arms": {"pipe-0.00/eager": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.00/graph": {"windows": 1, "fixed": true, "spreads": {"graph_nodes.work": 0.0, "graph_nodes.kernel": 0.0, "graph_nodes.memcpy": 0.0, "graph_nodes.memset": 0.0}}, "pipe-0.25/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.25/graph": {"windows": 4, "fixed": true, "spreads": {"graph_nodes.work": 0.0, "graph_nodes.kernel": 0.0, "graph_nodes.memcpy": 0.0, "graph_nodes.memset`
- **P2: HOLDS** -- `{"arms": {"pipe-0.00": {"delta_per_layer": {"launches": 7.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 7, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 5.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 5, "memcpy": 2, "sync": 0}, "holds": true, "delta_per_token": {"launches": 336.0, "async_copies": 96.0, "syncs": 0.0}}, "pipe-0.25": {"delta_per_layer": {"launches": 10.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 10, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 8.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 8, "`
- **P3: HOLDS** -- `{"band_ms": [0.5, 1.5], "families": {"qwen3-30b-a3b": {"graph": {"device_ms_per_token": 1.1947396249999789, "added_kernels_ms_per_token": 1.1840007499999816, "shared_kernels_ms_per_token": 0.010738874999997278, "shared_by_kernel_us": {"_gemv_nf4_dotpad": 10.738874999997279}, "submissions_per_token": 0.0, "wall_ms_per_token": 1.401218760292977}, "eager": {"device_ms_per_token": 1.237421749999888, "added_kernels_ms_per_token": 1.2325908749999352, "shared_kernels_ms_per_token": 0.004830874999952812, "shared_by_kernel_us": {"_gemv_nf4_dotpad": 4.830874999952812}, "submissions_per_token": 576.0, "w`
- **P4: HOLDS** -- `{"arms": {"mnvme-0.00": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.25": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.50": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1,`
- **P5: HOLDS** -- `{"arms": {"mpin-0.00": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}, "mpin-0.50": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}}}`
- **P6: HOLDS** -- `{"arms": {"hyb-0.25": {"launch_spread_per_token": 1632.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed", "syncs_per_layer": 10.0, "launches_per_layer": 75.0, "want_syncs": 10}, "ctrl8": {"composition": "cold_only", "syncs_per_layer": 8.0, "launches_per_layer": 50.0, "want_syncs": 8}}, "holds": true}, "hyb-0.50": {"launch_spread_per_token": 1632.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed`
- **P7: REFUTED** -- `{"band": [0.8, 1.25], "refute_above": 1.5, "windows": {"hyb-0.25/uniform": {"path": "hybrid", "measured_us": 185852.00787499774, "hot_d2d_us": 0.0, "predicted_us": 33592.684861908485, "ratio": 5.5325142553800335}, "hyb-0.25/ctrl4": {"path": "hybrid", "measured_us": 197902.85575000357, "hot_d2d_us": 0.0, "predicted_us": 22414.58034226386, "ratio": 8.829201918041152}, "hyb-0.25/ctrl8": {"path": "hybrid", "measured_us": 408220.08599999687, "hot_d2d_us": 0.0, "predicted_us": 44829.16068452772, "ratio": 9.106128238106617}, "hyb-0.50/uniform": {"path": "hybrid", "measured_us": 176195.160624999, "hot`
- **P8 (informational)** -- `{"status": "INFORMATIONAL", "rows": {"qwen3-30b-a3b/pipe-0.00/graph": {"cold_fraction": 1.0, "residency_ms": 75.93955937772989, "fixed_tax_ms": 1.401218760292977, "tax_share": 0.01845176310970142}, "qwen3-30b-a3b/pipe-0.00/eager": {"cold_fraction": 1.0, "residency_ms": 43.46445814007893, "fixed_tax_ms": 8.05387951550074, "tax_share": 0.1852980540915611}, "qwen3-30b-a3b/pipe-0.25/graph": {"cold_fraction": 0.75, "residency_ms": 53.285561880329624, "fixed_tax_ms": 1.401218760292977, "tax_share": 0.026296405833908215}, "qwen3-30b-a3b/pipe-0.25/eager": {"cold_fraction": 0.75, "residency_ms": 47.332494374131784, "fixed_tax_ms": 8.05387951550074, "tax_share": 0.17015540004801344}, "qwen3-30b-a3b/pipe-0.50/graph": {"cold_fraction": 0.5, "residency_ms": 41.44928851746954, "fixed_tax_ms": 1.401218760292977, "tax_share": 0.0338056167044317}, "qwen3-30b-a3b/pipe-0.50/eager": {"cold_fraction": 0.5, "`
- **P9 (informational)** -- `{"status": "INFORMATIONAL", "cross_level": {"M_pipe-0.00": {"level_m_per_step": {"cudaLaunchKernel": 384.0, "cudaMemcpyAsync": 96.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 384.0, "cudaMemcpyAsync": 96.0, "cudaStreamSynchronize": 0.0}, "agree": true}, "M_pipe-1.00": {"level_m_per_step": {"cudaLaunchKernel": 528.0, "cudaMemcpyAsync": 96.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 528.0, "cudaMemcpyAsync": 96.0, "cudaStreamSynchronize": 0.0}, "agree": true}}, "level_m": {"M_hyb-0.50": {"steps": 12, "coverage": 0.9923009917488927, "kernel_calls_per_step": 5263.0, "device_ms_per_step": 78.8021775, "manifest_counts": {"vram": 3072, "dram": 0, "nvme": 3072}, "active_steps": 12, "per_step": {"cudaLaunchKernel": 5296.0, "cudaMemcpyAsync": 379.0, "cudaStreamSynchronize": 211.0, "cudaDeviceSynchronize": 0.08333333333333333, "aten::it`

**Decision rule:** P1 ∧ P2: the pipelined engine's residency cost is a FIXED per-layer count (as registered), independent of cold fraction -> register the measured claim; the cost model gets a per-step fixed term (P3's graph device time, and submissions x the host's unit cost when eager), never a per-cold-expert launch term (grouped-nf4-gemm follow-up on kernel/cold_deadline.py) | P4: the MXFP4 NVMe engine's +3 syncs/layer are structural; a pinned-staging follow-up (two of the three are pageable H2D copies) is licensed only if their host cost >= 10 % of that token | P6: the hybrid tier's cost is layer-COMPOSITION-dependent (hot-only / mixed / cold-only); cold_dest='deadline' prices only bytes, so its GPU side omits the mixed-layer dispatch term | P7 REFUTED: the transfer itself departs from bytes/link -- the model needs a measured efficiency factor (UVA-read efficiency, #105 candidate 4) before any RFC comparison | P8: at the RFC's 17.2 % cold the fixed tax is 9% of residency's cost -- in this engine launches cannot explain an RFC-size (2-4x) gap; the gap would be link-side
