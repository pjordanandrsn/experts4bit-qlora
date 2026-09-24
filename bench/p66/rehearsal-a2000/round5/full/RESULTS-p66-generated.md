## Generated read (p66_reduce.py, unedited)

Box: NVIDIA RTX A2000 12GB, 12282 MiB, 575.64.05, 1, 3, 8, 16, 70.00 W, 2100 MHz (26 SMs) on Intel(R) Xeon(R) W-1250 CPU @ 3.30GHz; torch 2.8.0+cu128, triton 3.4.0. Host unit costs: {"aten_launch_host_us": 7.482761610299349, "triton_launch_host_us": 19.42523461766541, "sync_idle_us": 4.761422984302044, "item_roundtrip_us": 7.201813976280391, "h2d_pinned_64mb_gbs": 6.253998883353825, "d2d_copy_rw_gbs": 253.97844956981262}. Transfer constants: {"source": "calib:calib.json", "b_link_gbs": 6.04, "b_vram_gbs": 257.4, "b_dram_gbs": 16.51}.

### olmoe-1b-7b (reference: `ref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb-0.25 | uniform | 0.734 | eager | 1183.2 | 317.5 | 159.0 | 1500.8 | 111.694 | +1055.2 | +159.0 | 1637.293 |
| hyb-0.25 | uniform | 0.734 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 764.0 | 17.901 | +528.0 | +64.0 | 30.977 |
| hyb-0.25 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1520.0 | 74.379 | +1072.0 | +160.0 | 2159.176 |
| hyb-0.25 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1040.0 | 115.973 | +672.0 | +128.0 | 4147.960 |
| hyb-0.25 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | uniform | 0.465 | eager | 1199.5 | 320.0 | 160.0 | 1519.5 | 78.386 | +1071.5 | +160.0 | 1324.956 |
| hyb-0.50 | uniform | 0.465 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 768.0 | 17.948 | +528.0 | +64.0 | 32.141 |
| hyb-0.50 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1520.0 | 91.760 | +1072.0 | +160.0 | 2949.918 |
| hyb-0.50 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1040.0 | 215.499 | +672.0 | +128.0 | 1924.257 |
| hyb-0.50 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| pipe-0.00 | uniform | 1.000 | eager | 240.0 | 32.0 | 0.0 | 271.8 | 89.464 | +112.0 | +0.0 | 87.418 |
| pipe-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 273.8 | 87.378 | +0.0 | +0.0 | 86.067 |
| pipe-0.25 | uniform | 0.734 | eager | 288.0 | 32.0 | 0.0 | 319.8 | 65.173 | +160.0 | +0.0 | 65.656 |
| pipe-0.25 | uniform | 0.734 | graph | 0.0 | 2.0 | 0.0 | 321.8 | 65.523 | +0.0 | +0.0 | 68.656 |
| pipe-0.25 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 319.8 | 10.737 | +160.0 | +0.0 | 13.189 |
| pipe-0.25 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.676 | +0.0 | +0.0 | 10.791 |
| pipe-0.25 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 319.8 | 46.868 | +160.0 | +0.0 | 47.802 |
| pipe-0.25 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 321.8 | 47.099 | +0.0 | +0.0 | 46.486 |
| pipe-0.25 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 319.8 | 85.152 | +160.0 | +0.0 | 86.848 |
| pipe-0.25 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 84.775 | +0.0 | +0.0 | 85.947 |
| pipe-0.50 | uniform | 0.465 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.092 | +160.0 | +0.0 | 46.800 |
| pipe-0.50 | uniform | 0.465 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.924 | +0.0 | +0.0 | 48.306 |
| pipe-0.50 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 10.838 | +160.0 | +0.0 | 12.093 |
| pipe-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.670 | +0.0 | +0.0 | 10.806 |
| pipe-0.50 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 53.418 | +160.0 | +0.0 | 48.092 |
| pipe-0.50 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 46.286 | +0.0 | +0.0 | 46.949 |
| pipe-0.50 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 89.826 | +160.0 | +0.0 | 86.679 |
| pipe-0.50 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 85.174 | +0.0 | +0.0 | 100.147 |
| pipe-0.83 | uniform | 0.172 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 27.179 | +160.0 | +0.0 | 21.471 |
| pipe-0.83 | uniform | 0.172 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 25.157 | +0.0 | +0.0 | 22.916 |
| pipe-0.83 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 10.511 | +160.0 | +0.0 | 30.322 |
| pipe-0.83 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 13.724 | +0.0 | +0.0 | 14.209 |
| pipe-0.83 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 48.339 | +160.0 | +0.0 | 46.247 |
| pipe-0.83 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 49.841 | +0.0 | +0.0 | 45.378 |
| pipe-0.83 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 82.752 | +160.0 | +0.0 | 81.386 |
| pipe-0.83 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 83.403 | +0.0 | +0.0 | 86.894 |
| pipe-1.00 | uniform | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 16.927 | +160.0 | +0.0 | 27.575 |
| pipe-1.00 | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.885 | +0.0 | +0.0 | 11.078 |
| ref | uniform | 0.000 | eager | 128.0 | 0.0 | 0.0 | 128.0 | 16.040 | +0.0 | +0.0 | 13.446 |
| ref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 130.0 | 10.102 | +0.0 | +0.0 | 10.354 |

### gpt-oss-20b (reference: `mref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnvme-0.00 | uniform | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 79.967 | -8.0 | +24.0 | 151.320 |
| mnvme-0.00 | uniform | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | uniform | 0.766 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 71.218 | -8.0 | +24.0 | 193.984 |
| mnvme-0.25 | uniform | 0.766 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 11.611 | -8.0 | +24.0 | 20.811 |
| mnvme-0.25 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 43.683 | -8.0 | +24.0 | 94.516 |
| mnvme-0.25 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 77.416 | -8.0 | +24.0 | 169.419 |
| mnvme-0.25 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | uniform | 0.523 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 49.089 | -8.0 | +24.0 | 89.597 |
| mnvme-0.50 | uniform | 0.523 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.125 | -8.0 | +24.0 | 20.445 |
| mnvme-0.50 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 42.951 | -8.0 | +24.0 | 67.938 |
| mnvme-0.50 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 84.216 | -8.0 | +24.0 | 234.905 |
| mnvme-0.50 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-1.00 | uniform | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 12.017 | -8.0 | +24.0 | 20.642 |
| mnvme-1.00 | uniform | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mpin-0.00 | uniform | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 76.647 | +0.0 | +0.0 | 93.300 |
| mpin-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 79.646 | +0.0 | +0.0 | 84.923 |
| mpin-0.50 | uniform | 0.523 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 45.446 | +0.0 | +0.0 | 44.971 |
| mpin-0.50 | uniform | 0.523 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 37.621 | +0.0 | +0.0 | 36.748 |
| mpin-0.50 | ctrl0 | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 9.958 | +0.0 | +0.0 | 13.177 |
| mpin-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 9.897 | +0.0 | +0.0 | 10.016 |
| mpin-0.50 | ctrl2 | 0.500 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 41.262 | +0.0 | +0.0 | 48.655 |
| mpin-0.50 | ctrl2 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 41.032 | +0.0 | +0.0 | 41.129 |
| mpin-0.50 | ctrl4 | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 75.839 | +0.0 | +0.0 | 84.276 |
| mpin-0.50 | ctrl4 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 75.817 | +0.0 | +0.0 | 76.145 |
| mref | uniform | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 12.058 | +0.0 | +0.0 | 19.315 |
| mref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 11.069 | +0.0 | +0.0 | 11.111 |

### Verdicts (pre-registered)

- **P0 (gates): FAILED** -- nesting 39/39, simulator 14/14, graph check 21/21, graph parity 18/21, budget coverage 39/39, record completeness 38/39 (24 dropped device record(s) in total)
- **P1: REFUTED** -- `{"arms": {"pipe-0.00/eager": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.00/graph": {"windows": 1, "fixed": true, "spreads": {"kernels": 0.0}}, "pipe-0.25/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.25/graph": {"windows": 4, "fixed": false, "spreads": {"kernels": 0.25}}, "pipe-0.50/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.50/graph": {"windows": 4, "fi`
- **P2: HOLDS** -- `{"arms": {"pipe-0.00": {"delta_per_layer": {"launches": 7.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 7, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 5.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 5, "memcpy": 2, "sync": 0}, "holds": true, "delta_per_token": {"launches": 112.0, "async_copies": 32.0, "syncs": 0.0}}, "pipe-0.25": {"delta_per_layer": {"launches": 10.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 10, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 8.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 8, "`
- **P3: RECORDED (not the registered box)** -- `{"band_ms": [0.5, 1.5], "families": {"olmoe-1b-7b": {"graph": {"device_ms_per_token": 0.7834232499999726, "added_kernels_ms_per_token": 0.5020867499999845, "shared_kernels_ms_per_token": 0.2813364999999885, "shared_by_kernel_us": {"_gemv_nf4_grouped": 281.3364999999885}, "submissions_per_token": 0.0, "wall_ms_per_token": 0.7244222797453403}, "eager": {"device_ms_per_token": 0.8877900000000173, "added_kernels_ms_per_token": 0.8282332500000175, "shared_kernels_ms_per_token": 0.05955674999999792, "shared_by_kernel_us": {"_gemv_nf4_grouped": 59.55674999999792}, "submissions_per_token": 192.0, "wal`
- **P4: HOLDS** -- `{"arms": {"mnvme-0.00": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.25": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.50": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1,`
- **P5: HOLDS** -- `{"arms": {"mpin-0.00": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}, "mpin-0.50": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}}}`
- **P6: HOLDS** -- `{"arms": {"hyb-0.25": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed", "syncs_per_layer": 10.0, "launches_per_layer": 75.0, "want_syncs": 10}, "ctrl8": {"composition": "cold_only", "syncs_per_layer": 8.0, "launches_per_layer": 50.0, "want_syncs": 8}}, "holds": true}, "hyb-0.50": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed",`
- **P7: RECORDED (not the registered box)** -- `{"band": [0.8, 1.25], "refute_above": 1.5, "windows": {"hyb-0.25/uniform": {"path": "hybrid", "measured_us": 90663.80674999882, "hot_d2d_us": 0.0, "predicted_us": 56368.66904645024, "ratio": 1.6084077960983592}, "hyb-0.25/ctrl4": {"path": "hybrid", "measured_us": 54909.10300000057, "hot_d2d_us": 0.0, "predicted_us": 38378.66828694485, "ratio": 1.4307193410011787}, "hyb-0.25/ctrl8": {"path": "hybrid", "measured_us": 97656.98300000074, "hot_d2d_us": 0.0, "predicted_us": 76757.3365738897, "ratio": 1.2722820691673193}, "hyb-0.50/uniform": {"path": "hybrid", "measured_us": 57720.084499999415, "hot_`
- **P8 (informational)** -- `{"status": "INFORMATIONAL", "rows": {"olmoe-1b-7b/pipe-0.00/graph": {"cold_fraction": 1.0, "residency_ms": 75.71285724407062, "fixed_tax_ms": 0.7244222797453403, "tax_share": 0.00956802194652445}, "olmoe-1b-7b/pipe-0.00/eager": {"cold_fraction": 1.0, "residency_ms": 73.97162506822497, "fixed_tax_ms": 14.128709735814482, "tax_share": 0.1910017486135176}, "olmoe-1b-7b/pipe-0.25/graph": {"cold_fraction": 0.75, "residency_ms": 58.30246349796653, "fixed_tax_ms": 0.7244222797453403, "tax_share": 0.01242524305633512}, "olmoe-1b-7b/pipe-0.25/eager": {"cold_fraction": 0.75, "residency_ms": 52.21005401108414, "fixed_tax_ms": 14.128709735814482, "tax_share": 0.2706128159303373}, "olmoe-1b-7b/pipe-0.50/graph": {"cold_fraction": 0.5, "residency_ms": 37.95184276532382, "fixed_tax_ms": 0.7244222797453403, "tax_share": 0.019087934259867798}, "olmoe-1b-7b/pipe-0.50/eager": {"cold_fraction": 0.5, "residen`
- **P9 (informational)** -- `{"status": "INFORMATIONAL", "cross_level": {"M_pipe-0.00": {"level_m_per_step": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}, "M_pipe-1.00": {"level_m_per_step": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}}, "level_m": {"M_hyb-0.50": {"steps": 12, "coverage": 0.9928530367282237, "kernel_calls_per_step": 1719.0000000000002, "device_ms_per_step": 73.90665625, "manifest_counts": {"vram": 512, "dram": 0, "nvme": 512}, "active_steps": 12, "per_step": {"cudaLaunchKernel": 1778.6666666666667, "cudaMemcpyAsync": 128.33333333333334, "cudaStreamSynchronize": 72.33333333333333, "cudaDe`

**Decision rule:** ¬P0: an instrument gate failed -- nothing is read; the failing gate is filed and the run repeated once
