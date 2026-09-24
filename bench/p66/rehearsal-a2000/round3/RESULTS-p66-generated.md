## Generated read (p66_reduce.py, unedited)

Box: NVIDIA RTX A2000 12GB, 12282 MiB, 575.64.05, 3, 3, 8, 16, 70.00 W, 2100 MHz (26 SMs) on Intel(R) Xeon(R) W-1250 CPU @ 3.30GHz; torch 2.8.0+cu128, triton 3.4.0. Host unit costs: {"aten_launch_host_us": 6.389903975650668, "triton_launch_host_us": 21.351088024675846, "sync_idle_us": 7.607964100316167, "item_roundtrip_us": 8.782081422396004, "h2d_pinned_64mb_gbs": 6.199414293482654, "d2d_copy_rw_gbs": 253.32396808133527}. Transfer constants: {"source": "calib:calib.json", "b_link_gbs": 6.25, "b_vram_gbs": 257.4, "b_dram_gbs": 22.26}.

### olmoe-1b-7b (reference: `ref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb-0.25 | uniform | 0.734 | eager | 1183.2 | 317.5 | 159.0 | 1500.2 | 79.270 | +1055.2 | +159.0 | 231.444 |
| hyb-0.25 | uniform | 0.734 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 767.5 | 17.478 | +528.0 | +64.0 | 29.145 |
| hyb-0.25 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1519.5 | 53.511 | +1072.0 | +160.0 | 308.276 |
| hyb-0.25 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.2 | 101.152 | +672.0 | +128.0 | 476.183 |
| hyb-0.25 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | uniform | 0.465 | eager | 1199.5 | 320.0 | 160.0 | 1519.5 | 59.016 | +1071.5 | +160.0 | 328.753 |
| hyb-0.50 | uniform | 0.465 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 768.0 | 16.907 | +528.0 | +64.0 | 28.809 |
| hyb-0.50 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1520.0 | 60.286 | +1072.0 | +160.0 | 224.126 |
| hyb-0.50 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.8 | 113.659 | +672.0 | +128.0 | 243.193 |
| hyb-0.50 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| pipe-0.00 | uniform | 1.000 | eager | 240.0 | 32.0 | 0.0 | 272.0 | 84.718 | +112.0 | +0.0 | 85.511 |
| pipe-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 274.0 | 84.953 | +0.0 | +0.0 | 85.473 |
| pipe-0.25 | uniform | 0.734 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 65.541 | +160.0 | +0.0 | 65.645 |
| pipe-0.25 | uniform | 0.734 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 65.856 | +0.0 | +0.0 | 68.760 |
| pipe-0.25 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.432 | +160.0 | +0.0 | 12.047 |
| pipe-0.25 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.960 | +0.0 | +0.0 | 10.997 |
| pipe-0.25 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.417 | +160.0 | +0.0 | 47.090 |
| pipe-0.25 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 46.678 | +0.0 | +0.0 | 46.113 |
| pipe-0.25 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 84.969 | +160.0 | +0.0 | 85.607 |
| pipe-0.25 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 84.412 | +0.0 | +0.0 | 84.913 |
| pipe-0.50 | uniform | 0.465 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 45.378 | +160.0 | +0.0 | 46.847 |
| pipe-0.50 | uniform | 0.465 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.843 | +0.0 | +0.0 | 47.615 |
| pipe-0.50 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 10.714 | +160.0 | +0.0 | 11.921 |
| pipe-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.709 | +0.0 | +0.0 | 10.711 |
| pipe-0.50 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 46.948 | +160.0 | +0.0 | 46.811 |
| pipe-0.50 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 45.456 | +0.0 | +0.0 | 45.926 |
| pipe-0.50 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 83.666 | +160.0 | +0.0 | 84.210 |
| pipe-0.50 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 82.652 | +0.0 | +0.0 | 82.706 |
| pipe-0.83 | uniform | 0.172 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 23.232 | +160.0 | +0.0 | 21.292 |
| pipe-0.83 | uniform | 0.172 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 23.089 | +0.0 | +0.0 | 21.408 |
| pipe-0.83 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 10.673 | +160.0 | +0.0 | 14.001 |
| pipe-0.83 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.714 | +0.0 | +0.0 | 10.621 |
| pipe-0.83 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 43.994 | +160.0 | +0.0 | 45.509 |
| pipe-0.83 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.584 | +0.0 | +0.0 | 44.961 |
| pipe-0.83 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 76.287 | +160.0 | +0.0 | 80.061 |
| pipe-0.83 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 83.022 | +0.0 | +0.0 | 79.318 |
| pipe-1.00 | uniform | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 16.911 | +160.0 | +0.0 | 17.281 |
| pipe-1.00 | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.729 | +0.0 | +0.0 | 10.602 |
| ref | uniform | 0.000 | eager | 128.0 | 0.0 | 0.0 | 128.0 | 16.080 | +0.0 | +0.0 | 16.279 |
| ref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 130.0 | 10.194 | +0.0 | +0.0 | 10.333 |

### gpt-oss-20b (reference: `mref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnvme-0.00 | uniform | 1.000 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 76.093 | -8.0 | +24.0 | 137.728 |
| mnvme-0.00 | uniform | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | uniform | 0.766 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 61.830 | -8.0 | +24.0 | 131.090 |
| mnvme-0.25 | uniform | 0.766 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 10.224 | -8.0 | +24.0 | 14.894 |
| mnvme-0.25 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 41.147 | -8.0 | +24.0 | 73.675 |
| mnvme-0.25 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.160 | -8.0 | +24.0 | 168.482 |
| mnvme-0.25 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | uniform | 0.523 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 44.922 | -8.0 | +24.0 | 78.695 |
| mnvme-0.50 | uniform | 0.523 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 10.183 | -8.0 | +24.0 | 13.284 |
| mnvme-0.50 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.142 | -8.0 | +24.0 | 80.746 |
| mnvme-0.50 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 343.8 | 76.002 | -8.0 | +24.0 | 161.478 |
| mnvme-0.50 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-1.00 | uniform | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 12.019 | -8.0 | +24.0 | 16.379 |
| mnvme-1.00 | uniform | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mpin-0.00 | uniform | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 75.910 | +0.0 | +0.0 | 88.987 |
| mpin-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 75.911 | +0.0 | +0.0 | 76.089 |
| mpin-0.50 | uniform | 0.523 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 45.732 | +0.0 | +0.0 | 46.926 |
| mpin-0.50 | uniform | 0.523 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 38.541 | +0.0 | +0.0 | 38.337 |
| mpin-0.50 | ctrl0 | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 10.053 | +0.0 | +0.0 | 13.355 |
| mpin-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 10.015 | +0.0 | +0.0 | 10.058 |
| mpin-0.50 | ctrl2 | 0.500 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 42.489 | +0.0 | +0.0 | 56.873 |
| mpin-0.50 | ctrl2 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 41.315 | +0.0 | +0.0 | 41.281 |
| mpin-0.50 | ctrl4 | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 76.542 | +0.0 | +0.0 | 86.840 |
| mpin-0.50 | ctrl4 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 76.236 | +0.0 | +0.0 | 77.001 |
| mref | uniform | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 12.058 | +0.0 | +0.0 | 24.708 |
| mref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 11.888 | +0.0 | +0.0 | 11.965 |

### Verdicts (pre-registered)

- **P0 (gates): ALL HOLD** -- nesting 39/39, simulator 14/14, graph check 21/21, graph parity 21/21, budget coverage 39/39
- **P1: HOLDS** -- `{"arms": {"pipe-0.00/eager": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "kernels": 0.0, "memsets": 0.0}}, "pipe-0.00/graph": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "kernels": 0.0, "memsets": 0.0}}, "pipe-0.25/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "kernels": 0.0, "memsets": 0.0}}, "pipe-0.25/graph": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "kernels": 0.0, "memsets": 0.0}}, "p`
- **P2: HOLDS** -- `{"arms": {"pipe-0.00": {"delta_per_layer": {"launches": 7.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 7, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 5.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 5, "memcpy": 2, "sync": 0}, "holds": true, "delta_per_token": {"launches": 112.0, "async_copies": 32.0, "syncs": 0.0}}, "pipe-0.25": {"delta_per_layer": {"launches": 10.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 10, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 8.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 8, "`
- **P3: RECORDED (not the registered box)** -- `{"band_ms": [0.5, 1.5], "families": {"olmoe-1b-7b": {"graph": {"device_ms_per_token": 0.5348027500000134, "added_kernels_ms_per_token": 0.4922630000000132, "shared_kernels_ms_per_token": 0.04253975000000173, "shared_by_kernel_us": {"_gemv_nf4_grouped": 42.53975000000173}, "submissions_per_token": 0.0, "wall_ms_per_token": 0.26848644483834505}, "eager": {"device_ms_per_token": 0.8309205000000075, "added_kernels_ms_per_token": 0.8246687500000188, "shared_kernels_ms_per_token": 0.006251749999986714, "shared_by_kernel_us": {"_gemv_nf4_grouped": 6.251749999986714}, "submissions_per_token": 192.0, "`
- **P4: REFUTED** -- `{"arms": {"mnvme-0.00": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.25": {"fixed": false, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": false}, "mnvme-0.50": {"fixed": false, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": `
- **P5: HOLDS** -- `{"arms": {"mpin-0.00": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}, "mpin-0.50": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}}}`
- **P6: HOLDS** -- `{"arms": {"hyb-0.25": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed", "syncs_per_layer": 10.0, "launches_per_layer": 75.0, "want_syncs": 10}, "ctrl8": {"composition": "cold_only", "syncs_per_layer": 8.0, "launches_per_layer": 50.0, "want_syncs": 8}}, "holds": true}, "hyb-0.50": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed",`
- **P7: RECORDED (not the registered box)** -- `{"band": [0.8, 1.25], "refute_above": 1.5, "windows": {"hyb-0.25/uniform": {"path": "hybrid", "measured_us": 58162.90624999992, "hot_d2d_us": 0.0, "predicted_us": 54518.106011748256, "ratio": 1.0668548580441557}, "hyb-0.25/ctrl4": {"path": "hybrid", "measured_us": 38835.11450000016, "hot_d2d_us": 0.0, "predicted_us": 37118.710476083914, "ratio": 1.046240938920067}, "hyb-0.25/ctrl8": {"path": "hybrid", "measured_us": 82833.08825000052, "hot_d2d_us": 0.0, "predicted_us": 74237.42095216783, "ratio": 1.1157861788244368}, "hyb-0.50/uniform": {"path": "hybrid", "measured_us": 38290.53299999979, "hot`
- **P8 (informational)** -- `{"status": "INFORMATIONAL", "rows": {"olmoe-1b-7b/pipe-0.00/graph": {"cold_fraction": 1.0, "residency_ms": 75.13986149569973, "fixed_tax_ms": 0.26848644483834505, "tax_share": 0.0035731559719964427}, "olmoe-1b-7b/pipe-0.00/eager": {"cold_fraction": 1.0, "residency_ms": 69.23233746783808, "fixed_tax_ms": 1.0023982031270862, "tax_share": 0.014478757178937528}, "olmoe-1b-7b/pipe-0.25/graph": {"cold_fraction": 0.75, "residency_ms": 58.427220676094294, "fixed_tax_ms": 0.26848644483834505, "tax_share": 0.0045952287603541146}, "olmoe-1b-7b/pipe-0.25/eager": {"cold_fraction": 0.75, "residency_ms": 49.36620203079656, "fixed_tax_ms": 1.0023982031270862, "tax_share": 0.020305353903906788}, "olmoe-1b-7b/pipe-0.50/graph": {"cold_fraction": 0.5, "residency_ms": 37.28136874269694, "fixed_tax_ms": 0.26848644483834505, "tax_share": 0.007201625205644815}, "olmoe-1b-7b/pipe-0.50/eager": {"cold_fraction": 0`
- **P9 (informational)** -- `{"status": "INFORMATIONAL", "level_m": {"M_hyb-0.50": {"steps": 12, "coverage": 0.9957912931873482, "kernel_calls_per_step": 1743.0000000000002, "device_ms_per_step": 136.4234071666667, "active_steps": 12, "per_step": {"cudaLaunchKernel": 1778.6666666666667, "cudaMemcpyAsync": 128.33333333333334, "cudaStreamSynchronize": 72.33333333333333, "cudaDeviceSynchronize": 0.08333333333333333, "aten::item": 8.0, "aten::nonzero": 32.0}, "delta_kernel_calls_per_step": 193.33333333333348}, "M_pipe-0.00": {"steps": 12, "coverage": 0.9999826616399623, "kernel_calls_per_step": 1682.3333333333335, "device_ms_per_step": 88.41513366666668, "active_steps": 12, "per_step": {"cudaLaunchKernel": 1562.6666666666667, "cudaMemcpyAsync": 40.333333333333336, "cudaStreamSynchronize": 8.333333333333334, "cudaDeviceSynchronize": 0.08333333333333333, "aten::item": 0.0, "aten::nonzero": 0.0}, "delta_kernel_calls_per_st`

**Decision rule:** P1 ∧ P2: the pipelined engine's residency cost is a FIXED per-layer count (as registered), independent of cold fraction -> register the measured claim; the cost model gets a per-step fixed term (P3's graph device time, and submissions x the host's unit cost when eager), never a per-cold-expert launch term (grouped-nf4-gemm follow-up on kernel/cold_deadline.py) | P6: the hybrid tier's cost is layer-COMPOSITION-dependent (hot-only / mixed / cold-only); cold_dest='deadline' prices only bytes, so its GPU side omits the mixed-layer dispatch term | P8: at the RFC's 17.2 % cold the fixed tax is 2% of residency's cost -- in this engine launches cannot explain an RFC-size (2-4x) gap; the gap would be link-side
