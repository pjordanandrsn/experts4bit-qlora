## Generated read (p66_reduce.py, unedited)

Box: NVIDIA RTX A2000 12GB, 12282 MiB, 575.64.05, 3, 3, 8, 16, 70.00 W, 2100 MHz (26 SMs) on Intel(R) Xeon(R) W-1250 CPU @ 3.30GHz; torch 2.8.0+cu128, triton 3.4.0. Host unit costs: {"aten_launch_host_us": 6.082575814798474, "triton_launch_host_us": 17.16322721913457, "sync_idle_us": 4.753533052280545, "item_roundtrip_us": 7.1851639077067375, "h2d_pinned_64mb_gbs": 6.252547515013926, "d2d_copy_rw_gbs": 254.71277396428385}. Transfer constants: {"source": "calib:calib.json", "b_link_gbs": 6.28, "b_vram_gbs": 257.6, "b_dram_gbs": 27.27}.

### olmoe-1b-7b (reference: `ref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb-0.25 | uniform | 0.734 | eager | 1183.2 | 317.5 | 159.0 | 1500.5 | 75.539 | +1055.2 | +159.0 | 178.541 |
| hyb-0.25 | uniform | 0.734 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 767.8 | 13.315 | +528.0 | +64.0 | 27.441 |
| hyb-0.25 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1519.8 | 50.054 | +1072.0 | +160.0 | 135.279 |
| hyb-0.25 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.25 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.5 | 88.313 | +672.0 | +128.0 | 217.185 |
| hyb-0.25 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | uniform | 0.465 | eager | 1199.5 | 320.0 | 160.0 | 1519.5 | 56.133 | +1071.5 | +160.0 | 149.344 |
| hyb-0.50 | uniform | 0.465 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl0 | 0.000 | eager | 656.0 | 112.0 | 64.0 | 768.0 | 16.575 | +528.0 | +64.0 | 26.857 |
| hyb-0.50 | ctrl0 | 0.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl4 | 0.500 | eager | 1200.0 | 320.0 | 160.0 | 1520.0 | 53.251 | +1072.0 | +160.0 | 142.981 |
| hyb-0.50 | ctrl4 | 0.500 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| hyb-0.50 | ctrl8 | 1.000 | eager | 800.0 | 240.0 | 128.0 | 1039.8 | 91.300 | +672.0 | +128.0 | 223.907 |
| hyb-0.50 | ctrl8 | 1.000 | graph | refused: the v0 dispatch under the hybrid tier synchronizes (nonzero  | | | | | | | |
| pipe-0.00 | uniform | 1.000 | eager | 240.0 | 32.0 | 0.0 | 272.0 | 86.562 | +112.0 | +0.0 | 86.023 |
| pipe-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 274.0 | 85.410 | +0.0 | +0.0 | 85.718 |
| pipe-0.25 | uniform | 0.734 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 65.304 | +160.0 | +0.0 | 66.037 |
| pipe-0.25 | uniform | 0.734 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 65.680 | +0.0 | +0.0 | 68.849 |
| pipe-0.25 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.168 | +160.0 | +0.0 | 11.898 |
| pipe-0.25 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.868 | +0.0 | +0.0 | 10.770 |
| pipe-0.25 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.118 | +160.0 | +0.0 | 49.091 |
| pipe-0.25 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 49.089 | +0.0 | +0.0 | 48.040 |
| pipe-0.25 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 87.952 | +160.0 | +0.0 | 89.500 |
| pipe-0.25 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 87.067 | +0.0 | +0.0 | 88.008 |
| pipe-0.50 | uniform | 0.465 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 44.358 | +160.0 | +0.0 | 46.811 |
| pipe-0.50 | uniform | 0.465 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.723 | +0.0 | +0.0 | 47.555 |
| pipe-0.50 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.093 | +160.0 | +0.0 | 11.857 |
| pipe-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.767 | +0.0 | +0.0 | 10.649 |
| pipe-0.50 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 47.352 | +160.0 | +0.0 | 46.713 |
| pipe-0.50 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 45.450 | +0.0 | +0.0 | 45.759 |
| pipe-0.50 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 83.554 | +160.0 | +0.0 | 84.766 |
| pipe-0.50 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 82.771 | +0.0 | +0.0 | 83.497 |
| pipe-0.83 | uniform | 0.172 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 22.720 | +160.0 | +0.0 | 21.107 |
| pipe-0.83 | uniform | 0.172 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 22.989 | +0.0 | +0.0 | 21.172 |
| pipe-0.83 | ctrl0 | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 11.029 | +160.0 | +0.0 | 11.712 |
| pipe-0.83 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 10.790 | +0.0 | +0.0 | 10.701 |
| pipe-0.83 | ctrl4 | 0.500 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 43.886 | +160.0 | +0.0 | 45.439 |
| pipe-0.83 | ctrl4 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 43.516 | +0.0 | +0.0 | 44.620 |
| pipe-0.83 | ctrl8 | 1.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 76.178 | +160.0 | +0.0 | 79.858 |
| pipe-0.83 | ctrl8 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 81.050 | +0.0 | +0.0 | 78.994 |
| pipe-1.00 | uniform | 0.000 | eager | 288.0 | 32.0 | 0.0 | 320.0 | 16.914 | +160.0 | +0.0 | 12.851 |
| pipe-1.00 | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 322.0 | 12.060 | +0.0 | +0.0 | 12.098 |
| ref | uniform | 0.000 | eager | 128.0 | 0.0 | 0.0 | 128.0 | 16.081 | +0.0 | +0.0 | 15.292 |
| ref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 130.0 | 10.126 | +0.0 | +0.0 | 10.357 |

### gpt-oss-20b (reference: `mref`)

| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnvme-0.00 | uniform | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 75.955 | -8.0 | +24.0 | 115.101 |
| mnvme-0.00 | uniform | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | uniform | 0.766 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 59.533 | -8.0 | +24.0 | 86.854 |
| mnvme-0.25 | uniform | 0.766 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.075 | -8.0 | +24.0 | 13.250 |
| mnvme-0.25 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.051 | -8.0 | +24.0 | 64.106 |
| mnvme-0.25 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.25 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 76.003 | -8.0 | +24.0 | 114.286 |
| mnvme-0.25 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | uniform | 0.523 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 44.536 | -8.0 | +24.0 | 56.011 |
| mnvme-0.50 | uniform | 0.523 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl0 | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 10.121 | -8.0 | +24.0 | 13.250 |
| mnvme-0.50 | ctrl0 | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl2 | 0.500 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 41.139 | -8.0 | +24.0 | 59.890 |
| mnvme-0.50 | ctrl2 | 0.500 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-0.50 | ctrl4 | 1.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 75.899 | -8.0 | +24.0 | 107.547 |
| mnvme-0.50 | ctrl4 | 1.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mnvme-1.00 | uniform | 0.000 | eager | 296.0 | 48.0 | 32.0 | 344.0 | 12.021 | -8.0 | +24.0 | 15.393 |
| mnvme-1.00 | uniform | 0.000 | graph | refused: Mxfp4NvmeResidency._resolve_src reads the wanted ids to the  | | | | | | | |
| mpin-0.00 | uniform | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 76.156 | +0.0 | +0.0 | 80.374 |
| mpin-0.00 | uniform | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 75.877 | +0.0 | +0.0 | 75.909 |
| mpin-0.50 | uniform | 0.523 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 45.327 | +0.0 | +0.0 | 42.792 |
| mpin-0.50 | uniform | 0.523 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 37.963 | +0.0 | +0.0 | 37.146 |
| mpin-0.50 | ctrl0 | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 10.142 | +0.0 | +0.0 | 18.530 |
| mpin-0.50 | ctrl0 | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 10.037 | +0.0 | +0.0 | 10.048 |
| mpin-0.50 | ctrl2 | 0.500 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 41.548 | +0.0 | +0.0 | 45.333 |
| mpin-0.50 | ctrl2 | 0.500 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 41.270 | +0.0 | +0.0 | 41.420 |
| mpin-0.50 | ctrl4 | 1.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 76.897 | +0.0 | +0.0 | 80.476 |
| mpin-0.50 | ctrl4 | 1.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 75.813 | +0.0 | +0.0 | 76.025 |
| mref | uniform | 0.000 | eager | 304.0 | 24.0 | 8.0 | 328.0 | 12.057 | +0.0 | +0.0 | 15.123 |
| mref | uniform | 0.000 | graph | 0.0 | 2.0 | 0.0 | 306.0 | 11.891 | +0.0 | +0.0 | 11.914 |

### Verdicts (pre-registered)

- **P0 (gates): ALL HOLD** -- nesting 39/39, simulator 14/14, graph check 21/21, graph parity 21/21, budget coverage 39/39, record completeness 39/39 (6 dropped device record(s) in total)
- **P1: HOLDS** -- `{"arms": {"pipe-0.00/eager": {"windows": 1, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.00/graph": {"windows": 1, "fixed": true, "spreads": {"kernels": 0.0}}, "pipe-0.25/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.25/graph": {"windows": 4, "fixed": true, "spreads": {"kernels": 0.0}}, "pipe-0.50/eager": {"windows": 4, "fixed": true, "spreads": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0, "memsets": 0.0}}, "pipe-0.50/graph": {"windows": 4, "fixe`
- **P2: HOLDS** -- `{"arms": {"pipe-0.00": {"delta_per_layer": {"launches": 7.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 7, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 5.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 5, "memcpy": 2, "sync": 0}, "holds": true, "delta_per_token": {"launches": 112.0, "async_copies": 32.0, "syncs": 0.0}}, "pipe-0.25": {"delta_per_layer": {"launches": 10.0, "async_copies": 2.0, "syncs": 0.0}, "want": {"launches": 10, "async_copies": 2, "syncs": 0}, "fetch_per_layer": {"launch": 8.0, "memcpy": 2.0, "sync": 0.0}, "want_fetch": {"launch": 8, "`
- **P3: RECORDED (not the registered box)** -- `{"band_ms": [0.5, 1.5], "families": {"olmoe-1b-7b": {"graph": {"device_ms_per_token": 1.934088499999996, "added_kernels_ms_per_token": 0.5601594999999973, "shared_kernels_ms_per_token": 1.3739289999999982, "shared_by_kernel_us": {"_gemv_nf4_grouped": 1373.9289999999983}, "submissions_per_token": 0.0, "wall_ms_per_token": 1.7402112716808915}, "eager": {"device_ms_per_token": 0.8328634999999994, "added_kernels_ms_per_token": 0.8257509999999908, "shared_kernels_ms_per_token": 0.00711250000000473, "shared_by_kernel_us": {"_gemv_nf4_grouped": 7.112500000004729}, "submissions_per_token": 192.0, "wal`
- **P4: HOLDS** -- `{"arms": {"mnvme-0.00": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.25": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1, "async_copies": 3, "syncs": 3}, "syncs_per_layer": 4.0, "fetch_syncs_per_layer": 3.0, "holds": true}, "mnvme-0.50": {"fixed": true, "delta_per_layer": {"launches": -1.0, "async_copies": 3.0, "syncs": 3.0}, "want": {"launches": -1,`
- **P5: HOLDS** -- `{"arms": {"mpin-0.00": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}, "mpin-0.50": {"fixed": true, "delta_per_layer": {"launches": 0.0, "async_copies": 0.0, "syncs": 0.0}, "want": {"launches": 0, "async_copies": 0, "syncs": 0}, "holds": true}}}`
- **P6: HOLDS** -- `{"arms": {"hyb-0.25": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed", "syncs_per_layer": 10.0, "launches_per_layer": 75.0, "want_syncs": 10}, "ctrl8": {"composition": "cold_only", "syncs_per_layer": 8.0, "launches_per_layer": 50.0, "want_syncs": 8}}, "holds": true}, "hyb-0.50": {"launch_spread_per_token": 544.0, "windows": {"ctrl0": {"composition": "ctrl0", "syncs_per_layer": 4.0, "launches_per_layer": 41.0, "want_syncs": 4}, "ctrl4": {"composition": "mixed",`
- **P7: RECORDED (not the registered box)** -- `{"band": [0.8, 1.25], "refute_above": 1.5, "windows": {"hyb-0.25/uniform": {"path": "hybrid", "measured_us": 56793.374499999874, "hot_d2d_us": 0.0, "predicted_us": 54262.839621790576, "ratio": 1.0466347669205482}, "hyb-0.25/ctrl4": {"path": "hybrid", "measured_us": 37543.6277500001, "hot_d2d_us": 0.0, "predicted_us": 36944.91208292124, "ratio": 1.0162056324761328}, "hyb-0.25/ctrl8": {"path": "hybrid", "measured_us": 75063.24025000002, "hot_d2d_us": 0.0, "predicted_us": 73889.82416584248, "ratio": 1.015880618169071}, "hyb-0.50/uniform": {"path": "hybrid", "measured_us": 35460.82500000005, "hot_`
- **P8 (informational)** -- `{"status": "INFORMATIONAL", "rows": {"olmoe-1b-7b/pipe-0.00/graph": {"cold_fraction": 1.0, "residency_ms": 75.36078727571294, "fixed_tax_ms": 1.7402112716808915, "tax_share": 0.023091734237252613}, "olmoe-1b-7b/pipe-0.00/eager": {"cold_fraction": 1.0, "residency_ms": 70.73091849451885, "fixed_tax_ms": -2.4402065319009125, "tax_share": -0.03449985641130917}, "olmoe-1b-7b/pipe-0.25/graph": {"cold_fraction": 0.75, "residency_ms": 58.49151453003287, "fixed_tax_ms": 1.7402112716808915, "tax_share": 0.02975151670568161}, "olmoe-1b-7b/pipe-0.25/eager": {"cold_fraction": 0.75, "residency_ms": 50.744963984470814, "fixed_tax_ms": -2.4402065319009125, "tax_share": -0.0480876591546636}, "olmoe-1b-7b/pipe-0.50/graph": {"cold_fraction": 0.5, "residency_ms": 37.19721728703007, "fixed_tax_ms": 1.7402112716808915, "tax_share": 0.04678337248328704}, "olmoe-1b-7b/pipe-0.50/eager": {"cold_fraction": 0.5, "r`
- **P9 (informational)** -- `{"status": "INFORMATIONAL", "cross_level": {"M_pipe-0.00": {"level_m_per_step": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 128.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}, "M_pipe-1.00": {"level_m_per_step": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "level_l_per_token": {"cudaLaunchKernel": 176.0, "cudaMemcpyAsync": 32.0, "cudaStreamSynchronize": 0.0}, "agree": true}}, "level_m": {"M_hyb-0.50": {"steps": 12, "coverage": 0.9934673015144562, "kernel_calls_per_step": 1735.0, "device_ms_per_step": 54.84071966666667, "manifest_counts": {"vram": 512, "dram": 0, "nvme": 512}, "active_steps": 12, "per_step": {"cudaLaunchKernel": 1778.6666666666667, "cudaMemcpyAsync": 128.33333333333334, "cudaStreamSynchronize": 72.33333333333333, "cudaDeviceSy`

**Decision rule:** P1 ∧ P2: the pipelined engine's residency cost is a FIXED per-layer count (as registered), independent of cold fraction -> register the measured claim; the cost model gets a per-step fixed term (P3's graph device time, and submissions x the host's unit cost when eager), never a per-cold-expert launch term (grouped-nf4-gemm follow-up on kernel/cold_deadline.py) | P4: the MXFP4 NVMe engine's +3 syncs/layer are structural; a pinned-staging follow-up (two of the three are pageable H2D copies) is licensed only if their host cost >= 10 % of that token | P6: the hybrid tier's cost is layer-COMPOSITION-dependent (hot-only / mixed / cold-only); cold_dest='deadline' prices only bytes, so its GPU side omits the mixed-layer dispatch term | P8: at the RFC's 17.2 % cold the fixed tax is 16% of residency's cost -- partial; stated as measured, no attribution claimed for the RFC
