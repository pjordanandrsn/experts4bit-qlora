# sc2b-5090-1: e4b's request traces per server (post-hoc, descriptive; bench/sc2/sc2_trace.py --plan)

## e4b_off_d1
```
SC2_TRACE {"workload": "warm", "ttft_p50_s": 0.2566, "queue_wait_p50_s": 0.0009, "queue_wait_max_s": 0.001, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.9407}
SC2_TRACE {"workload": "serial", "ttft_p50_s": 0.261, "queue_wait_p50_s": 0.0007, "queue_wait_max_s": 0.0023, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.7835}
SC2_TRACE {"workload": "serial_repeat", "ttft_p50_s": 0.267, "queue_wait_p50_s": 0.0006, "queue_wait_max_s": 0.0037, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.7896}
SC2_TRACE {"workload": "r1", "ttft_p50_s": 0.2308, "queue_wait_p50_s": 0.0024, "queue_wait_max_s": 0.215, "prefills_during_decode_mean": 1.517, "decode_s_mean": 1.3755}
SC2_TRACE {"workload": "r2", "ttft_p50_s": 0.7565, "queue_wait_p50_s": 0.2003, "queue_wait_max_s": 3.2214, "prefills_during_decode_mean": 10.383, "decode_s_mean": 4.3768}
SC2_TRACE {"workload": "r4", "ttft_p50_s": 8.5957, "queue_wait_p50_s": 8.3724, "queue_wait_max_s": 15.3691, "prefills_during_decode_mean": 13.933, "decode_s_mean": 5.373}
SC2_TRACE {"workload": "r8", "ttft_p50_s": 14.8145, "queue_wait_p50_s": 14.4072, "queue_wait_max_s": 30.6244, "prefills_during_decode_mean": 13.967, "decode_s_mean": 5.6319}
SC2_TRACE_FIT {"decode_ms_per_token": 5.681, "stall_s_per_prefill": 0.3265, "r2": 0.9868, "n": 528}
```

## e4b_on_d1
```
SC2_TRACE {"workload": "warm", "ttft_p50_s": 0.162, "queue_wait_p50_s": 0.0006, "queue_wait_max_s": 0.0012, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.9332}
SC2_TRACE {"workload": "serial", "ttft_p50_s": 0.1575, "queue_wait_p50_s": 0.0007, "queue_wait_max_s": 0.001, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.777}
SC2_TRACE {"workload": "r1", "ttft_p50_s": 0.1622, "queue_wait_p50_s": 0.0012, "queue_wait_max_s": 0.1632, "prefills_during_decode_mean": 1.208, "decode_s_mean": 1.1734}
SC2_TRACE {"workload": "r2", "ttft_p50_s": 0.2711, "queue_wait_p50_s": 0.0466, "queue_wait_max_s": 1.5901, "prefills_during_decode_mean": 8.442, "decode_s_mean": 3.3521}
SC2_TRACE {"workload": "r4", "ttft_p50_s": 6.4774, "queue_wait_p50_s": 6.3047, "queue_wait_max_s": 11.2875, "prefills_during_decode_mean": 13.958, "decode_s_mean": 4.8845}
SC2_TRACE {"workload": "r8", "ttft_p50_s": 9.6806, "queue_wait_p50_s": 9.4698, "queue_wait_max_s": 20.3054, "prefills_during_decode_mean": 13.967, "decode_s_mean": 4.3673}
SC2_TRACE_FIT {"decode_ms_per_token": 6.064, "stall_s_per_prefill": 0.2617, "r2": 0.9845, "n": 504}
```

## e4b_on_d2
```
SC2_TRACE {"workload": "warm", "ttft_p50_s": 0.1713, "queue_wait_p50_s": 0.001, "queue_wait_max_s": 0.0013, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.9482}
SC2_TRACE {"workload": "serial", "ttft_p50_s": 0.1691, "queue_wait_p50_s": 0.0008, "queue_wait_max_s": 0.0012, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.7588}
SC2_TRACE {"workload": "r1", "ttft_p50_s": 0.1733, "queue_wait_p50_s": 0.0017, "queue_wait_max_s": 0.2158, "prefills_during_decode_mean": 1.483, "decode_s_mean": 1.264}
SC2_TRACE {"workload": "r2", "ttft_p50_s": 0.1927, "queue_wait_p50_s": 0.0085, "queue_wait_max_s": 0.2231, "prefills_during_decode_mean": 4.933, "decode_s_mean": 2.437}
SC2_TRACE {"workload": "r4", "ttft_p50_s": 1.974, "queue_wait_p50_s": 1.7965, "queue_wait_max_s": 4.4395, "prefills_during_decode_mean": 13.883, "decode_s_mean": 4.8554}
SC2_TRACE {"workload": "r8", "ttft_p50_s": 12.283, "queue_wait_p50_s": 12.1145, "queue_wait_max_s": 22.4427, "prefills_during_decode_mean": 13.925, "decode_s_mean": 4.735}
SC2_TRACE_FIT {"decode_ms_per_token": 6.127, "stall_s_per_prefill": 0.2692, "r2": 0.9904, "n": 504}
```

## e4b_off_d2
```
SC2_TRACE {"workload": "warm", "ttft_p50_s": 0.2779, "queue_wait_p50_s": 0.0012, "queue_wait_max_s": 0.0012, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.9518}
SC2_TRACE {"workload": "serial", "ttft_p50_s": 0.2194, "queue_wait_p50_s": 0.001, "queue_wait_max_s": 0.0023, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.7519}
SC2_TRACE {"workload": "r1", "ttft_p50_s": 0.2247, "queue_wait_p50_s": 0.0025, "queue_wait_max_s": 0.3291, "prefills_during_decode_mean": 1.725, "decode_s_mean": 1.4211}
SC2_TRACE {"workload": "r2", "ttft_p50_s": 0.3414, "queue_wait_p50_s": 0.0725, "queue_wait_max_s": 1.5002, "prefills_during_decode_mean": 6.775, "decode_s_mean": 3.3399}
SC2_TRACE {"workload": "r4", "ttft_p50_s": 3.3286, "queue_wait_p50_s": 3.0925, "queue_wait_max_s": 7.6877, "prefills_during_decode_mean": 13.883, "decode_s_mean": 5.2573}
SC2_TRACE {"workload": "r8", "ttft_p50_s": 16.2462, "queue_wait_p50_s": 16.0325, "queue_wait_max_s": 28.6607, "prefills_during_decode_mean": 13.925, "decode_s_mean": 5.4127}
SC2_TRACE_FIT {"decode_ms_per_token": 6.101, "stall_s_per_prefill": 0.3131, "r2": 0.9879, "n": 504}
```
