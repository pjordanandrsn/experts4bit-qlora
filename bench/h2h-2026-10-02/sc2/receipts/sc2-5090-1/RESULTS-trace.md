# sc2-5090-1: e4b's request traces, by workload (post-hoc, descriptive; bench/sc2/sc2_trace.py)

## e4b_int4
```
SC2_TRACE {"workload": "warm", "ttft_p50_s": 0.303, "queue_wait_p50_s": 0.0008, "queue_wait_max_s": 0.001, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.9742}
SC2_TRACE {"workload": "serial_d1", "ttft_p50_s": 0.2891, "queue_wait_p50_s": 0.0007, "queue_wait_max_s": 0.0016, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.7881}
SC2_TRACE {"workload": "r1_d1", "ttft_p50_s": 0.2834, "queue_wait_p50_s": 0.0024, "queue_wait_max_s": 0.3278, "prefills_during_decode_mean": 1.717, "decode_s_mean": 1.4923}
SC2_TRACE {"workload": "r2_d1", "ttft_p50_s": 1.4517, "queue_wait_p50_s": 0.9066, "queue_wait_max_s": 4.0936, "prefills_during_decode_mean": 11.65, "decode_s_mean": 5.121}
SC2_TRACE {"workload": "r4_d1", "ttft_p50_s": 10.9896, "queue_wait_p50_s": 10.6809, "queue_wait_max_s": 22.4471, "prefills_during_decode_mean": 13.992, "decode_s_mean": 6.2831}
SC2_TRACE {"workload": "r8_d1", "ttft_p50_s": 17.1963, "queue_wait_p50_s": 16.8329, "queue_wait_max_s": 33.603, "prefills_during_decode_mean": 13.967, "decode_s_mean": 5.9262}
SC2_TRACE {"workload": "serial_d2", "ttft_p50_s": 0.2444, "queue_wait_p50_s": 0.0007, "queue_wait_max_s": 0.0013, "prefills_during_decode_mean": 0.0, "decode_s_mean": 0.7536}
SC2_TRACE {"workload": "r1_d2", "ttft_p50_s": 0.2723, "queue_wait_p50_s": 0.0026, "queue_wait_max_s": 0.3073, "prefills_during_decode_mean": 1.967, "decode_s_mean": 1.5537}
SC2_TRACE {"workload": "r2_d2", "ttft_p50_s": 0.4777, "queue_wait_p50_s": 0.1396, "queue_wait_max_s": 1.5085, "prefills_during_decode_mean": 7.608, "decode_s_mean": 3.8303}
SC2_TRACE {"workload": "r4_d2", "ttft_p50_s": 5.8437, "queue_wait_p50_s": 5.5914, "queue_wait_max_s": 14.2473, "prefills_during_decode_mean": 13.925, "decode_s_mean": 6.1036}
SC2_TRACE {"workload": "r8_d2", "ttft_p50_s": 17.9863, "queue_wait_p50_s": 17.7455, "queue_wait_max_s": 32.5263, "prefills_during_decode_mean": 13.925, "decode_s_mean": 5.8743}
SC2_TRACE_FIT {"decode_ms_per_token": 6.146, "stall_s_per_prefill": 0.3561, "r2": 0.9845, "n": 1008}
```

## e4b_nf4
```
SC2_TRACE {"workload": "warm", "ttft_p50_s": 0.2653, "queue_wait_p50_s": 0.0009, "queue_wait_max_s": 0.0012, "prefills_during_decode_mean": 0.0, "decode_s_mean": 2.0102}
SC2_TRACE {"workload": "serial_d1", "ttft_p50_s": 0.2836, "queue_wait_p50_s": 0.0007, "queue_wait_max_s": 0.0012, "prefills_during_decode_mean": 0.0, "decode_s_mean": 1.6716}
SC2_TRACE {"workload": "r1_d1", "ttft_p50_s": 0.2954, "queue_wait_p50_s": 0.0165, "queue_wait_max_s": 0.2908, "prefills_during_decode_mean": 4.733, "decode_s_mean": 5.373}
SC2_TRACE {"workload": "r2_d1", "ttft_p50_s": 18.0968, "queue_wait_p50_s": 17.7832, "queue_wait_max_s": 31.2961, "prefills_during_decode_mean": 13.975, "decode_s_mean": 10.5624}
SC2_TRACE {"workload": "r4_d1", "ttft_p50_s": 25.1255, "queue_wait_p50_s": 24.8434, "queue_wait_max_s": 49.7887, "prefills_during_decode_mean": 13.942, "decode_s_mean": 10.1876}
SC2_TRACE {"workload": "r8_d1", "ttft_p50_s": 28.5569, "queue_wait_p50_s": 28.1148, "queue_wait_max_s": 58.2485, "prefills_during_decode_mean": 13.967, "decode_s_mean": 9.351}
SC2_TRACE {"workload": "serial_d2", "ttft_p50_s": 0.2602, "queue_wait_p50_s": 0.0007, "queue_wait_max_s": 0.0011, "prefills_during_decode_mean": 0.0, "decode_s_mean": 1.6045}
SC2_TRACE {"workload": "r1_d2", "ttft_p50_s": 0.2876, "queue_wait_p50_s": 0.0187, "queue_wait_max_s": 1.4004, "prefills_during_decode_mean": 6.792, "decode_s_mean": 6.2142}
SC2_TRACE {"workload": "r2_d2", "ttft_p50_s": 3.9115, "queue_wait_p50_s": 3.5223, "queue_wait_max_s": 20.095, "prefills_during_decode_mean": 13.233, "decode_s_mean": 9.8556}
SC2_TRACE {"workload": "r4_d2", "ttft_p50_s": 21.488, "queue_wait_p50_s": 21.2068, "queue_wait_max_s": 44.5617, "prefills_during_decode_mean": 13.917, "decode_s_mean": 10.333}
SC2_TRACE {"workload": "r8_d2", "ttft_p50_s": 33.004, "queue_wait_p50_s": 32.7194, "queue_wait_max_s": 62.4919, "prefills_during_decode_mean": 13.925, "decode_s_mean": 10.1451}
SC2_TRACE_FIT {"decode_ms_per_token": 19.758, "stall_s_per_prefill": 0.4835, "r2": 0.9708, "n": 1008}
```
