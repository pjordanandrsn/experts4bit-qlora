# SC2g results: `sc2g-5090-2` (rendered from `sc2/verdict_sc2g.json` by the reducer's output; re-derived identically)

### Arithmetic per engine (every e4b ratio is ARITH_MISMATCH)

| engine | arithmetic |
|---|---|
| e4b_gptoss | MXFP4 decode: T==1 GEMV on int8 activations (W4A8), K21 <= 256 rows on bf16 (W4A16); NF4 prefill (KEEP_NF4); sinks explicit-mask prefill |
| vllm | Marlin W4A16; TRITON_ATTN |
| sglang | SGLang default MXFP4 runner on sm_120 (recorded); triton attention |
| llamacpp | published GGUF (attention Q8_0); MMVQ W4A8 decode, MMQ W4A4 prefill |

### Serial (one request in flight): mean of the two draws

| engine | status | p50 TTFT (s) | p50 TPOT (ms) | output tok/s |
|---|---|---|---|---|
| e4b_gptoss | VALID | 0.166 | 6.21 | 139.9 |
| vllm | VALID | 0.033 | 3.56 | 267.9 |
| sglang | VALID | 0.024 | 4.06 | 239.2 |
| llamacpp | UNSTABLE | 0.048 | 3.03 | 302.1 |

### Poisson rates: SLO attainment per draw, status, p50 TPOT

| engine | 1 req/s | 2 req/s | 4 req/s | 8 req/s | ceiling |
|---|---|---|---|---|---|
| e4b_gptoss | 1.00/1.00 VALID · 10.3 ms | 0.81/0.88 VALID · 24.4 ms | 0.18/0.18 VALID · 31.1 ms | 0.12/0.10 VALID · 33.0 ms | 1 |
| vllm | 1.00/1.00 VALID · 4.3 ms | 1.00/1.00 VALID · 5.6 ms | 1.00/1.00 VALID · 7.0 ms | 1.00/1.00 VALID · 9.0 ms | 8 |
| sglang | 1.00/1.00 VALID · 4.7 ms | 0.97/1.00 VALID · 6.3 ms | 1.00/1.00 VALID · 7.6 ms | 1.00/1.00 VALID · 9.2 ms | 8 |
| llamacpp | 1.00/1.00 VALID · 5.7 ms | 1.00/1.00 VALID · 22.3 ms | 0.49/0.47 VALID · 22.0 ms | 0.17/0.17 VALID · 23.0 ms | 2 |

### Predictions

| # | verdict | detail |
|---|---|---|
| Q1 | **HOLDS** | `{"ratio": 5.0535, "label": "ARITH_MISMATCH"}` |
| Q2 | **REFUTED** | `{"ratio": 1.7429, "label": "ARITH_MISMATCH"}` |
| Q3 | **HOLDS** | `{"ceilings": {"e4b_gptoss": 1, "vllm": 8, "sglang": 8, "llamacpp": 2}}` |
| Q4 | **HOLDS** | `{"fit": {"decode_ms_per_token": 8.939, "stall_s_per_prefill": 0.2452, "r2": 0.9841, "n": 1008}, "stall_over_token": 27.43}` |
| Q5 | **REFUTED** | `{}` |

### e4b_gptoss against each comparator (ARITH_MISMATCH; no bar beyond Q1-Q3)

| comparator | serial TTFT ratio | serial TPOT ratio | attainment @1 | attainment @2 | attainment @4 | attainment @8 |
|---|---|---|---|---|---|---|
| vllm | 5.053 | 1.743 | 1.000 | 0.846 | 0.183 | 0.108 |
| sglang | 6.972 | 1.528 | 1.000 | 0.857 | 0.183 | 0.108 |
| llamacpp | — | — | 1.000 | 0.846 | 0.379 | 0.619 |
