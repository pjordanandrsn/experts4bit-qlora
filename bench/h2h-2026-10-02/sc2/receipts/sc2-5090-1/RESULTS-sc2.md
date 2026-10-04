# sc2-5090-1: the registered rule's tables (rendered from `sc2/verdict.json`, which `sc2_reduce.py` re-derives identically from the committed run files)

### Serial (Q1: one request in flight)

| engine | status | p50 TTFT (s) | p50 TPOT (ms) | output tok/s |
|---|---|---|---|---|
| e4b_int4 | UNSTABLE | 0.269 | 4.51 | 165.1 |
| vllm | VALID | 0.052 | 3.50 | 264.5 |
| sglang | VALID | 0.034 | 3.27 | 289.1 |
| llamacpp | VALID | 0.095 | 3.06 | 274.2 |
| e4b_nf4 | VALID | 0.275 | 9.58 | 89.5 |

### Poisson rates (Q2): SLO attainment per draw, status, p50 TPOT

| engine | 1 req/s | 2 req/s | 4 req/s | 8 req/s | ceiling |
|---|---|---|---|---|---|
| e4b_int4 | 1.00/0.98 VALID · 8.0 ms | 0.34/0.81 UNSTABLE · 27.0 ms ⚠spread | 0.07/0.10 VALID · 38.7 ms | 0.02/0.03 VALID · 38.7 ms | 1 |
| vllm | 1.00/1.00 VALID · 4.2 ms | 1.00/1.00 VALID · 5.3 ms | 1.00/1.00 VALID · 6.4 ms ⚠spread | 1.00/0.98 VALID · 10.1 ms ⚠spread | 8 |
| sglang | 1.00/1.00 VALID · 3.9 ms | 1.00/1.00 VALID · 5.4 ms | 1.00/1.00 VALID · 6.7 ms ⚠spread | 1.00/1.00 VALID · 10.4 ms | 8 |
| llamacpp | 1.00/1.00 VALID · 12.0 ms ⚠spread | 0.97/1.00 VALID · 28.3 ms | 0.27/0.87 UNSTABLE · 27.0 ms | 0.12/0.10 VALID · 27.6 ms | 2 |
| e4b_nf4 | 1.00/0.95 VALID · 35.7 ms ⚠spread | 0.11/0.35 UNSTABLE · 61.0 ms | 0.05/0.04 VALID · 63.2 ms | 0.02/0.03 VALID · 64.3 ms | 1 |

### Predictions

| # | verdict | detail |
|---|---|---|
| Q1 | **UNREAD** | `{}` |
| Q2 | **UNREAD** | `{}` |
| Q3 | **UNREAD** | `{"missing": ["e4b_int4"]}` |
| Q4 | **REFUTED** | `{"ceilings": {"e4b_int4": 1, "vllm": 8, "sglang": 8, "llamacpp": 2, "e4b_nf4": 1}}` |
| Q5 | **REFUTED** | `{"ceilings": {"e4b_int4": 1, "vllm": 8, "sglang": 8, "llamacpp": 2, "e4b_nf4": 1}}` |
| Q6 | **REFUTED** | `{"statuses": {"e4b_int4": {"serial": "UNSTABLE", "1": "VALID", "2": "UNSTABLE", "4": "VALID", "8": "VALID"}, "vllm": {"serial": "VALID", "1": "VALID", "2": "VALID", "4": "VALID", "8": "VALID"}, "sglang": {"serial": "VALI` |

### e4b_int4 against each comparator (no bar)

| comparator | serial TTFT ratio | serial TPOT ratio | goodput @1 | goodput @2 | goodput @4 | goodput @8 |
|---|---|---|---|---|---|---|
| vllm | — | — | 0.992 | — | 0.087 | 0.025 |
| sglang | — | — | 0.992 | — | 0.087 | 0.025 |
| llamacpp | — | — | 0.992 | — | — | 0.222 |
