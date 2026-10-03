# SC1b read: where each engine's decode step goes (box D, sc1d-5090-2 re-reduced under A3: EXPLORATORY)

> **EXPLORATORY, not a registered reading.** This is `sc1d-5090-2` re-reduced from its own nsys exports by main's A3
> reducer and scorer. A3 was written after this data was seen, and Q6–Q8 were derived from it, so they hold here by
> construction. The registered reading of this run is `../RESULTS-sc1b-registered.md`. The confirmatory run is
> `sc1d-5090-3`.

Verdicts by `bench/sc1b/sc1b_read.py` (amendment A1, merged before box D's data). Terms are per-step medians in ms.

## Predictions

| Q | verdict | detail |
|---|---|---|
| Q1 | UNREAD | largest=moe_route, contribution_ms=0.238113, I_in_contribution_ms=0.057145, delta_P_ms=1.10763, I_in_band_ms=[-0.004448, 0.145665], note=idle_out not nameable, why=the largest term is undecided inside the node-trace bands or behind an untrusted idle_out |
| Q2 | REFUTED | ratio=1.3964, kernels_in_graph=[1550, 1110.0] |
| Q3 | UNREAD | delta_moe_expert_ms=0.038513, delta_I_in_ms=0.057145, moe_expert_band_ms=[-0.109704, 0.102658], I_in_band_ms=[-0.004448, 0.145665], why=the bands overlap |
| Q4 | UNREAD | why=idle_out is not nameable (G-inflate failed on an arm) |
| Q5 | UNREAD | why=the gap is unread: ['NODE_TRACE_AMBIGUOUS: the nominal reading (spread, None) changes inside the node-trace bands (omega e4b 0.148 ms, comparator 0.103 ms)'] |
| Q6 | HOLDS | readings=[('named', 'overlap_in')] |
| Q7 | HOLDS | readings=[('named', 'norm_elem'), ('named', 'norm_elem')] |
| Q8 | HOLDS | llamacpp_fraction=0.9558, e4b_fraction=0.0 |

## Arms

| engine | B | status | labels | P | unprofiled | moe_expert | moe_route | attn | dense_gemm | norm_elem | sample | input_prep | memcpy | residual | I_in | idle_out | overlap_in | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | 1 | labelled | PROFILER_INFLATED NODE_TRACE_INFLATED | 4.406 | 4.160 | 1.113 | 0.441 | 0.683 | 1.292 | 0.497 | 0.005 | 0.002 | 0.003 | 0.001 | -0.003 | 0.370 | -0.000 | -0.001 |
| e4b | 16 | ok |  | 9.767 | 9.863 | 4.916 | 0.726 | 0.832 | 1.351 | 1.461 | 0.007 | 0.003 | 0.002 | 0.001 | 0.090 | 0.351 | -0.006 | -0.034 |
| llamacpp | 1 | labelled | NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.298 | 3.273 | 1.075 | 0.203 | 0.557 | 1.791 | 0.465 | 0.000 | 0.000 | 0.195 | 0.000 | -0.060 | 0.482 | -1.352 | 0.058 |
| llamacpp | 16 | labelled | PROFILER_INFLATED NSYS_GRAPH_ID_MAPPING | 16.082 | 13.977 | 6.453 | 0.378 | 1.265 | 2.789 | 0.567 | 0.000 | 0.000 | 0.354 | 0.000 | 0.071 | 4.311 | -0.177 | -0.071 |
| vllm | 1 | labelled | NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.522 | 3.546 | 1.083 | 0.395 | 0.597 | 1.219 | 0.118 | 0.022 | 0.023 | 0.037 | 0.004 | -0.022 | 0.016 | -0.000 | -0.033 |
| vllm | 16 | ok |  | 8.159 | 8.271 | 4.529 | 0.411 | 1.164 | 1.394 | 0.411 | 0.074 | 0.027 | 0.003 | 0.006 | 0.044 | 0.017 | -0.004 | -0.084 |
| sglang | 1 | labelled | NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.296 | 3.332 | 0.935 | 0.260 | 0.463 | 1.364 | 0.251 | 0.005 | 0.026 | 0.005 | 0.004 | 0.001 | 0.022 | -0.037 | 0.002 |
| sglang | 16 | labelled | CLOCK_MISMATCH | 7.893 | 8.024 | 4.501 | 0.350 | 0.995 | 1.532 | 0.454 | 0.006 | 0.057 | 0.012 | 0.004 | 0.016 | 0.017 | -0.038 | 0.012 |

## Gaps (e4b minus the comparator; positive = e4b slower)

| gap | comparator | B | status | reading | named | dP | dO | census ratio | SC1 ratio | top three |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | llamacpp | 1 | ok | named | overlap_in | +1.108 | -0.058 | 1.271 | 1.484 | overlap_in +1.352; dense_gemm -0.498; moe_route +0.238 |
| G2 | llamacpp | 16 | ok | spread | -- | -6.315 | +0.037 | 0.706 | 0.624 | moe_expert -1.537; dense_gemm -1.438; norm_elem +0.894 |
| G3 | vllm | 1 | unread | NODE_TRACE_AMBIGUOUS: the nominal reading (spread, None) changes inside the node-trace bands (omega e4b 0.148 ms, comparator 0.103 ms) (nominal: spread, --) | | | | | 1.203 | |
| G3 | vllm | 16 | ok | named | norm_elem | +1.608 | +0.050 | 1.192 | 1.172 | norm_elem +1.049; moe_expert +0.387; idle_out +0.335 |
| G4 | sglang | 1 | ok | spread | -- | +1.110 | -0.003 | 1.248 | 1.268 | norm_elem +0.246; attn +0.220; moe_route +0.181 |
| G4 | sglang | 16 | ok | named | norm_elem | +1.874 | -0.046 | 1.229 | 1.191 | norm_elem +1.007; moe_expert +0.415; moe_route +0.376 |
