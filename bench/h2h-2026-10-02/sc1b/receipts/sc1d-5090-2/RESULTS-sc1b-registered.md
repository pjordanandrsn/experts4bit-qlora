# SC1b read: where each engine's decode step goes (box D)

Verdicts by `bench/sc1b/sc1b_read.py` (amendment A1, merged before box D's data). Terms are per-step medians in ms.

## Predictions

| Q | verdict | detail |
|---|---|---|
| Q1 | UNREAD | why=the gap is unread: ['e4b: NODE_TRACE_INFLATED', 'llamacpp: NODE_TRACE_INFLATED'] |
| Q2 | REFUTED | ratio=1.3964, kernels_in_graph=[1550, 1110.0] |
| Q3 | UNREAD | why=G1 is unread: ['e4b: NODE_TRACE_INFLATED', 'llamacpp: NODE_TRACE_INFLATED'] |
| Q4 | UNREAD | why=G2 is unread: ['llamacpp: NSYS_DIAGNOSTIC_ERRORS'] |
| Q5 | UNREAD | why=the gap is unread: ['e4b: NODE_TRACE_INFLATED'] |

## Arms

| engine | B | status | labels | P | unprofiled | moe_expert | moe_route | attn | dense_gemm | norm_elem | sample | input_prep | memcpy | residual | I_in | idle_out | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | 1 | labelled | PROFILER_INFLATED NODE_TRACE_INFLATED | 4.406 | 4.160 | 1.113 | 0.441 | 0.683 | 1.292 | 0.497 | 0.005 | 0.002 | 0.003 | 0.001 | -0.003 | 0.370 | -0.001 |
| e4b | 16 | ok |  | 9.767 | 9.863 | 4.916 | 0.726 | 0.832 | 1.351 | 1.461 | 0.007 | 0.003 | 0.002 | 0.001 | 0.090 | 0.351 | -0.028 |
| llamacpp | 1 | labelled | NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.298 | 3.273 | 1.075 | 0.203 | 0.557 | 1.791 | 0.465 | 0.000 | 0.000 | 0.195 | 0.000 | -0.060 | 0.482 | 1.409 |
| llamacpp | 16 | labelled | PROFILER_INFLATED NSYS_DIAGNOSTIC_ERRORS | 16.082 | 13.977 | 6.453 | 0.378 | 1.265 | 2.789 | 0.567 | 0.000 | 0.000 | 0.354 | 0.000 | 0.071 | 4.311 | 0.107 |
| vllm | 1 | void |  | -- | 3.546 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| vllm | 16 | void |  | -- | 8.271 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| sglang | 1 | labelled | NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.296 | 3.332 | 0.935 | 0.260 | 0.463 | 1.364 | 0.251 | 0.005 | 0.026 | 0.005 | 0.004 | 0.001 | 0.022 | 0.039 |
| sglang | 16 | labelled | CLOCK_MISMATCH | 7.893 | 8.024 | 4.501 | 0.350 | 0.995 | 1.532 | 0.454 | 0.006 | 0.057 | 0.012 | 0.004 | 0.016 | 0.017 | 0.050 |

## Gaps (e4b minus the comparator; positive = e4b slower)

| gap | comparator | B | status | reading | named | dP | dO | census ratio | SC1 ratio | top three |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | llamacpp | 1 | unread | e4b: NODE_TRACE_INFLATED; llamacpp: NODE_TRACE_INFLATED | | | | | 1.484 | |
| G2 | llamacpp | 16 | unread | llamacpp: NSYS_DIAGNOSTIC_ERRORS | | | | | 0.624 | |
| G3 | vllm | 1 | unread | e4b: NODE_TRACE_INFLATED | | | | | 1.203 | |
| G3 | vllm | 16 | unread | an arm's capture is VOID | | | | | 1.172 | |
| G4 | sglang | 1 | unread | e4b: NODE_TRACE_INFLATED; sglang: NODE_TRACE_INFLATED | | | | | 1.268 | |
| G4 | sglang | 16 | ok | named | norm_elem | +1.874 | -0.077 | 1.229 | 1.191 | norm_elem +1.007; moe_expert +0.415; moe_route +0.376 |
