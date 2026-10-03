# SC1b read: where each engine's decode step goes (box D)

Verdicts by `bench/sc1b/sc1b_read.py` (amendment A1, merged before box D's data). Terms are per-step medians in ms.

## Predictions

| Q | verdict | detail |
|---|---|---|
| Q1 | UNREAD | largest=moe_route, contribution_ms=0.244624, I_in_contribution_ms=0.031377, delta_P_ms=1.199314, I_in_band_ms=[-0.004688, 0.15325], why=the largest term is undecided inside the node-trace bands |
| Q2 | REFUTED | ratio=1.3964, kernels_in_graph=[1550, 1110.0] |
| Q3 | UNREAD | delta_moe_expert_ms=0.052542, delta_I_in_ms=0.031377, moe_expert_band_ms=[-0.112516, 0.100415], I_in_band_ms=[-0.004688, 0.15325], why=the bands overlap |
| Q4 | UNREAD | why=idle_out is not nameable (G-inflate failed on an arm) |
| Q5 | UNREAD | why=the gap is unread: ['NODE_TRACE_AMBIGUOUS: the nominal reading (spread, None) changes inside the node-trace bands (omega e4b 0.165 ms, comparator 0.096 ms)'] |
| Q6 | HOLDS | readings=[('named', 'overlap_in')] |
| Q7 | UNREAD | why=a gap is unread, readings=[('named', 'norm_elem'), 'unread'] |
| Q8 | HOLDS | llamacpp_fraction=0.9549, e4b_fraction=0.0 |

## Arms

| engine | B | status | labels | P | unprofiled | moe_expert | moe_route | attn | dense_gemm | norm_elem | sample | input_prep | memcpy | residual | I_in | idle_out | overlap_in | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | 1 | labelled | NODE_TRACE_INFLATED | 4.443 | 4.277 | 1.123 | 0.450 | 0.695 | 1.303 | 0.509 | 0.005 | 0.002 | 0.003 | 0.001 | -0.012 | 0.365 | -0.000 | -0.000 |
| e4b | 16 | ok |  | 9.801 | 9.663 | 4.916 | 0.732 | 0.840 | 1.355 | 1.470 | 0.007 | 0.003 | 0.002 | 0.001 | 0.047 | 0.409 | -0.006 | -0.025 |
| llamacpp | 1 | labelled | CLOCK_MISMATCH | 3.244 | 3.151 | 1.070 | 0.205 | 0.578 | 1.809 | 0.472 | 0.000 | 0.000 | 0.024 | 0.000 | -0.043 | 0.507 | -1.380 | -0.002 |
| llamacpp | 16 | labelled | PROFILER_INFLATED NSYS_GRAPH_ID_MAPPING CLOCK_MISMATCH | 15.408 | 13.431 | 6.221 | 0.382 | 1.280 | 2.803 | 0.571 | 0.000 | 0.000 | 0.890 | 0.000 | 0.129 | 3.841 | -0.179 | 0.531 |
| vllm | 1 | labelled | NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.535 | 3.528 | 1.101 | 0.400 | 0.600 | 1.223 | 0.120 | 0.026 | 0.023 | 0.037 | 0.004 | -0.011 | 0.014 | -0.000 | 0.002 |
| vllm | 16 | labelled | NODE_TRACE_INFLATED | 7.989 | 8.298 | 4.567 | 0.418 | 1.165 | 1.403 | 0.412 | 0.071 | 0.027 | 0.003 | 0.006 | -0.089 | 0.014 | -0.004 | 0.004 |
| sglang | 1 | labelled | PROFILER_INFLATED NODE_TRACE_INFLATED CLOCK_MISMATCH | 3.325 | 3.108 | 0.940 | 0.264 | 0.468 | 1.371 | 0.254 | 0.005 | 0.027 | 0.006 | 0.004 | 0.017 | 0.015 | -0.036 | 0.010 |
| sglang | 16 | labelled | CLOCK_MISMATCH | 7.933 | 7.952 | 4.484 | 0.353 | 0.994 | 1.543 | 0.455 | 0.006 | 0.056 | 0.008 | 0.004 | 0.073 | 0.015 | -0.037 | 0.021 |

## Gaps (e4b minus the comparator; positive = e4b slower)

| gap | comparator | B | status | reading | named | dP | dO | census ratio | SC1 ratio | top three |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | llamacpp | 1 | ok | named | overlap_in | +1.199 | +0.002 | 1.358 | 1.484 | overlap_in +1.380; dense_gemm -0.506; moe_route +0.245 |
| G2 | llamacpp | 16 | ok | spread | -- | -5.607 | -0.556 | 0.719 | 0.624 | dense_gemm -1.447; moe_expert -1.305; norm_elem +0.898 |
| G3 | vllm | 1 | unread | NODE_TRACE_AMBIGUOUS: the nominal reading (spread, None) changes inside the node-trace bands (omega e4b 0.165 ms, comparator 0.096 ms) (nominal: spread, --) | | | | | 1.203 | |
| G3 | vllm | 16 | ok | named | norm_elem | +1.812 | -0.029 | 1.164 | 1.172 | norm_elem +1.058; idle_out +0.394; moe_expert +0.350 |
| G4 | sglang | 1 | ok | spread | -- | +1.118 | -0.011 | 1.376 | 1.268 | norm_elem +0.254; attn +0.227; moe_route +0.186 |
| G4 | sglang | 16 | unread | NODE_TRACE_AMBIGUOUS: the nominal reading (named, norm_elem) changes inside the node-trace bands (omega e4b 0.097 ms, comparator 0.033 ms) (nominal: named, norm_elem) | | | | | 1.191 | |
