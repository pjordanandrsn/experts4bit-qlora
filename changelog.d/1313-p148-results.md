### P148 results (#1313): both dense-GEMV levers clear the microbench bar

In a CUDA-graph microbench on one RTX 5090, two prototype kernels read faster than cuBLAS at the B=1 step's dense
GEMV shapes. Projected onto P142's in-situ times:
- **D1**, a bf16 GEMV for fused q/k/v and o: 0.131 ms (3.0 % of the step).
- **D2**, an int8 lm_head: 0.190 ms (4.3 %).

Both clear the registered 2 % bar. These are microbench premises, never quoted gains. Nothing changes by default. D2
next owes a served quality ruling at both of the split premise's gates. D1 becomes a grouped-nf4-gemm kernel with its
own tolerance contract and served read. Record: `bench/p148/RESULTS-p148.md`.
