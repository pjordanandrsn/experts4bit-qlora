### CUDA smoke for the default MoE paged server

Add a reusable offline correctness gate that builds the real default paged
server for four tiny random MoE families, exercises prefill and decode with
NF4 and int4, and fails loudly on build, generation or supported decode-graph
capture/replay failures. A2000 output explicitly labels eager-default coverage.
