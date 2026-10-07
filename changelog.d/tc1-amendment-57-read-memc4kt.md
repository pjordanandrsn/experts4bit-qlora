### Read: TC1 amendment 57 -- e4b's 1.92 GB training-phase excess on packed rows, attributed (P149-P152 HELD)

- `tc1-5090-114` ($1.08, an EPYC 7713): the census of the training phase. Peaks: e4b fp32 28.14 GB, e4b with the double-quantized absmax
  26.79 GB, Unsloth 24.86 GB, all in a training backward. Row `e4b.train.memory.packed-4k-train-census.5090.2026-10-07`.
- With the double-quantized absmax every static class matches Unsloth's. The +1.91 GB is transient:
  - checkpoint activations kept on the GPU, 0.79 GB (Unsloth's own checkpointing holds none there);
  - fp32 temporaries in e4b's routed-expert combine backward, 1.07 GB;
  - grouped-nf4-gemm's bucketed delta block, 0.90 GB.
- STATUS says so. The next registration targets the checkpoint activations, with the combine backward as e4b's own candidate.
