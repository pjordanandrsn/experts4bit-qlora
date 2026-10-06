### TC1 amendment 53: the RTX A2000 bucket count is committed with its script

- `bench/tc1/bucket_count.py` counts grouped-nf4-gemm's LoRA-delta kernels, launches, syncs and allocations, single block against buckets, at
  Qwen3-30B-A3B's packed-row shape. The outputs under torch 2.8 and 2.11 are in `bench/h2h-2026-10-02/tc1/a2000-bucket-counts/`. It is a
  diagnostic that licenses nothing, and its millisecond fields are not readings.
- A prereg note corrects amendment 53's *Why*: fp32 and bf16 adapters give close kernel counts, not identical ones.
