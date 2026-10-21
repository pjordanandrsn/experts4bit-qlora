### P223 registered: the price of a batch-invariant decode for Mixtral-8x7B

`bench/p223/PREREG-p223.md` registers a price read of an opt-in mode before any data. On P149's one-layer cut, two
existing knobs gave schedule-invariant served logits: one attention split at every bucket (`E4B_ATTN_SPLIT_CAP=1`) and
no bucket 1 (`E4B_PAGED_BUCKETS=2,4,8,16`). P223 reads their throughput cost on the full model at W1 and W16, for the
bf16 and the int8 attention servers, against today's bucket set, and checks invariance on the full model. No default,
gate or claim moves. The kit follows in its own PR.
