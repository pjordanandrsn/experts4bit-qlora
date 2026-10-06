### CI exercises grouped-nf4-gemm 0.42.0 (CI only; the `[fast]` floor stays `>=0.30.0`)

- The CI install pin moves from v0.41.0's commit (`dc8f94ab…`) to v0.42.0's (`b4f93f1c…`). 0.42.0 makes bucketed LoRA-delta padding
  the default as `auto` (TC1 amendments 47–50), and turns the int4-b32 split-K R term off on every part (K30/K32).
- e4b needs nothing new from 0.42.0, so the `[fast]` floor is unchanged. A user with an older grouped-nf4-gemm keeps the single padded
  block; `NF4_QLORA_PAD_BUCKETS=0` restores it on 0.42.0.
