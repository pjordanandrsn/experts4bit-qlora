### P143 Amendment 2 (#1313): the served read of the attention split cap, registered

Step 1 (`p143-5090-2`) read GO_FIXED: 16 splits beat the library's 32 at every context on the 5090. e4b#1768 made that
an opt-in knob, `E4B_ATTN_SPLIT_CAP`, over grouped-nf4-gemm 0.47.1's `split_cap`. This amendment registers the read that
decides whether Qwen3-MoE takes it by default:
- a kernel tolerance contract against an fp64 reference (the cap's error at most 1.05x the plan's);
- a kernel timing check at buckets 2 and 4;
- a served ABBA speed read with P115 Phase D's SANE gate at T == 1. The flip bar is g1 >= 1.03, and W16 is a
  no-change check.

Phase D's quality pass builds its KV at 656 tokens, where the library plans 11 splits and a cap of 16 cannot bind. Here
that KV is built at the server's 4096 tokens. The kit is `bench/p143/p143a2_*` and `tests/test_p143a2.py`. Nothing
changes by default.
