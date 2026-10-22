### TC1 amendment 74 registered: grouped-nf4-gemm's padded-delta switches at today's pins, with the census

- One rented RTX 5090 at TC1's field recipe with the memory census, e4b only at today's pins (main at launch, grouped-nf4-gemm 0.49.0).
  The arms are e4b's matched arm at its defaults and three arms that each change one grouped-nf4-gemm switch:
  `NF4_QLORA_SINGLE_LADDER=0`, `NF4_QLORA_COMPACT_DELTA=1` and `NF4_QLORA_PAD_BUCKETS=1`. One draw each, worst case $3.20.
- Each arm is bound to the path its own counters record, or it is VOID. The read reports peak memory and the padded LoRA delta's live
  bytes at the peak. An arm that removes at least half of the default's, lowers its peak by at least half of what it removed, and
  holds held-out within 0.005 is a candidate for a default change. A candidate needs its own two-card read with speed first. With none,
  the fix is new grouped-nf4-gemm code.
- `bench/tc1/tc1_run.sh` (`qwen3pos74pd`), `bench/tc1/tc1_reduce.py` (`pos74_why`, `pos74_block`, `padded_site_bytes`, self-test case 129)
  and a test that the padded-delta span names only those functions at the pinned grouped-nf4-gemm.
