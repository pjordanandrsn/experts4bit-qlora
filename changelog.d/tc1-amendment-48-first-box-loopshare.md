### Read: TC1 amendment 48's first box reads UNTESTED -- the arm's loop share did not count grouped-nf4-gemm's bucketed calls; fixed

- `tc1-5090-101` ($1.27): every `NF4_QLORA_PAD_BUCKETS=1` arm read VOID on amendment 43's loop rule. Its `lora_loop_share` read 1.000
  because `tc1_arm.py` summed only loop + padded + grouped_mm calls, leaving out grouped-nf4-gemm#490's new `padded_bucketed` counter. The
  real share was 1.5 % (494 of 32,256 calls). P115–P118 UNTESTED; the VOID arms' numbers are not read.
- `tc1_arm.py`: `_lora_loop_share` divides by every `lora_path_*` counter (a self-test case pins this receipt's numbers). Amendment 48 records
  the re-ask, the same box on the fixed arm, before it runs.
