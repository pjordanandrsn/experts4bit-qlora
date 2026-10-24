### TC1 amendment 78: the field recipe on a 24 GiB RTX 4090 (the fit reading)

- A new family, `qwen3pos78fit`, with one arm, pb1cl256: e4b's matched arm at the field recipe for 60 steps, with every LoRA delta
  padded per bucket through the compact bucketed node (`NF4_QLORA_PAD_BUCKETS=1`) and the training loss in 256-token chunks
  (`E4B_CHUNKED_LM_LOSS=256`), at grouped-nf4-gemm `efdda943`. No recipe term changes.
- Before the arm, each refusing at cents: the card's `memory.total` must be a 24 GiB card's (23,500 to 24,600 MiB, rc 18); grouped-nf4-gemm's
  `test_compact_delta.py` and `test_compact_buckets.py` must pass on the GPU, from a clone outside the fetched work dir; a probe must find
  the installed module bucketing; and the tokens must equal `tc1-5090-155`'s.
- The rule: two boxes on two different machines; FIT iff both train all 60 steps with no out-of-memory error, NOT_FIT on any, VOID on a
  draw off its registration, a shared machine, or steps 1-2 that differ across the draws. The least of three headrooms over both draws
  is quoted with the verdict. Held-out is reported beside `tc1-5090-155`'s cd1cl1 and is not a bar.
- `bench/tc1/tc1_run.sh` (`tc1_pos78_family`, the card band), `bench/tc1/tc1_reduce.py` (`pos78_why`, `pos78_draw`, `pos78_pair`,
  `pos78_block`, `--pos78-pair`, self-test case 132), `tests/test_tc1_pos78_family.py`.
