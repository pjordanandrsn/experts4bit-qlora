### TC1 amendment 78: the field recipe on a 24 GiB RTX 4090 (the fit reading)

- A new family, `qwen3pos78fit`, with one arm, pb1cl256: e4b's matched arm at the field recipe for 60 steps, with every LoRA delta
  padded per bucket through the compact bucketed node (`NF4_QLORA_PAD_BUCKETS=1`) and the training loss in 256-token chunks
  (`E4B_CHUNKED_LM_LOSS=256`), at grouped-nf4-gemm `efdda943`. No recipe term changes.
- Before the arm, each refusing at cents: the card's `memory.total` must be a 24 GiB card's (23,500 to 24,600 MiB, rc 18); grouped-nf4-gemm's
  `test_compact_delta.py` and `test_compact_buckets.py` must pass on the GPU, from a clone outside the fetched work dir; a probe must find
  the installed module bucketing; and the tokens must equal `tc1-5090-155`'s.
- The rule, over the started boxes in launch order: one OOM whose record proves the arm ran as registered reads NOT_FIT and ends the
  lane; the first two complete draws on different machines read FIT, or VOID if steps 1-2 differ between them; a draw off its
  registration or not read never pairs, and two in a row stop the lane NOT_READ, as does the cap of three without two read draws. The
  least of three headrooms over the draws the verdict used is quoted with it. Held-out is reported beside `tc1-5090-155`'s cd1cl1 and is
  not a bar.
- `bench/tc1/tc1_arm.py`: an e4b OOM row carries `oom_path`, the knobs as the process saw them and grouped-nf4-gemm's and the chunked
  loss's counters up to the OOM, so the reducer can bind it.
- `bench/tc1/tc1_run.sh` (`tc1_pos78_family`, the card band), `bench/tc1/tc1_reduce.py` (`pos78_why`, `pos78_oom_why`, `pos78_draw`,
  `pos78_lane`, `pos78_block`, `--pos78-lane`, self-test case 132), `tests/test_tc1_pos78_family.py`.
- `bench/tc1/pos78_kit.py` and `bench/tc1/pos78_manifest_template.json`: the lane's launch caps. The kit's offer screen and the
  template's `max_download_usd_per_gb` both refuse download above $0.02/GB, so a started box costs at most $3.50 and the lane at most
  $13.94. `tests/test_tc1_pos78_kit.py` refuses an offer at $0.021/GB and checks that the template carries the screen's caps.
