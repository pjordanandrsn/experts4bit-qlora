### Added

- P223 results (`bench/p223/RESULTS-p223.md`): the price of a schedule-invariant decode for Mixtral-8x7B on
  `serve_paged`, read once on one RTX 5090 at 575 W: **READ**. With `E4B_ATTN_SPLIT_CAP=1` and
  `E4B_PAGED_BUCKETS=2,4,8,16`, both the bf16-attention and the int8-attention servers are **schedule-invariant** on the
  full model, where both defaults are not. At sixteen requests the knobs cost 0.09 % and 0.40 % (p16 0.9991, 0.9960);
  at one request they cost about 3.5x and 3.9x (p1 0.2849, 0.2568), below the predicted range, with the cause not
  named by the read. The knobs stay opt-in; receipts in `bench/p223/receipts/p223-5090-1/`.
