### TC1 amendment 46's gate checkout is deleted after the gate and never fetched (bench and tests only)

- **What happened.** On `tc1dec-5090-4`, the gate's grouped-nf4-gemm checkout (`gnf4-src`) was pulled into the live run's
  `tc1.partial/`. Its `docs/receipts-ab/receipt.json` failed adertha's reconciler, which refused every launch on the account
  (`REFUSED[97]`, sc1g-r-7).
- **The fix.**
  - `tc1_decoded_gate` records the checkout's HEAD in `decgate.json` (`checkout_head`), then deletes the checkout before any arm, on
    every path.
  - `tc1_drive.sh` adds `--exclude 'gnf4-src'` to `TC1_RSYNC_EXCLUDES`, which both the partial pulls and the final fetch use.
- **Tests.** The gate test's stand-in checkout carries a nested `receipt.json`, and each path asserts that the checkout and every
  `receipt.json` are gone. A new test pins the exclude on both rsyncs.
