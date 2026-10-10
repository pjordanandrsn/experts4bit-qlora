### Q235: the milestone verdict, mechanical (bench/q235/instrument)

- `q235_verdict.py` applies the registered decision rules to a box's fetched records and the full-depth layer-streamed bf16
  reference.
  - **VOID** for a run that is not science-eligible, a refusal (13, 14, 17 or 18, with its named floor), a host fault, a
    time-left guard or a harness error.
  - Otherwise **MILESTONE PASS** iff rules 1-7 hold, else **FAIL** naming each failing rule:
    1. completed on the registered class;
    2. finite losses;
    3-5. Amendment 2's STRICT rules and the frozen base, re-derived from `box.json`, fail-closed;
    6. the fresh-process repeat within tolerance;
    7. Amendment 1's bands.
  - The device peak against the estimate (above 15 % names an estimator issue), the bytes moved and the step times are reported,
    not gated.
  - Exit codes: PASS 0, FAIL 1, VOID 2.
- `milestone_read.sh` is the post-box job: the reference on the box's own tokens (~2.4 h on one card), then the verdict.
- Tests (CPU, fabricated records): a pass, a failure for each rule, a void for each cause, a rehearsal never counting, and the
  estimator flag.
