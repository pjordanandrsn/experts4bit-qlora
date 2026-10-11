### Q235: the milestone verdict, mechanical (bench/q235/instrument)

- `q235_verdict.py` applies the registered decision rules (as Amendments 2 and 3 amend them) to a box's fetched records and the
  full-depth layer-streamed bf16 reference.
  - **VOID** for a run that is not science-eligible, a refusal (13, 14, 17 or 18, with its named floor), a host fault, a
    time-left guard or a harness error. Amendment 3's instrument checks also void, before any band:
    - the null (the repeat's clean step-0 forward) must be the box, within 1e-3 relative per layer;
    - each in-process plant must engage only at its layer: its earlier layers within 1e-3 of the null's (bit-for-bit equality is
      reported), the planted layer differing;
    - each plant must fail the band.
  - Otherwise **MILESTONE PASS** iff rules 1-7 hold, else **FAIL** naming each failing rule:
    1. completed on the registered class;
    2. finite losses;
    3-5. Amendment 2's STRICT rules and the frozen base, re-derived from `box.json`, fail-closed;
    6. the fresh-process repeat within tolerance;
    7. Amendment 3's band: the contribution error u_i at most 0.775 over the interior class (layers 8-86 of 94).
  - A STRICT failure (exit 12) stops the run before the repeat: a FAIL on rules 3-5, with rules 6 and 7 not read.
  - Reported, not gated:
    - the full u profile;
    - the full-state profile, with its median (the upper middle) and maximum;
    - the router overlap, the loss gap and the device peak against the estimate (above 15 % names an estimator issue);
    - the bytes moved and the step times.
  - Exit codes: PASS 0, FAIL 1, VOID 2.
- `milestone_read.sh` is the post-box job: the reference on the box's own tokens (~2.4 h on one card), then the verdict.
- Tests (CPU, fabricated records):
  - a pass, a failure for each rule, a void for each cause and for each instrument check;
  - the band read on the contribution, not the state;
  - exit 12's FAIL without a repeat;
  - a rehearsal never counting, and the estimator flag.
