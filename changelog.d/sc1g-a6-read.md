### SC1g A6 read (#846): the NF4-prefilled prompt carries part of e4b's excess KL, not most of it (P1 PARTIAL); conv2's front-loading is not the prompt route (P2 FALSIFIED); P3 unread (bench and tests only)

- **The run.** `sc1g-diag-a6-1` (box J, $0.532) graded e4b within the box by the median of per-position full KL against
  box R's reference.
- **P1 PARTIAL.** With the prompt on MXFP4 (`KEEP_NF4=0`), the median falls to 0.70×, 0.76× and 0.58× of the baseline on
  conv1–conv3. No window reaches 0.5×, and none stays above 0.9×.
- **P2 FALSIFIED.** On conv2, the first-half / second-half median ratio rose from 3.24 to 5.79.
- **P3 UNREAD.** The 1.0 h guard's deadline dropped the four bf16-activation arms. Setup took about 24 minutes, including
  an 8-minute staging of the 4 GB of reference rows.
- **Determinism.** The baseline is bit-identical to its repeat, and to A5's reading on a different host.
- **Descriptive.** With the prompt on MXFP4, conv2's mean KL doubles (0.088 against 0.043) and its NLL rises 0.23, all of
  it in the first quarter of decode, while the later quarters improve. This is a lead on the MXFP4 large-row prefill path,
  not graded here.
- **Receipt.** Committed; the reading re-derives (pinned by a test).
- **Spend.** The lane is at $9.382.
