### DQ2 run 1 read C_UNCOVERED (instrument diagnosed); Amendment 1 registered before run 2 (bench and tests only)

- **`dq2-5090-6`** ($0.034, RTX 5090 on PCIe gen 5 x16, Threadripper PRO 7965WX). The lane READs: engagement and
  integrity hold, and 1/10 self-pairs is out of band. One of 25 streaming draws (copy under the forward, M = 512, not a
  graded row) was not covered end to end, so the rule withholds the stream verdict. No reading is reported.
- **Cause.** At M = 512 the forward is host-launch-bound, and the probe enqueued every forward before any copy.
- **Amendment 1.** The copies wait on the first forward's event, then run while the host issues the rest. The rule is
  byte-identical (pinned), and the predictions are unchanged. Run 2 is the registered single re-run.
- **Before run 1:** five launch attempts on gen 5 hosts died at the launcher's pre-flight (stuck loading, HF-CDN floor)
  or were refused before a rental. That cost $0.128, with every receipt committed. adertha-agents#164 tracks the
  stuck-loading exclusion gap.
