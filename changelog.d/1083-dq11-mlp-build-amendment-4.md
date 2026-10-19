### DQ11 MLP-adapter BUILD amendment 4: a bar that can judge the candidate, and a null for the knob check (#1083)

`bench/dq11/DQ11-MLP-BUILD-AMENDMENT-4.md` registers the maintainer's ruling on Phase 1's `PHASE1_FAIL`
(`dq11-mlp-phase1-a2000-1`), before any new data:
- **(a) The floor leaves out its candidate.** Every draw containing R2, the fused epilogue's own arithmetic, leaves the
  floor, as does any redraw that mechanically lands on the fused read.
- **(b) An FP64 accuracy judge decides.** It uses dequantized NF4 weights and no intermediate rounding. The fused path
  must be no less accurate than the least accurate neutral unfused ordering, and dx is read on this bar. Where (a)
  disagrees, the outcome names `ACCURACY_OVER_REORDER`.
- **(c) KNOB_OFF gets a null first.** It compares two identical runs of today's path. If they differ, the check moves to
  DQ11's deterministic configuration C; if they differ there too, it uses the null's spread (`KNOB_OFF_BY_SPREAD`).
- **(d)** One independent review before data.

The read stands as registered and the lever stays open. The new read is `dq11-mlp-phase1-a2000-2`, at $0 on the
A2000. No code or default changes.
