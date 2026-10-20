### DQ11 MLP-adapter BUILD amendment 4: a bar that can judge the candidate, and a null for the knob check (#1083)

`bench/dq11/DQ11-MLP-BUILD-AMENDMENT-4.md` registers the maintainer's ruling on Phase 1's `PHASE1_FAIL`
(`dq11-mlp-phase1-a2000-1`), before any new data:
- **(a) The floor leaves out its candidate.** Every draw containing R2, the fused epilogue's own arithmetic, leaves the
  floor, as does any redraw that mechanically lands on the fused read. With no draw left, (a) reads `NO_FLOOR`, which
  decides nothing.
- **(b) An FP64 accuracy judge decides.** It uses dequantized NF4 weights and no intermediate rounding. On relative RMS,
  the fused error must be at most 1.05 × the worst of today's path and the post-(a) draws, with dx read on this bar.
  - **False-reject control:** before the fused path is judged, a leave-one-out calibration runs on the read itself.
    If any neutral path fails as a pseudo-candidate, the read is `INDETERMINATE`, never `FAIL`. For a neutral fused
    path, P(FAIL) ≤ 1/17.
  - **Disagreement with (a):** where (a) disagrees, the outcome names `ACCURACY_OVER_REORDER`.
- **(c) KNOB_OFF gets a null first.** It runs three `step-today` processes:
  - in the read environment;
  - if those disagree, under DQ11's deterministic configuration C;
  - if those disagree too, an aggregate relative RMS over the B update: each process is scored by its mean distance
    to the others, and off fails only if its score exceeds 2× the largest today score (`KNOB_OFF_BY_SPREAD`). Its
    guaranteed false-reject bound is 1/4; the margin's further reduction is a stated assumption.

  It never mixes environments.
- **(d)** One independent review before data.

The read stands as registered and the lever stays open. The new read is `dq11-mlp-phase1-a2000-2`, at $0 on the
A2000. No code or default changes.
