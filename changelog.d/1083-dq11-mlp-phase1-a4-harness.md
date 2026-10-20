### DQ11 MLP-adapter BUILD Phase 1: Amendment 4's read in the harness (#1083)

`bench/dq11/dq11_mlp_phase1.py` and `dq11_mlp_phase1_run.py` implement `DQ11-MLP-BUILD-AMENDMENT-4.md` for the read
`dq11-mlp-phase1-a2000-2`.

**The per-module read, in two passes** (`read_layers_a4`):
- Pass 1 reads today's path, the fused path and a landing check on every layer.
- Then every draw containing R2, or any redraw that lands, leaves the floor (`landed`, `post_a_draws`).
- Pass 2 reads an FP64 reference (`fp64_read`): dequantized NF4 weights, autograd in FP64, no intermediate rounding.
  One streamed pass gives (a)'s floor and each unfused path's error.

**The bars:**
- **(b)** decides on relative RMS within a 5% margin (`bar_b`). It first runs a leave-one-out calibration (`calibrate`);
  if that fails, the read is `INDETERMINATE`, never `FAIL`.
- **The outcome** (`outcome_a4`) names `ACCURACY_OVER_REORDER` and `RANK1_UNRESOLVED` where they apply.

**KNOB_OFF (`knob_off`)**, checked in order:
- three `step-today` processes and `step-off` in the read environment;
- if their null fails, the same four under DQ11's `CONFIGURATION_C` (`observer_policy`), each saving its B update;
- if C's null fails too, a per-process mean-distance score over the 224 B slots (`KNOB_OFF_BY_SPREAD`).

`--part need-c` tells the driver whether the C parts run. `reduce_a4` composes the outcome. The `-1` read still rebuilds
`phase1.json` byte-identically through the original `reduce`.
