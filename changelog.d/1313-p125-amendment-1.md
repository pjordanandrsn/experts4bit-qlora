### P125 Amendment 1 (#1313): launch with the int4-head fix; the reading's guard to 3.0 h

- **What happened.** The first proof, `p125-prove-1`, found that the calibrated int4 lm_head could not build the default
  graph server (fixed in #1426). It also measured the attention calibration at about 49 s a build on Granite.
- **What changes.**
  - The launch commit must carry the fix; the box's tripwire refuses an e4b without it.
  - The reading's guard rises to 3.0 h, from the projected 3–5 min a build on Qwen3.
- **Unchanged.** Every gate, bound, window count, prediction and arm.
- **Granite's proof gates.** Calibrated < RTN ≪ rolled scales, all failing, is reported as instrument behaviour, not
  evidence about Qwen3.
