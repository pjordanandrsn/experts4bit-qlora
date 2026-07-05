# Divergence-curve rev1 — FAILED (harness OOM, pre-fix record)

Per Q4: the red is committed before the fix. This is the red.

- Job: `divergence_bf16`, commit 6c4812c-era harness, pod (RTX A5000, on-demand, 2026-07-05,
  25 GB container RAM cap: `memory.max` = 24999997440).
- Failure: `scripts/divergence_curve.py` accumulated a full **fp64** flat weight snapshot
  (60.8M params × 8 B = 486 MB) in a Python list every step. During leg A's (silent) step
  loop this grew ~486 MB/step and hit the 25 GB cgroup cap around step ~43; the process was
  OOM-killed with no `result.json`. The log froze at the last loader line because run_leg
  prints nothing per step (load succeeded; the death was in the step loop, not the load).
- Classification: **harness memory bug in the brand-new probe script** — no divergence curve,
  no first-divergence step, nothing about the 0.0108 gap was measured. The static diff (probe
  step 1, clean) is unaffected and stands.
- Fix (rev2): stream each leg's per-step flat **fp32** vector to a disk memmap on /workspace;
  legs B and C diff against leg A's on-disk vectors online (one vector resident at a time), so
  peak RAM is O(one snapshot) instead of O(steps × legs). fp32 (not fp64) is ample for a
  divergence *curve* and still resolves ~1e-7 onset.
