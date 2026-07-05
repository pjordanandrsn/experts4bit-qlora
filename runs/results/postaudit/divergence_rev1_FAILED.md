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

## rev2 — ALSO FAILED (distinct bug: network-FS memmap page-cache charged to cgroup)

Leg A completed and wrote its full 14.6 GB fp32 memmap (`snaps_A_resident.npy`), but the
process died at leg B's model load. On the RunPod network volume (mfs), the 14.6 GB of
memmap page cache from leg A is charged to the 25 GB container cgroup and was not reclaimed
before leg B's bf16 load needed it → OOM at leg B. Moving the snapshots to disk removed the
Python-object RAM but not the page-cache charge. No curve produced (leg A's raw memmap is not
a comparison). Static diff still stands.

## Fix (rev3): fixed random-subset sketch in RAM — no full vectors, no disk

Store per step only a fixed random **2M-index subset** (same indices across all three legs,
seeded) of the LoRA weights, fp32, in a small RAM list: 2M × 4 B × 60 steps × 3 legs ≈ 1.4 GB
total, no file. A fixed random subset is an unbiased sample of the full weight vector, so the
divergence *onset* (first step any sampled param differs), *growth shape*, and the
A-vs-B / A-vs-C *ratio* — the actual deliverables — are all preserved; only the absolute L2 is
a scaled estimate (reported as such). Sidesteps both the RAM-list OOM (rev1) and the
memmap-page-cache OOM (rev2).
