# PREREG — the slot controller, stage C2: in-engine swaps and the wall

Registered before any measurement. C1 certified the decision value
(16.0% net on fresh windows, [RESULTS-controller.md](RESULTS-controller.md));
this stage puts the rule INSIDE the engine and asks whether the value
survives contact with real swaps and a real clock.

## The engine increment (this PR)

* `_HybridTier` **controller mode** (`enable_hybrid_tier(...,
  swappable=True)`): the DRAM stacks `d_*` cover EVERY expert (identity
  `g2d`; ~16 GB pageable host at the operating point) and `is_dram` is
  all-true, so a demotion is a pure mask flip and a promotion sources
  its H2D locally from `d_*` — runtime swaps never touch the arena or
  any membership structure. The manifest still defines the hot set;
  `hot_ids`, `g2h`, `is_hot`, `is_vram` flip per swap.
* `swap_expert(promote, demote)`: four segment copies into the retired
  slot (+ biases on bias-carrying models), then the id algebra. Called
  only between steps.
* **The rule** is C1's, under the per-layer geometry the stacks force
  (fixed per-layer slot counts ⇒ same-layer swaps only). Receipt-checked
  on C1's own data before this registration: 17.5% raw / 15.2%
  swap-adjusted vs C1's global 18.4/16.0 — the constraint costs ~1
  point and is disclosed, not hidden.

## Protocol (one box)

1. **B3 (void gate): the swap round-trip self-test** — swap an expert
   in, verify the slot's bytes equal the DRAM source bit-exactly, swap
   back, verify the hot stacks are bit-identical to the pre-swap
   snapshot AND a fixed greedy continuation is unchanged. Must print
   `SWAP-SELFTEST-OK` before any timed arm; anything else voids the box.
2. **A/B/A**: three runs of the same sequential workload — the C1 fresh
   grid's ten windows served back to back in one process (base 14500,
   stride 28400, span 28000; B=16, gen 128, chunk 512, pool 32) — with
   the controller OFF, ON, OFF. Same seed, same prompts. The deployed
   static baseline is the solver's placement over the committed
   design-set masses ([receipts-online](receipts-online/)); the
   controller prior is the same set's touch rates.
3. NVMe tier must stay empty (assert in-driver).

## Bars ([c2_verdict.py](c2_verdict.py), self-tested on certify /
decision-only / refuted synthetic receipts)

* **B1 (decision value, deterministic counters)**: controller-arm
  `uniq_dram_total` ≤ **0.88×** the static arms' (which must agree
  exactly — any mismatch voids). C1's constrained sim says 17.5%; the
  12% bar leaves room for engine-reality drift (swap latency shifting
  epoch boundaries does not change routing, so the counters stay
  deterministic).
* **Wall scoreability (the regime cycle's lesson, applied)**: the
  dram-bucket delta must exceed **3× the static-pair spread**, else the
  wall is UNSCOREABLE and only B1 stands.
* **B2 (the wall, when scoreable)**: dram-bucket total reduction ≥
  **5%** (the ~16% unique-mass reduction priced at the dense marginal
  net of ~0.2 ms/swap pageable H2D and the controller's own python
  time, which is measured and reported).
* Whole-step wall is REPORTED, not scored (attention dominates it).

**C2-CERTIFIED** = B1 ∧ B2. **C2-DECISION-ONLY** = B1 ∧ wall
unscoreable — the engine mechanism is proven and the wall claim waits
for quieter hardware. **REFUTED** = B1 fails (the decision value did
not survive the engine) or B2 scoreable-and-failed. Hard stop: one
scored A/B/A; a REFUTED C2 closes the controller line at the engine
boundary with the C1 artifact intact.

## Cost

≈ $1.00 (three ~15-min serving arms + setup on the reference host
class).
