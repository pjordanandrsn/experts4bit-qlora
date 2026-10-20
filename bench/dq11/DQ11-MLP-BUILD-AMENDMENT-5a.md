# DQ11 MLP-adapter BUILD — amendment 5a: what the $0 A2000 shakedown fixed (#1083)

Written after Amendment 5's $0 shakedown (section 6) and before any paid second. The maintainer rules on it. Amendment 5
stays as merged: this file fixes the numbers Amendment 5 left to the shakedown, and it changes no rule.

## The shakedown

- **Run:** `shakedown-20261020T182839Z` (receipt `dq11-mlp-phase2-shakedown-a2000-1`), on the RTX A2000 at the
  rehearsal lock, under a `gpu:a2000` claim (18:28:48Z to 19:05:21Z).
- **Code:** the Phase 2 harness at e4b `6e29f0f9`. Every part ran in its own process:
  - stage-check, 51 s;
  - the (b) part, 336 s;
  - the eight timed arms at 8 updates, 158–241 s each;
  - the reduce.
- **The two drift arms** failed on an import (`torch.optim.optimizer` is not loaded by `import torch`). The driver read
  `drift: not read` and the reduce still ran, as section 5 registers. The fix is e4b `9d7c770b`, tested through real
  optimizer steps. The two drift arms were then rerun alone, into a copy of the shakedown's records
  (`shakedown-20261020T182839Z-drift-20261020T192058Z`, 19:21:07Z to 19:28:01Z). D0 took 197 s and D1 185 s, both
  ran, and the reduce read the drift report.
- **Status:** its records are a shakedown, never Phase 2 data, and no A2000 timing is a result.

## What it fixes

1. **The family list is section 3's, unchanged:** R1(c ∈ {1, 2, 4, 8}), R6(c ∈ {1, 2, 4}), R9(c ∈ {1, 2, 4, 8}),
   R3(c ∈ {1, 2, 4}), R4(c ∈ {1, 2, 4}) and R5.
   - On the outputs and on dh, every level combination of their grid gave distinct bits: k = 12, the grid's size.
   - No family level landed on any tensor at any checked layer.
2. **The draws:** the union of the four registered grids (42 draws) plus a seeded sample of **24** from the rest of the
   full product, with seed 3407. That is 66 draws.
3. **k_T on the A2000, the same at layers 0, 15 and 31:**

   | outputs (gate, up, down) | dh | dx | dA/dB gate, up | dA down | dB down |
   |---|---|---|---|---|---|
   | 12 | 12 | 57 | 55 | 44 | 29 |

   The smallest is 12, so every tensor is judged. This only PREDICTS the 5090's k_T. The paid read counts its own, and
   section 2's rule governs there.
4. **The (b) part's seconds:** 336 s on the A2000. The 5090 is the faster card, so this is its bound (`B_READ_S` in
   `dq11_mlp_phase2_run.py`).
5. **The guard fit**, rerun with that number:
   - the static fit: (300 + 2,216) × 1.25 = 3,145 s ≤ 7,200 s;
   - the box's own fit before its install (every part but the drift arms): 2,395 s.

   The registered guard stays **2 h**, with a worst case of about $2.05.
6. **"Fetched before D0" is best-effort** (Amendment 5 section 5 says the timed records are "fetched off the box, before
   D0 starts").
   - The box writes the timed block's records and a `TIMED_DONE` marker, then waits 150 s, about two of the controller's
     60 s polls, each of which copies the box's records in a partial fetch. Nothing confirms that a fetch landed.
   - A missed fetch matters only if the box is also lost during the drift arms. Otherwise the records stay on the box
     for the final fetch, and a drift arm that fails or meets the guard cannot change the outcome or the box's exit.
   - That residual risk is accepted (the maintainer, 2026-10-20).

## Reported from the shakedown (never Phase 2 data)

- **(b) on the A2000:** every one of the 33 tensors judged and PASS, with fused/band 0.963–1.011 (worst 31:dA_up).
  RANK1_UNRESOLVED on 0:out_down only (e_min 0.0664). The worst margin and the RANK1 tensor are Phase 1's `-2` numbers
  again, on the new families.
- **Engagement:**
  - The (b) part engaged 96 adapters at each layer, with nothing refused, fallback 0 and 0 PEFT MLP LoRA forwards.
  - Every F1 arm engaged 96 adapters, fallback 0, and its witness ran 0 PEFT MLP LoRA forwards (and 256 attention).
  - Every F0 arm ran 192 PEFT MLP LoRA forwards and never imported the engine.
- **Drift on the A2000** (D1 against D0, under C, 8 updates; REPORTED only): ‖θ_D1 − θ_D0‖ / ‖θ_D0‖ for layer 0's
  `up_proj`.

  | after update | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|
  | A | 0 | 0.0025 | 0.0040 | 0.0069 |
  | B | 1.7e-5 | 0.026 | 0.046 | 0.063 |

  The largest per-update loss difference is 0.0012. A is still equal at update 1 because B starts at 0, so dA is 0 at
  the first step. B's relative difference reaches 6% by update 8, above the scale Amendment 5's graded prediction gave
  for update 40 (below 1e-2). That prediction stands as registered; this shakedown number is reported for the
  maintainer.
