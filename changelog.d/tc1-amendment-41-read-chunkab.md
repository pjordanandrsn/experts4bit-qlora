### Read: TC1 amendment 41 — e4b's chunked LM loss costs the shipped arm 4.9 % at the field recipe; it stays opt-in (P91 FALSIFIED)

- `tc1-5090-89` ($2.51, EPYC 7B13, machine 145701, 60-step load-gated draws, every attempt above the gate): `E4B_CHUNKED_LM_LOSS=1`
  against the default in venv-unsloth. Shipped arm **1.049** [1.017, 1.082] (P91 FALSIFIED, registered ≤ 1.01), peak 24.673 → 23.508
  GB; the matched pair was unstable (P90, P92, P93 UNTESTED). By the registered rule the flag stays opt-in (row
  `e4b.train.chunked-lm-loss.default-decision.5090.2026-10-05`); STATUS says so.
