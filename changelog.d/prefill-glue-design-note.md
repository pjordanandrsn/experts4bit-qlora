### Design note: a fused path for norm calls above 64 rows in e4b's prefill (bench and tests only)

- `bench/prefill-glue/DESIGN.md` is zero rental and for review. It maps the eager prefill chain onto the fused kernels
  that already exist, and proposes two knobs, both default 0:
  - `E4B_FUSE_PREFILL_GLUE` (P1: norms and rotary, using decode's arithmetic, option B);
  - `E4B_PREFILL_LEAN_DISPATCH` (P2: the bit-identical MoE dispatch, with its route asserted).
- The router epilogue (P3) is out of this round.
- At C = 64, the glue these parts absorb is worth at most about 1.35 ms (P1) and 1.67 ms (P1 + P2) of TPOT. These are
  ceilings: the fused kernels' own time is not subtracted.
- The note lists the CPU checks and the checks on the project's own A2000, correctness only.
- `glue_map.py` assigns every kernel of P119's 512-token prefill profile to its source operation.
  `tests/test_prefill_glue_map.py` keeps the note equal to the script's output and checks that the map closes.
