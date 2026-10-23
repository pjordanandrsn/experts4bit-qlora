### SC2j Amendment 1 (pre-reading): QUALITY reduced by P117's reducer at box Q's stack constants

- The proof `sc2j-r3` passed every gate but QUALITY, which P117's reducer read VOID on two constants naming P117's own
  registration: its grouped-nf4-gemm commit (v0.42.0; box Q runs v0.50.0 and did not export `GNF4_SHA` to
  `p117_box.py`) and its window counts (the proof smokes 8 Granite windows).
- `bench/sc2/sc2j_quality.py` runs `p117_reduce.reduce` with exactly those two constants set to box Q's, P117's bar
  unchanged, and fails closed; `q_quality` exports `GNF4_SHA`. A test on `sc2j-r3`'s own record shows VOID as recorded and
  AT_PARITY at box Q's constants, with each refusal turning it VOID.
- The staged kit gains `sc2j_quality.py`; the pin is regenerated.
