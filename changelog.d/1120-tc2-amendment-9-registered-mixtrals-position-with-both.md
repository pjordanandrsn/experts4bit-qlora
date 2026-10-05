### TC2 amendment 9 registered: Mixtral's position with both frameworks on one stack (bench and tests only)

- **Why.** Mixtral-8x7B is e4b's one losing family at default settings (TC2 amendment 8: Unsloth/e4b 0.836). That box ran e4b on the
  field image's torch 2.8 against Unsloth's torch 2.12, on a 285K host where Unsloth's Mixtral step is at its fastest.
- **The box** (token `mixtralsamestack`): TC1 amendment 25's same-stack family on Mixtral, resident, e4b at default settings (the dense
  route), 60 steps, load-gated draws, a 192 GB host floor, off machines 151350, 45511 and 138786. P29 Unsloth/e4b on one stack in
  [0.85, 1.25]; P30 e4b venv-unsloth / venv-e4b in [0.85, 1.02]; P31 the dense route on every fused e4b arm. A stable reading becomes
  Mixtral's position to quote whichever side it favours.
- The reducer reads it with amendment 25's scorer; the route check now takes a predicate per family (`SAMESTACK_ROUTE`), so Mixtral's is
  read from the dense call counts. One new self-test case; TC2 box M's real Mixtral receipts reduce under the new family.
