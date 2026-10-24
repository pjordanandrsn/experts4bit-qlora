### Added

- P231 results (`bench/p231/RESULTS-p231.md`): `E4B_NF4_SMALLM_BW=1` on Mixtral-8x7B's default NF4 server, read once
  on one RTX 5090: **FUNCTION_FAIL**. B1's tokens differ from B0's on the W16 control (both pairs; the box is
  deterministic, and W1 is equal), so the switch stays opt-in until that is explained. Reported, never graded: W2 decodes
  2.096x and W4 1.729x as fast (W8 1.207x), with the controls unchanged. A reading that rows ended early is refuted by
  the records; the prefill ramp is an untested candidate. Receipts in `bench/p231/receipts/p231-5090-3/`.
