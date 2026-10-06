### SC1g A7 registered (#846): the router-flip instrument -- per-position decode expert ids on box J, to see whether (a)'s worst positions sit where its routing left (b)'s (bench and tests only)

- **Why.** A6 put conv2's excess with the prompt on MXFP4 in the first quarter of decode. The $0 kernel check (#1260)
  cleared the MXFP4 prompt kernel. The maintainer's explanation is router flips, and testing it needs routing data, not
  more KL.
- **Instrument.** `sc1g_k8.py`'s `route_ids_capture` (`SC1G_ROUTE_IDS_OUT`) wraps gnf4's decode GEMV.
  - Its expert ids are copied device-side, with no new sync.
  - Each scored log-prob row records the call count, which aligns positions.
  - The reducer gates every capture: exactly 48 calls per position, gate_up equal to down, every id written and in range.
- **Box J.** 9 arms: conv2's (a) and (b) captured, plus (b) uncaptured, a perturbation control that must be
  bit-identical. Then (a) and (b) on conv1, conv3, conv4.
- **Rule (conv2).** The share of (a)'s top-1 % KL positions with a flipped decode set against all positions (exact
  hypergeometric p).
  - Mean flip count reads instead when the share saturates (> 0.8).
  - A fragility control on (b)'s own top positions separates SUPPORTS from FRAGILE_POSITIONS.
  - Both thresholds are registered with their α: share +0.20 at hypergeometric p ≤ 0.01; intensity ×1.5 at
    permutation p ≤ 0.01.
  - Outcomes, in precedence order: UNREAD, FRAGILE_POSITIONS, SUPPORTS, CONTRADICTS, INCONCLUSIVE.
  - The top set is ranked by this box's own KL.
  - Descriptive: the same rule on flips in [t − 8, t], since a flip reaches later positions through the KV cache.
- **Scope.** Descriptive: no engine change is licensed. The consequences only pick the next registration (on SUPPORTS, a
  causal routing replay).
- **Code.** `box_j` now runs `i_arms_a7`; A6's continuation box is kept as `box_j_a6c`. The reducer self-test has 55 cases
  (10 new). `tests/test_sc1g_a7.py` drives the arms through the real script and runs the capture on CPU through the
  reducer's gate.
