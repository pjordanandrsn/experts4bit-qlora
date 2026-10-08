### Lane P118 registered (#1313): does the decode lookahead decode the default `serve_paged` server faster at one request, with identical tokens?

- **What.** `bench/p118/`: the pre-registration, an RTX 5090 runner and driver, the box and the reducer, and
  `tests/test_p118_staged_pin.py`. The subject is the default graph server at 16 slots on Qwen3-30B-A3B NF4.
  - **Arms:** four ABBA arms differing only in `E4B_PAGED_DECODE_LOOKAHEAD` (#1339), over P109's W16 and W1 workloads.
  - **Engagement:** counted on the runner's `issue_decode` / `collect_decode`; L1 must overlap at least 90 % of its
    collects.
  - **Mechanism:** a traced pass per workload, after the timed ones, prices the host gap the lookahead removes. L0's
    W1 gap is the rule's premise; the rest is reported.
- **Rule.** VOID → NOISY (W1 self-pairs outside [0.99, 1.01]) → FUNCTION_FAIL (any token or bucket difference between
  arms) → UNTESTED (premise unmet: L0's traced W1 host gap under 0.2 ms, so there is nothing to hide; added in review)
  → SLOWER (g1 < 1.02 or g16 < 0.99) → DEFAULT_ON.
- **Budget.** Proof on Granite-3.1-3b-a800m (guard 0.75 h), reading guard 1.25 h. Lane ceiling $2.50, hard stop $3.00.
- **Not measured yet.** No box runs before this page merges, after #1339 and the P117 reading (#1335).
