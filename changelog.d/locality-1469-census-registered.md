### #1469 item 1 registered: the expert-locality census on Qwen3-30B-A3B, a routing trace through NF4 host residency on one rented RTX A2000 (bench and tests only)

- **What it reports**, per MoE layer:
  - the distinct experts per W-token window, for W ∈ {1, 16, 32, 64, 128};
  - the churn of the routed ids from one token to the next;
  - the hit rate of an 8-hot-per-layer residency, from the pipelined engine's own counters.
- **The traces.** Decode: four prompt kinds × 512 greedy steps, through grouped-nf4-gemm's `capture_routing.py`
  reused unchanged at a pinned commit. Prefill: 16 wikitext-2 test windows of 641 tokens.
- **The tools** (`bench/locality-1469/`):
  - `locality_capture.py` (calibrate, then census; 9 self-test cases);
  - `locality_summary.py` (numpy only; 9 cases);
  - a box runner and a controller driver derived from K34's, with `staged.sha256` checked in CI.
- **Where.** One rented RTX A2000. Its sm_86 runs the NF4 host-residency path; the fp8 paged runner does not run there.
  Expert ids are correctness-class data, so no timing is quoted. It is a single run under #846's standing tier: guard
  3.0 h, at most $1.60 with a 70 GB pull.
- **No rule.** It is a census, and it licenses nothing. Its numbers feed #1469 items 2 and 3 (the bytes model and the
  planner) and #1470's speculative-decoding question.
