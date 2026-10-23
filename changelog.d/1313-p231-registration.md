### P231 registered: `E4B_NF4_SMALLM_BW=1` on Mixtral-8x7B's default NF4 server at 2 and 4 requests (serving roadmap lever B, step B3)

A served registration, no box yet. P227's microbench read the per-row route (K33, one (token, slot) row per program) at
0.356x and 0.461x K25's expert-call time at 2 and 4 tokens on Mixtral's shapes. This lane asks whether that survives the
served decode step, and at what quality cost.

- **Arms:** B0a B1a B1b B0b on the default server (graphs on, 16 x 768), B1 adding only `E4B_NF4_SMALLM_BW=1`;
  decode-only timing; route and K33 dispatch read per decode bucket at capture, failing closed.
- **Workloads:** W2 and W4 ruled (gain bar 1.03); W1 and W16 inert controls (no-regression line 0.99, and B1's tokens
  must equal B0's there); W8 reported only, so 5 to 8 tokens stay opt-in whatever the verdict.
- **Quality:** P115's instrument at group 4 only, behind an on-card premise: B1's expert call returns every
  (token, slot) row bitwise identically at T = 1, 2 and 4 at Mixtral's shapes (`p231_box.py --mode rows`).
- **Host screen:** a download probe before the start refuses a host below 40 MB/s (rc 19, HOST_REFUSED, never a
  strike), so the 93 GB fetch fits its need; three consecutive rc 19 stop the renting, and every box records the
  probe's bytes and MB/s (`download_probe.json`).
- **Budget:** guard 4.8 h; worst USD 6.12 a box, USD 14.56 for the lane (10 ids, at most 2 started).
- **Files:** `bench/p231/` (PREREG, box with a 24-case self-test, reducer with a 44-case self-test, runner, drive,
  staged pins) and `tests/test_p231.py` (15 tests).
