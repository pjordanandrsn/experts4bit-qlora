### SC2e registered (#846): `serve_paged` at 16, 32 and 64 slots, with decode buckets that end at `max_seqs` (`E4B_PAGED_BUCKETS=auto`) against the default list, box L (bench and tests only)

- **Why.** At 8 req/s SC2c's ON server holds 16 saturated slots: queue wait p50 0.22 / 1.17 s, TPOT 13 ms against a
  100 ms bound, the bucket-16 decode step 9.3 ms (`bench/sc2/sc2e_basis.py census` on `sc2c-5090-1`). `serve_capacity`
  on SC2c's costs puts 64 slots at 1.00 / 1.00 attainment at 8 req/s against 0.70 / 0.34 today.
- **The box.** `SC1_BOX=L` (`bench/sc2/sc2e_box_l.sh`): four servers a draw, SC2c's ON server at `max_seqs` 16 (control),
  32 with `auto`, 64 with the default list and 64 with `auto`; Qwen3-30B-A3B int4, one RTX 5090, grouped-nf4-gemm
  v0.42.0. Per server: warm, a 64-request burst that must replay the widest bucket, serial (twice on the control's draw
  1), Poisson at 1, 2, 4, 8, 12 and 16 req/s. A tripwire refuses an e4b without `E4B_PAGED_BUCKETS=auto` (#1319, the code under test).
- **The rule** (`bench/sc2/sc2e_reduce.py`, 23 self-test cases): per-server gates ROUTES, SLOTS, ENGAGED, PROMPTS; lane
  gates DETERMINISM and IDENTITY; predictions P1–P8 (P1b: the 64-row graph against four 16-row replays); a licence for
  the first of 64 `auto`, 64 default, 32 `auto` whose ceiling rises above the control's with no regression. A licence
  licenses `E4B_PAGED_MAX_SEQS=auto` sized by the serve estimate, never a bare 64, in a separate PR.
- **Census** (`bench/sc2/sc2e_census.py`): decode steps keyed by bucket and by piece count (a chained step's trace
  `bucket` names only its last piece), busy-loop gaps, mean requests in the server and queue wait per rate.
- **Harness.** `sc1_run.sh` and `sc1_drive.sh` gain box L; `staged.sha256` regenerated; the box tests that enumerate the
  box letters now include L.
