### SC2e read (#846): 64 slots with decode buckets up to 64 lift `serve_paged`'s capacity ceiling 4 → 12 req/s on one 5090; `SLOTS_LICENSED(64, auto)` (bench only)

- **The reading** (`sc2e-5090-1`, $1.017; lane total $1.262 over 3 receipts): Qwen3-30B-A3B int4 at `max_seqs` 16
  (control), 32 with `E4B_PAGED_BUCKETS=auto`, 64 on the default list and 64 with `auto`, paired over two draws. Every
  gate passed, and serial output was byte-identical to the control on every arm in both draws.
  - Ceilings: 4 / 8 / 8 / **12** req/s (P3, P4 HOLD).
  - One 64-row decode graph runs in 18.4 ms, half of four chained 16-row replays (36.5 ms; P1b HOLDS at 0.50).
  - Output at 16 req/s: 1,514 / 1,569 tok/s at 64 slots with `auto`, against 1,073 / 1,089 at 16 slots.
  - Serial TTFT and TPOT within 1 % on every arm (P2).
  - P1 was refuted on the fast side: above 16 rows each row costs ~0.18–0.21 ms, not the ~0.35 ms extrapolated.
  - P7 was refuted low: the `auto` arms used 263–320 MiB less than their KV pool growth, so the serve estimate
    over-prices them.
  - **`SLOTS_LICENSED(64, auto)`**, with every wide arm licensable. As ruled on #1320, the default flip is
    `E4B_PAGED_MAX_SEQS=auto` (estimate-sized) on the default bucket list, in a separate PR. `E4B_PAGED_BUCKETS=auto`
    as a default waits on P110's teacher-forced read at buckets 32 and 64.
- **The census** (`sc2e_census.py`): the queue, not the decode step, bound 16 slots (queue wait p50 0.3–4.6 s at
  8–16 req/s against ≤ 56 ms at 64 slots); the prefill step and its 40.5 ms stall did not change with slots. Text
  agreement with the control under load is 0.46–0.74 for 64 slots on ≤ 16-row buckets and 0.01–0.12 for the wide
  buckets (no bar).
- **Post hoc** (`sc2e_capacity_check.py`): `serve_capacity` on each server's own costs reproduces 42 of 48 attainment
  cells within 0.03, every cell within 0.075, and every ceiling.
