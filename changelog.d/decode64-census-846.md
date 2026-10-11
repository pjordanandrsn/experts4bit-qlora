### The 64-slot decode-step census (#846; bench and tests only, no registration)

- `bench/h2h-2026-10-02/sc2e/decode64/`: where SC2e's served bucket-64 decode step goes, from committed receipts. The
  step closes exactly into host before the replay, the device replay (by P119's classes plus an explicit residual) and
  the host tail.
- The four levers, ranked by what each can still save per step:
  - the exposed host gap, ≤ 2.47 ms;
  - `Int4Linear` above 16 rows after P124, ≤ 1.05 ms;
  - K19 at 512 routed rows, ≤ 0.75 ms;
  - the tile table after P122/P126, ≤ 0.69 ms.
- In `sc2f_basis`'s model the top one moves the 0.95 knee from 12.95/12.55 to 14.19/13.94 req/s.
- It lists what reached the 64-row step after SC2e and what of that is unread.
- `decode64_census.py --check` recomputes it byte for byte, and `tests/test_decode64_census.py` runs that check.
