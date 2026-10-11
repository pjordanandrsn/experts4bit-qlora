### SC2f Amendment 1 (#846): FLIPS reads K19's dispatch above 256 rows, not its table (bench and tests only)

- **Why.** SC2f's FLIPS check required K19's calls above 256 rows in each server's start `/health` to be exactly the
  chained table. Bucket 64's decode capture routes 512 rows, which on Qwen3-30B-A3B take the one-launch table under
  `E4B_INT4_WIDE_TILES=auto` (lane P122's bound). So the check would have stopped every reading server before its paid
  runs. #1625's A2000 serve smoke showed the one-launch table at that size. SC2e's committed s64a start `/health` shows
  bucket 64's 144 extra calls above 256 rows.
- **The change.** `sc2f_reduce.k19_gt256_bad`:
  - every call above 256 rows must be on the arm's dispatch (`lean` on, `gather` off);
  - the prefill capture's `chained` call must be among them;
  - no `split` table on either arm.
  The table of the other calls is not read. Self-test 34 cases (30 before); `bench/sc1/staged.sha256` re-pins
  `sc2f_reduce.py` only.
- **Real records first.** `tests/test_sc2f_real_records.py` runs the gate on the smoke's `prefill_routes`
  (`tests/fixtures/smoke1625/pass2_summary.json`, byte-identical to the receipt, sha256 checked) and on SC2e's six start
  `/health` files (digests checked), then on synthetic departures. Those include the `off` arm, which has no real record
  yet. The rule as registered fails 5 of its 24 tests.
- `tests/test_sc2f_box.py` pins the code's table choice rather than the dispatch call's first token, as #1634 does.
