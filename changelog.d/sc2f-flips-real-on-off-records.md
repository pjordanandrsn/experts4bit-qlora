### SC2f (#846): FLIPS is tested on real start-/health records of both arms (tests only)

- `tests/fixtures/smokeflips/summary.json` is the receipt of a record-only smoke on the project's RTX A2000, byte for
  byte (sha256 checked). It has one fresh process per arm of P130's two knobs, on the smoke's qwen3_moe at top-4, so
  both of an SC2f startup's K19 row counts are reached: 2,048-row chunks take the chained table, and a 512-row forward
  takes the one-launch table. Each cell records the two `/health` blocks FLIPS reads, built by the `/health` handler's
  own expressions.
- `tests/test_sc2f_real_records.py` (28 tests):
  - each arm passes FLIPS on its own real record with nothing added and departs on all seven keys under the other;
  - the rule as first registered refuses both real records;
  - the synthetic departures start from the real records.
  The SC2f PREREG's Amendment 1 records the result; the rule is unchanged.
