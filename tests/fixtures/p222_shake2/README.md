# p222_shake2: the bench record and verdict of `p222-shake-2`

These files are byte-identical to the records in the receipt of `p222-shake-2`, in the private receipt store. That run
was P222's second paid shakedown: one L4 (sm_89, a 72 W enforced limit) on 2026-10-21, e4b 035cedc4 (Amendment 1's
merge) plus grouped-nf4-gemm 41ef086e, `--shakedown --layers 2 --reps 10 --grid 6 --select-steps 2`. It cost $0.024,
and teardown was verified. A shakedown is never a reading, and nothing in these files is data. The tests that read
these files check their sha256 first:

| file | sha256 |
|---|---|
| `bench_p222.json` | `434cc9753c9a0b5c7daa1918c7df068f08c8681686d0ee7290008af3c391251d` |
| `verdict_p222.json` | `f93575790e8dc89f047817a02f96219145bde5dc4dd82704fc2a7b8a9b5961ca` |

**What they hold:**
- `bench_p222.json` is `p222_bench.py`'s record: the sweep and selection over 6 plans, then all five arms on 10 reps,
  every rep bitwise identical, with no arm fault. The per-arm `compiled` metadata is empty for T, T2, P and M, the gap
  that Amendment 2 closes.
- `verdict_p222.json` is the reducer's verdict at 035cedc4: **VOID**, because the self-pair T/T2 median was 1.0217,
  outside 0.99-1.01. Amendment 2 makes that band report-only in a shakedown, and `tests/test_p222.py` checks that the
  amended reducer reads this same record as SHAKEDOWN, with the band reported.
