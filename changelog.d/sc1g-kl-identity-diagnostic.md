### SC1g: a failed `sc1g_kl.py` self-test now reports its numbers

When `sc1g_kl.py --self-test` fails in CI, `tests/test_sc1g_checks.py` now re-runs the `identity reads 0` case's own
construction in a fresh interpreter. It prints what that reads: the identity maximum, the NaN count and the gap between
the two rest roundings, with the runner's torch and numpy versions, CPU capability and CPU flags. A guard test pins the
restated construction to the self-test's source. The instrument is unchanged, since A5's rule digest pins it whole. On
2026-10-19 one CI runner failed that case at a value float64 roundoff cannot reach; the SC1g README records it.
