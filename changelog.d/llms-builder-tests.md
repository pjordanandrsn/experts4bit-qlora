### Tests: the llms bundle builder's own tests run here too

`tests/test_build_llms_bundle.py` is a byte copy of grouped-nf4-gemm's `kernel/test_build_llms_bundle.py`, the tests
for the shared `scripts/build_llms_bundle.py` that 0.56.0 synced. They cover the plain-text-safe link forms, code that
is never rewritten, and the check that the committed `llms-full.txt` holds no unsafe URL.
