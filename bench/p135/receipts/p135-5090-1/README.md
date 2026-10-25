# `p135-5090-1`: the first attempt (HARNESS_ERROR at the premise; strike 1)

One RTX 5090 (cc 12.0, driver 595.71.05) on an Intel Xeon Gold 6240 host (36 CPUs): Vast instance redacted-d6d1165f, launched
2026-10-18T09:54:48Z, torn down 09:58:05Z. Cost $0.038. Code: e4b `7a3e048f` (the registration's merge, #1732),
grouped-nf4-gemm 0.45.0 (`724ccc45`).

**HARNESS_ERROR, rc 25.** The premise stopped the run on the card before any fetch: 34 passed, 1 failed
(`logs/premise.log`). `test_fused_glue_decode_graphs_gpu.py::test_the_folds_replay_exactly_as_their_padded_eager_step`
required `rope_norm_heads` calls and saw 0, because since grouped-nf4-gemm#528 round 2's attention fold runs a q/k-norm
model's norm and rotary as `rope_norm_qk`. Nothing was measured. Amendment 1 (#1737) staged P136's route-exact copy of
the test, and the reading ran as `p135-5090-2`.

The launcher's receipt and ledger row are in the receipt store (`8f5ac4b9`). `SHA256SUMS` covers every file here.
