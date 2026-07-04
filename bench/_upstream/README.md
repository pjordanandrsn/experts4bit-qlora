# Upstream benchmarks — require bitsandbytes ≥ 0.50

These A/B benchmarks measure the `bnb.matmul_4bit` forward optimization proposed upstream in
[bitsandbytes#1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965) (the `mem:`
commit). They require **bitsandbytes ≥ 0.50**.

On older releases (≤ 0.49.x) `matmul_4bit` mishandles the `[packed, 1]` weight layout that
`Experts4bit` uses in training shapes and returns wrong results — so these scripts will report a
large divergence / FAIL there. As of v0.2.0 the packaged library sidesteps the issue entirely:
training runs dequantize-then-linear with **recompute-in-backward** on every bitsandbytes release
(no version gate to probe), and the only remaining `matmul_4bit` use is the probe-gated inference
decode GEMV (passes on stock 0.49.x — `docs/METHODOLOGY.md` §12a). The "after" path measured here
is therefore reconstructed locally by each bench (`_matmul4bit_proj`), preserving the exact
commit-era call these numbers pin.

- `bench_matmul4bit.py` — equivalence + latency/peak-memory A/B of dequantize-then-linear vs `matmul_4bit`.
- `bench_energy.py` — joules/op: native bf16 vs dequantize vs `matmul_4bit`.

Their results (measured on the fork, bnb 0.50-dev) are written up in
[`../../docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) §9–§10.
