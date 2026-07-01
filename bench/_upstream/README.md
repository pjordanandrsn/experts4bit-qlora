# Upstream benchmarks — require bitsandbytes ≥ 0.50

These A/B benchmarks measure the `bnb.matmul_4bit` forward optimization proposed upstream in
[bitsandbytes#1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965) (the `mem:`
commit). They require **bitsandbytes ≥ 0.50**.

On older releases (≤ 0.49.x) `matmul_4bit` mishandles the `[packed, 1]` weight layout that
`Experts4bit` uses and returns wrong results — so these scripts will report a large divergence /
FAIL there. That is exactly why the packaged `Experts4bit` **auto-detects** support at runtime
(`experts4bit_qlora/_vendor/experts.py::_matmul_4bit_matches_dequant`) and uses the portable
dequantize forward on older bnb.

- `bench_matmul4bit.py` — equivalence + latency/peak-memory A/B of dequantize-then-linear vs `matmul_4bit`.
- `bench_energy.py` — joules/op: native bf16 vs dequantize vs `matmul_4bit`.

Their results (measured on the fork, bnb 0.50-dev) are written up in
[`../../docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) §9–§10.
