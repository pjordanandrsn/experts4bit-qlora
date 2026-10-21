# `p150-5090-1`: D2's quality ruling, the int8 lm_head on held-out windows (GO)

One RTX 5090 (cc 12.0, driver 615.71.09, a 575 W power limit) on an AMD Ryzen 7 9800X3D host (16 CPUs): Vast instance
56693297 on machine 147099, 2026-10-19T07:49:06Z to 08:24:15Z. Cost $0.615.
- **Harness:** e4b `658a0665` (the registration's merge, #1807).
- **Subject:** P125's: e4b 0.50.0 (`4934aba9`) and grouped-nf4-gemm 0.43.0 (`6ee2e104`), with torch 2.8.0+cu128,
  triton 3.4.0 and transformers 5.17.0. Qwen/Qwen3-30B-A3B @ `ad44e777`, on the default server built eager at all-VRAM.

**Verdict: GO.** A ruling on quality only. The self-tests passed and the premise read 39 passed.
- **The held-out windows:** wikitext-2-raw-v1 validation at stride 1024. Both the windows check and each arm read the
  registered sha256s: t1 `5c444982…`, k16 `7ab52a50…`.
- **The round trip:** I8HEAD rounded exactly the lm_head (one module, no attention module) to its per-row int8 grid.

I8HEAD against A:

| gate | windows | argmax | d (nats) | SE | d + 1.645 SE | bound | passes |
|---|---|---|---|---|---|---|---|
| t1 | 196 | 0.98904 | +0.000747 | 0.000291 | 0.001225 | 0.005205 | yes |
| k16 | 123 | 0.99035 | +0.000547 | 0.000354 | 0.001130 | 0.005659 | yes |

The bars compare unrounded values; the table rounds for reading. c4 (16 windows) is reported in `verdict_p150.json`.

The maintainer re-derived the verdict byte for byte from the private receipt store (the reducer at `658a0665`, equal to the
staged pin and the box's copy). The launcher's receipt and ledger row are in that store. `SHA256SUMS` covers every file
here. Five logs carry carriage returns from the box's progress output; they stay only in that store, byte-exact, and
`STORE-ONLY.sha256` lists them.

Re-derive: `PYTHONPATH=bench/p125 python bench/p150/p150_reduce.py --dir bench/p150/receipts/p150-5090-1 --out {out}` writes `bench/p150/receipts/p150-5090-1/verdict_p150.json`
