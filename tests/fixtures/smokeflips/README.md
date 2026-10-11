# smokeflips: real start-/health records of SC2f's two arms

`summary.json` is byte-identical to the receipt of a record-only smoke run on the project's RTX A2000 (sm_86) on
2026-10-11, correctness only (no timing in it is read). sha256 `1eeb6e02670cb63e1d7c28c3b0e8c2f34116575e09d62a7a41435f267166b0de`;
the tests that read it check that digest first.

Code under the smoke: e4b main at a6c60914, grouped-nf4-gemm main at 52bd6a44, torch 2.8.0+cu128, transformers 5.17.0.

**What it holds.** Two cells, one fresh process each, on `bench/smoke/gpu_serve_smoke.py`'s own fixtures (random local
checkpoint, real arena bake, real `build_engine`; int4 experts and attention, `E4B_INT4_PREFILL=k19`):
- `on`: `E4B_FUSE_PREFILL_GLUE` and `E4B_PREFILL_LEAN_DISPATCH` unset, the release's defaults;
- `off`: both `0`, SC2f's off arm.

The model is the smoke's `qwen3_moe` (2 layers, 4 experts) at top-4 instead of top-2. That way both K19 row counts an
SC2f server's startup captures make are reached at the release's `E4B_INT4_WIDE_TILES=auto`:
- the scheduler's serve path (two requests, a 512-token prompt) routes 2,048 rows a chunk, above `_WIDE_TILES_MAX`
  = 1024, so it takes the chained table;
- one eager 128-token forward with the graph server's grouping routes 512 rows, the count of bucket 64's decode
  capture, so it takes the one-launch table.

Per cell, `health` holds the two `/health` blocks FLIPS reads, built by the `/health` handler's own expressions:
`levers` is the engine's info without `graph_status`, and `prefill_routes` is `prefill_routes()`. The cell also holds
K19_DISPATCH_SEEN's delta per phase.

The harness asserted no route: it records, and the tests judge. Its `status` reads `RECORDED` in both cells.

**What it shows.** Above 256 rows, `on` took `chained|lean` and `one_launch|lean`, and `off` took `chained|gather` and
`one_launch|gather`. `on` resolved the glue default from the family allowlist and ran the three prefill folds; `off`
read an explicit 0 and ran no prefill fold. Calls at or below 256 rows (decode) are `one_launch|lean|le256` on both
arms, because the prefill dispatch knob governs rows above 256 only.

**Who reads it.** `tests/test_sc2f_real_records.py`, through SC2f's FLIPS gate.
