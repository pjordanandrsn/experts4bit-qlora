### `E4B_K25_PLAN` defaults to `c1` on qwen3_moe (P161, FLIP; #1313)

Lane P161 read K25 at plan `c1` in the served step (`bench/p161/RESULTS-p161.md`): on Qwen3-30B-A3B, under K39's decode,
`c1` decodes sixteen requests 1.4654x as fast as today's plan (the min of four ABBA pairs), one request unchanged, every
token identical.
- **The default:** an unset `E4B_K25_PLAN` resolves at engine build, once the model's config is read and before any
  weight loads. On `qwen3_moe` it is `c1` where K25's decode resolved to `prmt` (the decode `c1` was read with);
  everywhere else it is `today`. The resolution and its reason are in `info["k25_plan"]["source"]` (`explicit`,
  `default-allowlisted`, or `default-off: <why>`).
- **Never a refusal by default:** where the decode is not `prmt` (an older kernel package, a ROCm device,
  `E4B_K25_LUT=tree`), the default resolves to `today`, and a forward whose decode is not `prmt` serves `today` rather
  than refusing. An explicit `E4B_K25_PLAN=c1` keeps its refusals; `E4B_K25_PLAN=today` opts out.
- **Scope:** every `qwen3_moe` checkpoint takes the default. A plan changes tiling and staging, never the values, so the
  output is bit-identical to today's plan; the speed was read only at Qwen3-30B-A3B's expert shapes on one RTX 5090,
  and no speed is claimed for other shapes.
- **One engine per process:** the resolved default is process-wide, as with `E4B_K25_LUT`'s.
- **Other families** stay on today's plan until a read of their own.

Files: `experts4bit_qlora/engines/hot_residency.py`, `experts4bit_qlora/serve_paged.py`, `tests/test_k25_plan_default.py`,
`tests/test_k25_plan.py`, `tests/test_k25_lut_default.py`, `docs/SERVING.md`, `docs/STATUS.md`.
