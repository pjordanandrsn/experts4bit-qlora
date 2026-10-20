### `E4B_K25_LUT` defaults to `prmt` on qwen3_moe (P158, FLIP; #1313)

Lane P158 read K39's decode in the served step (`bench/p158/RESULTS-p158.md`): on Qwen3-30B-A3B, `prmt` decodes sixteen
requests 1.0925x as fast as the select tree (the min of four ABBA pairs), one request unchanged, every token identical.
- **The default:** an unset `E4B_K25_LUT` resolves at engine build, once the model's config is read and before any
  weight loads. On `qwen3_moe` it is `prmt` where the installed grouped-nf4-gemm carries K39 (0.50.0 or later) and the
  device is NVIDIA; everywhere else it is `tree`. The resolution and its reason are in `info["k25_lut"]["source"]`
  (`explicit`, `default-allowlisted`, or `default-off: <why>`).
- **Never a refusal by default:** an older kernel package or a ROCm device resolves the default to the tree (the same
  values), so no install breaks and the dependency floor does not move. An explicit `E4B_K25_LUT=prmt` keeps its
  refusals; `E4B_K25_LUT=tree` opts out.
- **Scope:** every `qwen3_moe` checkpoint (Qwen3-235B-A22B included) takes the default. The decode is bit-identical
  to the tree, so the values are the same; the speed was read only at Qwen3-30B-A3B's expert shapes.
- **Other families** stay on the tree until a read of their own.
- **Docs:** the measured claim `e4b.serve.p158.k25-prmt.qwen3.5090.2026-10-20` and its STATUS entry.

Files: `experts4bit_qlora/engines/hot_residency.py`, `experts4bit_qlora/serve_paged.py`, `tests/test_k25_lut_default.py`.
