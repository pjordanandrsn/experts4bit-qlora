### Changed

- P233 Amendment 1 (before any data): every arm of the solver-vs-all-VRAM quality read names `E4B_INT8_HEAD_BW=1` and
  `E4B_INT8_HEAD_ROWS=8` explicitly, since an unset `E4B_INT8_HEAD_ROWS` now resolves to 8 on `qwen3_moe` (e4b#2095).
  Each phase records the resolved head values, and the reducer requires them equal across phases. Nothing else changes.
