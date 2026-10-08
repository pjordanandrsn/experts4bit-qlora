### `serve_paged`: the B=1 fused stack is on by default for Qwen3-MoE, Qwen3.5/3.6-MoE and Granite-MoE (behaviour change; lane P115)

- **What changed.** The four B=1 fusion knobs now have a family-scoped default: fused q/k/v (`E4B_PAGED_FUSE_QKV`) and
  the three glue folds (`E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`, `E4B_FUSE_ROUTER_EPI`).
  - An **unset** knob resolves at startup to `auto` on a model whose `config.model_type` has a registered passing read:
    `qwen3_moe`, `qwen3_5_moe` and `granitemoe`. It resolves to `0` everywhere else.
  - `/health` reports each knob's resolution and source (`fusion_modes`, `fusion_sources`). Its `engine.fuse_qkv` is
    now the build's resolution, `null` before the build, rather than the config flag an unset knob sets.
  - Explicit `auto` stays structural. On a family outside the list it logs one warning naming the read the family lacks
    or failed; for gpt-oss that is P115 Phase C's SANE argmax agreement, 0.924 < 0.95 (#1342).
- **Why.**
  - On Qwen3-30B-A3B NF4 the stack decoded one request 1.4289× and 16 requests 1.2298× as fast, within P110's quality
    bar (#1328).
  - Phase C read engagement and SANE on Qwen3.6, and Granite's full quality read (#1342).
  - [Phase D pending: the stack on top of grouped-nf4-gemm 0.43.0's bandwidth GEMV.]
- **Who is affected.** `serve_paged` users serving those three families: decode is faster and the arithmetic differs
  within P110's bar, so greedy text can change. Everyone else: nothing.
- **The way back.** `E4B_PAGED_FUSE_QKV=0 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0`.
- **What each family read:**
  - Qwen3-30B-A3B: speed and quality.
  - Qwen3.6-35B-A3B: engagement and SANE; only the router epilogue engages, and it was token-identical.
  - Granite-3.1-3b-a800m: Phase B's quality read.
  - Every other family: nothing, so it stays off.
- **Tests:**
  - `tests/test_fusions_default.py`: the allowlist; unset resolving per family; explicit values passing through; the
    once-per-family warning; `_apply_fusions` resolving at the build; `0` as the way back.
  - `tests/test_fusion_modes.py`: its default test now pins the per-family resolution. Phase C's `staged-c.sha256` is
    re-pinned with a dated note; its receipts record the files as run.
