### `serve_paged`: the B=1 fused stack is on by default for Qwen3-MoE (behaviour change; lane P115)

- **What changed.** The four B=1 fusion knobs now have a family-scoped default: fused q/k/v (`E4B_PAGED_FUSE_QKV`) and
  the three glue folds (`E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`, `E4B_FUSE_ROUTER_EPI`).
  - An **unset** knob resolves at startup to `auto` on a model whose `config.model_type` has a SANE read at T == 1 at
    reading size: `qwen3_moe`. It resolves to `0` everywhere else.
  - `/health` reports each knob's resolution and source (`fusion_modes`, `fusion_sources`). Its `engine.fuse_qkv` is
    now the build's resolution, `null` before the build, rather than the config flag an unset knob sets.
  - Explicit `auto` stays structural. On any other family it logs one warning naming the read the family lacks or
    failed: for gpt-oss, P115 Phase C's SANE argmax agreement, 0.924 < 0.95 (#1342); for Qwen3.5/3.6-MoE and
    Granite-MoE, a SANE read at T == 1, which lane FAM takes (#1362).
- **Why.**
  - On Qwen3-30B-A3B NF4, on top of grouped-nf4-gemm 0.43.0's bandwidth GEMV, the stack decodes one request 1.5902×
    and 16 requests 1.1644× as fast. P115 Phase D's SANE read at T == 1 passes: bias +0.00541 nats, argmax agreement
    0.9674 (#1379). Phases A and B read it within P110's quality bar at grouped-nf4-gemm 0.42.0 (#1328).
  - The maintainer's rule (#1366): a family is on by default only with a SANE read at T == 1 at reading size. Phase D's
    proof read Granite at T == 1 with a SANE bias of −0.0139 nats, against −0.0009 at T == 12, so the served
    one-request path moves a family's arithmetic more than the 12-window read showed.
- **Who is affected.** `serve_paged` users serving Qwen3-MoE: decode is faster and the arithmetic differs within
  P110's bar, so greedy text can change. Everyone else: nothing.
- **The way back.** `E4B_PAGED_FUSE_QKV=0 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0`.
- **Tests:**
  - `tests/test_fusions_default.py`: the allowlist; unset resolving per family; explicit values passing through; the
    once-per-family warning; `_apply_fusions` resolving at the build; `0` as the way back.
  - `tests/test_fusion_modes.py`: its default test now pins the per-family resolution. Phase C's `staged-c.sha256` is
    re-pinned with a dated note; its receipts record the files as run.
  - `tests/test_serve_paged.py`: `/health`'s `engine.fuse_qkv` is the build's resolution.
