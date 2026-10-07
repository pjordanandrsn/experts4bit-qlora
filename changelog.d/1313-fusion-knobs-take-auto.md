### The fusion knobs take `auto` (`E4B_PAGED_FUSE_QKV`, `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`, `E4B_FUSE_ROUTER_EPI`); every default unchanged (#1313)

- **What `auto` does.** It applies a fusion where the module structure and the installed kernels license it, and
  patches nothing, without raising, where they do not: another family, a missing kernel module, or a kernel cut that
  lacks what a matched structure needs (GraniteMoe's scaled residual fold, the norm-less rotary). `1` keeps every
  refusal it had; `0` (also unset or empty) is off. Any other value is now refused at startup rather than read as off.
- **`serve_paged`:**
  - `_fusion_env()` parses the four knobs; `PagedServeConfig` gains `fusion_modes`, and `fuse_qkv` is true for `auto`
    and `1`.
  - `_apply_fusions` passes each fold its mode. Fused q/k/v at `auto` fuses what matches, and the folds run either way.
  - `/health` `levers` gains `fusion_modes` and `fusion_report` (what each fold patched, failed to probe, or skipped,
    and why).
  - A config built without modes, such as a bench harness's `PagedServeConfig(fuse_qkv=...)`, calls everything
    exactly as before.
- **The library:**
  - `glue_fuse.fold_mode()` is the shared parser.
  - `fuse_t1_glue`, `fuse_t1_glue_r2` and `fuse_router_epilogue` take `mode=` and `report=`. Without `mode` they still
    read their environment variable, but through the same parser. For a direct caller (a bench harness or your own
    code), `auto` there now applies the fold where licensed, where any value but `1` used to read as off. A value other
    than `auto`, `0` or `1` now raises when the fold is called.
  - `fuse_qkv` takes `fold_modes=` and `fold_reports=`.
- **Why now.** Lane P115 (#1314) registers the read that would make `auto` `serve_paged`'s default for the registered
  B=1 fused stack; its Phase C reads this code on gpt-oss-20b and Qwen3.6-35B-A3B. No default moves here.
- **Tests:** `tests/test_fusion_modes.py`, 34 cases, CPU, with the glue kernels stood in by torch functions:
  - the parser and `from_env`;
  - `_apply_fusions` under each mode;
  - each fold's `auto` with no matching module, no kernel module, and an older kernel cut (each against `1`'s
    refusal);
  - the `auto` census on tiny Qwen3-MoE, GraniteMoe, gpt-oss and Qwen3.5-MoE. The last two are P115 Phase C's
    predictions at small layer counts: gpt-oss `0 / 2L+1 / [L, 0] / L`, Qwen3.5 the router only.
