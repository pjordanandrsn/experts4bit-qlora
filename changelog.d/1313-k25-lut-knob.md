### Added

- **`E4B_K25_LUT`** (#1313): K25's codebook decode on the NF4 store's device-grouped decode route. `tree` is the select
  tree, the default (also when unset), unchanged. `prmt` is K39's byte-permute decode (K25's `lut="prmt"`,
  grouped-nf4-gemm#569), bit-identical to the tree, which lane P156 read at 1.1755x the tree in a microbench (at least
  1.110 under the sampled clock range; `bench/p156/RESULTS-p156.md`) at Qwen3-30B-A3B's W16 shapes.
  - **Opt-in:** no default changes, and no speed claim. Lane P158's served read decides any default.
  - **Refusals, never a silent fallback:** `prmt` is refused when the installed grouped-nf4-gemm lacks K39's decode
    or K25 itself (also under `E4B_NF4_GROUPED_SMALLM=auto`) and on ROCm (an inline-PTX decode); at startup also with
    K25 turned off (`E4B_NF4_GROUPED_SMALLM=0`), where it would select nothing. Any other value is refused.
  - **At startup:** `serve_paged.build_engine` resolves the knob before loading the model
    (`hot_residency.resolve_k25_lut`), so a misconfiguration fails there and not on the first request; `info["k25_lut"]`
    records the resolution.
  - **Engagement record:** `hot_residency.K25_LUT_SEEN` counts K25's calls by decode (`"<lut>|le256"`; eager calls and
    graph captures, not replays), and `/health`'s `prefill_routes()["seen"]["moe_k25_lut"]` carries it.
  - **Bitwise on a GPU:** through the served route's own call, a B=16 step (under the lean glue and with the expanded
    rows), T == 1, and a CUDA-graph replay give the same bits under `prmt` as under the tree, at Granite-3.1-3B-A800M's
    and Qwen3-30B-A3B's expert shapes (`tests/test_k25_lut_gpu.py`).
