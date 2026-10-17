# DeepSeek-V4.1-Flash-1017, gate A: architecture audit and support matrix (2026-10-17)

An audit from the checkpoint's small metadata only, at $0: no weight was downloaded and nothing was run on a GPU. It checks
each reported spec against a file, and says what e4b and gnf4 support today.

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e10`, public and not gated, 510.3 GB in 48 shards.
- **Files read:** `config.json`, `model.safetensors.index.json`, the README, and `inference/` (`model.py`, `kernel.py`,
  `engram.py`, `convert.py`, `config.json`). Their sha256 are in `UPSTREAM.sha256`.
- **Shapes and dtypes:** all 48 shard headers, fetched by HTTP range requests (10.7 MB, no tensor bytes).
  `ds41_metadata.py` reproduces every count below.
- **Compared against:** e4b `b67fd5ba` (its package is unchanged since `3834e696`), gnf4 `5c3d309`, and transformers
  5.12.1 plus its `main` at `d4804f9d` (2026-10-17).

## The reported specs, checked

| reported | verdict | what the files show |
|---|---|---|
| about 552B backbone | **read** | 551.88B parameters in the text backbone. That count leaves out the Engram tables (196.61B), the vision tower (0.49B) and the MTP / DSpark layers (14.23B). The checkpoint holds 763.21B in all. The routed experts are 543.58B, and 8.49B of them are active per token (6 of 384, 40 layers). |
| "Engram" memory | **read** | Two hashed n-gram tables, at layers 1 and 14: `F8_E4M3` [384,006,168 × 256] and [384,016,682 × 256], with one `E8M0` scale per 32. Together they are 202.76 GB, 40 % of the checkpoint. Each token looks up 24 rows per table ((4 − 1) n-gram sizes × 8 heads, `model.py:344`). Each row is dequantized on lookup (`model.py:296`) and gated into the residual stream (`model.py:328`). The hash ids come from a normalized, compressed copy of the tokenizer vocabulary (`engram.py`), computed on the host. |
| 384 routed experts, top-6 | **read** | `config.json`, and 15,360 = 40 × 384 tensors per projection in the index. There is one shared expert. The router scores with `sqrtsoftplus`, and image-span tokens use a second bias, `bias_vl` (`model.py:794`). |
| native MXFP4 experts | **read** | `w1`/`w3` are `I8` [2304, 2560] (two E2M1 values per byte) with `F8_E8M0` [2304, 160] scales, one per 32 along K. `w2` is `I8` [5120, 1152] with [5120, 72]. The codebook is `convert.py:13`; the low nibble is the even element (`convert.py:32`). |
| compressed sparse attention | **read** | Every layer attends over a 128-token sliding window. A layer with `compress_ratio` > 0 also attends over the top 512 compressed positions, which a two-level indexer picks (candidate blocks, then top-k; `model.py:488`, `:584`). Only the source layers compute: compressed KV at layers [2, 8, 14, 20] and indexers at [2, 8, … 36]. Later layers share them (`model.py:614`). The attention is one 512-wide KV head against 64 query heads, with attention sinks and a grouped low-rank output (8 groups). |
| FP4 KV | **read, as numerics** | Compressed KV latents are rounded to FP4 (E2M1, 16-element blocks, E4M3 scales) and written back *dequantized* into a bf16 cache (`model.py:761`, `kernel.py:184`). The window K is FP8-rounded the same way (`model.py:708`). The reference has no FP4 KV storage format. |
| hyper-connections | **read** | The residual stream is 4 copies. Each block mixes them with 24 weights ((2 + 4) × 4, from F32 `hc_*_fn` [24, 20480]), normalized by 20 Sinkhorn iterations (`model.py:909`). |
| DSpark speculative decoding | **read, in part** | Three MTP layers, which read the main model's hidden states at layers 37–39 (`main_proj`, [5120, 15360]). Each has 128 routed experts, top-3, plus one shared expert, a Markov head (rank 256) and a confidence head. The reference implements the draft forward only: the speculative loop is "out of scope for this repo" (`model.py:129`). |
| *not reported* | **read** | A 32-layer, 1024-wide ViT and an aligner (the model is image-text-to-text). FP8 E4M3 dense weights with `E8M0` scales on 32 × 32 tiles. The repository ships no transformers modeling code, only the reference `inference/` (tilelang kernels, tensor-parallel 8). |

## Support today

| component | e4b | gnf4 |
|---|---|---|
| model type `deepseek_v41` (a module tree to load into) | **missing.** e4b's DeepSeek-V4 path (`arch/deepseek_v4.py`) renames checkpoint keys into transformers' `deepseek_v4` module tree. transformers has no `deepseek_v41`, in 5.12.1 or on `main`. | n/a |
| routed experts, MXFP4 | **supported at the format level.** `formats/mxfp4.dequantize_mxfp4` already reads V4's `I8` blocks and `F8_E8M0` scales (`formats/mxfp4.py:71`), and V4's expert forward (clamped SwiGLU, `swiglu_limit` 10) matches. | **supported by the byte layout**: the codebook and nibble order match `mxfp4_grouped`. It needs a host-side stack per layer (gnf4 wants [E, N, K/2] blocks). |
| dense FP8, 32 × 32 tiles | **partial.** `formats/fp8_blocks.dequantize_fp8_blocks` takes a `block_size`, but `arch/moe_load.py:181` checks scales against V4's 128 × 128 and would refuse these. | n/a |
| router: `sqrtsoftplus`, `bias_vl` | **missing** | n/a |
| attention: window + compressed sparse, cross-layer sources, two-level indexer | **missing.** V4's compressor and indexer come from transformers' V4 tree, in a different layout: V4 nests the indexer in the compressor, has `ape` and `hc_head`, and does not share across layers. | **missing** for the indexed sparse gather. `fp8_paged_attn` covers dense decode at head_dim 512 only. |
| hyper-connections (Sinkhorn) | **missing** for V4.1. V4's come from transformers' V4 tree. | **missing** |
| Engram | **missing** | **missing** |
| vision tower | **missing** | **missing** |
| MTP / DSpark | **missing.** The V4 path drops `mtp.*` by design (`arch/deepseek_v4.py`, `rename_checkpoint_key`). | **missing** |

## Read, and inferred

Everything in the two tables above is read from the files cited. The following is inference, for gate B to test:

- **The Engram tables are a host or disk residency case, not a VRAM one.** They are 202.76 GB, read sparsely: 24 rows × 264 B
  (256 FP8 values + 8 scales) = 6.3 KB per token per table.
- **The experts would stream as V4's do.** The main model's are 288.78 GB, 7.22 GB per layer as stored.
- **The rest of the main model, 10.82 GB** (attention, the shared experts, the embedding and head, vision, the Engram
  projections, the hyper-connections), fits a large card.
- **The MTP / DSpark layers are 7.93 GB more** (7.22 GB of experts, 0.71 GB other). Only speculative decoding reads them, and
  e4b's V4 path drops them, so gate B leaves them out and says so.

## Recommendation

**Run gate B, scoped to memory and H2D.** It costs nothing and the Engram tables make the residency question new: a single box
needs 288.78 GB of experts plus 202.76 GB of tables somewhere (296.00 GB of experts with DSpark's), and gate B says where they can
live. **Do not start a build on gate B's answer alone.** e4b cannot load this model until a `deepseek_v41` module tree exists,
either in transformers or written in e4b. Writing one covers the attention, Engram, the router, the hyper-connections and the
vision tower: a large build, and a decision of its own.
