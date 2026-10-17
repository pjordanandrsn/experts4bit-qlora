# Fused q/k/v selected by structure: correctness on the A2000 (R-d, 2026-10-17)

A correctness check, with no timing, of fusing the q/k/v projections on the families `serve_paged` serves, with the
module selected by its **structure** instead of the class name `Qwen3MoeAttention` that `engines/qkv_fuse.py` and
`engines/train_qkv_fuse.py` gate on today (`bench/fastpath-portability/`, reading R-d). Nothing in the package changes.

- **Setup:** one RTX A2000 12 GB (sm_86, correctness only), torch 2.11.0, transformers 5.19.0, e4b `main` at `b8bdac77`.
- **Fixtures:** every served family's committed fixture (`tests/fixtures/topology_controls/`), one attention layer
  each, random bf16 weights, T = 1 and 17. Gemma-4 and gpt-oss get a sliding and a full layer.
- **Record:** `rd_qkv_structural.py` writes `rd_out.json` (sha256 `fe836ba5`; harness `ec650cf8`).

## Three selectors

| | selects on | forward |
|---|---|---|
| **A** | children exactly {q, k, v, o, q_norm, k_norm}, unbiased projections, head_dim-wide norms, rotate-half rotary | e4b's own fused forward (`qkv_fuse._fused_forward`), as shipped for Qwen3 |
| **A2** | A, plus q/k/v out_features = heads × head_dim | the same |
| **B** | q/k/v all `nn.Linear` with one `in_features` | the family's own, untouched: one fused GEMM serves q/k/v when that forward calls them on the same input |

## Results

The unfused reference repeats bitwise in every case, so every delta below is the fusion's reordering. "Proj" is the
largest per-projection max|Δ| / max|ref|, measured in `qkv_fuse`'s own frame (bar 2^-7 = 7.8e-3). "Out" is the attention
output's.

| family (layer) | A | A2 | B at T = 1 | B at T = 17 |
|---|---|---|---|---|
| Qwen3-30B-A3B | 1 | 1 | bitwise | proj 7.9e-3, out 8.9e-3: **identical to A**, the shipped path |
| Qwen3-235B-A22B | 1 | 1 | bitwise | proj 5.5e-3 |
| Qwen3.6-35B-A3B (full) | **crashes** (gated q) | declines | bitwise | proj 6.6e-3 |
| OLMoE | declines (full-width q/k norms) | declines | bitwise | proj 8.1e-3 |
| Granite-3.1 | declines | declines | bitwise | proj 2.7e-3 |
| gpt-oss-20b (full / sliding) | declines | declines | proj 5.3e-3 / 5.2e-3 (bias carried) | proj 6.9e-3 / 3.6e-3 |
| Gemma-4 (sliding) | declines | declines | proj 4.5e-3 | proj 7.2e-3, **out 2.8e-2** |
| Gemma-4 (full) | declines | declines | declines (`v_proj` is `None`: K = V) | declines |
| Mixtral | declines | declines | proj 7.0e-3 | proj 5.8e-3 |
| ERNIE-4.5 | declines | declines | bitwise | proj 7.1e-3 |

- **A is unsafe.** Qwen3.6's attention has Qwen3's children and norms but a gated q (`q_proj` out 8192 = 2 × heads ×
  head_dim), so A selects it and the replaced forward fails on the shapes. A2's shape check declines it. Even then, a
  forward-replacing selector covers only Qwen3's shape.
- **B engages wherever q/k/v are all Linear:** 8 families, plus Gemma-4's sliding layers. Every forward called q/k/v on
  the same tensor: one fused GEMM, two cache hits, no fallbacks, and the cache dropped after each forward. B is
  bitwise at T = 1 on six families.
- **At T = 17, Qwen3-30B and OLMoE read just over 2^-7 per projection.** Qwen3-30B's 7.9e-3 is identical to e4b's
  shipped, licensed fusion on the same harness. That is the reorder class at 17 bf16 rows, not a defect of B.
- **Gemma-4's sliding layer amplifies the reorder.** Its projections are inside the frame, but the attention output
  moves 2.8e-2 at T = 17, where the others move 3e-3 to 1.1e-2. Any read that licenses B for Gemma-4 needs its quality bar
  to cover this case.

## B's safety: how the shared projection decides

- **Identity.** A cached result is reused only for the same tensor object *and* the same `_version`. The cache holds a
  reference to the input, so its id can't be reused while cached, and `_version` catches an in-place write between
  the q and k calls.
- **Consumed once.** The cache is dropped as soon as all three projections have taken their slice, so a later forward
  never reads a stale result.
- **Misses fall back.** Only the q call starts a fused GEMM. A k or v call that misses the cache computes its own slice,
  which is the unfused math.

Negative cases (`rd_out.json`, `negative`):

| case | cache reuses | fallbacks | result |
|---|---|---|---|
| cross-attention shape: q on x, k and v on y | 0 | 2 | k and v bitwise equal to the unfused projections |
| in-place write between the q and k calls | 0 | 1 | k bitwise equal to the unfused projection of the written tensor |
| k called again after a full q/k/v round | 0 (the round: 2) | 1 | cache dropped; k bitwise equal to unfused |

**CUDA-graph capture** (`capture`): Qwen3-30B and Granite at T = 1, captured once and replayed on three new inputs
written into the static buffer. Every replay is bitwise equal to the unfused module run eagerly. The Python check runs
only at capture, so the captured call pattern is what replays. That is correct as long as a module's forward calls q/k/v
the same way for every input, which holds for the families above by code reading.

## What it does not say

- **No speed.** The A2000 is correctness only, and fusion's gain is a 5090 read with its own quality bar.
- **No quality.** The weights are random, so this says nothing about a model's tokens.
- **No change to the package.** B here is a harness prototype; making it a library path is a separate change.
