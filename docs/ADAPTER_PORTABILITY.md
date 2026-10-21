# Adapter portability: what a trained adapter means on each serving path

A LoRA adapter corrects one function: the base it was trained against, `Q_train(W)`, plus `s·B·A`. Serving it on any
other base representation `Q_serve(W)` changes what it corrects. The difference `Q_train(W) − Q_serve(W)` is generally
dense (full rank), so it is never free and never fits in the adapter's rank. Every train-to-serve combination below is
therefore labelled one of three ways:

- **exact by construction.** The served function is the trained one, up to the stated rounding.
- **bounded-loss (gated).** The served function differs in a known way. It may only be served behind a measured,
  registered gate, and never under an "exact" claim.
- **refused.** No path exists, or nothing can make it correct.

This is a matrix read from code. It measures nothing. Bounded-loss cells carry no number until their registered read
(the per-format retention premise and the cross-format read) produces one. For measured train/query storage-mode
transfer on the reference path, see [`MODE_DECOUPLED_ADAPTERS.md`](MODE_DECOUPLED_ADAPTERS.md). This document covers
the fast paged server (`serve_paged`), its arena, its levers and its fusions.

Code references are at experts4bit-qlora `4f81f055`, loggetta `90a7697` and grouped-nf4-gemm `c621596`.

## 1. What training produces

| fact | where |
|---|---|
| `QLoRASetup` defaults: `quant_type="nf4"`, `blocksize=64`, `r=8`, `alpha=16`, `adapter_dtype="bf16"`, `train_experts=True`, `train_attention=True`, `attn_4bit=False`. There is no rsLoRA or other scaling field. | `recipe.py:35-63` |
| **Scale** is `alpha / r` for experts and attention alike, applied to `F.linear(F.linear(x, A), B)` in the adapter dtype. | `lora.py:451`, `lora.py:778`, `lora.py:787-798` |
| **Expert base:** each expert is quantized with `F.quantize_4bit(..., blocksize, compress_statistics=False)`, giving an fp32 absmax, and `gate_up` is `cat(gate, up)`. | `_vendor/experts.py:412-417`, `loader.py:1806` |
| **Expert absmax is double-quantized by default** (`E4B_ABSMAX_DQ` unset means on) for resident `grouped_nf4` training at blocksize 64. The fp32 absmax is replaced by an 8-bit nested form, which is lossy against fp32. Expert offload and `TRAIN_ARENA` keep fp32. | `engines/fast.py:1008-1024`, `recipe.py:254-265`, `absmax_dq.py:288` |
| **Expert delta placement:** gate_up's delta is added **before** the SwiGLU epilogue; down's delta is added to the down projection's output. | `lora.py:676-700` |
| **`attn_4bit`:** `bnb.nn.Linear4bit(..., quant_type="nf4")` with bitsandbytes' defaults for blocksize and `compress_statistics` (blocksize 64, nested absmax), quantized on transfer to the GPU. Biased projections are refused. The repository states no determinism guarantee for re-quantizing the same weights. | `lora.py:953-998` |
| **Targets:** experts `gate_up_lora_{A,B}`, `down_lora_{A,B}` (`[E,r,·]`/`[E,·,r]`); attention `{q,k,v,o}_proj.lora_{A,B}` (`[r,in]`/`[out,r]`). The router, `lm_head` and the norms take no LoRA. A full-weight router run is not saved with the adapter. | `lora.py:459-462`, `lora.py:780-781`, `train.py:102-105` |
| **gpt-oss** experts are built bare by e4b's loader (biased, clamped, interleaved MXFP4), so e4b's trainer gives them no expert LoRA. Expert LoRA for that family is grouped-nf4-gemm's `mxfp4_qlora.ExpertsMxfp4LoRA`. | `loader.py:1295-1299`, `loader.py:1729-1735`, `lora.py:220-222` |

**Loggetta's export (`backend-adapter/1`)** writes `adapter.safetensors` and `adapter_manifest.json`. The manifest
carries `resolved_model_revision`, the full `setup` (so `alpha`, `r` and `attn_4bit`), per-file sha256 and per-tensor
shape and dtype (`loggetta/backends/experts4bit_adapters.py:69-80`). `load_adapter` rebuilds the training model at
that revision (`:139-140`).

The manifest does **not** record:
- whether the expert absmax was double-quantized;
- `E4B_TRAIN_FUSE_QKV`;
- the bitsandbytes version.

That is everything needed to rebuild `Q_train(W)` bit for bit, and §4 F1-F2 turn it into requirements.

## 2. What the paged server serves

`build_engine` (`serve_paged.py:1589-1764`), in order:
1. load (`:1643`, `load_moe_4bit_streaming(..., quant_type="nf4", arena=...)`);
2. placement;
3. the levers (`:1667`);
4. the fusions (`:1672`);
5. the dense GEMV route (`:1680`);
6. the int8 `lm_head` (`:1685`);
7. the KV, the runner and graph capture.

| fact | where |
|---|---|
| **No adapter support today.** The module docstring says "Not in v1: … adapters", and the config and requests have no adapter field. | `serve_paged.py:120`, `:605-650`, `:1846-1869` |
| **Attention is bf16** unless a lever is set. `E4B_SERVE_ATTN_INT4` (int4-b32, packed from the bf16 weight), `E4B_SERVE_ATTN_INT8` (per-row int8, no bf16 copy kept) and the `*_CALIB` levers (GPTQ-style, calibration-dependent) swap plain `nn.Linear` children of `*Attention` modules. | `serve_paged.py:1290-1357`, `engines/int4_attn.py:391-400`, `engines/int8_attn.py:5-7` |
| **Fusions read `.weight`:** `fuse_qkv` (Qwen3MoeAttention only) copies q/k/v into one `qkv_proj`; the dense GEMV route shares `.weight`. | `engines/qkv_fuse.py:105`, `:140-158`, `engines/dense_gemv_bw.py:46` |
| **The NF4 expert arena** is baked with `F.quantize_4bit(w, blocksize=64, quant_type="nf4")` over `cat[gate; up]`, with any nested absmax **dequantized to fp32** (or stored bf16 where that is lossless). Its index records `snapshot` (a path) and `quantizer` (kind, quant_type, blocksize, bnb and torch versions, absmax dtype). It records **no model revision**, and e4b never compares it with the served revision. | grouped-nf4-gemm `kernel/nvme_bake_nf4.py:128-146`, `:606-628`; `loader.py:1344-1346` |
| **int4 expert stores** (`E4B_SERVE_EXP_INT4`) are repacked from the checkpoint at load, all-VRAM only. gpt-oss keeps its native MXFP4 blocks. | `serve_paged.py:1309-1333`, `engines/int4_experts.py:456-623` |
| **The int8 `lm_head`** (`E4B_INT8_HEAD_BW`, default on for `qwen3_moe`) rounds the head in place. That changes the served base's head whether or not an adapter is present, but adapters never touch `lm_head`. | `serve_paged.py:212-214`, `engines/int8_head_bw.py:13-21` |

## 3. The matrix

### Attention (q/k/v/o)

**These labels are per module.** Every attention adapter was trained inside a whole model whose experts were
`Q_train(experts)`. The served model as a whole computes the trained function only when the expert base matches too.
Under the default training run (double-quantized expert absmax) and today's fp32-absmax arena, it does not (F1). So an
attention-only adapter that is exact at its own projections still runs inside a **bounded-loss** whole model until
one of these holds:
- the expert recipe matches, by a training preset or an arena variant;
- a registered read prices the difference.

A `backend-adapter/1` manifest does not record the absmax form, so a v1 adapter's expert base is **unverified**,
not known to match.

| trained on | served on | label | why |
|---|---|---|---|
| bf16 W (`attn_4bit=false`) | bf16, merged at load before the levers and fusions | **exact by construction**, numerically not bitwise | The same function, with two numerical differences. (1) Merging rounds each merged weight `W + s·B·A` (computed in fp32) once to bf16; that error is reported per projection, never assumed. (2) Training ran the delta UNMERGED, `base(x) + s·B(A(x))` in the adapter dtype (`lora.py:787-788`), with its own roundings, which the merge removes. So outputs are not bitwise equal to training's even at merge error 0. `fuse_qkv` and the GEMV route then copy or share the merged weight exactly. |
| NF4 (`attn_4bit=true`) | the dequantized `Q_train(W)` rebuilt at load, plus `s·B·A`, in bf16 | **exact by construction, conditional**: only if the rebuild matches training's `Q_train(W)` by fingerprint; **refused** otherwise | The correction was learned against `Q_train(W)`. A rebuild needs bitsandbytes' recipe (blocksize 64, nested absmax) and a reproducible quantize. Neither is recorded or guaranteed today (§1), hence a fingerprint check against values recorded at training. |
| NF4 (`attn_4bit=true`) | raw bf16 W, plus `s·B·A` | **bounded-loss (gated)**; refused until a read prices it | It serves the adapter on a base it never saw. The mismatch `W − Q_train(W)` is dense. |
| either | int4 / int8 / calibrated attention levers | **bounded-loss (gated)**; refused until a read gates it | The lever re-quantizes the merged weight, so the correction is rounded with the base. The calibrated levers also depend on their calibration set. |
| biased projections (gpt-oss) | bf16, merged | as the bf16 row, with the bias untouched | `attn_4bit` is refused for biased projections at training, so only the bf16 row applies. Whether a given run attached attention LoRA is read from its tensors. |

### Experts (gate_up, down)

| trained on | served on | label | why |
|---|---|---|---|
| NF4, absmax **fp32** (`E4B_ABSMAX_DQ=0`, expert offload, or `TRAIN_ARENA`) | the NF4 arena of the same revision and recipe, via a fused low-rank epilogue (gate_up delta before SwiGLU, down delta after) | **exact by construction**, once that epilogue exists (it does not today), with the base confirmed by fingerprint | Both sides make the same call: training's `F.quantize_4bit(..., blocksize=64, compress_statistics=False)` (`_vendor/experts.py:412-417`) and the bake's `F.quantize_4bit(w, blocksize=64, quant_type="nf4")` with bitsandbytes' default `compress_statistics=False` (grouped-nf4-gemm `nvme_bake_nf4.py:135`), both with fp32 absmax. A per-adapter fingerprint turns that into a check. The delta lands where training put it. |
| NF4, absmax **double-quantized** (**the default**) | the NF4 arena (fp32 absmax) | **bounded-loss (gated)**, **not exact** | The arena dequantizes with the fp32 absmax, and training saw the 8-bit nested one. `Q_train(W) ≠ Q_serve(W)` by the absmax rounding, in every block. |
| NF4, either absmax | a re-quantized merge `Q_target(Q_train(W) + s·B·A)`, into NF4, int4-b32 or MXFP4 | **bounded-loss (gated)** | Re-quantization can round a small delta away. Its retention is what the per-format premise measures. |
| NF4 | int4 expert stores or MXFP4 at serve time, with no merge | **refused** | Without a merge, the delta is simply not applied. |
| any | today's paged server | **refused** (no expert-LoRA path) | Arena loads build bare experts unless `arena_train=True` (`loader.py:1295-1299`). A nonzero expert `B` must refuse at startup, never be silently dropped. |
| gpt-oss, e4b's trainer | any | not applicable | e4b builds the experts bare, so no expert LoRA is trained. |
| gpt-oss, grouped-nf4-gemm's `ExpertsMxfp4LoRA` | today's paged server | **refused** | There is no expert-LoRA path on the paged server, and its own matrix row belongs to the MXFP4 expert path when one is built. |

### Other modules and conditions

| case | label | why |
|---|---|---|
| router, norms, `lm_head` deltas | **refused** | No LoRA is trained on them, and a full-weight router run is not saved with the adapter. |
| the base revision differs from `resolved_model_revision` | **refused** | The adapter corrects one base. `load_adapter` pins it, and the paged server must compare its own base, including the arena, which records no revision today. |
| `quant_type` or `blocksize` differs from the arena's `quantizer` | **refused** | A different `Q_serve(W)`. |
| a manifest without `resolved_model_revision`, or an unknown recipe | **refused** | Nothing to verify against. |
| placement (resident vs host-streamed arena rows) | **no change** | The same arena bytes are read wherever they live. Placement moves data, not math. |
| the int8 `lm_head` default | **allowed**: a base-serving property, reported | It is weight-only, per-row int8 with no activation calibration (each row's scale comes from its own absmax), on a module no adapter touches. So it is identical to the base server's, and the server reports it in `base_match`. A lever that depends on base-model calibration (the `*_INT4_CALIB` levers, the head's included) is refused under an adapter, because an adapter shifts the activations that calibration was fit to. |
| e4b's reference server (`serve.py`, `.pt` adapters) | **out of scope** | It cannot read `backend-adapter/1`, its files carry no `alpha` ("must match by convention", `serve.py:222-223`), and it checks no revision. The reference arm for a cross-format read is `loggetta.load_adapter`. |

## 4. What this asks of the work that follows

- **F1. The expert absmax mismatch.** Today's default training (double-quantized expert absmax) never serves exactly
  on today's arena (fp32 absmax). That holds for an expert adapter, and for the whole model around an attention-only
  adapter. Before any expert path ships, one of two things must change:
  - an arena variant that bakes the double-quantized absmax;
  - an explicit "servable" training preset with `E4B_ABSMAX_DQ=0`.

  Otherwise expert serving stays bounded-loss, behind the retention gate.
- **F2. The manifest must record the recipe** (`backend-adapter/2`): the expert absmax form, `E4B_TRAIN_FUSE_QKV`,
  the bitsandbytes version and, per quantized module, a fingerprint of `Q_train(W)`. Without these, the
  "exact, conditional" attention row cannot be verified.
- **F3. The arena must record its source revision,** so the server can refuse a mismatch rather than trust a path.
- **F4. Each serving lever that re-quantizes weights** (the attention int4/int8/calibrated levers, int4 expert
  stores) is refused under an adapter until a registered read gates it.
- **F5. Exactness is reported, never assumed.** The merge reports the max |merged − (W + s·B·A)| per projection, and the
  attention rebuild reports its fingerprint match.

## 5. What this does not say

- No cell carries a measured number. "Bounded-loss" means a gated read decides, not that the loss is small.
- Nothing here covers model types beyond those named, multi-adapter or per-request serving, or training-quality questions.
  Those belong to the registered reads that follow.
