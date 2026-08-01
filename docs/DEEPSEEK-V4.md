# DeepSeek-V4 (Flash / Pro)

Full **DeepSeek-V4-Flash** — 43 layers × 256 experts, 284B params — loads in ~10 s at
**8.74 GiB peak VRAM** and generates. The dense side measured 8.28 GiB against 8.40
predicted from the shard headers alone.

That is possible because V4 is >96% routed experts. Its non-expert half is 8.40 GiB, where
Kimi K3's is 114.4 GB and dense — the wall that put K3 out of reach on a small card is 13×
smaller here.

## Storage: the checkpoint is split two ways

| half | on disk | size (V4-Flash) | served as |
|---|---|---|---|
| routed experts | per-expert **MXFP4** at `mlp.experts.{e}.w{1,3,2}` (`I8` blocks + `F8_E8M0` scales) | 140 GiB | NF4 resident, or an arena |
| everything else | block-scaled **FP8** (e4m3 + one e8m0 scale per `[128,128]` tile) | 8.40 GiB | `Fp8BlockLinear`, decoded on use |

Keeping the dense half in FP8 is what makes it fit: ~1 byte/param instead of 2 is 8.4 GiB
resident against ~14, i.e. the difference between fitting a 12 GB card and not.

**The scale is used differently by each half, and crossing them is silent.**
`dequantize_mxfp4` wants the raw e8m0 *byte* (it feeds `ldexp`); `dequantize_fp8_blocks`
wants `2**(byte-127)` as a *multiplier*. Both tensors are labelled `F8_E8M0` on disk.

## The epilogue

V4 sits between the two this package already models — gpt-oss's **clamps** with SwiGLU's
**combination**:

```
gate = gate.clamp(max=L)              # one-sided
up   = up.clamp(min=-L, max=L)        # two-sided
out  = silu(gate) * up                # NOT (up+1) * gate * sigmoid(alpha*gate)
```

`L = swiglu_limit = 10.0` for both Flash and Pro. Two fidelity choices follow the
checkpoint's own `inference/model.py` rather than this package's gpt-oss path: the GLU is
evaluated in fp32, and the router weight is applied to the gated activation *before* the
down projection.

## Quantization

```python
model, cfg = load_moe_4bit_streaming(
    "deepseek-ai/DeepSeek-V4-Flash", "cuda", torch.bfloat16,
    r=8, alpha=16, quant_type="nf4",
)
```

Experts are re-quantized to NF4. The "exact released bytes" claim lives one step earlier, at
`dequantize_mxfp4`, which is verified **bit-identical** to DeepSeek's own
`inference/convert.py` decode on real V4 bytes (w1/w2/w3), with the high-nibble-first
reading scoring False so the test discriminates rather than coincides.

To compute on the released bytes instead, serve from a **native MXFP4 arena** — a
relocation bake, so it is smaller and faster to produce than a re-quantized one (147 GB and
~80 s, against 156 GB and a full quantize pass):

```bash
python -c "
from nvme_arena import bake_expert_tensors
from mxfp4_residency import V4_RESIDENCY_KINDS
bake_expert_tensors('/path/to/DeepSeek-V4-Flash', '/path/to/v4.mxarena',
                    name_template='layers.{layer}.ffn.experts.{expert}.{kind}',
                    kinds=V4_RESIDENCY_KINDS)"
```

```python
model, cfg = load_moe_4bit_streaming(SRC, "cuda", torch.bfloat16, r=8, alpha=16,
                                     quant_type="nf4", arena="/path/to/v4.mxarena")
enable_mxfp4_nvme_residency(model, "/path/to/v4.mxarena",
                            k_slots=cfg.num_experts_per_tok, hot_rows=16)
```

Arena mode and training are mutually exclusive — under `arena=` the base buffers are on
`meta`, so the enabler refuses to bind over an `ExpertsLoRA` rather than silently
discarding the adapter.

## Training

V4 is trainable. `ExpertsLoRA` re-implements the expert math inline so the low-rank delta
lands before the nonlinearity, which means it also owns the choice of nonlinearity — that
choice used to be hardcoded to plain SwiGLU, so wrapping any clamped-expert architecture
optimised a function the frozen base does not compute, silently, with the loss still
falling. The base now supplies the epilogue via `_apply_gate`.

## Key naming

The checkpoint ships in DeepSeek's own `inference/` spelling. transformers converts it via
the central `conversion_mapping.py`, but only inside `from_pretrained`, which the streaming
loader never enters — so `deepseek_v4.rename_checkpoint_key` does it here. The map is
cross-checked against upstream's table and agrees on every rule.

Three parts are not mechanical:

1. **The indexer nests the other way** — `attn.indexer.compressor.*` on disk,
   `self_attn.compressor.indexer.*` in the module tree.
2. **The shared expert uses `gate_proj`/`up_proj`/`down_proj`** while its routed siblings in
   the same block use `w1`/`w3`/`w2`.
3. **`hc_head` keeps an `hc_` prefix** on its own parameters while the per-layer
   hyper-connections drop it (`attn_hc.fn`).

`.scale` is also an overloaded suffix: FP8 tensors use `X.scale` beside `X.weight`, but
hyper-connections ship a standalone parameter literally *named* `scale`. Pair FP8 by sibling
`.weight` presence, never by suffix.

The `mtp.*` multi-token-prediction block is dropped — `num_nextn_predict_layers` does not
build it, and placing it would be worse than skipping it.

## Limits

- **Full-width resident loading does not fit a small card.** The loader stacks a whole
  layer's experts in bf16 before quantizing — `[256, 4096, 4096]` + `[256, 4096, 2048]` =
  12.9 GB transient per layer. Use the arena path, which never materializes a layer.
- A naive `from_pretrained` materializes the fp4 experts to bf16: **568 GB** for Flash.
- `o_a_proj` is a `DeepseekV4GroupedLinear` — an `nn.Linear` **subclass** with a
  block-diagonal forward. `convert_to_fp8_blocks` converts weights in place for exactly this
  reason; substituting a plain Linear silently turns a grouped projection into a dense one.

## Fixtures

The tests that assert against real checkpoint bytes skip unless fixtures are present.
Regenerate them (~62 MB) from any local V4 snapshot:

```bash
python tools/make_v4_fixtures.py --ckpt /path/to/DeepSeek-V4-Flash --out ./fixtures
E4B_V4_TENSORS=./fixtures/v4_expert.safetensors \
E4B_V4_KEYS=./fixtures/v4_keys_l0_5.txt \
E4B_V4_CONFIG=./fixtures/v4cfg pytest tests/
```
