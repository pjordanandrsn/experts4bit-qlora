# Relationship to bitsandbytes

*(Moved out of the top-level README for length; linked from it.)* [← back to README](../README.md)

`ExpertsNbit` / `Experts4bit` are bitsandbytes primitives, proposed upstream in
[bitsandbytes#1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965). Until that
ships in a release, this package **vendors** a copy (`experts4bit_qlora/_vendor/experts.py`) so it
runs on stock bitsandbytes today. The import shim prefers the upstream classes when present *and
still satisfying everything this package promises about them*: the internals `ExpertsLoRA` builds
on, `Experts4bit` a subclass of `ExpertsNbit`, and the state_dict metadata contract
(`get`/`set_extra_state` overrides). Both names must resolve to the same implementation, never a
mix — and anything less falls back to the vendored copy:

```python
try:
    from bitsandbytes.nn import Experts4bit, ExpertsNbit   # once bitsandbytes#1965 releases (if compatible)
except ImportError:
    from ._vendor.experts import Experts4bit, ExpertsNbit  # vendored fallback (stock bnb)
```

Nothing in training depends on the bitsandbytes version: the recompute-in-backward projection
delivers the activation-memory win on any release. The only `bnb.matmul_4bit` use left in the
package is the inference decode GEMV, which is probe-gated per configuration and passes on stock
0.49.x. When #1965 lands upstream: bump the `bitsandbytes` floor and delete `_vendor/` — no API
change.

### Prior art

The closest public prior art is
[woct0rdho/transformers-qwen3-moe-fused](https://github.com/woct0rdho/transformers-qwen3-moe-fused)
(Apache-2.0, June 2025 — a year before this package existed), which demonstrated bnb 4-bit
quantization of the fused 3-D expert stack and per-expert *stacked* LoRA
(`[num_experts, r, in]` / `[num_experts, out, r]`) for Qwen3-MoE, wrapped around a Triton
grouped-GEMM forward, and has since fed the Transformers-5-era fused-MoE ecosystem (Transformers,
PEFT, Unsloth). It reached the core primitive — 4-bit on the fused stack with trainable per-expert
adapters — first, and it is the better choice when fused-forward throughput is the goal. The two
projects optimize different axes: that one is a *kernel* project (grouped-GEMM speed; its fused
4-bit-dequant kernel is forward-only and listed as in-progress); this one is a *storage-contract*
project — deliberately per-expert-loop (see
[what ExpertsNbit is / is not](#what-expertsnbit-is--is-not)) — whose distinct contributions are
the tested training contract (recompute-in-backward holding no dequantized-expert activations,
offload asserted bit-identical to resident execution, packed storage asserted unchanged through
training steps), the fidelity-pinned N-bit storage matrix, the streaming loader + past-VRAM expert
offload, and train/serve byte-identity (the served base is asserted `torch.equal` to the base the
adapters were trained against).
