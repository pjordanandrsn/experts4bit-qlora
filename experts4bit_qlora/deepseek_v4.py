"""DeepSeek-V4 fused experts in 4-bit — a `DeepseekV4Experts` replacement.

V4's experts sit between the two epilogues this package already models. They use
gpt-oss's *clamps* but SwiGLU's *combination*::

    gate = gate.clamp(max=limit)                 # one-sided
    up   = up.clamp(min=-limit, max=limit)       # two-sided
    out  = silu(gate) * up                       # NOT (up + 1) * gate * sigmoid(alpha * gate)

so neither :class:`ExpertsNbit` (no clamps) nor :class:`GptOssExperts4bit`
(gpt-oss's ``alpha``/``+1`` GLU) is correct for it, and it gets its own subclass.
``limit`` is the checkpoint's ``swiglu_limit`` — 10.0 for both V4-Flash and V4-Pro.
``limit <= 0`` disables clamping, matching the reference's ``if self.swiglu_limit > 0``.

Unlike gpt-oss there is **no layout transform and there are no biases**. V4 ships
each projection separately and output-major — ``w1``/``w3`` are ``[inter, hidden]``,
``w2`` is ``[hidden, inter]`` — so ``cat([w1, w3], dim=0)`` is already e4b's
``[2*inter, hidden]`` gate-block-then-up-block, and ``w2`` is already ``[hidden, inter]``.
Nothing is interleaved and nothing needs transposing.

Two fidelity choices follow the checkpoint's own ``inference/model.py`` ``Expert.forward``
rather than this package's gpt-oss path:

* the clamp + SiLU are evaluated in **fp32** (the reference does ``self.w1(x).float()``),
  then cast back before the down projection;
* the router weight is applied to the **gated activation, before** the down projection.
  With no bias term that is algebraically the same as scaling the output, but it is not
  bit-identical in floating point, and the reference does it first.

Provenance is the same story as gpt-oss: the loaded experts are NF4, a re-quantization
of the released MXFP4 bytes; the "exact released bytes" claim lives one step earlier at
:func:`experts4bit_qlora.mxfp4.dequantize_mxfp4`, which is verified bit-identical to
DeepSeek's own ``inference/convert.py`` decode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._vendor.experts import Experts4bit, ExpertsNbit

DEFAULT_SWIGLU_LIMIT = 10.0

# --------------------------------------------------------------------------- keys
#
# DeepSeek publishes the V4 checkpoints in their own *reference* spelling — the one
# `inference/generate.py` reads. transformers DOES convert it, but the mapping lives
# centrally in `transformers/conversion_mapping.py` (keyed `"deepseek_v4"`) rather than as
# a `_checkpoint_conversion_mapping` attribute on the model class, and it only runs inside
# `from_pretrained`. This loader streams shards directly and never enters that path, so it
# still needs its own rename.
#
# Cross-checked against upstream's table: it agrees on every rule below, including the two
# that are not guessable. Independent agreement is the reason to trust this, since a
# plausible-but-wrong mapping loads with correct shapes and computes nonsense.
#
# Three things here are not guessable and were read off the built module tree rather
# than assumed:
#
# 1. The indexer NESTS THE OTHER WAY. On disk it is `attn.indexer.compressor.*`;
#    in the module tree it is `self_attn.compressor.indexer.*`. Same reversal K3
#    pulls with `language_model.model.` vs `model.language_model.`.
# 2. The shared expert is a plain MLP, so it uses `gate_proj`/`up_proj`/`down_proj`
#    — NOT the `w1`/`w3`/`w2` spelling its routed siblings use in the same block.
# 3. `hc_head` keeps an `hc_` prefix on its own parameters (`hc_head.hc_fn`) while the
#    per-layer hyper-connections drop it (`attn_hc.fn`). Inconsistent, and load-bearing.
#
# Order matters: these are substring rewrites applied in sequence, most specific first.
DEEPSEEK_V4_RENAMES = (
    # indexer-inside-compressor — must precede the bare `attn.compressor.` rules
    (".attn.indexer.compressor.wgate", ".self_attn.compressor.indexer.gate_proj"),
    (".attn.indexer.compressor.wkv", ".self_attn.compressor.indexer.kv_proj"),
    (".attn.indexer.compressor.norm", ".self_attn.compressor.indexer.kv_norm"),
    (".attn.indexer.compressor.ape", ".self_attn.compressor.indexer.position_bias"),
    (".attn.indexer.wq_b", ".self_attn.compressor.indexer.q_b_proj"),
    (".attn.indexer.weights_proj", ".self_attn.compressor.indexer.scorer.weights_proj"),
    (".attn.compressor.wgate", ".self_attn.compressor.gate_proj"),
    (".attn.compressor.wkv", ".self_attn.compressor.kv_proj"),
    (".attn.compressor.norm", ".self_attn.compressor.kv_norm"),
    (".attn.compressor.ape", ".self_attn.compressor.position_bias"),
    (".attn.wq_a", ".self_attn.q_a_proj"),
    (".attn.wq_b", ".self_attn.q_b_proj"),
    (".attn.wkv", ".self_attn.kv_proj"),
    (".attn.wo_a", ".self_attn.o_a_proj"),
    (".attn.wo_b", ".self_attn.o_b_proj"),
    (".attn.q_norm", ".self_attn.q_a_norm"),
    (".attn.kv_norm", ".self_attn.kv_norm"),
    (".attn.attn_sink", ".self_attn.sinks"),
    (".attn_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    # shared expert is a DeepseekV4MLP, so gate/up/down — not w1/w3/w2
    (".ffn.shared_experts.w1", ".mlp.shared_experts.gate_proj"),
    (".ffn.shared_experts.w3", ".mlp.shared_experts.up_proj"),
    (".ffn.shared_experts.w2", ".mlp.shared_experts.down_proj"),
    (".ffn.gate.bias", ".mlp.gate.e_score_correction_bias"),
    (".hc_attn_fn", ".attn_hc.fn"),
    (".hc_attn_base", ".attn_hc.base"),
    (".hc_attn_scale", ".attn_hc.scale"),
    (".hc_ffn_fn", ".ffn_hc.fn"),
    (".hc_ffn_base", ".ffn_hc.base"),
    (".hc_ffn_scale", ".ffn_hc.scale"),
    # routed experts + router keep their leaf names; only the block is renamed
    (".ffn.", ".mlp."),
)

# Whole-key renames for the tensors that sit outside `layers.`.
DEEPSEEK_V4_TOPLEVEL = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}


def rename_checkpoint_key(key: str) -> str | None:
    """Map one on-disk DeepSeek-V4 key to its name in the transformers module tree.

    Returns ``None`` for tensors the text model does not have — currently only the
    multi-token-prediction block (``mtp.*``), which ``num_nextn_predict_layers`` does
    not build. Dropping it is deliberate: it is a whole extra layer of weights with no
    module to receive it, and silently `_assign`-ing it would be worse than skipping it.
    """
    if key.startswith("mtp."):
        return None
    if key in DEEPSEEK_V4_TOPLEVEL:
        return DEEPSEEK_V4_TOPLEVEL[key]
    if not key.startswith("layers."):
        return None
    key = "model." + key
    for old, new in DEEPSEEK_V4_RENAMES:
        key = key.replace(old, new)
    return key


class _DeepseekV4ForwardMixin:
    """V4 expert forward (clamped SwiGLU, no biases) + the load-time builder.

    Applied over either storage base: :class:`Experts4bit` (nf4/fp4) or
    :class:`ExpertsNbit` (bf16/fp16/int8/fp8 — the structural parity path).
    ``from_deepseek_v4`` dispatches the base by ``quant_type`` and rebinds.
    """

    @classmethod
    def from_deepseek_v4(
        cls,
        gate_up_dense: torch.Tensor,   # [E, 2*inter, hidden] — cat([w1, w3], dim=0) per expert
        down_dense: torch.Tensor,      # [E, hidden, inter]   — w2 per expert
        *,
        limit: float = DEFAULT_SWIGLU_LIMIT,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "_DeepseekV4ForwardMixin":
        if gate_up_dense.ndim != 3 or down_dense.ndim != 3:
            raise ValueError("expected [E, *, *] dense stacks")
        E, twoI, H = gate_up_dense.shape
        if twoI % 2:
            raise ValueError(f"gate_up dim 1 = {twoI} is not even; cannot split gate/up")
        if down_dense.shape != (E, H, twoI // 2):
            raise ValueError(
                f"down {tuple(down_dense.shape)} does not match gate_up "
                f"{tuple(gate_up_dense.shape)}; expected {(E, H, twoI // 2)}. V4 ships w2 as "
                "[hidden, inter] — if this looks transposed, the `.T` after dequantize_mxfp4 "
                "was probably applied the wrong number of times."
            )

        base_cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit
        obj = base_cls.from_float(
            gate_up_dense, down_dense, has_gate=True,
            quant_type=quant_type, compute_dtype=compute_dtype,
        )
        # same slots + storage; rebind to the V4 forward over the matching base
        obj.__class__ = DeepseekV4Experts4bit if base_cls is Experts4bit else DeepseekV4ExpertsNbit
        obj.limit = float(limit)
        return obj

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """V4's epilogue, in ONE place: clamped SwiGLU over a clean-concat gate_up.

        Named to match the hook ``ExpertsLoRA`` looks for (and the method transformers'
        own ``DeepseekV4Experts`` carries), so the trainable adapter reproduces this
        instead of assuming a plain SwiGLU. Without it the adapter optimises a function
        the frozen base does not compute -- silently, since the loss still falls.

        Returns **fp32**: the reference computes the whole GLU in fp32 and only casts
        back before the down projection, so callers cast when they are ready rather than
        losing precision here.
        """
        gate, up = gate_up.chunk(2, dim=-1)
        gate, up = gate.float(), up.float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)        # one-sided, by design
            up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,   # [num_tokens, top_k]
        router_scores: torch.Tensor,    # [num_tokens, top_k]
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        cd = self.compute_dtype if self.compute_dtype is not None else input_dtype
        x = hidden_states.to(cd)
        out = torch.zeros_like(x, dtype=torch.float32)

        with torch.no_grad():
            mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for e in hit:
            pos, tok = torch.where(mask[e])
            cur = x[tok]
            gate_up = self._project(
                self.gate_up_proj, self.gate_up_absmax, self._gate_up_shape, e, cur, cd
            )
            gated = self._apply_gate(gate_up)            # fp32, per the reference
            gated = gated * router_scores[tok, pos, None].float()   # before w2, as the reference does
            h = self._project(
                self.down_proj, self.down_absmax, self._down_shape, e, gated.to(cd), cd
            )
            out.index_add_(0, tok, h.to(out.dtype))

        return out.to(input_dtype)


class DeepseekV4ExpertsNbit(_DeepseekV4ForwardMixin, ExpertsNbit):
    """V4 experts over the general N-bit storage base (bf16/fp16/int8/fp8)."""


class DeepseekV4Experts4bit(_DeepseekV4ForwardMixin, Experts4bit):
    """V4 experts over the NF4/FP4 base — the production path."""
