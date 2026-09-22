# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-f2-tail T2: fuse Q/K/V projections into one matmul per
attention module.

Stock Qwen3MoeAttention runs three cuBLAS GEMVs per layer over the SAME
input. One fused ``[Nq + Nk + Nv, hidden]`` weight computes all three in
a single launch; the outputs are recovered by ``split`` views. Each
output row is the same dot product over the same operands -- but NOT
guaranteed bitwise: BLAS kernel selection depends on the output dim, so
fusing can change a row's accumulation ORDER (measured on CPU sgemm at
a small shape: rel 2.9e-7, exactly fp32-reorder class; bitwise at the
real shape, but that is the box's kernel table, not a mechanism
property). The honest equivalence class is therefore reorder noise:
per-projection ``max|delta| <= max|ref| * 2^-7`` (the K6 relative
frame), with end-to-end token divergence step reported and gated, not
assumed zero. The patched forward mirrors transformers' own (5.15
source) line for line past the projections; structural drift (norm or
rotary misplacement) is an O(1) error and fails the same tolerance in
``tests/test_qkv_fuse.py`` long before the serving gates.

Arm-gated: nothing calls this unless the harness passes ``--fuse-qkv``
(the harness's default). On the int4 serving lanes it is licensed at
B=1 (P54: token-identical) and at B=16 (P59 amendment 1: 0.0044
nats/token through a 128-row prefill, reorder class; the K16 decode
path is bitwise invariant to fusing, the >16-row cuBLAS path is not).
Applied BEFORE ``--compile-layers`` so dynamo traces the fused forward.

**The projections may already be on the int4-b32 grid.** Every int4 lane
applies ``enable_serve_attn_int4`` at load (the hook rides the hybrid-tier
enable) and ``fuse_qkv`` runs after it, so until lane P54 the two were
exclusive: ``mod.q_proj.weight`` does not exist on an ``Int4Linear`` and the
arm refused, which is why every int4 census ran ``--no-fuse-qkv`` and paid
three attention launches per layer where the bf16 stack paid one. When
q/k/v are ``Int4Linear`` the fused projection is ``Int4Linear.fuse`` --
their packed rows and scales concatenated along N, byte-identical, so the
fused module computes the same function on one GEMV (rows == 1) or one
K16 small-M GEMM (rows 2..16) per layer. A mix of int4 and dense
projections is refused rather than half-fused.
"""

from __future__ import annotations

import types

import torch


def _fused_forward(self, hidden_states, position_embeddings,
                   attention_mask, past_key_values=None, **kwargs):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
        eager_attention_forward)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self._fused_nq, self._fused_nk,
                         self._fused_nv], dim=-1)
    query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward)

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def fuse_qkv(model) -> int:
    """Fuse every Qwen3MoeAttention's q/k/v projections in place.

    Returns the number of modules fused. REFUSES (raises) on a biased
    projection or a missing expected attribute rather than silently
    producing an unfused model the caller believes is fused -- the arm
    that requests fusion must get fusion or an error, never a quiet
    no-op ("green about other code").
    """
    fused = 0
    for mod in model.modules():
        if type(mod).__name__ != "Qwen3MoeAttention":
            continue
        for attr in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
                     "head_dim", "config"):
            if not hasattr(mod, attr):
                raise RuntimeError(
                    f"Qwen3MoeAttention missing {attr!r}: transformers "
                    "layout drifted; refusing to half-fuse")
        parts = (mod.q_proj, mod.k_proj, mod.v_proj)
        if any(p.bias is not None for p in parts):
            raise RuntimeError("biased q/k/v projections: the fused "
                               "split-view layout assumes bias=None "
                               "(Qwen3MoE ships attention_bias=False)")
        int4 = [type(p).__name__ == "Int4Linear" for p in parts]
        if all(int4):
            # the serving int4 store: fuse the BYTES, not a dequantised copy
            from .int4_attn import Int4Linear
            qkv = Int4Linear.fuse(parts)
            nq, nk, nv = (p.N for p in parts)
        elif any(int4):
            raise RuntimeError(
                "q/k/v projections are a mix of Int4Linear and dense modules "
                f"(int4: q={int4[0]} k={int4[1]} v={int4[2]}): refusing to half-fuse")
        else:
            wq, wk, wv = (p.weight for p in parts)
            qkv = torch.nn.Linear(wq.shape[1],
                                  wq.shape[0] + wk.shape[0] + wv.shape[0],
                                  bias=False, device=wq.device,
                                  dtype=wq.dtype)
            with torch.no_grad():
                qkv.weight[: wq.shape[0]].copy_(wq)
                qkv.weight[wq.shape[0]: wq.shape[0] + wk.shape[0]].copy_(wk)
                qkv.weight[wq.shape[0] + wk.shape[0]:].copy_(wv)
            nq, nk, nv = wq.shape[0], wk.shape[0], wv.shape[0]
        mod.qkv_proj = qkv
        mod._fused_nq = nq
        mod._fused_nk = nk
        mod._fused_nv = nv
        # drop the unfused Linears so their weights free and nothing can
        # accidentally run the old path
        del mod.q_proj, mod.k_proj, mod.v_proj
        mod.forward = types.MethodType(_fused_forward, mod)
        fused += 1
    # env-gated decode fusions ride the same serve assembly point so the
    # documented flags are LIVE on the advertised path (review finding:
    # an env var only read by a function nothing calls is dead)
    from .glue_fuse import fuse_t1_glue
    fuse_t1_glue(model)
    # round 2 rides the same point and runs AFTER the attention fusion
    # above, whose qkv_proj it requires (it replaces that forward)
    from .glue_r2 import fuse_t1_glue_r2
    fuse_t1_glue_r2(model)
    from .router_epilogue import fuse_router_epilogue
    fuse_router_epilogue(model)
    return fused
