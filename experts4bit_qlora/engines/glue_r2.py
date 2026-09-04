# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Glue round 2 for decode (``E4B_FUSE_T1_GLUE_R2=1``).

Round 1 fused the RMSNorm call itself. The Phase A / P15 censuses put
what is left of the non-GEMM step at ~1.9 ms of the 6.5 ms B=1 step
and ~4.0 ms of the 15.5 ms B=16 step, spread over elementwise adds,
small reductions and the rotary chain. Two folds take the bulk of it:

* **residual + post-attention norm** -- the decoder layer's
  ``hidden = residual + attn_out`` followed by
  ``post_attention_layernorm(hidden)`` becomes one kernel call that
  returns both the new residual and the normed activation.
* **q/k norm + rotary** -- the per-head norm plus the multi-kernel
  ``apply_rotary_pos_emb`` chain (slice, negate, concat, two muls, an
  add) becomes one launch per projection.

A second layer shape is folded the same way: GraniteMoe's body keeps
the four pre-norm children but names its MoE ``block_sparse_moe`` and
scales both residual adds by a Python-float ``residual_multiplier``
(``resid + x * m``). That fold needs the kernel side's scaled residual
fold (``rmsnorm_resid_rows(..., scale=)`` and ``scaled_resid_add_rows``,
grouped-nf4-gemm >= 0.27) and refuses loudly on an older cut rather
than silently skipping the layer.

Both patches are licensed the way round 1's was: structure is checked,
never assumed from a class name, and the norm modules must pass the
same semantic probe that rejects centered ``x * (1 + w)`` variants.
The attention fold comes in two licensed shapes: the module this package
already fused (``qkv_proj`` present) and the standard separate-projection
attention (q/k/v/o + per-head q/k norms) that the calibrated int4 lane
runs; anything else keeps its own forward rather than being half-patched.
Off decode shapes every patch falls through to the original chain.

Engagement is census PRESENCE of ``_rmsnorm_resid_rows`` and
``_rope_norm_heads``, never a symbol grep.
"""
from __future__ import annotations

import inspect
import os

import torch

from .glue_fuse import _is_rmsnorm, _norm_eps, _probe_matches

__all__ = ["fuse_t1_glue_r2"]

# decode rows stay small; prefill keeps the upstream chain
_MAX_DECODE_ROWS = 64


def _decode_rows(x: torch.Tensor, width: int) -> bool:
    return (x.dtype == torch.bfloat16
            and x.shape[-1] == width
            and x.numel() <= _MAX_DECODE_ROWS * width)


_PLAIN_LAYER_CHILDREN = frozenset(
    {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp"})


def _layer_is_plain(mod) -> bool:
    """True when the decoder layer is exactly the four-child pre-norm
    body the round-2 fold re-implements: no extra child modules (further
    norms, parallel branches), no parameters or buffers of its own
    (residual multipliers, layer scalars)."""
    children = {n for n, _ in mod.named_children()}
    if children != _PLAIN_LAYER_CHILDREN:
        return False
    if any(True for _ in mod.named_parameters(recurse=False)):
        return False
    if any(True for _ in mod.named_buffers(recurse=False)):
        return False
    return True


_SCALED_LAYER_CHILDREN = frozenset(
    {"input_layernorm", "self_attn", "post_attention_layernorm",
     "block_sparse_moe"})


def _layer_scale(mod):
    """The residual multiplier of a GraniteMoe-shaped layer, or None when
    the layer is not that shape: exactly the four pre-norm children with
    the MoE under ``block_sparse_moe``, nothing else on the layer itself
    (no parameters, no buffers), and ``residual_multiplier`` either
    absent (1.0 -- the older Mixtral cut of the same body) or a Python
    float. A tensor or integer multiplier is a body this fold has not
    read and is refused."""
    children = {n for n, _ in mod.named_children()}
    if children != _SCALED_LAYER_CHILDREN:
        return None
    if any(True for _ in mod.named_parameters(recurse=False)):
        return None
    if any(True for _ in mod.named_buffers(recurse=False)):
        return None
    if "residual_multiplier" not in vars(mod):
        return 1.0
    m = vars(mod)["residual_multiplier"]
    if type(m) is not float:
        return None
    return m


def _kernel_has_scaled_fold(int4_b32) -> bool:
    fn = getattr(int4_b32, "rmsnorm_resid_rows", None)
    add = getattr(int4_b32, "scaled_resid_add_rows", None)
    if fn is None or add is None:
        return False
    try:
        return "scale" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _patch_layer_scaled(mod, scale, int4_b32) -> bool:
    """Fold GraniteMoe's ``resid + attn * m`` into the post-attention
    norm and its tail ``resid + moe * m`` into one launch. Mirrors the
    upstream forward (transformers 5.5 source) line for line; only the
    two scaled add sites change. Requires the kernel cut with the
    scaled fold when ``m != 1``; at ``m == 1`` the body is the plain
    fold under another child name and the older cut suffices."""
    ln = getattr(mod, "post_attention_layernorm", None)
    if ln is None or not _is_rmsnorm(ln):
        return False
    eps = _norm_eps(ln)
    if eps is None or not _probe_matches(ln, eps):
        return False
    scaled = scale != 1.0
    if scaled and not _kernel_has_scaled_fold(int4_b32):
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE_R2=1 on a residual-scaled layer body "
            f"({type(mod).__name__}, residual_multiplier={scale}) needs "
            "the kernel side's scaled residual fold "
            "(rmsnorm_resid_rows(scale=) and scaled_resid_add_rows, "
            "grouped-nf4-gemm >= 0.27); install the matching cut or "
            "unset the flag")
    rmsnorm_resid_rows = int4_b32.rmsnorm_resid_rows
    scaled_add = int4_b32.scaled_resid_add_rows if scaled else None
    orig = mod.forward
    width = ln.weight.numel()

    def _fwd(hidden_states, attention_mask=None, past_key_values=None,
             position_embeddings=None, _m=mod, _ln=ln, _eps=eps,
             _orig=orig, _w=width, _s=scale, _add=scaled_add, **kwargs):
        if not _decode_rows(hidden_states, _w):
            return _orig(hidden_states, attention_mask=attention_mask,
                         past_key_values=past_key_values,
                         position_embeddings=position_embeddings,
                         **kwargs)
        residual = hidden_states
        hidden_states = _m.input_layernorm(hidden_states)
        hidden_states, _ = _m.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # residual + attn * m, then the post-attention norm: one launch
        if _add is None:
            hidden_states, residual = rmsnorm_resid_rows(
                hidden_states, residual, _ln.weight, _eps)
        else:
            hidden_states, residual = rmsnorm_resid_rows(
                hidden_states, residual, _ln.weight, _eps, scale=_s)
        hidden_states = _m.block_sparse_moe(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]    # (out, router_logits) cuts
        if _add is None:
            return residual + hidden_states
        return _add(hidden_states, residual, _s)

    mod.forward = _fwd
    return True


def _patch_layer(mod, rmsnorm_resid_rows) -> bool:
    """Fold ``residual + attn_out`` into the post-attention norm.

    Mirrors the upstream forward (transformers 5.5 source) line for
    line; only the add-then-norm pair changes."""
    ln = getattr(mod, "post_attention_layernorm", None)
    if ln is None or not _is_rmsnorm(ln):
        return False
    eps = _norm_eps(ln)
    if eps is None or not _probe_matches(ln, eps):
        return False
    for attr in ("input_layernorm", "self_attn", "mlp"):
        if not hasattr(mod, attr):
            return False
    # The fold REPLACES the layer's forward with the Qwen3-shaped body
    # (norm -> attn -> add+norm -> mlp -> add). A layer whose children are
    # a superset of that shape has a different body: Gemma-4 carries two
    # more norms, a parallel routed-expert branch and a layer scalar under
    # the SAME four attribute names, and GraniteMoe scales its residuals.
    # Name presence cannot tell them apart, so the structure must be
    # EXACTLY the four children and nothing else on the layer itself.
    if not _layer_is_plain(mod):
        return False
    orig = mod.forward
    width = ln.weight.numel()

    def _fwd(hidden_states, attention_mask=None, position_ids=None,
             past_key_values=None, use_cache=False,
             position_embeddings=None, _m=mod, _ln=ln, _eps=eps,
             _orig=orig, _w=width, **kwargs):
        if not _decode_rows(hidden_states, _w):
            return _orig(hidden_states, attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=past_key_values,
                         use_cache=use_cache,
                         position_embeddings=position_embeddings,
                         **kwargs)
        residual = hidden_states
        hidden_states = _m.input_layernorm(hidden_states)
        hidden_states, _ = _m.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # residual add + post-attention norm, one launch
        hidden_states, residual = rmsnorm_resid_rows(
            hidden_states, residual, _ln.weight, _eps)
        hidden_states = _m.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            # gpt-oss's MoE block returns (hidden, router_scores) and its
            # layer unpacks ``hidden_states, _ = self.mlp(...)``; adding
            # the tuple to the residual raised a TypeError on the
            # validation lane. Mirror the unpack.
            hidden_states = hidden_states[0]
        return residual + hidden_states

    mod.forward = _fwd
    return True


def _patch_attention(mod, rope_norm_heads) -> bool:
    """Fold each of q_norm/k_norm plus rotary into one launch."""
    if not hasattr(mod, "qkv_proj"):
        return False            # only this package's fused attention
    for attr in ("q_norm", "k_norm", "head_dim", "_fused_nq",
                 "_fused_nk", "_fused_nv", "o_proj", "scaling",
                 "sliding_window", "layer_idx", "config"):
        if not hasattr(mod, attr):
            return False
    qn, kn = mod.q_norm, mod.k_norm
    if not (_is_rmsnorm(qn) and _is_rmsnorm(kn)):
        return False
    qe, ke = _norm_eps(qn), _norm_eps(kn)
    if qe is None or ke is None:
        return False
    if not (_probe_matches(qn, qe) and _probe_matches(kn, ke)):
        return False
    if mod.head_dim % 2 or qn.weight.numel() != mod.head_dim:
        return False
    orig = mod.forward

    def _fwd(hidden_states, position_embeddings=None,
             attention_mask=None, past_key_values=None, _m=mod,
             _qn=qn, _kn=kn, _qe=qe, _ke=ke, _orig=orig, **kwargs):
        d = _m.head_dim
        rows = hidden_states.numel() // hidden_states.shape[-1]
        if (position_embeddings is None
                or hidden_states.dtype != torch.bfloat16
                or rows > _MAX_DECODE_ROWS):
            return _orig(hidden_states,
                         position_embeddings=position_embeddings,
                         attention_mask=attention_mask,
                         past_key_values=past_key_values, **kwargs)
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            ALL_ATTENTION_FUNCTIONS, eager_attention_forward)

        input_shape = hidden_states.shape[:-1]
        qkv = _m.qkv_proj(hidden_states)
        q, k, v = qkv.split([_m._fused_nq, _m._fused_nk, _m._fused_nv],
                            dim=-1)
        cos, sin = position_embeddings
        # upstream broadcasts cos/sin against the head axis, and may
        # carry a batch of 1 that broadcasts across rows as well. The
        # kernel indexes one cos/sin row PER row, so materialise that
        # broadcast here; any other layout keeps the upstream chain
        # rather than silently rotating with the wrong positions
        # (review finding, High).
        cos2 = cos.reshape(-1, d)
        sin2 = sin.reshape(-1, d)
        if cos2.shape[0] == 1 and rows > 1:
            cos2 = cos2.expand(rows, d)
            sin2 = sin2.expand(rows, d)
        if cos2.shape[0] != rows or sin2.shape[0] != rows:
            return _orig(hidden_states,
                         position_embeddings=position_embeddings,
                         attention_mask=attention_mask,
                         past_key_values=past_key_values, **kwargs)
        # norm + rotary, one launch per projection
        query_states = rope_norm_heads(
            q.reshape(rows, -1, d), _qn.weight, cos2, sin2, _qe
        ).reshape(*input_shape, -1, d).transpose(1, 2)
        key_states = rope_norm_heads(
            k.reshape(rows, -1, d), _kn.weight, cos2, sin2, _ke
        ).reshape(*input_shape, -1, d).transpose(1, 2)
        value_states = v.view(*input_shape, -1, d).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, _m.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            _m.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            _m, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not _m.training else _m.attention_dropout,
            scaling=_m.scaling,
            sliding_window=_m.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return _m.o_proj(attn_output), attn_weights

    mod.forward = _fwd
    return True


_UNFUSED_ATTN_CHILDREN = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"})


def _patch_attention_unfused(mod, rope_norm_heads) -> bool:
    """The same norm + rotary fold for the STANDARD separate-projection
    attention (Qwen3-MoE-shaped: q/k/v/o projections plus per-head q/k
    norms), which is what every family runs under the calibrated int4
    attention lane -- that lane packs the four projections separately
    and is exclusive with qkv fusion, so the fused-only fold above never
    engaged on the campaign's own best stack.

    Licensed on structure: exactly those six children, nothing of the
    module's own, norms whose weight is ``head_dim`` wide (OLMoE norms the
    full hidden width BEFORE the head split -- a different function,
    refused), and the round-1 semantic probe on both norms. The
    projections may be any module (``nn.Linear`` or the int4 store's
    replacement); they are called, never read."""
    children = {n for n, _ in mod.named_children()}
    if children != _UNFUSED_ATTN_CHILDREN:
        return False
    if any(True for _ in mod.named_parameters(recurse=False)):
        return False
    if any(True for _ in mod.named_buffers(recurse=False)):
        return False
    for attr in ("head_dim", "scaling", "sliding_window", "layer_idx",
                 "config", "attention_dropout"):
        if not hasattr(mod, attr):
            return False
    qn, kn = mod.q_norm, mod.k_norm
    if not (_is_rmsnorm(qn) and _is_rmsnorm(kn)):
        return False
    qe, ke = _norm_eps(qn), _norm_eps(kn)
    if qe is None or ke is None:
        return False
    if not (_probe_matches(qn, qe) and _probe_matches(kn, ke)):
        return False
    d = int(mod.head_dim)
    if d % 2 or qn.weight.numel() != d or kn.weight.numel() != d:
        return False
    orig = mod.forward

    def _fwd(hidden_states, position_embeddings=None,
             attention_mask=None, past_key_values=None, _m=mod,
             _qn=qn, _kn=kn, _qe=qe, _ke=ke, _orig=orig, _d=d, **kwargs):
        rows = hidden_states.numel() // hidden_states.shape[-1]
        if (position_embeddings is None
                or hidden_states.dtype != torch.bfloat16
                or rows > _MAX_DECODE_ROWS):
            return _orig(hidden_states,
                         position_embeddings=position_embeddings,
                         attention_mask=attention_mask,
                         past_key_values=past_key_values, **kwargs)
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            ALL_ATTENTION_FUNCTIONS, eager_attention_forward)

        input_shape = hidden_states.shape[:-1]
        q = _m.q_proj(hidden_states)
        k = _m.k_proj(hidden_states)
        v = _m.v_proj(hidden_states)
        cos, sin = position_embeddings
        cos2 = cos.reshape(-1, _d)
        sin2 = sin.reshape(-1, _d)
        if cos2.shape[0] == 1 and rows > 1:
            cos2 = cos2.expand(rows, _d)
            sin2 = sin2.expand(rows, _d)
        if cos2.shape[0] != rows or sin2.shape[0] != rows:
            return _orig(hidden_states,
                         position_embeddings=position_embeddings,
                         attention_mask=attention_mask,
                         past_key_values=past_key_values, **kwargs)
        query_states = rope_norm_heads(
            q.reshape(rows, -1, _d), _qn.weight, cos2, sin2, _qe
        ).reshape(*input_shape, -1, _d).transpose(1, 2)
        key_states = rope_norm_heads(
            k.reshape(rows, -1, _d), _kn.weight, cos2, sin2, _ke
        ).reshape(*input_shape, -1, _d).transpose(1, 2)
        value_states = v.view(*input_shape, -1, _d).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, _m.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            _m.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            _m, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not _m.training else _m.attention_dropout,
            scaling=_m.scaling,
            sliding_window=_m.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return _m.o_proj(attn_output), attn_weights

    mod.forward = _fwd
    return True


def fuse_t1_glue_r2(model) -> tuple[int, int]:
    """Apply the round-2 decode folds. Returns ``(layers, attentions)``.

    Refuses a vacuous enable: an arm that asks for the fusion must get
    it or an error, never a quiet no-op."""
    if os.environ.get("E4B_FUSE_T1_GLUE_R2", "0") != "1":
        return (0, 0)
    try:
        import int4_b32  # the module object is needed for the capability probe
        from int4_b32 import rmsnorm_resid_rows, rope_norm_heads
    except ImportError as e:
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE_R2=1 needs the kernel side's "
            "rmsnorm_resid_rows/rope_norm_heads; install the matching "
            "cut or unset the flag") from e

    layers = attns = 0
    for mod in model.modules():
        name = type(mod).__name__
        if name.endswith("DecoderLayer"):
            scale = _layer_scale(mod)
            if scale is None:
                layers += bool(_patch_layer(mod, rmsnorm_resid_rows))
            else:
                layers += bool(_patch_layer_scaled(mod, scale, int4_b32))
        elif name.endswith("Attention"):
            attns += bool(_patch_attention(mod, rope_norm_heads)
                          or _patch_attention_unfused(mod, rope_norm_heads))
    if layers == 0 and attns == 0:
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE_R2=1 patched nothing (no structurally "
            "matched decoder layer or fused attention passed the "
            "probes) -- refusing a vacuous enable")
    return (layers, attns)
