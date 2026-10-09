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

The rotary is licensed the same way (:func:`rotary_is_rotate_half`): the
kernels compute rotate-half, so the module's OWN rotary must compute the
same function, probed, never assumed. ERNIE-4.5's attention has exactly
the q/k/v/o structure, but it re-interleaves cos/sin and rotates
adjacent pairs; it is refused (the FAM lane's find, e4b#1362).

Engagement is census PRESENCE of ``_rmsnorm_resid_rows`` and
``_rope_norm_heads``, never a symbol grep.
"""
from __future__ import annotations

import inspect

import torch

from .glue_fuse import _is_rmsnorm, _norm_eps, _note, _probe_matches, fold_mode

__all__ = ["fuse_t1_glue_r2", "license_moe_residual"]

# decode rows stay small; prefill keeps the upstream chain
_MAX_DECODE_ROWS = 64


def _module_rotary(mod):
    """The rotary the module's own forward applies: ``apply_rotary_pos_emb`` in the namespace its class's ``forward``
    was defined in, which is how every transformers attention calls it (an instance-level forward patch does not
    change it). ``None`` when there is none to read."""
    fn = getattr(getattr(type(mod), "forward", None), "__globals__", {}).get("apply_rotary_pos_emb")
    return fn if callable(fn) else None


def _probe_rotary(fn, d: int) -> bool:
    """Does ``fn(q, k, cos, sin)`` compute ``x * cos + rotate_half(x) * sin`` with ``rotate_half(x) = cat(-x[d/2:],
    x[:d/2])`` -- what ``rope_heads`` and ``rope_norm_heads`` compute -- for ARBITRARY cos/sin, in transformers'
    calling convention (q/k ``[batch, heads, T, d]``, cos/sin ``[batch, T, d]``)?"""
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(1, 2, 3, d, generator=g), torch.randn(1, 1, 3, d, generator=g)
    cos, sin = torch.randn(1, 3, d, generator=g), torch.randn(1, 3, d, generator=g)
    try:
        with torch.no_grad():
            got = fn(q, k, cos, sin)
    except Exception:       # a rotary that cannot be probed is not licensed
        return False
    h = d // 2

    def want(x):
        return x * cos.unsqueeze(1) + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin.unsqueeze(1)
    return (isinstance(got, (tuple, list)) and len(got) == 2
            and all(isinstance(t, torch.Tensor) and t.shape == x.shape
                    and torch.allclose(t.float(), want(x), rtol=1e-5, atol=1e-5)
                    for t, x in zip(got, (q, k))))


_ROTARY_PROBED: dict = {}


def rotary_is_rotate_half(mod, d: int) -> bool:
    """License a fold's rotary on what the module computes: its own ``apply_rotary_pos_emb``
    (:func:`_module_rotary`) must be rotate-half over ``d`` (:func:`_probe_rotary`). ERNIE-4.5 (interleaved),
    a module whose rotary cannot be read, and one that raises on the probe are refused. One probe per function and
    width."""
    fn = _module_rotary(mod)
    if fn is None or d % 2:
        return False
    if (fn, d) not in _ROTARY_PROBED:
        _ROTARY_PROBED[(fn, d)] = _probe_rotary(fn, d)
    return _ROTARY_PROBED[(fn, d)]


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


class _KernelGap(RuntimeError):
    """A structure this fold licenses, on a kernel cut that lacks what its patch needs. Under ``1`` it is raised as the
    refusal it always was; under ``auto`` the module is left unpatched and counted."""


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
        raise _KernelGap(
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
        lic = _m.__dict__.get("_e4b_moe_resid")
        if lic is not None and hidden_states.numel() // _w in lic:
            # the MoE block's own composition with the residual add folded into the experts' combine, at a row count
            # license_moe_residual's probe of this layer AS SERVED found bitwise (e4b#1313, lane P127's item c)
            return _moe_with_residual(_m.mlp, hidden_states, residual, _w)
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


def _moe_with_residual(mlp, h, residual, width):
    """``residual + mlp(h)`` by the composition :func:`license_moe_residual` licenses: the block's gate, then its
    experts with the residual handed down (``_, weights, ids = gate(x)``; ``experts(x, ids, weights)``, the
    transformers 5.x sparse-MoE block's body), reshaped back."""
    h2 = h.reshape(-1, width)
    _, w, idx = mlp.gate(h2)
    return mlp.experts(h2, idx, w, residual=residual.reshape(-1, width)).reshape(residual.shape)


def _residual_candidate(mod) -> bool:
    """A decoder layer under :func:`_patch_layer`'s fold whose MLP is a bare gate + experts block and whose experts
    forward takes ``residual=`` (hot residency's, :mod:`.hot_residency`)."""
    fwd = getattr(mod, "forward", None)
    if getattr(fwd, "__qualname__", "") != "_patch_layer.<locals>._fwd":
        return False
    mlp = getattr(mod, "mlp", None)
    if mlp is None or {n for n, _ in mlp.named_children()} != {"gate", "experts"}:
        return False
    return bool(getattr(getattr(mlp.experts, "forward", None), "_e4b_takes_residual", False))


def license_moe_residual(model, rows, mode: str | None = None, report: dict | None = None) -> int:
    """License each folded decoder layer to hand its MoE residual add to the experts' combine (lane P127's item c,
    e4b#1313; the kernel side's ``combine_rows(..., residual=)``, grouped-nf4-gemm#527), one row count at a time.

    Run on the model AS SERVED -- after residency, the collapse and the batched grouping are configured, before any
    graph is captured -- with ``rows`` every row count the decode composition may run (the server's decode buckets;
    1 at least). At each, on distinct finite random bf16 inputs and without grad, the layer's composition
    (:func:`_moe_with_residual`) must be bitwise the layer's own ``residual + mlp(h)``; the counts that are become
    the layer's licence, and any other count keeps the layer's own body. Candidates: :func:`_residual_candidate`.

    ``mode`` is the layer fold's own (``E4B_FUSE_T1_GLUE_R2``, :func:`~.glue_fuse.fold_mode`): ``0`` licenses
    nothing; ``1`` refuses when no layer is licensed at every count. Returns the layers licensed at every count."""
    mode = fold_mode("E4B_FUSE_T1_GLUE_R2", mode)
    rows = sorted({int(r) for r in rows} | {1})
    _note(report, rows=rows)
    if mode == "0":
        return 0
    gen = torch.Generator(device="cpu").manual_seed(0x5127)
    full = partial = refused = 0
    errors = []
    for mod in model.modules():
        if not _residual_candidate(mod):
            continue
        width = mod.post_attention_layernorm.weight.numel()
        dev = mod.post_attention_layernorm.weight.device
        ok = []
        with torch.no_grad():
            for t in rows:
                h = torch.randn(1, t, width, generator=gen).to(device=dev, dtype=torch.bfloat16)
                r = (torch.randn(1, t, width, generator=gen) * 4).to(device=dev, dtype=torch.bfloat16)
                try:
                    want = mod.mlp(h)
                    want = r + (want[0] if isinstance(want, tuple) else want)
                    got = _moe_with_residual(mod.mlp, h, r, width)
                except Exception as e:  # noqa: BLE001 -- a probe that raises refuses the licence; it never stops a build
                    errors.append(f"{type(mod).__name__} at {t} rows: {type(e).__name__}: {e}"[:240])
                    break
                if got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want):
                    ok.append(t)
        if ok:
            mod._e4b_moe_resid = frozenset(ok)
        else:
            mod.__dict__.pop("_e4b_moe_resid", None)
        full += len(ok) == len(rows)
        partial += 0 < len(ok) < len(rows)
        refused += not ok
    _note(report, licensed=full, partial=partial, refused=refused, probe_errors=errors[:8])
    if mode == "1" and full == 0:
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE_R2=1: no folded layer's MoE residual composition was bitwise its own body at every "
            f"decode row count {rows} -- refusing a vacuous licence")
    return full


def _patch_attention(mod, rope_norm_heads, rope_norm_qk=None) -> bool:
    """Fold each of q_norm/k_norm plus rotary into one launch.

    ``rope_norm_qk`` (grouped-nf4-gemm#528, or None): q's and k's heads in one launch, bitwise the two
    ``rope_norm_heads`` calls it replaces (e4b#1313, lane P127's Phase 2, item d)."""
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
    if not rotary_is_rotate_half(mod, int(mod.head_dim)):
        return False
    orig = mod.forward

    def _fwd(hidden_states, position_embeddings=None,
             attention_mask=None, past_key_values=None, _m=mod,
             _qn=qn, _kn=kn, _qe=qe, _ke=ke, _orig=orig, _qk=rope_norm_qk, **kwargs):
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
        # norm + rotary: one launch for both projections where the kernel has it, else one per projection
        if _qk is not None:
            qo, ko = _qk(q.reshape(rows, -1, d), k.reshape(rows, -1, d), _qn.weight, _kn.weight, cos2, sin2, _qe, _ke)
        else:
            qo = rope_norm_heads(q.reshape(rows, -1, d), _qn.weight, cos2, sin2, _qe)
            ko = rope_norm_heads(k.reshape(rows, -1, d), _kn.weight, cos2, sin2, _ke)
        query_states = qo.reshape(*input_shape, -1, d).transpose(1, 2)
        key_states = ko.reshape(*input_shape, -1, d).transpose(1, 2)
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


def _patch_attention_unfused(mod, rope_norm_heads, rope_norm_qk=None) -> bool:
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
    if not rotary_is_rotate_half(mod, d):
        return False
    orig = mod.forward

    def _fwd(hidden_states, position_embeddings=None,
             attention_mask=None, past_key_values=None, _m=mod,
             _qn=qn, _kn=kn, _qe=qe, _ke=ke, _orig=orig, _d=d, _qk=rope_norm_qk, **kwargs):
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
        if _qk is not None:             # q's and k's heads in one launch (grouped-nf4-gemm#528)
            qo, ko = _qk(q.reshape(rows, -1, _d), k.reshape(rows, -1, _d), _qn.weight, _kn.weight, cos2, sin2, _qe,
                         _ke)
        else:
            qo = rope_norm_heads(q.reshape(rows, -1, _d), _qn.weight, cos2, sin2, _qe)
            ko = rope_norm_heads(k.reshape(rows, -1, _d), _kn.weight, cos2, sin2, _ke)
        query_states = qo.reshape(*input_shape, -1, _d).transpose(1, 2)
        key_states = ko.reshape(*input_shape, -1, _d).transpose(1, 2)
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


_NONORM_ATTN_CHILDREN = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


def _patch_attention_rope_only(mod, int4_b32) -> bool:
    """The rotary chain folded for attention WITHOUT a head norm (the
    Llama-shaped q/k/v/o module GraniteMoe and Mixtral use): exactly the
    four projections, nothing of the module's own, the usual attributes,
    a rotate-half rotary (:func:`rotary_is_rotate_half`; ERNIE-4.5 has the
    structure but an interleaved rotary), and the kernel side's
    ``rope_heads``; refuses loudly on a kernel cut without it. gpt-oss's
    attention carries ``sinks`` (a parameter of its own) and is refused by
    the structure rule."""
    children = {n for n, _ in mod.named_children()}
    if children != _NONORM_ATTN_CHILDREN:
        return False
    if any(True for _ in mod.named_parameters(recurse=False)):
        return False
    if any(True for _ in mod.named_buffers(recurse=False)):
        return False
    # No ``sliding_window`` in this list: GraniteMoe's and Mixtral's
    # attention modules do not set it (the Qwen3-shaped license above
    # requires it; copying that here skipped both families -- Bugbot,
    # e4b#379). It is read with a default below.
    for attr in ("head_dim", "scaling", "layer_idx", "config",
                 "attention_dropout"):
        if not hasattr(mod, attr):
            return False
    d = int(mod.head_dim)
    if d % 2:
        return False
    if not rotary_is_rotate_half(mod, d):
        return False            # e.g. ERNIE-4.5: q/k/v/o exactly, interleaved rotary (e4b#1362)
    rope_heads = getattr(int4_b32, "rope_heads", None)
    if rope_heads is None:
        raise _KernelGap(
            "E4B_FUSE_T1_GLUE_R2=1 on a norm-less attention "
            f"({type(mod).__name__}) needs the kernel side's rope_heads "
            "(grouped-nf4-gemm >= 0.28); install the matching cut or "
            "unset the flag")
    orig = mod.forward

    def _fwd(hidden_states, position_embeddings=None,
             attention_mask=None, past_key_values=None, _m=mod,
             _orig=orig, _d=d, **kwargs):
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
        query_states = rope_heads(q.reshape(rows, -1, _d), cos2, sin2
                                  ).reshape(*input_shape, -1, _d).transpose(1, 2)
        key_states = rope_heads(k.reshape(rows, -1, _d), cos2, sin2
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
            sliding_window=getattr(_m, "sliding_window", None), **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return _m.o_proj(attn_output), attn_weights

    mod.forward = _fwd
    return True


def fuse_t1_glue_r2(model, mode: str | None = None, report: dict | None = None) -> tuple[int, int]:
    """Apply the round-2 decode folds. Returns ``(layers, attentions)``.

    ``mode`` (:func:`~.glue_fuse.fold_mode`; ``None`` reads ``E4B_FUSE_T1_GLUE_R2``): under ``1`` it refuses a
    vacuous enable or a kernel cut that lacks what a matched structure needs -- an arm that asks for the fusion must
    get it or an error, never a quiet no-op. Under ``auto`` those modules stay unpatched and ``report`` (a dict, when
    given) counts them."""
    mode = fold_mode("E4B_FUSE_T1_GLUE_R2", mode)
    _note(report, mode=mode)
    if mode == "0":
        return (0, 0)
    try:
        import int4_b32  # the module object is needed for the capability probe
        from int4_b32 import rmsnorm_resid_rows, rope_norm_heads
    except ImportError as e:
        if mode == "auto":
            _note(report, skipped=f"no kernel: {e}")
            return (0, 0)
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE_R2=1 needs the kernel side's "
            "rmsnorm_resid_rows/rope_norm_heads; install the matching "
            "cut or unset the flag") from e

    layers = attns = gaps = 0
    for mod in model.modules():
        name = type(mod).__name__
        try:
            if name.endswith("DecoderLayer"):
                scale = _layer_scale(mod)
                if scale is None:
                    layers += bool(_patch_layer(mod, rmsnorm_resid_rows))
                else:
                    layers += bool(_patch_layer_scaled(mod, scale, int4_b32))
            elif name.endswith("Attention"):
                qk = getattr(int4_b32, "rope_norm_qk", None)
                attns += bool(_patch_attention(mod, rope_norm_heads, qk)
                              or _patch_attention_unfused(mod, rope_norm_heads, qk)
                              or _patch_attention_rope_only(mod, int4_b32))
        except _KernelGap:
            if mode == "1":
                raise
            gaps += 1                       # auto: this module keeps its own forward
    _note(report, patched=[layers, attns], kernel_gaps=gaps)
    if layers == 0 and attns == 0 and mode == "1":
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE_R2=1 patched nothing (no structurally "
            "matched decoder layer or fused attention passed the "
            "probes) -- refusing a vacuous enable")
    return (layers, attns)
