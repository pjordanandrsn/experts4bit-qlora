# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Paged FP8 attention as a transformers attention *implementation* —
hybrid Stage 2, Phase 9.

Modern transformers dispatches attention through
``ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation)``,
calling it with post-RoPE ``query/key/value`` and returning
``[B, T, H_q, D]``. Registering there rather than patching each
architecture's ``forward`` is the difference between one seam and one per
model family — the same reasoning that put the Phase-1 router patch on
the router modules instead of the blocks.

The Cache protocol is deliberately NOT used. Its contract is "return the
full K/V for attention to consume", which over paged storage means
gathering every slot's blocks into a padded ``[B, H, T_max, D]`` buffer
on every layer of every step: more expensive than the paging saves, and
a materialization of exactly the KV Phase 7 keeps packed. Run the model
with ``use_cache=False`` and this function owns the KV instead.

Two regimes, split by what the fused kernel can do:

* **Decode** (one query token per sequence — what the kernel scores):
  append K/V to the pool, then read it in place through the block
  tables. No gather, no dequantized KV.
* **Prefill** (many query tokens, compute-bound): the chunk's own K/V
  are already in hand as bf16. Attention runs over a per-sequence
  **staging buffer** that accumulates the prompt's K/V as chunks arrive,
  and the buffer is quantized into the pool once — when the prompt
  completes. This is a staged WRITE, not a dequant round trip: nothing
  is quantized and then read back to compute on. The cost is honest and
  bounded — one prompt's bf16 KV per sequence that is *currently*
  prefilling, freed at completion — and it is what buys true chunking,
  since a later chunk must attend to earlier ones.

The staging buffer is the reason chunk size is a real knob rather than a
free parameter: bigger chunks mean fewer, larger attentions and more
resident bf16; smaller chunks mean tighter TTFT control. G9 sweeps it
rather than picking one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PagedAttentionContext:
    """Per-step binding: which pool, which slots, which regime.

    ``slots[b]`` is the KV slot of batch row ``b``. Set by the runner
    around each forward; when unset the attention implementation falls
    back to stock SDPA so an un-bound forward behaves normally instead of
    reading someone else's KV.
    """
    kv: object                                  # Fp8PagedKV
    slots: list[int]
    mode: str = "decode"                        # "decode" | "prefill"
    # staging[(layer, slot)] -> [k_list, v_list] of [T, H, D] bf16 chunks
    staging: dict = field(default_factory=dict)

    def stage(self, layer: int, slot: int, k, v):
        buf = self.staging.setdefault((layer, slot), ([], []))
        buf[0].append(k)
        buf[1].append(v)
        return (torch.cat(buf[0]), torch.cat(buf[1]))

    def flush(self, layer: int, slot: int):
        """Quantize a completed prompt's staged K/V into the pool."""
        buf = self.staging.pop((layer, slot), None)
        if buf is None:
            return None
        return torch.cat(buf[0]), torch.cat(buf[1])

    def drop(self, slot: int):
        for key in [k for k in self.staging if k[1] == slot]:
            self.staging.pop(key, None)


_CTX: PagedAttentionContext | None = None
IMPL_NAME = "e4b_paged"


def set_context(ctx: PagedAttentionContext | None):
    """Bind (or unbind) the paged context for the forwards that follow."""
    global _CTX
    prev, _CTX = _CTX, ctx
    return prev


def current_context() -> PagedAttentionContext | None:
    return _CTX


def _sdpa(module, q, k, v, attention_mask, dropout, scaling, is_causal,
          **kw):
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward)
    return sdpa_attention_forward(module, q, k, v, attention_mask,
                                  dropout=dropout, scaling=scaling,
                                  is_causal=is_causal, **kw)


def _window_of(module, kwargs) -> int:
    """The layer's sliding window in tokens, 0 for full attention. gpt-oss
    passes ``sliding_window`` through the interface; Gemma-4 keeps it on
    the module (``module.sliding_window`` on sliding layers, None on full
    ones). The paged kernel skips tiles below the window and masks the
    boundary tile; the prefill branch masks explicitly."""
    w = kwargs.get("sliding_window")
    if w is None:
        w = getattr(module, "sliding_window", None)
    return int(w) if w else 0


def _sinks_of(module, kwargs):
    """gpt-oss attention sinks: one learned logit per q head (``s_aux``),
    in the softmax max and denominator, never the numerator."""
    s_aux = kwargs.get("s_aux")
    if s_aux is None:
        s_aux = getattr(module, "sinks", None)
    return s_aux


def _prefill_attend(q_b, kk, vv, mask, scaling, sinks):
    """Prefill attention for one sequence. Without sinks this is SDPA with
    the explicit (causal, windowed) mask. A sink cannot be expressed as a
    key (its logit does not depend on q), so the sink path computes the
    scores directly, appends the sink column, softmaxes, drops it."""
    if sinks is None:
        return torch.nn.functional.scaled_dot_product_attention(
            q_b, kk, vv, attn_mask=mask, scale=scaling, enable_gqa=True)
    hq, hkv = q_b.shape[1], kk.shape[1]
    g = hq // hkv
    kk = kk.repeat_interleave(g, dim=1) if g > 1 else kk
    vv = vv.repeat_interleave(g, dim=1) if g > 1 else vv
    scale = scaling if scaling is not None else q_b.shape[-1] ** -0.5
    att = torch.matmul(q_b.float(), kk.float().transpose(-1, -2)) * scale
    att = att.masked_fill(~mask, float("-inf"))
    sk = sinks.float().to(att.device).view(1, hq, 1, 1).expand(
        att.shape[0], hq, att.shape[2], 1)
    probs = torch.softmax(torch.cat([att, sk], dim=-1), dim=-1)[..., :-1]
    return torch.matmul(probs.to(vv.dtype), vv)


def paged_attention_forward(module, query, key, value, attention_mask,
                            dropout: float = 0.0,
                            scaling: float | None = None,
                            is_causal: bool | None = None, **kwargs):
    """``[B, H_q, T, D]`` in, ``[B, T, H_q, D]`` out."""
    ctx = _CTX
    if ctx is None:
        return _sdpa(module, query, key, value, attention_mask, dropout,
                     scaling, is_causal, **kwargs)

    layer = int(getattr(module, "layer_idx", 0))
    B, hq, T, D = query.shape
    if len(ctx.slots) != B:
        raise ValueError(f"paged context binds {len(ctx.slots)} slots but "
                         f"the batch is {B} — a mismatch here would attend "
                         f"one sequence over another's KV")

    if ctx.mode == "decode":
        if T != 1:
            raise ValueError(f"decode regime expects one query token per "
                             f"sequence, got {T}")
        if getattr(ctx.kv, "graph_t1", False):
            # Graph mode: one token per slot, device-computed write
            # addresses -- the only append shapes a CUDA graph may
            # capture (PREREG-b1d item 2b; PREREG-bv3 for B>1). The
            # batch form requires the slot list registered at
            # graph_mode_init_batch to MATCH ctx.slots -- silently
            # appending rows to the init-time slots while attention
            # reads ctx.slots would cross sequences.
            if key.shape[0] > 1 or len(ctx.slots) > 1:
                gslots = getattr(ctx.kv, "_g_slots", None)
                if gslots != list(ctx.slots):
                    raise RuntimeError(
                        f"graph batch append: ctx.slots {list(ctx.slots)} "
                        f"!= graph_mode_init_batch slots {gslots}")
                ctx.kv.append_graph_bt1(
                    layer, key.permute(0, 2, 1, 3).reshape(
                        key.shape[0], -1, key.shape[-1]).contiguous(),
                    value.permute(0, 2, 1, 3).reshape(
                        value.shape[0], -1, value.shape[-1]).contiguous())
            else:
                ctx.kv.append_graph_t1(
                    layer, key[0].permute(1, 0, 2).contiguous(),
                    value[0].permute(1, 0, 2).contiguous())
        elif getattr(ctx.kv, "batched_append", False):
            # one permute+quantize per side for the whole batch
            # (PREREG-g9-kvappend); bit-identical to the loop below
            ctx.kv.append_many(layer, ctx.slots,
                                key.permute(0, 2, 1, 3).contiguous(),
                                value.permute(0, 2, 1, 3).contiguous())
        else:
            for b, slot in enumerate(ctx.slots):
                ctx.kv.append(layer, slot,
                              key[b].permute(1, 0, 2).contiguous(),
                              value[b].permute(1, 0, 2).contiguous())
        # the family's attention scale rides through to the decode kernel:
        # GraniteMoe's attention_multiplier (0.0156 vs head_dim**-0.5 =
        # 0.125) and Gemma-4's 1.0 (folded into q_norm) served garbage
        # while this branch passed only q/k/v/slots (receipts P24-GEN-B)
        out = ctx.kv.attention(layer, query[:, :, 0].contiguous(),
                               slots=ctx.slots, sm_scale=scaling,
                               window=_window_of(module, kwargs),
                               sinks=_sinks_of(module, kwargs))
        return out[:, None].to(query.dtype), None

    if ctx.mode == "verify":
        # PREREG-s2lite: speculative verification. ONE sequence, T = K+1
        # query rows. The window's K/V are appended first (same shim
        # ordering as decode), then every row reads through the paged
        # kernel with a STAGGERED length: row i attends over the past
        # plus draft tokens 0..i. Causality comes from lengths over
        # already-appended K/V -- no mask, no new kernel. The rejected
        # tail is made unreadable afterward by kv.rewind(), which the
        # EXECUTOR owns (the shim appends; it must not also rewind,
        # or a multi-layer forward would rewind 47 times mid-step).
        if B != 1:
            raise ValueError(f"verify regime is single-sequence, got "
                             f"batch {B}")
        slot = ctx.slots[0]
        ctx.kv.append(layer, slot,
                      key[0].permute(1, 0, 2).contiguous(),
                      value[0].permute(1, 0, 2).contiguous())
        # after append, seq_lens[slot] = base + T; row i must read
        # base + i + 1 of it
        base_plus_t = ctx.kv.seq_lens[layer].narrow(0, slot, 1)
        stagger = torch.arange(1 - T, 1, dtype=torch.int32,
                               device=query.device)
        lens_override = (base_plus_t + stagger).contiguous()
        q_rows = query[0].permute(1, 0, 2).contiguous()   # [T, H_q, D]
        out = ctx.kv.attention(layer, q_rows, slots=[slot] * T,
                               lens_override=lens_override, sm_scale=scaling,
                               window=_window_of(module, kwargs),
                               sinks=_sinks_of(module, kwargs))
        # [T, H_q, D] -> [1, T, H_q, D]
        return out[None].to(query.dtype), None

    # ---- prefill: causal attention over this sequence's staged context
    outs = []
    for b, slot in enumerate(ctx.slots):
        k_all, v_all = ctx.stage(layer, slot,
                                 key[b].permute(1, 0, 2),
                                 value[b].permute(1, 0, 2))
        # [T_total, H, D] -> [1, H, T_total, D]
        kk = k_all.permute(1, 0, 2)[None]
        vv = v_all.permute(1, 0, 2)[None]
        q_b = query[b:b + 1]
        t_total = kk.shape[2]
        # the chunk's queries are the LAST T positions of the sequence;
        # each attends to everything up to and including itself
        pos = torch.arange(t_total - T, t_total, device=query.device)
        keys = torch.arange(t_total, device=query.device)
        mask = keys[None, :] <= pos[:, None]
        win = _window_of(module, kwargs)
        if win:
            # the last `win` keys of each query's past, itself included
            mask = mask & (keys[None, :] > pos[:, None] - win)
        mask = mask[None, None]
        o = _prefill_attend(q_b, kk, vv, mask, scaling,
                            _sinks_of(module, kwargs))
        outs.append(o)
    out = torch.cat(outs, dim=0).transpose(1, 2).contiguous()
    return out.to(query.dtype), None


def register(model=None) -> str:
    """Register the implementation and, given a model, select it."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = paged_attention_forward
    if model is not None:
        model.set_attn_implementation(IMPL_NAME) if hasattr(
            model, "set_attn_implementation") else None
        model.config._attn_implementation = IMPL_NAME
        for mod in model.modules():
            cfg = getattr(mod, "config", None)
            if cfg is not None:
                cfg._attn_implementation = IMPL_NAME
    return IMPL_NAME
