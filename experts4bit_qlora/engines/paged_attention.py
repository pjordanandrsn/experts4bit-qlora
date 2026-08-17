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
        for b, slot in enumerate(ctx.slots):
            ctx.kv.append(layer, slot,
                          key[b].permute(1, 0, 2).contiguous(),
                          value[b].permute(1, 0, 2).contiguous())
        out = ctx.kv.attention(layer, query[:, :, 0].contiguous())
        return out[:, None].to(query.dtype), None

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
        mask = (keys[None, :] <= pos[:, None])[None, None]
        o = torch.nn.functional.scaled_dot_product_attention(
            q_b, kk, vv, attn_mask=mask, scale=scaling, enable_gqa=True)
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
