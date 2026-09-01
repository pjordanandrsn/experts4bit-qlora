# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Opt-in fused T=1 routing (``E4B_FUSE_T1_ROUTER=1``).

The upstream sparse-MoE block computes routing as a small linear plus a
softmax/topk/renorm chain -- ~6-8 launches per layer that the B=1
census prices at ~0.45 ms/step. The kernel side ships the whole
decision as one launch; this module patches router modules to use it at
T=1 and fall through to the original forward everywhere else.

Routers are matched STRUCTURALLY (weight [E, H], integer ``top_k``,
boolean ``norm_topk_prob``, hidden_dim, and a forward returning the
(logits, scores, indices) triple), never by importing an upstream
class -- the activation-registry lesson: identity checks against
upstream objects rot silently.

Parity frame: fp32 dot rounding differs from the dense linear, so
near-tie index differences are the accepted reorder class (per-arm
determinism + the K8 quality gate adjudicate, as for every kernel-swap
treatment). Engagement is census PRESENCE of ``_router_topk_t1``.
"""
from __future__ import annotations

import os

import torch

__all__ = ["fuse_t1_router"]


def _is_router(mod) -> bool:
    w = getattr(mod, "weight", None)
    return (torch.is_tensor(w) and w.dim() == 2
            and isinstance(getattr(mod, "top_k", None), int)
            and isinstance(getattr(mod, "norm_topk_prob", None), bool)
            and getattr(mod, "hidden_dim", None) == w.shape[1])


def fuse_t1_router(model) -> int:
    """Patch every structurally-matched router for fused T=1 decisions.

    Returns the number of routers patched. Refuses loudly when the
    kernel is missing and the env asks for the fusion; a zero match
    with the env set is also an error -- a fusion that silently
    patches nothing is the failure mode the census kept catching.
    """
    if os.environ.get("E4B_FUSE_T1_ROUTER", "0") != "1":
        return 0
    try:
        from int4_b32 import router_topk_t1
    except ImportError as e:
        raise RuntimeError(
            "E4B_FUSE_T1_ROUTER=1 needs the kernel side's router_topk_t1; "
            "install the matching cut or unset the flag") from e

    n = 0
    for mod in model.modules():
        if not _is_router(mod):
            continue
        orig = mod.forward

        def _fwd(hidden_states, _m=mod, _orig=orig):
            hs = hidden_states.reshape(-1, _m.hidden_dim)
            if hs.shape[0] != 1 or _m.weight.dtype != torch.bfloat16:
                return _orig(hidden_states)
            return router_topk_t1(hs, _m.weight, _m.top_k,
                                  _m.norm_topk_prob)

        mod.forward = _fwd
        n += 1
    if n == 0:
        raise RuntimeError(
            "E4B_FUSE_T1_ROUTER=1 matched no router modules -- refusing a "
            "vacuous enable (structural match: weight [E, H], int top_k, "
            "bool norm_topk_prob, hidden_dim == H)")
    return n
