# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Opt-in fused RMSNorm for decode (``E4B_FUSE_T1_GLUE=1``).

The B=1 census's elementwise/reduce soup includes 3-4 kernels per
RMSNorm call at four sites per layer (input and post-attention layer
norms, plus the per-head q/k norms). The kernel side ships the whole
call as one launch; this module patches norm modules to use it and
falls through to the original forward off the decode shapes.

Norms are matched STRUCTURALLY (a 1-D ``weight`` and a float epsilon
attribute under either upstream spelling), never by importing an
upstream class -- the activation-registry lesson. Engagement is census
PRESENCE of ``_rmsnorm_rows``.
"""
from __future__ import annotations

import os

import torch

__all__ = ["fuse_t1_glue"]

_EPS_ATTRS = ("variance_epsilon", "eps")


def _norm_eps(mod):
    for a in _EPS_ATTRS:
        v = getattr(mod, a, None)
        if isinstance(v, float):
            return v
    return None


def _is_rmsnorm(mod) -> bool:
    w = getattr(mod, "weight", None)
    return (torch.is_tensor(w) and w.dim() == 1
            and _norm_eps(mod) is not None
            and type(mod).__name__.endswith("RMSNorm"))


def fuse_t1_glue(model) -> int:
    """Patch every structurally-matched RMSNorm for fused decode calls.

    Returns the number of norms patched; refuses loudly on a missing
    kernel or a zero-match enable."""
    if os.environ.get("E4B_FUSE_T1_GLUE", "0") != "1":
        return 0
    try:
        from int4_b32 import rmsnorm_rows
    except ImportError as e:
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE=1 needs the kernel side's rmsnorm_rows; "
            "install the matching cut or unset the flag") from e

    n = 0
    for mod in model.modules():
        if not _is_rmsnorm(mod):
            continue
        orig = mod.forward
        eps = _norm_eps(mod)

        def _fwd(hidden_states, _m=mod, _orig=orig, _eps=eps):
            # decode shapes only: few rows, bf16, last-dim matches the
            # weight. Prefill and exotic dtypes keep the original chain.
            if (hidden_states.dtype != torch.bfloat16
                    or hidden_states.shape[-1] != _m.weight.numel()
                    or hidden_states.numel()
                    > 64 * hidden_states.shape[-1]):
                return _orig(hidden_states)
            return rmsnorm_rows(hidden_states, _m.weight, _eps)

        mod.forward = _fwd
        n += 1
    if n == 0:
        raise RuntimeError(
            "E4B_FUSE_T1_GLUE=1 matched no RMSNorm modules -- refusing a "
            "vacuous enable")
    return n
