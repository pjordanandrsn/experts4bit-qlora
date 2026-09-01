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


def _probe_matches(mod, eps: float) -> bool:
    """The module's OWN forward licenses the patch: centered variants
    (``x_norm * (1 + weight)`` with a near-zero stored weight) share the
    RMSNorm name but a different formula, and patching one would nearly
    zero the residual stream (review finding, High). A deterministic
    probe through the module, compared against the non-centered
    reference, accepts exactly the semantics the fused kernel computes
    -- name matching alone cannot."""
    w = mod.weight
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(2, w.numel(), generator=g).to(w.device, torch.bfloat16)
    try:
        with torch.no_grad():
            got = mod(x)
    except Exception:
        return False
    if not torch.is_tensor(got) or got.shape != x.shape:
        return False
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
           * w.float())
    return torch.allclose(got.float(), ref, rtol=2 ** -5, atol=2 ** -7)


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
    skipped = 0
    for mod in model.modules():
        if not _is_rmsnorm(mod):
            continue
        eps = _norm_eps(mod)
        if not _probe_matches(mod, eps):
            skipped += 1        # centered or otherwise non-matching
            continue
        orig = mod.forward

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
            f"E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules "
            f"({skipped} name-matched but failed the semantic probe) -- "
            "refusing a vacuous enable")
    return n
