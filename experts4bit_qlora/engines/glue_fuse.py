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


def _probe_variant(mod, eps: float):
    """Which frozen-RMSNorm formula the module's OWN forward computes, or None (training fusion, ``rmsnorm_train``).

    Returns ``(offset, mul_fp32)``: ``offset`` 0.0 for ``w * norm(x)`` or 1.0 for the centered ``(1 + w) * norm(x)``
    (Qwen3.5/3.6, whose stored weight is near zero -- patching it as ``w * norm(x)`` would nearly zero the residual stream);
    ``mul_fp32`` False for the Llama rounding (``norm(x)`` rounded to the input dtype, then multiplied by the weight in that
    dtype) or True for the multiply in fp32, rounded once (Gemma-4, Qwen3.5/3.6). The probe's weight is the module's own
    perturbed by a fixed pattern, so a plain weight near one and a centered weight near zero cannot be confused.

    The candidate that reproduces the module's output most closely wins, and only if it is inside the tolerance the decode
    matcher (:func:`_probe_matches`) has always used; an exact candidate wins outright. Closest rather than exact-only
    because a composite may compute the statistic differently (Gemma-4 takes ``pow(ms, -0.5)``, not ``rsqrt``) -- the formula
    is what the probe identifies, and a formula with a different weight convention is orders of magnitude outside it."""
    w = mod.weight
    if w.dtype not in (torch.bfloat16, torch.float16):
        return None
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(4, w.numel(), generator=g).to(w.device, w.dtype)
    pert = (torch.randn(w.numel(), generator=g) * 0.25).to(w.device, w.dtype)
    saved = w.detach().clone()
    try:
        with torch.no_grad():
            w.data.add_(pert)
            weff = w.detach().clone()
            got = mod(x)
    except Exception:
        return None
    finally:
        with torch.no_grad():
            w.data.copy_(saved)
    if not torch.is_tensor(got) or got.shape != x.shape or got.dtype != x.dtype:
        return None
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    best = None
    for offset in (0.0, 1.0):
        for mul_fp32 in (False, True):
            if offset == 0.0 and not mul_fp32:
                ref = weff * n.to(x.dtype)
            elif offset == 0.0:
                ref = (n * weff.float()).to(x.dtype)
            elif not mul_fp32:
                ref = (1.0 + weff.float()).to(x.dtype) * n.to(x.dtype)
            else:
                ref = (n * (1.0 + weff.float())).to(x.dtype)
            if torch.equal(got, ref):
                return offset, mul_fp32
            if not torch.allclose(got.float(), ref.float(), rtol=2 ** -5, atol=2 ** -7):
                continue
            err = (got.float() - ref.float()).abs().sum().item()
            if best is None or err < best[0]:
                best = (err, (offset, mul_fp32))
    return None if best is None else best[1]


FOLD_MODES = ("auto", "0", "1")


def fold_mode(name: str, value: str | None = None) -> str:
    """One decode-fusion knob's mode (``E4B_FUSE_T1_GLUE``, ``E4B_FUSE_T1_GLUE_R2``, ``E4B_FUSE_ROUTER_EPI``,
    ``E4B_PAGED_FUSE_QKV``):

    * ``0`` -- off (also unset or empty: the library's default);
    * ``1`` -- apply, and refuse a missing kernel or a vacuous enable (the lanes' rule);
    * ``auto`` -- apply where the module structure and the installed kernels license it, and patch nothing,
      without raising, where they do not (another family, an older kernel cut). ``serve_paged`` is the caller that
      passes it.

    ``value`` overrides the environment (``serve_paged`` passes its parsed config). Anything else is refused rather
    than read as one of these."""
    raw = os.environ.get(name, "") if value is None else value
    v = (raw or "").strip().lower() or "0"
    if v in FOLD_MODES:
        return v
    raise ValueError(f"{name}={raw!r}: expected 'auto', '0' or '1'")


def _note(report, **kw) -> None:
    if report is not None:
        report.update(kw)


def fuse_t1_glue(model, mode: str | None = None, report: dict | None = None) -> int:
    """Patch every structurally-matched RMSNorm for fused decode calls.

    Returns the number of norms patched. ``mode`` (:func:`fold_mode`; ``None`` reads ``E4B_FUSE_T1_GLUE``): ``1``
    refuses loudly on a missing kernel or a zero-match enable; ``auto`` returns 0 there instead and says why in
    ``report`` (a dict, when given)."""
    mode = fold_mode("E4B_FUSE_T1_GLUE", mode)
    _note(report, mode=mode)
    if mode == "0":
        return 0
    try:
        from int4_b32 import rmsnorm_rows
    except ImportError as e:
        if mode == "auto":
            _note(report, skipped=f"no kernel: {e}")
            return 0
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
    _note(report, patched=n, failed_probe=skipped)
    if n == 0 and mode == "1":
        raise RuntimeError(
            f"E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules "
            f"({skipped} name-matched but failed the semantic probe) -- "
            "refusing a vacuous enable")
    return n
