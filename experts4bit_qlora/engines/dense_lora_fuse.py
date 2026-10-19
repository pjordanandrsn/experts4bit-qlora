# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Fused LoRA adapters for a DENSE MLP in training (DQ11's MLP-adapter BUILD, ``bench/dq11/DQ11-MLP-BUILD-PREREG.md``;
off by default, ``E4B_DENSE_MLP_LORA_FUSE=1`` turns it on).

A PEFT LoRA module around a bitsandbytes NF4 base pays, per projection and per call:
- a defensive clone of the base output;
- a cast of the input to the adapters' dtype;
- the A matmul, the B matmul and the scale;
- a cast back and an add.

In a SwiGLU MLP, ``gate_proj`` and ``up_proj`` read the same input, so they pay the cast and the A matmul twice. This
module computes each MLP's adapters through one autograd Function per input:

- **gate and up together:** one FP32 cast of the input, one rank matmul against their A matrices stacked (``[2r, K]``),
  then for each one an in-place ``addmm_`` of its B matmul into an FP32 copy of its base output (the scale as
  ``alpha``) and one cast back;
- **down alone:** the same, for one adapter.

The adapters' rank compute stays in their own dtype (FP32), so the precision is unchanged. The order of the FP32 sums
is not, so the outputs and gradients equal PEFT's to rounding, not bit for bit. The registration's Phase 1 holds each
tensor to the unfused path's neutral floor.

Gradients come from the Function's own backward:
- B: ``s · gᵀ·t``;
- the stacked A: ``(s · g·B)ᵀ·x``, split back per adapter;
- the input: one matmul of the stacked rank gradient against the stacked A;
- the base outputs: the incoming gradient, unchanged.

Checkpoint recompute re-runs the same Function. The A and B tensors stay the trainable parameters, on their own PEFT
modules and under their own names; nothing is copied or merged.

Like P129's ``train_qkv_fuse`` (which patches the attention module), this patches the MLP module's forward, not each
LoRA module's: the shared gate/up matmul needs both projections in one call. The fused forward is the class's own,
``down(act(gate(x)) * up(x))``, with each projection's adapter added through the Function, so only classes whose forward
is exactly that are taken (:data:`FUSED_MLP_CLASSES`). A fused MLP runs its class's own forward, PEFT's path, for any call
the fusion would get wrong, counted in ``DENSE_MLP_LORA_STATS["fallback"]``:
- under autocast, which the Function's FP32 rank compute would not match;
- when a projection's PEFT state changed after fusing: adapters disabled (PEFT's ``disable_adapter()`` context), another
  adapter set, an adapter merged, or the input cast turned off.

Only an MLP of those classes whose three projections are vanilla PEFT LoRA modules on an NF4 base is fused. Anything
else with ``gate_proj``/``up_proj``/``down_proj`` keeps its own path, with the reason in
``DENSE_MLP_LORA_STATS["refused"]``:
- a class not in :data:`FUSED_MLP_CLASSES`, or a forward already patched by something else;
- a hook on a projection or its ``lora_A`` / ``lora_B`` (the fused forward does not call those modules);
- dropout other than 0;
- DoRA or another LoRA variant;
- more than one active adapter, or adapters disabled;
- a merged adapter, or PEFT's input cast turned off;
- adapters that are not FP32, or a rank or scale that differs within the MLP;
- a base that is not a bitsandbytes NF4 ``Linear4bit``, or has a bias.

:func:`disable_dense_mlp_lora_fuse` puts every fused MLP's own forward back.
"""
from __future__ import annotations

import os
import types

import torch
import torch.nn as nn

#: What :func:`enable_dense_mlp_lora_fuse` did: MLPs fused, the adapters they hold, each refused MLP's reason, and fused
#: forward calls (``fallback``: calls that ran the class's own forward instead). ``calls`` counts checkpoint recomputes
#: too: under non-reentrant checkpointing it is about twice a step's MLP forwards.
DENSE_MLP_LORA_STATS = {"fused": 0, "adapters": 0, "refused": {}, "calls": 0, "fallback": 0}

#: ``E4B_DENSE_MLP_LORA_FUSE`` values that turn the fusion on; anything else, unset included, keeps PEFT's path.
DENSE_MLP_LORA_FUSE_ON = frozenset({"1", "true", "on", "yes"})

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

#: The MLP classes this module fuses (its refusal rule): each one's forward is ``down(act(gate(x)) * up(x))``. The
#: registration's fixture is Mistral-7B; another class joins only with its own check.
FUSED_MLP_CLASSES = frozenset({"MistralMLP"})


def dense_mlp_lora_fuse_requested() -> bool:
    """On only when ``E4B_DENSE_MLP_LORA_FUSE`` is ``1`` (or ``true`` / ``on`` / ``yes``); unset is off."""
    return os.environ.get("E4B_DENSE_MLP_LORA_FUSE", "").strip().lower() in DENSE_MLP_LORA_FUSE_ON


def _is_nf4_linear4bit(base) -> bool:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    if not isinstance(base, bnb.nn.Linear4bit):
        return False
    qs = getattr(base.weight, "quant_state", None)
    return qs is not None and getattr(qs, "quant_type", None) == "nf4"


def _hooked(mod) -> bool:
    return bool(mod._forward_hooks or mod._forward_pre_hooks or mod._backward_hooks or mod._backward_pre_hooks)


def _active(proj) -> list:
    return [a for a in proj.active_adapters if a in proj.lora_A.keys()]


def _state_ok(proj) -> bool:
    """The PEFT state a fused projection needs, re-read on every call: adapters on, unmerged, PEFT's input cast on."""
    return not (getattr(proj, "disable_adapters", False) or getattr(proj, "merged", False)
                or not getattr(proj, "cast_input_dtype_enabled", True))


def _adapter(proj, base_ok):
    """``(A, B, scale, name)`` of a vanilla single-adapter PEFT LoRA module (A and B its ``nn.Linear`` modules, ``name``
    the active adapter's), or the reason it cannot be fused (a ``str``)."""
    for attr in ("base_layer", "lora_A", "lora_B", "scaling", "lora_dropout", "active_adapters"):
        if not hasattr(proj, attr):
            return f"not a PEFT LoRA module (no {attr!r})"
    if getattr(proj, "disable_adapters", False):
        return "its adapters are disabled"
    if getattr(proj, "merged", False):
        return "its adapter is merged"
    if not getattr(proj, "cast_input_dtype_enabled", True):
        return "PEFT's input cast is disabled"
    active = _active(proj)
    if len(active) != 1:
        return f"{len(active)} active adapters, not one"
    name = active[0]
    if (getattr(proj, "use_dora", {}) or {}).get(name) or name in (getattr(proj, "lora_variant", {}) or {}):
        return "a DoRA or other LoRA variant"
    if not isinstance(proj.lora_dropout[name], nn.Identity):
        return "LoRA dropout is not 0"
    lora_a, lora_b = proj.lora_A[name], proj.lora_B[name]
    if not (type(lora_a) is nn.Linear and type(lora_b) is nn.Linear) or lora_a.bias is not None or lora_b.bias is not None:
        return "the adapter is not two bias-free nn.Linear"
    if any(_hooked(m) for m in (proj, lora_a, lora_b)):
        return "it or its adapter has a hook, which the fused forward would not run"
    if lora_a.weight.dtype != torch.float32 or lora_b.weight.dtype != torch.float32:
        return "the adapter is not FP32"
    base = proj.base_layer
    if getattr(base, "bias", None) is not None:
        return "the base has a bias"
    if not base_ok(base):
        return f"the base is {type(base).__name__}, not a bitsandbytes NF4 Linear4bit"
    return lora_a, lora_b, float(proj.scaling[name]), name


class _FusedLoRA(torch.autograd.Function):
    """Adapters that share one input: one cast of ``x``, one matmul against their A matrices stacked, and per adapter an
    ``addmm`` of its B matmul into its base output in FP32 with the scale as alpha, cast back once. Inputs are
    ``(x, scale, n, A_1..A_n, B_1..B_n, base_1..base_n)``; outputs are the n projections."""

    @staticmethod
    def forward(ctx, x, scale, n, *rest):
        As, Bs, bases = rest[:n], rest[n:2 * n], rest[2 * n:]
        lead = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1]).to(As[0].dtype)                    # the one cast of the input
        a_cat = torch.cat(As, 0) if n > 1 else As[0]
        t = xf @ a_cat.t()                                                  # one rank matmul for every adapter
        ranks = [a.shape[0] for a in As]
        ts = torch.split(t, ranks, dim=1)
        outs = []
        for ti, b, base in zip(ts, Bs, bases):
            o = base.reshape(-1, base.shape[-1]).to(t.dtype, copy=True)    # the base output in FP32, a fresh buffer
            o.addmm_(ti, b.t(), alpha=scale)                                # the epilogue: s * t B^T added in place
            outs.append(o.to(base.dtype).reshape(*lead, b.shape[0]))       # one cast back
        ctx.save_for_backward(x, t, a_cat, *Bs)
        ctx.scale, ctx.n, ctx.ranks = scale, n, ranks
        ctx.base_dtypes = [base.dtype for base in bases]
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        x, t, a_cat, *Bs = ctx.saved_tensors
        s, n, ranks = ctx.scale, ctx.n, ctx.ranks
        xf = x.reshape(-1, x.shape[-1]).to(a_cat.dtype)
        ts = torch.split(t, ranks, dim=1)
        dts, dBs = [], []
        for g, ti, b in zip(grads, ts, Bs):
            gf = g.reshape(-1, g.shape[-1]).to(t.dtype)
            dBs.append((gf.t() @ ti).mul_(s))                               # dB = s * g^T t
            dts.append((gf @ b).mul_(s))                                    # dt = s * g B
        dt = torch.cat(dts, 1) if n > 1 else dts[0]
        dA = dt.t() @ xf                                                    # one matmul for every stacked A
        dx = (dt @ a_cat).to(x.dtype).reshape(x.shape) if ctx.needs_input_grad[0] else None   # one matmul for dx
        dAs = list(torch.split(dA, ranks, dim=0))
        dbases = [g if g is None else g.to(dtype) for g, dtype in zip(grads, ctx.base_dtypes)]   # base outputs: passed through
        return (dx, None, None, *dAs, *dBs, *dbases)


def _fused_mlp_forward(self, x):
    """The MLP with its adapters fused: ``down(act(gate(x)) * up(x))``, each projection's base output computed by its own
    base layer and its adapter added through :class:`_FusedLoRA`."""
    f = self._e4b_dense_lora
    if torch.is_autocast_enabled(x.device.type) or not all(
            _state_ok(getattr(self, p)) and _active(getattr(self, p)) == [n] for p, n in zip(PROJECTIONS, f["names"])):
        DENSE_MLP_LORA_STATS["fallback"] += 1
        return type(self).forward(self, x)                              # PEFT's path, which handles both
    A, B = [m.weight for m in f["A"]], [m.weight for m in f["B"]]       # read at call time: the live parameters
    if any(w.dtype != torch.float32 for w in A + B):
        raise RuntimeError("dense_lora_fuse: an adapter of a fused MLP is no longer FP32; disable the fusion before casting")
    DENSE_MLP_LORA_STATS["calls"] += 1
    g_base = self.gate_proj.base_layer(x)
    u_base = self.up_proj.base_layer(x)
    g, u = _FusedLoRA.apply(x, f["scale"], 2, A[0], A[1], B[0], B[1], g_base, u_base)
    h = self.act_fn(g) * u
    (d,) = _FusedLoRA.apply(h, f["scale"], 1, A[2], B[2], self.down_proj.base_layer(h))
    return d


def _refusal(mlp, base_ok):
    """``(A, B, scale, names)`` for a fusable MLP (A and B lists of the adapters' ``nn.Linear`` modules, ``names`` their
    active adapters), or the reason it cannot be fused (a ``str``)."""
    if type(mlp).__name__ not in FUSED_MLP_CLASSES:
        return f"{type(mlp).__name__} is not one of {sorted(FUSED_MLP_CLASSES)}, whose forward the fused forward repeats"
    if "forward" in mlp.__dict__:
        return "its forward is already patched"
    if not hasattr(mlp, "act_fn"):
        return "no 'act_fn'"
    got = [_adapter(getattr(mlp, p), base_ok) for p in PROJECTIONS]
    for p, g in zip(PROJECTIONS, got):
        if isinstance(g, str):
            return f"{p}: {g}"
    if len({g[2] for g in got}) != 1 or len({g[0].weight.shape[0] for g in got}) != 1:
        return "the three adapters differ in scale or rank"
    return [g[0] for g in got], [g[1] for g in got], got[0][2], [g[3] for g in got]


def enable_dense_mlp_lora_fuse(model: nn.Module, *, base_ok=None) -> int:
    """Fuse the adapters of every decoder MLP whose ``gate_proj``/``up_proj``/``down_proj`` are vanilla PEFT LoRA modules
    on NF4 bases. Returns the number of adapters fused, three per MLP (96 on Mistral-7B; ``0`` means nothing was fused,
    and every refused MLP's reason is in ``DENSE_MLP_LORA_STATS``). ``base_ok`` overrides the base-layer test (CPU tests
    only)."""
    base_ok = base_ok or _is_nf4_linear4bit
    DENSE_MLP_LORA_STATS.update(fused=0, adapters=0, refused={}, calls=0, fallback=0)
    for name, mod in model.named_modules():
        if not all(hasattr(mod, p) for p in PROJECTIONS):
            continue
        if hasattr(mod, "_e4b_dense_lora"):                             # fused already: it counts as fused
            DENSE_MLP_LORA_STATS["fused"] += 1
            DENSE_MLP_LORA_STATS["adapters"] += len(PROJECTIONS)
            continue
        r = _refusal(mod, base_ok)
        if isinstance(r, str):
            DENSE_MLP_LORA_STATS["refused"][name] = r
            continue
        A, B, scale, names = r
        mod._e4b_dense_lora = {"A": A, "B": B, "scale": scale, "names": names}
        mod.forward = types.MethodType(_fused_mlp_forward, mod)        # an instance attribute over the class's forward
        DENSE_MLP_LORA_STATS["fused"] += 1
        DENSE_MLP_LORA_STATS["adapters"] += len(PROJECTIONS)
    return DENSE_MLP_LORA_STATS["adapters"]


def disable_dense_mlp_lora_fuse(model: nn.Module) -> int:
    """Put every fused MLP's own forward back; returns how many were restored."""
    n = 0
    for _name, mod in model.named_modules():
        f = getattr(mod, "_e4b_dense_lora", None)
        if f is None:
            continue
        mod.__dict__.pop("forward", None)                               # the class's own forward again
        del mod._e4b_dense_lora
        n += 1
    return n
