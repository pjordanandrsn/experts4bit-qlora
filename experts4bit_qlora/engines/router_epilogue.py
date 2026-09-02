# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Glue round 3: the router epilogue as one launch
(``E4B_FUSE_ROUTER_EPI=1``).

After rounds one and two folded the norms, the residual add and the
rotary chain, the census's largest remaining coherent cluster at
decode is what the router does AFTER its GEMM: a softmax over every
expert, a top-k (which torch serves with a gather plus a bitonic
sort), a sum and a divide -- five launches per layer, ~0.43 ms of a
5.69 ms single-stream step.

**This is not the fusion that was refused.** That one folded the
router's GEMM into the same program, so a single CTA pulled the whole
router weight matrix and lost on occupancy. Here the GEMM stays
exactly where it is and only the epilogue moves, so a program reads E
floats -- 512 bytes at E=128. What is left to win is launch count.

Licensing repeats the round-one lesson: structure is checked
attribute by attribute, and the module's OWN forward must agree with
the reference epilogue on a deterministic probe before it is patched.
Routers that select before the softmax, add a bias, or route in groups
share the class-name shape and would be silently mis-routed -- the
probe rejects them. Off decode shapes the original forward runs.

Engagement is census PRESENCE of ``_router_epilogue``.
"""
from __future__ import annotations

import os

import torch

__all__ = ["fuse_router_epilogue"]

_MAX_DECODE_ROWS = 64


def _ref_epilogue(logits: torch.Tensor, k: int, norm: bool):
    """The reference this lane implements: fp32 softmax over ALL
    experts, then top-k, then the optional renormalisation."""
    probs = torch.softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(probs, k, dim=-1)
    if norm:
        vals = vals / vals.sum(dim=-1, keepdim=True)
    return probs, vals, idx


def _structural(mod):
    w = getattr(mod, "weight", None)
    if not torch.is_tensor(w) or w.dim() != 2:
        return None
    try:
        k = int(mod.top_k)
        e = int(mod.num_experts)
        norm = bool(mod.norm_topk_prob)
        hidden = int(mod.hidden_dim)
    except Exception:
        return None
    if w.shape != (e, hidden) or not (0 < k <= e):
        return None
    return k, e, norm, hidden


def _probe_matches(mod, k: int, norm: bool, hidden: int) -> bool:
    """Run the module's own forward and require it to agree with the
    reference epilogue -- both the selected expert SET and the weights.
    A name match cannot tell a softmax-then-topk router from a
    topk-then-softmax one, and mis-routing is not a rounding error."""
    g = torch.Generator(device="cpu").manual_seed(4242)
    x = torch.randn(4, hidden, generator=g).to(mod.weight.device,
                                               mod.weight.dtype)
    try:
        with torch.no_grad():
            out = mod(x)
    except Exception:
        return False
    if not (isinstance(out, tuple) and len(out) == 3):
        return False
    _, got_w, got_i = out
    with torch.no_grad():
        logits = torch.nn.functional.linear(x, mod.weight)
        _, ref_w, ref_i = _ref_epilogue(logits, k, norm)
    if not torch.is_tensor(got_i) or got_i.shape != ref_i.shape:
        return False
    if not torch.equal(got_i.cpu(), ref_i.cpu()):
        return False
    return torch.allclose(got_w.float().cpu(), ref_w.float().cpu(),
                          rtol=2 ** -8, atol=2 ** -12)


def fuse_router_epilogue(model) -> int:
    """Patch every structurally-matched, probe-licensed router.

    Returns the count; refuses a vacuous enable."""
    if os.environ.get("E4B_FUSE_ROUTER_EPI", "0") != "1":
        return 0
    try:
        from int4_b32 import router_epilogue
    except ImportError as e:
        raise RuntimeError(
            "E4B_FUSE_ROUTER_EPI=1 needs the kernel side's "
            "router_epilogue; install the matching cut or unset the flag"
        ) from e

    n = 0
    skipped = 0
    for mod in model.modules():
        if "Router" not in type(mod).__name__:
            continue
        spec = _structural(mod)
        if spec is None:
            continue
        k, e, norm, hidden = spec
        if not _probe_matches(mod, k, norm, hidden):
            skipped += 1
            continue
        orig = mod.forward

        def _fwd(hidden_states, _m=mod, _orig=orig, _k=k, _e=e,
                 _norm=norm, _h=hidden):
            rows = hidden_states.reshape(-1, _h)
            if rows.shape[0] > _MAX_DECODE_ROWS:
                return _orig(hidden_states)
            logits = torch.nn.functional.linear(rows, _m.weight)
            return router_epilogue(logits.float(), _k, _norm)

        mod.forward = _fwd
        n += 1
    if n == 0:
        raise RuntimeError(
            f"E4B_FUSE_ROUTER_EPI=1 patched no routers ({skipped} "
            "structurally matched but failed the semantic probe) -- "
            "refusing a vacuous enable")
    return n
