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


def _ref_topk_softmax(logits: torch.Tensor, k: int, bias):
    """gpt-oss / GraniteMoe: top-k ON THE LOGITS (plus the router bias
    when it carries one), then a softmax over the k selected."""
    x = logits.float() + (bias.float() if bias is not None else 0.0)
    top, idx = torch.topk(x, k, dim=-1)
    return x, torch.softmax(top, dim=-1), idx


def _gemma4_pre(mod, rows):
    """Gemma-4's router input: an unscaled RMSNorm, a learned per-channel
    scale and hidden_size**-0.5 -- module-side, cheap, kept in torch."""
    return mod.norm(rows) * mod.scale * mod.scalar_root_size


def _out_positions(out):
    """(first, weights, index) positions in a router's output tuple. The
    index is the only integer tensor; the weights are the float tensor
    that shares its shape; ``first`` is the remaining slot. GraniteMoe
    returns (index, weights, logits); every other family (first, weights,
    index)."""
    if not isinstance(out, tuple) or len(out) != 3 or not all(torch.is_tensor(t) for t in out):
        return None
    ipos = [i for i, t in enumerate(out) if not t.is_floating_point()]
    if len(ipos) != 1:
        return None
    wpos = [i for i, t in enumerate(out) if i != ipos[0] and tuple(t.shape) == tuple(out[ipos[0]].shape)]
    if len(wpos) != 1:
        return None
    fpos = ({0, 1, 2} - {ipos[0], wpos[0]}).pop()
    return fpos, wpos[0], ipos[0]


def _structural(mod):
    """Which routers this could be, structurally, or None. Returns a list
    of ``(kind, spec)`` candidates in probe order; the module's own forward
    decides among them (attribute presence alone cannot: a module may
    carry ``norm_topk_prob`` and still select on the logits). Kinds: ``softmax_topk`` (Qwen3-MoE / OLMoE / Mixtral: softmax
    over all, top-k, optional renormalise), ``topk_softmax`` (gpt-oss with
    a bias, GraniteMoe without: top-k on the logits, softmax over the k),
    ``gemma4`` (Gemma4TextRouter: norm and scale before the projection,
    softmax-topk-renormalise, a learned per-expert scale after)."""
    name = type(mod).__name__
    if name == "Gemma4TextRouter":
        try:
            k = int(mod.config.top_k_experts)
            w = mod.proj.weight
            e, hidden = int(w.shape[0]), int(w.shape[1])
            pes = mod.per_expert_scale
            _ = mod.norm, mod.scale, float(mod.scalar_root_size)
        except Exception:
            return None
        if getattr(mod.proj, "bias", None) is not None or tuple(pes.shape) != (e,) or not (0 < k <= e):
            return None
        return [("gemma4", {"k": k, "e": e, "hidden": hidden})]
    w = getattr(mod, "weight", None)
    if not torch.is_tensor(w) or w.dim() != 2:
        return None
    e, hidden = int(w.shape[0]), int(w.shape[1])
    try:
        k = int(mod.top_k)
    except Exception:
        return None
    if not (0 < k <= e):
        return None
    cands = []
    try:
        dims_ok = int(mod.num_experts) == e and int(mod.hidden_dim) == hidden
    except Exception:
        dims_ok = False
    if hasattr(mod, "norm_topk_prob"):
        if dims_ok:
            cands.append(("softmax_topk", {"k": k, "e": e, "hidden": hidden, "norm": bool(mod.norm_topk_prob)}))
    elif dims_ok:
        # Mixtral's router (transformers 5.16) is softmax over all, top-k,
        # ALWAYS renormalised, with no ``norm_topk_prob`` attribute to say
        # so. Offer both renormalisation choices; the probe keeps the one
        # the module's own forward computes (the validation lane refused
        # Mixtral with "0 structurally matched" for want of this).
        for norm in (True, False):
            cands.append(("softmax_topk", {"k": k, "e": e, "hidden": hidden, "norm": norm}))
    bias = getattr(mod, "bias", None)
    if bias is None or (torch.is_tensor(bias) and tuple(bias.shape) == (e,)):
        cands.append(("topk_softmax", {"k": k, "e": e, "hidden": hidden, "bias": bias}))
    return cands or None


def _reference_for(mod, kind, spec, x):
    """What the kind's fused path computes, in torch, on the probe input."""
    if kind == "gemma4":
        logits = torch.nn.functional.linear(_gemma4_pre(mod, x), mod.proj.weight)
        probs, vals, idx = _ref_epilogue(logits, spec["k"], True)
        return probs, vals * mod.per_expert_scale.float()[idx], idx
    logits = torch.nn.functional.linear(x, mod.weight)
    if kind == "softmax_topk":
        return _ref_epilogue(logits, spec["k"], spec["norm"])
    return _ref_topk_softmax(logits, spec["k"], spec["bias"])


def _probe_matches(mod, kind, spec) -> bool:
    """Run the module's own forward and require it to agree with the
    kind's reference -- both the selected expert SET and the weights.
    A name match cannot tell a softmax-then-topk router from a
    topk-then-softmax one, and mis-routing is not a rounding error.
    Records where the module puts (first, weights, index) in its tuple."""
    g = torch.Generator(device="cpu").manual_seed(4242)
    w = mod.proj.weight if kind == "gemma4" else mod.weight
    x = torch.randn(4, spec["hidden"], generator=g).to(w.device, w.dtype)
    try:
        with torch.no_grad():
            out = mod(x)
    except Exception:
        return False
    pos = _out_positions(out)
    if pos is None:
        return False
    got_f, got_w, got_i = out[pos[0]], out[pos[1]], out[pos[2]]
    with torch.no_grad():
        ref_f, ref_w, ref_i = _reference_for(mod, kind, spec, x)
        pre = _gemma4_pre(mod, x) if kind == "gemma4" else x
        raw = torch.nn.functional.linear(pre, w)
    if got_i.shape != ref_i.shape or not torch.equal(got_i.cpu(), ref_i.cpu()):
        return False
    if not torch.allclose(got_w.float().cpu(), ref_w.float().cpu(), rtol=2 ** -8, atol=2 ** -12):
        return False
    # The FIRST slot is part of the module's contract too (callers that
    # record router logits read it): it is either the kind's first output
    # (Qwen3-MoE returns the softmax probabilities there; the
    # select-on-logits kinds return the biased logits) or the RAW
    # projection (Mixtral, Gemma-4). Record which, and refuse a module
    # whose first slot is neither -- the fused path would hand it
    # something else (review finding on #370).
    def _same(a, b):
        return (torch.is_tensor(a) and a.shape == b.shape
                and torch.allclose(a.float().cpu(), b.float().cpu(), rtol=2 ** -6, atol=2 ** -8))
    if _same(got_f, ref_f):
        spec["first"] = "ref"
    elif _same(got_f, raw):
        spec["first"] = "raw"
    else:
        return False
    spec["out_pos"] = pos
    return True


def _kernel_supports_select_on_logits(router_epilogue) -> bool:
    import inspect
    try:
        return "select_on_logits" in inspect.signature(router_epilogue).parameters
    except (TypeError, ValueError):
        return False


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
    has_sol = _kernel_supports_select_on_logits(router_epilogue)
    n = 0
    skipped = 0
    no_kernel_mode = 0
    for mod in model.modules():
        if "Router" not in type(mod).__name__:
            continue
        cands = _structural(mod)
        if not cands:
            continue
        chosen = None
        needs_mode = False
        for kind, spec in cands:
            if kind == "topk_softmax" and not has_sol:
                needs_mode = True   # the installed kernel predates select_on_logits
                continue
            if _probe_matches(mod, kind, spec):
                chosen = (kind, spec)
                break
        if chosen is None:
            if needs_mode:
                no_kernel_mode += 1
            else:
                skipped += 1
            continue
        kind, spec = chosen
        orig = mod.forward
        k, hidden, pos = spec["k"], spec["hidden"], spec["out_pos"]

        raw_first = spec["first"] == "raw"
        if kind == "gemma4":
            def _fwd(hidden_states, _m=mod, _orig=orig, _k=k, _h=hidden, _pos=pos, _raw=raw_first):
                rows = hidden_states.reshape(-1, _h)
                if rows.shape[0] > _MAX_DECODE_ROWS:
                    return _orig(hidden_states)
                logits = torch.nn.functional.linear(_gemma4_pre(_m, rows), _m.proj.weight)
                probs, w, idx = router_epilogue(logits.float(), _k, True)
                w = w * _m.per_expert_scale.float()[idx]
                return _assemble(_pos, logits if _raw else probs, w.to(hidden_states.dtype), idx)
        elif kind == "softmax_topk":
            def _fwd(hidden_states, _m=mod, _orig=orig, _k=k, _h=hidden, _pos=pos, _norm=spec["norm"], _raw=raw_first):
                rows = hidden_states.reshape(-1, _h)
                if rows.shape[0] > _MAX_DECODE_ROWS:
                    return _orig(hidden_states)
                logits = torch.nn.functional.linear(rows, _m.weight)
                first, w, idx = router_epilogue(logits.float(), _k, _norm)
                return _assemble(_pos, logits if _raw else first, w, idx)
        else:
            def _fwd(hidden_states, _m=mod, _orig=orig, _k=k, _h=hidden, _pos=pos, _bias=spec["bias"], _raw=raw_first):
                rows = hidden_states.reshape(-1, _h)
                if rows.shape[0] > _MAX_DECODE_ROWS:
                    return _orig(hidden_states)
                logits = torch.nn.functional.linear(rows, _m.weight)
                first, w, idx = router_epilogue(logits.float(), _k, False,
                                                select_on_logits=True, bias=_bias)
                return _assemble(_pos, logits if _raw else first, w, idx)
        mod.forward = _fwd
        n += 1
    if n == 0:
        raise RuntimeError(
            f"E4B_FUSE_ROUTER_EPI=1 patched no routers ({skipped} "
            "structurally matched but failed the semantic probe"
            + (f"; {no_kernel_mode} need a grouped-nf4-gemm whose router_epilogue "
               "takes select_on_logits" if no_kernel_mode else "")
            + ") -- refusing a vacuous enable")
    return n


def _assemble(pos, first, w, idx):
    """Return the three outputs in the module's own order."""
    out = [None, None, None]
    out[pos[0]], out[pos[1]], out[pos[2]] = first, w, idx
    return tuple(out)
