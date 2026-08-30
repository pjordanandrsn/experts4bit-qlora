# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Opt-in uniform-int4 expert store for SINGLE-STREAM serving decode.

``enable_serve_experts_int4(model, source_dir)`` repacks every hot
expert stack to the int4-b32 grid **from the source checkpoint** (never
from the resident NF4 bytes: quantising onto an already-quantised grid
measured ~7x the pure grid's ppl cost), installs the stores on each
expert wrapper, and frees the NF4 stacks layer by layer so peak
overhead is one layer's worth. The singleton decode branch then runs
the int4-b32 grouped GEMV -- measured 2.7-3.3x over the NF4 path at the
census cells, with the grid itself at +0.007 ppl over 8,192
teacher-forced tokens.

Fusion layout MUST match the loader's: gate_up is ``cat(gate, up)``
along the intermediate axis, **gate first** (moe_conventions.fuse_experts).
The parity test pins one repacked expert against the source weights.

Scope: T == 1 decode only (the singleton branch). Batched decode keeps
the NF4 M-tile path -- at mean group 1.6+ the grouped kernel's weight
sharing wins, and that lane's engine is the certified graph loop.
"""
from __future__ import annotations

import json
import os



def safetensors_reader(source_dir: str):
    """``read_tensor`` over a safetensors snapshot: (keys, reader)."""
    from safetensors import safe_open

    idx = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
    elif os.path.exists(os.path.join(source_dir, "model.safetensors")):
        with safe_open(os.path.join(source_dir, "model.safetensors"),
                       framework="pt") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    else:
        raise FileNotFoundError(f"no safetensors under {source_dir}")

    def read_tensor(key):
        with safe_open(os.path.join(source_dir, wmap[key]),
                       framework="pt") as f:
            return f.get_tensor(key)
    return list(wmap.keys()), read_tensor


def _wrapper_for(model, param_name: str):
    """The hot-residency state owning ``param_name``.

    Resolves the fused target's parent module (the experts module in the
    plan's tree), searches it and its descendants, then walks up toward
    the root. Plan names are rooted at the causal-LM tree; when ``model``
    is the bare decoder the leading component is stripped and retried.
    """
    for name in (param_name, param_name.split(".", 1)[-1]):
        parts = name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            try:
                mod = model.get_submodule(".".join(parts[:cut]))
            except AttributeError:
                continue
            search = mod.modules() if cut == len(parts) - 1 else (mod,)
            for m in search:
                st = getattr(m, "_hot_residency", None)
                if st is not None and hasattr(st, "h_gu_p"):
                    return st
    return None


def _meta_twin(model):
    """A plannable twin of the live model on ``meta``.

    The live tree can never validate a plan -- its expert modules are
    residency wrappers, so the checkpoint's expert keys are unclaimable
    and the wrapper buffers would be unclaimed. Rebuild the SAME
    architecture fresh from config instead: constructing the live
    model's own class keeps composite trees (a vision-language wrapper
    whose text decoder lives under a prefix) shaped exactly as the
    checkpoint's keys expect. The causal-LM auto class is only the
    fallback for objects that cannot be rebuilt from their config.
    """
    import torch as _torch

    with _torch.device("meta"):
        try:
            return model.__class__(model.config)
        except Exception:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_config(model.config)


def enable_serve_experts_int4(model, source_dir: str, *,
                              model_type: str | None = None,
                              plan_model=None) -> int:
    """Repack + install for EVERY family the load plan understands.

    Routes the source read through the same machinery the loader uses --
    ``plan_moe_checkpoint`` (per-family keymaps), ``make_plan_reader``
    (dequantizes FP8/GPTQ/AWQ/NVFP4/MXFP4 sources), and
    ``read_fused_expert_layer`` (gate-first fusion, or non-gated
    stacking) -- so the int4 stacks are byte-consistent with what the
    NF4 loader would have produced from the same checkpoint, for every
    model in the coverage matrix. Never reads resident NF4 (composition
    measured ~7x the pure grid's ppl cost). Collapsed-path-only, as
    before: refuses tiered layers loudly.
    """
    import torch as _torch

    from int4_b32 import _plan
    from int4_pack_ref import pack_int4_b32

    from ..arch.moe_load import make_plan_reader, read_fused_expert_layer
    from ..arch.moe_plan import plan_moe_checkpoint

    keys, read_tensor = safetensors_reader(source_dir)
    mt = model_type or getattr(getattr(model, "config", None),
                               "model_type", None)
    if not mt:
        raise RuntimeError("enable_serve_experts_int4: model_type unknown; "
                           "pass model_type= explicitly")
    if plan_model is None:
        plan_model = _meta_twin(model)
    plan = plan_moe_checkpoint(keys, plan_model, mt, skip_extra_layers=True)
    read = make_plan_reader(plan, read_tensor, _torch.float32)
    keep_nf4 = os.environ.get("E4B_INT4_KEEP_NF4", "0") == "1"

    n_layers = 0
    for layer in plan.experts:
        first_name, _down_name = plan.expert_targets[layer]
        w = _wrapper_for(model, first_name)
        if w is None:
            raise RuntimeError(
                f"layer {layer}: no hot-residency state near {first_name} "
                "-- enable hot residency (all-VRAM) before the int4 lane")
        if not w._all_hot():
            raise RuntimeError(
                f"layer {layer} is tiered; this lane is the all-VRAM "
                "collapsed path -- use placement-override all-vram")
        first, down = read_fused_expert_layer(plan, layer, read,
                                              device="cpu",
                                              dtype=_torch.float32)
        dev = w.h_gu_p.device
        E = first.shape[0]

        def _pack_stack(stack, E=E, dev=dev):
            pk, sc = zip(*[pack_int4_b32(stack[e]) for e in range(E)])
            return (_torch.stack(pk).to(dev).contiguous(),
                    _torch.stack(sc).to(dev).contiguous())

        gu_p, gu_s = _pack_stack(first)
        dn_p, dn_s = _pack_stack(down)
        Ngu, Kgu = first.shape[1], first.shape[2]
        Ndn, Kdn = down.shape[1], down.shape[2]
        _b, _w2, sk_gu, _k = _plan(Ngu, Kgu)
        _b, _w2, sk_dn, _k = _plan(Ndn, Kdn)
        R = 8
        w._int4_stores = {
            "gu": {"packed": gu_p, "scales": gu_s, "N": Ngu, "K": Kgu,
                   "part": _torch.empty(sk_gu * R, Ngu,
                                        dtype=_torch.float32, device=dev)},
            "dn": {"packed": dn_p, "scales": dn_s, "N": Ndn, "K": Kdn,
                   "part": _torch.empty(sk_dn * R, Ndn,
                                        dtype=_torch.float32, device=dev)},
        }
        if not keep_nf4:
            for attr in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a"):
                t = getattr(w, attr)
                setattr(w, attr, t.new_empty((0,) * t.dim()))
            _torch.cuda.empty_cache()
        n_layers += 1
    if n_layers == 0:
        raise RuntimeError("enable_serve_experts_int4: the plan holds no "
                           "expert layers -- refusing a vacuous enable")
    return n_layers
