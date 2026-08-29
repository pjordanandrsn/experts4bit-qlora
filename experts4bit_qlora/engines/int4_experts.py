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

import torch


def _shard_map(source_dir: str) -> dict:
    idx = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    lone = os.path.join(source_dir, "model.safetensors")
    if os.path.exists(lone):
        return {"*": "model.safetensors"}
    raise FileNotFoundError(f"no safetensors index under {source_dir}")


def enable_serve_experts_int4(model, source_dir: str) -> int:
    """Repack + install; returns the number of layers converted."""
    from safetensors import safe_open

    from int4_b32 import _plan
    from int4_pack_ref import pack_int4_b32

    wmap = _shard_map(source_dir)

    def load(name):
        shard = wmap.get(name) or wmap.get("*")
        if shard is None:
            raise KeyError(f"{name} not in checkpoint index")
        with safe_open(os.path.join(source_dir, shard), framework="pt") as f:
            return f.get_tensor(name)

    layers = model.model.layers if hasattr(model, "model") else model.layers
    todo = []
    for li, layer in enumerate(layers):
        for mod in layer.modules():
            st = getattr(mod, "_hot_residency", None)
            if st is not None and hasattr(st, "h_gu_p"):
                todo.append((li, st))
                break
    if not todo:
        raise RuntimeError("enable_serve_experts_int4 found no hot expert "
                           "state -- refusing a vacuous enable")
    keep_nf4 = os.environ.get("E4B_INT4_KEEP_NF4", "0") == "1"
    n_layers = 0
    for li, w in todo:
        E = w.h_gu_p.shape[0]
        dev = w.h_gu_p.device
        gu_pk, gu_sc, dn_pk, dn_sc = [], [], [], []
        for e in range(E):
            base = f"model.layers.{li}.mlp.experts.{e}"
            gate = load(f"{base}.gate_proj.weight").float()
            up = load(f"{base}.up_proj.weight").float()
            down = load(f"{base}.down_proj.weight").float()
            gu = torch.cat([gate, up], dim=0)          # gate FIRST
            pk_, sc_ = pack_int4_b32(gu)
            gu_pk.append(pk_)
            gu_sc.append(sc_)
            pk_, sc_ = pack_int4_b32(down)
            dn_pk.append(pk_)
            dn_sc.append(sc_)
        Ngu, Kgu = gu.shape
        Ndn, Kdn = down.shape
        _bn, _wp, sk_gu, _ku = _plan(Ngu, Kgu)
        _bn, _wp, sk_dn, _ku = _plan(Ndn, Kdn)
        R = 8   # top-k rows per decode token; parts sized for that
        w._int4_stores = {
            "gu": {"packed": torch.stack(gu_pk).to(dev).contiguous(),
                   "scales": torch.stack(gu_sc).to(dev).contiguous(),
                   "N": Ngu, "K": Kgu,
                   "part": torch.empty(sk_gu * R, Ngu,
                                       dtype=torch.float32, device=dev)},
            "dn": {"packed": torch.stack(dn_pk).to(dev).contiguous(),
                   "scales": torch.stack(dn_sc).to(dev).contiguous(),
                   "N": Ndn, "K": Kdn,
                   "part": torch.empty(sk_dn * R, Ndn,
                                       dtype=torch.float32, device=dev)},
        }
        if not keep_nf4:
            # the NF4 and int4 expert stores cannot co-reside at 30B
            # scale on a 32 GB part; drop NF4 layer-by-layer so peak
            # overhead is one layer. This makes the process
            # SINGLETON-DECODE-ONLY (the batched M-tile path needs the
            # NF4 stacks) -- E4B_INT4_KEEP_NF4=1 keeps both when VRAM
            # allows.
            for attr in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a"):
                t = getattr(w, attr)
                setattr(w, attr, t.new_empty((0,) * t.dim()))
            torch.cuda.empty_cache()
        n_layers += 1
    return n_layers
