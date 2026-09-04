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
import re



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


def _top_k(model):
    """Routed experts per token, from the config (text tower for
    composites). The split-K partial buffer is sized ``sk * R * N`` and
    the kernel reshapes it EXACTLY, so a wrong R is a loud crash at the
    first decode step -- a top-8 constant sized it for two lucky
    families and broke the first top-4 one (Qwen1.5-MoE, K8 campaign).
    """
    cfg = getattr(model, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        for attr in ("num_experts_per_tok", "moe_top_k", "moe_topk",
                     "n_routed_experts_per_tok", "top_k_experts", "top_k"):
            v = getattr(c, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    raise RuntimeError(
        "enable_serve_experts_int4: cannot read routed-experts-per-token "
        "from the config; the split-K buffer cannot be sized safely")


_FUSED_TARGET = re.compile(
    r"^(?P<pfx>.*\.layers\.(?P<layer>\d+)\..*experts)\."
    r"(?P<role>gate_up_proj|down_proj)$")


def _prefused_layers(plan):
    """``{layer: (gate_up_key, down_key)}`` for PRE-FUSED families.

    Granite/Gemma-4/Qwen3-VL ship one stacked tensor per projection, so
    the planner routes them through ``passthrough`` and ``plan.experts``
    is EMPTY -- the per-expert fusion helper has nothing to fuse. Their
    stacks are already in the module's layout (any axis swap is recorded
    in ``plan.transforms`` and applied by the reader), so the int4 lane
    can pack them directly instead of refusing the family.
    """
    out = {}
    for ckpt_key, target in plan.passthrough.items():
        m = _FUSED_TARGET.match(target)
        if not m:
            continue
        layer = int(m.group("layer"))
        gu, dn = out.get(layer, (None, None))
        if m.group("role") == "gate_up_proj":
            gu = ckpt_key
        else:
            dn = ckpt_key
        out[layer] = (gu, dn)
    return {k: v for k, v in out.items() if v[0] and v[1]}


def _mxfp4_store_layout(gu_blocks, gu_scales, dn_blocks, dn_scales):
    """gpt-oss's released MXFP4 stacks, kept as released: ``gate_up_proj_blocks
    [E, 2I, H//32, 16]`` (fp4 nibble pairs, gate/up rows INTERLEAVED) with
    ``gate_up_proj_scales [E, 2I, H//32]`` (e8m0), and ``down_proj_blocks
    [E, H, I//32, 16]`` / ``[E, H, I//32]``. The kernel side's grouped MXFP4
    GEMM takes ``blocks [E, N, K//2]`` + ``scales [E, N, K//32]`` in exactly
    this orientation, so the only transform is the ROW de-interleave of the
    gate/up stack (gate block first, as the serve epilogue's ``chunk(2)``
    and the loader builder expect) and a flatten of the group axis. No
    value is re-quantised: the int4-b32 re-quantisation of these weights
    measured +0.626 nats (P30) because the uniform grid cannot hold e2m1
    levels; NF4's can, and this path holds them exactly."""
    import torch as _torch

    def _u8(t):
        return t if t.dtype == _torch.uint8 else t.view(_torch.uint8)
    gu_blocks, gu_scales = _u8(gu_blocks), _u8(gu_scales)
    dn_blocks, dn_scales = _u8(dn_blocks), _u8(dn_scales)
    if gu_blocks.ndim != 4 or dn_blocks.ndim != 4 or gu_blocks.shape[-1] != 16:
        raise RuntimeError("gpt_oss: expected MXFP4 blocks [E, rows, K//32, 16], got "
                           f"{tuple(gu_blocks.shape)} / {tuple(dn_blocks.shape)}")
    E, twoI, gH, _ = gu_blocks.shape
    E2, H, gI, _ = dn_blocks.shape
    if (twoI % 2 or E2 != E or tuple(gu_scales.shape) != (E, twoI, gH)
            or tuple(dn_scales.shape) != (E, H, gI) or gH * 32 != H or gI * 32 != twoI // 2):
        raise RuntimeError(
            "gpt_oss: MXFP4 stacks disagree with [E, 2I, H//32, 16] / [E, H, I//32, 16]: "
            f"blocks {tuple(gu_blocks.shape)} / {tuple(dn_blocks.shape)}, scales "
            f"{tuple(gu_scales.shape)} / {tuple(dn_scales.shape)}")
    gub = _torch.cat([gu_blocks[:, 0::2], gu_blocks[:, 1::2]], dim=1)   # de-interleave rows
    gus = _torch.cat([gu_scales[:, 0::2], gu_scales[:, 1::2]], dim=1)
    return (gub.reshape(E, twoI, gH * 16).contiguous(), gus.contiguous(),
            dn_blocks.reshape(E, H, gI * 16).contiguous(), dn_scales.contiguous())


def _gptoss_packer_layout(gate_up, down):
    """gpt-oss's dense stacks leave the plan reader in the MODULE layout:
    ``gate_up [E, H, 2I]`` input-major with the gate and up rows
    INTERLEAVED (``[..., ::2]`` gate, ``[..., 1::2]`` up), ``down
    [E, I, H]``. The packer wants ``[E, N, K]`` with the gate block
    before the up block (the serve epilogue splits with ``chunk(2)``).
    This is the loader builder's transform (``arch/gptoss.py``,
    ``from_gptoss``) line for line, so the int4 bytes pair gate rows
    with the wrapper's de-interleaved gate biases."""
    import torch as _torch

    if gate_up.ndim != 3 or down.ndim != 3:
        raise RuntimeError("gpt_oss: expected [E, *, *] expert stacks, got "
                           f"{tuple(gate_up.shape)} / {tuple(down.shape)}")
    E, H, twoI = gate_up.shape
    if twoI % 2 or tuple(down.shape) != (E, twoI // 2, H):
        raise RuntimeError(
            "gpt_oss: stacks disagree with the module layout [E, H, 2I] / "
            f"[E, I, H]: gate_up {tuple(gate_up.shape)}, down "
            f"{tuple(down.shape)}")
    gu = gate_up.transpose(1, 2).contiguous()                 # [E, 2I, H]
    gu = _torch.cat([gu[:, 0::2, :], gu[:, 1::2, :]], dim=1)  # de-interleave rows
    return gu, down.transpose(1, 2).contiguous()              # [E, H, I]


def _check_gptoss_wrapper(w, layer, first, down):
    """The int4 bytes are served by the hot-residency forward, whose
    gpt-oss epilogue (biases, clamped GLU) engages only on a wrapper the
    loader built as gpt-oss. A wrapper without that flag would serve the
    bytes through the plain SwiGLU -- shapes fine, model wrong."""
    gub = getattr(w, "h_gu_b", None)
    dnb = getattr(w, "h_dn_b", None)
    if not getattr(w, "gptoss", False) or gub is None or dnb is None:
        raise RuntimeError(
            f"layer {layer}: the hot-residency wrapper is not gpt-oss "
            "flagged (no bias epilogue); the int4 bytes would be served "
            "through a plain SwiGLU -- refusing")
    if gub.shape[-1] != first.shape[1] or dnb.shape[-1] != down.shape[1]:
        raise RuntimeError(
            f"layer {layer}: gpt-oss bias widths {tuple(gub.shape)} / "
            f"{tuple(dnb.shape)} do not match the packed stacks "
            f"{tuple(first.shape)} / {tuple(down.shape)}")


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


class _ExpertHessianSink:
    """Per-(layer, expert) ``2 X X^T`` accumulators fed by the fused
    forward's calibration tap. Layers are recognised by the identity of
    their hot NF4 gate/up stack (``w.h_gu_p``), which is what the fused
    forward is handed; ``layers`` restricts a pass to a subset so the
    running Hessians fit ``hessian_device`` (a 128-expert layer at hidden
    2048 / inter 768 is 2.4 GB in fp32)."""

    def __init__(self, layer_of: dict, layers, hessian_device="cpu"):
        self.layer_of = layer_of            # id(h_gu_p) -> layer index
        self.layers = set(layers)
        self.hessian_device = hessian_device
        self.gu = {}                        # (layer, e) -> HessianAccumulator
        self.dn = {}
        self.rows = {}                      # (layer, e) -> rows seen
        self.unmatched = 0

    def __call__(self, gu_p, sorted_ids, x_sorted, h):
        from gptq_pack import HessianAccumulator
        layer = self.layer_of.get(id(gu_p))
        if layer is None:
            self.unmatched += 1
            return
        if layer not in self.layers:
            return
        import torch
        ids = sorted_ids.to("cpu")
        for e in torch.unique(ids).tolist():
            m = (ids == e)
            idx = m.nonzero().flatten().to(x_sorted.device)
            key = (layer, int(e))
            if key not in self.gu:
                self.gu[key] = HessianAccumulator(x_sorted.shape[-1], device=self.hessian_device)
                self.dn[key] = HessianAccumulator(h.shape[-1], device=self.hessian_device)
                self.rows[key] = 0
            self.gu[key].add(x_sorted.index_select(0, idx))
            self.dn[key].add(h.index_select(0, idx))
            self.rows[key] += int(idx.numel())

    def hessians(self):
        out = {}
        for (layer, e), acc in self.gu.items():
            out.setdefault(layer, {})[e] = (acc.H, self.dn[(layer, e)].H, self.rows[(layer, e)])
        return out


def _expert_layers(model, source_dir, model_type=None, plan_model=None):
    """(plan, [(layer, wrapper state)]) -- the enabler's own enumeration,
    shared with calibration so both see the same layers in the same order."""
    import torch as _torch

    from ..arch.moe_plan import plan_moe_checkpoint
    keys, _read_tensor = safetensors_reader(source_dir)
    mt = model_type or getattr(getattr(model, "config", None), "model_type", None)
    if not mt:
        raise RuntimeError("model_type unknown; pass model_type= explicitly")
    if plan_model is None:
        plan_model = _meta_twin(model)
    plan = plan_moe_checkpoint(keys, plan_model, mt, skip_extra_layers=True)
    prefused = _prefused_layers(plan) if not plan.experts else {}
    out = []
    for layer in (plan.experts or prefused):
        if plan.experts:
            first_name, _down_name = plan.expert_targets[layer]
        else:
            first_name = plan.passthrough[prefused[layer][0]]
        w = _wrapper_for(model, first_name)
        if w is None:
            raise RuntimeError(f"layer {layer}: no hot-residency state near {first_name}")
        out.append((layer, w))
    del _torch
    return plan, out


def calibrate_expert_hessians(model, source_dir: str, batches, *,
                              model_type: str | None = None,
                              plan_model=None, device=None,
                              hessian_device="cpu",
                              max_hessian_bytes: int = 24 << 30,
                              layers_per_pass: int | None = None) -> dict:
    """Run ``batches`` (token-id tensors ``[B, T]``) through ``model`` on
    its NF4 expert stacks and return ``{layer: {expert: (H_gu, H_dn,
    rows)}}`` with ``H = 2 X X^T`` over the rows each expert actually
    saw -- the gate/up input for ``H_gu`` and the down-projection input
    (post-activation) for ``H_dn``, both taken at the fused forward's
    calibration tap. Must run BEFORE ``enable_serve_experts_int4``: the
    tap keys layers by their live NF4 stack.

    Memory is the constraint, not time: a 128-expert layer's fp32
    Hessians are ~2.4 GB, so the layers are calibrated in passes sized
    by ``max_hessian_bytes`` (or ``layers_per_pass``), each pass a full
    run over ``batches``."""
    from . import hot_residency as _hr
    _plan, layers = _expert_layers(model, source_dir, model_type, plan_model)
    if not layers:
        raise RuntimeError("calibrate_expert_hessians: no expert layers")
    layer_of = {id(w.h_gu_p): layer for layer, w in layers}
    if len(layer_of) != len(layers):
        raise RuntimeError("calibrate_expert_hessians: expert stacks are not "
                           "distinct objects (freed?) -- calibrate before the int4 enable")
    cfg = getattr(model, "config", None)
    if layers_per_pass is None:
        hid = getattr(cfg, "hidden_size", None)
        inter = (getattr(cfg, "moe_intermediate_size", None)
                 or getattr(cfg, "intermediate_size", None))
        n_exp = (getattr(cfg, "num_local_experts", None)
                 or getattr(cfg, "num_experts", None))
        if hid and inter and n_exp:
            per_layer = int(n_exp) * (int(hid) ** 2 + int(inter) ** 2) * 4
            layers_per_pass = max(1, int(max_hessian_bytes // per_layer))
        else:
            layers_per_pass = len(layers)
    import torch
    batches = list(batches)
    dev = device or next(model.parameters()).device
    order = [layer for layer, _w in layers]
    result = {}
    prev = _hr._CALIB_SINK
    try:
        for i in range(0, len(order), layers_per_pass):
            chunk = order[i:i + layers_per_pass]
            sink = _ExpertHessianSink(layer_of, chunk, hessian_device)
            _hr._CALIB_SINK = sink
            with torch.no_grad():
                for ids in batches:
                    model(ids.to(dev))
            if not sink.gu:
                raise RuntimeError(
                    f"calibrate_expert_hessians: the tap saw no expert rows for layers "
                    f"{chunk[:4]}... (unmatched calls: {sink.unmatched}) -- is the model "
                    "on the fused NF4 path (all-VRAM hot residency)?")
            result.update(sink.hessians())
    finally:
        _hr._CALIB_SINK = prev
    return result


def enable_serve_experts_int4(model, source_dir: str, *,
                              model_type: str | None = None,
                              plan_model=None,
                              expert_hessians: dict | None = None,
                              min_rows: int = 32) -> int:
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

    ``expert_hessians`` (from :func:`calibrate_expert_hessians`) switches
    the packer to the kernel side's GPTQ-style ``gptq_pack_int4_b32`` per
    expert, with each expert's own gate/up and down Hessians. An expert
    the calibration text never routed to (fewer than ``min_rows`` rows)
    is packed round-to-nearest and COUNTED; the store records
    ``calibrated=(n_gptq, n_rtn)`` so a lane can refuse a mostly-RTN pack
    under the calibrated banner.
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

    prefused = _prefused_layers(plan) if not plan.experts else {}
    # gpt-oss's stacks are MXFP4 with INTERLEAVED gate/up rows and a
    # bias-carrying epilogue. They are served NATIVELY -- the released fp4
    # blocks and e8m0 scales, row-de-interleaved, through the kernel side's
    # grouped MXFP4 GEMM -- never re-quantised onto the int4-b32 grid (that
    # measured +0.626 nats, P30: a uniform grid cannot hold e2m1 levels).
    # The biases and the clamped GLU are applied by the serve forward's
    # gpt-oss epilogue, which runs after whichever GEMM branch served the
    # store.
    gptoss = bool(prefused) and mt == "gpt_oss"

    n_layers = 0
    tot_gptq = tot_rtn = 0
    for layer in (plan.experts or prefused):
        if plan.experts:
            first_name, _down_name = plan.expert_targets[layer]
        else:
            first_name = plan.passthrough[prefused[layer][0]]
        w = _wrapper_for(model, first_name)
        if w is None:
            raise RuntimeError(
                f"layer {layer}: no hot-residency state near {first_name} "
                "-- enable hot residency (all-VRAM) before the int4 lane")
        if not w._all_hot():
            raise RuntimeError(
                f"layer {layer} is tiered; this lane is the all-VRAM "
                "collapsed path -- use placement-override all-vram")
        if plan.experts:
            first, down = read_fused_expert_layer(plan, layer, read,
                                                  device="cpu",
                                                  dtype=_torch.float32)
        else:
            gu_key, dn_key = prefused[layer]
            if gptoss:
                # native MXFP4: the released blocks and scales, never the
                # dequantised stacks (see _mxfp4_store_layout)
                gkind, gblk, gsc, _x = plan.scales[gu_key]
                dkind, dblk, dsc, _x = plan.scales[dn_key]
                if gkind != "mxfp4" or dkind != "mxfp4":
                    raise RuntimeError(f"layer {layer}: gpt_oss stacks are not MXFP4 "
                                       f"in this checkpoint ({gkind}/{dkind})")
                gub, gus, dnb, dns = _mxfp4_store_layout(
                    read_tensor(gblk), read_tensor(gsc), read_tensor(dblk), read_tensor(dsc))
                dev = w.h_gu_p.device
                Ngu, Kgu = gub.shape[1], gub.shape[2] * 2
                Ndn, Kdn = dnb.shape[1], dnb.shape[2] * 2
                _check_gptoss_wrapper(w, layer, gub, dnb)
                w._int4_stores = {
                    "kind": "mxfp4",
                    "gu": {"blocks": gub.to(dev), "scales": gus.to(dev), "N": Ngu, "K": Kgu},
                    "dn": {"blocks": dnb.to(dev), "scales": dns.to(dev), "N": Ndn, "K": Kdn},
                }
                if not keep_nf4:
                    for attr in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a"):
                        t = getattr(w, attr)
                        setattr(w, attr, t.new_empty((0,) * t.dim()))
                    _torch.cuda.empty_cache()
                n_layers += 1
                continue
            first = read(gu_key).to(_torch.float32)
            down = read(dn_key).to(_torch.float32)
        dev = w.h_gu_p.device
        E = first.shape[0]

        hl = (expert_hessians or {}).get(layer)
        if expert_hessians is not None and hl is None:
            raise RuntimeError(f"layer {layer}: calibrated enable but no expert "
                               "Hessians for this layer -- refusing to pack it "
                               "round-to-nearest under the calibrated banner")
        n_gptq = n_rtn = 0

        def _pack_stack(stack, role, E=E, dev=dev):
            nonlocal n_gptq, n_rtn
            pk, sc = [], []
            for e in range(E):
                H = None
                if hl is not None and e in hl:
                    H_gu, H_dn, rows = hl[e]
                    if rows >= min_rows:
                        H = H_gu if role == "gu" else H_dn
                if H is not None:
                    from gptq_pack import gptq_pack_int4_b32
                    p, c = gptq_pack_int4_b32(stack[e], H.to("cpu"))
                    n_gptq += 1
                else:
                    p, c = pack_int4_b32(stack[e])
                    n_rtn += 1
                pk.append(p)
                sc.append(c)
            return (_torch.stack(pk).to(dev).contiguous(),
                    _torch.stack(sc).to(dev).contiguous())

        gu_p, gu_s = _pack_stack(first, "gu")
        dn_p, dn_s = _pack_stack(down, "dn")
        Ngu, Kgu = first.shape[1], first.shape[2]
        Ndn, Kdn = down.shape[1], down.shape[2]
        _b, _w2, sk_gu, _k = _plan(Ngu, Kgu)
        _b, _w2, sk_dn, _k = _plan(Ndn, Kdn)
        R = _top_k(model)
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
        if expert_hessians is not None:
            w._int4_stores["calibrated"] = (n_gptq, n_rtn)
        n_layers += 1
        tot_gptq += n_gptq
        tot_rtn += n_rtn
    if n_layers == 0:
        raise RuntimeError("enable_serve_experts_int4: the plan holds "
                           "neither per-expert stacks nor pre-fused expert "
                           "tensors -- refusing a vacuous enable")
    if expert_hessians is not None:
        print(f"INT4EXP calibrated experts: {tot_gptq} gptq / {tot_rtn} rtn "
              f"(min_rows={min_rows}) over {n_layers} layers", flush=True)
    return n_layers
