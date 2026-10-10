# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""One fused q/k/v projection for TRAINING attention (P129; on by default, ``E4B_TRAIN_FUSE_QKV=0`` keeps the three projections).

e4b trains attention through four ``LoRALinear`` adapters, each around a bitsandbytes NF4 ``Linear4bit``. q, k and v read the same
input, but each pays its own launches: a nested-absmax dequantize, an NF4 dequantize, the base matmul, a cast to the adapters'
dtype, two LoRA matmuls, a cast back and an add. That is about ten a projection, in the forward, again in the checkpoint recompute
and in the backward (P128's census: attention was the largest launch bucket of a training step). This module computes the three
at once:

- **The base.** The three projections' packed NF4 rows are concatenated along N. A 64-element block lies within one row of K, so
  the bytes are the same bytes. Each projection's nested (double-quantized) absmax carries its own offset, so the nested form
  cannot be concatenated. It is expanded once to fp32 with bitsandbytes' own steps (``dequantize_blockwise`` plus the offset), and
  the fused dequantize therefore produces, bit for bit, the three dequantized weights stacked. One dequantize and one matmul
  replace six dequantizes and three matmuls.
- **The adapters.** One cast of the input, one matmul against the three A matrices concatenated (``[3r, K]``), three B matmuls
  and one cast back. The nine adapter tensors stay the trainable parameters, on their own modules and under their own names.

Concatenating changes the GEMM shapes, and cuBLAS may then pick another algorithm or reduction order, so the projections are
equal to rounding, not bit for bit. The forward is serving's fused attention forward (``qkv_fuse._fused_forward``), which looks
the modeling module's rotary up at call time, so e4b's fused training rope and RMSNorm stay in force.

Memory: the q/k/v NF4 bases are released once the fused copy holds their bytes, so the packed weights are not duplicated. The
expanded fp32 absmax replaces the nested 8-bit one, about 3 bytes more per 64-element block: roughly 24 MB over Qwen3-30B-A3B's
48 layers. A released projection called on its own raises a ``RuntimeError`` that names the fusion.

``disable_fast_train`` undoes it (:func:`disable_train_fuse_qkv`): each projection gets its base back as a view of the fused bytes
with its slice of the expanded fp32 absmax, which dequantizes bit for bit as the nested statistics did, so the three projections
compute what they computed before the fusion. The 3 bytes per 64 values stay until the model is reloaded.

Anything that does not match is refused and keeps today's path (``TRAIN_QKV_STATS["refused"]`` says why).

On by default since P129's DEFAULT_ON read (Qwen3-30B-A3B at TC1's field recipe on two RTX 5090 hosts: 13.9-14.6 % fewer launches,
0.868-0.906 of the step, the step-0 held-out inside an fp32-anchored rounding envelope, held-out at N within 0.002).
``E4B_TRAIN_FUSE_QKV=0`` (or ``false`` / ``off`` / ``no``) keeps the three projections.
"""
from __future__ import annotations

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

#: What :func:`enable_train_fuse_qkv` did: modules fused, and every refused module's reason.
TRAIN_QKV_STATS = {"fused": 0, "refused": {}, "calls": 0}


#: The attention classes this module fuses (its refusal rule); the training estimate counts their q/k/v (``arch.topology``).
FUSED_ATTENTION_CLASSES = frozenset({"Qwen3MoeAttention"})

#: ``E4B_TRAIN_FUSE_QKV`` values that keep the three projections; anything else, unset included, fuses (P129 DEFAULT_ON).
TRAIN_FUSE_QKV_OFF = frozenset({"0", "false", "off", "no"})


def train_fuse_qkv_requested() -> bool:
    """On unless ``E4B_TRAIN_FUSE_QKV`` is ``0`` (or ``false`` / ``off`` / ``no``); unset is on."""
    return os.environ.get("E4B_TRAIN_FUSE_QKV", "1").strip().lower() not in TRAIN_FUSE_QKV_OFF


def _expanded_absmax(qs) -> torch.Tensor:
    """The fp32 absmax bitsandbytes' ``dequantize_4bit`` computes from ``qs``: the same steps in the same order."""
    from bitsandbytes.functional import dequantize_blockwise
    if not qs.nested:
        a = qs.absmax
        return a if a.dtype == torch.float32 else a.float()
    a = dequantize_blockwise(qs.absmax, qs.state2)
    a += qs.offset
    return a if a.dtype == torch.float32 else a.float()


def _refusal(mod) -> str | None:
    """Why ``mod``'s q/k/v cannot be fused (``None``: they can)."""
    from ..lora import LoRALinear
    if type(mod).__name__ not in FUSED_ATTENTION_CLASSES:
        return f"{type(mod).__name__} is not Qwen3MoeAttention, the class serving's fused forward covers"
    for attr in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "head_dim", "config"):
        if not hasattr(mod, attr):
            return f"missing {attr!r}"
    parts = (mod.q_proj, mod.k_proj, mod.v_proj)
    if not all(isinstance(p, LoRALinear) for p in parts):
        return "q/k/v are not all LoRALinear"
    if len({(p.lora_A.shape[0], p.lora_A.dtype, p.lora_B.dtype, float(p.scaling)) for p in parts}) != 1:
        return "the three adapters differ in rank, dtype or scaling"
    if any(getattr(p, "lora_dropout", None) is not None for p in parts):
        return "an adapter carries dropout"
    bases = [p.base for p in parts]
    if any(getattr(b, "bias", None) is not None for b in bases):
        return "a projection carries a bias"
    try:
        qss = [b.weight.quant_state for b in bases]
    except AttributeError:
        return "a base is not a bitsandbytes 4-bit Linear"
    if any(qs is None for qs in qss):
        return "a base is not quantized"
    if len({(qs.quant_type, qs.blocksize, qs.dtype) for qs in qss}) != 1 or qss[0].quant_type != "nf4":
        return "the bases differ in quant type, blocksize or compute dtype, or are not NF4"
    K = {int(qs.shape[1]) for qs in qss}
    if len(K) != 1 or next(iter(K)) % int(qss[0].blocksize):
        return "the bases' K differ, or K is not a multiple of the blocksize (a block would span rows)"
    if any(getattr(b, "compute_dtype", None) != getattr(bases[0], "compute_dtype", None) for b in bases):
        return "the bases' compute dtypes differ"
    from .glue_r2 import rotary_is_rotate_half
    if not rotary_is_rotate_half(mod, int(mod.head_dim)):
        return "the module's rotary is not rotate-half"
    return None


class FusedQKVLoRA(nn.Module):
    """q, k and v as one NF4 matmul plus one shared LoRA-A matmul. Holds the fused packed bytes and the expanded absmax as buffers;
    reads the adapters from the attention module's own ``q_proj``/``k_proj``/``v_proj`` at call time."""

    def __init__(self, attn):
        super().__init__()
        from bitsandbytes.functional import QuantState
        parts = (attn.q_proj, attn.k_proj, attn.v_proj)
        bases = [p.base for p in parts]
        qss = [b.weight.quant_state for b in bases]
        self.ns = tuple(int(qs.shape[0]) for qs in qss)
        K = int(qss[0].shape[1])
        packed = torch.cat([b.weight.data.reshape(-1) for b in bases]).reshape(-1, 1)
        absmaxes = [_expanded_absmax(qs) for qs in qss]
        absmax = torch.cat(absmaxes)
        #: Per projection, what :func:`disable_train_fuse_qkv` needs to hand each base its bytes back.
        self.parts = tuple((tuple(b.weight.data.shape), int(b.weight.data.numel()), int(a.numel()), qs.shape)
                           for b, a, qs in zip(bases, absmaxes, qss))
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("absmax", absmax, persistent=False)
        self.qs = QuantState(absmax=self.absmax, shape=torch.Size([sum(self.ns), K]), code=qss[0].code,
                             blocksize=qss[0].blocksize, quant_type=qss[0].quant_type, dtype=qss[0].dtype)
        self.compute_dtype = getattr(bases[0], "compute_dtype", None)
        self.scaling = float(parts[0].scaling)
        self.r = int(parts[0].lora_A.shape[0])
        self._attn = [attn]                      # a list, so the attention module is not registered as a child (no cycle)

    def dequantized(self) -> torch.Tensor:
        from bitsandbytes.functional import dequantize_4bit
        return dequantize_4bit(self.packed, self.qs)

    def forward(self, x):
        import bitsandbytes as bnb
        from ..lora import _scaled
        TRAIN_QKV_STATS["calls"] += 1
        attn = self._attn[0]
        q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
        xin = x if self.compute_dtype is None else x.to(self.compute_dtype)
        base = bnb.matmul_4bit(xin, self.packed.t(), bias=None, quant_state=self.qs)
        A = torch.cat([q.lora_A, k.lora_A, v.lora_A], 0)
        h = F.linear(x.to(A.dtype), A)
        hq, hk, hv = h.split(self.r, dim=-1)
        delta = torch.cat([F.linear(hq, q.lora_B), F.linear(hk, k.lora_B), F.linear(hv, v.lora_B)], dim=-1)
        return base.to(x.dtype) + _scaled(delta, self.scaling).to(x.dtype)


def _fused_away_forward(self, x):
    raise RuntimeError(f"{self._e4b_fused_name} was fused into its attention module's qkv_proj by enable_fast_train "
                       "(E4B_TRAIN_FUSE_QKV, on by default) and its NF4 base released; call the attention module, which runs the "
                       "fused projection, or disable_fast_train to restore the projection")


def _release_base(p, name: str) -> None:
    """Mark a q/k/v ``LoRALinear`` as fused: drop its NF4 base (the fused copy holds those bytes) and make a direct call fail with a
    clear error instead of a ``TypeError`` on ``None``. Its adapters stay the parameters. The emptied base module is kept, outside
    the module tree, for :func:`disable_train_fuse_qkv`."""
    base = p.base
    base.weight.data = base.weight.data.new_empty(0)
    for holder in (base.weight, base):                   # the nested statistics go with the bytes (Linear4bit keeps both refs)
        if getattr(holder, "quant_state", None) is not None:
            holder.quant_state = None
    p._e4b_released = [base, p.__dict__.get("forward")]  # a list: the base must not be a registered child
    p._e4b_fused_into_qkv = True
    p._e4b_fused_name = name
    p.base = None
    p.forward = types.MethodType(_fused_away_forward, p)


def _restore_base(p, packed: torch.Tensor, absmax: torch.Tensor, part, like) -> None:
    """Give a released projection its base back: the fused bytes ``packed`` (a view) with its expanded fp32 ``absmax`` slice."""
    from bitsandbytes.functional import QuantState
    base, forward = p._e4b_released
    shape, _, _, qs_shape = part
    base.weight.data = packed.view(shape)
    qs = QuantState(absmax=absmax, shape=qs_shape, code=like.code, blocksize=like.blocksize, quant_type=like.quant_type,
                    dtype=like.dtype)
    base.weight.quant_state = qs
    if hasattr(base, "quant_state"):
        base.quant_state = qs
    p.base = base
    del p._e4b_released, p._e4b_fused_into_qkv, p._e4b_fused_name
    if forward is None:
        del p.forward                                    # the instance override goes; the class forward runs again
    else:
        p.forward = forward


def enable_train_fuse_qkv(model, verbose: bool = False) -> int:
    """Fuse every eligible attention module's q/k/v for training; returns the number fused. Refused modules keep today's path, with
    the reason in ``TRAIN_QKV_STATS["refused"]``. The q/k/v ``LoRALinear`` modules stay (their adapters are the parameters); their
    NF4 bases are released, since only the fused copy runs."""
    from .qkv_fuse import _fused_forward
    TRAIN_QKV_STATS["refused"] = {}
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ not in FUSED_ATTENTION_CLASSES:
            continue
        why = _refusal(mod)
        if why is not None:
            TRAIN_QKV_STATS["refused"][name] = why
            continue
        mod.qkv_proj = FusedQKVLoRA(mod)
        mod._fused_nq, mod._fused_nk, mod._fused_nv = mod.qkv_proj.ns
        for pn in ("q_proj", "k_proj", "v_proj"):
            _release_base(getattr(mod, pn), f"{name}.{pn}" if name else pn)
        mod._e4b_unfused_forward = mod.__dict__.get("forward")      # None: the class forward
        mod.forward = types.MethodType(_fused_forward, mod)
        n += 1
    TRAIN_QKV_STATS["fused"] = n
    if verbose:
        print(f"[e4b.train_qkv_fuse] fused {n} attention modules; refused {len(TRAIN_QKV_STATS['refused'])}")
    return n


def disable_train_fuse_qkv(model) -> int:
    """Undo :func:`enable_train_fuse_qkv` (``disable_fast_train`` calls it): every fused attention module gets its q/k/v bases back,
    as views of the fused bytes with their slices of the expanded fp32 absmax, and its own forward. Returns the number restored."""
    n = 0
    for mod in list(model.modules()):
        fq = getattr(mod, "qkv_proj", None)
        if not isinstance(fq, FusedQKVLoRA):
            continue
        po = ao = 0
        for pn, part in zip(("q_proj", "k_proj", "v_proj"), fq.parts):
            _, np_, na, _ = part
            _restore_base(getattr(mod, pn), fq.packed[po:po + np_], fq.absmax[ao:ao + na], part, fq.qs)
            po, ao = po + np_, ao + na
        forward = mod._e4b_unfused_forward
        del mod.qkv_proj, mod._fused_nq, mod._fused_nk, mod._fused_nv, mod._e4b_unfused_forward
        if forward is None:
            del mod.forward
        else:
            mod.forward = forward
        n += 1
    if n:
        TRAIN_QKV_STATS["fused"] = 0
        TRAIN_QKV_STATS["refused"] = {}
    return n
