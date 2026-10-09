# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""One fused q/k/v projection for TRAINING attention (P129; opt-in: ``E4B_TRAIN_FUSE_QKV=1``, off by default).

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

Anything that does not match is refused and keeps today's path (``TRAIN_QKV_STATS["refused"]`` says why). With the knob unset
nothing here runs.
"""
from __future__ import annotations

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

#: What :func:`enable_train_fuse_qkv` did: modules fused, and every refused module's reason.
TRAIN_QKV_STATS = {"fused": 0, "refused": {}, "calls": 0}


def train_fuse_qkv_requested() -> bool:
    """``E4B_TRAIN_FUSE_QKV=1``; anything else (unset included) is off."""
    return os.environ.get("E4B_TRAIN_FUSE_QKV", "0").strip() == "1"


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
    if type(mod).__name__ != "Qwen3MoeAttention":
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
        absmax = torch.cat([_expanded_absmax(qs) for qs in qss])
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


def enable_train_fuse_qkv(model, verbose: bool = False) -> int:
    """Fuse every eligible attention module's q/k/v for training; returns the number fused. Refused modules keep today's path, with
    the reason in ``TRAIN_QKV_STATS["refused"]``. The q/k/v ``LoRALinear`` modules stay (their adapters are the parameters); their
    NF4 bases are released, since only the fused copy runs."""
    from .qkv_fuse import _fused_forward
    TRAIN_QKV_STATS["refused"] = {}
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Qwen3MoeAttention":
            continue
        why = _refusal(mod)
        if why is not None:
            TRAIN_QKV_STATS["refused"][name] = why
            continue
        mod.qkv_proj = FusedQKVLoRA(mod)
        mod._fused_nq, mod._fused_nk, mod._fused_nv = mod.qkv_proj.ns
        for p in (mod.q_proj, mod.k_proj, mod.v_proj):
            p._e4b_fused_into_qkv = True
            p.base = None                         # the fused copy holds these bytes now; nothing calls the unfused path
        mod._e4b_unfused_forward = mod.forward
        mod.forward = types.MethodType(_fused_forward, mod)
        n += 1
    TRAIN_QKV_STATS["fused"] = n
    if verbose:
        print(f"[e4b.train_qkv_fuse] fused {n} attention modules; refused {len(TRAIN_QKV_STATS['refused'])}")
    return n
