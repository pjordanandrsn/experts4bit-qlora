"""D0 shared pieces: the trained base from fixture-base, the raw checkpoint, the adapter delta, and the target formats.

Everything is fp32 on CPU. Layout facts (verified at run time, never assumed):
- an e4b expert stack stores, per projection, packed NF4 codes [E, N*K/2] (bitsandbytes: element 2j in the HIGH nibble)
  and an absmax [E, N*K/64] (fp32, or the double-quantized payload *_absmax_q/_s/_off/_code);
- gate_up is [2I, H] (gate rows then up rows), down is [H, I];
- ExpertsLoRA: delta[e] = s * B[e] @ A[e], gate_up A [E, r, H] / B [E, 2I, r], down A [E, r, I] / B [E, H, r].
"""
from __future__ import annotations

import json
import os
import sys

import torch

GNF4_KERNEL = "/home/node/work/d0/gnf4/kernel"
BLOCK = 64
NESTED = 256
WHICH = ("gate_up", "down")


def _lut():
    import bitsandbytes.functional as F
    return F.get_4bit_type("nf4", device="cpu").float()


NF4_LUT = None


def nf4_lut():
    global NF4_LUT
    if NF4_LUT is None:
        NF4_LUT = _lut()
    return NF4_LUT


def deq_nf4(packed: torch.Tensor, absmax: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """One expert's packed bytes [n*k/2] and fp32 absmax [n*k/64] -> fp32 [n, k], as lut[code] * absmax."""
    hi = (packed >> 4).long()
    lo = (packed & 0xF).long()
    codes = torch.stack([hi, lo], dim=-1).reshape(-1)
    vals = nf4_lut()[codes].view(-1, BLOCK) * absmax.float().view(-1, 1)
    return vals.view(n, k)


def absmax_from_dq(q, s, off, code) -> torch.Tensor:
    """The fp32 absmax [E, M] from the nested payload: code[q] * s[block] + off (e4b's _CompressedAbsmaxRows)."""
    E, M = q.shape
    v = code.float()[q.long()].reshape(-1)
    nb = (E * M) // NESTED
    if (E * M) % NESTED:
        raise ValueError("nested payload not a whole number of 256-blocks")
    out = (v.view(nb, NESTED) * s.float().view(nb, 1)).view(E, M)
    return out + off.float()


class Checkpoint:
    """The raw bf16 checkpoint, one tensor at a time."""

    def __init__(self, path: str):
        from safetensors import safe_open
        self.path = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._open = {}
        self._safe_open = safe_open

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(os.path.join(self.path, shard), "pt")
        return self._open[shard].get_tensor(name)

    def expert(self, layer: int, e: int, which: str) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp.experts.{e}."
        if which == "gate_up":
            return torch.cat([self.get(p + "gate_proj.weight"), self.get(p + "up_proj.weight")]).float()
        return self.get(p + "down_proj.weight").float()


class FixtureBase:
    """fixture-base: stacks.safetensors (codes + fp32 absmax) and absmax_dq.safetensors (the DQ payloads)."""

    def __init__(self, path: str, dq: bool):
        from safetensors import safe_open
        self.st = safe_open(os.path.join(path, "stacks.safetensors"), "pt")
        self.dq = safe_open(os.path.join(path, "absmax_dq.safetensors"), "pt") if dq else None
        self.keys = set(self.st.keys())

    def absmax(self, layer: int, which: str) -> torch.Tensor:
        if self.dq is None:
            return self.st.get_tensor(f"{layer}.{which}_absmax").float()
        q, s, off, code = (self.dq.get_tensor(f"{layer}.{which}_absmax{x}") for x in ("_q", "_s", "_off", "_code"))
        return absmax_from_dq(q, s, off, code)

    def codes(self, layer: int, which: str) -> torch.Tensor:
        return self.st.get_tensor(f"{layer}.{which}_proj")

    def q_train(self, layer: int, which: str, n: int, k: int):
        """Every expert's Q_train(W) for one projection: fp32 [E, n, k]."""
        packed, am = self.codes(layer, which), self.absmax(layer, which)
        E = packed.shape[0]
        return torch.stack([deq_nf4(packed[e].reshape(-1), am[e].reshape(-1), n, k) for e in range(E)])


class Adapter:
    """adapter.safetensors + adapter_manifest.json: the expert deltas, fp32."""

    def __init__(self, path: str):
        from safetensors import safe_open
        self.man = json.load(open(os.path.join(path, "adapter_manifest.json")))
        self.st = safe_open(os.path.join(path, "adapter.safetensors"), "pt")
        self.names = sorted(self.st.keys())

    def scale(self) -> float:
        v2 = self.man.get("lora_scale_applied")
        if isinstance(v2, dict) and v2.get("experts") is not None:
            return float(v2["experts"])
        setup = self.man.get("setup") or self.man.get("backend", {}).get("setup") or {}
        return float(setup["alpha"]) / float(setup["r"])

    def expert_name(self, layer: int, which: str, ab: str) -> str:
        suffix = f"layers.{layer}.mlp.experts.{which}_lora_{ab}"
        hits = [n for n in self.names if n.endswith(suffix)]
        if len(hits) != 1:
            raise KeyError(f"{suffix}: {len(hits)} matches")
        return hits[0]

    def delta(self, layer: int, which: str) -> torch.Tensor:
        A = self.st.get_tensor(self.expert_name(layer, which, "A")).float()
        B = self.st.get_tensor(self.expert_name(layer, which, "B")).float()
        return self.scale() * torch.bmm(B, A)


def targets():
    """name -> fn(x fp32 [n, k]) -> deq fp32 [n, k], the three registered target formats."""
    import bitsandbytes.functional as F
    if GNF4_KERNEL not in sys.path:
        sys.path.insert(0, GNF4_KERNEL)
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32
    from mxfp4_pack_ref import dequant_mxfp4, quantize_pack_mxfp4

    def nf4(x):
        q, st = F.quantize_4bit(x.contiguous(), blocksize=BLOCK, quant_type="nf4", compress_statistics=False)
        return F.dequantize_4bit(q, st).float().view(x.shape)

    def int4(x):
        p, s = pack_int4_b32(x)
        return dequant_int4_ref(p, s, *x.shape)

    def mxfp4(x):
        b, s = quantize_pack_mxfp4(x)
        return dequant_mxfp4(b, s).reshape(x.shape)

    return {"NF4": nf4, "INT4": int4, "MXFP4": mxfp4}


def load_cfg(path: str) -> dict:
    """The model config with transformers' own defaults filled in (config.json omits rms_norm_eps, for one); refuses
    what d0_forward does not implement."""
    from transformers import OlmoeConfig
    d = OlmoeConfig.from_pretrained(path).to_dict()
    rp = d.get("rope_parameters") or {}
    d["rope_theta"] = d.get("rope_theta") or rp.get("rope_theta")
    if rp.get("rope_type", "default") != "default" or d.get("clip_qkv") is not None or d.get("norm_topk_prob"):
        raise NotImplementedError(f"config outside d0_forward: rope {rp}, clip_qkv {d.get('clip_qkv')}, "
                                  f"norm_topk_prob {d.get('norm_topk_prob')}")
    return d
