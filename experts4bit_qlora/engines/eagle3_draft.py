# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The EAGLE-3 draft in the paged server, for ONE speculating sequence (lane SD2, ``bench/sd2/PREREG-sd2.md`` B3, B4).

The arithmetic is SD1's (``bench/sd1/sd1_eagle3.py``), which implements vLLM's EAGLE-3 inference semantics for the pinned
head ``RedHatAI/Qwen3-30B-A3B-speculator.eagle3`` @6afc5aa2: one Llama layer, ``norm_before_residual``, no fc norms.
SD1 measured its acceptance on the target. This module repeats that arithmetic with a KV cache, so its chains equal
``sd1_eagle3.Eagle3Draft.chain_at`` on the same inputs (``tests/test_eagle3_draft.py``).

**The context convention.** Context entry ``j`` takes the token ``x[j + 1]`` (the token the target produced from position
``j``), the hidden ``fc(aux[j])`` (the target's residual stream entering layers 2, 24 and 45 at position ``j``) and RoPE
position ``j``. A query at position ``p`` attends to the cache's entries ``0..p``.

**The cycle.** The last emitted token sits at position ``base``, in neither KV yet:
1. the target verifies rows ``[x[base], d_1 .. d_k]`` at positions ``base .. base + k``. Its argmax ``g_0 .. g_k`` and its
   states ``aux_0 .. aux_k`` come back, and ``a`` is the number of leading matches ``d_{i+1} == g_i``;
2. :meth:`Eagle3Drafter.extend_and_draft` writes context entries ``base .. base + n - 1`` from ``(g_i, aux_i)``: every
   verified row at a static shape, SGLang's draft-extend. Rows past ``a`` hold rejected work and are overwritten by the
   next cycle, whose entries start at ``base + a + 1``;
3. the chain starts from row ``a``'s output: ``d'_1``, then ``k - 1`` single steps at positions ``base + a + 1 ..``,
   each writing its own entry first and attending to everything up to itself.

Every input of step 2 is a device tensor, ``a`` and ``base`` included, and every shape is static, so the runner can
capture it as a CUDA graph per verified row count.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch

#: the head SD1 measured (``bench/sd1/PREREG-sd1.md``, "The draft head (pinned)")
HEAD_REPO = "RedHatAI/Qwen3-30B-A3B-speculator.eagle3"
HEAD_REVISION = "6afc5aa2477b923467fb9a8d906782b984a9a6ba"
HEAD_SHA256 = "d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa"
HEAD_BYTES = 1_044_539_336

NAMES = ("d2t", "embed_tokens.weight", "fc.weight", "layers.0.hidden_norm.weight", "layers.0.input_layernorm.weight",
         "layers.0.mlp.down_proj.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight",
         "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
         "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight", "lm_head.weight", "norm.weight")


def aux_layers(n_layers: int) -> tuple:
    """The layers whose input residual stream feeds the draft: ``(2, L // 2, L - 3)``, vLLM's and SGLang's default when
    the head names none (``eagle_aux_hidden_state_layer_ids`` null). (2, 24, 45) at Qwen3-30B-A3B's 48 layers."""
    if n_layers < 6:
        raise ValueError(f"an EAGLE-3 target needs at least 6 layers, got {n_layers}")
    return (2, n_layers // 2, n_layers - 3)


def head_geometry(config: dict) -> dict:
    """The draft layer's geometry from the head's ``config.json``, refusing what this module does not implement."""
    lc = config.get("transformer_layer_config") or {}
    refusals = []
    if config.get("speculators_model_type") != "eagle3":
        refusals.append(f"speculators_model_type {config.get('speculators_model_type')!r}, not 'eagle3'")
    if not config.get("norm_before_residual"):
        refusals.append("norm_before_residual is not set")
    if config.get("eagle_aux_hidden_state_layer_ids") is not None:
        refusals.append("explicit eagle_aux_hidden_state_layer_ids (only the (2, L // 2, L - 3) default is implemented)")
    if lc.get("num_hidden_layers") != 1:
        refusals.append(f"{lc.get('num_hidden_layers')} draft layers (one is implemented)")
    if lc.get("rope_scaling") is not None:
        refusals.append("rope_scaling")
    if lc.get("hidden_act", "silu") != "silu" or lc.get("attention_bias") or lc.get("mlp_bias"):
        refusals.append("an activation other than silu, or a bias")
    if refusals:
        raise ValueError("unsupported EAGLE-3 head: " + "; ".join(refusals))
    return {"n_heads": int(lc["num_attention_heads"]), "n_kv": int(lc["num_key_value_heads"]),
            "head_dim": int(lc["head_dim"]), "rope_theta": float(lc["rope_theta"]), "eps": float(lc["rms_norm_eps"]),
            "hidden": int(lc["hidden_size"]), "draft_vocab": int(config["draft_vocab_size"])}


def load_head(path: str, *, sha256: str = HEAD_SHA256, size: int = HEAD_BYTES):
    """``(tensors, geometry)`` from a head directory, checked against the pinned bytes. The size is read through any
    symlink (a Hugging Face cache snapshot is one; SD1's first run measured the link). Anything else raises."""
    st_path = os.path.join(path, "model.safetensors")
    got = os.stat(st_path).st_size                    # os.stat follows symlinks
    if got != size:
        raise ValueError(f"EAGLE-3 head {st_path}: {got} bytes, expected {size}")
    h = hashlib.sha256()
    with open(st_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != sha256:
        raise ValueError(f"EAGLE-3 head {st_path}: sha256 {h.hexdigest()}, expected {sha256}")
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        geometry = head_geometry(json.load(f))
    from safetensors.torch import load_file
    return load_file(st_path), geometry


class Eagle3Drafter:
    """One speculating sequence's EAGLE-3 draft: the head's weights, a contiguous KV cache of ``max_positions`` entries,
    and the three operations of the cycle (:meth:`prefill`, :meth:`extend_and_draft`, the chain inside both)."""

    def __init__(self, tensors: dict, *, k: int, max_positions: int, n_heads: int = 32, n_kv: int = 4,
                 head_dim: int = 128, rope_theta: float = 10000.0, eps: float = 1e-6, device="cuda",
                 dtype=torch.bfloat16, **_unused_geometry):
        missing = [n for n in NAMES if n not in tensors]
        if missing:
            raise ValueError(f"the head lacks {missing}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.w = {n: (tensors[n].to(device) if n == "d2t" else tensors[n].to(device=device, dtype=dtype))
                  for n in NAMES}
        self.k, self.max_positions = int(k), int(max_positions)
        self.nh, self.nkv, self.hd, self.theta, self.eps, self.dtype = n_heads, n_kv, head_dim, rope_theta, eps, dtype
        self.device = torch.device(device)
        if self.nh % self.nkv:
            raise ValueError(f"{self.nh} heads over {self.nkv} KV heads")
        self.kc = torch.zeros(self.max_positions, self.nkv, self.hd, dtype=dtype, device=self.device)
        self.vc = torch.zeros_like(self.kc)
        self._keys = torch.arange(self.max_positions, device=self.device)
        half = self.hd // 2
        self._inv = 1.0 / (self.theta ** (torch.arange(0, half, device=self.device, dtype=torch.float32) / half))

    # ---- SD1's pieces, unchanged in arithmetic ------------------------------------------------------------------
    def _rms(self, x, w):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(self.dtype) * w

    def _rope(self, x, pos):
        """x [N, heads, hd]; rotate-half RoPE at integer positions pos [N]."""
        half = self.hd // 2
        ang = pos.to(torch.float32)[:, None] * self._inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
        x32 = x.float()
        rot = torch.cat([-x32[..., half:], x32[..., :half]], -1)
        return (x32 * cos + rot * sin).to(x.dtype)

    def fc(self, aux):
        """aux [N, 3 * hidden], the target's residual stream at the auxiliary layers, low|mid|high."""
        return aux.to(self.dtype) @ self.w["fc.weight"].T

    def _qkv(self, tok, hid, pos):
        w = self.w
        e = self._rms(w["embed_tokens.weight"][tok], w["layers.0.input_layernorm.weight"])
        h = self._rms(hid.to(self.dtype), w["layers.0.hidden_norm.weight"])
        x = torch.cat([e, h], -1)
        n = x.shape[0]
        q = (x @ w["layers.0.self_attn.q_proj.weight"].T).view(n, self.nh, self.hd)
        k = (x @ w["layers.0.self_attn.k_proj.weight"].T).view(n, self.nkv, self.hd)
        v = (x @ w["layers.0.self_attn.v_proj.weight"].T).view(n, self.nkv, self.hd)
        return self._rope(q, pos), self._rope(k, pos), v, h

    def _after_attn(self, attn, residual):
        """attn [N, heads, hd] -> (target ids [N], prenorm [N, hidden])."""
        w = self.w
        n = attn.shape[0]
        residual = residual + attn.reshape(n, self.nh * self.hd) @ w["layers.0.self_attn.o_proj.weight"].T
        m = self._rms(residual, w["layers.0.post_attention_layernorm.weight"])
        m = (torch.nn.functional.silu(m @ w["layers.0.mlp.gate_proj.weight"].T) * (m @ w["layers.0.mlp.up_proj.weight"].T)) \
            @ w["layers.0.mlp.down_proj.weight"].T
        prenorm = m + residual
        logits = self._rms(prenorm, w["norm.weight"]) @ w["lm_head.weight"].T
        did = logits.float().argmax(-1)
        return did + w["d2t"][did], prenorm

    # ---- the cache ----------------------------------------------------------------------------------------------
    def _write(self, pos, k, v):
        self.kc.index_copy_(0, pos, k)
        self.vc.index_copy_(0, pos, v)

    def _attend(self, q, pos, upto: int | None = None):
        """q [N, heads, hd] at positions pos [N] -> [N, heads, hd]: each row over the cache's entries 0..pos. fp32
        scores and softmax, as SD1's reference. ``upto`` bounds the keys read (prefill); a graph reads them all,
        masked, so its shapes are static."""
        M = self.max_positions if upto is None else upto
        rep = self.nh // self.nkv
        n = q.shape[0]
        qg = q.float().view(n, self.nkv, rep, self.hd)
        K, V = self.kc[:M].float(), self.vc[:M].float()
        s = torch.einsum("ngrd,mgd->ngrm", qg, K) / math.sqrt(self.hd)
        mask = self._keys[:M][None, :] <= pos[:, None]                     # [N, M]
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        o = torch.einsum("ngrm,mgd->ngrd", s.softmax(-1), V)
        return o.reshape(n, self.nh, self.hd).to(self.dtype)

    def _chain(self, did, prenorm, t):
        """The chain from a context entry at device position ``t`` whose output was (did, prenorm): k ids."""
        out = [did]
        for s in range(2, self.k + 1):
            p = (t + (s - 1)).reshape(1)
            q, k1, v1, h = self._qkv(did, prenorm, p)
            self._write(p, k1, v1)
            did, prenorm = self._after_attn(self._attend(q, p), h)
            out.append(did)
        return torch.cat(out)

    # ---- the cycle ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, tokens, aux, *, chunk: int = 256):
        """The prompt's context: ``tokens`` [P] = x[1..P] (the prompt's tokens after the first, then the first
        generated token), ``aux`` [P, 3 * hidden] the target's states at positions 0..P-1. Writes entries 0..P-1 and
        returns the first k drafts (device ids). Eager: it runs once per request, at a length the graphs do not fix."""
        P = int(tokens.shape[0])
        if aux.shape[0] != P:
            raise ValueError(f"aux has {aux.shape[0]} positions, expected {P}")
        if P + self.k > self.max_positions:
            raise ValueError(f"a {P}-token context and a {self.k}-step chain pass the cache's {self.max_positions}")
        tok = tokens.to(self.device)
        pos = torch.arange(P, device=self.device)
        q, k, v, h = self._qkv(tok, self.fc(aux.to(self.device)), pos)
        self._write(pos, k, v)
        last = slice(P - 1, P)                           # only the last row's output starts the chain
        a = self._attend(q[last], pos[last], upto=P)
        did, prenorm = self._after_attn(a, h[last])
        return self._chain(did, prenorm, pos[P - 1])

    @torch.no_grad()
    def extend_and_draft(self, g, aux_v, base, a):
        """After a verify: ``g`` [n] the target's argmax at its rows, ``aux_v`` [n, 3 * hidden] its states there,
        ``base`` the position of the verify's first row and ``a`` the accepted count (device long scalars). Writes
        entries base .. base + n - 1 and returns the next k drafts, chained from row ``a`` (device ids)."""
        n = g.shape[0]
        pos = base.reshape(1) + torch.arange(n, device=self.device)
        q, k, v, h = self._qkv(g, self.fc(aux_v), pos)
        self._write(pos, k, v)
        ai = a.reshape(1).to(torch.long)
        qa, pa, ha = q.index_select(0, ai), pos.index_select(0, ai), h.index_select(0, ai)
        did, prenorm = self._after_attn(self._attend(qa, pa), ha)
        return self._chain(did, prenorm, pa[0])


class AuxStates:
    """Forward pre-hooks on the target's decoder layers :func:`aux_layers` that copy each row's input residual stream
    into static buffers (SD2 B3, SD1's ``AuxCapture``):
    - ``dec`` [max_rows, 3 * hidden]: a decode or verify step's rows (inputs ``[b, 1, hidden]``), rows 0..b-1;
    - ``pre`` [max_prompt, 3 * hidden]: a prefill chunk's positions (input ``[1, T, hidden]``) at its offset.

    ``mode`` selects the target: ``"decode"``, ``("prefill", start)``, or None (nothing copied). The copies are
    ordinary ops, so a CUDA graph captured with ``mode`` set records them. :meth:`install` returns the hook count or
    raises."""

    def __init__(self, model, *, max_rows: int, max_prompt: int, device=None, dtype=torch.bfloat16):
        layers = getattr(getattr(model, "model", None), "layers", None)
        if not isinstance(layers, torch.nn.ModuleList):
            raise ValueError("AuxStates needs model.model.layers, a ModuleList of decoder layers")
        cfg = getattr(model, "config", None)
        hidden = getattr(getattr(cfg, "text_config", None) or cfg, "hidden_size", None)
        if not hidden:
            raise ValueError("AuxStates needs config.hidden_size")
        self.layers_idx = aux_layers(len(layers))
        self._layers = layers
        self.hidden = int(hidden)
        dev = device if device is not None else next(model.parameters()).device
        self.dec = torch.zeros(max_rows, 3 * self.hidden, dtype=dtype, device=dev)
        self.pre = torch.zeros(max_prompt, 3 * self.hidden, dtype=dtype, device=dev)
        self.mode = None
        self._handles = []

    def _hook(self, j: int):
        lo, hi = j * self.hidden, (j + 1) * self.hidden

        def pre_hook(module, args, kwargs):
            mode = self.mode
            if mode is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            if mode == "decode":
                self.dec[:hs.shape[0], lo:hi].copy_(hs[:, -1, :])
            else:
                start = int(mode[1])
                self.pre[start:start + hs.shape[1], lo:hi].copy_(hs[0])
            return None
        return pre_hook

    def install(self) -> int:
        if self._handles:
            raise RuntimeError("AuxStates is already installed")
        for j, li in enumerate(self.layers_idx):
            self._handles.append(self._layers[li].register_forward_pre_hook(self._hook(j), with_kwargs=True))
        return len(self._handles)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
