#!/usr/bin/env python3
"""sd1_eagle3.py -- lane SD1 (e4b#1313): an EAGLE-3 draft head's greedy chains, in plain PyTorch (bench/sd1/PREREG-sd1.md).

Why not the `speculators` package: speculators 0.8.0 requires torch >= 2.9 and transformers < 5.17. The box's e4b stack is
torch 2.8.0 + transformers 5.17.0, and installing it would replace the stack under test. The head's own repo ships remote
code that is not executed. This file implements the inference semantics of vLLM's EAGLE-3 path
(vllm/model_executor/models/llama_eagle3.py and v1/spec_decode/llm_base_proposer.py, read 2026-10-10) for the pinned head
`RedHatAI/Qwen3-30B-A3B-speculator.eagle3` @6afc5aa2. That head has one Llama layer, `norm_before_residual`, and no fc norms.

**Inputs.** The target's auxiliary hidden states are the residual stream entering decoder layers (2, 24, 45), vLLM's default
(2, L // 2, L - 3) for L = 48. They are concatenated low|mid|high -> `fc` -> 2048.

**The context pass.** Index j takes token x[j + 1] (the token the target produced from position j), the hidden fc(aux[j]) and
RoPE position j. It is causal over the indices.

**One layer.**
- e = input_layernorm(embed(token)); h = hidden_norm(hidden); residual = h; x = cat(e, h), width 4096.
- attn = o_proj(GQA attention(q, k, v of x, RoPE)); residual += attn; mlp = MLP(post_attention_layernorm(residual)).
- prenorm = mlp + residual; logits = lm_head(norm(prenorm)) over the 64,000 draft ids. target id = draft id + d2t[draft id].

**Chain step s >= 2 at index t.** It takes the previous step's drafted target id and its PRENORM hidden, at RoPE position
t + s - 1. It attends to the context keys 0..t plus the chain's own steps 2..s, and never to context index t + 1 or later.

The chain at index t proposes x[t + 2 .. t + 1 + K], given the target's tokens through x[t + 1]. Greedy chains are
prefix-consistent, so one K = 5 chain per index gives every k <= 5.
"""
from __future__ import annotations

import math

import torch

NAMES = ("d2t", "embed_tokens.weight", "fc.weight", "layers.0.hidden_norm.weight", "layers.0.input_layernorm.weight",
         "layers.0.mlp.down_proj.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight",
         "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
         "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight", "lm_head.weight", "norm.weight")
AUX_LAYERS = (2, 24, 45)


class Eagle3Draft:
    def __init__(self, tensors: dict, *, n_heads: int = 32, n_kv: int = 4, head_dim: int = 128, rope_theta: float = 10000.0,
                 eps: float = 1e-6, device="cpu", dtype=torch.bfloat16):
        missing = [n for n in NAMES if n not in tensors]
        if missing:
            raise ValueError(f"the head lacks {missing}")
        self.w = {k: (v.to(device) if k == "d2t" else v.to(device=device, dtype=dtype)) for k, v in tensors.items() if k in NAMES}
        self.nh, self.nkv, self.hd, self.theta, self.eps, self.dtype = n_heads, n_kv, head_dim, rope_theta, eps, dtype
        self.device = device

    # ---- pieces -------------------------------------------------------------------------------------------------
    def _rms(self, x, w):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(self.dtype) * w

    def _rope(self, x, pos):
        """x [N, heads, hd]; Llama/NeoX rotate-half RoPE at integer positions pos [N]."""
        half = self.hd // 2
        inv = 1.0 / (self.theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
        ang = pos.to(torch.float32)[:, None] * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
        x32 = x.float()
        rot = torch.cat([-x32[..., half:], x32[..., :half]], -1)
        return (x32 * cos + rot * sin).to(x.dtype)

    def fc(self, aux):
        """aux [N, 3 * 2048], the target's residual stream at AUX_LAYERS, low|mid|high."""
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
        """attn [N, heads, hd] -> (target ids [N], prenorm [N, 2048])."""
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

    def _attend_ref(self, q, keys, vals, mask):
        """The reference (chain_at only, N = 1): q [N, H, hd]; keys/vals [N, M, kv, hd], explicit per row; mask [N, M]."""
        rep = self.nh // self.nkv
        k = keys.repeat_interleave(rep, dim=2).float()
        v = vals.repeat_interleave(rep, dim=2).float()
        s = torch.einsum("nhd,nmhd->nhm", q.float(), k) / math.sqrt(self.hd)
        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        return torch.einsum("nhm,nmhd->nhd", s.softmax(-1), v).to(self.dtype)

    def _attend_shared(self, q, kc, vc, ctx_mask, k_extra=None, v_extra=None):
        """The batched path (Amendment 1). The context keys/values kc/vc [M, kv, hd] are SHARED by every query row
        (ctx_mask [N, M]); k_extra/v_extra [N, S, kv, hd] are each row's own chain steps, all visible. The scores are
        [N, H, M + S] and nothing of size N x M x H x hd is built. The first cut built it, about 50 GB at a
        1,750-position chat row."""
        rep, scale = self.nh // self.nkv, 1.0 / math.sqrt(self.hd)
        qf = q.float()
        K = kc.repeat_interleave(rep, dim=1).float()
        V = vc.repeat_interleave(rep, dim=1).float()
        s = torch.einsum("nhd,mhd->nhm", qf, K) * scale
        s = s.masked_fill(~ctx_mask[:, None, :], float("-inf"))
        if k_extra is not None:
            Ke = k_extra.repeat_interleave(rep, dim=2).float()
            s = torch.cat([s, torch.einsum("nhd,nshd->nhs", qf, Ke) * scale], -1)
        p = s.softmax(-1)
        M = kc.shape[0]
        out = torch.einsum("nhm,mhd->nhd", p[..., :M], V)
        if k_extra is not None:
            out = out + torch.einsum("nhs,nshd->nhd", p[..., M:], v_extra.repeat_interleave(rep, dim=2).float())
        return out.to(self.dtype)

    # ---- batched chains (the box) -------------------------------------------------------------------------------
    @torch.no_grad()
    def chains(self, tokens, aux, K: int = 5):
        """tokens [L] (prompt + generated, target ids); aux [L - 1, 6144] (the target's states at positions 0..L-2).
        Returns chains [L - 1, K]: chain t proposes tokens[t + 2 .. t + 1 + K]."""
        L = tokens.shape[0]
        n = L - 1
        if aux.shape[0] != n:
            raise ValueError(f"aux has {aux.shape[0]} positions, expected {n}")
        tok = tokens[1:].to(self.device)
        pos = torch.arange(n, device=self.device)
        q, kc, vc, h = self._qkv(tok, self.fc(aux.to(self.device)), pos)
        causal = torch.ones(n, n, dtype=torch.bool, device=self.device).tril()
        ids, prenorm = self._after_attn(self._attend_shared(q, kc, vc, causal), h)
        out = [ids]
        ck, cv = [], []                                  # the chain's own keys/values, steps 2..s
        for s in range(2, K + 1):
            q, k, v, h = self._qkv(out[-1], prenorm, pos + (s - 1))
            ck.append(k)
            cv.append(v)
            a = self._attend_shared(q, kc, vc, causal, torch.stack(ck, 1), torch.stack(cv, 1))
            ids, prenorm = self._after_attn(a, h)
            out.append(ids)
        return torch.stack(out, 1).cpu()

    # ---- the step-by-step reference (tests) ---------------------------------------------------------------------
    @torch.no_grad()
    def chain_at(self, tokens, aux, t: int, K: int = 5):
        """One chain at index t, one step at a time with an explicit cache: the oracle for chains()."""
        tok = tokens[1:t + 2].to(self.device)
        pos = torch.arange(t + 1, device=self.device)
        q, k, v, h = self._qkv(tok, self.fc(aux[:t + 1].to(self.device)), pos)
        keys, vals = [k], [v]
        mask = torch.ones(1, t + 1, dtype=torch.bool, device=self.device)
        a = self._attend_ref(q[t:t + 1], k[None, :t + 1], v[None, :t + 1], mask)
        did, prenorm = self._after_attn(a, h[t:t + 1])
        out = [int(did)]
        for s in range(2, K + 1):
            q, k1, v1, h = self._qkv(did, prenorm, torch.tensor([t + s - 1], device=self.device))
            keys.append(k1)
            vals.append(v1)
            kk = torch.cat(keys, 0)[None]
            vv = torch.cat(vals, 0)[None]
            a = self._attend_ref(q, kk, vv, torch.ones(1, kk.shape[1], dtype=torch.bool, device=self.device))
            did, prenorm = self._after_attn(a, h)
            out.append(int(did))
        return out
