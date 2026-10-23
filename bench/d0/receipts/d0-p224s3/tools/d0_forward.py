"""D0's layer-major OLMoE forward on CPU, fp32 compute (PREREG-d0.md, "The forward").

Hidden states are token-flat ([N, H] over all rows, with row offsets): attention runs per row (causal, no padding),
the MoE runs over every token at once. Expert weights come in per arm, as bf16 values; LoRA terms are applied
unmerged in fp32. Matches transformers' OlmoeDecoderLayer: q_norm/k_norm over the full projections, rotate-half RoPE,
softmax-then-top-k routing without renormalisation (norm_topk_prob false), SwiGLU over cat(gate, up).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope_tables(max_len: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def lin(x, w, lora=None):
    """x @ w.T, plus s * (x @ A.T) @ B.T when ``lora = (A, B, s)`` (fp32, unmerged)."""
    y = F.linear(x, w)
    if lora is not None:
        A, B, s = lora
        y = y + s * F.linear(F.linear(x, A), B)
    return y


class LayerWeights:
    """One decoder layer's non-expert weights (fp32) and its attention LoRA, if any."""

    def __init__(self, ck, layer: int, attn_lora: dict | None):
        p = f"model.layers.{layer}."
        g = lambda n: ck.get(p + n).float()  # noqa: E731
        self.ln1, self.ln2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        self.q, self.k, self.v, self.o = (g(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
        self.router = g("mlp.gate.weight")
        self.lora = attn_lora or {}


def attention(x, offsets, lw: LayerWeights, cfg, cos, sin, use_lora: bool):
    """x [N, H] -> x + attn(x), row by row."""
    eps, nh = cfg["rms_norm_eps"], cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // nh
    h = rms(x, lw.ln1, eps)
    L = lw.lora if use_lora else {}
    q = rms(lin(h, lw.q, L.get("q")), lw.qn, eps)
    k = rms(lin(h, lw.k, L.get("k")), lw.kn, eps)
    v = lin(h, lw.v, L.get("v"))
    out = torch.empty_like(x)
    for a, b in zip(offsets[:-1], offsets[1:]):
        T = b - a
        qr, kr, vr = (t[a:b].view(T, nh, hd).transpose(0, 1) for t in (q, k, v))
        c, s = cos[:T], sin[:T]
        qr, kr = qr * c + _rot(qr) * s, kr * c + _rot(kr) * s
        att = F.scaled_dot_product_attention(qr.unsqueeze(0), kr.unsqueeze(0), vr.unsqueeze(0), is_causal=True)
        out[a:b] = att.squeeze(0).transpose(0, 1).reshape(T, nh * hd)
    return x + lin(out, lw.o, L.get("o"))


def route(x, lw: LayerWeights, cfg):
    """(h, top-k weights [N, k], top-k ids [N, k]) after post_attention_layernorm."""
    h = rms(x, lw.ln2, cfg["rms_norm_eps"])
    probs = torch.softmax(F.linear(h, lw.router), dim=-1, dtype=torch.float32)
    topv, topi = torch.topk(probs, cfg["num_experts_per_tok"], dim=-1)
    return h, topv, topi


def moe(x, h, topv, topi, experts, expert_lora=None):
    """x + sum_k w_k * expert_k(h). ``experts(e) -> (gate_up bf16 [2I, H], down bf16 [H, I])``;
    ``expert_lora(e) -> ((A, B, s), (A, B, s))`` adds the unmerged expert deltas (REF)."""
    out = torch.zeros_like(x)
    counts = torch.zeros(experts.n, dtype=torch.int64)
    for e in range(experts.n):
        tok, slot = torch.where(topi == e)
        if tok.numel() == 0:
            continue
        counts[e] = tok.numel()
        gu, dn = experts(e)
        lg, ld = expert_lora(e) if expert_lora is not None else (None, None)
        xe = h[tok]
        gate, up = lin(xe, gu.float(), lg).chunk(2, dim=-1)
        y = lin(F.silu(gate) * up, dn.float(), ld)
        out.index_add_(0, tok, y * topv[tok, slot, None])
    return x + out, counts


def row_nll(x, offsets, norm_w, lm_head, ids, eps, ref_logits=None):
    """Per row: (sum NLL over tokens 1..T-1, count, max|dlogit|, sum|dlogit| vs ``ref_logits`` rows if given)."""
    res = []
    for r, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        logits = F.linear(rms(x[a:b], norm_w, eps), lm_head)[:-1]
        tgt = ids[a + 1:b]
        nll = F.cross_entropy(logits, tgt, reduction="sum").item()
        d = (None, None) if ref_logits is None else (
            (logits - ref_logits[r]).abs().max().item(), (logits - ref_logits[r]).abs().mean().item())
        res.append({"nll": nll, "n": b - a - 1, "dmax": d[0], "dmean": d[1], "logits": logits})
    return res
