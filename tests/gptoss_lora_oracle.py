# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The gpt-oss expert-LoRA oracle: an unfused, bias-carrying reference that later kernel work is graded against.

gpt-oss's experts differ from the stock SwiGLU stack ``ExpertsLoRA`` can represent (``arch/gptoss.py``): per-expert
biases on both projections, an interleaved gate/up on disk, and a clamped GLU, ``(up + 1) * gate * sigmoid(alpha *
gate)`` with ``gate <= limit`` and ``|up| <= limit``. This module states the trained function those structures imply,
in e4b's own layout, so a fused route can be checked against it rather than against itself.

**The layout is e4b's**, as :meth:`GptOssExperts4bit.from_gptoss` stores the base: ``gate_up [E, 2I, H]`` in
gate-block-then-up-block row order (the checkpoint's interleaved columns, transposed and de-interleaved),
``gate_up_bias [E, 2I]`` in the same order, ``down [E, H, I]``, ``down_bias [E, H]``. The LoRA tensors are
``ExpertsLoRA``'s: ``gate_up`` A ``[E, r, H]`` / B ``[E, 2I, r]`` and ``down`` A ``[E, r, I]`` / B ``[E, H, r]``, with
the delta ``scaling * (x @ A.T) @ B.T`` added BEFORE the bias-free nonlinearity and its bias, exactly as the base's own
projection. So an adapter trained here carries its gate_up B rows in the de-interleaved order, and any conversion to
the checkpoint's (or grouped-nf4-gemm's MXFP4) interleaved layout permutes those rows: :func:`interleave_rows`.

:func:`to_hf_experts_state` gives the dense ``GptOssExperts`` state that computes the same function with the delta
merged, which is what the tests compare against (forward, and the chain rule for the backward).
"""
import torch
import torch.nn.functional as F
from torch import nn

ALPHA, LIMIT = 1.702, 7.0          # arch/gptoss.py's GPTOSS_ALPHA / GPTOSS_LIMIT, and GptOssExperts' alpha / limit


def deinterleave_rows(w: torch.Tensor) -> torch.Tensor:
    """``[..., 2I, *]`` with gate/up alternating -> gate rows then up rows (``from_gptoss``'s transform)."""
    return torch.cat([w[..., 0::2, :], w[..., 1::2, :]], dim=-2)


def interleave_rows(w: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`deinterleave_rows`: gate rows then up rows -> alternating."""
    i = w.shape[-2] // 2
    out = torch.empty_like(w)
    out[..., 0::2, :], out[..., 1::2, :] = w[..., :i, :], w[..., i:, :]
    return out


def clamped_glu(gate_up: torch.Tensor, alpha: float = ALPHA, limit: float = LIMIT) -> torch.Tensor:
    """gpt-oss's activation on a gate-block-then-up-block ``[..., 2I]``."""
    i = gate_up.shape[-1] // 2
    gate = gate_up[..., :i].clamp(max=limit)
    up = gate_up[..., i:].clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


class GptOssExpertsLoRARef(nn.Module):
    """The oracle: frozen base and biases in e4b's layout, trainable LoRA in ``ExpertsLoRA``'s, one expert at a time.

    ``forward(hidden [T, H], router_indices [T, k], routing_weights [T, k])`` has ``GptOssExperts``' signature.
    """

    def __init__(self, gate_up, gate_up_bias, down, down_bias, r: int = 8, lora_alpha: int = 16,
                 alpha: float = ALPHA, limit: float = LIMIT):
        super().__init__()
        E, n2, h = gate_up.shape
        self.num_experts, self.alpha, self.limit, self.scaling = E, float(alpha), float(limit), lora_alpha / r
        for name, t in (("gate_up", gate_up), ("gate_up_bias", gate_up_bias), ("down", down), ("down_bias", down_bias)):
            self.register_buffer(name, t.detach().clone())
        i = down.shape[-1]
        self.gate_up_lora_A = nn.Parameter(torch.zeros(E, r, h, dtype=gate_up.dtype))
        self.gate_up_lora_B = nn.Parameter(torch.zeros(E, n2, r, dtype=gate_up.dtype))
        self.down_lora_A = nn.Parameter(torch.zeros(E, r, i, dtype=gate_up.dtype))
        self.down_lora_B = nn.Parameter(torch.zeros(E, h, r, dtype=gate_up.dtype))

    def _proj(self, x, w, b, A, B):
        return F.linear(x, w) + b + self.scaling * F.linear(F.linear(x, A), B)

    def forward(self, hidden_states, router_indices, routing_weights):
        out = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            tok, pos = torch.where(router_indices == e)
            if tok.numel() == 0:
                continue
            x = hidden_states[tok]
            gu = self._proj(x, self.gate_up[e], self.gate_up_bias[e], self.gate_up_lora_A[e], self.gate_up_lora_B[e])
            h = clamped_glu(gu, self.alpha, self.limit)
            y = self._proj(h, self.down[e], self.down_bias[e], self.down_lora_A[e], self.down_lora_B[e])
            out = out.index_add(0, tok, y * routing_weights[tok, pos, None])
        return out

    def merged(self):
        """The dense base with the delta merged, in e4b's layout: ``(gate_up', down')``."""
        s = self.scaling
        return (self.gate_up + s * torch.bmm(self.gate_up_lora_B, self.gate_up_lora_A),
                self.down + s * torch.bmm(self.down_lora_B, self.down_lora_A))


def from_hf_experts_state(sd: dict) -> tuple:
    """A ``GptOssExperts`` state (``gate_up_proj [E, H, 2I]`` interleaved, input-major) -> e4b's layout."""
    gu = deinterleave_rows(sd["gate_up_proj"].transpose(1, 2))
    gub = deinterleave_rows(sd["gate_up_proj_bias"].unsqueeze(-1)).squeeze(-1)
    return gu, gub, sd["down_proj"].transpose(1, 2), sd["down_proj_bias"]


def to_hf_experts_state(gate_up, gate_up_bias, down, down_bias) -> dict:
    """e4b's layout -> a ``GptOssExperts`` state, the inverse of :func:`from_hf_experts_state`."""
    return {"gate_up_proj": interleave_rows(gate_up).transpose(1, 2).contiguous(),
            "gate_up_proj_bias": interleave_rows(gate_up_bias.unsqueeze(-1)).squeeze(-1).contiguous(),
            "down_proj": down.transpose(1, 2).contiguous(), "down_proj_bias": down_bias.contiguous()}
