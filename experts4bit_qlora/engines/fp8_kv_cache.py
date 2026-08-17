# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""FP8 KV cache — reference path (hybrid Stage 2, Phase 7).

Stores keys and values as E4M3 with a per-token-per-head scale and hands
stock attention a dequantized view. This is the **quality oracle**: it
measures exactly what the format costs a model, independently of any
kernel, which is why it exists before the fused kernel does. The shipping
read path will dequantize in registers inside a paged-attention kernel and
must match this object's numerics; nothing in the serving path will call
:func:`dequant_kv_fp8_ref`.

The precedent this design answers to lives one module over. ``kv_cache``'s
NF4 store shipped as a capacity feature whose decode-latency cost went
unmeasured for months, because every comparison it published was against
another configuration of itself rather than against an unquantized
baseline. So this module ships with its measurement harness
(``bench/hybrid-g7/``), and the harness carries a null control and a
positive control before it is allowed to report a verdict.

``mode`` selects the storage format, and the non-default modes are there to
make the measurement honest rather than to be shipped:

``"fp8"``    E4M3, per-token-per-head scale. The Phase-7 format.
``"off"``    bf16 passthrough. The NULL control — quantization disabled
             through the identical code path, so a nonzero quality delta
             against it isolates the format rather than the plumbing.
``"int4"``   Symmetric 4-bit, same scale axis. The directive's flag:
             evaluated, default off, never shipped on unmeasured quality.
``"crush"``  2-bit. The POSITIVE control: deliberately destructive, so a
             harness that reports "no quality change" for it is broken and
             says so loudly instead of certifying a format it cannot see.
"""
from __future__ import annotations

import torch

_MODES = ("fp8", "off", "int4", "crush")


def _quant_int(x: torch.Tensor, n_bits: int):
    """Symmetric integer quant on the same per-row axis as the fp8 path.

    Returns the scale WITHOUT a trailing singleton, matching
    ``quantize_kv_fp8`` exactly: every format in this module stores its
    scale as ``payload.shape[:-1]``, so growth, load and packing share one
    axis convention. A keepdim here would put the token axis at -2 for one
    format and -1 for the others — which is a broadcast error at best and a
    silently mis-scaled cache at worst.
    """
    qmax = 2 ** (n_bits - 1) - 1
    xf = x.float()
    amax = xf.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    q = torch.round(xf / scale.unsqueeze(-1)).clamp(-qmax - 1, qmax)
    return q, scale


class Fp8KVCache:
    """transformers-protocol cache with quantized storage.

    Batch and layer agnostic; keeps one packed store per layer. Sequence
    growth is a concatenate, as in the stock dynamic cache — paged block
    storage is Phase 6's object, and joining the two is the fused
    kernel's job, not the oracle's.
    """

    def __init__(self, mode: str = "fp8", compute_dtype=torch.bfloat16,
                 key_mode: str | None = None, key_group: int | None = None,
                 value_group: int | None = None):
        """``key_mode`` overrides the format for KEYS only.

        The asymmetry is measured, not stylistic: this repo's NF4-KV work
        found keys roughly 6x more expensive to quantize than values
        (+0.083 vs +0.013 perplexity), and the Phase-7 outlier probe says
        why — key rows carry ``amax/rms`` around 10 where value rows sit
        near 3. A caller trading fidelity for capacity should know it can
        spend its bytes asymmetrically.
        """
        for m in (mode, key_mode or mode):
            if m not in _MODES:
                raise ValueError(f"mode must be one of {_MODES}, got {m!r}")
        self.mode = mode
        self.key_mode = key_mode or mode
        # sub-row scale groups; None = one scale per (token, head) row.
        # Finer KEY groups are the measured middle ground between full-row
        # fp8 (cheap, but +0.574% ppl on one model) and bf16 keys (free,
        # but gives back a third of the compression).
        self.key_group = key_group
        self.value_group = value_group
        self.compute_dtype = compute_dtype
        self._k: dict[int, tuple] = {}
        self._v: dict[int, tuple] = {}
        self._seen: dict[int, int] = {}

    # ------------------------------------------------------------ storage --
    def _store(self, x: torch.Tensor, mode: str | None = None,
               group: int | None = None):
        mode = mode or self.mode
        if mode == "off":
            return ("raw", x.to(self.compute_dtype))
        if mode == "fp8":
            from fp8_kv import quantize_kv_fp8
            q, s = quantize_kv_fp8(x, group=group)
            slot = ("fp8", q, s)
        else:
            bits = 4 if mode == "int4" else 2
            q, s = _quant_int(x, bits)
            slot = ("int", q.to(torch.int8), s)
        # ONE scale convention, enforced where it is created rather than
        # discovered as a broadcast error three frames away: the scale is
        # the payload's shape minus its last axis, so tokens live at dim -1
        # for every format this module knows.
        grouped = group is not None and group != x.shape[-1]
        want = (slot[1].shape[:-1] + (x.shape[-1] // group,) if grouped
                else slot[1].shape[:-1])
        if tuple(slot[2].shape) != tuple(want):
            raise AssertionError(
                f"{mode}: scale shape {tuple(slot[2].shape)} is not the "
                f"payload's {tuple(want)} — one scale per group, and per "
                f"row when ungrouped (payload minus its last axis)")
        return slot

    def _load(self, slot):
        kind = slot[0]
        if kind == "raw":
            return slot[1]
        if kind == "fp8":
            from fp8_kv import dequant_kv_fp8_ref
            return dequant_kv_fp8_ref(slot[1], slot[2],
                                      dtype=self.compute_dtype)
        return (slot[1].float() * slot[2].unsqueeze(-1)).to(
            self.compute_dtype)

    def _cat(self, old, new_x, mode=None, group=None):
        """Append raw values to a stored slot, re-quantizing only the new
        tokens: an already-stored token must never be quantized twice, which
        would compound error the serving path does not pay.

        The two tensors grow on DIFFERENT axes and that is not cosmetic:
        the payload is ``[..., T, D]`` so tokens are dim -2, while the scale
        dropped the ``D`` axis and is ``[..., T]`` so its tokens are dim -1.
        Growing both on -2 appends scales along HEADS — which only fails
        loudly because head count and token count differ, and would silently
        corrupt the cache on the step where they happen to match.
        """
        if old is None:
            return self._store(new_x, mode, group)
        kind = old[0]
        fresh = self._store(new_x, mode, group)
        if kind == "raw":
            return ("raw", torch.cat([old[1], fresh[1]], dim=-2))
        # The scale's token axis DEPENDS ON ITS LAYOUT and must be derived,
        # never hardcoded: ungrouped scales are [..., T] (tokens at -1),
        # grouped scales are [..., T, n_groups] (tokens at -2). Hardcoding
        # -1 concatenates grouped scales along GROUPS — no error, wrong
        # cache. Two earlier bugs in this module were this same mistake.
        s_axis = -1 if old[2].ndim == old[1].ndim - 1 else -2
        out = (kind,
               torch.cat([old[1], fresh[1]], dim=-2),      # payload: -2
               torch.cat([old[2], fresh[2]], dim=s_axis))
        # tokens grew by the same amount on both, or the layout was misread
        if out[1].shape[-2] != out[2].shape[s_axis]:
            raise AssertionError(
                f"token counts diverged after growth: payload "
                f"{out[1].shape[-2]} vs scale {out[2].shape[s_axis]} "
                f"(scale layout {tuple(out[2].shape)})")
        return out

    # ---------------------------------------------------------------- API --
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._k[layer_idx] = self._cat(self._k.get(layer_idx), key_states,
                                       self.key_mode, self.key_group)
        self._v[layer_idx] = self._cat(self._v.get(layer_idx), value_states,
                                       self.mode, self.value_group)
        self._seen[layer_idx] = (self._seen.get(layer_idx, 0)
                                 + key_states.shape[-2])
        return self._load(self._k[layer_idx]), self._load(self._v[layer_idx])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen.get(layer_idx, 0)

    def get_mask_sizes(self, query_length: int, layer_idx: int):
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> int:
        return -1

    @property
    def is_sliding(self) -> list:
        return [False] * max(1, len(self._seen))

    @property
    def is_compileable(self) -> bool:
        return False

    def bytes_per_token(self, n_kv_heads: int, head_dim: int,
                        n_layers: int) -> float:
        """Stored bytes per token across all layers, K and V — the number
        the batch ceiling is computed from. Counts the scale tail, which is
        what separates the honest ratio from a flat 2x."""
        def per_row(m, g):
            n_scales = (head_dim // g) if g else 1
            return {"off": head_dim * 2, "fp8": head_dim + 4 * n_scales,
                    "int4": head_dim / 2 + 4 * n_scales,
                    "crush": head_dim / 4 + 4 * n_scales}[m]
        # K and V are counted separately so a split-precision configuration
        # reports its real cost rather than twice one side's
        return (per_row(self.key_mode, self.key_group)
                + per_row(self.mode, self.value_group)) * n_kv_heads * n_layers
