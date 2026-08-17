# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Tiered paged KV cache (hybrid Stage 2, Phase 6).

Block-based KV over grouped-nf4-gemm's :class:`row_pool.RowPool` — the
weight tier abstraction generalized to writable rows (the Stage-2
directive's invariant 8: one tier vocabulary, never a parallel system).
Like :mod:`.kv_cache`, this is a cache OBJECT implementing the
``update``/``get_seq_length`` protocol the generation loop uses; no
attention forward is patched, so per-arch attention differences stay
upstream's problem.

Geometry that makes "paged overhead ≤2%" a non-event at batch 1 (gate
G6): each block row stores 16 tokens TOKENS-MAJOR (``[16, H_kv, D]``),
K and V in separate pool partitions, and the pool's append-only
partitions keep a layer's blocks physically contiguous. The full K for a
layer is then ``run.view(T', H, D).narrow(0, 0, T).permute(1, 0, 2)`` —
every step a VIEW, zero gather, zero copy; SDPA consumes the strided
tensor directly. Paged bookkeeping is a handful of integers per layer,
computed between forwards, off the critical path.

Demotion (the tiered part): with ``hot_window`` set, blocks beyond the
window demote VRAM→pinned-DRAM on a side stream (copy-on-demote,
publish-after-drain — the pool's contract), and reads of demoted context
stream back through a per-layer staging buffer. That path is the
constrained regime G6's nsys clause audits; with the window unset the
feature is structurally free (invariant 9): no side stream, no staging,
views only.

Batch 1 in v1 — the gate's regime. Batched paged KV arrives with Phase 9's
scheduler, where block tables become per-sequence.
"""
from __future__ import annotations

import math

import torch

BLOCK_TOKENS = 16


class TieredPagedKV:
    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, *,
                 dtype=torch.bfloat16, device: str = "cuda",
                 max_tokens: int = 4096, hot_window: int | None = None,
                 host_tokens: int = 0):
        from row_pool import RowPool

        self.L = n_layers
        self.H = n_kv_heads
        self.D = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.bt = BLOCK_TOKENS
        self.esize = torch.empty(0, dtype=dtype).element_size()
        self.row_bytes = self.bt * self.H * self.D * self.esize

        dev_blocks = math.ceil(max_tokens / self.bt)
        host_blocks = math.ceil(host_tokens / self.bt) if host_tokens else 0
        if hot_window is not None:
            if hot_window % self.bt:
                raise ValueError(f"hot_window must be a multiple of "
                                 f"{self.bt} tokens")
            if not host_blocks:
                raise ValueError("hot_window needs host_tokens > 0 to "
                                 "demote into")
        # partitions: K = [0, L), V = [L, 2L) — separate so each side's
        # token axis stays contiguous for the zero-copy view
        self.pool = RowPool(2 * self.L, dev_blocks, host_blocks,
                            self.row_bytes, device=device)
        self.hot_window = hot_window
        self._seen = [0] * n_layers            # tokens appended per layer
        self._side = (torch.cuda.Stream(self.device)
                      if (self.device.type == "cuda"
                          and hot_window is not None) else None)
        self._staging: dict[int, torch.Tensor] = {}
        self.gather_returns = 0                # non-view returns, for stats

    # ------------------------------------------------------------ plumbing --
    def _rows(self, part: int):
        """Typed [resident_tokens_capacity, H, D] view of the partition's
        resident run, plus the run's first absolute block index."""
        lo, hi = self.pool.resident_run(part)
        run = self.pool.run_view(part, lo, hi)
        t = run.flatten()
        return (t.view(self.dtype)
                .view((hi - lo) * self.bt, self.H, self.D)), lo

    def _append_tokens(self, part: int, layer: int, x):
        """x [T_new, H, D] → tail blocks of `part` (allocating as needed)."""
        t_new = x.shape[0]
        written = 0
        seen = self._seen[layer]
        while written < t_new:
            fill = seen % self.bt
            if fill == 0:
                _, row = self.pool.append(part)
            else:
                row = self.pool.row_view(part, seen // self.bt)
            take = min(self.bt - fill, t_new - written)
            dst = (row.view(self.dtype)
                   .view(self.bt, self.H, self.D)
                   .narrow(0, fill, take))
            dst.copy_(x.narrow(0, written, take))
            written += take
            seen += take

    # ---------------------------------------------------------------- API --
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[0] != 1:
            raise ValueError("TieredPagedKV v1 is batch-1 (gate G6's "
                             "regime); batched block tables land with the "
                             "Phase-9 scheduler")
        k = key_states[0].permute(1, 0, 2).to(self.dtype)   # [T, H, D]
        v = value_states[0].permute(1, 0, 2).to(self.dtype)
        self._append_tokens(layer_idx, layer_idx, k.contiguous())
        self._append_tokens(self.L + layer_idx, layer_idx, v.contiguous())
        self._seen[layer_idx] += k.shape[0]

        t = self._seen[layer_idx]
        out = []
        for part in (layer_idx, self.L + layer_idx):
            hosted = self.pool.demoted[part] * self.bt
            typed, lo = self._rows(part)
            res = typed.narrow(0, 0, t - lo * self.bt) if lo * self.bt < t \
                else typed.narrow(0, 0, 0)
            if hosted:
                res_from = lo * self.bt
                full = self._gathered(part, layer_idx, res, res_from, t)
                out.append(full.permute(1, 0, 2)[None])
            else:
                out.append(res.permute(1, 0, 2)[None])
        if layer_idx == self.L - 1:
            self._maybe_demote()
        return out[0], out[1]

    def _gathered(self, part, layer_idx, res, res_from, t):
        """Demoted prefix streamed back + resident view, concatenated.
        The copy is the price of the constrained regime and is counted."""
        n_host = self.pool.demoted[part]
        host = self.pool.host_run(part, 0, n_host)
        key = part
        need = t * self.H * self.D
        buf = self._staging.get(key)
        if buf is None or buf.numel() < need:
            buf = torch.empty(t + self.bt * 4, self.H, self.D,
                              dtype=self.dtype, device=self.device
                              ).flatten()
            self._staging[key] = buf
        dst = buf.narrow(0, 0, need).view(t, self.H, self.D)
        hosted_tokens = min(n_host * self.bt, res_from)
        dst.narrow(0, 0, hosted_tokens).copy_(
            host.view(torch.uint8).flatten().view(self.dtype)
                .view(n_host * self.bt, self.H, self.D)
                .narrow(0, 0, hosted_tokens),
            non_blocking=False)
        dst.narrow(0, hosted_tokens, t - hosted_tokens).copy_(res)
        self.gather_returns += 1
        return dst

    def _maybe_demote(self):
        if self.hot_window is None:
            return
        self.pool.settle()
        win_blocks = self.hot_window // self.bt
        for layer in range(self.L):
            done_blocks = self._seen[layer] // self.bt   # full blocks only
            upto = done_blocks - win_blocks
            if upto > 0:
                for part in (layer, self.L + layer):
                    self.pool.demote_head(part, upto, stream=self._side)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen[layer_idx]

    def stats(self) -> dict:
        s = self.pool.stats()
        s["gather_returns"] = self.gather_returns
        return s
