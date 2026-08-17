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
            # append margin: demotion is asynchronous (enqueue → settle a
            # step later), so the device ring needs slack beyond the window
            # or the first append after it fills has nowhere to go
            # (max_tokens == hot_window deadlocks the ring — Bugbot)
            if dev_blocks < hot_window // self.bt + 2:
                raise ValueError(
                    f"max_tokens ({max_tokens}) must give the device ring "
                    f"at least 2 blocks of slack over hot_window "
                    f"({hot_window}) — demotion settles asynchronously")
        # partitions: K = [0, L), V = [L, 2L) — separate so each side's
        # token axis stays contiguous for the zero-copy view
        self.pool = RowPool(2 * self.L, dev_blocks, host_blocks,
                            self.row_bytes, device=device)
        self.hot_window = hot_window
        self._seen = [0] * n_layers            # tokens appended per layer
        # per-partition typed view of the WHOLE device partition, built once
        # (the pool tensor never reallocates). The everything-fits return is
        # then one narrow + one permute — python bookkeeping was 6.7% of a
        # 0.6B model's step before this, vs a 2% gate bar
        self._typed_dev = [
            self.pool.dev[p].flatten().view(self.dtype)
            .view(dev_blocks * self.bt, self.H, self.D)
            for p in range(2 * self.L)
        ]
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
        seen = self._seen[layer_idx]
        t_new = key_states.shape[2]
        pk, pv = layer_idx, self.L + layer_idx
        # Decode fast path: one token, nothing demoted, no ring wrap. The
        # measured cost of this method is python DISPATCH COUNT, not any
        # kernel (SDPA is stride-indifferent and narrow+copy beats cat —
        # see bench/hybrid-g6): keep it to a handful of torch calls.
        if (t_new == 1 and self.pool.head[pk] == 0
                and self.pool.tail[pk] <= self.pool.device_rows
                and key_states.dtype == self.dtype):
            if seen % self.bt == 0:
                self.pool.append(pk)
                self.pool.append(pv)
            t = seen + 1
            kd = self._typed_dev[pk]
            vd = self._typed_dev[pv]
            kd.narrow(0, seen, 1).copy_(key_states[0].permute(1, 0, 2))
            vd.narrow(0, seen, 1).copy_(value_states[0].permute(1, 0, 2))
            self._seen[layer_idx] = t
            out_k = kd.narrow(0, 0, t).permute(1, 0, 2)[None]
            out_v = vd.narrow(0, 0, t).permute(1, 0, 2)[None]
            if layer_idx == self.L - 1:
                self._maybe_demote()
            return out_k, out_v

        k = key_states[0].permute(1, 0, 2).to(self.dtype)   # [T, H, D]
        v = value_states[0].permute(1, 0, 2).to(self.dtype)
        self._append_tokens(layer_idx, layer_idx, k)
        self._append_tokens(self.L + layer_idx, layer_idx, v)
        self._seen[layer_idx] += k.shape[0]

        t = self._seen[layer_idx]
        out = []
        for part in (layer_idx, self.L + layer_idx):
            head = self.pool.head[part]
            if head == 0 and self.pool.tail[part] <= self.pool.device_rows:
                # everything-fits fast path: the precomputed typed view,
                # one narrow + one permute, nothing else
                res = self._typed_dev[part].narrow(0, 0, t)
                out.append(res.permute(1, 0, 2)[None])
                continue
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
        if self._side is not None:
            # ORDER the side stream after everything the compute stream has
            # queued — including the appends that WROTE the rows about to be
            # copied. Without this fence the DtoH can read a row whose write
            # has not executed yet; the bench never saw it only because its
            # greedy loop host-syncs every step on the argmax (Bugbot, HIGH).
            self._side.wait_stream(torch.cuda.current_stream())
        win_blocks = self.hot_window // self.bt
        for layer in range(self.L):
            done_blocks = self._seen[layer] // self.bt   # full blocks only
            upto = done_blocks - win_blocks
            if upto > 0:
                for part in (layer, self.L + layer):
                    self.pool.demote_head(part, upto, stream=self._side)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen[layer_idx]

    # -- the rest of the mask-preprocessing surface modern transformers
    #    queries before layers run (contracts read off the installed
    #    DynamicCache: dynamic growth, no offset, nothing sliding)
    def get_mask_sizes(self, query_length: int, layer_idx: int):
        return self._seen[layer_idx] + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def get_query_offset(self, layer_idx: int = 0) -> int:
        # newer transformers' mask preprocessing; equals the cache length
        # for everything but MTP caches (per the installed contract)
        return self._seen[layer_idx]

    @property
    def is_sliding(self) -> list:
        return [False] * self.L

    @property
    def is_compileable(self) -> bool:
        return False

    def stats(self) -> dict:
        s = self.pool.stats()
        s["gather_returns"] = self.gather_returns
        return s
