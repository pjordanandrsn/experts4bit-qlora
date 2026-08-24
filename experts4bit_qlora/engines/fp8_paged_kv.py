# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Batched FP8 paged KV over RowPool (hybrid Stage 2, Phase 7).

The serving-side integration of the E4M3 KV format: quantize on WRITE into
:class:`row_pool.RowPool` block rows, and on READ hand the fused Triton
kernel ``(pool bytes, block table, seq lens)`` — never a dequantized
tensor. This is the piece that makes Phase 7's format usable by Phase 8/9's
scheduler, and it is deliberately NOT a transformers cache object: the
protocol's ``update() -> (k, v)`` contract is a promise to materialize
tensors for stock attention, which is exactly what invariant 2 forbids on
this path. The oracle that keeps that contract for quality measurement is
:mod:`.fp8_kv_cache`; this class serves kernels.

Tier vocabulary (invariant 8): two RowPools — keys and values row sizes
differ once key scales are grouped — each with one partition per layer.
Blocks are pool rows; a block belongs to one sequence; the kernel's block
table is partition-relative row indices, which v1 can equate with append
order because nothing demotes (``head == 0`` always, asserted). Demotion /
park of FP8 blocks is Phase 10's work and changes only who owns the ring,
not this format.

Free when unused (invariant 9): construct nothing, pay nothing — the pools
allocate at construction and only when a caller opts into FP8 KV.

Decode appends are per (layer, sequence) and cost a handful of narrow+copy
dispatches each; at large batch that python count is the known cost of v1
and lands on Phase 9's batched-update path (single quantize over the batch
is already provided here via :meth:`append_batch`; the per-sequence row
writes are the part the scheduler will fuse).
"""
from __future__ import annotations

import torch

BLOCK_TOKENS = 16


class Fp8PagedKV:
    """FP8 paged KV for ``L`` layers, ``B`` sequences, one model.

    ``k_groups`` is the sub-row scale granularity on KEYS (the measured
    quality-passing config is ``head_dim // 32`` groups, i.e. 32-wide);
    values keep the full-row scale — the asymmetry is measured, not
    assumed (keys carry ~3x the dynamic range; see RESULTS-g7-quality).
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, *,
                 batch: int, max_tokens_per_seq: int, k_groups: int = 4,
                 batched_append: bool = False, device: str = "cuda"):
        from fp8_kv import kv_block_bytes
        from row_pool import RowPool

        if head_dim % max(k_groups, 1):
            raise ValueError(f"k_groups {k_groups} must divide "
                             f"head_dim {head_dim}")
        self.L = n_layers
        self.H = n_kv_heads
        self.D = head_dim
        self.B = batch
        self.bt = BLOCK_TOKENS
        self.k_groups = k_groups
        # decode call-site switch (PREREG-g9-kvappend): callers that see
        # this flag use append_batch for the whole batch at each layer
        # instead of one append per sequence
        self.batched_append = batched_append
        self._batch_idx_cache = {}
        self.blocks_per_seq = -(-max_tokens_per_seq // self.bt)
        self.k_row = (kv_block_bytes(self.bt, self.H, self.D)
                      + self.bt * self.H * 4 * (k_groups - 1))
        self.v_row = kv_block_bytes(self.bt, self.H, self.D)
        rows = batch * self.blocks_per_seq
        self.kp = RowPool(n_layers, rows, 0, self.k_row, device=device)
        self.vp = RowPool(n_layers, rows, 0, self.v_row, device=device)
        self.device = self.kp.device
        # payload bytes before the scale tail of a row, per side
        self._k_pay = self.bt * self.H * self.D
        self._v_pay = self.bt * self.H * self.D
        # block tables live on-device and are written IN PLACE when a block
        # opens (one scalar copy per 16 tokens) — rebuilding [B, blocks]
        # tables per decode step would be an H2D per layer per step
        self.block_table = [torch.zeros(batch, self.blocks_per_seq,
                                        dtype=torch.int32, device=self.device)
                            for _ in range(n_layers)]
        self.seq_lens = torch.zeros(n_layers, batch, dtype=torch.int32,
                                    device=self.device)
        self._seen = [[0] * batch for _ in range(n_layers)]
        # Block rows a (layer, slot) currently owns, so a finished
        # sequence can hand them back. A serving loop recycles slots
        # continuously; without this the pool is a one-shot arena and the
        # engine would die at ``blocks`` sequences regardless of how many
        # ever ran at once (Phase 9).
        self._rows: dict[tuple[int, int], list[int]] = {}
        # device selector per active-set tuple (see kernel_args) — bounded
        # by the distinct decode subsets a run ever sees (arrival/finish
        # order over <= batch slots), each an 8-byte-per-slot tensor
        self._slot_sel: dict[tuple, torch.Tensor] = {}
        self._free: list[list[int]] = [
            list(range(rows)) for _ in range(n_layers)]
        # Claim the whole arena up front, once. RowPool's append/demote
        # ring exists for the WEIGHT tier, where rows stream in and out in
        # order; KV with per-sequence block tables wants the opposite — a
        # flat resident arena whose blocks this class hands out and takes
        # back. Appending here makes every row device-resident so
        # row_view serves any block index; allocation order is then ours,
        # which is what makes slot reuse possible at all.
        for layer in range(n_layers):
            for _ in range(rows):
                self.kp.append(layer)
                self.vp.append(layer)

    # ---------------------------------------------------------------- write --
    def _quant_bytes(self, x, groups):
        """All of append's FALLIBLE work (allocating quantize + reshapes)
        for one side, done before either pool is touched — so an
        exception (OOM, bad input) leaves both pools' tails in lockstep.
        A throw between V's append and K's would otherwise desync the
        shared block table's row pairing for every later block (review)."""
        from fp8_kv import quantize_kv_fp8

        q, s = quantize_kv_fp8(x, group=None if groups == 1
                               else self.D // groups)
        qb = q.view(torch.uint8).reshape(-1, self.H * self.D)   # [T, H*D]
        sb = s.float().reshape(x.shape[0], -1).view(torch.uint8)  # [T, H*g*4]
        return qb, sb

    def _write_side(self, pool, pay_bytes, groups, layer, seq, qb, sb,
                    t_new, seen):
        """Write pre-quantized bytes into `pool`'s block rows for
        (layer, seq) from absolute token position `seen`. Allocation-free
        (narrow + copy_ into existing rows; pool.append cannot overrun —
        capacity is checked before any write). Row indices come from the
        HOST mirror (`_rows`), never the device block table — reading a
        device scalar here (`int(tbl[seq, blk])`) is a stream sync per
        block write, which at L layers x 2 sides x B sequences serializes
        the whole decode step behind the GPU (measured: attention host
        time == attention device time, ~60 ms/step on the dev box)."""
        srow = self.H * groups * 4
        written = 0
        rows = self._rows[(layer, seq)]
        while written < t_new:
            fill = seen % self.bt
            blk = seen // self.bt
            row = pool.row_view(layer, rows[blk])
            take = min(self.bt - fill, t_new - written)
            row.narrow(0, fill * self.H * self.D,
                       take * self.H * self.D).copy_(
                qb.narrow(0, written, take).reshape(-1))
            row.narrow(0, pay_bytes + fill * srow, take * srow).copy_(
                sb.narrow(0, written, take).reshape(-1))
            written += take
            seen += take

    def append(self, layer: int, seq: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [T, H, D] new tokens for one sequence at one layer."""
        if k.shape != v.shape or k.shape[1:] != (self.H, self.D):
            raise ValueError(f"expected [T, {self.H}, {self.D}], got "
                             f"{tuple(k.shape)} / {tuple(v.shape)}")
        seen = self._seen[layer][seq]
        if seen + k.shape[0] > self.blocks_per_seq * self.bt:
            raise ValueError(f"sequence {seq} overflows its "
                             f"{self.blocks_per_seq} blocks")
        # quantize BOTH sides before touching ANY shared state — the
        # lockstep invariant is that a failed append changes NOTHING, and
        # _quant_bytes is the fallible part (allocating). Only then claim
        # blocks: allocated ONCE per append, before either side writes —
        # tying allocation to the K side would leave V (which writes
        # first, by the publish-last discipline below) landing on an
        # unassigned table entry, i.e. row 0 for every block.
        vq = self._quant_bytes(v, 1)
        kq = self._quant_bytes(k, self.k_groups)
        self._ensure_blocks(layer, seq, (seen + k.shape[0] - 1) // self.bt)
        # V first: K's writer owns the shared block-table entry, and writing
        # K last means a table row is never published for a block whose V
        # bytes haven't landed yet (same publish-last discipline as the pool)
        self._write_side(self.vp, self._v_pay, 1, layer, seq, *vq,
                         k.shape[0], seen)
        self._write_side(self.kp, self._k_pay, self.k_groups, layer, seq,
                         *kq, k.shape[0], seen)
        self._seen[layer][seq] = seen + k.shape[0]
        # device-side scalar add, value in kernel args: a plain
        # `seq_lens[layer, seq] = n` wraps the int in a CPU tensor and
        # the blocking copy stream-syncs — B x L times per decode step
        self.seq_lens[layer].narrow(0, seq, 1).add_(k.shape[0])

    def append_many(self, layer: int, seqs, k: torch.Tensor,
                    v: torch.Tensor):
        """k, v: [B, T, H, D] new tokens for B sequences at ONE layer.

        The batched form of append(): ONE quantize kernel per side for
        the whole batch instead of one per sequence — the attribution
        receipts measured the per-sequence form at 196,608 single-token
        quantize calls (~112 ms/step at B=16), essentially the entire
        attention bucket. Bit-identical by construction: FP8 scales are
        per (token, head) or finer (amax over the last dim only), so
        quantizing [B*T, H, D] in one call equals B separate calls.

        The lockstep discipline is append()'s: ALL fallible allocating
        work (both sides' quantize, overflow checks) happens before any
        shared state changes; per-sequence block writes then follow the
        same V-first publish-last order. A failure inside the write loop
        leaves a prefix of sequences appended — fatal to the run, as a
        mid-batch OOM is today.
        """
        if k.shape != v.shape or k.dim() != 4 \
                or k.shape[2:] != (self.H, self.D):
            raise ValueError(f"expected [B, T, {self.H}, {self.D}], got "
                             f"{tuple(k.shape)} / {tuple(v.shape)}")
        B, T = k.shape[0], k.shape[1]
        if B != len(seqs):
            raise ValueError(f"{B} rows for {len(seqs)} sequences")
        for seq in seqs:
            if self._seen[layer][seq] + T > self.blocks_per_seq * self.bt:
                raise ValueError(f"sequence {seq} overflows its "
                                 f"{self.blocks_per_seq} blocks")
        vq_all = self._quant_bytes(v.reshape(B * T, self.H, self.D), 1)
        kq_all = self._quant_bytes(k.reshape(B * T, self.H, self.D),
                                   self.k_groups)
        for b, seq in enumerate(seqs):
            seen = self._seen[layer][seq]
            self._ensure_blocks(layer, seq, (seen + T - 1) // self.bt)
            vq = (vq_all[0].narrow(0, b * T, T),
                  vq_all[1].narrow(0, b * T, T))
            kq = (kq_all[0].narrow(0, b * T, T),
                  kq_all[1].narrow(0, b * T, T))
            self._write_side(self.vp, self._v_pay, 1, layer, seq, *vq,
                             T, seen)
            self._write_side(self.kp, self._k_pay, self.k_groups, layer,
                             seq, *kq, T, seen)
            self._seen[layer][seq] = seen + T
        key = tuple(seqs)
        cached = self._batch_idx_cache.get(key)
        if cached is None:
            idx = torch.as_tensor(list(seqs), dtype=torch.long,
                                  device=self.seq_lens.device)
            ones = torch.ones(len(seqs), dtype=self.seq_lens.dtype,
                              device=self.seq_lens.device)
            cached = (idx, ones)
            self._batch_idx_cache[key] = cached
        idx, ones = cached
        # one async device-side add for the whole batch (same reasoning
        # as append()'s narrow().add_: no CPU-tensor stream sync)
        self.seq_lens[layer].index_add_(0, idx, ones * T)

    def _ensure_blocks(self, layer: int, seq: int, upto_blk: int) -> None:
        """Back every block up to ``upto_blk`` with a pool row."""
        rows = self._rows.setdefault((layer, seq), [])
        tbl = self.block_table[layer]
        while len(rows) <= upto_blk:
            if not self._free[layer]:
                raise RuntimeError(
                    f"layer {layer} is out of KV blocks — the scheduler "
                    f"admitted past capacity")
            idx = self._free[layer].pop(0)
            # fill_ keeps this async (see append's seq_lens note);
            # prefill flushes allocate prompt/bt blocks x L layers in
            # one burst, so a sync here dominates the flush wall
            tbl[seq].narrow(0, len(rows), 1).fill_(idx)
            rows.append(idx)

    def reset(self, seq: int) -> None:
        """Release a finished sequence's blocks back to the free lists.

        Rows return in ascending order so a fresh run reuses them
        deterministically — a scheduler's block assignment is part of
        what makes a serving run reproducible."""
        for layer in range(self.L):
            rows = self._rows.pop((layer, seq), [])
            if rows:
                self._free[layer].extend(rows)
                self._free[layer].sort()
            self._seen[layer][seq] = 0
            self.seq_lens[layer].narrow(0, seq, 1).fill_(0)
            self.block_table[layer][seq].zero_()

    def free_blocks(self, layer: int = 0) -> int:
        return len(self._free[layer])

    def append_batch(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Decode-step append: k, v [B, H, 1, D] (attention layout) for all
        sequences at once. One call point for Phase 9's scheduler.
        Delegates to append_many, so it now quantizes the whole batch in
        one kernel per side instead of one per sequence."""
        self.append_many(layer, list(range(self.B)),
                         k.permute(0, 2, 1, 3).contiguous(),
                         v.permute(0, 2, 1, 3).contiguous())

    # ----------------------------------------------------------------- read --
    def kernel_args(self, layer: int, slots=None):
        """What the fused kernel consumes: flat pool bytes, block table,
        per-sequence lengths.

        ``slots`` selects WHICH sequences form the kernel's batch. The
        kernel indexes its tables by BATCH ROW, so a caller decoding a
        subset — which is every step of a real serving loop, where
        sequences finish at different times — must hand it rows in that
        subset's order. Passing the full tables works only while the
        active set happens to be slots 0..B-1 in order; anything else
        silently attends one sequence over another's KV, and it reads as
        a model that starts coherent and degenerates."""
        tbl, lens = self.block_table[layer], self.seq_lens[layer]
        if slots is not None:
            # the selector is CACHED per active-set: building it fresh is a
            # pageable H2D copy, and torch's non_blocking=False copy ends
            # in a full stream synchronize — one per LAYER per step, which
            # serializes decode behind the GPU exactly like the block-table
            # read did (measured together: attention host time == device
            # time until both were removed)
            key = tuple(slots)
            sel = self._slot_sel.get(key)
            if sel is None:
                sel = torch.as_tensor(list(key), dtype=torch.long,
                                      device=tbl.device)
                self._slot_sel[key] = sel
            tbl, lens = tbl.index_select(0, sel), lens.index_select(0, sel)
        return (self.kp.dev[layer].flatten(), self.vp.dev[layer].flatten(),
                tbl, lens)

    def attention(self, layer: int, q: torch.Tensor, slots=None,
                  **kw) -> torch.Tensor:
        """Paged FP8 decode attention for q [B, H_q, D]; dequantization
        happens in the kernel's registers (invariant 2). ``slots`` maps
        q's rows to sequences (see :meth:`kernel_args`)."""
        from fp8_paged_attn import fp8_paged_decode_attention

        kf, vf, tbl, lens = self.kernel_args(layer, slots)
        if tbl.shape[0] != q.shape[0]:
            raise ValueError(
                f"q has {q.shape[0]} rows but the block table has "
                f"{tbl.shape[0]} — pass slots= so rows map to sequences")
        return fp8_paged_decode_attention(
            q, kf, vf, tbl, lens, n_kv_heads=self.H, head_dim=self.D,
            block_tokens=self.bt, k_groups=self.k_groups, **kw)

    def reference_kv(self, layer: int, seq: int, dtype=torch.bfloat16):
        """ORACLE ONLY (tests/quality): materialize this sequence's K, V
        [T, H, D] by reference dequant. Never on the serving path."""
        from fp8_kv import dequant_kv_fp8_ref, unpack_kv_block_grouped

        t = self._seen[layer][seq]
        ks, vs = [], []
        for blk in range(-(-t // self.bt)):
            row_i = int(self.block_table[layer][seq, blk])
            qk, sk = unpack_kv_block_grouped(
                self.kp.dev[layer, row_i], self.bt, self.H, self.D,
                self.k_groups)
            qv, sv = unpack_kv_block_grouped(
                self.vp.dev[layer, row_i], self.bt, self.H, self.D, 1)
            ks.append(dequant_kv_fp8_ref(qk, sk, dtype=dtype))
            vs.append(dequant_kv_fp8_ref(qv, sv, dtype=dtype))
        if not ks:
            z = torch.zeros(0, self.H, self.D, dtype=dtype,
                            device=self.device)
            return z, z
        return (torch.cat(ks)[:t], torch.cat(vs)[:t])

    def bytes_per_token(self) -> int:
        """Honest per-token cost across K and V (payload + scale tails)."""
        return (self.k_row + self.v_row) // self.bt

    def stats(self) -> dict:
        return {"k_pool": self.kp.stats(), "v_pool": self.vp.stats(),
                "seq_lens": [list(map(int, r)) for r in
                             self.seq_lens.cpu().tolist()]}
