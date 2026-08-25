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

import os

import torch

BLOCK_TOKENS = 16



def _resolve_fused_append(env, device_str, kernel_present) -> bool:
    """B2-certified default (RESULTS-f1-stageB-b2: PASS, gain
    2.08 ms/step, bitwise 13/13, token-identity exact), resolved at
    CONSTRUCTION. Three gates, each with a loud-refuse twin when the
    caller EXPLICITLY set E4B_FUSED_KV_APPEND=1:

    - env "0" rolls back to the eager ~25-launch append.
    - the fused path is a CUDA triton kernel: a non-cuda KV degrades.
      Presence of the kernel is not usability -- once gnf4 >= 0.15.0
      ships fp8_kv_append_t1 the import SUCCEEDS on a CPU host and the
      first append would die inside triton's driver ("0 active
      drivers"; caught by e4b#251 CI the hour 0.15.0 hit PyPI).
    - a pre-gnf4#253 install has no kernel: degrade (the lazy import
      used to crash graph decode on every such install -- Bugbot,
      e4b#238). ``kernel_present`` is a callable so the import cost is
      paid only when the answer matters.
    """
    want = (env or "1") == "1"
    if not want:
        return False
    if not device_str.startswith("cuda"):
        if env == "1":
            raise RuntimeError(
                "E4B_FUSED_KV_APPEND=1 but this Fp8PagedKV is on "
                f"device {device_str!r} -- the fused append is a CUDA "
                "triton kernel")
        return False
    if not kernel_present():
        if env == "1":
            raise RuntimeError(
                "E4B_FUSED_KV_APPEND=1 but the installed "
                "grouped-nf4-gemm has no fp8_kv_append_t1 "
                "(needs the gnf4#253 kernel)")
        return False
    return True

class Fp8PagedKV:
    """FP8 paged KV for ``L`` layers, ``B`` sequences, one model.

    ``k_groups`` is the sub-row scale granularity on KEYS (the measured
    quality-passing config is ``head_dim // 32`` groups, i.e. 32-wide);
    values keep the full-row scale — the asymmetry is measured, not
    assumed (keys carry ~3x the dynamic range; see RESULTS-g7-quality).
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, *,
                 batch: int, max_tokens_per_seq: int, k_groups: int = 4,
                 batched_append: bool = True, device: str = "cuda"):
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
        self.graph_t1 = False   # B1d graph mode; graph_mode_init flips it
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
        # resolution semantics + history: _resolve_fused_append
        def _kernel_present():
            try:
                from fp8_kv import fp8_kv_append_t1  # noqa: F401
                return True
            except ImportError:
                return False

        self._fused_append = _resolve_fused_append(
            os.environ.get("E4B_FUSED_KV_APPEND"), str(device),
            _kernel_present)
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
    # ------------------------------------------------- graph mode (B1d) --
    def graph_mode_init(self, seq: int = 0, upto_tokens: int | None = None):
        """Prepare device-addressed T=1 appends for ONE slot so a decode
        step is CUDA-graph-capturable (PREREG-b1d). Two moves:

        * every block the window can touch is ensured UP FRONT (host
          allocation leaves the step entirely — no boundary fallback);
        * flat uint8 views of each side's layer arenas are cached, and
          `append_graph_t1` addresses them from DEVICE state (seq_lens +
          block table) each execution — never from host ``_seen``, whose
          capture-time value a graph would bake (the e4b#227 finding).

        Requires the KV arena in its up-front-claimed, never-demoted
        state (asserted): row ids then equal ring slots and the flat
        views are stable for the life of the pool."""
        for pool in (self.kp, self.vp):
            assert all(h == 0 for h in pool.head) and \
                all(t == self.B * self.blocks_per_seq for t in pool.tail), \
                "graph mode needs the up-front-claimed KV arena"
        upto = (self.blocks_per_seq * self.bt if upto_tokens is None
                else upto_tokens)
        for layer in range(self.L):
            self._ensure_blocks(layer, seq, (upto - 1) // self.bt)
        self._g_seq = seq
        dev = self.device
        hd = self.H * self.D
        self._g_ar_hd = torch.arange(hd, device=dev)
        self._g_ar_sk = torch.arange(self.H * self.k_groups * 4, device=dev)
        self._g_ar_sv = torch.arange(self.H * 4, device=dev)
        self._g_kflat = [self.kp.dev[layer].reshape(-1)
                         for layer in range(self.L)]
        self._g_vflat = [self.vp.dev[layer].reshape(-1)
                         for layer in range(self.L)]
        self.graph_t1 = True

    def append_graph_t1(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """One-token, one-slot append with every address computed on
        device (capture-safe). k, v: [1, H, D]. The write position comes
        from ``seq_lens`` and the row id from the device block table, so
        a captured replay writes to the ADVANCING position; the tail
        ``seq_lens.add_(1)`` is the same in-place publish the attention
        kernel reads."""
        seq = self._g_seq
        if self._fused_append:
            # AMENDMENT-f1-stageB-b2: one launch per side replaces the
            # ~25-launch eager sequence below. Bitwise against it
            # (gnf4 test_fp8_kv_append.py, re-asserted on-box before any
            # timed arm); the seq_lens publish stays the separate
            # in-stream op it is today.
            from fp8_kv import fp8_kv_append_t1
            fp8_kv_append_t1(v, self._g_vflat[layer],
                             self.block_table[layer][seq],
                             self.seq_lens[layer].narrow(0, seq, 1),
                             self.v_row, self._v_pay, self.bt, 1)
            fp8_kv_append_t1(k, self._g_kflat[layer],
                             self.block_table[layer][seq],
                             self.seq_lens[layer].narrow(0, seq, 1),
                             self.k_row, self._k_pay, self.bt,
                             self.k_groups)
            self.seq_lens[layer].narrow(0, seq, 1).add_(1)
            return
        vq, vs = self._quant_bytes(v, 1)
        kq, ks = self._quant_bytes(k, self.k_groups)
        pos = self.seq_lens[layer, seq].to(torch.long)
        blk = torch.div(pos, self.bt, rounding_mode="floor")
        fill = pos - blk * self.bt
        row = (self.block_table[layer][seq]
               .gather(0, blk.reshape(1).to(torch.int64))
               .reshape(()).to(torch.long))
        hd = self.H * self.D
        vbase = row * self.v_row + fill * hd
        self._g_vflat[layer].scatter_(0, vbase + self._g_ar_hd, vq.reshape(-1))
        vsb = row * self.v_row + self._v_pay + fill * (self.H * 4)
        self._g_vflat[layer].scatter_(0, vsb + self._g_ar_sv, vs.reshape(-1))
        kbase = row * self.k_row + fill * hd
        self._g_kflat[layer].scatter_(0, kbase + self._g_ar_hd, kq.reshape(-1))
        ksb = (row * self.k_row + self._k_pay
               + fill * (self.H * self.k_groups * 4))
        self._g_kflat[layer].scatter_(0, ksb + self._g_ar_sk, ks.reshape(-1))
        self.seq_lens[layer].narrow(0, seq, 1).add_(1)

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
                  lens_override: torch.Tensor | None = None,
                  **kw) -> torch.Tensor:
        """Paged FP8 decode attention for q [B, H_q, D]; dequantization
        happens in the kernel's registers (invariant 2). ``slots`` maps
        q's rows to sequences (see :meth:`kernel_args`).

        ``lens_override`` replaces the per-row lengths read from
        ``seq_lens`` (PREREG-s2lite): speculative verification maps K+1
        q-rows to ONE slot with staggered lengths ``base + 1 .. base +
        K+1`` so row i attends over the past plus draft tokens 0..i —
        causality enforced by lengths over already-appended K/V, with no
        kernel change (the kernel always consumed a lens tensor). Must
        be int32, on-device, one entry per q row."""
        from fp8_paged_attn import fp8_paged_decode_attention

        kf, vf, tbl, lens = self.kernel_args(layer, slots)
        if lens_override is not None:
            if lens_override.shape != lens.shape:
                raise ValueError(f"lens_override {tuple(lens_override.shape)}"
                                 f" must match rows {tuple(lens.shape)}")
            lens = lens_override.to(dtype=lens.dtype)
        if tbl.shape[0] != q.shape[0]:
            raise ValueError(
                f"q has {q.shape[0]} rows but the block table has "
                f"{tbl.shape[0]} — pass slots= so rows map to sequences")
        return fp8_paged_decode_attention(
            q, kf, vf, tbl, lens, n_kv_heads=self.H, head_dim=self.D,
            block_tokens=self.bt, k_groups=self.k_groups, **kw)

    def seen_device(self, layer: int, seq: int) -> int:
        """Authoritative token count for (layer, seq), read from DEVICE
        state. ``_seen`` is the host mirror maintained by the eager
        append paths; ``append_graph_t1`` deliberately advances only
        ``seq_lens`` (its whole point is that a captured replay needs no
        host state), so after any graph-mode decoding ``_seen`` is
        STALE. Anything that must know the true length -- rewind, a
        speculative base capture -- reads here and pays the sync
        (Bugbot, e4b#241)."""
        return int(self.seq_lens[layer, seq].item())

    def rewind(self, seq: int, to_tokens: int):
        """Roll a sequence back to ``to_tokens`` tokens across ALL
        layers (PREREG-s2lite): speculative verification appends the
        full K+1 window before reading, and rejected tokens must not be
        readable afterward. Reads are governed entirely by lengths --
        the stale bytes past the new length are unreachable -- so
        rewind is a length update with no data movement.

        Both mirrors are set: ``seq_lens`` (what the kernel reads) and
        ``_seen`` (what the eager append paths advance), so a rewind is
        safe whichever append path runs next. The forward check uses the
        DEVICE length, since ``_seen`` may be stale under graph mode --
        and that check is a ``.item()``, i.e. a SYNC, which is illegal
        under stream capture. Callers inside a capture (or timing a
        capture's replays) use :meth:`rewind_nosync`, which does the
        same length writes with no read-back."""
        for layer in range(self.L):
            cur = self.seen_device(layer, seq)
            if to_tokens > cur:
                raise ValueError(f"rewind forward (layer {layer}: "
                                 f"{cur} -> {to_tokens})")
        self.rewind_nosync(seq, to_tokens)

    def rewind_nosync(self, seq: int, to_tokens: int):
        """:meth:`rewind` without the device-length read-back.

        Capture-safe and sync-free: ``fill_`` is an in-stream op. The
        caller owns the guarantee that ``to_tokens`` is not forward of
        the true length -- inside a capture there is no legal way to
        check it, so the check happens once outside (Bugbot-adjacent:
        found live when the S2 timing arm's rewind hit
        cudaErrorStreamCaptureUnsupported through ``.item()``)."""
        for layer in range(self.L):
            self._seen[layer][seq] = to_tokens
            # fill_ keeps this async (see append's seq_lens note)
            self.seq_lens[layer].narrow(0, seq, 1).fill_(to_tokens)

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
