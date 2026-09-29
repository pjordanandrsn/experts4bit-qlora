# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Native-MXFP4 pipelined residency (Phase 4) — the fused mxfp4 kernel + the
address-table residency engine + the gpt-oss clamped-GLU epilogue, composed.

Self-contained in the unpublished implementation (rail: the shipped integration threads a
FORMAT-AGNOSTIC row_bytes/codebook param through e4b's public pipelined.py; the
word mxfp4 never enters a public file). The gather kernel here mirrors e4b
pipelined.py's format-agnostic `_gather_rows_addr` byte-for-byte — it copies a
per-slot row-block by absolute address and cannot see the format.

What differs from the NF4 engine (the seam map, realized in the arena):
  - row-block segments are native blocks (uint8) + e8m0 scales (uint8) for
    gate_up and down — NO fp32 absmax segment (scales are 4x smaller).
  - the fused call is gemm_mxfp4_grouped (flipped nibble, exp2 e8m0, BLOCK_K=32).
  - the epilogue is gpt-oss clamped-GLU + per-expert biases.
K (hot experts/layer resident) is a table rebuild, not a code path — identical
to the NF4 engine's contract. Decode (T==1) on CUDA; correctness-gated vs the
dequant reference at every K.
"""
from __future__ import annotations

import torch

from mxfp4_grouped import gemm_mxfp4_grouped


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _index_add_ordered_(out, rows, src):
    """``out.index_add_(0, rows, src)`` with each row's terms added in index order.

    On CUDA ``index_add_`` accumulates with float atomics. When ``rows`` repeats, as it
    does when a prefill token meets two of its experts in one chunk, that row's terms
    land in whatever order the threads win. The fp32 sum then changes from call to
    call, and so, now and then, do its bf16 bits. On the RTX A2000, 50 identical calls
    at Kimi-K3 geometry gave 50 different fp32 outputs, and the downstream top-16
    routing drifted from process to process (experts4bit-qlora#761).

    Here no two threads ever add into one address. Pass ``r`` adds the ``r``-th
    occurrence of every row, so each ``index_add_`` call sees unique rows. The result
    is bitwise the sequential ``for i in range(len(rows)): out[rows[i]] += src[i]``,
    on CPU and on CUDA. The cost is a sort of ``rows``, one host sync for the pass
    count, and one ``index_add_`` per pass. The pass count is the largest number of
    times any row repeats (at most ``topk`` on the prefill path).
    """
    n = rows.numel()
    if n == 0:
        return out
    srt, perm = torch.sort(rows, stable=True)
    # occurrence number of each sorted position: its distance from its row's first
    occ = torch.arange(n, device=rows.device) - torch.searchsorted(srt, srt)
    by_occ = torch.argsort(occ, stable=True)
    _, per_pass = torch.unique_consecutive(occ.index_select(0, by_occ), return_counts=True)
    pos = perm.index_select(0, by_occ)                     # positions in `rows`, pass by pass
    lo = 0
    for cnt in per_pass.tolist():
        sel = pos[lo:lo + cnt]
        out.index_add_(0, rows.index_select(0, sel), src.index_select(0, sel))
        lo += cnt
    return out


_KERNEL = None


def _gather_kernel():
    """Per-slot absolute-address gather with have-skip — mirrors e4b
    pipelined.py::_gather_rows_addr (format-agnostic; copies bytes)."""
    # `tl` into MODULE globals, not locals: with `from __future__ import
    # annotations` the `BLOCK: tl.constexpr` annotation is the string
    # "tl.constexpr", which triton resolves against the jitted function's
    # __globals__. A local import raises NameError('tl is not defined') from
    # inside the triton compiler on 3.2. 3.4 tolerates it, so the bug is
    # invisible on a current box and fatal on an older one.
    global _KERNEL, tl
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _gather_rows_addr(dst_ptr, src_ptr, have_ptr, row_words, BLOCK: tl.constexpr):
            slot = tl.program_id(0)
            chunk = tl.program_id(1)
            want = tl.load(src_ptr + slot)
            have = tl.load(have_ptr + slot)
            if want == have:
                return
            offs = chunk * BLOCK + tl.arange(0, BLOCK)
            mask = offs < row_words
            src = tl.cast(want, tl.pointer_type(tl.int64))
            vals = tl.load(src + offs, mask=mask)
            tl.store(dst_ptr + slot.to(tl.int64) * row_words + offs, vals, mask=mask)

        _KERNEL = _gather_rows_addr
    return _KERNEL


class SlotStore:
    """The k device slots, factored out so every layer's engine can share ONE.

    Slots are the engine's only large device allocation: ``k`` rows of ``row_bytes``,
    which for Kimi K3 is 16 x 17,547,264 = **281 MB**. Fine for one layer, fatal for
    92 — 25.8 GB of VRAM in slot buffers alone, for a model whose entire point is
    fitting on a card that size. Nothing justifies the duplication: decode walks the
    layers in sequence, so at most one engine's slots are live at any instant.

    What sharing costs is that residency state stops being per-engine. ``have`` moves in
    here with the buffer it describes, and an engine taking the buffer over from another
    layer must FORGET what it believed was resident — see
    :meth:`Mxfp4PipelinedGptOss._claim`. Under a tier that is not paranoia: tier slot
    addresses are reused across layers, so the previous owner's recorded address can
    equal what this layer now wants, and the gather's ``src == have`` skip would then
    serve another layer's expert. Same unsoundness as within one layer
    (:meth:`mxfp4_residency.Mxfp4NvmeResidency._invalidate`), one level up.

    Not shared, on purpose: ``a_buf`` (k x hidden activations, 114 KB at K3's latent
    width — 10 MB across 92 layers, and sharing it would mean touching ``forward``),
    the traffic counters, and ``hot_stack``, which is genuinely per-layer data.
    """

    __slots__ = ("k", "row_bytes", "off", "geo", "device", "slots", "slots64",
                 "gu_p_v", "gu_a_v", "dn_p_v", "dn_a_v", "have", "want_buf",
                 "slot_eids", "owner", "claims", "users")

    def __init__(self, engine):
        k, off, row_bytes, dev = engine.k, engine.off, engine.row_bytes, engine.device
        n1, half1, nb1 = engine.n1, engine.half1, engine.nb1
        n2, half2, nb2 = engine.n2, engine.half2, engine.nb2
        self.k, self.row_bytes, self.off = k, row_bytes, list(off)
        self.geo = (n1, half1, nb1, n2, half2, nb2)
        self.device = dev
        slots = torch.empty(k, row_bytes, dtype=torch.uint8, device=dev)
        self.slots, self.slots64 = slots, slots.view(torch.int64)
        self.gu_p_v = torch.as_strided(slots, (k, n1, half1), (row_bytes, half1, 1), off[0])
        self.gu_a_v = torch.as_strided(slots, (k, n1, nb1), (row_bytes, nb1, 1), off[1])
        self.dn_p_v = torch.as_strided(slots, (k, n2, half2), (row_bytes, half2, 1), off[2])
        self.dn_a_v = torch.as_strided(slots, (k, n2, nb2), (row_bytes, nb2, 1), off[3])
        self.slot_eids = torch.arange(k, dtype=torch.int32, device=dev)
        self.have = torch.full((k,), -1, dtype=torch.long, device=dev)
        self.want_buf = torch.zeros(k, dtype=torch.long, device=dev)
        self.owner = None          # the engine whose rows the slots currently hold
        self.claims = 0            # buffer handovers; per token this is the layer count
        self.users = 0

    def _shape(self, engine):
        return (engine.k, engine.row_bytes, list(engine.off),
                (engine.n1, engine.half1, engine.nb1,
                 engine.n2, engine.half2, engine.nb2), engine.device)

    def check(self, engine):
        """Refuse a store whose row layout is not this engine's, byte for byte.

        Slots are raw bytes read at fixed segment offsets; a store built for a
        different layout would be read as this one and give plausible garbage.
        """
        mine = (self.k, self.row_bytes, list(self.off), self.geo, self.device)
        theirs = self._shape(engine)
        if mine != theirs:
            raise ValueError(
                f"shared SlotStore does not match this engine: store {mine} vs "
                f"engine {theirs}. Every engine sharing a store must have identical "
                "row geometry.")

    @property
    def bytes(self) -> int:
        return self.k * self.row_bytes

    def __repr__(self) -> str:
        return (f"SlotStore(k={self.k}, row_bytes={self.row_bytes}, "
                f"{self.bytes/1e6:.0f} MB, users={self.users}, claims={self.claims})")


class Mxfp4PipelinedGptOss:
    """Per-layer engine: pinned native-mxfp4 arena + resident hot stack + k-slot
    store, address-table dispatch, fused mxfp4 GEMM, gpt-oss GLU epilogue.

    gu_blocks [E, n1, k1//2] u8, gu_scales [E, n1, k1//32] u8   (k1 = hidden)
    dn_blocks [E, n2, k2//2] u8, dn_scales [E, n2, k2//32] u8   (k2 = inter)
    gate_up_bias [E, n1] bf16, down_bias [E, n2] bf16.
    """

    def __init__(self, gu_blocks, gu_scales, dn_blocks, dn_scales,
                 gate_up_bias, down_bias, hot_ids, k_slots, device="cuda",
                 alpha=1.702, limit=7.0, compute_dtype=torch.bfloat16,
                 store=None):
        E, n1, half1 = gu_blocks.shape
        _, n2, half2 = dn_blocks.shape
        nb1, nb2 = gu_scales.shape[-1], dn_scales.shape[-1]
        assert nb1 == (half1 * 2) // 32 and nb2 == (half2 * 2) // 32
        self._init_geometry(E, n1, half1, nb1, n2, half2, nb2, k_slots=k_slots,
                            device=device, alpha=alpha, limit=limit,
                            compute_dtype=compute_dtype)
        self._build_source(gu_blocks, gu_scales, dn_blocks, dn_scales, hot_ids)
        self._init_slots(store)
        self._init_bias(gate_up_bias, down_bias)
        self._prime()

    # ------------------------------------------------------------- geometry --
    def _init_geometry(self, E, n1, half1, nb1, n2, half2, nb2, *, k_slots,
                       device, alpha, limit, compute_dtype):
        """Row layout only — no weights, so a subclass that gets its geometry
        from a baked arena's index rather than from tensors shares this exactly."""
        self.device = torch.device(device)
        self.k = int(k_slots)
        self.cd = compute_dtype
        self.alpha, self.limit = float(alpha), float(limit)
        self.E, self.n1, self.n2 = E, n1, n2
        self.half1, self.half2, self.nb1, self.nb2 = half1, half2, nb1, nb2
        self.k1, self.k2 = half1 * 2, half2 * 2

        seg = [n1 * half1, n1 * nb1, n2 * half2, n2 * nb2]
        off = [0]
        for s in seg[:-1]:
            off.append(_align8(off[-1] + s))
        row_bytes = _align8(off[-1] + seg[-1])
        self.seg = seg
        self.row_bytes, self.off, self.row_words = row_bytes, off, row_bytes // 8

    def _build_source(self, gu_blocks, gu_scales, dn_blocks, dn_scales, hot_ids):
        """Where rows come from: here, a pinned arena holding ALL E of them.

        That is the piece a >host-RAM model cannot afford — see
        :mod:`mxfp4_residency`, which REPLACES this step (and refuses this
        method outright) to keep only the hot rows and stream the tail from
        NVMe."""
        E, off, seg, row_bytes = self.E, self.off, self.seg, self.row_bytes
        # pinned arena [E, row_bytes]: native bytes, laid out, not converted
        arena = torch.zeros(E, row_bytes, dtype=torch.uint8)
        try:
            arena = arena.pin_memory()
            self.pinned = arena.is_pinned()
        except (RuntimeError, AssertionError):
            self.pinned = False
        arena[:, off[0]:off[0] + seg[0]] = gu_blocks.reshape(E, -1)
        arena[:, off[1]:off[1] + seg[1]] = gu_scales.reshape(E, -1)
        arena[:, off[2]:off[2] + seg[2]] = dn_blocks.reshape(E, -1)
        arena[:, off[3]:off[3] + seg[3]] = dn_scales.reshape(E, -1)
        self.arena = arena

        hot_ids = torch.as_tensor(hot_ids, dtype=torch.long).unique()
        if hot_ids.numel():
            self.hot_stack = arena.index_select(0, hot_ids).to(self.device)
        else:
            self.hot_stack = torch.empty(0, row_bytes, dtype=torch.uint8, device=self.device)
        is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        is_hot[hot_ids.to(self.device)] = True
        self.is_hot = is_hot
        h_row = torch.zeros(E, dtype=torch.long, device=self.device)
        h_row[hot_ids.to(self.device)] = torch.arange(hot_ids.numel(), device=self.device)
        host_addr = self.arena.data_ptr() + torch.arange(E, device=self.device, dtype=torch.long) * row_bytes
        hot_addr = self.hot_stack.data_ptr() + h_row * row_bytes
        self.src_of_expert = torch.where(is_hot, hot_addr, host_addr)

    def _init_slots(self, store=None):
        """Bind the device slots — a fresh :class:`SlotStore`, or a shared one.

        Pass another engine's ``.store`` to share; the idiom mirrors ``tier=``::

            engines = []
            for layer in range(1, 93):
                engines.append(Mxfp4NvmeResidencyK3(
                    arena, layer, k_slots=16, tier=tier,
                    store=engines[0].store if engines else None))
        """
        if store is None:
            store = SlotStore(self)
        else:
            store.check(self)
        store.users += 1
        self.store = store
        self.slots, self.slots64 = store.slots, store.slots64
        self.gu_p_v, self.gu_a_v = store.gu_p_v, store.gu_a_v
        self.dn_p_v, self.dn_a_v = store.dn_p_v, store.dn_a_v
        self.slot_eids = store.slot_eids
        self.have, self.want_buf = store.have, store.want_buf

        self.sizes = [1] * self.k
        self.a_buf = None
        self.hot_d2d_bytes = torch.zeros((), dtype=torch.long, device=self.device)
        self.cold_pcie_bytes = torch.zeros((), dtype=torch.long, device=self.device)

    def _init_bias(self, gate_up_bias, down_bias):
        """``None`` means this family has no per-expert bias — SKIP the add, do
        not synthesize zeros. gpt-oss ships biases; Kimi K3's experts are three
        bias-free Linears, and an `[E, n]` zeros pair is real VRAM at E=896."""
        for name, b, n in (("gate_up_bias", gate_up_bias, self.n1),
                           ("down_bias", down_bias, self.n2)):
            if b is None:
                setattr(self, name, None)
                continue
            if tuple(b.shape) != (self.E, n):
                raise ValueError(f"{name} must be [E={self.E}, {n}], "
                                 f"got {tuple(b.shape)}")
            setattr(self, name, b.to(self.device))

    # -------------------------------------------------------------- fetch ----
    def _resolve_src(self):
        """Absolute byte address of the row each slot wants, from ``want_buf``.

        A static table suffices here because pinning every row makes
        address <-> expert a BIJECTION. A tiering subclass must override this
        together with :meth:`_invalidate`, because a tier slot's address
        identifies the SLOT, not what currently lives in it."""
        return self.src_of_expert.index_select(0, self.want_buf)

    def _invalidate(self, src):
        """Force a re-gather of device slots whose CONTENTS changed while their
        source address did not. Nothing to do under a bijection; overridden by
        the tiering subclass, where it is the difference between correct output
        and silently stale experts."""

    def _commit(self, src):
        self.have.copy_(src)

    def _gather(self, src):
        """Copy each wanted row into its device slot, verbatim.

        The single place the gather is launched, so a subclass can substitute a
        kernel that REORDERS segments while copying — see
        :meth:`mxfp4_residency.Mxfp4NvmeResidency._gather`, which reads an arena
        whose segments are laid out in a different order than this engine's."""
        kern = _gather_kernel()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src, self.have, self.row_words, BLOCK=2048, num_warps=4)

    def _claim(self):
        """Take the shared slot buffer, discarding residency another layer left.

        A no-op when this engine already holds it, which is the single-engine case and
        the repeated-decode-step case — so the have-skip keeps working exactly as
        before. On a handover, ``have`` is poisoned wholesale (the slots hold another
        layer's rows now) and :meth:`_forget` drops whatever else the engine believed.
        """
        store = self.store
        if store.owner is self:
            return
        store.owner = self
        store.claims += 1
        self.have.fill_(-1)
        self._forget()

    def _forget(self):
        """Per-engine residency bookkeeping to discard on a buffer handover.

        Nothing here: with every row pinned, address <-> expert is a BIJECTION, so an
        address another layer's engine recorded cannot name one of this layer's rows.
        Overridden by the tiering subclass, where that stops being true.
        """

    def _prime(self):
        self._claim()
        self.want_buf.zero_()
        src0 = self._resolve_src()
        self._gather(src0)
        self._commit(src0)

    def _fetch(self, want):
        self._claim()
        self.want_buf.copy_(want)
        src = self._resolve_src()
        self._invalidate(src)                             # before the miss count
        miss = src != self.have
        hot = self.is_hot.index_select(0, self.want_buf)   # resident -> D2D, cold -> UVA
        self.cold_pcie_bytes += (miss & ~hot).sum() * self.row_bytes
        self.hot_d2d_bytes += (miss & hot).sum() * self.row_bytes
        self._gather(src)
        self._commit(src)

    def forward(self, hidden_states, router_indices, router_scores):
        """``x [T, K]``, ``router_indices/scores [T, topk]`` -> ``[T, N]``.

        T == 1 takes the decode path unchanged — it is the validated one, and its
        reduction order is part of what was validated. T > 1 takes
        :meth:`_forward_prefill`.
        """
        if router_indices.reshape(-1).numel() > self.k:
            return self._forward_prefill(hidden_states, router_indices, router_scores)
        return self._forward_decode(hidden_states, router_indices, router_scores)

    def _forward_prefill(self, hidden_states, router_indices, router_scores):
        """Many tokens per call, with the expert set chunked to the slot budget.

        Prefill is not merely decode in a loop, and the difference is I/O, not
        convenience. Stepping T tokens one at a time re-reads the DENSE side T times
        (107.3 GB each) and every routed expert row T times. Here each layer is
        entered once for the whole prompt, and each distinct expert is read once no
        matter how many of the T tokens routed to it. On Kimi K3 at T=10 that is
        ~310 GB against ~1,346 GB — a 4x cut, at batch one.

        The device slot budget stays exactly k. A prompt can route to far more than k
        distinct experts (up to ``T * topk``), so the distinct set is processed in
        chunks of at most k: make a chunk resident, run every (token, expert) pair
        belonging to it, accumulate, move on. The grouped GEMM was built for this —
        ``sizes`` is its per-group token count and it switches to the tiled kernel on
        its own once any group holds more than one row.
        """
        in_dtype, in_dev = hidden_states.dtype, hidden_states.device
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(
            device=self.device, dtype=self.cd)
        T = x.shape[0]
        ids = router_indices.reshape(T, -1).to(device=self.device, dtype=torch.long)
        topk = ids.shape[1]
        if not torch.cuda.is_current_stream_capturing():
            if bool((ids >= self.E).any()):
                raise ValueError(
                    "padded routing index (== num_experts) reached the pipelined "
                    "prefill engine — drop padding upstream")
        wt = router_scores.reshape(-1).to(device=self.device, dtype=torch.float32)

        # Sort the T*topk (token, expert) pairs by EXPERT so each expert's rows are
        # contiguous — which is what `sizes` describes. stable=True only to make the
        # row order reproducible; correctness does not depend on it, since every row
        # is scattered back to its own token below.
        flat_e = ids.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        pair_tok = torch.div(order, topk, rounding_mode="floor")
        sorted_e = flat_e.index_select(0, order)
        uniq, counts = torch.unique_consecutive(sorted_e, return_counts=True)
        uniq_l, counts_l = uniq.tolist(), counts.tolist()

        out = torch.zeros(T, self.n2, dtype=torch.float32, device=self.device)
        k, pstart = self.k, 0
        for c0 in range(0, len(uniq_l), k):
            chunk, ccounts = uniq_l[c0:c0 + k], counts_l[c0:c0 + k]
            npairs = sum(ccounts)
            # `_fetch` gathers all k slots, so a short chunk is padded by REPEATING an
            # id already in it. The padded slots are never referenced by `sizes` or
            # `slot_eids`, and a duplicate id is already routine on this path —
            # `_prime` asks all k slots for expert 0.
            want = torch.tensor(chunk + [chunk[-1]] * (k - len(chunk)),
                                dtype=torch.long, device=self.device)
            self._fetch(want)

            rows = pair_tok[pstart:pstart + npairs]
            eids = sorted_e[pstart:pstart + npairs]
            a = x.index_select(0, rows).contiguous()      # group-sorted, [npairs, K]
            gu = gemm_mxfp4_grouped(a, self.gu_p_v, self.gu_a_v, ccounts,
                                    self.slot_eids[:len(chunk)])
            if self.gate_up_bias is not None:
                # per ROW now, not per slot: a row's bias is its own expert's
                gu = gu + self.gate_up_bias.index_select(0, eids)
            h = self._glu(gu)
            dn = gemm_mxfp4_grouped(h.contiguous().to(self.cd), self.dn_p_v,
                                    self.dn_a_v, ccounts,
                                    self.slot_eids[:len(chunk)]).to(torch.float32)
            if self.down_bias is not None:
                dn = dn + self.down_bias.index_select(0, eids).to(torch.float32)
            # weight each pair by its own router score, then sum into its token. The
            # chunks run in ascending expert id and the pairs inside one are expert-sorted,
            # so each token's terms are added in ascending expert id, one fp32 rounding
            # each: the same bits on every call (see _index_add_ordered_)
            _index_add_ordered_(
                out, rows, dn * wt.index_select(0, order[pstart:pstart + npairs])[:, None])
            pstart += npairs

        return out.to(device=in_dev, dtype=in_dtype)

    def _forward_decode(self, hidden_states, router_indices, router_scores):
        in_dtype, in_dev = hidden_states.dtype, hidden_states.device
        x = hidden_states.to(device=self.device, dtype=self.cd)
        want = router_indices.reshape(-1).to(device=self.device, dtype=torch.long)
        # the decode engine takes exactly k VALID ids; transformers' padded
        # routing index (== num_experts) would index bias/expert data OOB.
        # Checked eagerly only — a sync inside CUDA-graph capture is illegal.
        if not torch.cuda.is_current_stream_capturing():
            if bool((want >= self.E).any()):
                raise ValueError(
                    "padded routing index (== num_experts) reached the pipelined "
                    "decode engine — drop padding upstream; this engine takes "
                    "exactly k valid expert ids per token")
        k = self.k
        self._fetch(want)
        if self.a_buf is None or self.a_buf.dtype != self.cd:
            self.a_buf = torch.empty(k, x.shape[-1], dtype=self.cd, device=self.device)
        self.a_buf.copy_(x.expand(k, -1))
        gu = gemm_mxfp4_grouped(self.a_buf, self.gu_p_v, self.gu_a_v, self.sizes, self.slot_eids)
        if self.gate_up_bias is not None:
            gu = gu + self.gate_up_bias.index_select(0, self.want_buf)
        h = self._glu(gu)
        dn = gemm_mxfp4_grouped(h.contiguous().to(self.cd), self.dn_p_v, self.dn_a_v,
                                self.sizes, self.slot_eids)
        dn = dn.to(torch.float32)
        if self.down_bias is not None:
            dn = dn + self.down_bias.index_select(0, self.want_buf).to(torch.float32)
        w = router_scores.reshape(-1).to(device=self.device, dtype=torch.float32)
        out = (dn * w[:, None]).sum(0, keepdim=True)
        return out.to(device=in_dev, dtype=in_dtype)

    def _glu(self, gu):
        """gpt-oss epilogue: gate/up are INTERLEAVED in the output columns (not
        two halves), then clamped-GLU. Other families differ in BOTH respects —
        Kimi K3 concatenates (``cat([w1(x), w3(x)])``) and activates with SiTU —
        so this is a hook, not inlined arithmetic."""
        gate, up = gu[..., ::2], gu[..., 1::2]     # gpt-oss INTERLEAVED, not chunk(2)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return (up + 1) * (gate * torch.sigmoid(gate * self.alpha))

    def traffic(self):
        return {"hot_d2d_bytes": int(self.hot_d2d_bytes.item()),
                "cold_pcie_bytes": int(self.cold_pcie_bytes.item())}
