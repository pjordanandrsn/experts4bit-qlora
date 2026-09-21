# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Opt-in uniform-int4 storage for the SERVING attention projections.

Callers opt in with :func:`enable_serve_attn_int4` (after any q/k/v
fusion pass), the ``enable_fast``/``enable_nvme_residency`` pattern --
free when unused. Harnesses conventionally gate the call on
``E4B_SERVE_ATTN_INT4=1``. Each projection becomes :class:`Int4Linear`: weights packed to the
int4-b32 grid at load FROM THE SOURCE bf16 tensor (never from an
already-quantised grid -- composition measured ~7x the pure grid's ppl
cost), decode (M = 1) served by the grouped int4 GEMV, prefill (M > 1)
by dequant-then-matmul -- the winning regime per the fused/dequant
crossover, paid once per request.

Measured basis (receipts: int4port P1, INT4GATE/INT4SPLIT): the GEMV
runs the qkv shape at 1,044 GB/s -- 6.9x over the NF4 register-LUT path
and 2.6x over the bf16 dense baseline -- and the int4 grid on attention
costs -0.006 ppl over 8,192 teacher-forced tokens. The lm_head is NOT
eligible uncalibrated (+0.18 ppl measured); the calibrated lane takes it
under its own flag. A projection bias (gpt-oss) rides beside the int4
weight in bf16 and is added after the GEMV; the weight alone is on the
grid.

Capture-legality: each module preallocates its split-K partials buffer
and its activation-quant outputs at swap time, so a captured decode
step allocates nothing here.
"""
from __future__ import annotations

import torch
from torch import nn


def _kernels():
    """Lazy, loud: the int4 kernels ship in grouped-nf4-gemm >= the cut
    carrying ``int4_b32``. Absence degrades at ENABLE time with a
    sentence, never silently at forward time (the fp8-append pattern)."""
    from int4_b32 import gemv_int4_b32, quant_x_rows  # noqa: F401
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: F401
    return gemv_int4_b32, quant_x_rows, dequant_int4_ref, pack_int4_b32


def _smallm_kernels():
    """The K16 small-M GEMM (grouped-nf4-gemm ``int4_smallm``, lane K16). Optional and OPT-IN:
    absent, ``enable_serve_attn_int4(..., smallm=True)`` refuses with a sentence; it is never
    substituted silently."""
    from int4_smallm import gemm_int4_b32_smallm, plan_smallm, smallm_workspace  # noqa: F401
    return gemm_int4_b32_smallm, plan_smallm, smallm_workspace


class Int4Linear(nn.Module):
    """Frozen serving projection stored on the int4-b32 grid."""

    def __init__(self, lin: nn.Linear, packer=None, smallm: bool = False):
        """``packer(w_fp32_cpu) -> (packed, scales)`` defaults to the
        shipped round-to-nearest packer; the calibrated lane passes one
        closed over that projection's Hessian. Same bytes either way.

        ``smallm=True`` routes ``1 < rows <= SMALLM_ROWS_MAX`` to the K16
        small-M int4 GEMM (``int4_smallm.gemm_int4_b32_smallm``) instead of
        the cached bf16 matmul, on the SAME packed bytes; its split-K
        workspace is preallocated here so a captured decode step allocates
        nothing. Default ``auto`` since the K16 P5 read (see
        :func:`resolve_smallm`); callers pass the resolved flag."""
        super().__init__()
        _gemv, _qx, _dref, pack = _kernels()
        N, K = lin.out_features, lin.in_features
        dev = lin.weight.device
        packed, scales = (packer or pack)(lin.weight.detach().float().cpu())
        # A projection bias (gpt-oss's q/k/v/o carry one) rides beside the
        # int4 weight in bf16 and is added after the GEMV / matmul -- the
        # weight is what the grid stores, the bias is not quantised. Kept
        # as a buffer of the module's own so the swap stays weight-exact.
        bias = None if lin.bias is None else lin.bias.detach().to(torch.bfloat16).clone()
        self._install(packed.reshape(1, N, K // 2).to(dev),
                      scales.reshape(1, N, K // 32).to(dev),
                      N, K, bias, dev, smallm)

    @classmethod
    def from_packed(cls, packed: torch.Tensor, scales: torch.Tensor, N: int, K: int, *,
                    bias: torch.Tensor | None = None, smallm: bool = False) -> "Int4Linear":
        """An ``Int4Linear`` over bytes that are ALREADY on the int4-b32 grid
        (``packed [N, K//2] uint8`` or ``[1, N, K//2]``; ``scales [N, K//32]`` or
        ``[1, N, K//32]``). Nothing is re-quantised: the grid is the one the
        caller hands over, so two modules built from the same bytes serve the
        same function. This is the constructor :meth:`fuse` uses."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        _kernels()
        if packed.numel() != N * (K // 2) or scales.numel() != N * (K // 32):
            raise ValueError(
                f"from_packed: bytes do not describe an [N={N}, K={K}] int4-b32 store "
                f"(packed {tuple(packed.shape)}, scales {tuple(scales.shape)})")
        dev = packed.device
        self._install(packed.reshape(1, N, K // 2), scales.reshape(1, N, K // 32),
                      N, K, None if bias is None else bias.to(torch.bfloat16), dev, smallm)
        return self

    @classmethod
    def fuse(cls, mods) -> "Int4Linear":
        """ONE ``Int4Linear`` whose output is the concatenation, in order, of
        ``mods``' outputs on the same input -- the int4 counterpart of
        stacking q/k/v weights into one ``nn.Linear`` (``engines.qkv_fuse``).

        The int4-b32 grid quantises each output row over its own K-blocks, so
        concatenating the packed rows and their scales along N is exact: every
        fused row is byte-identical to the row it came from, and the fused
        module computes the same function as the parts. What changes is the
        launch count -- one GEMV / small-M GEMM per layer where there were
        three -- which is the whole point at decode (k/v_proj sit within 1 us
        of the launch floor on a 5090, K16 rows). Refuses (``ValueError``)
        rather than half-fuses: every part must be an ``Int4Linear`` with the
        same ``K``, the same small-M routing, and either all or none biased."""
        mods = list(mods)
        if not mods or not all(isinstance(m, cls) for m in mods):
            raise ValueError("fuse: every part must be an Int4Linear")
        K = mods[0].K
        if any(m.K != K for m in mods):
            raise ValueError(f"fuse: parts disagree on K: {[m.K for m in mods]}")
        smallm = mods[0]._smallm is not None
        if any((m._smallm is not None) != smallm for m in mods):
            raise ValueError("fuse: parts disagree on the smallm route (some route rows 2..16 to K16, some do not)")
        biased = [m.bias is not None for m in mods]
        if any(biased) and not all(biased):
            raise ValueError("fuse: some parts carry a bias and some do not -- refusing to half-fuse")
        packed = torch.cat([m.packed for m in mods], dim=1)
        scales = torch.cat([m.scales for m in mods], dim=1)
        bias = torch.cat([m.bias for m in mods]) if all(biased) else None
        return cls.from_packed(packed, scales, sum(m.N for m in mods), K, bias=bias, smallm=smallm)

    def _install(self, packed, scales, N, K, bias, dev, smallm):
        gemv, qx, dref, _pack = _kernels()
        self._gemv, self._qx, self._dref = gemv, qx, dref
        self._smallm = None
        if smallm:
            self._smallm, plan_smallm, smallm_workspace = _smallm_kernels()
        self.N, self.K = N, K
        self.register_buffer("packed", packed.contiguous(), persistent=False)
        self.register_buffer("scales", scales.contiguous(), persistent=False)
        self.register_buffer("_eid0",
                             torch.zeros(1, dtype=torch.int32, device=dev),
                             persistent=False)
        from int4_b32 import _plan
        _bn, _wp, sk, _ku = _plan(self.N, self.K)
        self.register_buffer("_part",
                             torch.empty(sk, self.N, dtype=torch.float32,
                                         device=dev),
                             persistent=False)
        self._decode_cache = {}        # R -> (eids, part) for 1 < R <= cap
        self._bf16_cache = None        # dequantised weight for rows > cap
        if self._smallm is not None:
            bn, kc, sk_sm = plan_smallm(self.N, self.K)
            self._smallm_cfg = (bn, kc, sk_sm)
            part_sm, cnt_sm = smallm_workspace(self.N, block_n=bn, sk=sk_sm, device=dev)
            self.register_buffer("_smallm_part", part_sm, persistent=False)
            self.register_buffer("_smallm_cnt", cnt_sm, persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    # The int4 GEMV is a ONE-ROW lever. Its grid has a row axis but no
    # weight reuse across rows: each row-program re-streams the projection,
    # so 16 rows cost ~16 L2 reads of a 4-5 MB tile (~15 us per launch x 96
    # launches). Measured on a 5090 (receipts INT4B16/P22C): rows=1 x1.06,
    # rows=16 x0.90 -- and the dequant-per-call path P21 had was x0.52.
    # Above one row the fastest thing we have is cuBLAS on a bf16 weight
    # read ONCE with tensor cores, so the dequantised weight is built once
    # and CACHED (+~1.8 GB for 96 projections); batched decode then costs
    # exactly what bf16 attention costs. A batched int4 attention that WINS
    # needs a small-M int4 GEMM with weight-tile reuse; not this module.
    GEMV_ROWS_MAX = 1
    # K16: the small-M GEMM serves 1 < rows <= 16 -- ONE M-tile, in-register dequant, bf16 MMA over
    # fat K chunks, fused split-K -- reading the int4 bytes once instead of the cached bf16 copy
    # (#561: the bf16 cache holds a second, 4x larger representation of every projection). Opt-in
    # until the K16 lane's 5090 numbers meet its registered decision rule.
    SMALLM_ROWS_MAX = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.K)
        R = rows.shape[0]
        if R <= self.GEMV_ROWS_MAX:
            eids, part = self._decode_bufs(R)
            xq, xs = self._qx(rows)
            out = self._gemv(xq, xs, self.packed, self.scales, eids,
                             self.N, self.K, part=part)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*x.shape[:-1], self.N).to(x.dtype)
        if self._smallm is not None and R <= self.SMALLM_ROWS_MAX:
            bn, kc, sk_sm = self._smallm_cfg
            out = self._smallm(rows.to(torch.bfloat16), self.packed[0], self.scales[0],
                               block_n=bn, kc=kc, sk=sk_sm,
                               workspace=(self._smallm_part, self._smallm_cnt))
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*x.shape[:-1], self.N).to(x.dtype)
        w = self._bf16_weight()
        out = rows.to(torch.bfloat16) @ w.t()
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.N).to(x.dtype)

    def _bf16_weight(self) -> torch.Tensor:
        """The dequantised weight, built on first use (warm-up, before any
        graph capture) and kept: the same int4 values the GEMV serves, in
        the layout cuBLAS reads once."""
        w = self._bf16_cache
        if w is None:
            w = self._deq()
            self._bf16_cache = w
        return w

    def _decode_bufs(self, R: int):
        """Row-count-keyed GEMV buffers. Allocated on first use for each R
        (the warm-up pass, before any CUDA-graph capture) and reused, so a
        captured replay never allocates. R=1 reuses the buffers built at
        construction."""
        if R == 1:
            return self._eid0, self._part
        cache = self._decode_cache
        if R not in cache:
            sk = self._part.shape[0]
            eids = torch.zeros(R, dtype=torch.int32, device=self._part.device)
            part = torch.empty(sk * R, self.N, dtype=torch.float32,
                               device=self._part.device)
            cache[R] = (eids, part)
        return cache[R]

    def _deq(self) -> torch.Tensor:
        lo = (self.packed.to(torch.int16) & 0xF) - 8
        hi = ((self.packed.to(torch.int16) >> 4) & 0xF) - 8
        q = torch.stack([lo, hi], dim=-1).reshape(self.N, self.K).float()
        w = (q.reshape(self.N, self.K // 32, 32)
             * self.scales.reshape(self.N, self.K // 32).float()[..., None])
        return w.reshape(self.N, self.K).to(torch.bfloat16)


def resolve_smallm(smallm=None, *, banner=print) -> bool:
    """The K16 route's default, decided ONCE per enable (K16 P5 read, bench/k16/RESULTS-k16-p5.md: -1.06 ms/step
    at B=16 on the 5090). ``E4B_ATTN_INT4_SMALLM``: ``1`` requires the kernel (refuses without it), ``0`` keeps the
    cached-bf16 path, unset/``auto`` routes when the installed grouped-nf4-gemm carries ``int4_smallm`` (>= 0.32.0)
    and says so in one line when it does not -- never a silent fallback, never a refusal on an older cut."""
    import os
    if smallm is None:
        v = os.environ.get("E4B_ATTN_INT4_SMALLM", "auto").strip().lower()
        smallm = {"1": True, "true": True, "0": False, "false": False}.get(v, "auto")
    if smallm is True:
        try:
            _smallm_kernels()
        except ImportError as e:
            raise RuntimeError(
                "E4B_ATTN_INT4_SMALLM=1 needs grouped-nf4-gemm with int4_smallm (lane K16, >= 0.32.0) "
                f"(missing: {e}); install that cut or unset the flag -- the route is never substituted silently"
            ) from e
        return True
    if smallm is False:
        return False
    try:
        _smallm_kernels()
        return True
    except ImportError as e:
        banner(f"[e4b.int4_attn] K16 small-M route OFF: the installed grouped-nf4-gemm has no int4_smallm ({e}); "
               "rows 2..16 take the cached-bf16 matmul (E4B_ATTN_INT4_SMALLM=1 to require the route, =0 to silence this)")
        return False


def enable_serve_attn_int4(model, smallm: bool | None = None) -> int:
    """Swap every structural attention projection for Int4Linear.
    Returns the count; refuses a vacuous enable. lm_head untouched.

    ``smallm`` (default: ``E4B_ATTN_INT4_SMALLM``, ``auto``) routes ``1 < rows <= 16``
    to the K16 small-M int4 GEMM when the kernel is installed (see
    :func:`resolve_smallm`); ``=1`` with the kernel absent refuses here, never
    at forward time."""
    try:
        _kernels()
    except ImportError as e:
        raise RuntimeError(
            "E4B_SERVE_ATTN_INT4=1 needs grouped-nf4-gemm with int4_b32 "
            f"(missing: {e}); install the matching cut or unset the flag"
        ) from e
    smallm = resolve_smallm(smallm)
    n = 0
    for mod in model.modules():
        if type(mod).__name__.endswith("Attention"):
            for name, child in list(mod.named_children()):
                if type(child) is nn.Linear:
                    setattr(mod, name, Int4Linear(child, smallm=smallm))
                    n += 1
    if n == 0:
        raise RuntimeError("E4B_SERVE_ATTN_INT4=1 matched no attention "
                           "projections -- refusing a vacuous enable")
    return n
