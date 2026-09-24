# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P64 (e4b#709): ``E4B_INT4_DECODE_A16`` routes the int4 store's T == 1 expert
calls through the prefill branch -- dequant, bf16 matmul -- instead of the GEMV
on int8-quantised activations.

What is pinned, on CPU, with the int4 kernels stubbed by pure-torch references
(the real pack/dequant from ``int4_pack_ref``):

* off (the default): T == 1 calls ``quant_x_rows`` then ``gemv_int4_b32`` per
  projection, never the dequant loop, and returns exactly what those two calls
  return -- the route as it is today;
* on: T == 1 never calls ``quant_x_rows`` or the GEMV, dequantises each routed
  expert, and returns bit for bit what the prefill branch returns for the same
  rows; T > 1 is untouched either way;
* the batched-decode GEMV route (device grouping, <= 256 rows) is NOT covered by
  the flag, as documented;
* the flag refuses under CUDA-graph capture (the dequant loop reads the expert
  ids on the host).

The stubs are installed per test with monkeypatch (test_singleton_groups.py's
rule: never module-level, so the real kernels are neither shadowed nor leaked).
"""
import sys
import types

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("int4_pack_ref")   # gnf4 builds predating the int4 lane

from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

E, H, INTER, TOPK = 6, 64, 32, 4        # K is a multiple of 32 on both projections


def _quant_ref(x):
    """Per-(row, 32-block) symmetric int8, int4_b32._quant_x_rows' arithmetic in torch."""
    R, K = x.shape
    b = x.float().reshape(R, K // 32, 32)
    s = b.abs().amax(-1) / 127.0 + 1e-12
    q = torch.clamp(torch.floor(b / s[..., None] + 0.5), -127, 127)
    return q.to(torch.int8).reshape(R, K), s


def _gemv_ref(xq, xs, packed, scales, eids, N, K, part=None):
    """The GEMV's contract on the int8 rows: fp32 accumulate, one bf16 cast."""
    R = xq.shape[0]
    x = (xq.float().reshape(R, K // 32, 32) * xs[..., None]).reshape(R, K)
    out = torch.empty(R, N, dtype=torch.bfloat16)
    for r, e in enumerate(eids.tolist()):
        out[r] = (x[r] @ dequant_int4_ref(packed[e], scales[e], N, K).t()).to(torch.bfloat16)
    return out


@pytest.fixture
def kernels(monkeypatch):
    """Counting stubs for int4_b32 (quant + GEMV), a counting wrapper on the REAL
    dequant_int4_ref, and an nf4_grouped stand-in where the real one cannot import
    (macOS: no triton). Resets the flag to off around every test."""
    calls = {"quant": 0, "gemv": 0, "dequant": 0}
    stub = types.ModuleType("int4_b32")

    def quant_x_rows(x):
        calls["quant"] += 1
        return _quant_ref(x)

    def gemv_int4_b32(*a, **k):
        calls["gemv"] += 1
        return _gemv_ref(*a, **k)

    stub.quant_x_rows = quant_x_rows
    stub.gemv_int4_b32 = gemv_int4_b32
    monkeypatch.setitem(sys.modules, "int4_b32", stub)

    import int4_pack_ref
    real = int4_pack_ref.dequant_int4_ref

    def counting_dequant(*a, **k):
        calls["dequant"] += 1
        return real(*a, **k)

    monkeypatch.setattr(int4_pack_ref, "dequant_int4_ref", counting_dequant)
    try:
        import nf4_grouped  # noqa: F401
    except ImportError:
        def _no_nf4(*a, **k):
            raise AssertionError("the NF4 grouped GEMM was reached on an int4-store call")
        monkeypatch.setitem(sys.modules, "nf4_grouped", types.SimpleNamespace(gemm_4bit_grouped=_no_nf4))
    old = hr.DECODE_A16[0]            # a one-element list (the FORCE_SINGLETON_GROUPS pattern): saved, restored
    hr.DECODE_A16[0] = False
    hr._SWIGLU.clear()
    hr._COMBINE.clear()
    try:
        yield calls
    finally:
        hr.DECODE_A16[0] = old
        hr._SWIGLU.clear()
        hr._COMBINE.clear()


def _stores(seed=11):
    g = torch.Generator().manual_seed(seed)
    gu_w = torch.randn(E, 2 * INTER, H, generator=g) * 0.1
    dn_w = torch.randn(E, H, INTER, generator=g) * 0.1

    def pack_stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk), torch.stack(sc)
    gu_p, gu_s = pack_stack(gu_w)
    dn_p, dn_s = pack_stack(dn_w)
    return {"gu": {"packed": gu_p, "scales": gu_s, "N": 2 * INTER, "K": H},
            "dn": {"packed": dn_p, "scales": dn_s, "N": H, "K": INTER}}


FREED_GU = torch.empty(0, 0, 0, dtype=torch.uint8)     # distinct sentinels: `pk is gu_p` names the slot
FREED_DN = torch.empty(0, 0, 0, dtype=torch.uint8)
FREED_A = torch.empty(0, 0, 0)
SHAPES = (2 * INTER, H, H, INTER)


def _call(x, ids, stores, **kw):
    return hr._fused_over_stack(x, ids, FREED_GU, FREED_A, FREED_DN, FREED_A, SHAPES, True, F.silu,
                                int4_stores=stores, **kw)


def _decode_rows(seed=5):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(1, H, generator=g) * 0.5).to(torch.bfloat16).expand(TOPK, H).contiguous()
    ids = torch.randperm(E, generator=g)[:TOPK]           # a token's top-k ids are distinct
    return x, ids


def test_default_is_off_and_reads_the_env_once(monkeypatch):
    monkeypatch.delenv("E4B_INT4_DECODE_A16", raising=False)
    assert hr._decode_a16_default() is False
    for v, want in (("1", True), ("0", False), ("true", False), ("", False)):
        monkeypatch.setenv("E4B_INT4_DECODE_A16", v)
        assert hr._decode_a16_default() is want, v
    assert isinstance(hr.DECODE_A16, list) and len(hr.DECODE_A16) == 1


def test_off_t1_is_the_gemv_on_int8_activations(kernels):
    stores = _stores()
    x, ids = _decode_rows()
    out = _call(x, ids, stores, singleton_groups=True)
    assert kernels == {"quant": 2, "gemv": 2, "dequant": 0}
    # what the route returns is exactly its two kernel calls composed -- today's bytes
    e32 = ids.to(torch.int32)
    gu = _gemv_ref(*_quant_ref(x), stores["gu"]["packed"], stores["gu"]["scales"], e32, 2 * INTER, H)
    gate, up = gu.chunk(2, dim=-1)
    h = (F.silu(gate) * up).contiguous()
    want = _gemv_ref(*_quant_ref(h), stores["dn"]["packed"], stores["dn"]["scales"], e32, H, INTER)
    assert torch.equal(out, want)


def test_on_t1_takes_the_prefill_branch_and_never_quantises(kernels, monkeypatch):
    stores = _stores()
    x, ids = _decode_rows()
    a8 = _call(x, ids, stores, singleton_groups=True)
    kernels.update(quant=0, gemv=0, dequant=0)

    hr.DECODE_A16[0] = True
    a16 = _call(x, ids, stores, singleton_groups=True)
    assert kernels["quant"] == 0 and kernels["gemv"] == 0, kernels
    assert kernels["dequant"] == 2 * TOPK, kernels            # one per routed row per projection

    # bit for bit the prefill branch on the same rows (host grouping: sort, one row per distinct id, unsort)
    kernels.update(dequant=0)
    hr.DECODE_A16[0] = False
    prefill = _call(x, ids, stores)
    assert kernels["quant"] == 0 and kernels["dequant"] == 2 * TOPK
    assert torch.equal(a16, prefill)
    # and against an independent bf16 reference of the same int4 values
    ref = torch.empty(TOPK, H, dtype=torch.bfloat16)
    for r, e in enumerate(ids.tolist()):
        w_gu = dequant_int4_ref(stores["gu"]["packed"][e], stores["gu"]["scales"][e], 2 * INTER, H)
        w_dn = dequant_int4_ref(stores["dn"]["packed"][e], stores["dn"]["scales"][e], H, INTER)
        gu = x[r:r + 1] @ w_gu.to(torch.bfloat16).t()
        g_, u_ = gu.chunk(2, dim=-1)
        ref[r] = (((F.silu(g_) * u_)) @ w_dn.to(torch.bfloat16).t())[0]
    assert torch.equal(a16, ref)
    # the lever is real: bf16 and int8 activations give different outputs
    assert not torch.equal(a16, a8)


def test_on_leaves_prefill_untouched(kernels, monkeypatch):
    stores = _stores()
    g = torch.Generator().manual_seed(8)
    x = (torch.randn(24, H, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, E, (24,), generator=g)           # duplicates: real grouping
    off = _call(x, ids, stores)
    n_off = dict(kernels)
    kernels.update(quant=0, gemv=0, dequant=0)
    hr.DECODE_A16[0] = True
    on = _call(x, ids, stores)
    assert torch.equal(off, on)
    assert kernels == n_off and n_off["quant"] == 0 and n_off["gemv"] == 0


def test_on_does_not_cover_the_device_grouped_decode_gemv(kernels, monkeypatch):
    """The batched-decode GEMV (device grouping, <= 256 rows, hot_residency's
    `_int4_gemv_decode`) keeps its int8 activations: the flag is T == 1's only."""
    stores = _stores()
    x = (torch.randn(8, H) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, E, (8,))
    hr.DECODE_A16[0] = True
    _call(x, ids, stores, device_grouping=True)
    assert kernels["quant"] == 2 and kernels["gemv"] == 2 and kernels["dequant"] == 0


def test_forward_collapsed_t1_follows_the_flag(kernels, monkeypatch):
    """Through the collapse the serving stack runs: T == 1 is where the flag bites,
    T > 1 is the prefill branch in both states."""
    stores = _stores()
    m = hr._HotResidency.__new__(hr._HotResidency)
    m._rt_cache = None
    m.h_gu_p, m.h_gu_a, m.h_dn_p, m.h_dn_a = FREED_GU, FREED_A, FREED_DN, FREED_A
    m.shapes, m.has_gate, m.act_fn = SHAPES, True, F.silu
    m.gptoss, m.clamp_limit, m._int4_stores = False, None, stores

    def fwd(T, seed):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
        flat = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).reshape(-1)
        w = torch.softmax(torch.randn(T, TOPK, generator=g), -1)
        return m._forward_collapsed(x, flat, w, T, TOPK, H, torch.device("cpu"), torch.device("cpu"), torch.bfloat16)

    off1 = fwd(1, 3)
    assert kernels["quant"] == 2 and kernels["dequant"] == 0
    off4 = fwd(4, 4)
    kernels.update(quant=0, gemv=0, dequant=0)
    hr.DECODE_A16[0] = True
    on1 = fwd(1, 3)
    assert kernels["quant"] == 0 and kernels["gemv"] == 0 and kernels["dequant"] == 2 * TOPK
    on4 = fwd(4, 4)
    assert torch.equal(off4, on4)
    assert not torch.equal(off1, on1)


def test_on_refuses_under_capture(kernels, monkeypatch):
    """The dequant loop's `eids.tolist()` is a host read: under capture the flag
    refuses with a sentence instead of failing deep in the graph."""
    stores = _stores()
    x, ids = _decode_rows()

    class _CudaView(torch.Tensor):     # a CPU tensor that reports itself as CUDA, nothing else changed
        @property
        def is_cuda(self):
            return True

    hr.DECODE_A16[0] = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager-only"):
        _call(x.as_subclass(_CudaView), ids, stores, singleton_groups=True)
    # off: the capture check is never consulted
    hr.DECODE_A16[0] = False
    kernels.update(quant=0)
    _call(x.as_subclass(_CudaView), ids, stores, singleton_groups=True)
    assert kernels["quant"] == 2
