# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_ATTN_INT4_WIDE (e4b#846): the attention projections' 17..64-row route on the K16 small-M GEMM.

A batched decode step above 16 rows takes ``Int4Linear``'s cached bf16 copy and cuBLAS. grouped-nf4-gemm's
``gemm_int4_b32_smallm(..., block_m=)`` (#522) serves up to 64 rows from the int4 bytes with a 32- or 64-row tile.
These pin, on the CPU with the kernel package stubbed as ``tests/test_int4_attn.py`` stubs it:
  - ``wide=True`` sends 17..64 rows to the small-M GEMM through a workspace sized for 64 rows; 2..16 rows keep their
    construction-time workspace (K16, unchanged), one row the GEMV, more than 64 rows the bf16 matmul;
  - the workspace is ONE per (device, width, plan), shared by every module of that width and built at construction,
    zeroed, before any capture; a lookup that misses under a capture raises rather than allocates;
  - ``E4B_ATTN_INT4_WIDE``: ``0`` (the default) changes nothing; ``1`` needs the K16 route and a kernel with
    ``block_m=`` (detected by signature), refused otherwise; anything else refused; fuse and enable carry it.
And on a CUDA device with the real kernel (the A2000 correctness run; skipped elsewhere): two ``Int4Linear`` of one
width captured in ONE graph, sharing the workspace, replay to their eager bits and within one bf16 ulp of the dequant
reference; and two graphs that both take the route (32 rows captured first, 64 second) replay to their eager bits with
the second replayed FIRST -- the case a workspace born inside the first capture would get wrong.
"""
import inspect
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(__file__))
attn_tests = pytest.importorskip("test_int4_attn")          # its CPU kernel stubs (skips where int4_pack_ref is absent)

from torch import nn  # noqa: E402


def _stubs(monkeypatch, calls, block_m=True):
    """test_int4_attn's stubs, with a ``gemm_int4_b32_smallm`` that takes ``block_m=`` (#522) or, for an older
    kernel package, does not."""
    attn_tests._cpu_kernel_stubs(monkeypatch, calls)
    import int4_pack_ref
    import int4_smallm

    def _gemm(x, packed, scales, block_n, kc, sk, workspace, bm):
        M = int(x.shape[0])
        if M > (64 if block_m else 16):
            raise ValueError(f"stub kernel serves at most {64 if block_m else 16} rows (got {M})")
        bm = bm or (16 if M <= 16 else (32 if M <= 32 else 64))
        if workspace is not None:
            part, cnt = workspace
            assert part.numel() >= sk * bm * part.shape[-1], "workspace too small for the tile"
        calls.append(("smallm", M, bm, (block_n, kc, sk), None if workspace is None else workspace))
        N, kh = packed.shape
        w = int4_pack_ref.dequant_int4_ref(packed.reshape(N, kh), scales.reshape(N, kh * 2 // 32), N, kh * 2)
        return (x.float() @ w.t()).to(torch.bfloat16)

    if block_m:
        def gemm_int4_b32_smallm(x, packed, scales, *, block_n=64, kc=128, sk=4, warps=4, stages=2, workspace=None,
                                 dot_bf16=None, block_m=None):
            return _gemm(x, packed, scales, block_n, kc, sk, workspace, block_m)
    else:
        def gemm_int4_b32_smallm(x, packed, scales, *, block_n=64, kc=128, sk=4, warps=4, stages=2, workspace=None,
                                 dot_bf16=None):
            return _gemm(x, packed, scales, block_n, kc, sk, workspace, None)
    int4_smallm.gemm_int4_b32_smallm = gemm_int4_b32_smallm
    from experts4bit_qlora.engines import int4_attn
    monkeypatch.setattr(int4_attn, "_WIDE_WS", {})                # every test starts with no shared workspace
    return int4_attn


def _lin(n_out=96, k=64, seed=5):
    torch.manual_seed(seed)
    return nn.Linear(k, n_out, bias=False, dtype=torch.bfloat16)


def test_wide_serves_17_to_64_rows_through_a_64_row_workspace(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    m = ia.Int4Linear(_lin(), smallm=True, wide=True)
    import int4_pack_ref
    ref_w = int4_pack_ref.dequant_int4_ref(m.packed[0], m.scales[0], 96, 64)
    for R, tile in ((17, 32), (32, 32), (33, 64), (64, 64)):
        x = torch.randn(R, 64, dtype=torch.bfloat16)
        y = m(x)
        kind, rows, bm, cfg, ws = calls[-1]
        assert (kind, rows, bm, cfg) == ("smallm", R, tile, m._smallm_cfg), calls[-1]
        assert tuple(ws[0].shape) == (m._smallm_cfg[2], 64, 96), "the wide workspace is sized for 64 rows"
        assert torch.allclose(y.float(), x.float() @ ref_w.t(), rtol=2e-2, atol=2e-2)
    assert m._bf16_cache is None                                                     # no second copy up to 64 rows
    m(torch.randn(16, 64, dtype=torch.bfloat16))
    assert calls[-1][1:3] == (16, 16) and calls[-1][4][0] is m._smallm_part          # 2..16: K16's own workspace
    m(torch.randn(65, 64, dtype=torch.bfloat16))
    assert calls[-1][0] == "smallm" and m._bf16_cache is not None                    # 65 rows: the bf16 matmul
    m(torch.randn(1, 64, dtype=torch.bfloat16))
    assert calls[-1][0] == "gemv"                                                    # one row: the GEMV


def test_without_wide_17_rows_keep_the_bf16_matmul(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    m = ia.Int4Linear(_lin(), smallm=True)
    m(torch.randn(17, 64, dtype=torch.bfloat16))
    assert not any(c[0] == "smallm" and c[1] == 17 for c in calls) and m._bf16_cache is not None
    assert ia._WIDE_WS == {}


def test_the_workspace_is_one_per_width_and_plan_built_at_construction(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    a, b = (ia.Int4Linear(_lin(seed=s), smallm=True, wide=True) for s in (1, 2))
    c = ia.Int4Linear(_lin(n_out=32, seed=3), smallm=True, wide=True)
    built = dict(ia._WIDE_WS)
    assert len(built) == 2 and int(sum(int(w[1].sum()) for w in built.values())) == 0      # before any forward, zeroed
    x = torch.randn(40, 64, dtype=torch.bfloat16)
    a(x)
    b(x)
    c(x)
    a(x[:20])
    ws = [call[4] for call in calls if call[0] == "smallm"]
    assert ws[0] is ws[1] is ws[3], "modules of one width share one workspace on a stream"
    assert ws[2] is not ws[0] and ws[2][0].shape[-1] == 32, "another width gets its own"
    assert ia._WIDE_WS == built and all(any(w is v for v in built.values()) for w in ws), "forwards allocate nothing"
    assert ia.wide_workspace_bytes() == sum(t.numel() * t.element_size() for w in (ws[0], ws[2]) for t in w)


def test_a_lookup_that_misses_under_a_capture_raises(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="this stream is capturing"):
        ia._wide_workspace(96, 64, 4, torch.device("cuda", 0))          # raises before touching the device
    assert ia._WIDE_WS == {}


def test_resolve_wide(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    for v in (None, "", "0", " 0 "):
        if v is None:
            monkeypatch.delenv("E4B_ATTN_INT4_WIDE", raising=False)
        else:
            monkeypatch.setenv("E4B_ATTN_INT4_WIDE", v)
        assert ia.resolve_wide(True) is False
    monkeypatch.setenv("E4B_ATTN_INT4_WIDE", "1")
    assert ia.resolve_wide(True) is True
    with pytest.raises(RuntimeError, match="E4B_ATTN_INT4_WIDE=1 needs the K16 small-M route"):
        ia.resolve_wide(False)
    monkeypatch.setenv("E4B_ATTN_INT4_WIDE", "2")
    with pytest.raises(ValueError, match="E4B_ATTN_INT4_WIDE='2': expected '0' or '1'"):
        ia.resolve_wide(True)
    assert ia.resolve_wide(True, wide=False) is False                                # an explicit flag wins


def test_wide_is_refused_on_a_kernel_without_block_m(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls, block_m=False)
    import int4_smallm
    assert "block_m" not in inspect.signature(int4_smallm.gemm_int4_b32_smallm).parameters
    monkeypatch.setenv("E4B_ATTN_INT4_WIDE", "1")
    with pytest.raises(RuntimeError, match="takes block_m="):
        ia.resolve_wide(True)
    monkeypatch.delenv("E4B_ATTN_INT4_WIDE")
    assert ia.resolve_wide(True) is False                                            # the default never asks


def test_wide_needs_smallm_at_construction(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    with pytest.raises(ValueError, match="needs smallm=True"):
        ia.Int4Linear(_lin(), smallm=False, wide=True)


def test_enable_resolves_and_carries_the_flag(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _lin()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    monkeypatch.delenv("E4B_ATTN_INT4_SMALLM", raising=False)
    monkeypatch.delenv("E4B_ATTN_INT4_WIDE", raising=False)
    m = M()
    assert ia.enable_serve_attn_int4(m) == 1 and m.attn.q_proj._wide is False        # default: off
    monkeypatch.setenv("E4B_ATTN_INT4_WIDE", "1")
    m1 = M()
    assert ia.enable_serve_attn_int4(m1) == 1 and m1.attn.q_proj._wide is True
    monkeypatch.setenv("E4B_ATTN_INT4_SMALLM", "0")
    with pytest.raises(RuntimeError, match="needs the K16 small-M route"):
        ia.enable_serve_attn_int4(M())


def test_fuse_carries_wide_and_refuses_mixed(monkeypatch):
    calls = []
    ia = _stubs(monkeypatch, calls)
    parts = [ia.Int4Linear(_lin(n_out=n, seed=n), smallm=True, wide=True) for n in (64, 16, 16)]
    fused = ia.Int4Linear.fuse(parts)
    assert fused._wide is True and fused.N == 96
    x = torch.randn(48, 64, dtype=torch.bfloat16)
    y = fused(x)
    assert calls[-1][1:3] == (48, 64)
    assert torch.allclose(y.float(), torch.cat([p(x) for p in parts], dim=-1).float(), rtol=1e-2, atol=1e-2)
    with pytest.raises(ValueError, match="wide route"):
        ia.Int4Linear.fuse([parts[0], ia.Int4Linear(_lin(n_out=16), smallm=True)])


# -- CUDA, the real kernel (the A2000 correctness run) ----------------------------------------------------------------
def _real_wide_kernel():
    if not torch.cuda.is_available():
        return None
    for name in ("int4_smallm", "int4_b32", "int4_pack_ref"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None) is None:
            return None                                                              # a stub is installed
    try:
        from int4_smallm import gemm_int4_b32_smallm
    except ImportError:
        return None
    return gemm_int4_b32_smallm if "block_m" in inspect.signature(gemm_int4_b32_smallm).parameters else None


needs_wide_cuda = pytest.mark.skipif(_real_wide_kernel() is None,
                                     reason="needs CUDA and grouped-nf4-gemm whose gemm_int4_b32_smallm takes block_m=")


def _ulp_ok(y, ref):
    return bool((y.float() - ref).abs().max() <= 2 ** -7 * ref.abs().max())


@needs_wide_cuda
def test_two_modules_of_one_width_in_one_graph_replay_to_their_references(monkeypatch):
    from experts4bit_qlora.engines import int4_attn as ia
    monkeypatch.setattr(ia, "_WIDE_WS", {})
    dev = torch.device("cuda")
    N, K, R = 1024, 2048, 48                                     # K16's plan splits K four ways: partials are used
    mods = []
    for seed in (11, 12):
        torch.manual_seed(seed)
        lin = nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device=dev)
        mods.append(ia.Int4Linear(lin, smallm=True, wide=True))
    a, b = mods
    assert a._smallm_cfg == b._smallm_cfg and a._smallm_cfg[2] > 1
    ref_w = [m._deq().float() for m in mods]
    x1 = torch.randn(R, K, dtype=torch.bfloat16, device=dev)
    x2 = torch.randn(R, K, dtype=torch.bfloat16, device=dev)

    def step():
        return a(x1), b(x2), a(x2)                               # three launches, one width, one stream

    side = torch.cuda.Stream(dev)                                # warm up as paged_runner does: a side stream ...
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream(dev).wait_stream(side)
    torch.cuda.synchronize(dev)
    built = dict(ia._WIDE_WS)
    assert len(built) == 1, "one workspace for the width, built at construction"
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                                    # ... then capture on the capture stream
        outs = step()
    torch.cuda.synchronize(dev)
    assert ia._WIDE_WS == built, "the capture allocated a workspace"
    for seed in (21, 22, 23):
        torch.manual_seed(seed)
        x1.copy_(torch.randn_like(x1))
        x2.copy_(torch.randn_like(x2))
        g.replay()
        torch.cuda.synchronize(dev)
        eager = step()                                           # the default stream: its own workspace
        for got, want in zip(outs, eager):
            assert torch.equal(got, want), "a graph replay sharing one workspace moved a bit"
        refs = (x1.float() @ ref_w[0].bfloat16().float().t(), x2.float() @ ref_w[1].bfloat16().float().t(),
                x2.float() @ ref_w[0].bfloat16().float().t())
        for got, ref in zip(outs, refs):
            assert _ulp_ok(got, ref), "replayed output is not within one bf16 ulp of the dequant reference"
    assert ia.wide_workspace_bytes() > 0


@needs_wide_cuda
def test_two_graphs_on_the_route_replay_in_either_order(monkeypatch):
    """Graph A (32 rows) is captured first, graph B (64 rows) second, both through the one workspace; B replays first.
    A workspace born inside A's capture would be zeroed only when A replays, and B would start from garbage counters."""
    from experts4bit_qlora.engines import int4_attn as ia
    monkeypatch.setattr(ia, "_WIDE_WS", {})
    dev = torch.device("cuda")
    N, K = 1024, 2048
    torch.manual_seed(31)
    a = ia.Int4Linear(nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device=dev), smallm=True, wide=True)
    b = ia.Int4Linear(nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device=dev), smallm=True, wide=True)
    x32 = torch.randn(32, K, dtype=torch.bfloat16, device=dev)
    x64 = torch.randn(64, K, dtype=torch.bfloat16, device=dev)
    graphs, outs = [], []
    for fn in (lambda: (a(x32), b(x32)), lambda: (b(x64), a(x64))):
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs.append(fn())
        graphs.append(g)
    torch.cuda.synchronize(dev)
    assert len(ia._WIDE_WS) == 1
    for seed in (41, 42):
        torch.manual_seed(seed)
        x32.copy_(torch.randn_like(x32))
        x64.copy_(torch.randn_like(x64))
        graphs[1].replay()                                       # B first
        torch.cuda.synchronize(dev)
        want = (b(x64), a(x64))
        assert all(torch.equal(g_, w_) for g_, w_ in zip(outs[1], want)), "B replayed first moved a bit"
        graphs[0].replay()
        torch.cuda.synchronize(dev)
        want = (a(x32), b(x32))
        assert all(torch.equal(g_, w_) for g_, w_ in zip(outs[0], want)), "A replayed after B moved a bit"
