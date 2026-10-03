# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The all-resident collapse's row-to-token index under bucketed CUDA graphs (#913, lane P98's replay fault).

``_HotResidency._forward_collapsed`` builds ``rt`` (the token each expert row reads) once per row count and caches it.
``PagedModelRunner.enable_decode_graphs`` captures one graph per bucket, each after an EAGER warm-up of its own row
count. A single-entry cache therefore frees the earlier bucket's ``rt`` when the next bucket warms up, while the
earlier bucket's graph still reads that address; a CUDA graph keeps no reference to tensors it reads. Once the
caching allocator hands the block out again, the earlier bucket's replay gathers token rows through whatever was
written there. P98 hit it as ``indexSelectSmallIndex: srcIndex < srcSelectDimSize`` at bucket 1, whose singleton
route gathers ``x.index_select(0, rt)``.

Here K25 is off (``E4B_NF4_GROUPED_SMALLM=0``), so every row count gathers through ``rt``. Buckets 2 and 4 are captured
in that order, warmed on one side stream as ``_capture_bucket`` warms them. The allocator then hands that stream's
small blocks back out zero-filled (a freed block is reused only on its own stream; a reused ``rt`` reads token 0 for
every row: a clean mismatch rather than a process-killing out-of-range read), and bucket 2 replays new inputs. It must
equal the eager call on the same inputs, bit for bit. A first version churned on the default stream and passed with
the fix reverted: an inert check.

Any CUDA card (the NF4 M-tile and the tile builder are Triton, no fp8); skips without one.
"""
import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for graph capture")

DEV = "cuda"
E, H, INTER, TOPK = 16, 512, 256, 4


def _stub():
    from nf4_pack_ref import quantize_pack_nf4

    from experts4bit_qlora.engines import hot_residency as hr
    g = torch.Generator().manual_seed(0)

    def stack(n, k):
        ps, as_ = zip(*[quantize_pack_nf4(torch.randn(n, k, generator=g) * 0.02) for _ in range(E)])
        return torch.stack(ps).to(DEV), torch.stack(as_).to(DEV)

    m = hr._HotResidency.__new__(hr._HotResidency)
    m._rt_cache = None
    m.h_gu_p, m.h_gu_a = stack(2 * INTER, H)
    m.h_dn_p, m.h_dn_a = stack(H, INTER)
    m.shapes, m.has_gate, m.act_fn = (2 * INTER, H, H, INTER), True, F.silu
    m.gptoss, m.clamp_limit, m._int4_stores = False, None, None
    return hr, m


def _inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, H, generator=g) * 0.5).to(DEV, torch.bfloat16)
    flat = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).reshape(-1).to(DEV)
    w = torch.softmax(torch.randn(T, TOPK, generator=g), -1).to(DEV)
    return x, flat, w


@needs_cuda
def test_an_earlier_buckets_replay_keeps_its_row_to_token_index(monkeypatch):
    pytest.importorskip("nf4_pack_ref", reason="needs grouped-nf4-gemm's NF4 packer")
    pytest.importorskip("nf4_grouped", reason="needs grouped-nf4-gemm's NF4 grouped GEMM")
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "0")             # every row count gathers through rt
    hr, m = _stub()
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])            # the batched lane's capture-safe grouping
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    dev = torch.device(DEV)

    def call(x, flat, w, T):
        return m._forward_collapsed(x, flat, w, T, TOPK, H, dev, dev, torch.bfloat16)

    graphs = {}
    # ONE side stream for every warm-up and for the churn below: the caching allocator reuses a freed block only on the
    # stream it was allocated on, and the warm-ups (where rt is built) run on a side stream, as in _capture_bucket
    side = torch.cuda.Stream()
    for T in (2, 4):                                              # enable_decode_graphs' order: warm, then capture
        x_s, f_s, w_s = (t.clone() for t in _inputs(T, 10 + T))
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(2):
                call(x_s, f_s, w_s, T)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = call(x_s, f_s, w_s, T)
        graphs[T] = (g, x_s, f_s, w_s, out)
    # hand the side stream's small blocks back out, zero-filled: a freed rt now reads token 0 for every row
    with torch.cuda.stream(side):
        churn = [torch.zeros(n, dtype=torch.long, device=DEV) for n in (2 * TOPK, 4 * TOPK) for _ in range(64)]
    torch.cuda.synchronize()
    g, x_s, f_s, w_s, out = graphs[2]
    for seed in (101, 102, 103):
        x, flat, w = _inputs(2, seed)
        x_s.copy_(x)
        f_s.copy_(flat)
        w_s.copy_(w)
        g.replay()
        torch.cuda.synchronize()
        with torch.no_grad():
            eager = call(x, flat, w, 2)
        torch.cuda.synchronize()
        assert torch.equal(out, eager), (
            f"seed {seed}: bucket 2's replay differs from eager after bucket 4 warmed up "
            f"(maxabs {(out.float() - eager.float()).abs().max():.3e})")
    del churn
