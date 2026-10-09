# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Three host casts the T == 1 decode no longer launches (e4b#1313, lane P127's Phase 1).

Each kernel already converts what it reads, so each host cast was one launch a layer for nothing:
- the router epilogue loads its logits with ``.to(tl.float32)``: the forwards hand it the projection's own dtype;
- ``combine_rows`` loads its weights with ``.to(tl.float32)``: it takes them in the router's dtype;
- ``gemm_4bit_grouped`` casts the expert ids to int32 on every call: the NF4 route casts them once for both calls.

The CPU tests assert what each kernel is handed (the launch is gone) and that the result is the one the cast gave
(bf16 -> fp32 widening is exact, and the ids are the same integers); the router's are in test_router_epilogue.py, with
its look-alike routers. The CUDA test runs the real kernels both ways and asserts bitwise-identical outputs."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402


def test_the_combine_kernel_is_handed_the_weights_in_the_router_dtype(monkeypatch):
    T, k, H = 1, 8, 16
    torch.manual_seed(1)
    dn = torch.randn(T * k, H, dtype=torch.bfloat16)
    tw = torch.softmax(torch.randn(T, k), dim=-1).to(torch.bfloat16)
    handed = []

    def ck(dn_, w_, k_):                       # the kernel widens on load: fp32 weight-and-sum, bf16 out
        handed.append(w_)
        return (dn_.float() * w_.float()[:, None]).view(-1, k_, dn_.shape[1]).sum(1).to(torch.bfloat16)
    monkeypatch.setattr(hr, "_combine_kernel", lambda: ck)
    monkeypatch.setattr(hr, "_kernel_tensor", lambda t: True)
    out = hr._combine_topk(dn, tw, k, T, H, torch.bfloat16, "cpu")
    assert len(handed) == 1 and handed[0].dtype == torch.bfloat16, "no host .to(float32) before the kernel"
    assert torch.equal(handed[0], tw.reshape(-1))
    assert torch.equal(out, ck(dn, tw.reshape(-1).to(torch.float32), k)), "bitwise what the fp32 weights gave"
    monkeypatch.setattr(hr, "_combine_kernel", lambda: None)           # the torch chain keeps its fp32 weights
    chain = hr._combine_topk(dn, tw, k, T, H, torch.bfloat16, "cpu")
    ref = (dn.to(torch.float32) * tw.reshape(-1).to(torch.float32)[:, None]).view(T, k, H).sum(1).to(torch.bfloat16)
    assert torch.equal(chain, ref)


def test_the_nf4_route_casts_the_expert_ids_once_for_both_calls(monkeypatch):
    seen = []
    stub = types.ModuleType("nf4_grouped")

    def gemm_4bit_grouped(xr, pk, am, sizes, eids):
        seen.append(eids)
        return torch.zeros(xr.shape[0], pk.shape[1], dtype=torch.bfloat16)
    stub.gemm_4bit_grouped = gemm_4bit_grouped
    monkeypatch.setitem(sys.modules, "nf4_grouped", stub)
    G, inter, H, k = 4, 8, 16, 8
    x_rows = torch.randn(k, H, dtype=torch.bfloat16)
    local_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64)
    gu_p = torch.zeros(G, 2 * inter, H // 2, dtype=torch.uint8)
    dn_p = torch.zeros(G, H, inter // 2, dtype=torch.uint8)
    am = torch.ones(1)
    out = hr._fused_over_stack(x_rows, local_ids, gu_p, am, dn_p, am, (2 * inter, H, H, inter), True, F.silu,
                               singleton_groups=True)
    assert out.shape == (k, H)
    assert len(seen) == 2, "gate_up and down"
    assert seen[0].dtype == torch.int32 and seen[0] is seen[1], "cast once, the same int32 ids for both calls"
    assert torch.equal(seen[0].long(), local_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the real kernels need CUDA")
def test_the_real_kernels_are_bitwise_with_the_host_casts_gone():
    """Both kernels, handed bf16 and handed its fp32 widening: every output identical, at the decode's 1 row and the
    16-row path, in both router modes and with Qwen3's (128, 8) and smaller shapes."""
    int4_b32 = pytest.importorskip("int4_b32")
    torch.manual_seed(2)
    dev = "cuda"
    for rows in (1, 16):
        for E_, k_, norm, sel in ((128, 8, True, False), (40, 8, True, False), (32, 4, False, False),
                                  (128, 4, False, True), (32, 4, False, True)):
            logits = torch.randn(rows, E_, device=dev, dtype=torch.bfloat16) * 4
            a = int4_b32.router_epilogue(logits, k_, norm, select_on_logits=sel)
            b = int4_b32.router_epilogue(logits.float(), k_, norm, select_on_logits=sel)
            assert all(x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(a, b)), (rows, E_, k_, norm, sel)
        for k_, H in ((8, 2048), (4, 2880)):
            dn = torch.randn(rows * k_, H, device=dev, dtype=torch.bfloat16)
            w = torch.softmax(torch.randn(rows, k_, device=dev), dim=-1).to(torch.bfloat16).reshape(-1)
            assert torch.equal(int4_b32.combine_rows(dn, w, k_), int4_b32.combine_rows(dn, w.float(), k_)), (rows, k_, H)


def _nf4_stub(monkeypatch, seen, *, ids_dtypes=None, gather=False):
    """A stand-in ``nf4_grouped`` whose GEMM is real arithmetic over float weight stacks (``pk [G, N, K]``), so routes
    can be compared value for value: row r reads token r // gather_div. ``ids_dtypes`` sets ``EXPERT_ID_DTYPES``
    (grouped-nf4-gemm#529); ``gather`` gives the GEMM ``gather_div=`` (#530)."""
    stub = types.ModuleType("nf4_grouped")

    def _gemm(xr, pk, am, sizes, eids, gather_div=1):
        seen.append({"xr": xr, "eids": eids, "gather_div": gather_div})
        rows = xr.repeat_interleave(gather_div, dim=0).float()
        w = pk.index_select(0, eids.long()).float()                   # [R, N, K]
        return torch.einsum("rk,rnk->rn", rows, w).to(torch.bfloat16)
    if gather:
        stub.gemm_4bit_grouped = _gemm
    else:
        def gemm_4bit_grouped(xr, pk, am, sizes, eids):
            return _gemm(xr, pk, am, sizes, eids)
        stub.gemm_4bit_grouped = gemm_4bit_grouped
    if ids_dtypes is not None:
        stub.EXPERT_ID_DTYPES = ids_dtypes
    monkeypatch.setitem(sys.modules, "nf4_grouped", stub)


def _singleton_call(T=2, k=4, G=6, inter=8, H=16, seed=3):
    torch.manual_seed(seed)
    x_t = torch.randn(T, H, dtype=torch.bfloat16)
    row_token = torch.arange(T * k) // k
    local_ids = torch.randint(0, G, (T * k,), dtype=torch.int64)
    gu_w = torch.randn(G, 2 * inter, H) / H ** 0.5
    dn_w = torch.randn(G, H, inter) / inter ** 0.5
    am = torch.ones(1)
    return dict(x_t=x_t, row_token=row_token, k=k, local_ids=local_ids,
                args=(local_ids, gu_w, am, dn_w, am, (2 * inter, H, H, inter), True, F.silu))


def test_the_nf4_route_hands_int64_ids_over_uncast(monkeypatch):
    """With a kernel package that reads int64 ids as they are (``EXPERT_ID_DTYPES``, grouped-nf4-gemm#529) both NF4
    calls get the caller's int64 ids themselves: no cast launch (P127 Phase 2, item a1)."""
    seen = []
    _nf4_stub(monkeypatch, seen, ids_dtypes=(torch.int32, torch.int64))
    c = _singleton_call()
    hr._fused_over_stack(c["x_t"].index_select(0, c["row_token"]), *c["args"], singleton_groups=True)
    assert len(seen) == 2 and seen[0]["eids"] is c["local_ids"] and seen[1]["eids"] is c["local_ids"]


@pytest.mark.parametrize("ids_dtypes", [None, (torch.int32, torch.int64)])
def test_the_nf4_singleton_route_reads_the_token_rows_bitwise(monkeypatch, ids_dtypes):
    """With ``gather_div`` in the kernel package (grouped-nf4-gemm#530) the NF4 singleton route, at ONE token, hands
    gate_up the TOKEN row with ``gather_div=top_k`` -- no (token, slot) copy -- and down its own rows; the output is
    bitwise the copied rows' (P127 Phase 2, item b2)."""
    c = _singleton_call(T=1, k=8)
    outs = []
    for gather in (False, True):
        seen = []
        _nf4_stub(monkeypatch, seen, ids_dtypes=ids_dtypes, gather=gather)
        outs.append(hr._fused_over_stack(None, *c["args"], singleton_groups=True,
                                         x_tokens=(c["x_t"], c["row_token"], c["k"])))
        if gather:
            assert seen[0]["xr"] is c["x_t"] and seen[0]["gather_div"] == c["k"], "gate_up reads the token rows"
            assert seen[1]["gather_div"] == 1 and seen[1]["xr"].shape[0] == c["local_ids"].numel()
        else:
            assert seen[0]["xr"].shape[0] == c["local_ids"].numel(), "the copied (token, slot) rows"
    assert outs[0].shape == (c["local_ids"].numel(), 16)
    assert torch.equal(outs[0], outs[1])

def test_expert_sorted_rows_at_two_tokens_keep_the_copied_rows(monkeypatch):
    """``gather_div`` assumes token-major rows (row r is token r // top_k). With more than one token the route does not
    use it at all: rows that arrive expert-sorted are copied by their own map, and the output is bitwise that copy's
    (gnf4 #530's review)."""
    c = _singleton_call(T=2, k=4)
    order = torch.argsort(c["local_ids"], stable=True)              # the rows sorted by expert, not by token
    row_token, local_ids = c["row_token"][order], c["local_ids"][order]
    assert not torch.equal(row_token, torch.arange(8) // 4), "the fixture must not be token-major"
    args = (local_ids,) + c["args"][1:]
    outs = []
    for gather in (False, True):
        seen = []
        _nf4_stub(monkeypatch, seen, gather=gather)
        outs.append(hr._fused_over_stack(None, *args, singleton_groups=True, x_tokens=(c["x_t"], row_token, c["k"])))
        assert all(s_["gather_div"] == 1 for s_ in seen), "no gather_div with two tokens"
    want = hr._fused_over_stack(c["x_t"].index_select(0, row_token), *args, singleton_groups=True)
    assert torch.equal(outs[0], want) and torch.equal(outs[1], want)
