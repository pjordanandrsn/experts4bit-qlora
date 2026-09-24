# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The expert dispatch routes lane P63 read EXACT, asserted with ``torch.equal`` on a GPU (#708).

A token decoded alone (T = 1: ``_fused_over_stack`` on the singleton route, then the combine) and the same token
inside a T-token call must come out bit for bit the same on these routes:

- **the int4 store under** ``FORCE_SINGLETON_GROUPS`` (the S2 verify default): one group per row, so every row
  takes the T = 1 kernel, ``gemv_int4_b32``; with ``_int4_part_or_none`` handing it its own buffer when the
  store's does not fit (``e4b.serve.p63.qwen3.int4-singleton.row-exact.5090.2026-09-24``);
- **the int4 store under** ``DEVICE_GROUPING`` at <= 256 routed rows: the same GEMV in input order
  (``e4b.serve.p63.qwen3.int4-device-gemv.row-exact.5090.2026-09-24``); above 256 rows it takes the grouped
  int4 GEMM, which is reorder-class and not asserted here;
- **the NF4 stack under** ``FORCE_SINGLETON_GROUPS``: the decode GEMV at every row count, which on a >= 160-SM
  part at Qwen3's shapes is the dot-pad kernel (``e4b.serve.p63.qwen3.nf4-singleton.row-exact.5090.2026-09-24``).

The kernels' own row-invariance is grouped-nf4-gemm's to pin (``kernel/test_row_invariance_gpu.py`` there); this
file pins that the DISPATCH keeps every row on them. The tests force the plan an RTX 5090 gets (above
``SPLITK_R_TERM_MAX_SMS``, and dot-pad engaged), so they assert the routes P63 read on any CUDA part, on random
weights at Qwen3-30B-A3B's expert shapes. ``tests/test_singleton_groups.py`` pins the dispatch algebra through a
mocked GEMM on CPU; it cannot see a kernel switch, which is why these exist.

``test_the_check_sees_the_default_route_change_the_function`` is the control: the default T > 1 route on the
int4 store (host-grouped dequant + bf16 matmul, which P63 read PRECISION) must NOT come out bit-equal.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the routes' row-count invariance is a property of the compiled kernels")

DEV = "cuda"
TOP_K = 8
EXPERTS = 16
H, INTER = 2048, 768                       # Qwen3-30B-A3B: hidden, per-expert intermediate
SHAPES = (2 * INTER, H, H, INTER)          # (n1, k1, n2, k2): gate_up (1536, 2048), down (2048, 768)
BIG_PART_SMS = 170                         # an RTX 5090


def _routing(T: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(T)])
    w = torch.softmax(torch.randn(T, TOP_K, generator=g), -1)
    return ids.reshape(-1).to(DEV), w.reshape(-1).to(torch.float32).to(DEV)


def _rows(T: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    return x.repeat_interleave(TOP_K, 0).contiguous()


def _int4_stores():
    """Stores shaped as ``enable_serve_experts_int4`` builds them: the NF4 stacks freed to 0-sized sentinels, a
    partials buffer sized for ONE token's ``top_k`` rows (the B=1 capture pattern)."""
    from int4_b32 import _plan
    from int4_pack_ref import pack_int4_b32
    g = torch.Generator(device="cpu").manual_seed(7)

    def stack(N, K):
        packs = [pack_int4_b32(torch.randn(N, K, generator=g) * 0.02) for _ in range(EXPERTS)]
        return (torch.stack([p for p, _ in packs]).to(DEV), torch.stack([s for _, s in packs]).to(DEV))
    stores = {}
    for name, (N, K) in (("gu", SHAPES[:2]), ("dn", SHAPES[2:])):
        P, S = stack(N, K)
        sk = _plan(N, K)[2]
        stores[name] = {"packed": P, "scales": S, "N": N, "K": K,
                        "part": torch.empty(sk * TOP_K, N, dtype=torch.float32, device=DEV)}
    freed = [torch.empty(0, 0, 0, device=DEV) for _ in range(4)]      # distinct objects: `pk is gu_p` picks the slot
    return stores, freed


def _module_rows(stacks, x_rows, ids, w, T, **route):
    """What the experts module computes after routing: the per-(token, slot) down outputs, then the combine."""
    from experts4bit_qlora.engines.hot_residency import _combine_kernel, _fused_over_stack
    gu_p, gu_a, dn_p, dn_a, stores = stacks
    dn = _fused_over_stack(x_rows, ids, gu_p, gu_a, dn_p, dn_a, SHAPES, True, F.silu,
                           int4_stores=stores, **route)
    ck = _combine_kernel()
    out = ck(dn, w, TOP_K) if ck is not None else \
        (dn.float() * w[:, None]).view(T, TOP_K, -1).sum(1).to(torch.bfloat16)
    return dn, out


def _assert_rows_match_t1(stacks, T, route, what):
    x_rows = _rows(T, seed=T)
    ids, w = _routing(T, seed=T + 1)
    t1 = dict(singleton_groups=True, device_grouping=False)            # the module's T = 1 call
    dn_b, out_b = _module_rows(stacks, x_rows, ids, w, T, **route)
    singles = [_module_rows(stacks, x_rows[t * TOP_K:(t + 1) * TOP_K].contiguous(),
                            ids[t * TOP_K:(t + 1) * TOP_K], w[t * TOP_K:(t + 1) * TOP_K], 1, **t1)
               for t in range(T)]
    torch.cuda.synchronize()
    dn_s = torch.cat([d for d, _ in singles], 0)
    out_s = torch.cat([o for _, o in singles], 0)
    bad = int((~(dn_b.view(torch.int16) == dn_s.view(torch.int16)).all(1)).sum())
    assert torch.equal(dn_b, dn_s), f"{what}: {bad} of {dn_s.shape[0]} routed rows differ from the T = 1 calls"
    assert torch.equal(out_b, out_s), f"{what}: combined token rows differ from the T = 1 calls"


@pytest.fixture
def big_part(monkeypatch):
    """The plan and the dispatch an RTX 5090 gets, on whatever CUDA part runs this."""
    import int4_b32
    import nf4_grouped
    monkeypatch.setattr(int4_b32, "_sm_count", lambda device: BIG_PART_SMS)
    monkeypatch.setattr(nf4_grouped, "_sm_count", lambda device: BIG_PART_SMS)
    for v in ("GNF4_GEMV_DOTPAD", "GNF4_GEMV_SPLITK", "GNF4_GEMV_FUSED_REDUCE", "E4B_INT4_DECODE_A16"):
        monkeypatch.delenv(v, raising=False)


@needs_cuda
@pytest.mark.parametrize("T", [16, 17, 160])
def test_int4_singleton_route_is_row_exact(big_part, T):
    stores, (gu_p, gu_a, dn_p, dn_a) = _int4_stores()
    _assert_rows_match_t1((gu_p, gu_a, dn_p, dn_a, stores), T,
                          dict(singleton_groups=True, device_grouping=False), f"int4 singleton at T={T}")


@needs_cuda
@pytest.mark.parametrize("T", [16, 17])                                # 128 / 136 routed rows: <= 256
def test_int4_device_grouping_gemv_is_row_exact(big_part, T):
    stores, (gu_p, gu_a, dn_p, dn_a) = _int4_stores()
    _assert_rows_match_t1((gu_p, gu_a, dn_p, dn_a, stores), T,
                          dict(singleton_groups=False, device_grouping=True), f"int4 device GEMV at T={T}")


@needs_cuda
@pytest.mark.parametrize("T", [16, 17, 160])
def test_nf4_singleton_route_is_row_exact(big_part, T):
    import nf4_grouped
    from nf4_pack_ref import make_stack
    gu_p, gu_a = make_stack(EXPERTS, *SHAPES[:2], seed=11, device=DEV)
    dn_p, dn_a = make_stack(EXPERTS, *SHAPES[2:], seed=12, device=DEV)
    nf4_grouped.reset_dispatch_counts()
    _assert_rows_match_t1((gu_p, gu_a, dn_p, dn_a, None), T,
                          dict(singleton_groups=True, device_grouping=False), f"NF4 singleton at T={T}")
    c = nf4_grouped.dispatch_counts()
    assert c["dotpad"] > 0 and c["dotpad"] == sum(c.values()), f"expected dot-pad on every decode call, got {c}"


@needs_cuda
def test_the_check_sees_the_default_route_change_the_function(big_part):
    stores, (gu_p, gu_a, dn_p, dn_a) = _int4_stores()
    with pytest.raises(AssertionError, match="routed rows differ"):
        _assert_rows_match_t1((gu_p, gu_a, dn_p, dn_a, stores), 17,
                              dict(singleton_groups=False, device_grouping=False), "control: int4 default at T=17")
