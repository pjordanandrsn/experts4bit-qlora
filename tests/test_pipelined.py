# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Pipelined residency correctness + the zero-sync token loop.

Proves the address-dispatched engine (pinned arena + hot stack + k-slot
store) reproduces the reference forward at decode across hot splits, that
every guarded fallback lands on the reference path, that the three forward
patches ([fast], v0 hot-residency, pipelined) are mutually exclusive, and —
the design law — that the decode step issues **no host synchronization**
(``torch.cuda.set_sync_debug_mode``-enforced). Skips unless CUDA +
grouped-nf4-gemm are present.
"""
import pytest
import torch

pytest.importorskip("nf4_grouped")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

@pytest.fixture(autouse=True)
def _no_triton_interpreter():
    """Runtime (order-proof) guard: the address-gather is compiled-only — raw
    device/UVA pointers segfault the host-side Triton interpreter. When an
    interpreter-contract suite has set TRITON_INTERPRET=1 in this process
    (it does so at import), these tests skip; run them in separate pytest
    invocations to execute both."""
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("Triton interpreter mode active (raw-pointer gather is compiled-only)")


from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.pipelined import (  # noqa: E402
    disable_pipelined_residency,
    enable_pipelined_residency,
    pipelined_available,
)


def _make(E=8, H=128, inter=64, k=3, seed=0, has_gate=True):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * inter if has_gate else inter, H)
    down = torch.randn(E, H, inter)
    mod = Experts4bit.from_float(gate_up_proj=gate_up.cuda(), down_proj=down.cuda(),
                                 compute_dtype=torch.bfloat16, has_gate=has_gate)
    return mod


def _route(E, k, seed):
    torch.manual_seed(seed)
    hs = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
    tw, ti = torch.topk(torch.softmax(torch.randn(1, E, device="cuda"), -1), k=k, dim=-1)
    return hs, ti, tw.to(torch.bfloat16)


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def test_pipelined_available():
    assert pipelined_available()


@pytest.mark.parametrize("hot", [[], [0, 1, 2, 3], list(range(8)), [5]])
def test_decode_matches_reference_across_hot_splits(hot):
    mod = _make()
    hs, ti, tw = _route(8, 3, seed=1)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    n = enable_pipelined_residency(mod, [torch.tensor(hot, dtype=torch.long)],
                                   device="cuda", k_slots=3)
    assert n == 1
    with torch.no_grad():
        got = mod(hs, ti, tw)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert _b_rel(got, ref) < 1.5e-2, (hot, _b_rel(got, ref))
    assert disable_pipelined_residency(mod) == 1
    with torch.no_grad():
        back = mod(hs, ti, tw)
    torch.testing.assert_close(back.float(), ref.float(), rtol=0, atol=0)


def test_sequential_decode_slot_reuse():
    # 12 steps with churning routes: have-skip caching must never serve stale
    # bytes for a *different* expert, and repeated routes must stay exact.
    mod = _make(seed=3)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    for s in range(12):
        hs, ti, tw = _route(8, 3, seed=100 + (s % 5))  # cycle: repeats + changes
        with torch.no_grad():
            got = mod(hs, ti, tw)
        disable_pipelined_residency(mod)
        with torch.no_grad():
            ref = mod(hs, ti, tw)
        enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
        assert _b_rel(got, ref) < 1.5e-2, (s, _b_rel(got, ref))
    disable_pipelined_residency(mod)


def test_k_is_config_not_code_path():
    # same module, K swept by re-enable: pure streaming -> full residency,
    # one code path, correctness flat.
    mod = _make(seed=4)
    hs, ti, tw = _route(8, 3, seed=9)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    for K in (0, 2, 4, 8):
        enable_pipelined_residency(mod, [torch.arange(K)], device="cuda", k_slots=3)
        with torch.no_grad():
            got = mod(hs, ti, tw)
        assert _b_rel(got, ref) < 1.5e-2, (K, _b_rel(got, ref))
        disable_pipelined_residency(mod)


def test_zero_sync_token_loop():
    # THE exit criterion (kill sheet): after warmup, N decode steps under
    # torch.cuda.set_sync_debug_mode("error") — any .item()/nonzero/D2H in
    # the loop raises RuntimeError and fails this test.
    mod = _make(seed=6)
    enable_pipelined_residency(mod, [torch.tensor([0, 1, 2])], device="cuda", k_slots=3)
    routes = [_route(8, 3, seed=200 + s) for s in range(8)]
    with torch.no_grad():
        for hs, ti, tw in routes[:3]:   # warmup: triton JIT compile syncs, allowed
            mod(hs, ti, tw)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.no_grad():
            for hs, ti, tw in routes[3:]:
                mod(hs, ti, tw)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    disable_pipelined_residency(mod)


def test_prefill_and_wrong_k_fall_back():
    mod = _make(seed=7)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    torch.manual_seed(11)
    hs = torch.randn(5, 128, dtype=torch.bfloat16, device="cuda")   # T>1: prefill
    tw, ti = torch.topk(torch.softmax(torch.randn(5, 8, device="cuda"), -1), k=3, dim=-1)
    with torch.no_grad():
        got = mod(hs, ti, tw.to(torch.bfloat16))
    disable_pipelined_residency(mod)
    with torch.no_grad():
        ref = mod(hs, ti, tw.to(torch.bfloat16))
    torch.testing.assert_close(got.float(), ref.float(), rtol=0, atol=0)  # same path, exact

    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    hs1, ti2, tw2 = _route(8, 2, seed=12)   # k=2 != k_slots=3: fallback
    with torch.no_grad():
        got2 = mod(hs1, ti2, tw2)
    disable_pipelined_residency(mod)
    with torch.no_grad():
        ref2 = mod(hs1, ti2, tw2)
    torch.testing.assert_close(got2.float(), ref2.float(), rtol=0, atol=0)


def test_training_falls_back():
    mod = _make(seed=8)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    hs, ti, tw = _route(8, 3, seed=13)
    x = hs.clone().requires_grad_(True)
    out = mod(x, ti, tw)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    disable_pipelined_residency(mod)


def test_mutual_exclusion_all_three_patches():
    from experts4bit_qlora.fast import disable_fast, enable_fast
    from experts4bit_qlora import disable_hot_residency, enable_hot_residency

    # pipelined active -> fast and v0 refuse
    mod = _make(seed=9)
    enable_pipelined_residency(mod, [torch.tensor([0])], device="cuda", k_slots=3)
    assert enable_fast(mod) == 0
    assert enable_hot_residency(mod, [torch.tensor([0])], device="cuda") == 0
    disable_pipelined_residency(mod)

    # fast active -> pipelined refuses
    mod2 = _make(seed=9)
    assert enable_fast(mod2) == 1
    assert enable_pipelined_residency(mod2, [torch.tensor([0])], device="cuda", k_slots=3) == 0
    disable_fast(mod2)

    # v0 active -> pipelined refuses
    mod3 = _make(seed=9)
    assert enable_hot_residency(mod3, [torch.tensor([0])], device="cuda") == 1
    assert enable_pipelined_residency(mod3, [torch.tensor([0])], device="cuda", k_slots=3) == 0
    disable_hot_residency(mod3)


def test_validation_errors():
    mod = _make(seed=10)
    with pytest.raises(ValueError, match="k_slots"):
        enable_pipelined_residency(mod, [torch.tensor([0])], device="cuda")
    with pytest.raises(ValueError, match="hot ids"):
        enable_pipelined_residency(mod, [torch.tensor([99])], device="cuda", k_slots=3)
    with pytest.raises(ValueError, match="entries"):
        enable_pipelined_residency(mod, [torch.tensor([0]), torch.tensor([1])],
                                   device="cuda", k_slots=3)


def test_reenable_refreshes_from_current_weights():
    # the Bugbot rule from v0: a cached partition must never go stale
    mod = _make(seed=14)
    hs, ti, tw = _route(8, 3, seed=15)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    with torch.no_grad():
        before = mod(hs, ti, tw)
    with torch.no_grad():   # perturb the packed weights (simulates a reload)
        mod.gate_up_proj.data.copy_(torch.roll(mod.gate_up_proj.data, 1, dims=-1))
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    with torch.no_grad():
        after = mod(hs, ti, tw)
    disable_pipelined_residency(mod)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    assert _b_rel(after, ref) < 1.5e-2
    assert _b_rel(before, ref) > 1.5e-2  # stale partition WOULD have been wrong


def test_gptoss_epilogue_matches_reference():
    from experts4bit_qlora.gptoss import GptOssExperts4bit

    torch.manual_seed(0)
    E, H, I, k = 8, 128, 64, 4
    gate_up_dense = torch.randn(E, H, 2 * I) * 0.1
    gate_up_bias = torch.randn(E, 2 * I) * 0.1
    down_dense = torch.randn(E, I, H) * 0.1
    down_bias = torch.randn(E, H) * 0.1
    mod = GptOssExperts4bit.from_gptoss(gate_up_dense.cuda(), gate_up_bias.cuda(),
                                        down_dense.cuda(), down_bias.cuda(),
                                        compute_dtype=torch.bfloat16).cuda()
    hs = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    sc, idx = torch.topk(torch.softmax(torch.randn(1, E, device="cuda"), -1), k=k, dim=-1)
    sc = sc.to(torch.bfloat16)
    with torch.no_grad():
        ref = mod(hs, idx, sc)
    for hot in ([], [0, 1, 2, 3], list(range(E))):
        n = enable_pipelined_residency(mod, [torch.tensor(hot, dtype=torch.long)],
                                       device="cuda", k_slots=k)
        assert n == 1
        with torch.no_grad():
            got = mod(hs, idx, sc)
        assert _b_rel(got, ref) < 1.5e-2, (hot, _b_rel(got, ref))
        disable_pipelined_residency(mod)
