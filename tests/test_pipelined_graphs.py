# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CUDA-graph capture of the pipelined decode step (Phase 3, guarded).

The engine's step is fixed-shape, id-indirect through device memory, and
allocation-stable, so a captured step must replay correctly as routes churn —
including the gather's have-skip behavior evolving across replays. Graph
correctness is its own gate: replay must match eager within the P1 tolerance
at every K; environments whose capture fails must fall back to eager, so a
capture failure here SKIPS (reported), never blocks. Lead-time hint tests
ride along (mechanism, not tuning).
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
)


def _make(E=8, H=128, inter=64, seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * inter, H)
    down = torch.randn(E, H, inter)
    return Experts4bit.from_float(gate_up_proj=gate_up.cuda(), down_proj=down.cuda(),
                                  compute_dtype=torch.bfloat16, has_gate=True)


def _routes(E, k, n, seed):
    torch.manual_seed(seed)
    out = []
    for _ in range(n):
        hs = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        tw, ti = torch.topk(torch.softmax(torch.randn(1, E, device="cuda"), -1), k=k, dim=-1)
        out.append((hs, ti, tw.to(torch.bfloat16)))
    return out


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.parametrize("hot", [[], [0, 1, 2, 3], list(range(8))])
def test_graph_replay_matches_eager(hot):
    mod = _make()
    enable_pipelined_residency(mod, [torch.tensor(hot, dtype=torch.long)],
                               device="cuda", k_slots=3)
    routes = _routes(8, 3, 10, seed=21)

    # static I/O buffers + warmup on a side stream (standard capture recipe)
    x_st = torch.zeros_like(routes[0][0])
    i_st = torch.zeros_like(routes[0][1])
    w_st = torch.zeros_like(routes[0][2])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for hs, ti, tw in routes[:3]:
            x_st.copy_(hs); i_st.copy_(ti); w_st.copy_(tw)
            mod(x_st, i_st, w_st)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g), torch.no_grad():
            out_st = mod(x_st, i_st, w_st)
    except RuntimeError as e:
        disable_pipelined_residency(mod)
        pytest.skip(f"capture unavailable here (eager fallback is the contract): {e}")

    # replay across churning routes; compare to eager on a fresh identical module
    got = []
    for hs, ti, tw in routes[3:]:
        x_st.copy_(hs); i_st.copy_(ti); w_st.copy_(tw)
        g.replay()
        got.append(out_st.clone())
    torch.cuda.synchronize()
    disable_pipelined_residency(mod)
    for (hs, ti, tw), o in zip(routes[3:], got):
        with torch.no_grad():
            ref = mod(hs, ti, tw)          # reference forward (patch disabled)
        assert _b_rel(o, ref) < 1.5e-2, (hot, _b_rel(o, ref))


def test_hint_preserves_correctness():
    # wrong hints, right hints, absent hints — the forward's own gather must
    # keep the result identical (hint is overlap, never semantics)
    mod = _make(seed=31)
    routes = _routes(8, 3, 6, seed=33)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=3)
    st = mod._pipelined
    outs = []
    with torch.no_grad():
        for j, (hs, ti, tw) in enumerate(routes):
            if j % 3 == 1:
                st.hint(ti)                                  # perfect hint
            elif j % 3 == 2:
                st.hint(torch.randint(0, 8, (1, 3), device="cuda"))  # wrong hint
            outs.append(mod(hs, ti, tw))
    disable_pipelined_residency(mod)
    with torch.no_grad():
        for (hs, ti, tw), o in zip(routes, outs):
            ref = mod(hs, ti, tw)
            assert _b_rel(o, ref) < 1.5e-2
