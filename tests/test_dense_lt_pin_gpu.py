# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_DENSE_LT_PIN`` against the REAL cuBLASLt torch loaded (engines/dense_lt_pin.py; lever K, step K1), on a GPU.

The pins here are taken from cuBLASLt's own heuristic list on the card under test (the registration's selection reads
the same list and times it; these tests only need a valid configuration), for a toy o_proj and Qwen3-30B-A3B's o_proj
and router shapes, at M = 2, 4 and 8. Pins:

* the mapped library is the one torch loaded, and its version is a positive integer;
* the heuristic list holds no INPLACE reduction;
* each pinned GEMM is within the served tolerance of ``F.linear`` (max |d| / max |ref| <= 1.5e-2) and bitwise equal
  run to run;
* a CUDA-graph capture of the pinned projection replays bitwise equal to eager;
* a configuration cuBLASLt rejects is refused at build.

Skips without CUDA or off Linux. Run on a rented box, never on the A2000.
"""
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                                reason="needs CUDA on Linux")

from experts4bit_qlora.engines import dense_lt_pin as dl  # noqa: E402

SHAPES = [(256, 512), (2048, 4096), (128, 2048)]       # toy, Qwen3's o_proj, Qwen3's router gate (N, K)
TOL = 1.5e-2


class ToyAttention(nn.Module):                         # "Attention" in the name: the site the pin looks for
    def __init__(self, N, K):
        super().__init__()
        self.o_proj = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)


class Model(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.layers = nn.ModuleList([ToyAttention(N, K)])


@pytest.fixture(scope="module")
def lib():
    F.linear(torch.ones(2, 8, device="cuda", dtype=torch.bfloat16), torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    return dl.LtLib(dl.mapped_cublaslt())


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", {})
    monkeypatch.setattr(dl, "SEEN", {})


def _key(lib):
    p = torch.cuda.get_device_properties(0)
    return (lib.version(), (p.major, p.minor), p.multi_processor_count)


def _pin_all(lib, N, K, prefer_split=False):
    """A pin per M = 2, 4, 8 from cuBLASLt's own heuristic list (a split-K one when ``prefer_split`` and one exists)."""
    for M in (2, 4, 8):
        cands = lib.heuristics(lib.problem(N, K, M))
        assert cands, f"no heuristic candidate for ({N}, {K}, {M})"
        assert all(c["reduction_scheme"] in dl.REDUCTION_OK for c, _ in cands)
        pick = next((c for c, _ in cands if c["split_k"] > 1), cands[0][0]) if prefer_split else cands[0][0]
        dl.LT_PINS[(*_key(lib), N, K, M)] = pick


def test_the_mapped_library_and_its_version(lib):
    assert "libcublasLt.so" in lib.path and lib.version() > 0


@pytest.mark.parametrize("prefer_split", [False, True], ids=["first", "split-k"])
@pytest.mark.parametrize("N, K", SHAPES)
def test_the_pinned_gemm_is_within_tolerance_and_bitwise_repeatable(lib, N, K, prefer_split):
    _pin_all(lib, N, K, prefer_split)
    m = Model(N, K)
    rec = dl.apply_dense_lt_pin(m, (1, 2, 4, 8, 16), device="cuda", value="o_proj")
    assert rec["on"] and rec["pinned_m"] == {f"o_proj|{N}x{K}": [2, 4, 8]}
    lin = m.layers[0].o_proj
    for M in (2, 4, 8):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        a, b = lin(x), lin(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        ref = F.linear(x, lin.weight)
        rel = ((a.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        assert rel <= TOL, (N, K, M, rel, dl.LT_PINS[(*_key(lib), N, K, M)])
    assert all(k.endswith("|pinned") for k in dl.SEEN) and len(dl.SEEN) == 3


@pytest.mark.parametrize("N, K", SHAPES[1:])
def test_a_graph_capture_replays_bitwise_as_eager(lib, N, K):
    _pin_all(lib, N, K, prefer_split=True)
    m = Model(N, K)
    dl.apply_dense_lt_pin(m, (2, 4, 8), device="cuda", value="o_proj")
    lin = m.layers[0].o_proj
    for M in (2, 4, 8):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        with torch.no_grad():
            eager = lin(x)                               # also the warm-up: an eager call on another stream is refused
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static = lin(x)
            g.replay()
            torch.cuda.synchronize()
        torch.testing.assert_close(static, eager, rtol=0, atol=0)


def test_a_configuration_cublaslt_rejects_is_refused(lib):
    N, K = SHAPES[0]
    _pin_all(lib, N, K)
    bad = dict(dl.LT_PINS[(*_key(lib), N, K, 2)], algo_id=9999)
    dl.LT_PINS[(*_key(lib), N, K, 2)] = bad
    with pytest.raises(RuntimeError, match=r"E4B_DENSE_LT_PIN: (cuBLASLt rejects the pinned configuration|"
                                           r"MatmulAlgoInit returned)"):
        dl.apply_dense_lt_pin(Model(N, K), (2, 4, 8), device="cuda", value="o_proj")
