#!/usr/bin/env python3
"""$0 harness validation: with SIMULATED uniform routing (no model), the counter must reproduce
the occupancy null  read_fraction(n) = 1 - (1-k/E)^n  to within sampling error, and its degenerate
behaviors (per-forward reset, per-layer independence) must hold. Run: python -m pytest this -q."""

import math

import torch

from access_counter import ExpertAccessCounter, attach


def _uniform_logits(n_tokens: int, E: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_tokens, E, generator=g)  # argsort of iid normals == uniform top-k


def test_reproduces_occupancy_null():
    E, k = 128, 8
    q = 1 - k / E
    for n in [1, 4, 16, 36, 128, 512]:
        # average many independent forwards to beat down sampling noise on E[distinct]
        fracs = []
        c = ExpertAccessCounter(E, k)
        for s in range(200):
            c.note_logits(0, _uniform_logits(n, E, seed=1000 * s + n))
            c.close_step(s, batch=1, seq=n)
        fracs = [r["read_fraction"] for r in c.records]
        got = sum(fracs) / len(fracs)
        want = 1 - q**n
        assert abs(got - want) < 0.03, f"n={n}: got {got:.3f} want {want:.3f}"


def test_per_forward_reset():
    """Distinct sets must reset each close_step — otherwise the union only ever grows."""
    c = ExpertAccessCounter(128, 8)
    c.note_logits(0, _uniform_logits(2000, 128, seed=1))  # saturates ~all experts
    c.close_step(0, 1, 2000)
    c.note_logits(0, _uniform_logits(1, 128, seed=2))  # a single token -> exactly k distinct
    c.close_step(1, 1, 1)
    assert c.records[1]["n_distinct"] == 8


def test_per_layer_independent():
    c = ExpertAccessCounter(128, 8)
    c.note_logits(0, _uniform_logits(4, 128, seed=1))
    c.note_logits(1, _uniform_logits(4, 128, seed=2))
    c.close_step(0, 1, 4)
    layers = {r["layer"] for r in c.records}
    assert layers == {0, 1}
    # each layer sees <= 4*8 = 32 distinct, independently
    for r in c.records:
        assert r["n_distinct"] <= 32


def test_indices_seam_matches_logits_seam():
    c1 = ExpertAccessCounter(128, 8)
    logits = _uniform_logits(50, 128, seed=7)
    idx = torch.topk(logits, 8, dim=-1).indices
    c1.note_logits(0, logits)
    c1.close_step(0, 1, 50)
    c2 = ExpertAccessCounter(128, 8)
    c2.note_indices(0, idx)
    c2.close_step(0, 1, 50)
    assert c1.records[0]["n_distinct"] == c2.records[0]["n_distinct"]


def test_attach_finds_gate_by_output_dim():
    class FakeMoEBlock(torch.nn.Module):
        def __init__(self, hidden, E):
            super().__init__()
            self.gate = torch.nn.Linear(hidden, E, bias=False)
            self.attn_o = torch.nn.Linear(hidden, hidden, bias=False)  # must NOT be hooked

        def forward(self, x):
            _ = self.attn_o(x)
            return self.gate(x)

    class FakeModel(torch.nn.Module):
        def __init__(self, hidden, E, n_layers):
            super().__init__()
            self.blocks = torch.nn.ModuleList(FakeMoEBlock(hidden, E) for _ in range(n_layers))

        def forward(self, x):
            for b in self.blocks:
                b(x)
            return x

    E = 128
    m = FakeModel(64, E, n_layers=3)
    c = ExpertAccessCounter(E, 8)
    n_hooked = attach(m, c)
    assert n_hooked == 3  # three gates, the attn_o square layers excluded
    m(torch.randn(10, 64))
    c.close_step(0, 1, 10)
    assert len({r["layer"] for r in c.records}) == 3
