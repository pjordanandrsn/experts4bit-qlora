# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The determinism invariant, repaired: serve and train combine per-token
expert contributions via unique (token, slot) writes + one fixed-order
reduction — never index_add_ with duplicate token rows, whose CUDA atomic
ordering flipped output bits run to run (found by the G4 B1/B2 sandwich at
235B, filed with the G3 formal results)."""

import pytest
import torch

pytest.importorskip("nvme_arena")
pytest.importorskip("nvme_residency")
pytest.importorskip("cpu_grouped")

from experts4bit_qlora.engines import hybrid as hy  # noqa: E402
from experts4bit_qlora.engines import hybrid_train as ht  # noqa: E402

from test_hybrid_prefetch import E, H, K, two_layer  # noqa: E402,F401

needs_stack = pytest.mark.skipif(
    not hy.hybrid_available(), reason="needs CUDA + gnf4_native CPU kernels"
)


@needs_stack
def test_serve_forward_is_bit_stable_across_runs(two_layer):  # noqa: F811
    model, path, _ = two_layer
    man = {"schema": "e4b-placement/1",
           "tiers": {"vram": [[0, 0], [0, 1], [0, 2]],
                     "dram": [[0, 3], [0, 4], [0, 5]],
                     "nvme": [[0, 6], [0, 7]],
                     },
           "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}
    man["tiers"]["vram"] += [[1, e] for e in range(4)]
    man["tiers"]["dram"] += [[1, e] for e in range(4, 8)]
    hy.enable_hybrid_tier(model, path, man, hot_rows=E, threads=2)
    try:
        torch.manual_seed(7)
        hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
        idx = torch.tensor([[3, 6]], device="cuda")   # dram + nvme collide
        wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            first = model[0].experts(hidden, idx, wts)
            for _ in range(25):
                again = model[0].experts(hidden, idx, wts)
                assert torch.equal(first, again), \
                    "serve forward flipped bits between identical runs"
    finally:
        hy.disable_hybrid_tier(model)


@needs_stack
def test_train_grads_are_bit_stable_across_runs(two_layer):  # noqa: F811
    model, path, _ = two_layer
    man = {"schema": "e4b-placement/1",
           "tiers": {"vram": [[le, e] for le in range(2) for e in (0, 1, 2)],
                     "dram": [[le, e] for le in range(2) for e in (3, 4, 5)],
                     "nvme": [[le, e] for le in range(2) for e in (6, 7)]},
           "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}
    ht.enable_hybrid_train(model, path, man, hot_rows=E, threads=2)
    try:
        torch.manual_seed(9)
        hidden0 = torch.randn(2, H, dtype=torch.bfloat16, device="cuda")
        idx = torch.tensor([[0, 4], [6, 3]], device="cuda")
        wts = torch.rand(2, K, device="cuda", dtype=torch.bfloat16)
        gout = torch.randn(2, H, dtype=torch.bfloat16, device="cuda")

        def run():
            h = hidden0.clone().requires_grad_()
            out = model[0].experts(h, idx, wts)
            out.backward(gout)
            return out.detach().clone(), h.grad.clone()

        o0, g0 = run()
        for _ in range(8):
            o, g = run()
            assert torch.equal(o0, o), "train forward flipped bits"
            assert torch.equal(g0, g), "train backward flipped bits"
    finally:
        ht.disable_hybrid_train(model)
