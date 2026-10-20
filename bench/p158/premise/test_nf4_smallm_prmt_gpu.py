# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K25's ``lut="prmt"`` (lane K39), COMPILED on CUDA: bit-identical to ``lut="tree"`` at experts4bit-qlora's served
plan. That is the BITWISE tier: the same codebook values and the same tiling, instruction shape and accumulation
order, with only the decode differing. Skips without a CUDA device.

The cases mirror the A2000 suite K39 was ruled on (394/394 bit-equal on real W16 routing, under triton 3.4.0):
- **Routings:** seeded W16 routings (16 tokens, top-8 distinct experts of 128) through gate_up on the lean route
  (``gather_div`` 8 + ``scatter``) and through down (gather + scatter).
- **Edge cases:** one expert carrying all 128 rows (8 tiles); N not a multiple of BLOCK_N; a masked K tail; the bf16
  path; no gather.
- **Determinism and capture:** the same call twice, and a CUDA-graph capture with the tile table built inside it,
  replayed over changed routings against eager ``"tree"``."""
import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip:
    pytest.skip("needs an NVIDIA CUDA device (K39's decode is inline PTX)", allow_module_level=True)

import int4_b32  # noqa: E402
from nf4_smallm import gemm_nf4_grouped_smallm  # noqa: E402

DEV = torch.device("cuda")
E, TOPK = 128, 8
SERVED = dict(block_n=32, kc=64, warps=4, stages=3, dot_bf16=False)


def _store(n, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, generator=g).to(DEV)
    absmax = (torch.rand(E, n, k // 64, generator=g) * 0.09 + 0.01).to(DEV)
    return packed, absmax


def _routing(seed, tokens=16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(tokens)]).reshape(-1).to(DEV)


def _tables(ids):
    t_row0, t_rows, t_group, order, _counts = int4_b32.build_group_tiles_fused(ids, E, 16)
    return t_row0, t_rows, t_group, order


def _both(*args, **kw):
    return gemm_nf4_grouped_smallm(*args, lut="prmt", **kw), gemm_nf4_grouped_smallm(*args, lut="tree", **kw)


@pytest.mark.parametrize("seed", range(8))
def test_gate_up_on_the_lean_route_is_bit_identical(seed):
    packed, absmax = _store(1536, 2048, 100 + seed)
    t = _tables(_routing(seed))
    x = torch.randn(16, 2048, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)).to(torch.bfloat16)
    a, b = _both(x, packed, absmax, *t, scatter=t[3], gather_div=TOPK, **SERVED)
    assert torch.equal(a, b), f"{int((a != b).sum())} outputs differ"


@pytest.mark.parametrize("seed", range(8))
def test_down_is_bit_identical(seed):
    packed, absmax = _store(2048, 768, 200 + seed)
    t = _tables(_routing(seed))
    x = torch.randn(128, 768, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)).to(torch.bfloat16)
    a, b = _both(x, packed, absmax, *t, scatter=t[3], **SERVED)
    assert torch.equal(a, b), f"{int((a != b).sum())} outputs differ"


@pytest.mark.parametrize("case", ["one_expert_8_tiles", "n_1544", "kc128_k832", "bf16", "no_gather"])
def test_the_edge_cases_are_bit_identical(case):
    n, k, kw, ids = 2048, 768, dict(SERVED), _routing(7)
    if case == "one_expert_8_tiles":
        ids = torch.full((128,), 5, dtype=torch.long, device=DEV)
    elif case == "n_1544":
        n = 1544
    elif case == "kc128_k832":
        k, kw["kc"] = 832, 128
    elif case == "bf16":
        kw["dot_bf16"] = True
    packed, absmax = _store(n, k, 300)
    t = _tables(ids)
    x = torch.randn(ids.numel(), k, device=DEV).to(torch.bfloat16)
    if case == "no_gather":
        a, b = _both(x, packed, absmax, t[0], t[1], t[2], **kw)
    else:
        a, b = _both(x, packed, absmax, *t, scatter=t[3], **kw)
    assert torch.equal(a, b), f"{case}: {int((a != b).sum())} outputs differ"


def test_it_is_deterministic():
    packed, absmax = _store(2048, 768, 400)
    t = _tables(_routing(3))
    x = torch.randn(128, 768, device=DEV).to(torch.bfloat16)
    one = gemm_nf4_grouped_smallm(x, packed, absmax, *t, scatter=t[3], lut="prmt", **SERVED)
    assert torch.equal(one, gemm_nf4_grouped_smallm(x, packed, absmax, *t, scatter=t[3], lut="prmt", **SERVED))


def test_a_captured_graph_replays_bit_identical_to_eager_tree_as_the_routing_changes():
    packed, absmax = _store(1536, 2048, 500)
    ids = _routing(0).clone()
    x = torch.randn(16, 2048, device=DEV).to(torch.bfloat16)
    kw = dict(SERVED, gather_div=TOPK, lut="prmt")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            t = _tables(ids)
            gemm_nf4_grouped_smallm(x, packed, absmax, *t, scatter=t[3], **kw)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        t = _tables(ids)
        y = gemm_nf4_grouped_smallm(x, packed, absmax, *t, scatter=t[3], **kw)
    for seed in (11, 12, 13):
        ids.copy_(_routing(seed))
        g.replay()
        torch.cuda.synchronize()
        te = _tables(ids)
        want = gemm_nf4_grouped_smallm(x, packed, absmax, *te, scatter=te[3], gather_div=TOPK, lut="tree", **SERVED)
        assert torch.equal(y, want), f"routing {seed}: {int((y != want).sum())} outputs differ"
