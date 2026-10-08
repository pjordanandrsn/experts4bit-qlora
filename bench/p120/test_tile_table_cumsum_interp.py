"""``build_group_tiles_fused(..., rank="cumsum")`` (e4b#846): the one-launch tile table above 256 routed rows.

The pairwise rank caps the one-launch builder at R <= 256 (its [RB, RB] compare). The cumsum rank counts each expert's
rows up to row ``r`` along the existing [EB, RB] hit matrix, so R up to 1024 fits one program. What must hold, bit for
bit, at every R it accepts:
- the permutation ``order`` equals ``torch.argsort(expert_ids, stable=True)``;
- all five tables (row0, rows, grp, order, counts) equal the chained builder's (``build_group_tiles_device``), and
  the lean variant's six outputs equal the default's;
- at R <= 256, the cumsum and pairwise kernels give the same integers.
Routing covers uniform, skewed (a few experts take most rows), single-expert and empty experts.

Runs under ``TRITON_INTERPRET=1`` on CPU (named in ``.github/workflows/ci.yml``'s interpreter job and guarded in
``conftest._INTERP_FILES``); compiled on a CUDA device (``TRITON_INTERPRET=0``) the same cases run."""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from int4_b32 import build_group_tiles_fused      # noqa: E402
from nf4_grouped import build_group_tiles_device  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
if not INTERP and not torch.cuda.is_available():
    pytest.skip("compiled mode needs a CUDA device", allow_module_level=True)
DEV = "cpu" if INTERP else "cuda"
NAMES = ("row0", "rows", "grp", "order", "counts")


def _ids(R, E, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        ids = torch.randint(0, E, (R,), generator=g)
    elif kind == "skewed":                     # a few experts take most rows; many experts stay empty
        p = torch.softmax(torch.randn(E, generator=g) * 3, 0)
        ids = torch.multinomial(p, R, replacement=True, generator=g)
    elif kind == "one":                        # every row on one expert: the longest tie run
        ids = torch.full((R,), E // 2, dtype=torch.int64)
    else:
        raise ValueError(kind)
    return ids.to(DEV)


@pytest.mark.parametrize("R", [1, 7, 255, 256, 257, 384, 512, 1000, 1024])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "one"])
@pytest.mark.parametrize("E,BM", [(128, 16), (64, 32)])
def test_cumsum_rank_order_is_stable_argsort_and_tables_match_the_chained_builder(R, kind, E, BM):
    ids = _ids(R, E, kind, seed=R * 7 + E)
    got = build_group_tiles_fused(ids.to(torch.int32), E, BM, rank="cumsum")
    assert torch.equal(got[3], torch.argsort(ids, stable=True)), "order is not the stable argsort"
    ref = build_group_tiles_device(ids.to(torch.int32), E, BM)
    for n, x, y in zip(NAMES, ref, got):
        assert x.dtype == y.dtype, (n, x.dtype, y.dtype)
        assert torch.equal(x, y), (n, R, kind)


@pytest.mark.parametrize("R", [257, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_cumsum_lean_variant_gives_the_default_calls_integers(R, dtype):
    ids = _ids(R, 128, "skewed", seed=R).to(dtype)
    a = build_group_tiles_fused(ids, 128, 16, rank="cumsum")
    b = build_group_tiles_fused(ids, 128, 16, rank="cumsum", lean=True, sorted_ids=True)
    assert len(b) == 6
    for n, x, y in zip(NAMES, a, b):
        assert x.dtype == y.dtype and torch.equal(x, y), n
    assert b[5].dtype == dtype and torch.equal(b[5], ids.index_select(0, a[3]))


@pytest.mark.parametrize("R", [1, 64, 200, 256])
def test_cumsum_and_pairwise_agree_where_both_apply(R):
    ids = _ids(R, 128, "uniform", seed=100 + R).to(torch.int32)
    a = build_group_tiles_fused(ids, 128, 16)                    # rank="pairwise", the default
    b = build_group_tiles_fused(ids, 128, 16, rank="cumsum")
    for n, x, y in zip(NAMES, a, b):
        assert torch.equal(x, y), n


def test_the_caps_and_an_unknown_rank_refuse():
    ids = torch.zeros(1025, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="decode-only"):
        build_group_tiles_fused(ids, 128, 16, rank="cumsum")
    with pytest.raises(ValueError, match="decode-only"):
        build_group_tiles_fused(ids[:257], 128, 16)              # the pairwise default keeps its 256 cap
    with pytest.raises(ValueError, match="rank="):
        build_group_tiles_fused(ids[:8], 128, 16, rank="radix")


def test_the_order_check_can_fail():
    """The stable-argsort comparison is not vacuous: swapping two tied rows of a correct order is detected."""
    ids = torch.tensor([3, 1, 3, 1, 3], device=DEV)
    order = build_group_tiles_fused(ids.to(torch.int32), 8, 16, rank="cumsum")[3]
    bad = order.clone()
    bad[[2, 3]] = bad[[3, 2]]                    # rows 0, 2, 4 share expert 3; swap two of them
    assert torch.equal(order, torch.argsort(ids, stable=True))
    assert not torch.equal(bad, torch.argsort(ids, stable=True))
