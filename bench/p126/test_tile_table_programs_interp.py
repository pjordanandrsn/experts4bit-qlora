"""``build_group_tiles_fused(..., rank="cumsum", programs=P)`` (e4b#846): the cumsum tile table split over P programs.

The one-program table at 128 experts and 512 routed rows walks every row twice in ``[128, 64]`` chunks (e4b P122: 2.51
ms a 64-row step, 52 us a launch). ``_tile_table_cumsum_mp`` gives each program a slice of the experts: every program
takes all experts' counts from one histogram of the ids, then ranks and writes only its own experts' rows. What must
hold, bit for bit, at every P it accepts:
- ``programs=None`` and ``programs=1`` are the one-program kernel, launched exactly as before;
- the five tables (and the lean variant's six) equal ``programs=1``'s, the chained builder's, and the pairwise rank's
  where it applies; ``order`` equals ``torch.argsort(expert_ids, stable=True)``;
- every row is written by exactly ONE program (the kernel's ``OWNERS`` count), including P larger than the expert
  count and P that does not divide the padded expert axis;
- the histogram (``tl.histogram``, the masked rows' sentinel in a bin of its own) and the counting pass give the same
  integers.

Runs under ``TRITON_INTERPRET=1`` on CPU (named in ``.github/workflows/ci.yml``'s interpreter job and guarded in
``conftest._INTERP_FILES``); compiled on a CUDA device (``TRITON_INTERPRET=0``) the same cases run."""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

import int4_b32                                                                       # noqa: E402
from int4_b32 import _programs_slice, build_group_tiles_fused                       # noqa: E402
from nf4_grouped import build_group_tiles_device                                     # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
if not INTERP and not torch.cuda.is_available():
    pytest.skip("compiled mode needs a CUDA device", allow_module_level=True)
DEV = "cpu" if INTERP else "cuda"
NAMES = ("row0", "rows", "grp", "order", "counts")
BM = 16


def _ids(R, E, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        ids = torch.randint(0, E, (R,), generator=g)
    elif kind == "skewed":                     # a few experts take most rows; many experts stay empty
        p = torch.softmax(torch.randn(E, generator=g) * 3, 0)
        ids = torch.multinomial(p, R, replacement=True, generator=g)
    else:                                      # one expert, owned by one program, holds every row
        ids = torch.full((R,), E - 1, dtype=torch.int64)
    return ids.to(torch.int32).to(DEV)


def _eq(a, b, what):
    for name, x, y in zip(NAMES + ("sids",), a, b):
        assert torch.equal(x.cpu(), y.cpu()), f"{what}: {name} differs"


class _Spy:
    def __init__(self, fn):
        self.fn, self.launches = fn, []

    def __getitem__(self, grid):
        def launch(*a, **kw):
            self.launches.append((tuple(grid), {k: v for k, v in kw.items()}))
            return self.fn[grid](*a, **kw)
        return launch


@pytest.mark.parametrize("programs", [None, 1])
def test_one_program_is_the_kernel_as_before(monkeypatch, programs):
    ids = _ids(512, 128, "uniform", 0)
    r1, mp = _Spy(int4_b32._tile_table_r1), _Spy(int4_b32._tile_table_cumsum_mp)
    monkeypatch.setattr(int4_b32, "_tile_table_r1", r1)
    monkeypatch.setattr(int4_b32, "_tile_table_cumsum_mp", mp)
    got = build_group_tiles_fused(ids, 128, BM, rank="cumsum", programs=programs)
    ref = build_group_tiles_fused(ids, 128, BM, rank="cumsum")
    assert mp.launches == [] and len(r1.launches) == 2 and r1.launches[0] == r1.launches[1]
    assert r1.launches[0][0] == (1,)
    _eq(got, ref, f"programs={programs}")


@pytest.mark.parametrize("E", [40, 128, 256])
@pytest.mark.parametrize("P", [2, 3, 8, 64])
@pytest.mark.parametrize("R", [1, 100, 257, 512, 1024])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "one"])
def test_every_program_count_gives_the_same_integers(E, P, R, kind):
    ids = _ids(R, E, kind, seed=R + E + P)
    ref = build_group_tiles_fused(ids, E, BM, rank="cumsum")                   # one program, the shipped table
    got = build_group_tiles_fused(ids, E, BM, rank="cumsum", programs=P)
    _eq(got, ref, f"E={E} P={P} R={R} {kind}")
    assert torch.equal(got[3].cpu(), torch.argsort(ids.cpu().long(), stable=True))
    if R <= 256:
        _eq(got, build_group_tiles_fused(ids, E, BM, rank="pairwise"), f"pairwise E={E} P={P} R={R} {kind}")
        _eq(got, build_group_tiles_device(ids, E, BM), f"chained E={E} P={P} R={R} {kind}")


@pytest.mark.parametrize("E,P", [(40, 3), (40, 64), (128, 3), (128, 8), (256, 7), (128, 200)])
@pytest.mark.parametrize("hist", [True, False])
def test_every_row_has_exactly_one_owner(E, P, hist):
    """P > E (programs that own no expert) and P not dividing the padded expert axis (3, 7) included."""
    for R, kind in ((512, "uniform"), (300, "skewed"), (257, "one")):
        ids = _ids(R, E, kind, seed=E * P + R)
        owners = torch.zeros(R, dtype=torch.int32, device=DEV)
        got = build_group_tiles_fused(ids, E, BM, rank="cumsum", programs=P, _hist=hist, _owners=owners)
        assert torch.equal(owners.cpu(), torch.ones(R, dtype=torch.int32)), \
            f"E={E} P={P} R={R} {kind}: rows written {owners.unique().tolist()} times"
        _eq(got, build_group_tiles_fused(ids, E, BM, rank="cumsum"), f"E={E} P={P} R={R} {kind} hist={hist}")


def test_the_slices_partition_the_experts():
    for E in range(1, 300):
        for P in range(1, 70):
            ebp = _programs_slice(E, P)
            assert ebp & (ebp - 1) == 0 and P * ebp >= E, (E, P, ebp)
            owner = [e // ebp for e in range(E)]                        # slice [p*ebp, (p+1)*ebp) owns e
            assert all(0 <= o < P for o in owner), (E, P)


@pytest.mark.parametrize("hist", [True, False])
@pytest.mark.parametrize("E,P", [(128, 8), (40, 3), (256, 64)])
def test_lean_and_sorted_ids_match_at_every_program_count(E, P, hist):
    ids = _ids(512, E, "skewed", seed=E + P)
    ref = build_group_tiles_fused(ids, E, BM, rank="cumsum", lean=True, sorted_ids=True)
    got = build_group_tiles_fused(ids, E, BM, rank="cumsum", lean=True, sorted_ids=True, programs=P, _hist=hist)
    assert len(got) == 6
    _eq(got, ref, f"lean E={E} P={P} hist={hist}")
    budget = -(-512 // BM) + E
    assert got[0].numel() == budget and int(got[1][int((got[1] > 0).sum()):].abs().sum()) == 0   # padding zeroed


def test_refusals():
    ids = _ids(64, 40, "uniform", 1)
    for bad in (0, -2, 1.5):
        with pytest.raises(ValueError, match="programs="):
            build_group_tiles_fused(ids, 40, BM, rank="cumsum", programs=bad)
    with pytest.raises(ValueError, match="splits the cumsum rank"):
        build_group_tiles_fused(ids, 40, BM, rank="pairwise", programs=4)
