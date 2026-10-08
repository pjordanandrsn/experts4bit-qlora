"""``build_group_tiles_fused(..., rank="cumsum", rchunk=...)`` (e4b#846): the cumsum rank walked in row chunks.

At 128 experts and 512 routed rows the one-piece cumsum built a [128, 512] hit matrix, its running count and the rank
select in one program (e4b P120: 218 us a launch, the 64-row step 1.44x slower). ``RCHUNK`` walks the rows in chunks with
a per-expert carry, so no tile exceeds [EB, RCHUNK]. What must hold, bit for bit, at every chunk and every R it accepts:
- the permutation ``order`` equals ``torch.argsort(expert_ids, stable=True)``;
- all five tables equal the chained builder's (``build_group_tiles_device``) and the one-piece cumsum's (``rchunk=0``);
- the lean variant's six outputs equal the default's;
- the automatic chunk (``rchunk=None``) is ``_cumsum_rchunk``: one piece while ``EB * RB <= CUMSUM_TILE_ELEMS``.
Chunk boundaries are where a carry error would show: routing covers one expert spanning every chunk, skewed and uniform
routing, and R that is not a multiple of the chunk.

Runs under ``TRITON_INTERPRET=1`` on CPU (named in ``.github/workflows/ci.yml``'s interpreter job and guarded in
``conftest._INTERP_FILES``); compiled on a CUDA device (``TRITON_INTERPRET=0``) the same cases run."""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from int4_b32 import CUMSUM_TILE_ELEMS, _cumsum_rchunk, build_group_tiles_fused   # noqa: E402
from nf4_grouped import build_group_tiles_device                                     # noqa: E402

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
    elif kind == "one":                        # one expert's rows span every chunk: the carry's whole job
        ids = torch.full((R,), E // 2, dtype=torch.int64)
    else:
        raise ValueError(kind)
    return ids.to(DEV)


@pytest.mark.parametrize("R", [1, 17, 257, 1024])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "one"])
@pytest.mark.parametrize("E,BM", [(128, 16), (40, 16), (256, 16)])
@pytest.mark.parametrize("rchunk", [None, 16, 64])
def test_chunked_rank_order_is_stable_argsort_and_tables_match_the_chained_builder(R, kind, E, BM, rchunk):
    ids = _ids(R, E, kind, seed=R * 7 + E)
    got = build_group_tiles_fused(ids.to(torch.int32), E, BM, rank="cumsum", rchunk=rchunk)
    assert torch.equal(got[3], torch.argsort(ids, stable=True)), "order is not the stable argsort"
    ref = build_group_tiles_device(ids.to(torch.int32), E, BM)
    for n, x, y in zip(NAMES, ref, got):
        assert x.dtype == y.dtype, (n, x.dtype, y.dtype)
        assert torch.equal(x, y), (n, R, kind, E, rchunk)


@pytest.mark.parametrize("R", [257, 512, 1024])
def test_chunked_and_one_piece_cumsum_give_the_same_integers(R):
    ids = _ids(R, 128, "skewed", seed=R).to(torch.int32)
    a = build_group_tiles_fused(ids, 128, 16, rank="cumsum", rchunk=0)
    b = build_group_tiles_fused(ids, 128, 16, rank="cumsum")
    for n, x, y in zip(NAMES, a, b):
        assert torch.equal(x, y), n


@pytest.mark.parametrize("R", [257, 512])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_chunked_lean_variant_gives_the_default_calls_integers(R, dtype):
    ids = _ids(R, 128, "skewed", seed=R).to(dtype)
    a = build_group_tiles_fused(ids, 128, 16, rank="cumsum")
    b = build_group_tiles_fused(ids, 128, 16, rank="cumsum", lean=True, sorted_ids=True)
    assert len(b) == 6
    for n, x, y in zip(NAMES, a, b):
        assert x.dtype == y.dtype and torch.equal(x, y), n
    assert b[5].dtype == dtype and torch.equal(b[5], ids.index_select(0, a[3]))


def test_the_automatic_chunk():
    assert CUMSUM_TILE_ELEMS == 8192
    assert _cumsum_rchunk(128, 512) == 64            # Qwen3-30B-A3B's 64-row step: eight 64-row chunks
    assert _cumsum_rchunk(40, 512) == 128            # Granite: EB 64
    assert _cumsum_rchunk(512, 1024) == 16           # never below 16 rows a chunk
    assert _cumsum_rchunk(8, 64) == 0                # a small table stays one piece
    assert _cumsum_rchunk(64, 128) == 0 and _cumsum_rchunk(64, 129) == 128


def test_bad_chunks_refuse():
    ids = torch.zeros(300, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="power of two"):
        build_group_tiles_fused(ids, 128, 16, rank="cumsum", rchunk=24)
    with pytest.raises(ValueError, match="power of two"):
        build_group_tiles_fused(ids, 128, 16, rank="cumsum", rchunk=8)
    with pytest.raises(ValueError, match="rchunk="):
        build_group_tiles_fused(ids[:200], 128, 16, rchunk=64)          # the pairwise rank has no chunks
