"""CI guard for Fp8PagedKV.append_batch: bit-identical to per-sequence
append across pool bytes, seen counters, seq_lens, and block tables.
This is PREREG-g9-kvappend's void-gate invariant, machine-checked."""
import pytest
import torch

pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402


def make(device="cpu", **kw):
    return Fp8PagedKV(3, 2, 64, batch=4, max_tokens_per_seq=48,
                      k_groups=4, device=device, **kw)


def drive(kv, steps, T=1, seed=3):
    g = torch.Generator().manual_seed(seed)
    seqs = list(range(4))
    for _ in range(steps):
        k = torch.randn(4, T, 2, 64, generator=g)
        v = torch.randn(4, T, 2, 64, generator=g)
        yield seqs, k, v


def pages_equal(a, b):
    assert a._seen == b._seen
    assert torch.equal(a.seq_lens, b.seq_lens)
    for layer in range(a.L):
        assert torch.equal(a.block_table[layer], b.block_table[layer])
    for key, rows in a._rows.items():
        assert b._rows[key] == rows
        layer, _seq = key
        for r in rows:
            for pa, pb in ((a.kp, b.kp), (a.vp, b.vp)):
                assert torch.equal(pa.row_view(layer, r),
                                   pb.row_view(layer, r))


@pytest.mark.parametrize("T", [1, 5])
def test_batched_equals_sequential(T):
    a = make()
    b = make()
    for seqs, k, v in drive(a, 6, T=T):
        for layer in range(a.L):
            for i, s in enumerate(seqs):
                a.append(layer, s, k[i], v[i])
    for seqs, k, v in drive(b, 6, T=T):
        for layer in range(b.L):
            b.append_many(layer, seqs, k, v)
    pages_equal(a, b)


def test_overflow_rejected_before_any_write():
    kv = make()
    seqs = list(range(4))
    k = torch.randn(4, 100, 2, 64)     # > max_tokens_per_seq
    before = {s: kv._seen[0][s] for s in seqs}
    with pytest.raises(ValueError):
        kv.append_many(0, seqs, k, k)
    assert {s: kv._seen[0][s] for s in seqs} == before


def test_shape_validation():
    kv = make()
    with pytest.raises(ValueError):
        kv.append_many(0, [0, 1], torch.randn(2, 1, 2, 32),
                        torch.randn(2, 1, 2, 32))
