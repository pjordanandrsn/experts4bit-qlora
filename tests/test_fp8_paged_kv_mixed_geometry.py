# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""One paged KV cache serving layers of different (kv heads, head_dim):
Gemma-4 runs sliding layers at 256/8 and full layers at 512/2. Rows are
sized for the largest layer; each layer writes and reads its own
geometry inside that stride. CPU-runnable (no kernel launch)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GEOM = [(4, 32), (2, 64), (4, 32)]          # (kv heads, head_dim) per layer


def _kv(batch=2, k_groups=1):
    return Fp8PagedKV(3, [h for h, _ in GEOM], [d for _, d in GEOM],
                      batch=batch, max_tokens_per_seq=64, k_groups=k_groups,
                      device=DEV)


def test_geometry_is_per_layer_and_rows_take_the_largest():
    kv = _kv()
    assert kv.Hs == [4, 2, 4] and kv.Ds == [32, 64, 32]
    assert kv.H == 4 and kv.D == 64                       # the largest of each
    assert kv.k_row == max(kv.k_rows) and kv.v_row == max(kv.v_rows)
    # layer 1 (2 x 64) and layer 0 (4 x 32) have the same payload bytes;
    # the pool row is the max natural row
    assert kv._k_pays == [16 * 4 * 32, 16 * 2 * 64, 16 * 4 * 32]
    assert kv.kp.dev[0].numel() % kv.k_row == 0


def test_int_geometry_broadcasts_as_before():
    kv = Fp8PagedKV(2, 4, 32, batch=1, max_tokens_per_seq=32, device=DEV)
    assert kv.Hs == [4, 4] and kv.Ds == [32, 32] and kv.k_rows == [kv.k_row] * 2


def test_wrong_geometry_for_a_layer_is_refused():
    kv = _kv()
    with pytest.raises(ValueError, match="layer 1"):
        kv.append(1, 0, torch.zeros(3, 4, 32), torch.zeros(3, 4, 32))


@pytest.mark.parametrize("k_groups", [1, 2])
def test_each_layer_round_trips_its_own_geometry(k_groups):
    torch.manual_seed(5)
    kv = _kv(k_groups=k_groups)
    truth = {}
    for layer, (h, d) in enumerate(GEOM):
        for seq in range(2):
            k = torch.randn(21, h, d, dtype=torch.bfloat16, device=DEV) * 0.5
            v = torch.randn(21, h, d, dtype=torch.bfloat16, device=DEV) * 0.5
            kv.append(layer, seq, k, v)                     # crosses a block
            truth[(layer, seq)] = (k, v)
    for (layer, seq), (k, v) in truth.items():
        kk, vv = kv.reference_kv(layer, seq)
        assert kk.shape == k.shape and vv.shape == v.shape
        torch.testing.assert_close(kk.float(), k.float(), rtol=0.13, atol=0.05)
        torch.testing.assert_close(vv.float(), v.float(), rtol=0.13, atol=0.05)
    assert int(kv.seq_lens[1, 0]) == 21


def test_append_many_and_kernel_args_per_layer():
    torch.manual_seed(6)
    kv = _kv()
    for layer, (h, d) in enumerate(GEOM):
        k = torch.randn(2, 5, h, d, dtype=torch.bfloat16, device=DEV)
        v = torch.randn(2, 5, h, d, dtype=torch.bfloat16, device=DEV)
        kv.append_many(layer, [0, 1], k, v)
        kf, vf, tbl, lens = kv.kernel_args(layer, [0, 1])
        assert kf.numel() % kv.k_row == 0 and vf.numel() % kv.v_row == 0
        assert lens.tolist() == [5, 5]
        kk, _ = kv.reference_kv(layer, 1)
        torch.testing.assert_close(kk.float(), k[1].float(), rtol=0.13, atol=0.05)
