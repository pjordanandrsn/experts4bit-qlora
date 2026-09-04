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


def test_harness_reads_per_layer_geometry():
    """The harness sizes the cache per layer on a heterogeneous config and
    with scalars on a uniform one."""
    import importlib.util
    import os
    import sys
    from types import SimpleNamespace as N
    spec = importlib.util.spec_from_file_location(
        "step_decomp", os.path.join(os.path.dirname(__file__), "..", "bench", "hybrid-g9", "step_decomp.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step_decomp"] = mod
    spec.loader.exec_module(mod)
    assert mod._kv_geometry(N(num_key_value_heads=8, head_dim=128, hidden_size=2048, num_attention_heads=32)) == (8, 128)

    class Hetero:
        per_layer_config = [N(num_key_value_heads=8, head_dim=256, hidden_size=2816, num_attention_heads=16),
                            N(num_key_value_heads=2, head_dim=512, hidden_size=2816, num_attention_heads=16)]

        def __getattr__(self, k):
            raise RuntimeError(f"'{k}' is a per-layer attribute and may vary across layers")
    assert mod._kv_geometry(Hetero()) == ([8, 2], [256, 512])


def test_auto_key_groups_keep_32_wide_scales_per_layer(monkeypatch):
    """k_groups=None sizes each layer's key scale groups to 32-wide (4 at
    head_dim 128, 8 at 256, 16 at 512) when the installed kernel unrolls
    that many, and falls back to 4 when it refuses (capability probe,
    never a version string). Gemma-4's 512-dim layers measured 0.046 nats
    of fp8 cost with 128-wide groups and 0.017 with 32-wide (e4b#359)."""
    import experts4bit_qlora.engines.fp8_paged_kv as mod
    calls = []
    def fake_probe(q, head_dim, k_groups, v_groups, **kw):
        calls.append((head_dim, k_groups))
        return None                       # a kernel that unrolls the count
    import sys
    import types
    fake = types.SimpleNamespace(fp8_compute_unsupported=fake_probe)
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", fake)
    kv = mod.Fp8PagedKV(3, [8, 8, 2], [128, 256, 512], batch=1,
                        max_tokens_per_seq=32, device="cpu")
    assert kv.kgs == [4, 8, 16] and kv.k_groups is None
    assert calls == [(256, 8), (512, 16)], "128 needs no probe: 4 groups is the old default"
    # per-layer row sizing carries the extra scale bytes of the finer layers
    assert kv.k_rows[1] - (16 * 8 * 256) == 16 * 8 * 8 * 4
    assert kv.k_rows[2] - (16 * 2 * 512) == 16 * 2 * 16 * 4
    # round trip through append + read-back on every layer: the finer
    # layers' scale bytes must land where the reader expects them
    for layer in range(3):
        H, D = kv.Hs[layer], kv.Ds[layer]
        k, v = torch.randn(5, H, D), torch.randn(5, H, D)
        kv.append(layer, 0, k, v)
        kk, vv = kv.reference_kv(layer, 0)
        assert kk.shape == (5, H, D) and vv.shape == (5, H, D)
        assert (kk.float() - k).abs().max() < 0.1 * k.abs().max(), layer
        assert (vv.float() - v).abs().max() < 0.1 * v.abs().max(), layer

    def refusing_probe(q, head_dim, k_groups, v_groups, **kw):
        return f"fp8 compute unrolls k_groups in (1, 2, 4), got {k_groups}"
    fake.fp8_compute_unsupported = refusing_probe
    kv_old = mod.Fp8PagedKV(3, [8, 8, 2], [128, 256, 512], batch=1,
                            max_tokens_per_seq=32, device="cpu")
    assert kv_old.kgs == [4, 4, 4] and kv_old.k_groups == 4

    monkeypatch.delitem(sys.modules, "fp8_paged_attn")
    monkeypatch.setattr("builtins.__import__", _import_blocker("fp8_paged_attn"))
    kv_none = mod.Fp8PagedKV(1, 2, 512, batch=1, max_tokens_per_seq=32, device="cpu")
    assert kv_none.kgs == [4], "no kernel installed: the old default"


def _import_blocker(name):
    real = __import__
    def blocked(n, *a, **k):
        if n == name:
            raise ImportError(name)
        return real(n, *a, **k)
    return blocked


def test_explicit_key_groups_still_broadcast():
    import experts4bit_qlora.engines.fp8_paged_kv as mod
    kv = mod.Fp8PagedKV(2, [8, 2], [256, 512], batch=1, max_tokens_per_seq=32,
                        k_groups=4, device="cpu")
    assert kv.kgs == [4, 4] and kv.k_groups == 4


def test_auto_key_groups_are_powers_of_two_and_floor_at_four():
    """32-wide scales at head_dim >= 128 (4 / 8 / 16 groups); below that the
    count floors at 4 -- the 2-group default at head_dim 64 was measured on
    one box, paired twice, and refused (no throughput on Granite or gpt-oss,
    +0.0136 / +0.108 nats). Counts are powers of two only: 96 // 32 = 3 would
    be refused by the kernel's unroll, 192 // 32 = 6 rounds to 4."""
    import experts4bit_qlora.engines.fp8_paged_kv as mod
    assert mod._auto_k_groups(64) == 4
    assert mod._auto_k_groups(32) == 4
    assert mod._auto_k_groups(128) == 4
    assert mod._auto_k_groups(96) == 4
    assert mod._auto_k_groups(192) == 4
    assert mod._auto_k_groups(80) == 4
    kv = mod.Fp8PagedKV(2, 8, 64, batch=1, max_tokens_per_seq=32, device="cpu")
    assert kv.kgs == [4, 4] and kv.k_groups == 4
    # round trip on the new default
    k, v = torch.randn(5, 8, 64), torch.randn(5, 8, 64)
    kv.append(0, 0, k, v)
    kk, vv = kv.reference_kv(0, 0)
    assert (kk.float() - k).abs().max() < 0.1 * k.abs().max()

