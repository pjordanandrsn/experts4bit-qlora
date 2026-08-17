# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""FP8 KV cache oracle (Phase 7): growth across calls is exactly equal to
quantizing the whole sequence at once, the null control is a true
passthrough, and the positive control is genuinely destructive — the three
properties the G7 quality verdict rests on."""

import pytest
import torch

pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.fp8_kv_cache import Fp8KVCache  # noqa: E402

B, H, D = 1, 4, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _kv(t, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, H, t, D, generator=g, dtype=torch.float32)
    return x.to(DEV).to(torch.bfloat16)


@pytest.mark.parametrize("mode,kgroup", [
    ("fp8", None), ("off", None), ("int4", None), ("crush", None),
    ("fp8", 32), ("fp8", 16),          # grouped scales: tokens move to -2
])
def test_chunked_growth_equals_whole_sequence_quantization(mode, kgroup):
    """The scale is per (token, head), so splitting a sequence across
    update() calls cannot change any token's scale — chunked and
    whole-sequence storage must agree BIT for bit. This is what fails when
    the payload and the scale tensor are grown on different axes than
    their own token axes (payload -2, scales -1).

    EVERY mode is exercised: the first version of this test covered only
    fp8, and the integer path shipped a keepdim scale whose token axis sat
    one place over — a bug the fp8-only test could not see."""
    pieces = [_kv(5, 1), _kv(1, 2), _kv(11, 3), _kv(1, 4)]
    whole = torch.cat(pieces, dim=-2)

    chunked = Fp8KVCache(mode=mode, key_group=kgroup, value_group=kgroup)
    for p in pieces:
        k_out, v_out = chunked.update(p, p, layer_idx=0)

    single = Fp8KVCache(mode=mode, key_group=kgroup, value_group=kgroup)
    k_ref, v_ref = single.update(whole, whole, layer_idx=0)

    assert k_out.shape == k_ref.shape == whole.shape
    assert torch.equal(k_out, k_ref), "chunked growth changed the bytes"
    assert torch.equal(v_out, v_ref)


def test_heads_and_tokens_differ_so_an_axis_slip_cannot_hide():
    """Guard the guard: the shapes used here must make a wrong-axis
    concatenate fail loudly rather than coincidentally line up."""
    assert H != 5 and H != 11


def test_null_control_is_a_true_passthrough():
    x = _kv(7, 9)
    off = Fp8KVCache(mode="off")
    k, v = off.update(x, x, layer_idx=0)
    assert torch.equal(k, x), "mode='off' must not touch the values"
    assert torch.equal(v, x)


def test_positive_control_is_genuinely_destructive():
    """`crush` exists so the harness can prove it detects KV damage. If it
    ever stops being destructive, every quality verdict built on it is
    worthless — so the property is pinned here, not assumed there."""
    x = _kv(32, 11)
    fp8 = Fp8KVCache(mode="fp8").update(x, x, layer_idx=0)[0]
    crush = Fp8KVCache(mode="crush").update(x, x, layer_idx=0)[0]
    xf = x.float()
    err_fp8 = (fp8.float() - xf).norm() / xf.norm()
    err_crush = (crush.float() - xf).norm() / xf.norm()
    assert err_fp8 < 0.05, f"fp8 error above its own floor: {err_fp8}"
    assert err_crush > 6 * err_fp8, (
        f"positive control is not destructive enough to be a control: "
        f"crush={err_crush} vs fp8={err_fp8}")


def test_int4_sits_between_fp8_and_crush():
    x = _kv(32, 13)
    xf = x.float()

    def err(mode):
        out = Fp8KVCache(mode=mode).update(x, x, layer_idx=0)[0]
        return ((out.float() - xf).norm() / xf.norm()).item()

    e8, e4, ec = err("fp8"), err("int4"), err("crush")
    assert e8 < e4 < ec, f"format ordering violated: {e8} {e4} {ec}"


def test_per_layer_stores_are_independent():
    a, b = _kv(3, 21), _kv(3, 22)
    c = Fp8KVCache(mode="fp8")
    c.update(a, a, layer_idx=0)
    k1, _ = c.update(b, b, layer_idx=1)
    assert c.get_seq_length(0) == 3 and c.get_seq_length(1) == 3
    assert not torch.equal(k1, c.update(torch.zeros_like(b), b, 0)[0])


def test_bytes_per_token_counts_both_sides_and_the_scale_tail():
    n_layers, hkv = 24, 8
    off = Fp8KVCache(mode="off").bytes_per_token(hkv, D, n_layers)
    fp8 = Fp8KVCache(mode="fp8").bytes_per_token(hkv, D, n_layers)
    assert off == D * 2 * hkv * n_layers * 2          # K and V, bf16
    assert fp8 == (D + 4) * hkv * n_layers * 2        # payload + fp32 scale
    assert off / fp8 == pytest.approx(2 * D / (D + 4))


@pytest.mark.parametrize("mode", ["fp8", "int4", "crush"])
def test_scale_axis_convention_is_enforced_at_write(mode, monkeypatch):
    """The module's one axis convention — scale is payload.shape[:-1] — is
    checked where the slot is BUILT, so a future format that gets it wrong
    fails with that sentence instead of a broadcast error three frames
    away in the read path."""
    c = Fp8KVCache(mode=mode)
    x = _kv(4, 31)
    slot = c._store(x)
    assert slot[2].shape == slot[1].shape[:-1]

    import experts4bit_qlora.engines.fp8_kv_cache as m
    monkeypatch.setattr(m, "_quant_int",
                        lambda t, b: (t.float(),
                                      t.float().abs().amax(-1, keepdim=True)))
    if mode != "fp8":
        with pytest.raises(AssertionError, match="minus its last axis"):
            Fp8KVCache(mode=mode)._store(x)
