# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_PAGED_PREFILL_ATTN (e4b#960): how a prefill chunk attends on a layer without sinks or a sliding window.

``math`` (default) passes SDPA the explicit lower-right causal boolean mask -- with GQA that call lands on SDPA's
math backend (fp32 on CUDA). ``flash`` passes the same mask as ``causal_lower_right(T, t_total)``, which the flash
kernel takes. Pinned here on CPU, where SDPA serves the bias itself:

- the knob's default, values and refusal;
- ``flash`` across chunk boundaries equals the whole-sequence reference, as ``math`` does (the chunked-prefill
  property), and the two routes agree on every chunk;
- ``flash`` hands SDPA a ``CausalBias`` with ``enable_gqa`` on a plain layer, and never on a layer with a sliding
  window or sinks, which keep the explicit mask.

The kernel the bias reaches on a GPU is tests/test_paged_prefill_attn_route_gpu.py's.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from experts4bit_qlora.engines import paged_attention as pa  # noqa: E402

H_Q, H_KV, D = 8, 2, 64


class _Mod(torch.nn.Module):
    def __init__(self, layer_idx=0, sliding_window=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = H_Q // H_KV
        self.is_causal = True
        self.sliding_window = sliding_window


def _qkv(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, H_Q, T, D, generator=g), torch.randn(1, H_KV, T, D, generator=g),
            torch.randn(1, H_KV, T, D, generator=g))


def _reference(q, k, v):
    T = q.shape[2]
    mask = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=D ** -0.5, enable_gqa=True)
    return o.transpose(1, 2).contiguous()


def _chunked(q, k, v, cuts, mod=None, **kw):
    ctx = pa.PagedAttentionContext(kv=None, slots=[0], mode="prefill")
    pa.set_context(ctx)
    outs, lo = [], 0
    try:
        for hi in list(cuts) + [q.shape[2]]:
            o, _ = pa.paged_attention_forward(mod or _Mod(), q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], None,
                                              scaling=D ** -0.5, **kw)
            outs.append(o)
            lo = hi
    finally:
        pa.set_context(None)
    return torch.cat(outs, dim=1)


def test_the_knob_defaults_to_math_and_refuses_unknown_values(monkeypatch):
    monkeypatch.delenv("E4B_PAGED_PREFILL_ATTN", raising=False)
    assert pa._prefill_attn_mode_env() == "math"
    for v, want in (("math", "math"), (" Flash ", "flash"), ("", "math")):
        monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", v)
        assert pa._prefill_attn_mode_env() == want
    assert pa.PREFILL_ATTN_ROUTES == ("math", "flash")
    for bad in ("1", "sdpa", "efficient"):
        monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", bad)
        with pytest.raises(ValueError, match="E4B_PAGED_PREFILL_ATTN"):
            pa._prefill_attn_mode_env()


@pytest.mark.parametrize("cuts", [(), (5,), (3, 7, 11)])
def test_both_routes_match_the_whole_sequence_across_chunk_boundaries(monkeypatch, cuts):
    q, k, v = _qkv(13, seed=len(cuts))
    want = _reference(q, k, v)
    got = {}
    for route in ("math", "flash"):
        monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", route)
        got[route] = _chunked(q, k, v, cuts)
        torch.testing.assert_close(got[route], want, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(got["flash"], got["math"], rtol=2e-5, atol=2e-5)


def _recording_sdpa(monkeypatch):
    seen = []
    real = torch.nn.functional.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, **kw):
        seen.append((type(attn_mask).__name__, kw.get("enable_gqa")))
        # a CausalBias dispatches only when called through the real function's own attribute: restore it for the call
        torch.nn.functional.scaled_dot_product_attention = real
        try:
            return real(q, k, v, attn_mask=attn_mask, **kw)
        finally:
            torch.nn.functional.scaled_dot_product_attention = sdpa
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    return seen


def test_flash_hands_sdpa_a_lower_right_bias_on_a_plain_layer(monkeypatch):
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", "flash")
    seen = _recording_sdpa(monkeypatch)
    q, k, v = _qkv(9, seed=4)
    _chunked(q, k, v, (4,))
    assert seen == [("CausalBias", True), ("CausalBias", True)], seen
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", "math")
    seen.clear()
    _chunked(q, k, v, (4,))
    assert seen == [("Tensor", True), ("Tensor", True)], seen


def test_windowed_and_sink_layers_keep_the_explicit_mask(monkeypatch):
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", "flash")
    seen = _recording_sdpa(monkeypatch)
    q, k, v = _qkv(9, seed=5)
    _chunked(q, k, v, (4,), mod=_Mod(sliding_window=3))
    assert [m for m, _g in seen] == ["Tensor", "Tensor"], seen
    seen.clear()
    sinks = torch.zeros(H_Q)
    out = _chunked(q, k, v, (4,), s_aux=sinks)          # the sink path computes its own scores: no SDPA call
    assert seen == [] and out.shape == (1, 9, H_Q, D)
