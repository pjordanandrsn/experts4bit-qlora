# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Sliding windows and attention sinks reach the paged kernel from the
model's own attributes (Gemma-4 keeps the window on the module, gpt-oss
passes window and sinks through the interface), and the prefill branch
applies the same window mask and the same sink softmax as transformers'
eager forward."""
import pytest
import torch

pa = pytest.importorskip("experts4bit_qlora.engines.paged_attention")


class _KV:
    def __init__(self):
        self.calls = []
        self.seq_lens = torch.zeros(1, 8, dtype=torch.int32)

    def append(self, layer, slot, k, v):
        self.seq_lens[layer, slot] += k.shape[0]

    def append_many(self, layer, slots, k, v):
        pass

    def attention(self, layer, q, slots=None, lens_override=None, **kw):
        self.calls.append(dict(kw))
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=q.dtype)


class _M:
    layer_idx = 0


class _Gemma(_M):
    sliding_window = 1024          # Gemma-4 sliding layer: on the module


class _Full(_M):
    sliding_window = None          # Gemma-4 full layer


def _decode(mod, **kw):
    kv = _KV()
    prev = pa.set_context(pa.PagedAttentionContext(kv=kv, slots=[0], mode="decode"))
    try:
        q = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
        pa.paged_attention_forward(mod, q, q, q, None, scaling=0.125, **kw)
    finally:
        pa.set_context(prev)
    return kv.calls[-1]


def test_gemma_window_comes_from_the_module():
    c = _decode(_Gemma())
    assert c["window"] == 1024 and c["sinks"] is None
    c = _decode(_Full())
    assert c["window"] == 0


def test_gptoss_window_and_sinks_come_from_kwargs():
    sinks = torch.randn(4)
    c = _decode(_M(), sliding_window=128, s_aux=sinks)
    assert c["window"] == 128 and torch.equal(c["sinks"], sinks)


def test_plain_families_pass_nothing():
    c = _decode(_M())
    assert c["window"] == 0 and c["sinks"] is None


def _eager(q, k, v, mask, scale, sinks=None):
    """transformers-style eager attention with GQA repeat, explicit mask,
    optional sink column."""
    g = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    att = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    att = att.masked_fill(~mask, float("-inf"))
    if sinks is not None:
        sk = sinks.float().view(1, -1, 1, 1).expand(att.shape[0], att.shape[1], att.shape[2], 1)
        att = torch.cat([att, sk], dim=-1)
        p = torch.softmax(att, dim=-1)[..., :-1]
    else:
        p = torch.softmax(att, dim=-1)
    return torch.matmul(p.to(v.dtype), v)


@pytest.mark.parametrize("window,sinks", [(0, False), (4, False), (0, True), (4, True)])
def test_prefill_matches_eager_with_window_and_sinks(window, sinks):
    torch.manual_seed(3)
    T, hq, hkv, d = 12, 4, 2, 8
    q = torch.randn(1, hq, T, d, dtype=torch.bfloat16)
    k = torch.randn(1, hkv, T, d, dtype=torch.bfloat16)
    v = torch.randn(1, hkv, T, d, dtype=torch.bfloat16)
    s_aux = torch.randn(hq) if sinks else None
    pos = torch.arange(T)
    keys = torch.arange(T)
    mask = keys[None, :] <= pos[:, None]
    if window:
        mask = mask & (keys[None, :] > pos[:, None] - window)
    want = _eager(q, k, v, mask[None, None], 0.125, s_aux)
    got = pa._prefill_attend(q, k, v, mask[None, None], 0.125, s_aux)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


def test_window_of_prefers_kwargs_then_module():
    assert pa._window_of(_Gemma(), {}) == 1024
    assert pa._window_of(_Gemma(), {"sliding_window": 64}) == 64
    assert pa._window_of(_Full(), {}) == 0
    assert pa._window_of(_M(), {"sliding_window": None}) == 0
