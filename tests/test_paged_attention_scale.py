# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The family's attention scale reaches the paged decode kernel.

GraniteMoe (attention_multiplier 0.015625) and Gemma-4 (1.0, folded
into q_norm) served word salad through the paged decode loop because the
decode and verify branches passed only q/k/v/slots and the kernel fell
back to head_dim**-0.5 (receipts INT4B16/P24-GEN-B)."""
import pytest
import torch

pa = pytest.importorskip("experts4bit_qlora.engines.paged_attention")


class _KV:
    """Records what the shim hands the kernel wrapper."""

    def __init__(self):
        self.calls = []
        # verify mode reads the slot's current length to stagger rows
        self.seq_lens = torch.zeros(1, 8, dtype=torch.int32)

    def append(self, layer, slot, k, v):
        self.seq_lens[layer, slot] += k.shape[0]

    def append_many(self, layer, slots, k, v):
        pass

    def attention(self, layer, q, slots=None, lens_override=None, **kw):
        self.calls.append(dict(kw, rows=q.shape[0]))
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=q.dtype)


class _M:
    layer_idx = 0


@pytest.mark.parametrize("scaling", [0.015625, 1.0, 0.125])
def test_decode_passes_the_module_scale(scaling):
    kv = _KV()
    ctx = pa.PagedAttentionContext(kv=kv, slots=[0], mode="decode")
    prev = pa.set_context(ctx)
    try:
        q = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
        pa.paged_attention_forward(_M(), q, q, q, None, scaling=scaling)
    finally:
        pa.set_context(prev)
    assert kv.calls and kv.calls[-1]["sm_scale"] == scaling


def test_verify_passes_the_module_scale():
    kv = _KV()
    ctx = pa.PagedAttentionContext(kv=kv, slots=[3], mode="verify")
    prev = pa.set_context(ctx)
    try:
        q = torch.zeros(1, 4, 3, 8, dtype=torch.bfloat16)      # T = K+1 rows
        pa.paged_attention_forward(_M(), q, q, q, None, scaling=0.015625)
    finally:
        pa.set_context(prev)
    assert kv.calls and kv.calls[-1]["sm_scale"] == 0.015625
    assert kv.calls[-1]["rows"] == 3
