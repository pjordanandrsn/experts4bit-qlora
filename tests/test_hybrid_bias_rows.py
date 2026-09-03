# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""gpt-oss per-expert biases: the row gather moves the index to the
bias's device before index_select (P24-GEN-A: every gpt-oss arm died on
a CPU index against a CUDA bias)."""
import torch

from experts4bit_qlora.engines.hybrid import _bias_rows


class _BiasOnDevice:
    """A stand-in bias that claims a device and records where the index
    arrived; the gather itself runs on the CPU copy with the original
    ids (a meta index cannot be copied back)."""

    def __init__(self, t, device, ids_cpu):
        self.t, self.device, self.ids_cpu, self.seen = t, device, ids_cpu, None

    def detach(self):
        return self

    def index_select(self, dim, ids):
        self.seen = ids.device
        return self.t.index_select(dim, self.ids_cpu)


def test_index_follows_the_bias_device():
    """Without a GPU, 'meta' stands in for the bias's device: the index
    handed to index_select must arrive on it, not on the CPU."""
    t = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    ids = torch.tensor([3, 1])
    fake = _BiasOnDevice(t, torch.device("meta"), ids)
    out = _bias_rows(fake, ids)
    assert fake.seen == torch.device("meta")
    assert out.dtype == torch.float32 and out.device.type == "cpu"
    assert torch.equal(out, t[[3, 1]].float())


def test_rows_and_layout():
    t = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    out = _bias_rows(t, torch.tensor([3, 1]))
    assert out.is_contiguous() and out.dtype == torch.float32
    assert torch.equal(out, t[[3, 1]].float())
