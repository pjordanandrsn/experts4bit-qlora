# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-phase pool engagement (core-sweep receipts): the runner must flip
tier threads around prefill and RESTORE the enable-time value for
decode — a stuck value would silently run decode at the prefill count,
which measured as a 2.7x DRAM regression."""
from types import SimpleNamespace

import torch

from experts4bit_qlora.engines.paged_runner import PagedModelRunner


class _Tier:
    def __init__(self, threads):
        self._threads = threads
        self.gpu_only_calls = []

    def prefill_gpu_only(self, on):
        self.gpu_only_calls.append(bool(on))


def _runner(prefill_threads):
    model = torch.nn.Module()
    child = torch.nn.Module()
    child._hot_residency = _Tier(threads=32)
    model.add_module("m", child)
    kv = SimpleNamespace(L=2, reset=lambda s: None)
    return PagedModelRunner(model, kv, device="cpu",
                            prefill_threads=prefill_threads), \
        child._hot_residency


def test_mode_flips_threads_and_restores():
    r, tier = _runner(prefill_threads=64)
    assert tier._threads == 32
    r._mode(True)
    assert tier._threads == 64 and tier.gpu_only_calls[-1] is True
    r._mode(False)
    assert tier._threads == 32 and tier.gpu_only_calls[-1] is False


def test_none_leaves_threads_untouched():
    r, tier = _runner(prefill_threads=None)
    r._mode(True)
    assert tier._threads == 32
    r._mode(False)
    assert tier._threads == 32
