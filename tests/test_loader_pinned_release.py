"""The streaming loader returns the caching host allocator's free pinned blocks to CUDA once the load is done.

safetensors reads every tensor opened with ``device="cuda"`` through a pinned block that the allocator rounds up to a power
of two and keeps cached, so a load used to leave one cached block per size class of its dense tensors behind (2.22 GB on
Qwen3-235B, 0.69 GB on Mixtral-8x22B, measured). These tests pin the release without a GPU."""
import ast
import inspect

import torch

from experts4bit_qlora import loader


class _Allocator:
    def __init__(self, owned, live):
        self.owned, self.live, self.calls = owned, live, []

    def stats(self):
        return {"allocated_bytes.current": self.owned}

    def empty(self):
        self.calls.append("empty")
        self.owned = self.live                # cached blocks go back to CUDA; blocks a tensor holds stay


def _patch(monkeypatch, alloc, cuda=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: alloc.calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "host_memory_stats", alloc.stats)
    monkeypatch.setattr(torch._C, "_host_emptyCache", alloc.empty, raising=False)


def test_a_cuda_load_releases_the_cached_blocks_and_reports_them(monkeypatch):
    alloc = _Allocator(owned=5_843_722_497, live=3_623_878_656)      # the Qwen3-235B 2-layer record at the load end
    _patch(monkeypatch, alloc)
    assert loader._release_cached_pinned_blocks("cuda") == 2_219_843_841
    assert alloc.calls == ["sync", "empty"]                            # synchronised first: an in-flight block is not free


def test_a_cpu_load_touches_nothing(monkeypatch):
    alloc = _Allocator(owned=1 << 30, live=0)
    _patch(monkeypatch, alloc)
    assert loader._release_cached_pinned_blocks("cpu") is None
    assert alloc.calls == []


def test_no_cuda_or_no_private_api_is_a_no_op(monkeypatch):
    alloc = _Allocator(owned=1 << 30, live=0)
    _patch(monkeypatch, alloc, cuda=False)
    assert loader._release_cached_pinned_blocks("cuda") is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delattr(torch._C, "_host_emptyCache")
    assert loader._release_cached_pinned_blocks("cuda") is None
    assert alloc.calls == []


def test_the_streaming_loader_releases_just_before_it_returns():
    fn = next(n for n in ast.parse(inspect.getsource(loader)).body
              if isinstance(n, ast.FunctionDef) and n.name == "load_moe_4bit_streaming")
    last, before = fn.body[-1], fn.body[-2]
    assert isinstance(last, ast.Return)
    assert isinstance(before, ast.Expr) and isinstance(before.value, ast.Call)
    assert getattr(before.value.func, "id", None) == "_release_cached_pinned_blocks"
