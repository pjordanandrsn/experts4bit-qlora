"""DQ9 instrumentation/restoration checks; CPU fixtures are never measured GPU evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("dq9_phase_fixture", Path(__file__).parents[1]/"bench/dq9/dq9_phase.py")
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


class Counters:
    def __init__(self):
        self.reserved, self.empty_calls = 2048, 0

    def memory_allocated(self):
        return 1024

    def memory_reserved(self):
        return self.reserved

    def max_memory_allocated(self):
        return 1536

    def max_memory_reserved(self):
        return 2048

    def empty_cache(self):
        self.empty_calls += 1
        self.reserved = 1024

    def synchronize(self):
        raise AssertionError("instrument added CUDA sync")

    def reset_peak_memory_stats(self):
        raise AssertionError("instrument reset the existing interval")


@pytest.mark.parametrize("mode", ["baseline", "empty"])
def test_instrument_observes_real_cpu_optimizer_and_restores_on_success_or_failure(mode):
    torch.manual_seed(731)
    model = torch.nn.Linear(4, 4)
    counters = Counters()
    api = SimpleNamespace(cuda=counters, optim=torch.optim, autograd=torch.autograd, nn=torch.nn)
    loader = SimpleNamespace(load_base=lambda: model)
    trainer = SimpleNamespace(prepare=lambda: SimpleNamespace(model=loader.load_base()))
    original_forward, original_init = model.forward, torch.optim.AdamW.__init__
    original_backward, original_step = torch.autograd.backward, torch.optim.AdamW.step
    census = phase.PhaseCensus(api, mode)
    with census.installed(loader, trainer):
        prepared = trainer.prepare()
        opt = torch.optim.AdamW(prepared.model.parameters(), lr=2e-4)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            prepared.model(torch.ones(1, 4)).sum().backward()
            torch.nn.utils.clip_grad_norm_(prepared.model.parameters(), 1.0)
            opt.step()
    assert model.forward == original_forward
    assert torch.optim.AdamW.__init__ is original_init and torch.optim.AdamW.step is original_step
    assert torch.autograd.backward is original_backward
    report = census.report()
    assert counters.empty_calls == report["cache_calls"] == int(mode == "empty")
    names = [r["phase"] for r in report["rows"]]
    for name in ("loader-done", "setup-done", "cache-boundary-done", "optimizer-created"):
        assert names.count(name) == 1
    for name in ("forward-loss-done", "backward-done", "clip-done", "optimizer-step-done"):
        assert names.count(name) == 2
    assert census.step == 2 and report["new_sync_calls"] == 0
    assert report["setup_storages"]
    assert "forward" not in vars(model)
    assert len(report["optimizer_storages"]) == 2
    assert all(x["storages"] for x in report["optimizer_storages"])
    with pytest.raises(RuntimeError), census.installed(loader, trainer):
        raise RuntimeError("injected failure")
    assert model.forward == original_forward and torch.autograd.backward is original_backward


def test_inventory_deduplicates_tied_storage_and_retains_no_tensor_objects():
    model = torch.nn.Module()
    model.a = torch.nn.Linear(4, 4, bias=False)
    model.b = torch.nn.Linear(4, 4, bias=False)
    model.b.weight = model.a.weight
    rows = phase.inventory(model)
    assert sum(x["storage_bytes"] for x in rows) == 64
    assert len(rows) == 1 and rows[0]["dtype"] == "torch.float32"
    assert not any(isinstance(value, torch.Tensor) for row in rows for value in row.values())


def test_unregistered_cache_mode_is_refused():
    with pytest.raises(ValueError, match="unregistered"):
        phase.PhaseCensus(torch, "adaptive")
