# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The hot_rows floor (``serve_paged.hot_rows_floor``) tied to the REAL failure: the hybrid tier built on CPU through
``enable_hybrid_tier`` (the tier's landing buffer unpinned, which needs no driver, and the fused NF4 kernel stubbed,
since what is measured is the request the tier sees, not the math). A forward of F tokens that routes every NVMe
expert of the binding layer passes at the floor and raises nvme_residency's own error one row below it, random
routings at F never ask for more than the floor, and VRAM-hot and DRAM experts never reach the tier.
"""
import random

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series")
nvme_residency = pytest.importorskip("nvme_residency")
pytest.importorskip("cpu_grouped")
pytest.importorskip("bitsandbytes", reason="Experts4bit quantization needs bnb")

from test_hybrid import E, H, K, _manifest, _module, _Wrap, arena  # noqa: E402,F401  (arena: the fixture)

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402
from experts4bit_qlora.engines.hybrid import disable_hybrid_tier, enable_hybrid_tier  # noqa: E402


def _man(n_layers, per_layer):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for lay in range(n_layers):
        v, d, n = per_layer.get(lay, ((), (), ()))
        for t, ids in (("vram", v), ("dram", d), ("nvme", n)):
            tiers[t] += [[lay, e] for e in ids]
    return {"tiers": tiers}


@pytest.fixture
def cold_path(monkeypatch):
    """The real hybrid tier on CPU: ColdTier with an unpinned landing buffer (pinned host memory needs a driver),
    recording every ensure() request's expert ids; the fused NF4 kernel stubbed after the gather."""
    requests = []
    real = nvme_residency.ColdTier

    class CpuTier(real):
        def __init__(self, *a, **k):
            k["pinned"] = False
            super().__init__(*a, **k)

        def ensure(self, layer, experts, *a, **k):
            requests.append(sorted(int(e) for e in experts))
            return super().ensure(layer, experts, *a, **k)

    monkeypatch.setattr(nvme_residency, "ColdTier", CpuTier)
    monkeypatch.setattr(hr, "_fused_over_stack", lambda xr, *a, **k: torch.zeros_like(xr))
    return requests


def _forward(model, ids):
    T = ids.shape[0]
    hidden = torch.randn(T, H, dtype=torch.bfloat16) * 0.3
    with torch.no_grad():
        return model.experts(hidden, ids, torch.rand(T, K, dtype=torch.bfloat16))


def _served(path, split, hot_rows, requests, **kw):
    """The tier built at ``hot_rows``. The build's own bulk reads (the VRAM and DRAM stacks, chunked to hot_rows) go
    through ensure() too; they are not a forward's request, so the record starts after the build."""
    model = _Wrap(_module())
    assert enable_hybrid_tier(model, path, _manifest(*split), hot_rows=hot_rows, device="cpu", verbose=False, **kw) == 1
    requests.clear()
    return model


def _floor(split, chunk, max_seqs):
    v, d, n = split
    return sp.hot_rows_floor(_man(1, {0: (v, d, n)}), 1, E, K, chunk, max_seqs)["floor"]


def test_the_layer_count_floor_passes_and_one_row_below_raises_in_the_forward(arena, cold_path):
    """F = 3 tokens x top-2 = 6 rows, and the layer has 6 NVMe experts: the floor is 6. A forward of F tokens that
    routes all six asks the tier for exactly 6 rows; at hot_rows 6 it runs, at 5 nvme_residency raises."""
    _, path, _ = arena
    split = ([0], [1], [2, 3, 4, 5, 6, 7])
    floor = _floor(split, chunk=3, max_seqs=1)
    assert floor == 6
    every = torch.tensor([[2, 3], [4, 5], [6, 7]])
    model = _served(path, split, floor, cold_path)
    try:
        _forward(model, every)
        assert max(len(r) for r in cold_path) == floor, cold_path
    finally:
        disable_hybrid_tier(model)
    model = _served(path, split, floor - 1, cold_path)
    try:
        with pytest.raises(ValueError, match=f"exceeds hot_rows={floor - 1}"):
            _forward(model, every)
    finally:
        disable_hybrid_tier(model)


def test_the_f_times_k_floor_passes_and_one_row_below_raises_in_the_forward(arena, cold_path):
    """F = 2 tokens x top-2 = 4 rows, below the layer's 7 NVMe experts: the floor is 4. Random routings at F never ask
    for more than 4 rows at hot_rows 4; four distinct NVMe experts in one forward raise at 3."""
    _, path, _ = arena
    split = ([0], [], [1, 2, 3, 4, 5, 6, 7])
    floor = _floor(split, chunk=2, max_seqs=1)
    assert floor == 4
    model = _served(path, split, floor, cold_path)
    try:
        rng = random.Random(20261024)
        for _ in range(30):
            _forward(model, torch.tensor([rng.sample(range(E), K) for _ in range(2)]))
        assert cold_path and max(len(r) for r in cold_path) <= floor, cold_path
    finally:
        disable_hybrid_tier(model)
    model = _served(path, split, floor - 1, cold_path)
    try:
        with pytest.raises(ValueError, match=f"exceeds hot_rows={floor - 1}"):
            _forward(model, torch.tensor([[1, 2], [3, 4]]))
    finally:
        disable_hybrid_tier(model)


@pytest.mark.parametrize("offload_rows", [None, 1], ids=["dram-in-place", "dram-moved-to-the-device"])
def test_dram_and_vram_experts_never_reach_the_cold_tier(arena, cold_path, offload_rows):
    """The premise the floor stands on: with most experts in DRAM, random routings at F (4 tokens x top-2 = 8 rows)
    ask the tier only for NVMe ids, never more than the floor (the layer's 3 NVMe experts) at hot_rows 3. Both DRAM
    routes: computed in place (the default) and moved to the device from the DRAM stacks (offload_rows 1 takes that
    route on every step)."""
    _, path, _ = arena
    split = ([0], [1, 2, 3, 4], [5, 6, 7])
    floor = _floor(split, chunk=4, max_seqs=1)
    assert floor == 3
    model = _served(path, split, floor, cold_path, offload_rows=offload_rows)
    try:
        rng = random.Random(7)
        for _ in range(40):
            _forward(model, torch.tensor([rng.sample(range(E), K) for _ in range(4)]))
        asked = {e for r in cold_path for e in r}
        assert asked and asked <= {5, 6, 7}, asked
        assert max(len(r) for r in cold_path) <= floor
        st = model.experts._hot_residency
        assert (st.offload_steps > 0) == (offload_rows is not None), st.offload_steps     # the route under test ran
    finally:
        disable_hybrid_tier(model)


# ------------------------------------------------------------------------------- build_engine itself, on CPU --
class _Reached(Exception):
    """enable_hybrid_tier was called: the build got past the floor."""


CALIB = {"cpu_bench": {"scatter_best": {"gbs": 200.0}}, "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1600.0}]}}


def _build(monkeypatch, tmp_path, path, **cfg_kw):
    """build_engine with the model load and tokenizer stubbed to the tiny MoE (no hub, no weights), the REAL solver
    on the tiny arena, and enable_hybrid_tier replaced by a marker: returns the placement it was handed, or raises
    the build's own refusal."""
    import json
    import types

    import transformers

    import experts4bit_qlora
    from experts4bit_qlora.engines import hybrid as hy

    model = _Wrap(_module())
    model.config = types.SimpleNamespace(num_experts_per_tok=K, model_type=None)
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps(CALIB))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: (_ for _ in ()).throw(OSError("stub")))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(experts4bit_qlora, "load_moe_4bit_streaming", lambda *a, **k: (model, None), raising=False)
    seen = {}

    def reached(m, arena, man, **k):
        seen.update(man=man, hot_rows=k.get("hot_rows"))
        raise _Reached
    monkeypatch.setattr(hy, "enable_hybrid_tier", reached)
    cfg = sp.PagedServeConfig(model="tiny/moe", arena=path, calib=str(calib), device="cpu", **cfg_kw)
    with pytest.raises(_Reached):
        sp.build_engine(cfg)
    return seen


def _solver_budget(path):
    """vram and dram budgets (GB) for ONE expert each, so the solver leaves the other six of eight on NVMe."""
    one = sp._bytes_per_expert(path) / 2**30
    return {"vram_gb": one * 1.01, "dram_gb": one * 1.01}


def test_build_engine_refuses_below_the_floor_by_name_before_the_tier(arena, monkeypatch, tmp_path):
    """At solver with one forward of 3 tokens x top-2 and six NVMe experts, the floor is 6: hot_rows 5 refuses at
    build, naming the knob, before enable_hybrid_tier; hot_rows 6 reaches the tier. all-vram reaches it at any
    hot_rows, the default included."""
    _, path, _ = arena
    kw = dict(placement="solver", chunk_tokens=3, max_seqs=1, max_seqs_requested="1", **_solver_budget(path))
    seen = _build(monkeypatch, tmp_path, path, hot_rows=6, **kw)
    assert len(seen["man"]["tiers"]["nvme"]) == 6 and seen["hot_rows"] == 6, seen
    with pytest.raises(ValueError, match=r"E4B_PAGED_HOT_ROWS=5 is below its floor 6 at this placement") as ei:
        _build(monkeypatch, tmp_path, path, hot_rows=5, **kw)
    assert "max_seqs 1" in str(ei.value) and "E4B_PAGED_CHUNK_TOKENS 3" in str(ei.value)
    for hot in (1, 64):
        seen = _build(monkeypatch, tmp_path, path, hot_rows=hot, placement="all-vram", chunk_tokens=3, max_seqs=1,
                      max_seqs_requested="1")
        assert seen["man"]["tiers"]["nvme"] == [] and seen["hot_rows"] == hot
