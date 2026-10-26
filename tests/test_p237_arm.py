"""P237's observing hook (bench/p237/p237_arm.py), on CPU fakes of the objects it wraps, plus the real tc1_arm loader.

- Every wrapper passes its call through unchanged: same arguments in, the wrapped function's own output out.
- Counts land in the interval of the last step_edge (``pre`` before edge 0), split by the kept model's ``training`` flag, so the
  drift guard's layer-major call and eval0's stagings never enter a step, and an eval between steps is never a training count.
- Unsloth: a fetch is counted only when the block was NOT resident and is after (block_swap.py:214-216 returns early otherwise),
  an eviction only when it was resident; blocks are named by the swap's own indices. The edge snapshot flags an evicted block
  whose frozen parameters are not 0-element, and a resident one holding a 0-element parameter.
- e4b: one count per ``_OffloadStats.record_copy`` call (a real staging), by policy, with its bytes and copies; layer_major_step
  calls per interval; gnf4's route per interval.
- The source launches no device work and sets no environment variable; tc1_arm's real ``load_arm_hook`` accepts the file by its
  sha256 and finds all four phases; ``uninstall`` restores every original.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "bench" / "p237" / "p237_arm.py"
ARM = REPO / "bench" / "tc1" / "tc1_arm.py"
sys.path.insert(0, str(HOOK.parent))
p237_arm = importlib.import_module("p237_arm")


# -- fakes ------------------------------------------------------------------------------------------------------------------------
class FakeBlock:
    """block_swap._Block's residency contract (unsloth_zoo 2026.10.8, block_swap.py:204-227): ``evict`` returns at once when not
    resident, else points every frozen parameter at a 0-element tensor; ``prefetch`` returns at once when resident, else points them
    back at the slot views. The return values are sentinels, so the tests can see the wrapper hand them back."""

    def __init__(self, shapes, nbytes):
        self.params = [torch.nn.Parameter(torch.zeros(s), requires_grad=False) for s in shapes]
        self._full = [p.data for p in self.params]
        self.resident = True
        self._n = nbytes

    def nbytes(self):
        return self._n

    def evict(self):
        if not self.resident:
            return "not-resident"
        for p in self.params:
            p.data = torch.empty(0)
        self.resident = False
        return ("freed", id(self))

    def prefetch(self, slot):
        if self.resident:
            return "already"
        for p, f in zip(self.params, self._full):
            p.data = f
        self.resident = True
        return ("fetched", slot)


class FakeCounter:
    """UnslothMoeCounter's surface the hook calls (pinned against the real class below)."""

    def __init__(self):
        self.marks, self.loaded, self.removed = [], None, False

    def install(self):
        return self

    def patch_loaded(self, experts):
        self.loaded = list(experts)
        return self

    def mark_step(self, i):
        self.marks.append(int(i))

    def snapshot(self):
        return {"passes_n": len(self.marks)}

    def structural(self, micro_batches, n_steps):
        return {"micro_batches": micro_batches, "steps": n_steps, "faults": []}

    def uninstall(self):
        self.removed = True


class FakeModel(torch.nn.Module):
    def __init__(self, swap=None):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        if swap is not None:
            self._unsloth_block_swap = swap
        self.config = types.SimpleNamespace(num_hidden_layers=4)


def _args(**kw):
    a = types.SimpleNamespace(framework="unsloth", accum=4, layer_major=0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def unsloth_env(monkeypatch):
    """A fake ``unsloth_zoo.block_swap`` whose ``_Block`` is FakeBlock, and a fake ``unsloth_moe_counter`` module."""
    pkg = types.ModuleType("unsloth_zoo")
    bs = types.ModuleType("unsloth_zoo.block_swap")
    cls = type("_Block", (FakeBlock,), {})
    bs._Block = cls
    pkg.block_swap = bs
    monkeypatch.setitem(sys.modules, "unsloth_zoo", pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_swap", bs)
    counters = []

    def make_counter():
        c = FakeCounter()
        counters.append(c)
        return c

    monkeypatch.setitem(sys.modules, "unsloth_moe_counter", types.SimpleNamespace(UnslothMoeCounter=make_counter))
    orig = (cls.prefetch, cls.evict)
    yield cls, counters
    assert (cls.prefetch, cls.evict) == orig            # every test uninstalls; a leak would show here


def _swap(cls, n=2, depth=2):
    blocks = [cls([(3, 2), (4,)], nbytes=100 * (k + 1)) for k in range(n)]
    return types.SimpleNamespace(indices=[1, 3][:n], blocks=blocks, depth=depth, pinned_bytes=50,
                                 host_bytes=lambda: sum(b.nbytes() for b in blocks))


# -- Unsloth ----------------------------------------------------------------------------------------------------------------------
def test_unsloth_fetch_and_evict_counted_per_interval_and_mode(unsloth_env):
    cls, counters = unsloth_env
    swap = _swap(cls)
    b0, b1 = swap.blocks
    h = p237_arm.make()
    h.post_import(_args())
    try:
        assert b0.evict() == ("freed", id(b0))           # the load's own eviction: interval pre, mode load
        assert b0.prefetch("slotA") == ("fetched", "slotA")
        assert b0.prefetch("slotB") == "already"          # resident: returns early, NOT a fetch
        model = FakeModel(swap)
        h.post_load(_args(), model)
        model.eval()
        b1.evict()
        b1.prefetch("s")                                  # eval0: pre / eval
        model.train()
        h.step_edge(0)
        for blk in (b0, b1):                              # two micro-batches of step 0
            for _ in range(2):
                assert blk.evict() == ("freed", id(blk))
                assert blk.evict() == "not-resident"      # not resident: NOT an eviction
                blk.prefetch("s")
        h.step_edge(1)
        rec = h.record()
    finally:
        h.uninstall()
    json.dumps(rec)
    assert rec["fetch"]["pre"] == {"load": {"1": [1, 100]}, "eval": {"3": [1, 200]}}, rec["fetch"]
    assert rec["evict"]["pre"] == {"load": {"1": 1}, "eval": {"3": 1}}
    assert rec["fetch"]["0"] == {"train": {"1": [2, 200], "3": [2, 400]}}
    assert rec["evict"]["0"] == {"train": {"1": 2, "3": 2}}
    assert "1" not in rec["fetch"] and rec["n_steps"] == 1
    assert counters[0].marks == [0, 1] and counters[0].loaded == [] and counters[0].removed
    assert rec["structural"] == {"micro_batches": 4, "steps": 1, "faults": []}
    assert rec["facts"]["block_swap"] == {"indices": [1, 3], "blocks": 2, "depth": 2, "host_bytes": 300, "pinned_bytes": 50,
                                          "block_bytes": [100, 200], "params_per_block": [2, 2]}


def test_swap_state_flags_disagreeing_blocks(unsloth_env):
    cls, _ = unsloth_env
    swap = _swap(cls)
    b0, b1 = swap.blocks
    h = p237_arm.make()
    h.post_import(_args())
    try:
        h.post_load(_args(), FakeModel(swap))
        b0.evict()
        h.step_edge(0)
        assert h.edges[-1]["swap"] == {"resident": [3], "evicted_nonempty": [], "resident_empty": []}
        b0.params[0].data = torch.zeros(3, 2)             # MUTANT: a forward that skipped the swap left real weights in an evicted block
        b1.params[1].data = torch.empty(0)                 # MUTANT: a resident block holding a 0-element parameter
        h.step_edge(1)
        assert h.edges[-1]["swap"] == {"resident": [3], "evicted_nonempty": [1], "resident_empty": [3]}
    finally:
        h.uninstall()


def test_blocks_outside_the_swap_stay_distinct(unsloth_env):
    """A block the swap does not list (built during the load, then dropped) is named by its identity, never merged with another."""
    cls, _ = unsloth_env
    stray = [cls([(2,)], nbytes=8), cls([(2,)], nbytes=16)]
    h = p237_arm.make()
    h.post_import(_args())
    try:
        for b in stray:
            b.evict()
            b.prefetch("s")
        h.post_load(_args(), FakeModel(_swap(cls)))
        h.step_edge(0)
        rec = h.record()
    finally:
        h.uninstall()
    load = rec["fetch"]["pre"]["load"]
    assert sorted(load.values()) == [[1, 8], [1, 16]] and all(k.startswith("unmapped:") for k in load), load


def test_no_block_swap_is_recorded_as_none(unsloth_env):
    h = p237_arm.make()
    h.post_import(_args())
    try:
        h.post_load(_args(), FakeModel())
        h.step_edge(0)
        rec = h.record()
    finally:
        h.uninstall()
    assert rec["facts"]["block_swap"] is None and rec["edges"][0]["swap"] is None


def test_the_fake_counter_matches_the_real_counter_surface():
    sys.path.insert(0, str(REPO / "bench" / "common"))
    from unsloth_moe_counter import UnslothMoeCounter
    for name in ("install", "patch_loaded", "mark_step", "snapshot", "structural", "uninstall"):
        assert callable(getattr(UnslothMoeCounter, name, None)), name
    assert list(inspect.signature(UnslothMoeCounter.structural).parameters) == ["self", "micro_batches", "n_steps"]
    assert list(inspect.signature(UnslothMoeCounter.patch_loaded).parameters) == ["self", "experts"]


# -- e4b --------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def e4b_env(monkeypatch):
    offload = importlib.import_module("experts4bit_qlora.engines.offload")
    layer_major = importlib.import_module("experts4bit_qlora.engines.layer_major")
    route = types.ModuleType("nf4_route")
    route.train_gemm_route = lambda dev=None, n_groups=None: "k25" if (n_groups or 0) <= 4 else "mtile"
    monkeypatch.setitem(sys.modules, "nf4_route", route)
    calls = []

    def fake_step(model, batches, weights, handles=None):
        calls.append((model, len(batches), tuple(weights)))
        return ["loss"] * len(batches)

    monkeypatch.setattr(layer_major, "layer_major_step", fake_step)
    monkeypatch.setattr(offload, "_STATS", None)
    handles = [object(), object()]
    monkeypatch.setattr(offload, "offload_handles", lambda model: list(handles))
    monkeypatch.setattr(offload._ExpertOffload, "_resident", None)
    orig = (offload._OffloadStats.record_copy, route.train_gemm_route, fake_step)
    yield offload, layer_major, route, calls, handles
    assert (offload._OffloadStats.record_copy, route.train_gemm_route, layer_major.layer_major_step) == orig


def test_e4b_stagings_layer_major_and_route_per_interval(e4b_env, monkeypatch):
    offload, layer_major, route, calls, handles = e4b_env
    monkeypatch.setenv("E4B_OFFLOAD_STATS", "1")
    h = p237_arm.make()
    h.post_import(_args(framework="e4b", layer_major=1, accum=4))
    try:
        st = offload._stats()
        st.record_copy("s", "e", 10, 4, "bulk")           # during the load: pre / load
        model = FakeModel()
        h.post_load(_args(framework="e4b"), model)
        model.eval()
        st.record_copy("s", "e", 10, 4, "bulk")           # eval0: pre / eval
        model.train()
        out = layer_major.layer_major_step(model, [{}] * 4, [0.25] * 4)    # the drift guard's call: pre, never a step
        assert out == ["loss"] * 4 and calls[-1] == (model, 4, (0.25,) * 4)
        assert route.train_gemm_route("cuda", 2) == "k25"
        h.step_edge(0)
        layer_major.layer_major_step(model, [{}] * 4, [0.25] * 4)
        for _ in range(3):
            st.record_copy("s", "e", 7, 2, "bulk")
        st.record_copy("s", "e", 5, 1, "routed")
        assert route.train_gemm_route("cuda", 8) == "mtile"
        monkeypatch.setattr(offload._ExpertOffload, "_resident", handles[1])
        h.step_edge(1)
        rec = h.record()
    finally:
        h.uninstall()
    json.dumps(rec)
    assert [c[:4] for c in st.copies] == [("s", "e", 10, 4)] * 2 + [("s", "e", 7, 2)] * 3 + [("s", "e", 5, 1)]   # passed through
    assert rec["stage"]["pre"] == {"load|bulk": [1, 10, 4], "eval|bulk": [1, 10, 4]}
    assert rec["stage"]["0"] == {"train|bulk": [3, 21, 6], "train|routed": [1, 5, 1]}
    assert rec["layer_major_calls"] == {"pre": 1, "0": 1}
    assert rec["route_log"] == {"pre": {"2:k25": 1}, "0": {"8:mtile": 1}}
    assert rec["edges"][0]["staging"] == {"copies_total": 2, "bytes_total": 20, "cold_misses": 0, "resident_handle": None}
    assert rec["edges"][1]["staging"] == {"copies_total": 6, "bytes_total": 46, "cold_misses": 0, "resident_handle": 1}
    assert rec["facts"]["offload_handles"] == 2 and rec["facts"]["stats_enabled"] is True
    assert rec["facts"]["env"]["E4B_OFFLOAD_STATS"] == "1" and rec["facts"]["num_hidden_layers"] == 4


def test_e4b_without_layer_major_leaves_the_step_unwrapped(e4b_env, monkeypatch):
    offload, layer_major, route, calls, handles = e4b_env
    monkeypatch.delenv("E4B_OFFLOAD_STATS", raising=False)
    before = layer_major.layer_major_step
    h = p237_arm.make()
    h.post_import(_args(framework="e4b", layer_major=0))
    try:
        assert layer_major.layer_major_step is before
        h.post_load(_args(framework="e4b"), FakeModel())
        h.step_edge(0)
        rec = h.record()
    finally:
        h.uninstall()
    assert "layer_major_calls" not in rec and rec["facts"]["stats_enabled"] is False


# -- the contract -----------------------------------------------------------------------------------------------------------------
def test_the_source_launches_no_device_work_and_sets_nothing():
    """Read off the parsed code (docstrings and comments can name these words): no attribute or name that launches device work or
    writes the environment, no torch import, and no assignment into os.environ."""
    import ast
    tree = ast.parse(HOOK.read_text(encoding="utf-8"))
    bad = {"cuda", "to", "synchronize", "copy_", "empty", "zeros", "Event", "record_stream", "elapsed_time", "item", "putenv",
           "setenv", "set_per_process_memory_fraction", "pop", "update"}
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not bad & used, sorted(bad & used)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "torch" not in imported, imported
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete)):
            for t in getattr(n, "targets", None) or [n.target]:
                assert "environ" not in ast.unparse(t), ast.unparse(n)
    assert "environ" in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}   # it reads the environment, for the record


def test_make_takes_no_arguments_and_defines_the_four_phases():
    assert list(inspect.signature(p237_arm.make).parameters) == []
    h = p237_arm.make()
    for phase in ("post_import", "post_load", "step_edge", "record"):
        assert callable(getattr(h, phase)), phase


def test_tc1_arm_loads_this_hook_by_its_sha(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("tc1_arm_p237_under_test", ARM)
    arm = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "tc1_arm_p237_under_test", arm)
    spec.loader.exec_module(arm)
    monkeypatch.setenv(arm.ARM_HOOK_ENV, "p237_arm:make")
    monkeypatch.setenv(arm.ARM_HOOK_SHA_ENV, hashlib.sha256(HOOK.read_bytes()).hexdigest())
    a = types.SimpleNamespace(framework="e4b", fam="tiny", model="m", revision="0" * 40, arm="e4b", tag="t", steps=1, seq=8, accum=1,
                              micro_batch=1, offload=0, prereg="p", note=None, out=str(tmp_path), adapter_dir=str(tmp_path / "a"),
                              vram_cap_gib=None, unsloth_offload_layers=None, layer_major=0)
    h = arm.load_arm_hook(a)
    assert type(h["obj"]).__name__ == "P237Arm" and h["file"] == "p237_arm.py"
    assert h["phases"] == ["post_import", "post_load", "step_edge", "record"], h["phases"]
