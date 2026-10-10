"""bench/train-routing-trace: the training routing recorder and the residency replay (CPU only)."""
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parents[1] / "bench" / "train-routing-trace"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load("routing_recorder")
rp = _load("replay")

E, K, H = 16, 4, 8


class Router(torch.nn.Module):
    """A Qwen3-MoE-shaped router: returns (logits, scores, int indices). `perturb` routes differently on demand."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, H))
        self.perturb = False

    def forward(self, h):
        logits = h @ self.weight.t()
        if self.perturb:
            logits = logits.flip(-1)                      # a different top-k: the deliberately divergent recompute
        top, idx = torch.topk(logits.softmax(-1), K, dim=-1)
        return logits, top, idx


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Router()
        self.w = torch.nn.Parameter(torch.randn(E, H, H) * 0.1)

    def forward(self, h):
        _, scores, idx = self.gate(h)
        return h + torch.einsum("tk,tkij,tj->ti", scores, self.w[idx], h)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()

    def forward(self, h):
        return self.mlp(h)


class Toy(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer() for _ in range(n_layers)])

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = checkpoint(layer, h, use_reentrant=False)  # gradient checkpointing: the router runs again in backward
        return h.square().mean()


def _train(model, rec, steps=2, accum=2, perturb_at=None, backward=None):
    """`backward(loss)` defaults to the labelled Tensor.backward path (rec.backward_phase)."""
    torch.manual_seed(0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for s in range(steps):
        for mb in range(accum):
            x = torch.randn(6, H)
            loss = model(x)
            if perturb_at == (s, mb):                     # only this micro-batch's RECOMPUTE routes differently
                model.model.layers[1].mlp.gate.perturb = True
            if backward is None:
                with rec.backward_phase():
                    loss.backward()
            else:
                backward(loss)
            model.model.layers[1].mlp.gate.perturb = False
        opt.step()
        opt.zero_grad()
        rec.end_step()


def test_recorder_captures_forward_and_recompute_and_they_agree():
    torch.manual_seed(0)
    model, rec = Toy(), rr.RoutingRecorder(n_experts=E)
    assert rec.attach(model) == 3
    _train(model, rec)
    chk = rec.check_fwd_equals_recompute()
    assert chk == {"differ": [], "no_recompute": []}
    assert rec.check_every_layer_fired() == []
    for s in range(2):
        for mb in range(2):
            for layer in range(3):
                f, r = rec.records[(s, mb, "fwd", layer)], rec.records[(s, mb, "recompute", layer)]
                assert f.shape == (6, K) and np.array_equal(f, r)


def test_fwd_recompute_check_is_armed_by_a_perturbed_microbatch():
    """The check must be able to FAIL: one micro-batch whose recompute routes differently is caught, and only it."""
    torch.manual_seed(0)
    model, rec = Toy(), rr.RoutingRecorder(n_experts=E)
    rec.attach(model)
    _train(model, rec, perturb_at=(1, 0))
    chk = rec.check_fwd_equals_recompute()
    assert chk["differ"] == [(1, 0, 1)]
    assert chk["no_recompute"] == []


def test_eval_forward_is_not_recorded_and_forward_only_is_reported():
    torch.manual_seed(0)
    model, rec = Toy(), rr.RoutingRecorder(n_experts=E)
    rec.attach(model)
    with torch.no_grad():
        model(torch.randn(6, H))                           # an eval forward: ignored
    assert rec.records == {}
    model(torch.randn(6, H))                               # a grad-enabled forward with no backward: forward-only
    assert len(rec.check_fwd_equals_recompute()["no_recompute"]) == 3


def test_router_output_that_is_not_logits_scores_indices_is_refused():
    with pytest.raises(TypeError):
        rr.router_indices(torch.zeros(2, 4))
    with pytest.raises(TypeError):
        rr.router_indices((torch.zeros(2), torch.zeros(2), torch.zeros(2)))   # float "indices"


def test_attach_refuses_a_model_without_routers():
    with pytest.raises(RuntimeError):
        rr.RoutingRecorder().attach(torch.nn.Linear(2, 2))


class BypassMlp(Mlp):
    """Routes from gate.weight directly, never calling the hooked gate module (the zero-records failure mode)."""

    def forward(self, h):
        scores, idx = torch.topk((h @ self.gate.weight.t()).softmax(-1), K, dim=-1)
        return h + torch.einsum("tk,tkij,tj->ti", scores, self.w[idx], h)


def test_completeness_guard_voids_a_trace_with_zero_records(tmp_path):
    torch.manual_seed(0)
    model = Toy()
    for layer in model.model.layers:
        layer.mlp = BypassMlp()
    rec = rr.RoutingRecorder(n_experts=E)
    rec.attach(model)
    _train(model, rec)
    s = rec.save(str(tmp_path / "t.npz"), {}, expect_mb=2)
    assert s["verdict"] == "VOID" and "no records" in s["reasons"][0]


def test_completeness_guard_refuses_an_unlabelled_autograd_backward(tmp_path):
    """Armed: backward through torch.autograd.backward with NO labelling installed mislabels the recompute as new
    forward micro-batches; the guard must refuse rather than read clean."""
    torch.manual_seed(0)
    model, rec = Toy(), rr.RoutingRecorder(n_experts=E)
    rec.attach(model)
    _train(model, rec, backward=lambda loss: torch.autograd.backward(loss))
    s = rec.save(str(tmp_path / "t.npz"), {}, expect_mb=2)
    assert s["verdict"] == "VOID"
    assert any("micro-batches" in r for r in s["reasons"])
    with pytest.raises(rp.IncompleteTrace):
        rp.load_train_trace(str(tmp_path / "t.npz"), expect_mb=2)


def test_installed_labelling_covers_torch_autograd_backward(tmp_path):
    torch.manual_seed(0)
    model, rec = Toy(), rr.RoutingRecorder(n_experts=E)
    rec.attach(model)
    undo = rr.install_backward_labelling(rec)
    try:
        _train(model, rec, backward=lambda loss: torch.autograd.backward(loss))
    finally:
        undo()
    assert torch.autograd.backward.__module__ == "torch.autograd"      # restored
    s = rec.save(str(tmp_path / "t.npz"), {}, expect_mb=2)
    assert s["verdict"] == "OK", s["reasons"]
    mbs = rp.load_train_trace(str(tmp_path / "t.npz"), expect_mb=2)
    assert sorted(mbs) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_replay_refuses_a_wrong_expect_mb_and_a_tampered_file(tmp_path):
    torch.manual_seed(0)
    model, rec = Toy(), rr.RoutingRecorder(n_experts=E)
    rec.attach(model)
    _train(model, rec)
    path = str(tmp_path / "t.npz")
    assert rec.save(path, {}, expect_mb=2)["verdict"] == "OK"
    with pytest.raises(rp.IncompleteTrace):
        rp.load_train_trace(path, expect_mb=3)                         # the claim does not match the file
    data = dict(np.load(path))
    del data["s1_mb0_recompute_L2"]                                     # meta still says OK; the file no longer does
    np.savez_compressed(path, **data)
    with pytest.raises(rp.IncompleteTrace):
        rp.load_train_trace(path, expect_mb=2)


def test_expert_row_bytes_qwen3_shape_and_layout_is_cited():
    n = 3 * 2048 * 768
    assert rp.expert_row_bytes(2048, 768) == n // 2 + (n // 64) * 4 == 2_654_208
    assert rp.LAYOUT["blocksize"] == 64 and rp.LAYOUT["absmax_bytes"] == 4 and not rp.LAYOUT["double_quant"]
    assert all(":" in src for src in rp.LAYOUT["sources"])


def test_train_sequence_order_and_pass_labels():
    ids = {0: np.array([[1, 2], [2, 3]], np.uint8), 1: np.array([[5, 6], [6, 7]], np.uint8)}
    seq = rp.train_sequence({(0, 0): {"fwd": ids, "recompute": ids}})
    assert [(p, r) for _, p, r in seq[:6]] == [("fwd", (0, 1)), ("fwd", (0, 2)), ("fwd", (0, 3)),
                                               ("fwd", (1, 5)), ("fwd", (1, 6)), ("fwd", (1, 7))]
    assert [p for _, p, _ in seq[6:12]] == ["recompute"] * 3 + ["dgrad"] * 3            # layer 1, descending
    assert [r for _, _, r in seq[6:12]] == [(1, 5), (1, 6), (1, 7)] * 2
    assert [p for _, p, _ in seq[12:]] == ["recompute"] * 3 + ["dgrad"] * 3


def test_null_keeps_set_sizes_and_recompute_equal_to_forward():
    ids = {layer: np.array([[1, 2], [2, 9]], np.uint8) for layer in range(4)}
    seq = rp.train_sequence({(s, 0): {"fwd": ids, "recompute": ids} for s in range(3)}, null_seed=7, n_experts=E)
    per = {}
    for step, p, (layer, e) in seq:
        per.setdefault((step, layer, p), []).append(e)
    for (step, layer, p), v in per.items():
        assert len(v) == 3
        assert v == per[(step, layer, "fwd")]       # the same random set in all three passes


def _random_seq(n_steps=12, rows=40, per_step=60, seed=3):
    rnd = random.Random(seed)
    return [(s, "fwd", (rnd.randrange(4), rnd.randrange(rows // 4))) for s in range(n_steps) for _ in range(per_step)]


@pytest.mark.parametrize("budget", [4, 8, 16, 40])
def test_belady_is_an_upper_bound_on_the_online_policies(budget):
    seq = _random_seq()
    b = rp.simulate(seq, budget, "belady", 2)["pooled"][0]
    assert b >= rp.simulate(seq, budget, "lru", 2)["pooled"][0]
    assert b >= rp.simulate(seq, budget, "lfu", 2)["pooled"][0]


def test_a_budget_that_holds_every_row_misses_only_first_uses():
    """No evictions at a budget of every distinct row: the only scored misses are rows first seen in the scored window."""
    seq = _random_seq()
    distinct = len({r for _, _, r in seq})
    first = {}
    for s, _, r in seq:
        first.setdefault(r, s)
    cold = sum(1 for s in first.values() if s >= 6)
    for p in ("lru", "lfu", "belady"):
        h, t = rp.simulate(seq, distinct, p, fit_steps=6)["pooled"]
        assert t - h == cold


def test_profile_is_fitted_on_the_fit_window_and_scored_on_the_rest():
    seq = [(0, "fwd", (0, 1))] * 5 + [(0, "fwd", (0, 2))] * 3 + [(1, "fwd", r) for r in [(0, 1), (0, 3), (0, 2)]]
    assert rp.simulate(seq, budget=1, policy="profile", fit_steps=1)["pooled"] == (1, 3)


def test_dgrad_is_reported_separately_from_the_inflating_pooled_rate():
    """Each dgrad touch follows its layer's recompute, so at any budget holding one layer's set it always hits."""
    rng = np.random.default_rng(0)
    mbs = {(s, m): {"fwd": {L: rng.integers(0, E, size=(6, K)).astype(np.uint8) for L in range(3)},
                    "recompute": None} for s in range(4) for m in range(2)}
    for v in mbs.values():
        v["recompute"] = v["fwd"]
    out = rp.replay(rp.train_sequence(mbs), fit_steps=1, row_bytes=10, budgets=(E,), policies=("lru",))
    passes = out["cells"][0]["passes"]
    assert set(passes) == {"fwd", "recompute", "dgrad", "pooled"}
    assert passes["dgrad"]["hit_rate"] == 1.0
    assert passes["fwd"]["hit_rate"] < 1.0
    assert passes["pooled"]["hit_rate"] > passes["fwd"]["hit_rate"]
    assert passes["pooled"]["touches"] == sum(passes[p]["touches"] for p in ("fwd", "recompute", "dgrad"))


def test_run_capture_main_end_to_end_with_a_stub_harness(tmp_path):
    """Drives run_capture.main() itself: the wrapper's load_e4b hook, its backward labelling, and the optimizer
    step hook (an attribute lookup `torch.optim.optimizer.<name>` here failed at startup on real torch), then save."""
    import types

    rc_mod = _load("run_capture")

    def stub_main():
        model, _ = stub.load_e4b(None)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        torch.manual_seed(0)
        with torch.no_grad():
            model(torch.randn(6, H))                       # an eval forward: ignored by the recorder
        for _step in range(2):
            for _mb in range(3):
                model(torch.randn(6, H)).backward()
            opt.step()
            opt.zero_grad()
        return 0

    stub = types.ModuleType("tc1_arm")
    stub.load_e4b = lambda a: (Toy(), None)
    stub.main = stub_main
    real_load = rc_mod._load
    rc_mod._load = lambda name, path: stub if name == "tc1_arm" else real_load(name, path)
    before = (torch.Tensor.backward, torch.autograd.backward)
    out = tmp_path / "trace.npz"
    try:
        rc = rc_mod.main(["--trace-out", str(out), "--expect-mb", "3", "--", "--steps", "2"])
    finally:
        rc_mod._load = real_load
    assert rc == 0
    assert (torch.Tensor.backward, torch.autograd.backward) == before          # both entry points restored
    import json
    meta = json.loads(open(str(out) + ".meta.json").read())
    assert meta["summary"]["verdict"] == "OK", meta["summary"]["reasons"]
    assert meta["summary"]["steps"] == 2
    assert meta["meta"]["harness_args"] == ["--steps", "2"]
    assert sorted(rp.load_train_trace(str(out), expect_mb=3)) == [(s, m) for s in range(2) for m in range(3)]
