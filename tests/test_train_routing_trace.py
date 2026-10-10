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


def _train(model, rec, steps=2, accum=2, perturb_at=None):
    torch.manual_seed(0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for s in range(steps):
        for mb in range(accum):
            x = torch.randn(6, H)
            loss = model(x)
            if perturb_at == (s, mb):                     # only this micro-batch's RECOMPUTE routes differently
                model.model.layers[1].mlp.gate.perturb = True
            with rec.backward_phase():
                loss.backward()
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


def test_expert_row_bytes_qwen3_shape():
    n = 3 * 2048 * 768
    assert rp.expert_row_bytes(2048, 768) == n // 2 + (n // 64) * 4 == 2_654_208


def test_train_sequence_order_forward_then_recompute_and_dgrad_descending():
    ids = {0: np.array([[1, 2], [2, 3]], np.uint8), 1: np.array([[5, 6], [6, 7]], np.uint8)}
    mbs = {(0, 0): {"fwd": ids, "recompute": ids}}
    seq = [row for _, row in rp.train_sequence(mbs)]
    assert seq[:6] == [(0, 1), (0, 2), (0, 3), (1, 5), (1, 6), (1, 7)]          # forward, ascending layers
    assert seq[6:12] == [(1, 5), (1, 6), (1, 7)] * 2                              # layer 1: recompute, then dgrad
    assert seq[12:] == [(0, 1), (0, 2), (0, 3)] * 2


def test_null_keeps_set_sizes_and_recompute_equal_to_forward():
    ids = {layer: np.array([[1, 2], [2, 9]], np.uint8) for layer in range(4)}
    mbs = {(s, 0): {"fwd": ids, "recompute": ids} for s in range(3)}
    seq = rp.train_sequence(mbs, null_seed=7, n_experts=E)
    per = {}
    for step, (layer, e) in seq:
        per.setdefault((step, layer), []).append(e)
    for v in per.values():
        assert len(v) == 3 * 3                     # 3 distinct ids, touched in forward, recompute and dgrad
        assert v[:3] == v[3:6] == v[6:]            # the same set in all three passes


def _random_seq(n_steps=12, rows=40, per_step=60, seed=3):
    rnd = random.Random(seed)
    return [(s, (rnd.randrange(4), rnd.randrange(rows // 4))) for s in range(n_steps) for _ in range(per_step)]


@pytest.mark.parametrize("budget", [4, 8, 16, 40])
def test_belady_is_an_upper_bound_on_the_online_policies(budget):
    seq = _random_seq()
    b = rp.simulate(seq, budget, "belady", 2)[0]
    assert b >= rp.simulate(seq, budget, "lru", 2)[0]
    assert b >= rp.simulate(seq, budget, "lfu", 2)[0]


def test_a_budget_that_holds_every_row_misses_only_first_uses():
    """No evictions at a budget of every distinct row: the only scored misses are rows first seen in the scored window."""
    seq = _random_seq()
    distinct = len({r for _, r in seq})
    first = {}
    for s, r in seq:
        first.setdefault(r, s)
    cold = sum(1 for s in first.values() if s >= 6)
    for p in ("lru", "lfu", "belady"):
        h, t = rp.simulate(seq, distinct, p, fit_steps=6)
        assert t - h == cold


def test_profile_is_fitted_on_the_fit_window_and_scored_on_the_rest():
    seq = [(0, (0, 1))] * 5 + [(0, (0, 2))] * 3 + [(1, (0, 1)), (1, (0, 3)), (1, (0, 2))]
    h, t = rp.simulate(seq, budget=1, policy="profile", fit_steps=1)
    assert (h, t) == (1, 3)                           # only row (0, 1) is resident; scored on step 1 only


def test_replay_reports_hit_rate_and_staged_bytes():
    seq = _random_seq()
    out = rp.replay(seq, fit_steps=2, row_bytes=10, budgets=(8,), policies=("lru",))
    cell = out["cells"][0]
    assert cell["touches"] == sum(1 for s, _ in seq if s >= 2)
    assert cell["staged_bytes_per_step"] == (cell["touches"] - cell["hits"]) * 10 / out["scored_steps"]
