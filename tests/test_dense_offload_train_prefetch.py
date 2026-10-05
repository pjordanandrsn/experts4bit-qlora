"""DQ3 (bench/dq3/DQ3-PREREG.md): the opt-in overlapped TRAINING prefetch of ``enable_dense_offload(train_prefetch=True)``.

The contract is the one ``test_dense_offload.py`` sets for inference, extended to training:

1. **Off is byte-identical.** With ``train_prefetch=False`` no schedule, side stream or event is built, and training
   gives exactly what it gave before.
2. **Bitwise parity.** With it on, losses and gradients are ``torch.equal`` to an un-offloaded run, under gradient
   checkpointing and without it.
3. **Bounded residency.** Two layers at most (the one in use + one scheduled neighbour), at every use.
4. **The schedule.** Forward prefetches i+1, backward i-1; the exact counts are pinned on a 4-layer model.
5. **The fence.** A copy into a block compute is still reading would corrupt weights silently; ``record_stream`` at bind
   is the fence. The race test shows the corruption with the fence removed (the mutation arm) and its absence with it.
"""
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experts4bit_qlora.engines import dense_offload as do
from experts4bit_qlora.engines.dense_offload import (_DenseOffload, dense_offload_report, enable_dense_offload,
                                                     train_schedule)

H, INTER, NL = 512, 1024, 4
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class Block(nn.Module):
    """A decoder-layer stand-in: two big projections (offloaded, FROZEN as in QLoRA) and a trainable 1-D norm that stays
    resident -- the gradient the parity tests compare."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, INTER, bias=False)
        self.o_proj = nn.Linear(INTER, H, bias=False)
        self.norm = nn.Parameter(torch.ones(H))

    def forward(self, x):
        return x + self.o_proj(torch.relu(self.q_proj(x * self.norm)))


class Toy(nn.Module):
    def __init__(self, n=NL, ckpt=True):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(n))
        self.ckpt = ckpt

    def forward(self, x):
        for lay in self.layers:
            x = checkpoint(lay, x, use_reentrant=False) if self.ckpt else lay(x)
        return x


@pytest.fixture(autouse=True)
def _clean_class_state():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()


def _model(device, seed=7, n=NL, ckpt=True):
    torch.manual_seed(seed)
    m = Toy(n, ckpt).to(device)
    for lay in m.layers:
        lay.q_proj.weight.requires_grad_(False)
        lay.o_proj.weight.requires_grad_(False)
    m.train()
    return m


def _steps(m, device, steps=3, seed=3):
    """Run ``steps`` forward/backward passes on fixed inputs; return (losses, norm grads, input grads) per step."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(steps):
        x = torch.randn(8, H, generator=g).to(device).requires_grad_(True)
        for p in m.parameters():
            p.grad = None
        loss = m(x).pow(2).mean()
        loss.backward()
        out.append((loss.detach().clone(), [lay.norm.grad.detach().clone() for lay in m.layers],
                    x.grad.detach().clone()))
    return out


def _assert_equal_runs(a, b):
    assert len(a) == len(b)
    for (la, ga, xa), (lb, gb, xb) in zip(a, b):
        assert torch.equal(la, lb), (la.item(), lb.item())
        assert torch.equal(xa, xb), (xa - xb).abs().max().item()
        for i, (u, v) in enumerate(zip(ga, gb)):
            assert torch.equal(u, v), (i, (u - v).abs().max().item())


# ------------------------------------------------------------- the schedule (pure) --
def _walk(order, n):
    last = phase = None
    seen = []
    for i in order:
        phase, target = train_schedule(last, phase, i, n)
        seen.append((i, phase, target))
        last, phase = (None, "fwd") if phase == "unscheduled" else (i, phase)
    return seen


def test_schedule_forward_then_checkpoint_backward_with_repeats():
    """A checkpointed step: forward 0..3, then each layer's recompute and its backward pre-hook (a repeat), 3..0."""
    seen = _walk([0, 1, 2, 3, 3, 3, 2, 2, 1, 1, 0, 0], 4)
    assert seen == [(0, "fwd", 1), (1, "fwd", 2), (2, "fwd", 3), (3, "fwd", 2),
                    (3, "bwd", 2), (3, "bwd", 2), (2, "bwd", 1), (2, "bwd", 1),
                    (1, "bwd", 0), (1, "bwd", 0), (0, "bwd", 1), (0, "bwd", 1)]


def test_schedule_backward_without_checkpointing():
    """No recompute: only the backward pre-hooks run in the backward, descending, right after the forward."""
    seen = _walk([0, 1, 2, 3, 3, 2, 1, 0], 4)
    assert [p for _i, p, _t in seen] == ["fwd"] * 4 + ["bwd"] * 4
    assert [t for _i, _p, t in seen] == [1, 2, 3, 2, 2, 1, 0, 1]


def test_schedule_next_step_starts_forward_from_layer_zero():
    seen = _walk([0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2], 4)
    # 0 after the backward's 0 is the boundary (a repeat, still resident); 1 turns the walk forward again
    assert seen[8] == (0, "bwd", 1) and seen[9] == (1, "fwd", 2) and seen[10] == (2, "fwd", 3)


def test_schedule_out_of_order_is_unscheduled_then_restarts():
    seen = _walk([0, 1, 3, 2], 4)
    assert seen[2] == (3, "unscheduled", None)
    assert seen[3][1] == "fwd"          # the walk restarts from the next use, never prefetching on a guess


@pytest.mark.parametrize("n,expect", [(1, [(0, "fwd", None), (0, "bwd", None)]),      # 0 is the last layer: a turnaround
                                      (2, [(0, "fwd", 1), (1, "fwd", 0), (1, "bwd", 0), (0, "bwd", 1)])])
def test_schedule_tiny_chains(n, expect):
    order = [0, 0] if n == 1 else [0, 1, 1, 0]
    assert _walk(order, n) == expect


def test_schedule_rejects_an_index_outside_the_chain():
    assert train_schedule(None, None, 5, 4) == ("unscheduled", None)
    assert train_schedule(None, None, 0, 0) == ("unscheduled", None)


# ------------------------------------------------------------------ off path -------
def test_off_builds_no_schedule_stream_or_event(monkeypatch):
    """train_prefetch=False (the default) must not construct the DQ3 machinery at all."""
    def boom(*a, **k):
        raise AssertionError("the off path reached the train-prefetch machinery")

    monkeypatch.setattr(do, "_prefetch_stream", boom)
    monkeypatch.setattr(do._TrainPrefetch, "__init__", boom)
    m = _model("cpu")
    hs = enable_dense_offload(m, "cpu", pin=False, prefetch=False)
    assert all(h._train is None and h._train_idx is None for h in hs)
    assert dense_offload_report(hs)["train_prefetch"] is None
    _steps(m, "cpu", steps=2)


def test_off_training_matches_no_offload_exactly():
    ref = _steps(_model("cpu"), "cpu")
    m = _model("cpu")
    enable_dense_offload(m, "cpu", pin=False, prefetch=False)
    _assert_equal_runs(ref, _steps(m, "cpu"))


def test_a_second_enable_with_train_prefetch_false_turns_it_off():
    m = _model("cpu")
    enable_dense_offload(m, "cpu", pin=False, prefetch=False, train_prefetch=True)
    hs = enable_dense_offload(m, "cpu", pin=False, prefetch=False, train_prefetch=False)
    assert all(h._train is None for h in hs)


# --------------------------------------------------------------- on, CPU (CI) -----
@pytest.mark.parametrize("ckpt", [True, False])
def test_on_cpu_is_bitwise_identical_and_bounded(ckpt):
    ref = _steps(_model("cpu", ckpt=ckpt), "cpu")
    m = _model("cpu", ckpt=ckpt)
    hs = enable_dense_offload(m, "cpu", pin=False, prefetch=False, train_prefetch=True)
    _assert_equal_runs(ref, _steps(m, "cpu"))
    c = dense_offload_report(hs)["train_prefetch"]["cpu"]
    assert c["hwm_resident"] <= 2, c
    assert c["unscheduled"] == 0, c
    # on CPU nothing is copied ahead: every target is staged when used
    assert c["fwd_prefetch_issued"] == c["bwd_prefetch_issued"] == 0, c


# --------------------------------------------------------------- on, CUDA ---------
@cuda
@pytest.mark.parametrize("ckpt", [True, False])
def test_on_cuda_is_bitwise_identical(ckpt):
    ref = _steps(_model("cuda", ckpt=ckpt), "cuda")
    m = _model("cuda", ckpt=ckpt)
    enable_dense_offload(m, "cuda", pin=True, prefetch=False, train_prefetch=True)
    _assert_equal_runs(ref, _steps(m, "cuda"))


@cuda
def test_on_cuda_counts_match_the_schedule_exactly():
    """4 layers, checkpointed, 3 steps. Step 1 starts cold (layer 0 blocking, 3 forward prefetches); every later step
    issues 2 x (n - 2) = 4 prefetches (2 forward, 2 backward) and blocks on none -- the turnaround and the step boundary
    are resident hits. The backward's prefetches are what a forward-only schedule would lack."""
    m = _model("cuda")
    hs = enable_dense_offload(m, "cuda", pin=True, prefetch=False, train_prefetch=True)
    _steps(m, "cuda", steps=1)
    c1 = dict(dense_offload_report(hs)["train_prefetch"]["cuda"])
    _steps(m, "cuda", steps=2)
    c3 = dense_offload_report(hs)["train_prefetch"]["cuda"]
    assert c1["fwd_blocking"] + c1["bwd_blocking"] == 1, c1                 # the cold layer 0
    assert (c1["fwd_prefetch_issued"], c1["bwd_prefetch_issued"]) == (3, 2), c1
    per_step = {k: (c3[k] - c1[k]) / 2 for k in c1}
    assert per_step["fwd_blocking"] == per_step["bwd_blocking"] == 0, per_step
    assert (per_step["fwd_prefetch_issued"], per_step["bwd_prefetch_issued"]) == (2, 2), per_step
    issued = c3["fwd_prefetch_issued"] + c3["bwd_prefetch_issued"]
    consumed = sum(c3[f"{p}_{k}"] for p in ("fwd", "bwd") for k in ("overlapped", "waited"))
    assert consumed == issued, c3
    assert c3["hwm_resident"] <= 2 and c3["unscheduled"] == 0, c3


def _race(mutant: bool) -> bool:
    """True when the delayed read saw the weight it was given. Layer 1 is prefetched (its block belongs to the prefetch
    stream's pool) and bound; compute is held up by a sleep kernel, then reads layer 1's weight. The next use evicts
    layer 1 and prefetches layer 3 into a same-sized block: without the fence the allocator may hand layer 3's copy the
    block compute is about to read."""
    m = _model("cuda", n=4, ckpt=False)
    with torch.no_grad():
        for i, lay in enumerate(m.layers):           # distinct weights per layer, so a stale/overwritten read shows
            lay.q_proj.weight.fill_(float(i + 1))
    hs = enable_dense_offload(m, "cuda", pin=True, prefetch=False, train_prefetch=True)
    sched = hs[0]._train
    sched.use(hs[0])                                  # stages 0, prefetches 1
    sched.use(hs[1])                                  # binds 1 (record_stream unless mutant), prefetches 2
    x = torch.ones(4, H, device="cuda")
    # Warm the reader first. A process's first cuBLAS call (and any read that needs a fresh cudaMalloc) synchronises the
    # whole device, so an unwarmed matmul here finishes the sleep before returning and the race can never happen --
    # measured on the A2000: compute done 0-1 ms after the read was "enqueued", the mutant never corrupting.
    x @ m.layers[1].q_proj.weight.t()
    torch.cuda.synchronize()
    torch.cuda._sleep(int(3e8))                       # hold compute well past the next copy
    y = x @ m.layers[1].q_proj.weight.t()             # reads layer 1 AFTER the sleep
    sched.use(hs[2])                                  # evicts 1, prefetches 3 into the pool layer 1's block went back to
    torch.cuda.synchronize()
    return bool(torch.equal(y, torch.full_like(y, 2.0 * H)))


@cuda
def test_the_fence_holds():
    assert _race(mutant=False), "the delayed read saw another layer's bytes WITH the fence in place"


@cuda
def test_the_race_is_real_without_the_fence(monkeypatch):
    """The mutation arm: remove the fence (record_stream becomes a no-op). The same sequence must now corrupt the read,
    or this race test has no power and the passing test above proves nothing."""
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)
    assert not _race(mutant=True), "the race did not fire without the fence: the race test is not sensitive"


def test_no_path_streams_a_trainable_parameter():
    """A trainable 2-D parameter over MIN_BYTES (a LoRA matrix) stays resident on BOTH paths: streaming it would hand
    the optimizer an empty placeholder after eviction. Frozen projections still stream."""
    def model_with_trainable_big():
        m = _model("cpu")
        m.layers[0].o_proj.weight.requires_grad_(True)      # 2 MB, trainable
        return m

    def streams(h, mod, attr):
        return any(sm is mod and sa == attr for sm, sa, _p, _h in h.slots)

    for train_prefetch in (True, False):
        _DenseOffload._staged_now.clear()
        _DenseOffload._resident.clear()
        m = model_with_trainable_big()
        hs = enable_dense_offload(m, "cpu", pin=False, prefetch=False, train_prefetch=train_prefetch)
        assert not streams(hs[0], m.layers[0].o_proj, "weight"), f"a trainable parameter streams ({train_prefetch=})"
        assert streams(hs[0], m.layers[0].q_proj, "weight"), f"the frozen projection must still stream ({train_prefetch=})"
        assert m.layers[0].o_proj.weight.numel() == INTER * H, "the trainable weight must stay bound (not a placeholder)"


@pytest.mark.parametrize("train_prefetch", [False, True])
def test_an_optimizer_steps_a_large_trainable_parameter(train_prefetch):
    """The DQ3 rehearsal's crash, in miniature: AdamW over a trainable matrix over MIN_BYTES, through the offloaded
    model, for two steps. Before the fix the default path streamed that matrix and AdamW raised on the 0-element
    placeholder. Results match the un-offloaded model bit for bit."""
    def run(offload):
        _DenseOffload._staged_now.clear()
        _DenseOffload._resident.clear()
        m = _model("cpu")
        m.layers[1].o_proj.weight.requires_grad_(True)
        if offload:
            enable_dense_offload(m, "cpu", pin=False, prefetch=False, train_prefetch=train_prefetch)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(5)
        losses = []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = m(torch.randn(8, H, generator=g)).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.detach().clone())
        return losses, m.layers[1].o_proj.weight.detach().clone()

    (la, wa), (lb, wb) = run(False), run(True)
    assert all(torch.equal(a, b) for a, b in zip(la, lb))
    assert torch.equal(wa, wb), "the trained matrix diverged under offload"


@pytest.mark.parametrize("device,prefetch", [
    ("cpu", False),
    pytest.param("cuda", False, marks=cuda),
    pytest.param("cuda", True, marks=cuda),      # the inference chain wraps: eval leaves layer 0 PREFETCHED
])
def test_eval_mid_training_keeps_parity_and_the_schedule(device, prefetch):
    """The interleaving HF Trainer hits first: train step -> model.eval() + no_grad forward -> model.train() -> train
    step, twice. The eval forward runs through the inference path and the same residency registry, while the training
    schedule still holds its last (layer 0, bwd) state and possibly an in-flight target. Losses, eval outputs and the
    trained weights must match the un-offloaded model bit for bit, and the schedule must stay on its rails: no
    unscheduled use, residency never above two, at most one blocking fetch per (re)start."""
    n_evals = 2

    def run(offload):
        _DenseOffload._staged_now.clear()
        _DenseOffload._resident.clear()
        m = _model(device)
        hs = []
        if offload:
            hs = enable_dense_offload(m, device, pin=False, prefetch=prefetch, train_prefetch=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        g = torch.Generator(device="cpu").manual_seed(4)
        x_eval = torch.randn(8, H, generator=g).to(device)
        losses, evals = [], []
        for k in range(n_evals + 1):
            m.train()
            opt.zero_grad(set_to_none=True)
            loss = m(torch.randn(8, H, generator=g).to(device)).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
            if k < n_evals:
                m.eval()
                with torch.no_grad():
                    evals.append(m(x_eval).cpu())
        weights = [p.detach().cpu().clone() for p in m.parameters() if p.requires_grad]
        return losses, evals, weights, (dense_offload_report(hs)["train_prefetch"] if hs else None)

    l0, e0, w0, _ = run(False)
    l1, e1, w1, rep = run(True)
    assert all(torch.equal(a, b) for a, b in zip(l0, l1)), (l0, l1)
    assert all(torch.equal(a, b) for a, b in zip(e0, e1)), "an eval forward diverged from the un-offloaded model"
    assert len(w0) == len(w1) and all(torch.equal(a, b) for a, b in zip(w0, w1)), "trained weights diverged"
    (c,) = rep.values()
    steps = n_evals + 1
    assert c["unscheduled"] == 0, c
    assert c["hwm_resident"] <= 2, c
    assert c["uses"] == 3 * NL * steps, c          # forward + checkpoint recompute + backward pre-hook, per layer
    if device == "cuda":
        assert c["fwd_blocking"] + c["bwd_blocking"] <= 1 + n_evals, c
