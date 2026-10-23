"""E4B_TRAIN_DET_GATHER=1: the fused training forward's token gather with a deterministic backward (engines/fast.py).

The fused training forward copies every token to its k expert rows with ``hidden_states.index_select(0, order // k)``.
``index_select``'s autograd backward is ``index_add_``: on CUDA, float atomics into rows repeated k times, so dL/dx
changes in its last bits call to call on identical inputs. The knob takes the same gather through
``batched._GatherTokenRows`` (#765), whose backward assigns each row's gradient to its own cell and sums the k cells
along a fixed axis. Checked three ways, because the new backward cannot be bitwise equal to the one it replaces:

  (a) the forward is bitwise ``index_select``'s;
  (b) the backward is bitwise equal to ITSELF across repeated calls at k = 8 on CUDA (the property that matters);
      the CPU case is a proxy only, since CPU ``index_add_`` is sequential;
  (c) the backward is within float tolerance of ``index_select``'s, and exact at k = 1 (no repeated rows).
"""
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.engines import fast  # noqa: E402
from experts4bit_qlora.engines.batched import _GatherTokenRows  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _case(dev, tokens, k, hidden, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randint(0, 16, (tokens * k,), generator=g)
    order = torch.argsort(flat, stable=True).to(dev)
    hs = torch.randn(tokens, hidden, generator=g).to(dtype).to(dev)
    gout = torch.randn(tokens * k, hidden, generator=g).to(dtype).to(dev)
    return hs, order, gout


def _grad(hs0, order, k, gout, det):
    hs = hs0.clone().requires_grad_(True)
    if det:
        out = _GatherTokenRows.apply(hs, order, k)
    else:
        out = hs.index_select(0, torch.div(order, k, rounding_mode="floor"))
    out.backward(gout)
    return out.detach(), hs.grad


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("k", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_the_forward_is_index_select_bit_for_bit(dev, k, dtype):
    hs, order, gout = _case(dev, 97, k, 64, dtype)
    a, _ = _grad(hs, order, k, gout, det=False)
    b, _ = _grad(hs, order, k, gout, det=True)
    assert a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("k,dtype,atol,rtol", [(2, torch.float32, 1e-6, 1e-6), (8, torch.float32, 1e-5, 1e-5),
                                               (2, torch.bfloat16, 0.0, 0.0), (8, torch.bfloat16, 1.6e-1, 2e-2)])
def test_c_the_backward_is_index_selects_within_float_tolerance(dev, k, dtype, atol, rtol):
    """Two terms sum exactly in either order (a + b == b + a), so k = 2 is exact even in bf16; at k = 8 the k cells
    sum in another order, a float reordering bounded by the dtype's rounding."""
    hs, order, gout = _case(dev, 97, k, 64, dtype, seed=3)
    _, ga = _grad(hs, order, k, gout, det=False)
    _, gb = _grad(hs, order, k, gout, det=True)
    torch.testing.assert_close(gb, ga, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_c_exact_at_k1_where_no_row_repeats(dev, dtype):
    hs, order, gout = _case(dev, 97, 1, 64, dtype, seed=5)
    _, ga = _grad(hs, order, 1, gout, det=False)
    _, gb = _grad(hs, order, 1, gout, det=True)
    assert torch.equal(ga, gb)


def test_b_cpu_proxy_repeated_backwards_are_identical():
    """A PROXY for (b), not the check: CPU index_add_ is sequential, so both backwards are deterministic here. The
    CUDA case below, and the self-pair box, are the check."""
    hs, order, gout = _case("cpu", 97, 8, 64, torch.bfloat16, seed=7)
    runs = [_grad(hs, order, 8, gout, det=True)[1] for _ in range(3)]
    assert all(torch.equal(runs[0], r) for r in runs[1:])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the nondeterminism is CUDA float atomics; on CPU index_add_ is sequential")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_b_cuda_repeated_backwards_are_bitwise_identical_at_k8(dtype):
    """(b) on the card: identical inputs, k = 8, many repeats; every dL/dx bit-identical to the first."""
    hs, order, gout = _case("cuda", 512, 8, 256, dtype, seed=11)
    first = _grad(hs, order, 8, gout, det=True)[1]
    for _ in range(20):
        assert torch.equal(first, _grad(hs, order, 8, gout, det=True)[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused training path needs CUDA and grouped-nf4-gemm")
def test_b_the_fused_training_forward_takes_it_and_is_repeatable(monkeypatch):
    """Through ``fused_experts_train_forward`` itself: with the knob on the counter moves and repeated calls give
    bit-identical output and dL/dx; with it off the counter stays put (the default path, unchanged)."""
    pytest.importorskip("nf4_qlora")
    pytest.importorskip("bitsandbytes")
    from quant_guard import require_quantize
    from experts4bit_qlora import Experts4bit, ExpertsLoRA
    n_exp, k, hid, inter, n_tok = 16, 8, 256, 256, 512
    torch.manual_seed(0)
    gate_up = (torch.randn(n_exp, 2 * inter, hid) * 0.1).cuda()
    down = (torch.randn(n_exp, hid, inter) * 0.1).cuda()
    require_quantize("cuda", "nf4")
    base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.bfloat16)
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).cuda().train()
    assert fast.enable_fast_train(mod) == 1
    torch.manual_seed(1)
    hs0 = torch.randn(n_tok, hid, dtype=torch.bfloat16, device="cuda")
    idx = torch.rand(n_tok, n_exp, device="cuda").argsort(-1)[:, :k].contiguous()
    wts = torch.rand(n_tok, k, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(n_tok, hid, device="cuda")

    def run():
        hs = hs0.clone().requires_grad_(True)
        out = mod(hs, idx, wts)
        (out.float() * g).sum().backward()
        return out.detach(), hs.grad

    monkeypatch.setenv("E4B_TRAIN_DET_GATHER", "0")
    n0 = fast.FAST_TRAIN_STATS["det_gather"]
    run()
    assert fast.FAST_TRAIN_STATS["det_gather"] == n0
    monkeypatch.setenv("E4B_TRAIN_DET_GATHER", "1")
    first = run()
    assert fast.FAST_TRAIN_STATS["det_gather"] == n0 + 1
    for _ in range(5):
        again = run()
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    fast.disable_fast_train(mod)


def test_the_knob_is_off_unless_one(monkeypatch):
    for v, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False), ("2", False)):
        monkeypatch.setenv("E4B_TRAIN_DET_GATHER", v)
        assert fast.det_gather_enabled() is want
    monkeypatch.delenv("E4B_TRAIN_DET_GATHER")
    assert fast.det_gather_enabled() is False


def test_the_training_gather_is_the_only_one_switched():
    """The knob sits at the training forward's gather only: the inference forwards (no backward) keep theirs."""
    import inspect
    src = inspect.getsource(fast.fused_experts_train_forward)
    assert "_GatherTokenRows.apply(hidden_states, order, k)" in src and "det_gather_enabled()" in src
    for name in ("fused_experts_forward", "fused_experts_lora_forward"):
        assert "_GatherTokenRows" not in inspect.getsource(getattr(fast, name))
