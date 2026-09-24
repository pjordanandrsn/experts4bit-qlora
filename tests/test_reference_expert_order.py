"""P67's perturbed-reference switch (``E4B_REFERENCE_EXPERT_ORDER``, experts4bit_qlora/lora.py).

The floor arm is only a floor if three things hold, and each is pinned here:

1. **Default off, and the default is the shipped loop.** Unset (or ``ascending``), the re-ordering helper is
   never called and the counter never moves, so every existing receipt's arithmetic is untouched.
2. **Correct by construction.** A re-ordered loop visits the SAME experts and sums the SAME terms: its output
   and every gradient equal the default's to rounding, in fp32 compute where rounding is ~1e-7.
3. **Not a no-op.** It must actually move the rounding -- a switch that happened to reproduce the default bit
   for bit would make every floor draw a plain repeat, and a floor of zero is the band against zero again.

Plus the refusals: an unregistered value raises rather than silently running the default, and the counter the
receipt reads is per call and resettable.
"""

import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

import experts4bit_qlora.lora as L  # noqa: E402
from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E, HID, INTER, TOP_K, N_TOK = 8, 128, 192, 4, 64
ENV = "E4B_REFERENCE_EXPERT_ORDER"


def _build(seed=0):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    require_quantize(DEVICE)
    base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():                        # a TRAINED adapter, so the low-rank path contributes too
        for p in (mod.gate_up_lora_B, mod.down_lora_B):
            p.normal_(std=0.05)
    return mod


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(N_TOK, HID, generator=g).to(DEVICE).requires_grad_(True)
    idx = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(N_TOK)]).to(DEVICE)
    wts = torch.softmax(torch.randn(N_TOK, TOP_K, generator=g), -1).to(DEVICE)
    return hs, idx, wts


def _step(mod, order, monkeypatch):
    if order is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, order)
    mod.zero_grad(set_to_none=True)
    hs, idx, wts = _inputs()
    out = mod(hs, idx, wts)
    (out * torch.linspace(-1, 1, out.numel(), device=DEVICE).view_as(out)).sum().backward()
    grads = [hs.grad.detach().clone()] + [p.grad.detach().clone() for p in
                                          (mod.gate_up_lora_A, mod.gate_up_lora_B, mod.down_lora_A, mod.down_lora_B)]
    return out.detach().clone(), grads


def test_default_is_the_shipped_loop_and_never_reorders(monkeypatch):
    mod = _build()
    L.reset_reference_order_stats()
    called = []
    real = L._order_expert_hit
    monkeypatch.setattr(L, "_order_expert_hit", lambda *a: called.append(a) or real(*a))
    out_unset, g_unset = _step(mod, None, monkeypatch)
    out_asc, g_asc = _step(mod, "ascending", monkeypatch)
    assert not called, "the default path must not re-order anything"
    assert L.reference_order_stats() == {"order": None, "calls": 0}
    assert torch.equal(out_unset, out_asc) and all(torch.equal(a, b) for a, b in zip(g_unset, g_asc))


@pytest.mark.parametrize("order", ["descending", "perm:1", "perm:2", "perm:3", "perm:4"])
def test_a_reordered_loop_is_the_same_function_to_rounding(order, monkeypatch):
    mod = _build()
    out0, g0 = _step(mod, None, monkeypatch)
    out1, g1 = _step(mod, order, monkeypatch)
    torch.testing.assert_close(out1, out0, rtol=1e-5, atol=1e-6)
    for a, b in zip(g1, g0):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_the_perturbation_is_not_a_no_op(monkeypatch):
    """At least one registered order must change SOME bit of the forward or the input gradient; if none did,
    the floor arms could not differ from the reference by construction."""
    mod = _build()
    out0, g0 = _step(mod, None, monkeypatch)
    moved = []
    for order in ("perm:1", "perm:2", "perm:3", "perm:4"):
        out1, g1 = _step(mod, order, monkeypatch)
        moved.append((not torch.equal(out1, out0)) or (not torch.equal(g1[0], g0[0])))
    assert any(moved), "no permuted order moved a single bit: the floor arm would be a plain repeat"


def test_perm_is_a_fixed_permutation_per_seed():
    r1, r1b, r2 = L._perm_rank(128, 1), L._perm_rank(128, 1), L._perm_rank(128, 2)
    assert sorted(r1.tolist()) == list(range(128)) and torch.equal(r1, r1b) and not torch.equal(r1, r2)
    hit = torch.tensor([0, 3, 5, 17, 64, 127])
    for order in ("descending", "perm:1", "perm:7"):
        got = L._order_expert_hit(hit, 128, order)
        assert sorted(got.tolist()) == hit.tolist()          # the SET of experts is unchanged
    assert L._order_expert_hit(hit, 128, "descending").tolist() == hit.flip(0).tolist()


@pytest.mark.parametrize("bad", ["reverse", "perm:", "perm:x", "perm:-1", "desc", "1"])
def test_an_unregistered_value_is_refused_not_read_as_the_default(bad, monkeypatch):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError, match=ENV):
        L._reference_expert_order()


def test_the_counter_is_per_call_and_resettable(monkeypatch):
    mod = _build()
    L.reset_reference_order_stats()
    _step(mod, "perm:3", monkeypatch)
    st = L.reference_order_stats()
    assert st["order"] == "perm:3" and st["calls"] == 1      # one forward (no checkpoint recompute here)
    L.reset_reference_order_stats()
    assert L.reference_order_stats() == {"order": None, "calls": 0}


def test_the_decode_fast_path_is_untouched(monkeypatch):
    """A single-token eval forward takes the decode path, which the switch does not govern."""
    mod = _build().eval()
    L.reset_reference_order_stats()
    monkeypatch.setenv(ENV, "perm:1")
    hs, idx, wts = _inputs()
    with torch.no_grad():
        mod(hs[:1].detach(), idx[:1], wts[:1])
    assert L.reference_order_stats()["calls"] == 0
