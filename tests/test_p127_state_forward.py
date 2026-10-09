"""The MoE residual (lane P127's item c, #1477) through every residency state class (e4b#1313).

p127-prove-1 found hot residency's patched experts forward passing ``residual=`` (even ``None``) to hybrid's
``_HybridTier.forward``, which overrides ``forward`` without it: a TypeError on every MoE call of the served
all-resident build. The CPU tests had used a toy experts module. These pin the real classes:
- ``_state_forward`` sends no ``residual=`` keyword when there is none, the keyword where the class takes it, and the
  layer's own add otherwise;
- ``_HybridTier.forward`` takes the residual and hands it to the base forward;
- the base forward's non-collapsed residual path never re-enters a subclass override (its prefetch and amortization
  count would run twice);
- every residency state class in the package that overrides ``forward`` takes ``residual=``.
"""
import inspect
import types

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402


class _Plain:
    """A state class overriding forward WITHOUT residual (hybrid's before the fix, pipelined's)."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states * 2


class _Takes:
    def __init__(self):
        self.seen = []

    def forward(self, hidden_states, top_k_index, top_k_weights, residual=None):
        self.seen.append(residual)
        out = hidden_states * 2
        return out if residual is None else out + residual


def test_no_residual_sends_no_keyword_and_a_class_without_it_still_adds():
    h, idx, w = torch.randn(2, 4), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1)
    r = torch.randn(2, 4)
    assert torch.equal(hr._state_forward(_Plain(), h, idx, w, None), h * 2)              # no TypeError
    assert torch.equal(hr._state_forward(_Plain(), h, idx, w, r), h * 2 + r)             # the layer's own add
    t = _Takes()
    assert torch.equal(hr._state_forward(t, h, idx, w, r), h * 2 + r) and t.seen == [r]   # the keyword where taken
    hr._state_forward(t, h, idx, w, None)
    assert t.seen[-1] is None


def test_the_hybrid_tier_takes_the_residual_and_hands_it_to_the_base(monkeypatch):
    from experts4bit_qlora.engines.hybrid import _HybridTier
    assert "residual" in inspect.signature(_HybridTier.forward).parameters
    seen = []

    def base(self, hidden_states, top_k_index, top_k_weights, residual=None):
        seen.append(residual)
        return hidden_states
    monkeypatch.setattr(hr._HotResidency, "forward", base)
    st = object.__new__(_HybridTier)
    st.amort, st.pf_enabled, st.pf = None, False, None
    h, r = torch.randn(1, 4), torch.randn(1, 4)
    _HybridTier.forward(st, h, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1), residual=r)
    _HybridTier.forward(st, h, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))
    assert seen[0] is r and seen[1] is None


def test_the_base_residual_path_never_re_enters_a_subclass_override():
    class Sub(hr._HotResidency):
        calls = 0

        def forward(self, hidden_states, top_k_index, top_k_weights, residual=None):
            type(self).calls += 1
            return super().forward(hidden_states, top_k_index, top_k_weights, residual=residual)

        def _forward_diet(self, x, flat, top_k_weights, T, k, H, dev, input_dev, input_dtype):
            return torch.ones(T, H, dtype=input_dtype)
    st = object.__new__(Sub)
    st.mod = types.SimpleNamespace(compute_dtype=None)
    st.device = torch.device("cpu")
    st.collapse_resident, st.dispatch_diet = False, True
    h, r = torch.randn(3, 4, dtype=torch.bfloat16), torch.randn(3, 4, dtype=torch.bfloat16)
    out = Sub.forward(st, h, torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2), residual=r)
    assert Sub.calls == 1, "the subclass override ran once for one call"
    assert torch.equal(out, torch.ones(3, 4, dtype=torch.bfloat16) + r)


def _subclasses(cls):
    for c in cls.__subclasses__():
        yield c
        yield from _subclasses(c)


def test_every_residency_state_override_takes_the_residual():
    import importlib
    for name in ("hybrid", "nvme_experts", "cold_engine"):
        importlib.import_module(f"experts4bit_qlora.engines.{name}")
    bad = [c.__qualname__ for c in _subclasses(hr._HotResidency)
           if "forward" in c.__dict__ and c.__module__.startswith("experts4bit_qlora")
           and "residual" not in inspect.signature(c.forward).parameters]
    assert not bad, f"these override forward without residual=: {bad}"
