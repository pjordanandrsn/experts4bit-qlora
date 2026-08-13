"""`hot_rows` below its floor must be refused at attach, not many steps in.

A stage requests every expert one forward routed and each protects a slot from
eviction, so an undersized tier raises inside `ColdTier.ensure` — correct, but
only when a forward is finally unlucky enough to route more uniques than the
tier holds. On Qwen3-30B-A3B at seq 384 that is a MEDIAN of 63 and a MAX of 97
of 128: a tier sized to the median survives most forwards and kills the run on
one of them, after the checkpoint has loaded and the arena is open.

`num_experts` is the worst case that request can reach and is known at attach
for free, so the refusal belongs there — in the pre-flight, which opens nothing
and therefore has nothing to unwind.

Also covers the qd pass-through: e4b must NOT pin the reader's queue depth,
because grouped-nf4-gemm now sizes it from the host's CPU budget.
"""
import inspect

import pytest

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("nvme_residency")
pytest.importorskip("bitsandbytes", reason="Experts4bit quantization needs bnb")

from experts4bit_qlora.engines.nvme_train import (  # noqa: E402
    enable_nvme_train_residency,
)
from test_nvme_train_residency import (  # noqa: E402
    E,
    _bake,
    _model_of,
    _module,
)


def test_below_the_floor_is_refused_at_attach(tmp_path):
    mod = _module()
    path, _index = _bake(mod, tmp_path, name="floor.arena")
    model = _model_of(mod)
    with pytest.raises(ValueError, match="below the hard floor"):
        enable_nvme_train_residency(model, path, hot_rows=E - 1, device="cpu",
                                    pinned=False)


def test_the_refusal_opens_no_tier(tmp_path):
    """Same discipline as the geometry refusal: the check runs in the pre-flight,
    so a rejected configuration leaves nothing stamped and nothing to close."""
    mod = _module()
    path, _index = _bake(mod, tmp_path, name="floor2.arena")
    model = _model_of(mod)
    with pytest.raises(ValueError):
        enable_nvme_train_residency(model, path, hot_rows=1, device="cpu",
                                    pinned=False)
    assert not hasattr(mod, "_e4b_cold_tier"), \
        "a tier was opened for a configuration that was going to be refused"


def test_exactly_the_floor_is_accepted(tmp_path):
    """The floor is inclusive. `num_experts` rows hold one whole layer, which is
    the widest a single stage can ever request."""
    mod = _module()
    path, _index = _bake(mod, tmp_path, name="floor3.arena")
    model = _model_of(mod)
    n = enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    assert n > 0
    assert mod._e4b_cold_tier.hot_rows == E


def test_the_message_names_the_floor_and_its_cost(tmp_path):
    """An error that says only 'too small' makes the reader go measuring. This
    one has to carry the number to use and what it costs."""
    mod = _module()
    path, _index = _bake(mod, tmp_path, name="floor4.arena")
    model = _model_of(mod)
    with pytest.raises(ValueError) as ei:
        enable_nvme_train_residency(model, path, hot_rows=2, device="cpu",
                                    pinned=False)
    msg = str(ei.value)
    assert str(E) in msg, "the floor value itself is missing"
    assert "GB pinned" in msg, "the RAM cost of the floor is missing"
    assert "capacity_for_bytes" in msg, "no pointer to how to size it"


def test_qd_defaults_to_None_so_gnf4_sizes_it(tmp_path):
    """The reader's queue depth is measured against the HOST's CPU budget by
    grouped-nf4-gemm. A fixed default here would override that on every host and
    make the measurement inert -- which is exactly what `qd: int = 4` did."""
    sig = inspect.signature(enable_nvme_train_residency)
    assert sig.parameters["qd"].default is None


def test_default_qd_is_not_pinned_to_four(tmp_path, monkeypatch):
    """Route test, positive-controlled.

    Asserting `reader.qd == default_qd()` is VACUOUS on a host where default_qd()
    happens to be 4 — which is every small CI box, and was this one. So force
    default_qd to a value nothing would arrive at by accident and check the tier
    actually took it. Without this, a regression to `qd: int = 4` passes.
    """
    import nvme_reader

    monkeypatch.setattr(nvme_reader, "default_qd", lambda *a, **k: 11)
    mod = _module()
    path, _index = _bake(mod, tmp_path, name="qd.arena")
    model = _model_of(mod)
    enable_nvme_train_residency(model, path, hot_rows=E, device="cpu", pinned=False)
    assert mod._e4b_cold_tier.reader.qd == 11, (
        "the tier did not take grouped-nf4-gemm's CPU-scaled default — e4b is "
        "pinning the queue depth again")


def test_explicit_qd_is_still_honoured(tmp_path):
    mod = _module()
    path, _index = _bake(mod, tmp_path, name="qd2.arena")
    model = _model_of(mod)
    enable_nvme_train_residency(model, path, hot_rows=E, device="cpu", pinned=False,
                               qd=3)
    assert mod._e4b_cold_tier.reader.qd == 3
