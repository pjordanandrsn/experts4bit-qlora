# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_KV_STEP_SELECT`` is on by default since lane P111 (``bench/p111/RESULTS-p111.md``): an unset variable turns the
step-level KV-table selection on, and ``0`` keeps the per-layer form. ``tests/test_kv_step_select.py`` (staged by P111 at
its registered bytes) pins the switch's behaviour; this file pins the default.
"""
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV


def _kv():
    return Fp8PagedKV(2, 2, 32, batch=2, max_tokens_per_seq=32, device="cpu", scratch_slots=2)


def test_an_unset_switch_is_on(monkeypatch):
    monkeypatch.delenv("E4B_KV_STEP_SELECT", raising=False)
    kv = _kv()
    assert kv._step_select is True and "tbl" in kv.graph_bucket(2)


def test_zero_keeps_the_per_layer_form(monkeypatch):
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "0")
    kv = _kv()
    assert kv._step_select is False and "tbl" not in kv.graph_bucket(2)
