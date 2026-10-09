# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P106's in-process kernel switch and comparison helpers (``bench/p106/gdn_toggle.py``, ``bench/p106/p106_box.py``)
on CPU, before any GPU is rented.

- ``GdnToggle`` reads transformers' fallback closures for the four Gated DeltaNet functions in both Qwen3.5 modules.
  With the kernels masked at import (``tests/hybrid_reference.py``) its resolved record is transformers' own functions.
- With a stand-in "kernel" written into the closures (what the cells hold when fla imports), a tiny Qwen3.5 hybrid's
  forward calls the stand-in under ``use("resolved")`` and never under ``use("torch")``: the model looks the name up at
  call time, and the switch routes it.
- The mutant forces ``use_qk_l2norm_in_kernel=False`` on the delta rules only.
- ``compare`` reads KL on fp32 log-probs (zero for identical rows, positive otherwise), next-token NLL, argmax agreement
  and bit-identity; ``summarize`` reports d_nll as the alternative minus the reference.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

LANE = Path(__file__).resolve().parents[1] / "bench" / "p106"


def _load(name):
    if str(LANE) not in sys.path:
        sys.path.insert(0, str(LANE))
    if str(LANE.parent / "p98") not in sys.path:
        sys.path.insert(0, str(LANE.parent / "p98"))
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _modules():
    pytest.importorskip("transformers.models.qwen3_5", reason="needs transformers with Qwen3.5")
    from hybrid_reference import reference_modeling
    return reference_modeling("qwen3_5"), reference_modeling("qwen3_5_moe")


@pytest.fixture
def cells_restored():
    """Every test that writes a closure cell gets the original contents back, so no other test sees a stand-in."""
    m_dense, m_moe = _modules()
    toggle = _load("gdn_toggle")
    saved = []
    for mod in (m_dense, m_moe):
        for name in toggle.GDN_FUNCS:
            fn = getattr(mod, name)
            cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
            saved += [(c, c.cell_contents) for c in cells.values()]
    yield
    for c, v in saved:
        c.cell_contents = v


def test_without_the_kernels_every_function_resolves_to_transformers(cells_restored):
    m_dense, m_moe = _modules()
    tg = _load("gdn_toggle").GdnToggle([m_dense, m_moe])
    rec = tg.resolved_record()
    assert len(rec) == 8 and all(v in (m_dense.__name__, m_moe.__name__) for v in rec.values())
    tg.use("torch")
    assert tg.record() == rec
    with pytest.raises(ValueError):
        tg.use("fla")


def _tiny():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
                            max_position_embeddings=128)
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).eval()


def test_the_switch_routes_a_resolved_kernel_and_the_torch_path(cells_restored):
    m_dense, m_moe = _modules()
    gdn = _load("gdn_toggle")
    calls = {n: 0 for n in gdn.GDN_FUNCS}
    for name in gdn.GDN_FUNCS:                       # what the cells hold when a kernel package imports
        fn = getattr(m_dense, name)
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
        tf = cells["torch_function"].cell_contents

        def kernel(*a, _tf=tf, _n=name, **k):
            calls[_n] += 1
            return _tf(*a, **k)

        kernel.__module__ = "stand_in_kernel"
        cells["implementation"].cell_contents = kernel
        cells["is_new_implementation"].cell_contents = True
    tg = gdn.GdnToggle([m_dense, m_moe])
    assert {v for k, v in tg.resolved_record().items() if k.startswith(m_dense.__name__ + ".")} == {"stand_in_kernel"}
    model, ids = _tiny(), torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        tg.use("torch")
        ref = model(input_ids=ids).logits
        assert sum(calls.values()) == 0
        assert all(v.startswith("transformers.") for v in tg.record().values())
        tg.use("resolved")
        got = model(input_ids=ids).logits
        assert calls["torch_chunk_gated_delta_rule"] == 1 and calls["causal_conv1d_fn"] == 1
        assert torch.equal(got, ref)                 # the stand-in IS the torch function: same arithmetic, other route
        tg.use("torch")
        model(input_ids=ids)
        assert calls["torch_chunk_gated_delta_rule"] == 1


def test_the_mutant_forces_the_l2norm_flag_off_on_the_delta_rules_only(cells_restored):
    m_dense, m_moe = _modules()
    box = _load("p106_box")
    gdn = _load("gdn_toggle")
    seen = {}
    for name in gdn.GDN_FUNCS:
        fn = getattr(m_moe, name)
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))

        def kernel(*a, _n=name, **k):
            seen[_n] = k.get("use_qk_l2norm_in_kernel", "absent")

        cells["implementation"].cell_contents = kernel
    tg = gdn.GdnToggle([m_moe])
    box.mutant(tg)
    for name in gdn.GDN_FUNCS:
        impl = dict(zip(getattr(m_moe, name).__code__.co_freevars, getattr(m_moe, name).__closure__))["implementation"]
        impl.cell_contents(use_qk_l2norm_in_kernel=True)
    assert seen["torch_chunk_gated_delta_rule"] is False and seen["torch_recurrent_gated_delta_rule"] is False
    assert seen["causal_conv1d_fn"] is True and seen["causal_conv1d_update"] is True
    tg.use("resolved")
    assert tg.record() == tg.resolved_record()


def test_compare_reads_kl_nll_agreement_and_identity_on_fp32():
    box = _load("p106_box")
    torch.manual_seed(1)
    ref = torch.randn(300, 50).to(torch.bfloat16)
    alt = ref.clone()
    alt[7] += 0.5 * torch.randn(50).to(torch.bfloat16)
    tgt = torch.randint(0, 50, (300,))
    rows = box.compare(ref, alt, tgt)
    assert len(rows["kl"]) == 300 and rows["identical"].count(False) == 1 and not rows["identical"][7]
    assert rows["kl"][7] > 0 and all(k == 0 for i, k in enumerate(rows["kl"]) if i != 7)
    lr = ref.float().log_softmax(-1)
    assert rows["nll_ref"][0] == pytest.approx(float(-lr[0, tgt[0]]), rel=1e-6)
    s = box.summarize(box.merge([rows, box.compare(ref[:5], ref[:5], tgt[:5])]))
    assert s["n"] == 305 and s["identical_rows"] == 304
    assert s["mean_d_nll"] == pytest.approx((rows["nll_alt"][7] - rows["nll_ref"][7]) / 305, rel=1e-5)
    assert box.summarize(box.merge([])) == {"n": 0}
