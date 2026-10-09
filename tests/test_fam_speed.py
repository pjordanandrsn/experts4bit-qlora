# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM, Amendment 4: the router-epilogue speed box (``bench/fam/fam_speed.py``) and its rule
(``bench/fam/fam_speed_reduce.py``) on CPU, before any GPU is rented (bench/fam/PREREG-fam.md; e4b#1362).

1. **Self-tests.** The box's and the rule's own.
2. **One model, both settings.** On a tiny Qwen3.5-MoE with the router epilogue fused (P115's stand-in kernels),
   ``EpiRoute(on=False)`` reproduces the unfused model's logits bit for bit and touches every fused router;
   ``on=True`` is the fused model; leaving it restores the build's state.
3. **The hybrid's one linear-state pool.** Two runners alive on one tiny hybrid, decoding in strict alternation on their
   own slots (``SLOT``), emit exactly what each emits alone. On one shared slot they do not: each step reads the state the
   other runner just advanced.

What CPU cannot run: the captured bucket graphs (the fused fp8 KV append, sm_89+). The blocks here decode eagerly.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.cache_utils", reason="needs transformers")
pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
                    reason="needs transformers with Qwen3.5-MoE")

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("fam", "p115", "p110", "p108", "p97")]
LIN, ATT = "linear_attention", "full_attention"
P = 24


def _load(name, lane="fam"):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / lane / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hybrid(monkeypatch):
    """Builds a tiny Qwen3.6-shaped hybrid, seeded: the same weights on every call."""
    monkeypatch.setitem(sys.modules, "causal_conv1d", None)     # CUDA-only kernels: the reference path on CPU,
    monkeypatch.setitem(sys.modules, "fla", None)               # bound when the modeling module is first imported
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q35

    def make(layer_types=(LIN, LIN, LIN, ATT)):
        cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=len(layer_types),
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=32, moe_intermediate_size=32,
                                   shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                                   linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                                   linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                                   max_position_embeddings=256)
        torch.manual_seed(0)
        return q35.Qwen3_5MoeForCausalLM(cfg).to(torch.bfloat16).eval()
    return make


def test_self_tests():
    assert _load("fam_speed").self_test() == 0
    assert _load("fam_speed_reduce").self_test() == 0


def _fuse_epilogue(model, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    import test_p115_quality_box as t
    monkeypatch.setitem(sys.modules, "int4_b32", t._kernel_stub())
    from experts4bit_qlora.serve_paged import FUSION_KNOBS, PagedServeConfig, _apply_fusions
    box = _load("fam_speed")
    return _apply_fusions(model, PagedServeConfig(fusion_modes={k: box.KNOBS[k] for k in FUSION_KNOBS}))


def test_epiroute_off_is_the_unfused_model_and_on_is_the_fused_one(hybrid, monkeypatch):
    box = _load("fam_speed")
    plain = hybrid()
    model = copy.deepcopy(plain)
    census = _fuse_epilogue(model, monkeypatch)
    routers = box.fused_routers(model)
    assert census["fuse_router_epilogue_n"] == len(routers) == 4             # every MoE layer's router
    ids, pos = torch.randint(0, 128, (3, 1)), torch.full((3, 1), 8)
    with torch.no_grad():
        want_off = plain(input_ids=ids, position_ids=pos, use_cache=False).logits
        with box.EpiRoute(model, False, routers) as r:
            assert r.n == 4 and all(m.__dict__["forward"] is orig for m, _, orig in routers)
            off = model(input_ids=ids, position_ids=pos, use_cache=False).logits
        assert all(m.__dict__["forward"] is fused for m, fused, _ in routers)    # the build's state is back
        with box.EpiRoute(model, True, routers):
            on = model(input_ids=ids, position_ids=pos, use_cache=False).logits
        fused = model(input_ids=ids, position_ids=pos, use_cache=False).logits
    assert torch.equal(off, want_off)
    assert torch.equal(on, fused)


def _reference_attention(monkeypatch):
    """Paged decode attention on CPU: SDPA over the pool's own dequantized K/V, as tests/test_linear_state.py runs it."""
    pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV

    def attention(self, layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
        outs = []
        for b, slot in enumerate(slots):
            kr, vr = self.reference_kv(layer, slot)
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
        return torch.stack(outs)
    monkeypatch.setattr(Fp8PagedKV, "attention", attention)


def _state(model, slot):
    """Every linear layer's pooled conv and recurrent state for ``slot``."""
    pool = model._e4b_linear_state
    return {k: (pool.conv[k][slot].clone(), pool.rec[k][slot].clone()) for k in sorted(pool.conv)}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k][0], b[k][0]) and torch.equal(a[k][1], b[k][1]) for k in a)


def _alone(box, model, prompt, n):
    """One runner, one sequence, ``n`` greedy decode steps: the reference each alternating runner must reproduce."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    runner = PagedModelRunner(model, box._pool(model, len(prompt) + n + 16, 0, "cpu"), device="cpu")
    out = []
    with torch.no_grad():
        runner.bind(0, 0, list(prompt))
        runner.run_prefill([(0, 0, len(prompt))])
        for _ in range(n):
            out.append(int(runner.run_decode([0])[0]))
    return out


def test_two_alternating_runners_on_their_own_slots_emit_what_each_emits_alone(hybrid, monkeypatch):
    _reference_attention(monkeypatch)
    from experts4bit_qlora.engines import paged_attention
    box = _load("fam_speed")
    monkeypatch.setattr(box, "BUSY", 2)
    monkeypatch.setattr(box, "WARM", 1)
    steps = 6
    n = box.WARM + steps + box.BUSY
    prompt = torch.randint(0, 128, (P,), generator=torch.Generator().manual_seed(5)).tolist()
    ref_model = hybrid()
    paged_attention.register(ref_model)
    want = _alone(box, ref_model, prompt, n)
    model = hybrid()
    paged_attention.register(model)
    rec = box.interleaved_block(model, prompt, "cpu", block="a", steps=steps, buckets=(1,), graphs=False,
                                bulk_kv=False, routers=[])
    assert rec["slots"] == box.SLOT and rec["order"] == ["OFF", "ON"]
    for st in box.SETTINGS:
        assert rec["arms"][st]["tokens"] == want, st
        assert len(rec["arms"][st]["step_ms"]) == steps
    ref = _state(ref_model, 0)                                  # the recurrent state one runner alone leaves
    assert _same(_state(model, box.SLOT["OFF"]), ref) and _same(_state(model, box.SLOT["ON"]), ref)
    shared = hybrid()                                           # the hazard: both runners on slot 0
    paged_attention.register(shared)
    box.interleaved_block(shared, prompt, "cpu", block="a", steps=steps, buckets=(1,), graphs=False, bulk_kv=False,
                          routers=[], slots={"OFF": 0, "ON": 0})
    assert not _same(_state(shared, 0), ref)                    # advanced twice a step: the state the box must not share
