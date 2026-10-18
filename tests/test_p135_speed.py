# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P135 (Track B1's R-b): Mixtral's fold speed box (``bench/p135/p135_speed.py``) and its rule
(``bench/p135/p135_reduce.py``) on CPU, before any GPU is rented (bench/p135/PREREG-p135.md).

1. **Self-tests.** The box's and the rule's own.
2. **The registered tables, at the real depth.** A tiny Mixtral with 32 layers, built with FAM's ON_auto knobs (P115's
   stand-in kernels), reads exactly the census, the per-module fold counts and the fp32 router path the rule registers.
3. **One model, both settings.** ``FoldRoute(on=False)`` reproduces the unfused model's logits bit for bit and touches
   every fold-patched module; ``on=True`` is the fused model; leaving it restores the build's state.
4. **Two alternating runners.** One runner per setting, both alive, decoding in strict alternation on their own slots,
   emit exactly what each emits alone under its setting.

What CPU cannot run: the captured bucket graphs (the fused fp8 KV append, sm_89+). The blocks here decode eagerly.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.mixtral", reason="needs transformers with Mixtral")

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p135", "p115", "p110", "p108", "p97")]


def _load(name, lane="p135"):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / lane / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mixtral(layers=32):
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_local_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=512)
    torch.manual_seed(0)
    return MixtralForCausalLM(cfg).to(torch.bfloat16).eval()


def _fuse(model, monkeypatch, counters=False):
    """FAM's ON_auto on ``model`` with P115's stand-in kernels: -> (census, report[, counters]). The kernel counters, when
    asked for, are installed BEFORE the folds patch (each fused forward binds its kernels at patch time)."""
    sys.path.insert(0, str(ROOT / "tests"))
    import test_p115_quality_box as t
    monkeypatch.setitem(sys.modules, "int4_b32", t._kernel_stub())
    kc = _load("p115_quality", "p115").KernelCounters().install() if counters else None
    from experts4bit_qlora.serve_paged import FUSION_KNOBS, PagedServeConfig, _apply_fusions
    box = _load("p135_speed")
    report = {}
    census = _apply_fusions(model, PagedServeConfig(fusion_modes={k: box.KNOBS[k] for k in FUSION_KNOBS}), report=report)
    return (census, report, kc) if counters else (census, report)


def test_self_tests():
    assert _load("p135_speed").self_test() == 0
    assert _load("p135_reduce").self_test() == 0


def test_mixtral_at_full_depth_reads_the_registered_census_folds_and_fp32_routers(monkeypatch):
    box, red = _load("p135_speed"), _load("p135_reduce")
    model = _mixtral()
    census, report = _fuse(model, monkeypatch)
    assert [census[k] for k in box.CENSUS_KEYS] == red.CENSUS, census
    folds = box.fused_folds(model)
    assert box.by_module(folds) == red.FOLDS and len(folds) == red.N_FOLDS
    epi = report["folds"]["E4B_FUSE_ROUTER_EPI"]
    assert epi["patched"] == epi["fp32_upstream"] == red.FP32_ROUTERS, epi


def test_foldroute_off_is_the_unfused_model_and_on_is_the_fused_one(monkeypatch):
    box = _load("p135_speed")
    plain = _mixtral(layers=4)
    model = copy.deepcopy(plain)
    _, _, counters = _fuse(model, monkeypatch, counters=True)
    folds = box.fused_folds(model)
    assert box.by_module(folds) == {"glue_fuse": 9, "glue_r2": 8, "router_epilogue": 4}   # 2 x 4 + 1; 4 + 4; 4
    ids, pos = torch.randint(0, 256, (3, 1)), torch.full((3, 1), 8)

    def calls(fn):
        before = counters.snapshot()
        out = fn()
        return out, {k: v - before[k] for k, v in counters.snapshot().items() if v - before[k]}
    with torch.no_grad():
        want_off = plain(input_ids=ids, position_ids=pos, use_cache=False).logits
        fused = model(input_ids=ids, position_ids=pos, use_cache=False).logits
        with box.FoldRoute(model, False, folds) as r:
            assert r.n == len(folds) and all(m.__dict__["forward"] is orig for m, _, orig in folds)
            off, off_calls = calls(lambda: model(input_ids=ids, position_ids=pos, use_cache=False).logits)
        assert all(m.__dict__["forward"] is f for m, f, _ in folds)            # the build's state is back
        with box.FoldRoute(model, True, folds):
            on, on_calls = calls(lambda: model(input_ids=ids, position_ids=pos, use_cache=False).logits)
    assert torch.equal(off, want_off)
    assert torch.equal(on, fused)
    assert off_calls == {}                                                     # OFF runs no fold kernel at all
    assert on_calls.get("router_epilogue") == 4 and on_calls.get("rmsnorm_resid_rows") == 4, on_calls


def _reference_attention(monkeypatch):
    """Paged decode attention on CPU: SDPA over the pool's own dequantized K/V, as tests/test_fam_speed.py runs it."""
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


def _alone(box, model, prompt, n, on, folds):
    """One runner under one setting, one sequence, ``n`` greedy decode steps."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    runner = PagedModelRunner(model, box._pool(model, len(prompt) + n + 16, 0, "cpu"), device="cpu")
    out = []
    with torch.no_grad(), box.FoldRoute(model, on, folds):
        runner.bind(0, 0, list(prompt))
        runner.run_prefill([(0, 0, len(prompt))])
        for _ in range(n):
            out.append(int(runner.run_decode([0])[0]))
    return out


def test_two_alternating_runners_emit_what_each_emits_alone(monkeypatch):
    _reference_attention(monkeypatch)
    from experts4bit_qlora.engines import paged_attention
    box = _load("p135_speed")
    monkeypatch.setattr(box, "WARM", 1)
    monkeypatch.setattr(box, "BUSY", 2)
    model = _mixtral(layers=4)
    _fuse(model, monkeypatch)
    paged_attention.register(model)
    folds = box.fused_folds(model)
    prompt = torch.randint(0, 256, (24,), generator=torch.Generator().manual_seed(5)).tolist()
    steps = 4
    n = box.WARM + steps + box.BUSY
    rec = box.interleaved_block(model, prompt, "cpu", block="b", steps=steps, buckets=box.BUCKETS, graphs=False,
                                bulk_kv=False, folds=folds)
    assert rec["order"] == ["ON", "OFF"] and rec["slots"] == box.SLOT
    for st in box.SETTINGS:
        arm = rec["arms"][st]
        assert arm["folds"] == len(folds) and len(arm["step_ms"]) == steps and len(arm["tokens"]) == n
        assert arm["tokens"] == _alone(box, model, prompt, n, st == "ON", folds), st
