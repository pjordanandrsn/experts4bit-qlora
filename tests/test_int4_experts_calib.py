# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calibrated int4 experts: the fused forward's Hessian tap feeds a
per-(layer, expert) accumulator, and the enable packs each expert with
its own gate/up and down Hessians through the kernel side's GPTQ packer
-- round-to-nearest only for experts the calibration text never routed
to, and counted."""
import sys
import types

import pytest
import torch

from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.engines.hot_residency import _fused_over_stack
from experts4bit_qlora.engines.int4_experts import (
    _ExpertHessianSink, enable_serve_experts_int4)
from int4_pack_ref import dequant_int4_ref, pack_int4_b32

from test_int4_experts import E, K1, N1, _family_case, _live_model, _PlanTree, _write_ckpt


class _Acc:
    """A HessianAccumulator stand-in: 2 X X^T on the given device."""
    def __init__(self, k, device=None):
        self.H = torch.zeros(k, k, dtype=torch.float32, device=device or "cpu")

    def add(self, x):
        xf = x.float().to(self.H.device)
        self.H += 2.0 * (xf.t() @ xf)


def _fake_gemm(x, p, a, sizes, eids):
    ids = eids if torch.is_tensor(eids) else torch.as_tensor(eids)
    per_row = ids.to(torch.float32).repeat_interleave(torch.as_tensor(sizes))
    return x * (per_row.unsqueeze(1) + 2.0)


@pytest.fixture
def stubs(monkeypatch):
    gp = types.ModuleType("gptq_pack")
    gp.HessianAccumulator = _Acc
    calls = {"gptq": 0}

    def gptq_pack_int4_b32(w, hessian, **kw):
        # a visibly different pack: the calibrated grid is the RTN grid of
        # the Hessian-scaled weight, so the bytes differ from plain RTN
        calls["gptq"] += 1
        assert hessian.shape == (w.shape[1], w.shape[1]), (hessian.shape, w.shape)
        return pack_int4_b32(w * 0.5)
    gp.gptq_pack_int4_b32 = gptq_pack_int4_b32
    monkeypatch.setitem(sys.modules, "gptq_pack", gp)
    nf = sys.modules.get("nf4_grouped") or types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "nf4_grouped", nf)
    monkeypatch.setattr(nf, "gemm_4bit_grouped", _fake_gemm, raising=False)
    return calls


def test_tap_feeds_per_expert_hessians(stubs, monkeypatch):
    """One fused call (mock GEMM) with a sink installed: the sink sees the
    gate/up input rows and the down input rows per expert, keyed by the
    hot stack's identity, and ignores stacks it does not know."""
    gu_p = torch.ones(4, dtype=torch.uint8)
    sink = _ExpertHessianSink({id(gu_p): 3}, layers=[3], hessian_device="cpu")
    monkeypatch.setattr(hr, "_CALIB_SINK", sink)
    torch.manual_seed(1)
    x = torch.randn(6, 8)
    ids = torch.tensor([2, 0, 2, 1, 0, 2])
    _fused_over_stack(x, ids, gu_p, None, gu_p, None, (8, 16, 8, 8),
                      has_gate=True, act_fn=torch.nn.functional.silu)
    assert sink.unmatched == 0
    got = sink.hessians()
    assert set(got) == {3} and set(got[3]) == {0, 1, 2}
    H_gu, H_dn, rows = got[3][2]
    assert rows == 3 and H_gu.shape == (8, 8) and H_dn.shape == (4, 4)   # down input = gate half
    # the gate/up Hessian is 2 X X^T over exactly expert 2's rows
    xe = x[ids == 2]
    assert torch.allclose(H_gu, 2.0 * xe.t() @ xe, atol=1e-5)
    # an unknown stack is counted, not attributed
    other = torch.ones(4, dtype=torch.uint8)
    _fused_over_stack(x, ids, other, None, other, None, (8, 16, 8, 8),
                      has_gate=True, act_fn=torch.nn.functional.silu)
    assert sink.unmatched == 1 and set(sink.hessians()[3]) == {0, 1, 2}


def test_serving_path_has_no_tap_cost(monkeypatch, stubs):
    monkeypatch.setattr(hr, "_CALIB_SINK", None)
    x = torch.randn(3, 8)
    ids = torch.tensor([0, 1, 0])
    a = _fused_over_stack(x, ids, None, None, None, None, (8, 16, 8, 8),
                          has_gate=False, act_fn=torch.nn.functional.silu)
    assert a.shape == (3, 8)


def test_enable_packs_calibrated_experts_and_counts_rtn(tmp_path, stubs, monkeypatch):
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    Hg = torch.eye(K1) * 4.0
    Hd = torch.eye(N1) * 4.0
    # expert 0 calibrated (plenty of rows); expert 1 saw too few rows -> RTN
    hess = {0: {0: (Hg, Hd, 500), 1: (Hg, Hd, 3)}}
    n = enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                                  plan_model=_PlanTree(names), expert_hessians=hess)
    assert n == 1
    stores = st._int4_stores
    assert stores["calibrated"] == (2, 2)          # gu + dn for expert 0; gu + dn RTN for expert 1
    assert stubs["gptq"] == 2
    # the loader's own stacks: expert 1's bytes are plain RTN, expert 0's are the calibrated pack
    from experts4bit_qlora.arch.moe_load import make_plan_reader, read_fused_expert_layer
    from experts4bit_qlora.arch.moe_plan import plan_moe_checkpoint
    from experts4bit_qlora.engines.int4_experts import safetensors_reader
    keys, read_tensor = safetensors_reader(src)
    plan = plan_moe_checkpoint(keys, _PlanTree(names), "qwen3_moe")
    read = make_plan_reader(plan, read_tensor, torch.float32)
    first, down = read_fused_expert_layer(plan, 0, read, device="cpu", dtype=torch.float32)
    for role, stack in (("gu", first), ("dn", down)):
        nn_, kk = stores[role]["N"], stores[role]["K"]
        for e, w in ((1, stack[1]), (0, stack[0] * 0.5)):
            pk, sc = pack_int4_b32(w)
            got = dequant_int4_ref(stores[role]["packed"][e].cpu(), stores[role]["scales"][e].cpu(), nn_, kk)
            want = dequant_int4_ref(pk, sc, nn_, kk)
            assert torch.equal(got, want), (role, e)


def test_enable_refuses_a_layer_without_hessians(tmp_path, stubs, monkeypatch):
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, _st = _live_model(wrap, "qwen3_moe")
    with pytest.raises(RuntimeError, match="no expert Hessians"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                                  plan_model=_PlanTree(names), expert_hessians={7: {}})


def test_uncalibrated_enable_is_unchanged(tmp_path, stubs, monkeypatch):
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names))
    assert "calibrated" not in st._int4_stores and stubs["gptq"] == 0
