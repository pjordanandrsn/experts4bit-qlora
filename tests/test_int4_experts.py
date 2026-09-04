# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The int4 serve enabler rides the load-plan machinery.

What these tests pin down is the WIRING, hermetically (no GPU, no
network, no transformers): for any family the planner understands, the
enabler must produce int4 stores that are byte-identical to packing the
exact stacks the loader itself would have fused from the same
checkpoint. The dequant/fusion machinery is covered by the loader
suites; here it is the oracle.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("safetensors")

from experts4bit_qlora.arch.moe_load import (  # noqa: E402
    make_plan_reader, read_fused_expert_layer)
from experts4bit_qlora.arch.moe_plan import plan_moe_checkpoint  # noqa: E402
from experts4bit_qlora.engines.int4_experts import (  # noqa: E402
    enable_serve_experts_int4, safetensors_reader)

pytest.importorskip("int4_b32")   # the enabler sizes split-K via its _plan
from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

E, N1, K1, K2 = 2, 64, 64, 32   # gate/up: (N1,K1); down: (K1//2? no) see below


class _PlanTree(torch.nn.Module):
    """A state_dict-only planning tree (the _toy pattern from
    test_moe_plan): the planner needs names, not tensors."""
    def __init__(self, names):
        super().__init__()
        self._names = list(names)

    def state_dict(self, *a, **k):
        return {nm: torch.zeros(1) for nm in self._names}


class _FakeState:
    """Just enough hot-residency surface for the enabler. ``gptoss``
    mirrors the wrapper flag the loader sets on a gpt-oss build (biases
    resident, clamped-GLU epilogue)."""
    def __init__(self, gptoss=False, gu_bias=None, dn_bias=None):
        self.h_gu_p = torch.zeros(3, dtype=torch.uint8)
        self.h_gu_a = torch.zeros(3)
        self.h_dn_p = torch.zeros(3, dtype=torch.uint8)
        self.h_dn_a = torch.zeros(3)
        self.all_hot = True
        self.gptoss = gptoss
        if gptoss:
            self.h_gu_b = gu_bias
            self.h_dn_b = dn_bias

    def _all_hot(self):
        return self.all_hot


def _live_model(wrapper_path, model_type, top_k=8):
    """A live tree whose experts module carries the fake state at
    ``wrapper_path`` (dotted, rooted at the causal LM)."""
    root = torch.nn.Module()
    node = root
    for part in wrapper_path.split("."):
        child = torch.nn.Module()
        node.add_module(part, child)
        node = child
    node._hot_residency = _FakeState()

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.model_type = model_type
    cfg.num_experts_per_tok = top_k
    root.config = cfg
    return root, node._hot_residency


def _write_ckpt(tmp_path, tensors):
    from safetensors.torch import save_file
    fp = tmp_path / "model.safetensors"
    save_file(tensors, str(fp))
    return str(tmp_path)


def _family_case(model_type):
    """(checkpoint tensors, plan-tree names, wrapper path) per family."""
    g = torch.Generator().manual_seed(7)
    def w(n, k):
        return (torch.randn(n, k, generator=g) / 8).to(torch.float32)
    if model_type == "qwen3_moe":
        pre = "model.layers.0.mlp.experts"
        ck = {}
        for e in range(E):
            ck[f"{pre}.{e}.gate_proj.weight"] = w(N1, K1)
            ck[f"{pre}.{e}.up_proj.weight"] = w(N1, K1)
            ck[f"{pre}.{e}.down_proj.weight"] = w(K1, N1)
        names = [f"{pre}.gate_up_proj", f"{pre}.down_proj"]
        wrap = pre
    elif model_type == "mixtral":
        src = "model.layers.0.block_sparse_moe.experts"
        # fused targets are NORMALIZED to mlp.experts.* whatever the
        # source prefix -- the serve tree is uniform across families
        pre = "model.layers.0.mlp.experts"
        ck = {}
        for e in range(E):
            ck[f"{src}.{e}.w1.weight"] = w(N1, K1)   # gate
            ck[f"{src}.{e}.w3.weight"] = w(N1, K1)   # up
            ck[f"{src}.{e}.w2.weight"] = w(K1, N1)   # down
        names = [f"{pre}.gate_up_proj", f"{pre}.down_proj"]
        wrap = pre
    else:
        raise AssertionError(model_type)
    ck["model.embed_tokens.weight"] = w(N1, K1)
    names.append("model.embed_tokens.weight")
    return ck, names, wrap


@pytest.mark.parametrize("model_type", ["qwen3_moe", "mixtral"])
def test_enable_matches_the_loader_stacks(tmp_path, model_type, monkeypatch):
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _family_case(model_type)
    src = _write_ckpt(tmp_path, ck)
    tree = _PlanTree(names)
    live, st = _live_model(wrap, model_type)

    n = enable_serve_experts_int4(live, src, model_type=model_type,
                                  plan_model=tree)
    assert n == 1
    stores = st._int4_stores
    assert set(stores) == {"gu", "dn"}

    # Oracle: the loader's own fused stacks from the same checkpoint.
    keys, read_tensor = safetensors_reader(src)
    plan = plan_moe_checkpoint(keys, tree, model_type)
    read = make_plan_reader(plan, read_tensor, torch.float32)
    first, down = read_fused_expert_layer(plan, 0, read, device="cpu",
                                          dtype=torch.float32)
    assert stores["gu"]["N"] == first.shape[1] == 2 * N1
    assert stores["gu"]["K"] == first.shape[2] == K1
    assert stores["dn"]["N"] == down.shape[1]
    assert stores["dn"]["K"] == down.shape[2]
    for e in range(E):
        for role, stack in (("gu", first), ("dn", down)):
            nn, kk = stores[role]["N"], stores[role]["K"]
            pk, sc = pack_int4_b32(stack[e])
            got = dequant_int4_ref(stores[role]["packed"][e].cpu(),
                                   stores[role]["scales"][e].cpu(), nn, kk)
            want = dequant_int4_ref(pk, sc, nn, kk)
            assert torch.equal(got, want), (model_type, role, e)
    # NF4 dropped by default.
    assert st.h_gu_p.numel() == 0 and st.h_dn_p.numel() == 0


def test_split_k_partials_sized_by_config_top_k(tmp_path):
    """A top-4 router must get a sk*4*N buffer, not a top-8 one -- the
    kernel reshapes the buffer EXACTLY (found live: Qwen1.5-MoE, K8)."""
    from int4_b32 import _plan

    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe", top_k=4)
    enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                              plan_model=_PlanTree(names))
    for role in ("gu", "dn"):
        srow = st._int4_stores[role]
        _b, _w, sk, _k = _plan(srow["N"], srow["K"])
        assert srow["part"].shape == (sk * 4, srow["N"])


def test_missing_top_k_refused_loudly(tmp_path):
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    del live.config.num_experts_per_tok
    with pytest.raises(RuntimeError, match="routed-experts-per-token"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                                  plan_model=_PlanTree(names))


def test_enable_refuses_tiered_layers(tmp_path):
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    st.all_hot = False
    with pytest.raises(RuntimeError, match="tiered"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                                  plan_model=_PlanTree(names))


def test_enable_refuses_when_no_wrapper(tmp_path):
    ck, names, _ = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, _ = _live_model("model.layers.0.mlp.experts", "qwen3_moe")
    del live.get_submodule("model.layers.0.mlp.experts")._hot_residency
    with pytest.raises(RuntimeError, match="hot-residency"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                                  plan_model=_PlanTree(names))


def test_keep_nf4_env(tmp_path, monkeypatch):
    monkeypatch.setenv("E4B_INT4_KEEP_NF4", "1")
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    enable_serve_experts_int4(live, src, model_type="qwen3_moe",
                              plan_model=_PlanTree(names))
    assert st.h_gu_p.numel() != 0


def test_meta_twin_prefers_the_live_class():
    """Composite trees (text decoder under a prefix) must be twinned by
    rebuilding the live model's own class, not a causal-LM auto class."""
    from experts4bit_qlora.engines.int4_experts import _meta_twin

    class Composite(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            lang = torch.nn.Module()
            lang.proj = torch.nn.Linear(4, 4)
            self.add_module("language_model", lang)

    class _Cfg:
        model_type = "toy_composite"
    live = Composite(_Cfg())
    twin = _meta_twin(live)
    assert type(twin) is Composite
    assert "language_model.proj.weight" in twin.state_dict()
    assert next(twin.parameters()).is_meta


def _granite_case():
    """A PRE-FUSED family: one stacked tensor per projection, routed
    through the plan's passthrough (plan.experts stays empty)."""
    g = torch.Generator().manual_seed(23)
    pre = "model.layers.0.block_sparse_moe"
    ck = {
        f"{pre}.input_linear.weight": (torch.randn(E, 2 * N1, K1, generator=g) / 8),
        f"{pre}.output_linear.weight": (torch.randn(E, K1, N1, generator=g) / 8),
        f"{pre}.router.layer.weight": (torch.randn(E, K1, generator=g) / 8),
        "model.embed_tokens.weight": (torch.randn(N1, K1, generator=g) / 8),
    }
    names = [f"{pre}.experts.gate_up_proj", f"{pre}.experts.down_proj",
             f"{pre}.router.weight", "model.embed_tokens.weight"]
    return ck, names, f"{pre}.experts"


def test_prefused_family_is_packed_not_refused(tmp_path, monkeypatch):
    """Granite-style pre-fused stacks must pack, not hit the vacuous
    refusal: plan.experts is empty for them by design."""
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _granite_case()
    src = _write_ckpt(tmp_path, ck)
    tree = _PlanTree(names)
    live, st = _live_model(wrap, "granitemoe")

    plan = plan_moe_checkpoint(list(ck), tree, "granitemoe")
    assert not plan.experts, "fixture must exercise the PRE-FUSED path"

    n = enable_serve_experts_int4(live, src, model_type="granitemoe",
                                  plan_model=tree)
    assert n == 1
    stores = st._int4_stores
    assert stores["gu"]["N"] == 2 * N1 and stores["gu"]["K"] == K1
    assert stores["dn"]["N"] == K1 and stores["dn"]["K"] == N1
    for e in range(E):
        nn, kk = stores["gu"]["N"], stores["gu"]["K"]
        pk, sc = pack_int4_b32(ck["model.layers.0.block_sparse_moe."
                                  "input_linear.weight"][e])
        assert torch.equal(
            dequant_int4_ref(stores["gu"]["packed"][e].cpu(),
                             stores["gu"]["scales"][e].cpu(), nn, kk),
            dequant_int4_ref(pk, sc, nn, kk))


def _gptoss_case(E=2, H=128, inter=64):
    """A gpt-oss layer as RELEASED: MXFP4 ``_blocks``/``_scales`` pairs
    (uint8, the e8m0 exponent per 32-group) plus per-projection biases,
    gate/up rows interleaved. Returns (ckpt, plan names, prefix, dense
    stacks in the module layout, biases)."""
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4
    g = torch.Generator().manual_seed(11)
    pre = "model.layers.0.mlp.experts"
    gu_blocks = torch.randint(0, 256, (E, 2 * inter, H // 32, 16), generator=g,
                              dtype=torch.uint8)
    gu_scales = torch.randint(120, 130, (E, 2 * inter, H // 32), generator=g,
                              dtype=torch.uint8)
    dn_blocks = torch.randint(0, 256, (E, H, inter // 32, 16), generator=g,
                              dtype=torch.uint8)
    dn_scales = torch.randint(120, 130, (E, H, inter // 32), generator=g,
                              dtype=torch.uint8)
    gu_bias = torch.randn(E, 2 * inter, generator=g)
    dn_bias = torch.randn(E, H, generator=g)
    ck = {f"{pre}.gate_up_proj_blocks": gu_blocks,
          f"{pre}.gate_up_proj_scales": gu_scales,
          f"{pre}.gate_up_proj_bias": gu_bias,
          f"{pre}.down_proj_blocks": dn_blocks,
          f"{pre}.down_proj_scales": dn_scales,
          f"{pre}.down_proj_bias": dn_bias,
          "model.embed_tokens.weight": torch.randn(8, H, generator=g)}
    names = [f"{pre}.gate_up_proj", f"{pre}.down_proj",
             f"{pre}.gate_up_proj_bias", f"{pre}.down_proj_bias",
             "model.embed_tokens.weight"]
    gu_dense = dequantize_mxfp4(gu_blocks, gu_scales, dtype=torch.float32)
    dn_dense = dequantize_mxfp4(dn_blocks, dn_scales, dtype=torch.float32)
    assert tuple(gu_dense.shape) == (E, H, 2 * inter)  # module layout, interleaved
    assert tuple(dn_dense.shape) == (E, inter, H)
    return ck, names, pre, gu_dense, dn_dense, gu_bias, dn_bias


def test_gptoss_is_packed_in_the_builders_layout(tmp_path, monkeypatch):
    """gpt-oss packs: the int4 stores must be byte-identical to packing
    the stacks the loader's own builder (``from_gptoss``) produces from
    the same released bytes -- gate block then up block, output-major --
    with N/K oriented for the GEMV. The oracle is the builder's transform
    replicated on the raw dequant, never the lane's own helper."""
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, pre, gu_dense, dn_dense, gu_bias, dn_bias = _gptoss_case()
    E_, H_, twoI = gu_dense.shape
    I_ = twoI // 2
    src = _write_ckpt(tmp_path, ck)
    gub = torch.cat([gu_bias[:, 0::2], gu_bias[:, 1::2]], dim=1)  # as the loader stores it
    live, st = _live_model(pre, "gpt_oss", top_k=4)
    st.gptoss = True
    st.h_gu_b = gub
    st.h_dn_b = dn_bias

    n = enable_serve_experts_int4(live, src, model_type="gpt_oss",
                                  plan_model=_PlanTree(names))
    assert n == 1
    stores = st._int4_stores
    assert stores["gu"]["N"] == twoI and stores["gu"]["K"] == H_
    assert stores["dn"]["N"] == H_ and stores["dn"]["K"] == I_
    # the builder's transform, line for line (arch/gptoss.py from_gptoss)
    gt = gu_dense.transpose(1, 2).contiguous()
    gt = torch.cat([gt[:, 0::2, :], gt[:, 1::2, :]], dim=1)
    gt_dn = dn_dense.transpose(1, 2).contiguous()
    for e in range(E_):
        pk, sc = pack_int4_b32(gt[e])
        assert torch.equal(stores["gu"]["packed"][e].cpu(), pk)
        assert torch.equal(stores["gu"]["scales"][e].cpu(), sc)
        pk, sc = pack_int4_b32(gt_dn[e])
        assert torch.equal(stores["dn"]["packed"][e].cpu(), pk)
        assert torch.equal(stores["dn"]["scales"][e].cpu(), sc)
    # and NOT the interleaved rows: packing the raw transpose must differ
    raw = gu_dense.transpose(1, 2).contiguous()
    pk_raw, _ = pack_int4_b32(raw[0])
    assert not torch.equal(stores["gu"]["packed"][0].cpu(), pk_raw), \
        "an interleaved pack must not pass -- the test would prove nothing"
    # split-K partials sized from the config's top_k (4 for gpt-oss)
    assert stores["gu"]["part"].shape[0] % 4 == 0


def test_gptoss_refuses_a_wrapper_without_the_bias_epilogue(tmp_path):
    """A wrapper the loader did not build as gpt-oss has no bias epilogue;
    serving int4 bytes through it would run a plain SwiGLU. Refuse."""
    ck, names, pre, *_ = _gptoss_case()
    src = _write_ckpt(tmp_path, ck)
    live, _st = _live_model(pre, "gpt_oss", top_k=4)
    with pytest.raises(RuntimeError, match="not gpt-oss flagged"):
        enable_serve_experts_int4(live, src, model_type="gpt_oss",
                                  plan_model=_PlanTree(names))


def test_gptoss_refuses_bias_width_mismatch(tmp_path):
    ck, names, pre, gu_dense, dn_dense, gu_bias, dn_bias = _gptoss_case()
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(pre, "gpt_oss", top_k=4)
    st.gptoss = True
    st.h_gu_b = gu_bias[:, : gu_bias.shape[1] // 2]     # wrong width
    st.h_dn_b = dn_bias
    with pytest.raises(RuntimeError, match="bias widths"):
        enable_serve_experts_int4(live, src, model_type="gpt_oss",
                                  plan_model=_PlanTree(names))


def test_gptoss_layout_helper_refuses_misshapen_stacks():
    from experts4bit_qlora.engines.int4_experts import _gptoss_packer_layout
    with pytest.raises(RuntimeError, match="disagree"):
        _gptoss_packer_layout(torch.zeros(2, 8, 6), torch.zeros(2, 4, 8))
    with pytest.raises(RuntimeError, match="expected"):
        _gptoss_packer_layout(torch.zeros(8, 6), torch.zeros(3, 8))
