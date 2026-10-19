"""experts4bit_qlora.engines.dense_lora_fuse: DQ11's MLP-adapter fusion (bench/dq11/DQ11-MLP-BUILD-PREREG.md), on CPU.

A tiny Mistral MLP with PEFT LoRA on gate/up/down (bf16 bases, FP32 adapters, as DQ11's L path). The bases are
nn.Linear here (``base_ok`` stands in for the NF4 test, which needs bitsandbytes on CUDA). The fused path must equal
PEFT's to rounding, run no PEFT MLP forward, refuse what the registration refuses, and leave nothing behind when
disabled."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from transformers import MistralConfig  # noqa: E402
from transformers.models.mistral.modeling_mistral import MistralMLP  # noqa: E402

from experts4bit_qlora.engines import dense_lora_fuse as dlf  # noqa: E402

LINEAR = lambda b: type(b) is nn.Linear  # noqa: E731  (the CPU stand-in for "an NF4 Linear4bit")


class Tiny(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        cfg = MistralConfig(hidden_size=64, intermediate_size=96, hidden_act="silu")
        self.layers = nn.ModuleList(nn.ModuleDict({"mlp": MistralMLP(cfg)}) for _ in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer["mlp"](x)
        return x


def _model(*, dropout=0.0, use_dora=False, r=4, alpha=8, seed=0):
    torch.manual_seed(seed)
    m = Tiny().to(torch.bfloat16)
    cfg = peft.LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, use_dora=use_dora,
                          target_modules=["gate_proj", "up_proj", "down_proj"])
    m = peft.inject_adapter_in_model(cfg, m)
    g = torch.Generator().manual_seed(seed + 1)
    for name, p in m.named_parameters():
        if "lora_" in name:
            with torch.no_grad():
                p.data = (torch.randn(p.shape, generator=g) * 0.05).float()   # FP32 adapters, B non-zero
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    return m


def _run(m, x):
    x = x.clone().requires_grad_(True)
    y = m(x)
    (y.float() ** 2).mean().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return y.detach(), x.grad.clone(), grads


def _x(seed=3):
    return torch.randn(2, 8, 64, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def test_fused_equals_peft_to_rounding_forward_and_every_gradient():
    ref = _model()
    fused = copy.deepcopy(ref)
    assert dlf.enable_dense_mlp_lora_fuse(fused, base_ok=LINEAR) == 6                # adapters: 2 MLPs x 3
    assert dlf.DENSE_MLP_LORA_STATS["fused"] == 2 and dlf.DENSE_MLP_LORA_STATS["refused"] == {}
    y0, dx0, g0 = _run(ref, _x())
    y1, dx1, g1 = _run(fused, _x())
    assert set(g0) == set(g1) and len(g0) == 12                    # 2 layers x 3 adapters x (A, B)
    torch.testing.assert_close(y1.float(), y0.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dx1.float(), dx0.float(), rtol=2e-2, atol=2e-2)
    for n in g0:
        torch.testing.assert_close(g1[n], g0[n], rtol=2e-2, atol=2e-3, msg=n)
    assert dlf.DENSE_MLP_LORA_STATS["calls"] == 2


@pytest.mark.parametrize("n", [2, 1])
def test_the_function_passes_gradcheck_in_float64_and_leaves_its_inputs_alone(n):
    g = torch.Generator().manual_seed(5)
    d = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64, requires_grad=True)   # noqa: E731
    ranks, outs = [2, 3][:n], [6, 4][:n]                                    # unequal ranks: the split is exercised
    x = d(3, 5, 8)
    As, Bs, bases = [d(r, 8) for r in ranks], [d(o, r) for o, r in zip(outs, ranks)], [d(3, 5, o) for o in outs]
    kept = [t.detach().clone() for t in (x, *bases)]
    dlf._FusedLoRA.apply(x, 1.7, n, *As, *Bs, *bases)
    assert all(torch.equal(t, k) for t, k in zip((x, *bases), kept))      # the epilogue never writes into a base output
    assert torch.autograd.gradcheck(lambda x, *r: dlf._FusedLoRA.apply(x, 1.7, n, *r), (x, *As, *Bs, *bases))
    recomputed = lambda x, *r: torch.utils.checkpoint.checkpoint(                              # noqa: E731
        lambda *a: dlf._FusedLoRA.apply(a[0], 1.7, n, *a[1:]), x, *r, use_reentrant=False)
    assert torch.autograd.gradcheck(recomputed, (x, *As, *Bs, *bases))   # the same Function through the checkpoint recompute


def test_an_input_that_needs_no_gradient_gets_none_and_the_adapters_still_get_theirs():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(4, 8, generator=g)                                      # no requires_grad
    a, b = torch.randn(2, 8, generator=g).requires_grad_(), torch.randn(3, 2, generator=g).requires_grad_()
    base = torch.randn(4, 3, generator=g).requires_grad_()
    (y,) = dlf._FusedLoRA.apply(x, 2.0, 1, a, b, base)
    y.sum().backward()
    assert x.grad is None
    torch.testing.assert_close(b.grad, 2.0 * torch.ones(4, 3).t() @ (x @ a.detach().t()))
    torch.testing.assert_close(a.grad, (2.0 * torch.ones(4, 3) @ b.detach()).t() @ x)


def test_a_fused_mlp_reads_its_adapters_live_and_refuses_one_cast_after_fusing():
    m = _model()
    dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR)
    mlp = m.layers[0]["mlp"]
    mlp.gate_proj.lora_A["default"].weight = nn.Parameter(torch.zeros_like(mlp.gate_proj.lora_A["default"].weight))
    ref = copy.deepcopy(m)
    dlf.disable_dense_mlp_lora_fuse(ref)
    torch.testing.assert_close(m(_x()).float(), ref(_x()).float(), rtol=2e-2, atol=2e-2)
    mlp.up_proj.lora_B["default"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="no longer FP32"):
        m(_x())


def test_a_fused_step_runs_no_peft_mlp_forward():
    m = _model()
    dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR)
    seen = []
    for name, mod in m.named_modules():
        if name.endswith(("gate_proj", "up_proj", "down_proj")):
            mod.register_forward_hook(lambda *a, n=name: seen.append(n))
    _run(m, _x())
    assert seen == []                                               # the PEFT modules' own forwards never ran


def test_disable_restores_the_peft_path_bit_for_bit():
    ref = _model()
    m = copy.deepcopy(ref)
    dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR)
    assert dlf.disable_dense_mlp_lora_fuse(m) == 2
    y0, dx0, g0 = _run(ref, _x())
    y1, dx1, g1 = _run(m, _x())
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1) and all(torch.equal(g0[n], g1[n]) for n in g0)
    assert not any(hasattr(mod, "_e4b_dense_lora") or "forward" in mod.__dict__ for mod in m.modules()
                   if hasattr(mod, "gate_proj"))


def test_checkpoint_recompute_gives_the_same_gradients():
    m = _model()
    dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR)
    _y, dx0, g0 = _run(m, _x())
    for p in m.parameters():
        p.grad = None
    layers = list(m.layers)

    def ck(x):
        for layer in layers:
            x = x + torch.utils.checkpoint.checkpoint(layer["mlp"], x, use_reentrant=False)
        return x
    x = _x().requires_grad_(True)
    (ck(x).float() ** 2).mean().backward()
    assert torch.equal(x.grad, dx0)
    assert all(torch.equal(p.grad, g0[n]) for n, p in m.named_parameters() if p.grad is not None)


@pytest.mark.parametrize("kw,why", [
    ({"dropout": 0.1}, "dropout is not 0"),
    ({"use_dora": True}, "DoRA"),
])
def test_registered_refusals_keep_the_peft_path(kw, why):
    m = _model(**kw)
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 0
    assert dlf.DENSE_MLP_LORA_STATS["refused"] and all(why in r for r in dlf.DENSE_MLP_LORA_STATS["refused"].values())


def test_refuses_non_fp32_adapters_a_merged_adapter_extra_adapters_and_a_non_nf4_base():
    m = _model()
    m.layers[0]["mlp"].up_proj.lora_A["default"].weight.data = m.layers[0]["mlp"].up_proj.lora_A["default"].weight.data.half()
    m.layers[1]["mlp"].down_proj.merged_adapters = ["default"]
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 0
    refused = dlf.DENSE_MLP_LORA_STATS["refused"]
    assert "not FP32" in refused["layers.0.mlp"] and "merged" in refused["layers.1.mlp"]
    m = _model()
    m.layers[0]["mlp"].gate_proj.cast_input_dtype_enabled = False       # what peft.helpers.disable_input_dtype_casting sets
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 3
    assert "input cast is disabled" in dlf.DENSE_MLP_LORA_STATS["refused"]["layers.0.mlp"]
    m2 = _model()
    peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, target_modules=["gate_proj"]), m2, adapter_name="other")
    for name, mod in m2.named_modules():                                     # the injection activated "other" everywhere
        if name.endswith(("gate_proj", "up_proj", "down_proj")):
            mod.set_adapter("default")
    m2.layers[0]["mlp"].gate_proj.set_adapter(["default", "other"])          # two active adapters on one projection
    m2.layers[0]["mlp"].gate_proj.lora_A["other"].float()
    m2.layers[0]["mlp"].gate_proj.lora_B["other"].float()
    assert dlf.enable_dense_mlp_lora_fuse(m2, base_ok=LINEAR) == 3
    assert "2 active adapters" in dlf.DENSE_MLP_LORA_STATS["refused"]["layers.0.mlp"]
    assert dlf.enable_dense_mlp_lora_fuse(_model()) == 0                # no NF4 Linear4bit on CPU: refused, never faked
    assert all("not a bitsandbytes NF4 Linear4bit" in r for r in dlf.DENSE_MLP_LORA_STATS["refused"].values())


class NotMistral(nn.Module):                                             # the same projections, another forward
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(64, 96), nn.Linear(64, 96), nn.Linear(96, 64)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) + self.up_proj(x))


def test_refuses_an_unlisted_class_a_patched_forward_and_a_hooked_projection():
    m = nn.ModuleDict({"a": NotMistral()})
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 0
    assert "NotMistral is not one of ['MistralMLP']" in dlf.DENSE_MLP_LORA_STATS["refused"]["a"]
    m = _model()
    m.layers[0]["mlp"].forward = m.layers[0]["mlp"].forward                 # an instance forward someone else set
    m.layers[1]["mlp"].up_proj.register_forward_pre_hook(lambda *a: None)
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 0
    refused = dlf.DENSE_MLP_LORA_STATS["refused"]
    assert "already patched" in refused["layers.0.mlp"] and "has a hook" in refused["layers.1.mlp"]
    m = _model()
    m.layers[0]["mlp"].down_proj.lora_B["default"].register_forward_hook(lambda *a: None)
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 3
    assert "down_proj: it or its adapter has a hook" in dlf.DENSE_MLP_LORA_STATS["refused"]["layers.0.mlp"]
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 3              # a second call counts the fused MLP


def test_under_autocast_a_fused_mlp_runs_the_class_forward_bit_for_bit():
    ref = _model()
    m = copy.deepcopy(ref)
    dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y0, dx0, g0 = _run(ref, _x())
        y1, dx1, g1 = _run(m, _x())
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1) and all(torch.equal(g0[n], g1[n]) for n in g0)
    assert dlf.DENSE_MLP_LORA_STATS["fallback"] == 2 and dlf.DENSE_MLP_LORA_STATS["calls"] == 0


@pytest.mark.parametrize("change", ["disabled", "other adapter", "input cast off"])
def test_a_peft_state_changed_after_fusing_runs_the_class_forward_bit_for_bit(change):
    ref = _model()
    peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, target_modules=["up_proj"]), ref, adapter_name="other")
    for name, mod in ref.named_modules():
        if name.endswith(("gate_proj", "up_proj", "down_proj")):
            mod.set_adapter("default")
    m = copy.deepcopy(ref)
    assert dlf.enable_dense_mlp_lora_fuse(m, base_ok=LINEAR) == 6
    for model in (ref, m):                                                  # the same change on both, after fusing
        up = model.layers[1]["mlp"].up_proj
        if change == "disabled":
            up.enable_adapters(False)
        elif change == "other adapter":
            up.set_adapter("other")
        else:
            up.cast_input_dtype_enabled = False
    if change == "input cast off":                                          # PEFT's own path refuses bf16 x on FP32 A ...
        with pytest.raises(RuntimeError, match="same dtype"):
            ref(_x())
        with pytest.raises(RuntimeError, match="same dtype"):               # ... and so does the fused model's fallback
            m(_x())
    else:
        assert torch.equal(ref(_x()), m(_x()))
    assert dlf.DENSE_MLP_LORA_STATS["fallback"] == 1 and dlf.DENSE_MLP_LORA_STATS["calls"] == 1


def test_the_knob_is_off_unless_set(monkeypatch):
    monkeypatch.delenv("E4B_DENSE_MLP_LORA_FUSE", raising=False)
    assert dlf.dense_mlp_lora_fuse_requested() is False
    for v, on in (("1", True), ("true", True), ("ON", True), ("0", False), ("", False), ("auto", False)):
        monkeypatch.setenv("E4B_DENSE_MLP_LORA_FUSE", v)
        assert dlf.dense_mlp_lora_fuse_requested() is on
