"""The training estimate prices (1) the expert absmax as the run stores it -- double-quantized under
``enable_fast_train``'s default (``absmax_dq.compress_expert_absmax_``), fp32 otherwise -- and (2) the LM loss branch
from the family the run will instantiate, whether or not the config names its architecture.

(1) LOWERS charges, so it is pinned against MEASURED bytes: real NF4 stacks quantized on CPU, compressed (or not) by the
code the run uses, and the estimator's bytes must equal the stored tensors' nbytes exactly. Layout: absmax_dq.py
(``_compress_one``: ``_q`` uint8 [n], ``_s`` fp32 per 256, ``_off`` fp32 [1], ``_code`` fp32 [256])."""
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experts4bit_qlora import Experts4bit, ExpertsNbit, QLoRASetup, describe_moe, estimate_qlora_footprint, recipe
from experts4bit_qlora.absmax_dq import compress_expert_absmax_, is_absmax_compressed
from experts4bit_qlora.engines import fast
from experts4bit_qlora.lora import ExpertsLoRA

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from quant_guard import require_quantize  # noqa: E402


def _real_model(E, H, inter, seed=0):
    require_quantize("cpu", "nf4")
    g = torch.Generator().manual_seed(seed)
    gu = torch.randn(E, 2 * inter, H, generator=g) * 0.05
    dn = torch.randn(E, H, inter, generator=g) * 0.05
    base = Experts4bit.from_float(gu, dn, quant_type="nf4", blocksize=64, compute_dtype=torch.float32)
    root = nn.Module()
    root.layers = nn.ModuleList([nn.Module()])
    root.layers[0].mlp = nn.Module()
    root.layers[0].mlp.experts = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32)
    return root


def _stored_bytes(model):
    """What the stacks actually hold: every parameter and buffer of each ExpertsNbit base."""
    bases = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    return sum(t.numel() * t.element_size() for b in bases for t in list(b.parameters()) + list(b.buffers()))


def _estimated_bytes(E, H, inter, setup):
    stack = SimpleNamespace(n_experts=E, hidden=H, intermediate=inter, first_name="gate_up_proj")
    base, _ = recipe._stack_modules(stack, setup)
    return recipe._stack_device_bytes(base, setup)


SHAPES = [(4, 128, 64), (3, 64, 64)]           # absmax counts 512 / 256 (multiples of 256) and 384 / 192 (not)
RESIDENT = QLoRASetup(expert_kernel="grouped_nf4", expert_residency="device", quant_type="nf4", blocksize=64)


@pytest.mark.parametrize("E,H,inter", SHAPES)
def test_fp32_absmax_bytes_are_the_stored_bytes(E, H, inter, monkeypatch):
    monkeypatch.setenv("E4B_ABSMAX_DQ", "0")
    model = _real_model(E, H, inter)
    assert not recipe._absmax_dq_in_force(RESIDENT)
    assert _estimated_bytes(E, H, inter, RESIDENT) == _stored_bytes(model)


@pytest.mark.parametrize("E,H,inter", SHAPES)
@pytest.mark.parametrize("how", ["switch=1", "unset default"])
def test_double_quantized_absmax_bytes_are_the_stored_bytes(E, H, inter, how, monkeypatch):
    if how == "switch=1":
        monkeypatch.setenv("E4B_ABSMAX_DQ", "1")
    else:                                          # the shipped default (tests/conftest.py pins it off for other tests)
        monkeypatch.delenv("E4B_ABSMAX_DQ", raising=False)
        monkeypatch.delenv("OFFLOAD_EXPERTS", raising=False)
        monkeypatch.delenv("TRAIN_ARENA", raising=False)
        monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)
    model = _real_model(E, H, inter)
    assert compress_expert_absmax_(model) == 1 and all(
        is_absmax_compressed(m) for m in model.modules() if isinstance(m, ExpertsNbit))
    assert recipe._absmax_dq_in_force(RESIDENT)
    assert _estimated_bytes(E, H, inter, RESIDENT) == _stored_bytes(model)


@pytest.mark.parametrize("setup", [
    QLoRASetup(expert_kernel="grouped_nf4", expert_residency="host"),     # offload keeps fp32 by name
    QLoRASetup(expert_kernel="reference", expert_residency="device"),     # no enable_fast_train
])
def test_paths_that_keep_fp32_are_priced_at_fp32(setup, monkeypatch):
    monkeypatch.setenv("E4B_ABSMAX_DQ", "1")
    assert not recipe._absmax_dq_in_force(setup)
    stack = SimpleNamespace(n_experts=4, hidden=128, intermediate=64, first_name="gate_up_proj")
    base, _ = recipe._stack_modules(stack, setup)
    assert recipe._stack_device_bytes(base, setup) == recipe._module_bytes(base)


def test_estimate_env_reports_the_absmax_switch(monkeypatch):
    monkeypatch.setenv("E4B_ABSMAX_DQ", "0")
    assert recipe.estimate_env()["E4B_ABSMAX_DQ"] is False
    monkeypatch.setenv("E4B_ABSMAX_DQ", "1")
    assert recipe.estimate_env()["E4B_ABSMAX_DQ"] is True


# ----------------------------------------------------------------------------- (2) the loss family

def _qwen3_moe_config(with_architectures):
    tr = pytest.importorskip("transformers")
    cfg = tr.AutoConfig.for_model("qwen3_moe", hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=64, num_experts=8, num_experts_per_tok=2,
                                  moe_intermediate_size=128, intermediate_size=512, vocab_size=151936,
                                  mlp_only_layers=[], decoder_sparse_step=1)
    if with_architectures:
        cfg.architectures = ["Qwen3MoeForCausalLM"]
    return cfg


def _activations(cfg, monkeypatch):
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)
    fp = estimate_qlora_footprint(describe_moe(cfg), QLoRASetup(expert_kernel="grouped_nf4"), tokens_per_microbatch=2048)
    return next(i for i in fp.items if i.name == "activations"), fp.unmodelled


def test_a_config_without_architectures_prices_the_loss_the_run_takes(monkeypatch):
    with_arch, _ = _activations(_qwen3_moe_config(True), monkeypatch)
    without, unmodelled = _activations(_qwen3_moe_config(False), monkeypatch)
    assert "chunked LM loss" in with_arch.detail                       # 2048 x 151936 x 4 >= 1 GiB: auto chunks
    assert without.bytes == with_arch.bytes and without.detail == with_arch.detail
    assert not any("LM loss branch" in u for u in unmodelled)


def test_an_unidentifiable_family_is_said_loudly_and_priced_at_the_stock_loss(monkeypatch):
    topo = describe_moe(_qwen3_moe_config(False))
    topo = type(topo)(**{**topo.__dict__, "model_type": "not_a_family", "architecture": None}) \
        if hasattr(topo, "__dataclass_fields__") else topo
    assert recipe._loss_family(topo) is None
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)
    fp = estimate_qlora_footprint(topo, QLoRASetup(expert_kernel="grouped_nf4"), tokens_per_microbatch=2048)
    act = next(i for i in fp.items if i.name == "activations")
    assert "logits and loss" in act.detail                             # the stock (larger) branch
    assert any("LM loss branch" in u and "not_a_family" in u for u in fp.unmodelled)
