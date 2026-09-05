# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The stock-epilogue contract (#397): ``ExpertsLoRA`` refuses, by STRUCTURE, every
expert module whose forward it cannot re-implement, and every route that wraps or
attaches to a wrapper applies the same refusal.

The defect: ``ExpertsLoRA`` re-implements the expert math inline (the low-rank delta
must land before the nonlinearity), so it owns the epilogue. For gpt-oss -- interleaved
gate/up de-interleaved at load, per-expert biases, a clamped sigmoid GLU, no
``_apply_gate`` hook -- ``_epilogue`` fell back to ``silu(gate) * up`` with no biases and
no clamp: shapes agree, the loss falls, nothing raises. The arena loader wrapped it
anyway under ``arena_train=True`` and ``enable_nvme_train_residency`` attached to the
wrapper without looking.

What is asserted here:

* the contract enumerates gpt-oss's structure and refuses it -- the REAL
  ``GptOssExperts4bit`` / ``GptOssExpertsNbit`` classes, on a hidden-64, 2-expert
  fixture, in both NF4 and bf16 storage;
* the refusal is by structure, never by name: a stock class gains a bias buffer and is
  refused, loses it and is accepted; a custom forward is refused until it exposes
  ``_apply_gate``; an epilogue scalar no hook consumes is refused;
* the stock classes still wrap and take one training step: SiLU (OLMoE/Qwen3), Gemma-4's
  gelu_tanh with the dense MLP beside the routed experts, a non-gated stack, and V4's
  hooked clamp;
* every route refuses: the constructor, the loader's ``arena_train=True`` branch on a
  gpt-oss-shaped checkpoint end to end, ``enable_nvme_train_residency`` in its
  pre-flight (before a tier opens), ``enable_fast`` / ``enable_fast_train`` /
  ``enable_batched_train`` on a wrapper whose base was swapped in after construction,
  and ``enable_mxfp4_nvme_residency`` on a bias-carrying module.

CPU only. The NF4 arms ask ``quant_guard.require_quantize`` whether this host's
bitsandbytes can 4-bit quantize on CPU; the arena arms skip without grouped-nf4-gemm's
N-series modules -- and say so.
"""
from __future__ import annotations

import json
import os

import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    EpilogueContractError,
    Experts4bit,
    ExpertsLoRA,
    ExpertsNbit,
    assert_stock_epilogue,
    enable_batched_train,
    enable_fast,
)
from experts4bit_qlora.arch.deepseek_v4 import DeepseekV4Experts4bit  # noqa: E402
from experts4bit_qlora.arch.gptoss import GptOssExperts4bit, GptOssExpertsNbit  # noqa: E402
from experts4bit_qlora.lora import describe_epilogue_structure  # noqa: E402

# hidden 64 / intermediate 64: the NF4 blocksize (64) must divide both contracted dims.
E, H, INTER, K, TOKENS = 2, 64, 64, 1, 6


# ------------------------------------------------------------------ fixtures --

def _gptoss(quant_type="nf4", seed=11):
    """The gpt-oss expert stack as the loader builds it: input-major, INTERLEAVED gate/up
    with per-expert biases, alpha/limit, de-interleaved at load by ``from_gptoss``."""
    if quant_type in ("nf4", "fp4"):
        require_quantize("cpu", quant_type)
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(E, H, 2 * INTER, generator=g) / 4
    gate_up_bias = torch.randn(E, 2 * INTER, generator=g) / 4
    down = torch.randn(E, INTER, H, generator=g) / 4
    down_bias = torch.randn(E, H, generator=g) / 4
    return GptOssExperts4bit.from_gptoss(
        gate_up, gate_up_bias, down, down_bias,
        quant_type=quant_type, compute_dtype=torch.float32)


def _stock(activation=None, has_gate=True, seed=3):
    require_quantize("cpu")
    g = torch.Generator().manual_seed(seed)
    rows = 2 * INTER if has_gate else INTER
    gate_up = torch.randn(E, rows, H, generator=g) * 0.1
    down = torch.randn(E, H, INTER, generator=g) * 0.1
    return Experts4bit.from_float(gate_up, down, has_gate=has_gate, activation=activation,
                                  quant_type="nf4", compute_dtype=torch.float32)


def _v4(limit=2.0, seed=5):
    require_quantize("cpu")
    g = torch.Generator().manual_seed(seed)
    return DeepseekV4Experts4bit.from_deepseek_v4(
        torch.randn(E, 2 * INTER, H, generator=g) * 0.3,
        torch.randn(E, H, INTER, generator=g) * 0.3,
        limit=limit, quant_type="nf4", compute_dtype=torch.float32)


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(TOKENS, H, generator=g)
    logits = torch.randn(TOKENS, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    return x, idx, torch.softmax(val, dim=-1)


def _one_train_step(wrapper):
    """Forward, backward, optimiser step on the adapter alone. Returns the loss."""
    wrapper.train()
    params = [p for p in wrapper.parameters() if p.requires_grad]
    assert params and all("lora" in n for n, p in wrapper.named_parameters() if p.requires_grad)
    opt = torch.optim.SGD(params, lr=1e-2)
    x, idx, w = _inputs()
    loss = wrapper(x, idx, w).float().pow(2).mean()
    loss.backward()
    # B is zero-initialised, so dL/dB is the first signal that the delta is wired in.
    assert wrapper.down_lora_B.grad is not None and bool(wrapper.down_lora_B.grad.any())
    opt.step()
    assert torch.isfinite(loss)
    return float(loss.detach())


# ------------------------------------------------- the contract on gpt-oss --

def test_gptoss_structure_is_enumerated():
    """What the contract sees on the real class, as data: biases, scalars, a non-stock
    forward whose own body clamps, applies the sigmoid GLU and adds the biases."""
    d = describe_epilogue_structure(_gptoss("bf16"))
    assert d["class"] == "GptOssExpertsNbit"
    assert d["bias_tensors"] == ["down_bias", "gate_up_bias"]
    assert d["epilogue_scalars"] == ["alpha", "limit"]
    assert not d["stock_forward"] and not d["apply_gate_hook"]
    assert {"clamp", "sigmoid", "gate_up_bias", "down_bias"} <= set(d["forward_references"])
    # the de-interleave at load is what makes the LAYOUT stock even though nothing else is
    assert d["interleave_markers"] == [] and d["gate_up_shape"] == (2 * INTER, H)


@pytest.mark.parametrize("quant_type", ["nf4", "bf16"])
def test_expertslora_refuses_gptoss(quant_type):
    """The constructor is the seam every wrap goes through; it refuses the real class in
    the shipped storage (NF4) and in the bf16 structural-parity storage alike, and the
    message names the attributes and the faithful route."""
    base = _gptoss(quant_type)
    assert isinstance(base, (GptOssExperts4bit, GptOssExpertsNbit))
    with pytest.raises(EpilogueContractError) as ei:
        ExpertsLoRA(base, r=4, alpha=8)
    msg = str(ei.value)
    for name in ("gate_up_bias", "down_bias", "alpha", "limit", "_apply_gate"):
        assert name in msg, f"refusal does not name {name!r}"
    assert "ExpertsMxfp4LoRA" in msg and "mxfp4-moe-training-and-residency" in msg
    # `TypeError` on purpose: the quant-guard skip set catches NotImplementedError /
    # RuntimeError, and this refusal must never be reported as a green skip.
    assert isinstance(ei.value, TypeError) and not isinstance(ei.value, (RuntimeError, NotImplementedError))


def test_refusal_is_by_structure_not_by_name():
    """A stock class with a bias buffer bolted on is refused; take it off and it is
    accepted. Nothing about the class NAME enters into it."""
    base = _stock()
    assert_stock_epilogue(base)
    base.register_buffer("gate_up_proj_bias", torch.zeros(E, 2 * INTER))
    with pytest.raises(EpilogueContractError, match="gate_up_proj_bias"):
        assert_stock_epilogue(base)
    del base._buffers["gate_up_proj_bias"]
    assert_stock_epilogue(base)
    # a plain tensor attribute spelled with `bias` counts too
    base.some_bias = torch.zeros(E, H)
    with pytest.raises(EpilogueContractError, match="some_bias"):
        assert_stock_epilogue(base)
    del base.some_bias
    assert_stock_epilogue(base)


def test_custom_forward_is_refused_until_it_exposes_a_hook():
    """The V4 pattern, generically: a forward override is refused on its own and accepted
    once the epilogue is handed over through `_apply_gate`."""
    base = _stock()

    class _Custom(type(base)):
        def forward(self, *a, **kw):            # noqa: D401 - stands in for any override
            raise AssertionError("never called")

    base.__class__ = _Custom
    with pytest.raises(EpilogueContractError, match="_apply_gate"):
        assert_stock_epilogue(base)

    class _Hooked(_Custom):
        def _apply_gate(self, gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

    base.__class__ = _Hooked
    assert_stock_epilogue(base)


def test_epilogue_scalar_without_a_hook_is_refused():
    """`limit` on a stock forward: the adapter would ignore it, the serving engines
    (`hot_residency`) would clamp on it -- two answers for one module. With a hook the
    scalar is the hook's business (V4)."""
    base = _stock()
    base.limit = 7.0
    with pytest.raises(EpilogueContractError, match="limit"):
        assert_stock_epilogue(base)
    del base.limit
    assert_stock_epilogue(base)
    assert describe_epilogue_structure(_v4())["epilogue_scalars"] == ["limit"]
    assert_stock_epilogue(_v4())


def test_a_two_argument_act_fn_is_refused():
    """A GLU in disguise: an `act_fn` that wants (gate, up) is not something the stock
    epilogue applies to the gate alone."""
    base = _stock()
    base.act_fn = lambda gate, up: torch.nn.functional.silu(gate) * up
    with pytest.raises(EpilogueContractError, match="positional"):
        assert_stock_epilogue(base)


def test_an_interleaved_layout_marker_is_refused():
    base = _stock()
    base._e4b_interleaved = True
    with pytest.raises(EpilogueContractError, match="interleaved"):
        assert_stock_epilogue(base)


# ------------------------------------------------- the stock classes still work --

@pytest.mark.parametrize("build", [
    pytest.param(lambda: _stock(), id="silu"),
    pytest.param(lambda: _stock(activation=torch.nn.GELU(approximate="tanh")), id="gelu_tanh"),
    pytest.param(lambda: _stock(has_gate=False), id="nongated"),
    pytest.param(lambda: _v4(), id="v4_hooked"),
])
def test_stock_classes_wrap_and_train_one_step(build):
    base = build()
    assert_stock_epilogue(base)
    wrapper = ExpertsLoRA(base, r=4, alpha=8)
    _one_train_step(wrapper)


def test_gemma4_shape_wraps_and_trains():
    """Gemma-4's decoder block: routed experts (gelu_tanh) beside a DENSE `mlp`
    (gate/up/down `nn.Linear`), both summed into the residual. The contract sees only
    the expert module; the dense sibling is not its business and must not confuse it."""
    require_quantize("cpu")

    class _DenseMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(H, INTER, bias=False)
            self.up_proj = torch.nn.Linear(H, INTER, bias=False)
            self.down_proj = torch.nn.Linear(INTER, H, bias=False)
            self.act_fn = torch.nn.GELU(approximate="tanh")

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _DenseMlp()
            self.experts = ExpertsLoRA(
                _stock(activation=torch.nn.GELU(approximate="tanh")), r=4, alpha=8)

        def forward(self, x, idx, w):
            return x + self.mlp(x) + self.experts(x, idx, w)

    layer = _Layer()
    for p in layer.mlp.parameters():
        p.requires_grad_(False)
    layer.train()
    x, idx, w = _inputs()
    loss = layer(x, idx, w).pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert bool(layer.experts.down_lora_B.grad.any())
    assert all(p.grad is None for p in layer.mlp.parameters())


# ------------------------------------------------- every route refuses --

def _wrapper_with_swapped_base(quant_type="nf4"):
    """An `ExpertsLoRA` whose base VIOLATES the contract, reached the only way that is
    still possible: construct it over a stock base, then swap a gpt-oss stack of the same
    geometry underneath. This is the shape every enabler must refuse -- a wrapper is
    unfaithful on every path once its base is."""
    wrapper = ExpertsLoRA(_stock(), r=4, alpha=8)
    wrapper.base = _gptoss(quant_type)
    return wrapper


def test_enable_fast_refuses_a_wrapped_violating_base():
    with pytest.raises(EpilogueContractError, match="enable_fast"):
        enable_fast(_wrapper_with_swapped_base())


def test_enable_batched_train_refuses_a_wrapped_violating_base():
    wrapper = _wrapper_with_swapped_base()
    with pytest.raises(EpilogueContractError, match="enable_batched_train"):
        enable_batched_train(wrapper)
    assert not hasattr(wrapper, "_e4b_batched_ref"), "refused, yet patched"


def test_enable_fast_train_refuses_a_wrapped_violating_base():
    pytest.importorskip("nf4_qlora", reason="enable_fast_train returns 0 before its loop without the kernel")
    from experts4bit_qlora import enable_fast_train

    with pytest.raises(EpilogueContractError, match="enable_fast_train"):
        enable_fast_train(_wrapper_with_swapped_base())


def test_nvme_train_residency_refuses_in_the_preflight(tmp_path):
    """The attach the loader cannot be bypassed around: a violating base under a wrapper
    is refused before the tier opens, so nothing is stamped and nothing leaks."""
    pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
    pytest.importorskip("nvme_residency")
    from experts4bit_qlora.engines.nvme_train import enable_nvme_train_residency
    from test_nvme_train_residency import _bake

    wrapper = _wrapper_with_swapped_base("nf4")
    path, _index = _bake(wrapper.base, tmp_path, name="gptoss.arena")
    model = torch.nn.ModuleDict({"experts": wrapper})
    with pytest.raises(EpilogueContractError, match="arena-backed training"):
        enable_nvme_train_residency(model, path, hot_rows=E, device="cpu", pinned=False)
    assert not hasattr(wrapper.base, "_e4b_cold_tier"), "a tier was opened for a refused module"
    assert not hasattr(wrapper.base, "_e4b_arena_offload")


def test_mxfp4_nvme_residency_refuses_a_biased_module(tmp_path):
    """`enable_mxfp4_nvme_residency` hands the engine no biases and defaults to the V4
    epilogue; a bias-carrying module is refused on its structure before any engine or
    tier is built."""
    pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
    pytest.importorskip("mxfp4_residency", reason="needs grouped-nf4-gemm MXFP4 residency")
    from experts4bit_qlora.engines.nvme_experts import enable_mxfp4_nvme_residency
    from test_mxfp4_arena_train import _bake_mxfp4

    path, _index = _bake_mxfp4(tmp_path)
    model = torch.nn.ModuleDict({"experts": _gptoss("bf16")})
    with pytest.raises(RuntimeError, match="per-expert bias tensors"):
        enable_mxfp4_nvme_residency(model, path, k_slots=K, hot_rows=E, device="cpu")
    assert not hasattr(model["experts"], "_e4b_mxfp4_engine")


# ------------------------------------------------- the loader, end to end --

def _write_gptoss_ckpt(d, layers=2):
    """A gpt-oss-shaped checkpoint in the family's ON-DISK layout: MXFP4 blocks + e8m0
    scales for the fused expert stacks, per-projection biases, router bias, sinks."""
    from safetensors.torch import save_file
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssConfig, GptOssForCausalLM

    cfg = GptOssConfig(
        hidden_size=H, intermediate_size=INTER, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
        num_local_experts=E, num_experts_per_tok=K, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"][:layers],
        tie_word_embeddings=False)
    model = GptOssForCausalLM(cfg)
    g = torch.Generator().manual_seed(7)

    def u8(*shape):
        return torch.randint(0, 255, shape, generator=g, dtype=torch.uint8)

    def e8m0(*shape):
        # bounded exponents: a uniform draw dequantises half the stack to inf or 0
        return torch.randint(127 - 8, 127 + 8, shape, generator=g, dtype=torch.uint8)

    sd = {}
    for k, v in model.state_dict().items():
        if k.endswith("experts.gate_up_proj"):          # [E, H, 2I] -> blocks [E, 2I, H/32, 16]
            base = k[: -len("gate_up_proj")]
            sd[base + "gate_up_proj_blocks"] = u8(E, 2 * INTER, H // 32, 16)
            sd[base + "gate_up_proj_scales"] = e8m0(E, 2 * INTER, H // 32)
        elif k.endswith("experts.down_proj"):           # [E, I, H] -> blocks [E, H, I/32, 16]
            base = k[: -len("down_proj")]
            sd[base + "down_proj_blocks"] = u8(E, H, INTER // 32, 16)
            sd[base + "down_proj_scales"] = e8m0(E, H, INTER // 32)
        else:
            sd[k] = v.to(torch.float32).contiguous().clone()
    save_file(sd, os.path.join(d, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in sd}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    cfg.save_pretrained(d)
    return cfg


def test_loader_builds_gptoss_bare_and_refuses_arena_train(tmp_path):
    """The route the issue describes, end to end on CPU: the resident load builds the
    gpt-oss stacks BARE (no adapter, one NOTE); an `arena=` load carries the biases onto
    the meta stack; `arena_train=True` over that stack is refused with the layer and
    family named, instead of wrapping it."""
    pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
    pytest.importorskip("transformers.models.gpt_oss", reason="this transformers has no gpt_oss")
    from quant_guard import load_or_skip
    from test_loader_architectures import _bake_arena_for

    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_gptoss_ckpt(str(tmp_path))
    model, _cfg = load_or_skip(str(tmp_path), "cpu", torch.float32, r=4, alpha=8, what="gpt-oss nf4")
    stacks = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    assert stacks and all(isinstance(m, GptOssExperts4bit) for m in stacks)
    assert not [m for m in model.modules() if isinstance(m, ExpertsLoRA)], "gpt-oss must load bare"
    for m in stacks:
        with pytest.raises(EpilogueContractError):
            assert_stock_epilogue(m)

    arena_path = str(tmp_path / "gptoss.arena")
    n_layers = _bake_arena_for(model, arena_path)
    assert n_layers == len(stacks)

    # serving shape: bare meta stacks, biases carried (the regression #397's fix must keep)
    served, _ = load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8,
                                        arena=arena_path)
    meta = [m for m in served.modules() if isinstance(m, ExpertsNbit)]
    assert len(meta) == n_layers and all(m.gate_up_proj.is_meta for m in meta)
    assert all(hasattr(m, "gate_up_bias") and not m.gate_up_bias.is_meta for m in meta)
    assert not [m for m in served.modules() if isinstance(m, ExpertsLoRA)]

    # training shape: refused, by structure, naming the layer and the family
    with pytest.raises(EpilogueContractError) as ei:
        load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8,
                                arena=arena_path, arena_train=True)
    msg = str(ei.value)
    assert "layer 0" in msg and "gpt_oss" in msg and "arena_train=True" in msg
    assert "gate_up_bias" in msg and "ExpertsMxfp4LoRA" in msg
