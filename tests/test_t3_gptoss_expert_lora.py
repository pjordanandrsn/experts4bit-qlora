# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""T3 step 2 (USD 0, CPU): ``ExpertsLoRA`` over gpt-oss's experts, graded by the oracle in ``tests/gptoss_lora_oracle.py``.

``GptOssExperts4bit`` exposes its clamped GLU as ``_apply_gate``, and ``ExpertsLoRA`` adds the two per-expert biases right
after each base projection on its unfused loop (``assert_stock_epilogue(..., allow_bias=True)``). What is asserted:

* bf16-passthrough storage, fp32 compute: forward and backward (dx and every LoRA gradient) equal the oracle over the
  stack's own weights to 1e-5, through the expert-mask forward and the single-token decode forward, and in a 2-layer
  ``GptOssForCausalLM``; NF4 storage, the same bar against the oracle over the dequantized weights;
* the inputs engage both clamps with the LoRA delta in place;
* the ``_apply_gate`` refactor leaves the frozen forward bit-identical;
* ``allow_bias`` admits exactly the two buffers beside a hook, and every fused, batched, offload and arena entry point
  keeps refusing the biased stack;
* the loader builds gpt-oss bare by default, wraps it under ``E4B_GPTOSS_EXPERT_LORA=1``, and refuses the flag with
  offload.
"""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

from quant_guard import require_quantize  # noqa: E402
from gptoss_lora_oracle import LIMIT, GptOssExpertsLoRARef, clamped_glu  # noqa: E402
from test_t3_gptoss_lora_oracle import _cfg, _hf_experts, _rel, _route, _set_lora  # noqa: E402

from experts4bit_qlora import EpilogueContractError, ExpertsLoRA, assert_stock_epilogue  # noqa: E402
from experts4bit_qlora.arch.gptoss import GptOssExperts4bit  # noqa: E402
from experts4bit_qlora.lora import describe_epilogue_structure  # noqa: E402

TOL = 1e-5
LORA = ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B")


def _nf4_cfg():
    # the NF4 blocksize (64) must divide both contracted dims
    return _cfg(hidden_size=64, intermediate_size=64, head_dim=16)


def _base(hf, quant_type="bf16", compute_dtype=torch.float32):
    """The stack as the loader builds it: ``from_gptoss`` over a ``GptOssExperts`` state (interleaved, input-major)."""
    if quant_type in ("nf4", "fp4"):
        require_quantize("cpu", quant_type)
    sd = dict(hf.state_dict())
    return GptOssExperts4bit.from_gptoss(sd["gate_up_proj"], sd["gate_up_proj_bias"], sd["down_proj"],
                                         sd["down_proj_bias"], quant_type=quant_type, compute_dtype=compute_dtype)


def _oracle_over(base, r=8, lora_alpha=16):
    """The oracle on exactly the weights ``base`` computes with: each expert dequantized, biases as stored."""
    E = base.num_experts
    gu = torch.stack([base._dequantize_expert(base.gate_up_proj, getattr(base, "gate_up_absmax", None),
                                              base._gate_up_shape, e, torch.float32) for e in range(E)])
    dn = torch.stack([base._dequantize_expert(base.down_proj, getattr(base, "down_absmax", None),
                                              base._down_shape, e, torch.float32) for e in range(E)])
    return GptOssExpertsLoRARef(gu, base.gate_up_bias.float(), dn, base.down_bias.float(), r=r, lora_alpha=lora_alpha)


def _pair(hf, quant_type, seed):
    """``ExpertsLoRA`` over the e4b stack and the oracle over the same weights, holding the same random adapter."""
    base = _base(hf, quant_type)
    ref = _oracle_over(base)
    _set_lora(ref, seed)
    ours = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32)
    with torch.no_grad():
        for n in LORA:
            getattr(ours, n).copy_(getattr(ref, n))
    return ours, ref


def _inputs(cfg, seed, n=40):
    x = torch.randn(n, cfg.hidden_size, generator=torch.Generator().manual_seed(seed)) * 2
    idx, sc = _route(cfg, x, seed + 1)
    return x, idx, sc


# --------------------------------------------------------------- the forward and backward, graded --

@pytest.mark.parametrize("quant_type", ["bf16", "nf4"])
def test_one_stack_forward_and_backward_match_the_oracle(quant_type):
    """Training forward (the expert-mask loop): output, input gradient and every LoRA gradient equal the oracle's."""
    cfg = _cfg() if quant_type == "bf16" else _nf4_cfg()
    ours, ref = _pair(_hf_experts(cfg, 5), quant_type, 9)
    x0, idx, sc = _inputs(cfg, 6)
    w_out = torch.randn(x0.shape, generator=torch.Generator().manual_seed(8))

    ours.train()
    x = x0.clone().requires_grad_(True)
    (ours(x, idx, sc) * w_out).sum().backward()
    x_ref = x0.clone().requires_grad_(True)
    y_ref = ref(x_ref, idx, sc)
    (y_ref * w_out).sum().backward()

    with torch.no_grad():
        assert _rel(ours(x0, idx, sc), y_ref) <= TOL
    assert _rel(x.grad, x_ref.grad) <= TOL
    for n in LORA:
        assert _rel(getattr(ours, n).grad, getattr(ref, n).grad) <= TOL, n


def test_the_lora_inputs_engage_both_clamps():
    """With the delta in place, the gate passes the limit somewhere and stays under it elsewhere; so does |up|."""
    cfg = _cfg()
    _ours, ref = _pair(_hf_experts(cfg, 5), "bf16", 9)
    x, _, _ = _inputs(cfg, 6)
    with torch.no_grad():
        gu = torch.cat([ref._proj(x, ref.gate_up[e], ref.gate_up_bias[e], ref.gate_up_lora_A[e], ref.gate_up_lora_B[e])
                        for e in range(ref.num_experts)])
    i = gu.shape[-1] // 2
    gate, up = gu[:, :i], gu[:, i:]
    assert (gate > LIMIT).any() and (gate <= LIMIT).any()
    assert (up.abs() > LIMIT).any() and (up.abs() <= LIMIT).any()


@pytest.mark.parametrize("quant_type", ["bf16", "nf4"])
def test_eval_forwards_match_the_oracle(quant_type):
    """No-grad eval: the multi-token mask loop and the single-token decode forward both equal the oracle."""
    cfg = _cfg() if quant_type == "bf16" else _nf4_cfg()
    ours, ref = _pair(_hf_experts(cfg, 15), quant_type, 16)
    ours.eval()
    x, idx, sc = _inputs(cfg, 17)
    with torch.no_grad():
        assert _rel(ours(x, idx, sc), ref(x, idx, sc)) <= TOL
        for t in range(4):
            one = slice(t, t + 1)
            assert _rel(ours(x[one], idx[one], sc[one]), ref(x[one], idx[one], sc[one])) <= TOL, t


def test_two_layer_model_matches_the_oracle_model():
    """A 2-layer ``GptOssForCausalLM`` with ``ExpertsLoRA`` in every MoE block: logits, loss and every LoRA gradient
    equal the same model holding the oracle over the same weights and adapter, to 1e-5."""
    from transformers import GptOssForCausalLM
    cfg = _cfg()
    torch.manual_seed(10)
    base = GptOssForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in base.model.layers:
            for p in layer.mlp.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    for p in base.parameters():
        p.requires_grad_(False)
    ids = torch.randint(0, cfg.vocab_size, (2, 12), generator=torch.Generator().manual_seed(12))

    ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
    pairs = []
    for li, (lo, lr) in enumerate(zip(ours.model.layers, theirs.model.layers)):
        wrapped, ref = _pair(lo.mlp.experts, "bf16", 20 + li)
        lo.mlp.experts, lr.mlp.experts = wrapped, ref
        pairs.append((wrapped, ref))
    out, out_ref = ours(input_ids=ids, labels=ids), theirs(input_ids=ids, labels=ids)
    out.loss.backward()
    out_ref.loss.backward()

    assert _rel(out.logits, out_ref.logits) <= TOL
    assert abs(out.loss.item() - out_ref.loss.item()) <= TOL * abs(out_ref.loss.item())
    for wrapped, ref in pairs:
        for n in LORA:
            assert _rel(getattr(wrapped, n).grad, getattr(ref, n).grad) <= TOL, n


# --------------------------------------------------------------- the refactor --

def _pre_refactor_forward(mod, hidden_states, router_indices, router_scores):
    """``_GptOssForwardMixin.forward`` as it stood before ``_apply_gate``: the clamped GLU inline."""
    cd = mod.compute_dtype if mod.compute_dtype is not None else hidden_states.dtype
    x = hidden_states.to(cd)
    out = torch.zeros_like(x, dtype=torch.float32)
    mask = torch.nn.functional.one_hot(router_indices, num_classes=mod.num_experts).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)
    for e in hit:
        pos, tok = torch.where(mask[e])
        gate_up = mod._project(mod.gate_up_proj, mod.gate_up_absmax, mod._gate_up_shape, e, x[tok], cd) \
            + mod.gate_up_bias[e]
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=mod.limit)
        up = up.clamp(min=-mod.limit, max=mod.limit)
        gated = (up + 1) * (gate * torch.sigmoid(gate * mod.alpha))
        h = mod._project(mod.down_proj, mod.down_absmax, mod._down_shape, e, gated, cd) + mod.down_bias[e]
        out.index_add_(0, tok, (h * router_scores[tok, pos, None]).to(out.dtype))
    return out.to(hidden_states.dtype)


@pytest.mark.parametrize("quant_type,cd", [("bf16", torch.bfloat16), ("bf16", torch.float32), ("nf4", torch.float32)])
def test_the_frozen_forward_is_bit_identical_after_the_hook(quant_type, cd):
    cfg = _cfg() if quant_type == "bf16" else _nf4_cfg()
    mod = _base(_hf_experts(cfg, 30), quant_type, cd)
    x, idx, sc = _inputs(cfg, 31)
    with torch.no_grad():
        assert torch.equal(mod(x.to(cd), idx, sc.to(cd)), _pre_refactor_forward(mod, x.to(cd), idx, sc.to(cd)))
    t = torch.randn(9, 2 * cfg.intermediate_size, generator=torch.Generator().manual_seed(32)) * 9
    assert torch.equal(mod._apply_gate(t), clamped_glu(t, mod.alpha, mod.limit))


# --------------------------------------------------------------- the contract --

def test_the_contract_admits_exactly_the_two_buffers_beside_a_hook():
    base = _base(_hf_experts(_cfg(), 40))
    d = describe_epilogue_structure(base)
    assert d["apply_gate_hook"] and d["bias_buffers_representable"]
    assert d["bias_tensors"] == ["down_bias", "gate_up_bias"]
    assert d["forward_references"] == ["down_bias", "gate_up_bias"]   # the clamp and sigmoid moved into the hook
    assert_stock_epilogue(base, allow_bias=True)
    with pytest.raises(EpilogueContractError) as ei:
        assert_stock_epilogue(base)          # every caller but the adapter
    msg = str(ei.value)
    for name in ("gate_up_bias", "down_bias", "E4B_GPTOSS_EXPERT_LORA", "unmeasured", "ExpertsMxfp4LoRA"):
        assert name in msg, name
    assert isinstance(ei.value, TypeError) and not isinstance(ei.value, (RuntimeError, NotImplementedError))

    extra = _base(_hf_experts(_cfg(), 40))
    extra.register_buffer("router_bias", torch.zeros(extra.num_experts))
    wrong = _base(_hf_experts(_cfg(), 40))
    wrong.down_bias = torch.zeros(wrong.num_experts, 3)
    as_param = _base(_hf_experts(_cfg(), 40))
    b = as_param._buffers.pop("down_bias")
    as_param.down_bias = torch.nn.Parameter(b, requires_grad=False)
    for bad, what in ((extra, "router_bias"), (wrong, "down_bias"), (as_param, "down_bias")):
        assert not describe_epilogue_structure(bad)["bias_buffers_representable"]
        with pytest.raises(EpilogueContractError, match=what):
            assert_stock_epilogue(bad, allow_bias=True)
        with pytest.raises(EpilogueContractError):
            ExpertsLoRA(bad, r=4, alpha=8)


def test_biases_without_a_hook_are_still_refused():
    """A stock stack with the two buffers bolted on, right shapes, no hook: the adapter would apply a SwiGLU it was
    never told about, so ``allow_bias`` does not admit it."""
    require_quantize("cpu")
    from experts4bit_qlora import Experts4bit
    E, H, INTER = 2, 64, 64
    g = torch.Generator().manual_seed(41)
    base = Experts4bit.from_float(torch.randn(E, 2 * INTER, H, generator=g) * 0.1,
                                  torch.randn(E, H, INTER, generator=g) * 0.1,
                                  has_gate=True, quant_type="nf4", compute_dtype=torch.float32)
    base.register_buffer("gate_up_bias", torch.zeros(E, 2 * INTER))
    base.register_buffer("down_bias", torch.zeros(E, H))
    assert not describe_epilogue_structure(base)["bias_buffers_representable"]
    with pytest.raises(EpilogueContractError, match="gate_up_bias"):
        ExpertsLoRA(base, r=4, alpha=8)


# --------------------------------------------------------------- every other entry point refuses --

def _wrapped(quant_type="nf4"):
    return ExpertsLoRA(_base(_hf_experts(_nf4_cfg(), 50), quant_type), r=4, alpha=8)


def test_enable_fast_refuses_the_wrapper():
    from experts4bit_qlora import enable_fast
    with pytest.raises(EpilogueContractError, match="enable_fast"):
        enable_fast(_wrapped())


def test_enable_batched_train_refuses_the_wrapper():
    from experts4bit_qlora import enable_batched_train
    wrapper = _wrapped()
    with pytest.raises(EpilogueContractError, match="enable_batched_train"):
        enable_batched_train(wrapper)
    assert not hasattr(wrapper, "_e4b_batched_ref"), "refused, yet patched"


def test_enable_fast_train_refuses_the_wrapper():
    pytest.importorskip("nf4_qlora", reason="enable_fast_train returns 0 before its loop without the kernel")
    from experts4bit_qlora import enable_fast_train
    with pytest.raises(EpilogueContractError, match="enable_fast_train"):
        enable_fast_train(_wrapped())


def test_enable_expert_offload_refuses_the_wrapper():
    from experts4bit_qlora.engines.offload import enable_expert_offload
    wrapper = _wrapped()
    with pytest.raises(EpilogueContractError, match="gate_up_bias"):
        enable_expert_offload(wrapper, "cpu", pin=False)
    assert getattr(wrapper, "_offload", None) is None


def test_nvme_train_residency_refuses_the_wrapper(tmp_path):
    pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
    pytest.importorskip("nvme_residency")
    from experts4bit_qlora.engines.nvme_train import enable_nvme_train_residency
    from test_nvme_train_residency import _bake

    wrapper = _wrapped()
    path, _index = _bake(wrapper.base, tmp_path, name="gptoss.arena")
    with pytest.raises(EpilogueContractError, match="arena-backed training"):
        enable_nvme_train_residency(torch.nn.ModuleDict({"experts": wrapper}), path,
                                    hot_rows=wrapper.base.num_experts, device="cpu", pinned=False)
    assert not hasattr(wrapper.base, "_e4b_cold_tier")


# --------------------------------------------------------------- the loader --

def test_loader_wraps_gptoss_only_under_the_opt_in(tmp_path, monkeypatch):
    pytest.importorskip("transformers.models.gpt_oss", reason="this transformers has no gpt_oss")
    from quant_guard import load_or_skip
    from test_epilogue_contract import _write_gptoss_ckpt

    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_gptoss_ckpt(str(tmp_path))
    monkeypatch.delenv("E4B_GPTOSS_EXPERT_LORA", raising=False)
    bare, _ = load_or_skip(str(tmp_path), "cpu", torch.float32, r=4, alpha=8, what="gpt-oss nf4")
    assert not [m for m in bare.modules() if isinstance(m, ExpertsLoRA)], "the default must stay bare"

    monkeypatch.setenv("E4B_GPTOSS_EXPERT_LORA", "1")
    model, _ = load_or_skip(str(tmp_path), "cpu", torch.float32, r=4, alpha=8, what="gpt-oss nf4")
    wrapped = [m for m in model.modules() if isinstance(m, ExpertsLoRA)]
    stacks = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    assert wrapped and len(wrapped) == len(stacks)
    assert all(isinstance(w.base, GptOssExperts4bit) for w in wrapped)
    # it trains: the LoRA B gradients are live after one backward
    ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(51))
    model.train()
    model(input_ids=ids, labels=ids).loss.backward()
    assert all(w.down_lora_B.grad is not None and bool(w.down_lora_B.grad.any()) for w in wrapped)

    with pytest.raises(ValueError, match="E4B_GPTOSS_EXPERT_LORA=1"):
        load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8, offload=True)
