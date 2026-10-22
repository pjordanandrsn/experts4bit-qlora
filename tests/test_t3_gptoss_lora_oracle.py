# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""T3's premise for gpt-oss expert LoRA (USD 0, CPU): the oracle in ``tests/gptoss_lora_oracle.py`` matches
transformers' ``GptOssExperts``, forward AND backward, to 1e-5 in fp32, on one expert stack and on a 2-layer model.

The oracle is the function e4b's gpt-oss expert LoRA must compute, fixed here before any kernel work. What stood between
it and a trained gpt-oss expert, named (``test_gaps_one_and_two_are_closed`` pins the first two as closed by T3 step 2,
graded in ``test_t3_gptoss_expert_lora.py``; the third and fourth are documented on the code they name):

1. **Per-expert biases** (closed). ``ExpertsLoRA`` adds ``gate_up_bias`` / ``down_bias`` after each base projection on
   its unfused loop; ``assert_stock_epilogue`` admits exactly those buffers beside a hook for the adapter alone
   (``allow_bias=True``), and every other entry point still refuses them.
2. **The clamped GLU is not exposed** (closed). ``GptOssExperts4bit`` now hands it over as ``_apply_gate``, the hook
   ``ExpertsLoRA`` and the fused training paths (``engines/fast.py``) call, as DeepSeek-V4's clamped SwiGLU does.
3. **The fused training route applies no bias.** ``enable_fast_train`` skips gpt-oss (``engines/fast.py``'s
   eligibility note). grouped-nf4-gemm's grouped NF4 GEMM carries no bias (``kernel/nf4_grouped.py``'s header). Only
   the serving path adds gpt-oss's biases, per sorted row (``engines/hot_residency.py``).
4. **The interleave.** It is resolved at load: ``from_gptoss`` de-interleaves, so an adapter's gate_up B rows follow
   e4b's gate-block-then-up-block order. A conversion to the checkpoint's layout, or to grouped-nf4-gemm's MXFP4
   ``ExpertsMxfp4LoRA`` (interleaved), permutes those rows with ``interleave_rows``. ``test_layout_round_trips`` pins
   the mapping.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gptoss_lora_oracle import (  # noqa: E402
    LIMIT, GptOssExpertsLoRARef, deinterleave_rows, from_hf_experts_state, interleave_rows, to_hf_experts_state,
)

TOL = 1e-5


def _cfg(**kw):
    from transformers import GptOssConfig
    base = dict(vocab_size=97, hidden_size=48, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=12, num_local_experts=4, num_experts_per_tok=2, sliding_window=8,
                max_position_embeddings=64, layer_types=["sliding_attention", "full_attention"])
    base.update(kw)
    cfg = GptOssConfig(**base)
    cfg._attn_implementation = "eager"
    return cfg


def _rel(a, b) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def _hf_experts(cfg, seed, std=0.5):
    """A ``GptOssExperts`` with every tensor random (biases too), large enough that the clamp engages."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    g = torch.Generator().manual_seed(seed)
    m = GptOssExperts(cfg).float()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return m


def _route(cfg, x, seed):
    """transformers' own router on random weights: top-k of the logits, softmax over those k."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
    g = torch.Generator().manual_seed(seed)
    r = GptOssTopKRouter(cfg).float()
    with torch.no_grad():
        for p in r.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
        _, scores, idx = r(x)
    return idx, scores


def _set_lora(ref, seed, std=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (ref.gate_up_lora_A, ref.gate_up_lora_B, ref.down_lora_A, ref.down_lora_B):
            p.copy_(torch.randn(p.shape, generator=g) * std)


def _oracle_from(hf, **kw):
    return GptOssExpertsLoRARef(*from_hf_experts_state(dict(hf.state_dict())), **kw)


def _lora_grads_by_chain_rule(ref, g_gate_up, g_down):
    """d loss / d(A, B) from d loss / d(merged weight), e4b layout: dA = s B^T G, dB = s G A^T."""
    s = ref.scaling
    return {"gate_up_lora_A": s * torch.bmm(ref.gate_up_lora_B.transpose(1, 2), g_gate_up),
            "gate_up_lora_B": s * torch.bmm(g_gate_up, ref.gate_up_lora_A.transpose(1, 2)),
            "down_lora_A": s * torch.bmm(ref.down_lora_B.transpose(1, 2), g_down),
            "down_lora_B": s * torch.bmm(g_down, ref.down_lora_A.transpose(1, 2))}


def test_layout_round_trips():
    """The mapping to and from the checkpoint layout is exact, and its de-interleave is ``from_gptoss``'s own."""
    cfg = _cfg()
    hf = _hf_experts(cfg, 1)
    sd = dict(hf.state_dict())
    back = to_hf_experts_state(*from_hf_experts_state(sd))
    for k in sd:
        assert torch.equal(back[k], sd[k]), k
    w = torch.randn(3, 10, 5)
    assert torch.equal(interleave_rows(deinterleave_rows(w)), w)
    from experts4bit_qlora.arch.gptoss import GptOssExperts4bit
    e4b = GptOssExperts4bit.from_gptoss(sd["gate_up_proj"], sd["gate_up_proj_bias"], sd["down_proj"],
                                        sd["down_proj_bias"], quant_type="bf16", compute_dtype=torch.bfloat16)
    gu, gub, dn, dnb = from_hf_experts_state(sd)
    assert torch.equal(e4b.gate_up_bias, gub.to(torch.bfloat16))
    assert torch.equal(e4b.gate_up_proj.view(gu.shape), gu.to(torch.bfloat16))


def test_zero_lora_is_transformers_exactly():
    """With B = 0 the oracle is ``GptOssExperts``, biases, interleave, clamp and routing included."""
    cfg = _cfg()
    hf = _hf_experts(cfg, 2)
    x = torch.randn(40, cfg.hidden_size, generator=torch.Generator().manual_seed(3)) * 2
    idx, sc = _route(cfg, x, 4)
    ref = _oracle_from(hf)
    with torch.no_grad():
        assert _rel(ref(x, idx, sc), hf(x, idx, sc)) <= 1e-6


def test_the_clamp_and_routing_are_engaged():
    """The inputs the tests use exercise both clamps (and leave some lanes unclamped), and route every expert."""
    cfg = _cfg()
    hf = _hf_experts(cfg, 2)
    x = torch.randn(40, cfg.hidden_size, generator=torch.Generator().manual_seed(3)) * 2
    idx, _ = _route(cfg, x, 4)
    ref = _oracle_from(hf)
    gu = torch.cat([torch.nn.functional.linear(x, ref.gate_up[e]) + ref.gate_up_bias[e] for e in range(ref.num_experts)])
    i = gu.shape[-1] // 2
    gate, up = gu[:, :i], gu[:, i:]
    assert (gate > LIMIT).any() and (gate <= LIMIT).any()
    assert (up.abs() > LIMIT).any() and (up.abs() <= LIMIT).any()
    assert set(idx.unique().tolist()) == set(range(cfg.num_local_experts))


def test_lora_forward_and_backward_match_the_merged_transformers_stack():
    """LoRA active: the forward equals ``GptOssExperts`` holding the merged weights; every LoRA gradient equals the
    chain rule applied to that stack's weight gradients, and the input gradient matches, all to 1e-5."""
    cfg = _cfg()
    hf = _hf_experts(cfg, 5)
    x0 = torch.randn(40, cfg.hidden_size, generator=torch.Generator().manual_seed(6)) * 2
    idx, sc = _route(cfg, x0, 7)
    w_out = torch.randn(40, cfg.hidden_size, generator=torch.Generator().manual_seed(8))
    ref = _oracle_from(hf)
    _set_lora(ref, 9)
    x = x0.clone().requires_grad_(True)
    y = ref(x, idx, sc)
    (y * w_out).sum().backward()

    gu_m, dn_m = (t.detach().clone().requires_grad_(True) for t in ref.merged())
    x_ref = x0.clone().requires_grad_(True)
    state = to_hf_experts_state(gu_m, ref.gate_up_bias, dn_m, ref.down_bias)
    y_ref = torch.func.functional_call(hf, state, (x_ref, idx, sc))
    (y_ref * w_out).sum().backward()
    assert _rel(y, y_ref) <= TOL
    assert _rel(x.grad, x_ref.grad) <= TOL
    want = _lora_grads_by_chain_rule(ref, gu_m.grad, dn_m.grad)
    for name, g in want.items():
        assert _rel(getattr(ref, name).grad, g) <= TOL, name


def test_two_layer_model_matches_the_merged_transformers_model():
    """A 2-layer ``GptOssForCausalLM`` with the oracle (LoRA active) in every MoE block: logits, loss and every LoRA
    gradient match the same model holding the merged expert weights, to 1e-5."""
    from transformers import GptOssForCausalLM
    import copy
    cfg = _cfg()
    torch.manual_seed(10)
    base = GptOssForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in base.model.layers:
            for p in layer.mlp.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    ids = torch.randint(0, cfg.vocab_size, (2, 12), generator=torch.Generator().manual_seed(12))

    ours = copy.deepcopy(base)
    for p in ours.parameters():
        p.requires_grad_(False)
    refs = []
    for li, layer in enumerate(ours.model.layers):
        ref = _oracle_from(layer.mlp.experts)
        _set_lora(ref, 20 + li)
        layer.mlp.experts = ref
        refs.append(ref)
    out = ours(input_ids=ids, labels=ids)
    out.loss.backward()

    merged = copy.deepcopy(base)
    for p in merged.parameters():
        p.requires_grad_(False)
    leaves = []
    for layer, ref in zip(merged.model.layers, refs):
        gu_m, dn_m = (t.detach().clone().requires_grad_(True) for t in ref.merged())
        leaves.append((gu_m, dn_m))
        experts = layer.mlp.experts
        st = to_hf_experts_state(gu_m, ref.gate_up_bias, dn_m, ref.down_bias)
        layer.mlp.experts = _Functional(experts, st)
    out_m = merged(input_ids=ids, labels=ids)
    out_m.loss.backward()

    assert _rel(out.logits, out_m.logits) <= TOL
    assert abs(out.loss.item() - out_m.loss.item()) <= TOL * abs(out_m.loss.item())
    for ref, (gu_m, dn_m) in zip(refs, leaves):
        for name, g in _lora_grads_by_chain_rule(ref, gu_m.grad, dn_m.grad).items():
            assert _rel(getattr(ref, name).grad, g) <= TOL, name


class _Functional(torch.nn.Module):
    """``GptOssExperts`` run on an explicit state (the merged leaves), so gradients flow to them."""

    def __init__(self, experts, state):
        super().__init__()
        self.experts, self.state = experts, state

    def forward(self, *args):
        return torch.func.functional_call(self.experts, self.state, args)


def test_gaps_one_and_two_are_closed():
    """Gaps 1 and 2 in the module docstring, as the code stands after T3 step 2: the hook exists, the adapter wraps the
    stack, and the default contract (every caller but the adapter) still refuses its biases."""
    from experts4bit_qlora.arch.gptoss import GptOssExperts4bit
    from experts4bit_qlora.lora import EpilogueContractError, ExpertsLoRA, assert_stock_epilogue
    cfg = _cfg()
    sd = dict(_hf_experts(cfg, 13).state_dict())
    e4b = GptOssExperts4bit.from_gptoss(sd["gate_up_proj"], sd["gate_up_proj_bias"], sd["down_proj"],
                                        sd["down_proj_bias"], quant_type="bf16", compute_dtype=torch.bfloat16)
    assert hasattr(type(e4b), "_apply_gate")
    ExpertsLoRA(e4b, r=8, alpha=16)
    with pytest.raises(EpilogueContractError):
        assert_stock_epilogue(e4b)
