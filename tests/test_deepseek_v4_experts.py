"""DeepSeek-V4 expert epilogue: clamped SwiGLU, and specifically NOT gpt-oss's GLU.

The reference is a literal transcription of the checkpoint's own
``inference/model.py`` ``Expert.forward`` + ``MoE.forward``, so a structural mistake
in the fused path (wrong clamp side, wrong combination, router weight on the wrong
side of w2) shows up as a numeric disagreement rather than as plausible garbage.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from experts4bit_qlora.deepseek_v4 import (  # noqa: E402
    DEFAULT_SWIGLU_LIMIT,
    DeepseekV4Experts4bit,
    DeepseekV4ExpertsNbit,
    _DeepseekV4ForwardMixin,
)

# hidden_dim and intermediate_dim must both be multiples of the NF4 blocksize (64)
# so per-expert quantization blocks align with expert boundaries.
E, H, INTER, K, TOKENS = 6, 128, 64, 2, 12


def _weights(seed=0, scale=0.35):
    """Weights are rounded to bf16 at generation so the ``bf16`` storage base is
    LOSSLESS. Otherwise every comparison below carries ~0.4% of bf16 rounding and
    the tolerances would have to be loose enough to hide a real structural bug."""
    g = torch.Generator().manual_seed(seed)
    bf = lambda t: t.bfloat16().float()
    gate_up = bf(torch.randn(E, 2 * INTER, H, generator=g) * scale)   # [E, 2I, H]
    down = bf(torch.randn(E, H, INTER, generator=g) * scale)          # [E, H, I]
    x = bf(torch.randn(TOKENS, H, generator=g) * 1.5)
    logits = torch.randn(TOKENS, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    return gate_up, down, x, idx, torch.softmax(val, dim=-1)


def _ref(gate_up, down, x, idx, w, limit, *, two_sided_gate=False, gpt_oss=False):
    """inference/model.py Expert.forward + MoE.forward, transcribed."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            gate, up = F.linear(x[t].float(), gate_up[e].float()).chunk(2, dim=-1)
            if limit > 0:
                gate = gate.clamp(min=-limit, max=limit) if two_sided_gate else gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            h = (up + 1) * (gate * torch.sigmoid(gate * 1.702)) if gpt_oss else F.silu(gate) * up
            h = h * float(w[t, j])                     # before w2, as the reference does
            out[t] += F.linear(h, down[e].float())
    return out


def _build(gate_up, down, limit=DEFAULT_SWIGLU_LIMIT, quant_type="bf16", device="cpu"):
    return _DeepseekV4ForwardMixin.from_deepseek_v4(
        gate_up, down, limit=limit, quant_type=quant_type,
        compute_dtype=torch.float32,
    ).to(device)


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def test_matches_reference_expert_forward():
    gate_up, down, x, idx, w = _weights()
    mod = _build(gate_up, down)
    got = mod(x, idx, w)
    want = _ref(gate_up, down, x, idx, w, DEFAULT_SWIGLU_LIMIT)
    assert _rel(got, want) < 1e-5, _rel(got, want)


def test_rebinds_to_the_nbit_subclass():
    gate_up, down, *_ = _weights()
    assert isinstance(_build(gate_up, down), DeepseekV4ExpertsNbit)


def test_gate_clamp_is_one_sided():
    """gate clamps only from above; up clamps both ways. A two-sided gate must differ."""
    gate_up, down, x, idx, w = _weights(seed=3, scale=1.2)   # big enough to reach the limit
    got = _build(gate_up, down, limit=2.0)(x, idx, w)
    one = _ref(gate_up, down, x, idx, w, 2.0, two_sided_gate=False)
    two = _ref(gate_up, down, x, idx, w, 2.0, two_sided_gate=True)
    assert _rel(one, two) > 1e-3, "test weights never drive gate below -limit; raise scale"
    assert _rel(got, one) < 1e-5
    assert _rel(got, two) > 1e-3


def test_is_not_the_gptoss_glu():
    """Guards a refactor that collapses this into GptOssExperts4bit."""
    gate_up, down, x, idx, w = _weights(seed=4)
    got = _build(gate_up, down, limit=7.0)(x, idx, w)
    assert _rel(got, _ref(gate_up, down, x, idx, w, 7.0)) < 1e-5
    assert _rel(got, _ref(gate_up, down, x, idx, w, 7.0, gpt_oss=True)) > 1e-2


def test_limit_zero_disables_clamping():
    gate_up, down, x, idx, w = _weights(seed=5, scale=1.2)
    got = _build(gate_up, down, limit=0.0)(x, idx, w)
    assert _rel(got, _ref(gate_up, down, x, idx, w, 0.0)) < 1e-5
    assert _rel(got, _ref(gate_up, down, x, idx, w, 2.0)) > 1e-3


def test_limit_actually_binds():
    gate_up, down, x, idx, w = _weights(seed=6, scale=1.2)
    a = _build(gate_up, down, limit=0.5)(x, idx, w)
    b = _build(gate_up, down, limit=50.0)(x, idx, w)
    assert _rel(a, b) > 1e-2


def test_transposed_down_is_rejected():
    gate_up, down, *_ = _weights()
    with pytest.raises(ValueError, match="does not match gate_up"):
        _build(gate_up, down.transpose(1, 2).contiguous())


def test_odd_gate_up_is_rejected():
    gate_up, down, *_ = _weights()
    with pytest.raises(ValueError, match="not even"):
        _build(gate_up[:, :-1, :].contiguous(), down)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nf4_tracks_the_reference():
    pytest.importorskip("bitsandbytes")
    gate_up, down, x, idx, w = _weights(seed=7)
    mod = _DeepseekV4ForwardMixin.from_deepseek_v4(
        gate_up, down, limit=DEFAULT_SWIGLU_LIMIT,
        quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
    assert isinstance(mod, DeepseekV4Experts4bit)
    got = mod(x.to(torch.bfloat16).cuda(), idx.cuda(), w.to(torch.bfloat16).cuda()).cpu().float()
    want = _ref(gate_up, down, x, idx, w, DEFAULT_SWIGLU_LIMIT)
    # This is a STORAGE test, not an epilogue test. NF4 re-quantizes (~9% per weight,
    # through two projections), and measurement showed a cosine bound cannot tell the
    # epilogues apart at this scale anyway: the correct GLU scores 0.9858 and gpt-oss's
    # WRONG one scores 0.9676 on the same quantized weights — 0.018 apart, well inside
    # the quantization noise. At limit=10 the clamps rarely bind and the two GLUs are
    # simply similar functions. Correctness of the epilogue is proved exactly, at
    # <1e-5, by the bf16 tests above (`test_is_not_the_gptoss_glu` in particular);
    # what this test adds is that the NF4 path runs, rebinds, and stays in-family.
    cos = F.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
    assert cos > 0.97, f"cosine {cos}"
    assert _rel(got, want) < 0.35, _rel(got, want)
