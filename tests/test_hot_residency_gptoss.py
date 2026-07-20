"""gpt-oss hot-residency gate: the hot/cold fused path reproduces the bare
GptOssExperts4bit forward (clamped-GLU + per-expert biases) at every hot/cold
split. gpt-oss was previously SKIPPED by enable_hot_residency (custom forward);
this proves it is now supported. CUDA + nf4_grouped required."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import enable_hot_residency, disable_hot_residency  # noqa: E402


def _make_gptoss(E=8, H=128, inter=64, k=3, tokens=24, seed=0):
    """Synthetic gpt-oss expert stack via from_gptoss (interleaved input-major
    dense + biases + alpha/limit), quantized to NF4. Returns (mod, x, idx, w)."""
    from experts4bit_qlora.gptoss import GptOssExperts4bit
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(E, H, 2 * inter, generator=g) * 0.1   # [E, hidden, 2I] interleaved
    gate_up_bias = torch.randn(E, 2 * inter, generator=g) * 0.05
    down = torch.randn(E, inter, H, generator=g) * 0.1          # [E, inter, hidden]
    down_bias = torch.randn(E, H, generator=g) * 0.05
    mod = GptOssExperts4bit.from_gptoss(
        gate_up, gate_up_bias, down, down_bias,
        alpha=1.702, limit=7.0, compute_dtype=torch.bfloat16).cuda()
    x = (torch.randn(tokens, H, generator=g) * 0.5).to(torch.bfloat16).cuda()
    logits = torch.randn(tokens, E, generator=g)
    val, idx = torch.topk(logits, k, dim=-1)
    w = torch.softmax(val, dim=-1).to(torch.bfloat16).cuda()
    return mod, x, idx.cuda(), w


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def test_gptoss_is_now_eligible():
    """enable_hot_residency used to skip gpt-oss (custom forward); now patches it."""
    mod, x, idx, w = _make_gptoss()
    n = enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
    assert n == 1, "gpt-oss module should be patched, not skipped"
    disable_hot_residency(mod)


def test_gptoss_hot_matches_bare_reference():
    """Bare GptOss forward (clamped-GLU + biases) == hot-residency forward."""
    mod, x, idx, w = _make_gptoss(seed=1)
    with torch.no_grad():
        ref = mod(x, idx, w)
        enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
        got = mod(x, idx, w)
    assert _b_rel(got, ref) < 1.5e-2, _b_rel(got, ref)
    disable_hot_residency(mod)


def test_gptoss_all_hot_equals_all_cold():
    """K=E (all resident) and K=0 (all streamed) both reproduce the bare ref —
    isolates the epilogue+bias correctness from the residency split."""
    mod, x, idx, w = _make_gptoss(seed=2)
    with torch.no_grad():
        ref = mod(x, idx, w)
        for hot in (torch.arange(mod.num_experts), torch.tensor([], dtype=torch.long)):
            enable_hot_residency(mod, [hot], device="cuda")
            got = mod(x, idx, w)
            assert _b_rel(got, ref) < 1.5e-2, (hot.numel(), _b_rel(got, ref))
            disable_hot_residency(mod)


def test_gptoss_mixed_split_and_bias_matters():
    """A mixed hot/cold split matches; and zeroing biases changes the output
    (proving the bias path is live, not a no-op)."""
    mod, x, idx, w = _make_gptoss(seed=3)
    with torch.no_grad():
        ref = mod(x, idx, w)
        enable_hot_residency(mod, [torch.tensor([0, 2, 5])], device="cuda")
        got = mod(x, idx, w)
        assert _b_rel(got, ref) < 1.5e-2, _b_rel(got, ref)
        disable_hot_residency(mod)
        # bias is load-bearing: null it and the reference itself must move
        mod.gate_up_bias.zero_()
        mod.down_bias.zero_()
        ref0 = mod(x, idx, w)
    assert _b_rel(ref0, ref) > 1e-3, "biases had no effect — bias path suspect"
