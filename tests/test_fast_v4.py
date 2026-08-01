"""`[fast]` must not drop DeepSeek-V4's clamps.

V4 loads INSIDE an `ExpertsLoRA` (that is what `load_moe_4bit_streaming` produces), and
both fused paths re-implement the expert math rather than calling the base. `enable_fast`'s
wrapper loop only checked the WRAPPER's forward, and `enable_fast_train` had no eligibility
gate at all — so a V4 base reached the kernel and was served a plain `act_fn(gate) * up`
while the frozen reference applied `silu(clamp(gate)) * clamp(up)`. Bugbot, PR #58, High.

The fix routes every fused epilogue through `lora._epilogue`, the same hook
`ExpertsLoRA.forward` uses, so the two cannot drift. These tests pin that, and pin that
gpt-oss — whose forward also adds per-expert biases no fused path applies — stays skipped.

CUDA + nf4_grouped required.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("nf4_grouped")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import disable_fast, enable_fast  # noqa: E402
from experts4bit_qlora.lora import ExpertsLoRA  # noqa: E402

E, H, INTER, K, TOKENS = 8, 128, 64, 3, 24
LIMIT = 2.0          # far below the checkpoint's 10.0 so the clamp actually binds here


def _v4_lora(limit=LIMIT, seed=0, r=8, w_scale=0.35, x_scale=1.5):
    """A V4 expert stack wrapped exactly as the streaming loader wraps it."""
    from experts4bit_qlora.deepseek_v4 import _DeepseekV4ForwardMixin
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, H, generator=g) * w_scale
    down = torch.randn(E, H, INTER, generator=g) * w_scale
    base = _DeepseekV4ForwardMixin.from_deepseek_v4(
        gate_up, down, limit=limit, quant_type="nf4",
        compute_dtype=torch.bfloat16).cuda()
    mod = ExpertsLoRA(base, r=r, alpha=2 * r, dtype=torch.bfloat16).cuda().eval()
    x = (torch.randn(TOKENS, H, generator=g) * x_scale).to(torch.bfloat16).cuda()
    logits = torch.randn(TOKENS, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    w = torch.softmax(val, dim=-1).to(torch.bfloat16).cuda()
    return mod, x, idx.cuda(), w


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


TOL = 1.5e-2


def test_fast_patches_v4_inside_lora():
    mod, *_ = _v4_lora()
    assert enable_fast(mod) == 1, "V4-in-ExpertsLoRA should be fused, not skipped"
    disable_fast(mod)


def _v4_bare(limit=LIMIT, seed=0, w_scale=0.35, x_scale=1.5):
    from experts4bit_qlora.deepseek_v4 import _DeepseekV4ForwardMixin
    g = torch.Generator().manual_seed(seed)
    mod = _DeepseekV4ForwardMixin.from_deepseek_v4(
        torch.randn(E, 2 * INTER, H, generator=g) * w_scale,
        torch.randn(E, H, INTER, generator=g) * w_scale,
        limit=limit, quant_type="nf4", compute_dtype=torch.bfloat16).cuda().eval()
    x = (torch.randn(TOKENS, H, generator=g) * x_scale).to(torch.bfloat16).cuda()
    logits = torch.randn(TOKENS, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    return mod, x, idx.cuda(), torch.softmax(val, dim=-1).to(torch.bfloat16).cuda()


def test_fast_patches_bare_v4_and_keeps_the_clamps():
    """The bare loop had the same gap as the wrapper loop, the other way round: it checked
    `stock_forwards` and nothing else, so a BARE V4 module was skipped while the
    LoRA-wrapped one was fused. Both now accept a custom ACTIVATION via `_apply_gate`."""
    mod, x, idx, w = _v4_bare(seed=6, limit=LIMIT)
    unclamped, *_ = _v4_bare(seed=6, limit=0.0)
    with torch.no_grad():
        ref = mod(x, idx, w)
        ref_noclamp = unclamped(x, idx, w)
        assert _b_rel(ref, ref_noclamp) > 10 * TOL, "clamp barely binds"
        assert enable_fast(mod) == 1, "bare V4 skipped as a custom forward"
        got = mod(x, idx, w)
    assert _b_rel(got, ref) < TOL, _b_rel(got, ref)
    assert _b_rel(got, ref_noclamp) > 10 * TOL, "fused bare path served UNCLAMPED SwiGLU"
    disable_fast(mod)


def test_fast_keeps_the_clamps():
    """The discriminating test. A fused path that assumed plain SwiGLU would still match
    an UNCLAMPED reference, so assert against BOTH: close to clamped, far from unclamped."""
    mod, x, idx, w = _v4_lora(seed=1, limit=LIMIT)
    unclamped, *_ = _v4_lora(seed=1, limit=0.0)     # identical weights, clamping off
    with torch.no_grad():
        ref = mod(x, idx, w)
        ref_noclamp = unclamped(x, idx, w)
        assert _b_rel(ref, ref_noclamp) > 10 * TOL, (
            "clamp barely binds — this test cannot detect a dropped clamp")
        assert enable_fast(mod) == 1
        got = mod(x, idx, w)
    assert _b_rel(got, ref) < TOL, _b_rel(got, ref)
    assert _b_rel(got, ref_noclamp) > 10 * TOL, "fused path served UNCLAMPED SwiGLU"
    disable_fast(mod)


def test_fast_train_keeps_the_clamps():
    """Same for the differentiable path, which had no eligibility gate at all. Run under
    grad — that is the whole point of `enable_fast_train`, and the no-grad paths bail early."""
    pytest.importorskip("nf4_qlora")
    from experts4bit_qlora import disable_fast_train, enable_fast_train
    mod, x, idx, w = _v4_lora(seed=2, limit=LIMIT)
    unclamped, *_ = _v4_lora(seed=2, limit=0.0)
    mod.train()
    unclamped.train()
    ref = mod(x, idx, w)
    ref_noclamp = unclamped(x, idx, w)
    assert _b_rel(ref, ref_noclamp) > 10 * TOL, "clamp barely binds"

    assert enable_fast_train(mod) == 1
    got = mod(x, idx, w)
    assert got.requires_grad, "fused TRAINING path must stay differentiable"
    assert _b_rel(got, ref) < TOL, _b_rel(got, ref)
    assert _b_rel(got, ref_noclamp) > 10 * TOL, "fused train path served UNCLAMPED SwiGLU"
    got.float().sum().backward()                     # and the delta still gets a gradient
    assert mod.gate_up_lora_A.grad is not None
    disable_fast_train(mod)


def test_gptoss_inside_lora_is_still_skipped():
    """gpt-oss shares V4's clamps but ALSO carries per-expert biases, which no fused path
    here applies — `_apply_gate` cannot rescue it, so it must be skipped on both paths."""
    from experts4bit_qlora.gptoss import GptOssExperts4bit
    g = torch.Generator().manual_seed(3)
    base = GptOssExperts4bit.from_gptoss(
        torch.randn(E, H, 2 * INTER, generator=g) * 0.1,
        torch.randn(E, 2 * INTER, generator=g) * 0.05,
        torch.randn(E, INTER, H, generator=g) * 0.1,
        torch.randn(E, H, generator=g) * 0.05,
        alpha=1.702, limit=7.0, compute_dtype=torch.bfloat16).cuda()
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.bfloat16).cuda().eval()
    assert enable_fast(mod) == 0, "gpt-oss-in-LoRA was fused; its biases would be dropped"
    if pytest.importorskip("nf4_qlora"):
        from experts4bit_qlora import enable_fast_train
        assert enable_fast_train(mod) == 0, "fused TRAINING path fused gpt-oss"
