"""DeepSeek-V4 on the pipelined engine — the one that supersedes `enable_hot_residency`.

V4 was allowlisted for hot residency but not here, so it was skipped as "custom forward"
and V4 residency worked only on the engine that warns it is deprecated. Bugbot, PR #58.

`_PipelinedResidency.step` now takes the module's own epilogue via `lora._epilogue`, so V4
needs no subclass — unlike gpt-oss, whose forward also adds per-expert biases and therefore
keeps `_GptOssPipelined`.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("nf4_grouped")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _no_triton_interpreter():
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("Triton interpreter mode active (raw-pointer gather is compiled-only)")


from experts4bit_qlora.pipelined import (  # noqa: E402
    disable_pipelined_residency,
    enable_pipelined_residency,
)

E, H, INTER, K = 8, 128, 64, 3
LIMIT = 2.0          # far below the checkpoint's 10.0 so the clamp actually binds


def _v4(limit=LIMIT, seed=0, w_scale=0.35, x_scale=1.5):
    """Bare V4 stack (this engine requires standalone modules) + a T=1 decode row."""
    from experts4bit_qlora.deepseek_v4 import _DeepseekV4ForwardMixin
    g = torch.Generator().manual_seed(seed)
    mod = _DeepseekV4ForwardMixin.from_deepseek_v4(
        torch.randn(E, 2 * INTER, H, generator=g) * w_scale,
        torch.randn(E, H, INTER, generator=g) * w_scale,
        limit=limit, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
    x = (torch.randn(1, H, generator=g) * x_scale).to(torch.bfloat16).cuda()
    logits = torch.randn(1, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    return mod, x, idx.cuda(), torch.softmax(val, dim=-1).to(torch.bfloat16).cuda()


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


TOL = 1.5e-2


def test_v4_is_eligible_for_the_pipelined_engine():
    _no_triton_interpreter()
    mod, *_ = _v4()
    assert enable_pipelined_residency(
        mod, [torch.tensor([0, 1, 2, 3])], device="cuda", k_slots=K) == 1, \
        "V4 skipped as a custom forward — it runs only on the deprecated engine"
    disable_pipelined_residency(mod)


@pytest.mark.parametrize("hot", [[], [0, 2, 5], list(range(E))])
def test_v4_pipelined_keeps_the_clamps(hot):
    """Close to the clamped reference AND far from an unclamped one, at every hot/cold
    split — a path that assumed plain SwiGLU would match the unclamped one and look fine."""
    _no_triton_interpreter()
    mod, x, idx, w = _v4(seed=1, limit=LIMIT)
    unclamped, *_ = _v4(seed=1, limit=0.0)      # identical weights, clamping off
    with torch.no_grad():
        ref = mod(x, idx, w)
        ref_noclamp = unclamped(x, idx, w)
        assert _b_rel(ref, ref_noclamp) > 10 * TOL, "clamp barely binds; cannot discriminate"
        assert enable_pipelined_residency(
            mod, [torch.tensor(hot, dtype=torch.long)], device="cuda", k_slots=K) == 1
        got = mod(x, idx, w)
    assert _b_rel(got, ref) < TOL, (hot, _b_rel(got, ref))
    assert _b_rel(got, ref_noclamp) > 10 * TOL, "pipelined path served UNCLAMPED SwiGLU"
    assert disable_pipelined_residency(mod) == 1
    with torch.no_grad():
        back = mod(x, idx, w)
    torch.testing.assert_close(back.float(), ref.float(), rtol=0, atol=0)
