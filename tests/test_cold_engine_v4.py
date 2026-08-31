"""DeepSeek-V4 on the cold engine — host-CPU compute of the cold tail.

`enable_cold_engine` allowlisted gpt-oss but not V4, and its host epilogue was a plain
SwiGLU, so V4 was skipped as a custom forward and the strong-CPU regime was unavailable to
it. Bugbot, PR #58. The host path now takes the module's own epilogue via `lora._epilogue`.

CPU-runnable end to end, like `test_cold_engine.py`: the all-cold configuration needs
neither CUDA nor the fused kernel, so CI exercises the real math instead of skipping.
"""
import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import disable_cold_engine, enable_cold_engine  # noqa: E402

E, HID, INTER, K, T = 6, 128, 64, 2, 8
LIMIT = 2.0          # far below the checkpoint's 10.0 so the clamp actually binds


def _mk_v4(limit=LIMIT, seed=0, compute_dtype=torch.float32, scale=1.2):
    from experts4bit_qlora.arch.deepseek_v4 import _DeepseekV4ForwardMixin
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, HID) * scale
    down = torch.randn(E, HID, INTER) * scale
    require_quantize("cpu")
    base = _DeepseekV4ForwardMixin.from_deepseek_v4(
        gate_up, down, limit=limit, quant_type="nf4", compute_dtype=compute_dtype)
    hs = torch.randn(T, HID) * 1.5
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    wts = torch.softmax(torch.randn(T, K), dim=-1)
    return base, hs, idx, wts


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def test_v4_is_eligible_for_the_cold_engine():
    base, *_ = _mk_v4()
    assert enable_cold_engine(base, hot_sets=[[]], device="cpu") == 1, \
        "V4 skipped as a custom forward — the strong-CPU regime was unavailable to it"
    disable_cold_engine(base)


def test_v4_all_cold_keeps_the_clamps():
    """Both sides are fp32 here, so this is a TIGHT comparison — unlike the GPU gates,
    nothing hides behind bf16. Close to the clamped reference, far from an unclamped one."""
    base, hs, idx, wts = _mk_v4(seed=1, limit=LIMIT)
    unclamped, *_ = _mk_v4(seed=1, limit=0.0)       # identical weights, clamping off
    ref = base(hs, idx, wts)
    ref_noclamp = unclamped(hs, idx, wts)
    assert _rel(ref, ref_noclamp) > 1e-2, "clamp barely binds; cannot discriminate"

    assert enable_cold_engine(base, hot_sets=[[]], device="cpu") == 1
    got = base(hs, idx, wts)
    disable_cold_engine(base)
    # same decoded values, fp32 math on both sides — only the reduction order differs
    assert _rel(got, ref) < 1e-5, _rel(got, ref)
    assert _rel(got, ref_noclamp) > 1e-2, "cold engine served UNCLAMPED SwiGLU"
    assert torch.equal(base(hs, idx, wts), ref), "disable must restore the stock forward"


def test_v4_cold_engine_limit_is_load_bearing():
    """A different limit must move the cold-engine output, proving the engine reads the
    module's own bound rather than a constant baked in at patch time."""
    a, hs, idx, wts = _mk_v4(seed=2, limit=0.5)
    b, *_ = _mk_v4(seed=2, limit=50.0)
    enable_cold_engine(a, hot_sets=[[]], device="cpu")
    enable_cold_engine(b, hot_sets=[[]], device="cpu")
    out_a, out_b = a(hs, idx, wts), b(hs, idx, wts)
    disable_cold_engine(a)
    disable_cold_engine(b)
    assert _rel(out_a, out_b) > 1e-2, "clamp bound had no effect through the cold engine"
