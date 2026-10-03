# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_PAGED_PREFILL_ATTN on a GPU at Qwen3-30B-A3B's attention shapes (e4b#960): Hq 32, Hkv 4, head_dim 128, bf16,
a 1,536-token prompt prefilled in three 512-token chunks through ``paged_attention_forward``.

- ``math`` launches no flash kernel (SDPA's math backend: the fp32 path P102 profiled);
- ``flash`` launches the flash forward kernel on every chunk;
- the routes agree within 1 % relative Frobenius. On an A2000 each route's error against an fp64 reference was
  0.0016-0.0023, and the bound here is a tripwire for a broken mask, not the quality gate (lane P107 gates quality
  on the model).

Skips without CUDA, or on a device without bf16 flash attention (sm < 80).
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    pytest.skip("needs a CUDA device with bf16 flash attention (sm_80+)", allow_module_level=True)

from experts4bit_qlora.engines import paged_attention as pa  # noqa: E402

HQ, HKV, D, CHUNK, N_CHUNKS = 32, 4, 128, 512, 3


class _Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.num_key_value_groups = HQ // HKV
        self.is_causal = True
        self.sliding_window = None


def _run(route, monkeypatch, q, k, v):
    from torch.profiler import ProfilerActivity, profile
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", route)
    ctx = pa.PagedAttentionContext(kv=None, slots=[0], mode="prefill")
    pa.set_context(ctx)
    outs = []
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for c in range(N_CHUNKS):
                s = slice(c * CHUNK, (c + 1) * CHUNK)
                o, _ = pa.paged_attention_forward(_Mod(), q[:, :, s], k[:, :, s], v[:, :, s], None,
                                                  scaling=D ** -0.5)
                outs.append(o)
            torch.cuda.synchronize()
    finally:
        pa.set_context(None)
    flash = sum(1 for e in prof.events()
                if str(getattr(e, "device_type", "")).endswith("CUDA") and "flash" in e.name.lower())
    return torch.cat(outs, dim=1), flash


def test_flash_route_reaches_the_flash_kernel_and_agrees_with_math(monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(0)
    T = CHUNK * N_CHUNKS
    q = torch.randn(1, HQ, T, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, HKV, T, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, HKV, T, D, device="cuda", dtype=torch.bfloat16, generator=g)
    y_math, flash_math = _run("math", monkeypatch, q, k, v)
    y_flash, flash_flash = _run("flash", monkeypatch, q, k, v)
    assert flash_math == 0, "the math route reached a flash kernel"
    assert flash_flash >= N_CHUNKS, f"the flash route launched {flash_flash} flash kernels for {N_CHUNKS} chunks"
    rel = float((y_flash.float() - y_math.float()).norm() / y_math.float().norm())
    assert rel < 0.01, rel
    print(f"rel flash vs math {rel:.5f}")
