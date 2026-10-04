"""E4B_TRITON_PREBIND (opt-in): the prebound launch of the fused RMSNorm and rotary kernels is BIT-IDENTICAL to Triton's own
launch -- it launches the very compiled kernel Triton's lookup returns, across dtypes, odd sizes and misaligned pointers -- and
anything outside its contract (another Triton version, a launch hook, a callable grid, a changed global) takes Triton's path."""
import os
import subprocess
import sys

import pytest
import torch

from experts4bit_qlora.engines import triton_prebind as tp

try:
    import triton
    import triton.language as tl
except ImportError:                                    # pragma: no cover - macOS
    triton = tl = None

CUDA = torch.cuda.is_available() and triton is not None
SUPPORTED = triton is not None and tp._triton_version() in tp.SUPPORTED_TRITON
gpu = pytest.mark.skipif(not (CUDA and SUPPORTED), reason="needs CUDA and a Triton release the prebound path supports")


def test_off_unless_requested(monkeypatch):
    sentinel = object()
    monkeypatch.delenv("E4B_TRITON_PREBIND", raising=False)
    assert not tp.prebind_requested() and tp.prebind(sentinel) is sentinel
    monkeypatch.setenv("E4B_TRITON_PREBIND", "1")
    assert tp.prebind_requested()
    assert tp.prebind(sentinel) is sentinel            # not a Triton kernel (or no Triton at all): returned as is


@pytest.mark.skipif(triton is None, reason="needs Triton")
def test_unsupported_triton_version_keeps_tritons_launch(monkeypatch):
    from experts4bit_qlora.engines.rmsnorm_train import _rms_fwd
    if SUPPORTED:
        assert isinstance(tp.prebind(_rms_fwd, force=True), tp.Prebound)
    for v in ((3, 3), (3, 5), (3, 7), (4, 0), None):
        monkeypatch.setattr(tp, "_triton_version", lambda v=v: v)
        assert tp.prebind(_rms_fwd, force=True) is _rms_fwd


@pytest.mark.skipif(triton is None, reason="needs Triton")
def test_flag_binds_the_launchers_at_import():
    code = ("from experts4bit_qlora.engines import rmsnorm_train as r, rope_train as p\n"
            "from experts4bit_qlora.engines.triton_prebind import Prebound\n"
            "print([isinstance(f, Prebound) for f in (r._rms_fwd_launch, r._rms_bwd_launch, p._rope_launch)],"
            " r._rms_fwd_launch is r._rms_fwd)")
    for flag, want in (("1", f"[{SUPPORTED}, {SUPPORTED}, {SUPPORTED}] {not SUPPORTED}"), ("0", "[False, False, False] True")):
        env = dict(os.environ, E4B_TRITON_PREBIND=flag)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        assert out.strip().splitlines()[-1] == want


def _tritons_kernel(fn, args, kwargs):
    """The compiled kernel Triton's own ``run`` would launch for these arguments (its binder, key and per-device cache)."""
    from triton import knobs
    from triton.runtime.driver import driver
    kw = dict(kwargs)
    kw["debug"] = kw.get("debug", fn.debug) or knobs.runtime.debug
    if tp._triton_version() >= (3, 6):
        kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    caches = fn.device_caches[driver.active.get_current_device()]
    _, spec, opts = caches[-1](*args, **kw)
    if len(caches) == 5:                                # 3.6: (kernels, key cache, target, backend, binder)
        from triton.runtime.jit import compute_cache_key
        return caches[0].get(compute_cache_key(caches[1], spec, opts))
    return caches[0].get(str(spec) + str(opts))


def _rms(rt, base, skip, shape, w, off, f32, fwd_launch, bwd_launch, dy):
    """Forward and dx through ``rmsnorm_frozen`` with the given launchers, on ``base[skip:]`` (a view ``skip`` elements past an
    aligned allocation, so its pointer is misaligned for skip > 0)."""
    saved = rt._rms_fwd_launch, rt._rms_bwd_launch
    rt._rms_fwd_launch, rt._rms_bwd_launch = fwd_launch, bwd_launch
    try:
        b = base.detach().clone().requires_grad_(True)
        y = rt.rmsnorm_frozen(b[skip:].view(shape), w, 1e-6, off, f32)
        (g,) = torch.autograd.grad(y, b, dy)
        return y, g
    finally:
        rt._rms_fwd_launch, rt._rms_bwd_launch = saved


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("variant", [(0.0, False), (0.0, True), (1.0, False), (1.0, True)])
def test_rmsnorm_prebound_is_bit_identical(dtype, variant):
    from experts4bit_qlora.engines import rmsnorm_train as rt
    pf, pb = tp.Prebound(rt._rms_fwd), tp.Prebound(rt._rms_bwd)
    before = tp.PREBIND_STATS["prebound"]
    g = torch.Generator(device="cuda").manual_seed(0)
    # odd widths and row counts move Triton's divisibility specialization; offset views move the pointer off 16-byte alignment
    for rows, N in ((1, 7), (3, 33), (13, 100), (8, 127), (64, 128), (5, 1000), (4, 2047), (6, 2048), (2, 4096)):
        for skip in (0, 1, 3):
            base = torch.randn(rows * N + skip, generator=g, device="cuda").to(dtype)
            w = (torch.rand(N, generator=g, device="cuda") + 0.5).to(dtype)
            dy = torch.randn(rows, N, generator=g, device="cuda").to(dtype)
            for _ in range(2):                         # the first launch of a key fills the cache, the second is prebound
                ya, ga = _rms(rt, base, skip, (rows, N), w, *variant, rt._rms_fwd, rt._rms_bwd, dy)
                yb, gb = _rms(rt, base, skip, (rows, N), w, *variant, pf, pb, dy)
                assert torch.equal(ya, yb) and torch.equal(ga, gb), (rows, N, skip)
    assert tp.PREBIND_STATS["prebound"] > before


@gpu
@pytest.mark.parametrize("B,L,Hq,Hk,D", [(2, 190, 32, 4, 128), (1, 7, 4, 2, 64), (2, 33, 8, 8, 128), (1, 1, 2, 1, 32)])
def test_rope_prebound_is_bit_identical(B, L, Hq, Hk, D):
    from experts4bit_qlora.engines import rope_train as rp
    torch.manual_seed(0)
    q = torch.randn(B, L, Hq, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(B, L, Hk, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    ang = torch.randn(1, L, D // 2, device="cuda")
    ang = torch.cat((ang, ang), -1)
    cos, sin = ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    outs = []
    for launch in (rp._rope_fwd, tp.Prebound(rp._rope_fwd)):
        saved, rp._rope_launch = rp._rope_launch, launch
        try:
            for _ in range(2):
                qa, ka = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
                y = rp.rope_qk(qa, ka, cos, sin)
                outs.append(y + torch.autograd.grad(y, (qa, ka), (gq, gk)))
        finally:
            rp._rope_launch = saved
    for a, b in zip(outs[1], outs[3]):
        assert torch.equal(a, b)


@gpu
def test_prebound_launches_tritons_own_compiled_kernel():
    from experts4bit_qlora.engines import rmsnorm_train as rt
    p = tp.Prebound(rt._rms_fwd)
    for dtype in (torch.bfloat16, torch.float16):
        for rows, N in ((1, 2), (2, 16), (3, 17), (4, 48), (7, 1024), (1, 4096)):
            for skip in (0, 1, 8):
                base = torch.randn(rows * N + skip, device="cuda").to(dtype)
                x, w = base[skip:].view(rows, N), torch.ones(N, device="cuda", dtype=dtype)
                y, r = torch.empty_like(x), torch.empty(rows, device="cuda")
                args = (x, w, y, r, x.stride(0), N, 1e-6)
                kw = dict(ROWS=1, BLOCK=triton.next_power_of_2(N), num_warps=1, OFFSET=0.0, MUL_FP32=False)
                p[(rows,)](*args, **kw)                 # fills
                k = p[(rows,)](*args, **kw)             # prebound
                assert k is not None and k is _tritons_kernel(rt._rms_fwd, args, kw), (dtype, rows, N, skip)


@gpu
def test_launch_hook_and_callable_grid_take_tritons_path():
    from triton import knobs
    from experts4bit_qlora.engines import rmsnorm_train as rt
    p = tp.Prebound(rt._rms_fwd)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    y, r = torch.empty_like(x), torch.empty(4, device="cuda")
    kw = dict(ROWS=1, BLOCK=64, num_warps=1)
    p[(4,)](x, w, y, r, 64, 64, 1e-6, **kw)
    seen = []

    def hook(meta):
        seen.append(meta)

    chain = knobs.runtime.launch_enter_hook
    if hasattr(chain, "add"):                          # 3.6: a HookChain
        chain.add(hook)
    else:
        knobs.runtime.launch_enter_hook = hook
    try:
        before = tp.PREBIND_STATS["triton"]
        p[(4,)](x, w, y, r, 64, 64, 1e-6, **kw)
        torch.cuda.synchronize()
        assert seen and tp.PREBIND_STATS["triton"] == before + 1
    finally:
        if hasattr(chain, "remove"):
            chain.remove(hook)
        else:
            knobs.runtime.launch_enter_hook = None
    before = tp.PREBIND_STATS["triton"]
    p[lambda meta: (4,)](x, w, y, r, 64, 64, 1e-6, **kw)
    assert tp.PREBIND_STATS["triton"] == before         # a callable grid is Triton's own launch (not even counted)
    ref = torch.empty_like(y)
    rt._rms_fwd[(4,)](x, w, ref, r, 64, 64, 1e-6, **kw)
    assert torch.equal(y, ref)


if triton is not None:
    _SCALE = tl.constexpr(2)

    @triton.jit
    def _scaled_copy(X, Y, n, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        m = i < n
        tl.store(Y + i, tl.load(X + i, mask=m) * _SCALE, mask=m)


@gpu
def test_a_changed_global_is_refused_like_triton_refuses_it():
    global _SCALE
    p = tp.Prebound(_scaled_copy)
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    p[(1,)](x, y, 8, BLOCK=8)
    p[(1,)](x, y, 8, BLOCK=8)
    assert torch.equal(y, x * 2)
    if not _scaled_copy.used_global_vals:
        pytest.skip("this Triton does not record the kernel's globals")
    saved, _SCALE = _SCALE, tl.constexpr(3)
    try:
        with pytest.raises(RuntimeError, match="changed"):
            p[(1,)](x, y, 8, BLOCK=8)
    finally:
        _SCALE = saved


@gpu
def test_a_tensor_passed_by_keyword_is_never_cached():
    from experts4bit_qlora.engines import rmsnorm_train as rt
    p = tp.Prebound(rt._rms_fwd)
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(32, device="cuda", dtype=torch.bfloat16)
    y, r = torch.empty_like(x), torch.empty(2, device="cuda")
    for _ in range(3):
        p[(2,)](x, w, y, R=r, stride=32, N=32, eps=1e-6, ROWS=1, BLOCK=32, num_warps=1)
    assert not p.kernels
