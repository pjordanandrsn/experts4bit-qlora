# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SD2, Amendment 4 (``bench/sd2/PREREG-sd2.md``): the key-row gate's derivation, checked on CPU with real e4m3.

Keys are rotated by rotate-half RoPE (Qwen3's layout and base), quantized as the FP8 pool quantizes them (one scale
per 32-wide group, amax / 448, e4m3), and compared on pair 0 (dims 0 and D/2), which turns 1 rad per position at any
base. The registered bound: a verify key that differs from the T == 1 key only by its input drift eps (relative, on
pair 0) and by the two FP8 roundings sits within arcsin(eps + 2^-3 + 2^-10) of it; a key one position off sits within
the same of 1 rad. The decision boundary is 0.5 rad, the midpoint. These tests hold the quantizer to that bound."""
import importlib.util
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("this torch has no float8_e4m3fn", allow_module_level=True)

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sd2"
D, H, GROUP, BASE = 128, 4, 32, 1_000_000.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOX = _load("sd2_box_k", LANE / "sd2_box.py")
RED = _load("sd2_reduce_k", LANE / "sd2_reduce.py")


def _rope(x, pos):
    """Rotate-half RoPE: pair j = (j, j + D/2) turns pos * BASE^(-2j/D) rad. x [n, H, D]; pos [n]."""
    inv = 1.0 / (BASE ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = pos.double()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    x64 = x.double()
    rot = torch.cat([-x64[..., D // 2:], x64[..., :D // 2]], -1)
    return (x64 * cos + rot * sin).float()


def _fp8(x):
    """The pool's key quantization: one scale per 32-wide group (amax / 448), e4m3, dequantized to fp32."""
    g = x.reshape(*x.shape[:-1], D // GROUP, GROUP)
    s = (g.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
    return ((g / s).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)


def _pair0(k):
    hi = (D // 2 // GROUP) * GROUP
    return (torch.complex(k[..., 0], k[..., D // 2]),
            torch.maximum(k[..., :GROUP].abs().amax(-1), k[..., hi:hi + GROUP].abs().amax(-1)))


def _keys(n, gen):
    """Pre-RoPE keys with a spread of magnitudes across dims (as k_norm's per-dim weights give)."""
    scale = torch.exp(torch.randn(D, generator=gen) * 1.0)
    return torch.randn(n, H, D, generator=gen) * scale


def _drift(k, eps, gen):
    """Every pair-0 vector moved by exactly eps x its magnitude, in a random direction (the bound's worst case per
    cell); the other dims by a relative eps in random sign."""
    out = k * (1 + eps * torch.sign(torch.randn(k.shape, generator=gen)))
    z = torch.complex(k[..., 0], k[..., D // 2])
    phi = torch.rand(z.shape, generator=gen) * 2 * math.pi
    z2 = z + eps * z.abs() * torch.complex(phi.cos(), phi.sin())
    out[..., 0], out[..., D // 2] = z2.real, z2.imag
    return out


def _bound(eps):
    return math.asin(min(1.0, eps + 2.0 ** -3 + 2.0 ** -10))


@pytest.mark.parametrize("eps", [0.0, 2.0 ** -7, 0.05, 0.3])
def test_the_real_build_and_the_shifted_mutant_sit_inside_the_derived_bounds(eps):
    gen = torch.Generator().manual_seed(1313)
    n = 512
    pos = torch.randint(16, 4000, (n,), generator=gen)
    k = _keys(n, gen)
    zo, amax = _pair0(_fp8(_rope(k, pos)))
    zv, _ = _pair0(_fp8(_rope(_drift(k, eps, gen), pos)))
    zc, _ = _pair0(_fp8(_rope(_drift(k, eps, gen), pos + 1)))
    a_real, inf, _ = BOX.pair0_rotation(zo, zv, amax)
    a_mut, _, _ = BOX.pair0_rotation(zo, zc, amax)
    assert inf.float().mean() > 0.5, "most cells informative"
    b = _bound(eps)
    assert a_real[inf].abs().max() <= b + 1e-6, (float(a_real[inf].abs().max()), b)
    assert (a_mut[inf] - 1.0).abs().max() <= b + 1e-6, (float((a_mut[inf] - 1.0).abs().max()), b)
    if eps <= 0.3:                                        # eps + 2^-3 + 2^-10 <= sin(0.5): both sides of the boundary
        assert a_real[inf].abs().max() < RED.BOUND_ROT <= a_mut[inf].abs().min()


def test_the_boundary_is_the_midpoint_and_the_margin_condition_is_registered():
    assert RED.BOUND_ROT == 0.5
    assert BOX.INFORMATIVE == 2.0 ** -7
    eps_max = math.sin(0.5) - 2.0 ** -3 - 2.0 ** -10      # the drift both sides tolerate: about 0.353
    assert 0.35 < eps_max < 0.36
    assert _bound(0.0) < 0.13 and 1.0 - _bound(0.0) > 0.87
    assert abs(_bound(2.0 ** -6) - 0.142) < 5e-4              # layer 0's registered bound (eps0 <= 2^-6): 0.142 / 0.858


def test_an_uninformative_cell_is_left_out():
    zo = torch.tensor([1e-6 + 0j, 3.0 + 4.0j], dtype=torch.complex64)
    zv = torch.tensor([-1e-6 + 0j, 3.0 + 4.0j], dtype=torch.complex64)
    ang, inf, _ = BOX.pair0_rotation(zo, zv, torch.tensor([10.0, 10.0]))
    assert inf.tolist() == [False, True] and abs(float(ang[0])) > 3.0      # pi, but not counted
    summ = BOX.rot_summary([(ang.view(1, 2, 1), inf.view(1, 2, 1), torch.zeros(1, 2, 1))])
    assert summ["max_abs_angle"] == 0.0 and summ["informative"] == 1


def test_a_reversed_shift_reads_as_minus_one_radian():
    gen = torch.Generator().manual_seed(7)
    pos = torch.randint(16, 4000, (64,), generator=gen)
    k = _keys(64, gen)
    zo, amax = _pair0(_fp8(_rope(k, pos)))
    zm, _ = _pair0(_fp8(_rope(k, pos - 1)))
    ang, inf, _ = BOX.pair0_rotation(zo, zm, amax)
    assert (ang[inf] + 1.0).abs().max() <= _bound(0.0) + 1e-6
