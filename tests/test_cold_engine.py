"""Cold engine tests — host-CPU compute of the cold tail, anchored to the reference.

CPU-runnable end to end (the all-cold configuration needs neither CUDA nor the
fused kernel), so CI exercises the engine's actual math rather than skipping:

* ``test_torch_dequant_bit_exact_vs_bnb`` — the pure-torch NF4 decode against
  bitsandbytes' own CPU ``dequantize_4bit`` as the oracle, **bit-exact** (atol=0).
  This is the nibble-order / codebook / absmax-expansion gate.
* all-cold forward parity vs the stock ``ExpertsNbit`` reference forward, tight
  at fp32 compute (same decoded values, fp32 math both sides — only summation
  order differs) and loose only at bf16 (the reference matmuls in bf16).
* the gpt-oss epilogue (per-expert biases + clamped GLU) through the engine.
* backend forcing: ``dequant="torch"`` and ``dequant="bnb"`` agree.
* bookkeeping: enable/disable restore, hot_sets length validation, mutual
  exclusion with [fast], and the fp32-with-hot fallback gate.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    disable_cold_engine,
    enable_cold_engine,
    enable_fast,
    disable_fast,
)
from experts4bit_qlora.cold_engine import (  # noqa: E402
    _bnb_cpu_dequant_ok,
    _dequant_torch,
)

_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)

E, HID, INTER, T, K = 8, 128, 256, 5, 2


def _relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _mk(compute_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, HID)
    down = torch.randn(E, HID, INTER)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=compute_dtype)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bnb 4-bit quantize unavailable on this host: {e}")
    hs = torch.randn(T, HID)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    wts = torch.softmax(torch.randn(T, K), dim=-1)
    return base, hs, idx, wts


def test_torch_dequant_bit_exact_vs_bnb():
    if not _bnb_cpu_dequant_ok():
        pytest.skip("bnb CPU dequantize_4bit not functional here — no oracle")
    from bitsandbytes import functional as F
    torch.manual_seed(1)
    n, k = 6, 192
    w = torch.randn(n, k)
    q, st = F.quantize_4bit(w, blocksize=64, quant_type="nf4")
    oracle = F.dequantize_4bit(q, quant_state=st).float()
    got = _dequant_torch(q.reshape(n, k // 2), st.absmax.float(),
                         F.get_4bit_type("nf4", device="cpu").float(), n, k)
    assert torch.equal(got, oracle), "torch NF4 decode must be BIT-exact vs bnb CPU"


@pytest.mark.parametrize("dequant", ["torch", "bnb"])
def test_all_cold_matches_reference_fp32(dequant):
    if dequant == "bnb" and not _bnb_cpu_dequant_ok():
        pytest.skip("bnb CPU dequantize_4bit not functional here")
    base, hs, idx, wts = _mk(torch.float32)
    ref = base(hs, idx, wts)
    n = enable_cold_engine(base, hot_sets=[[]], device="cpu", dequant=dequant)
    assert n == 1
    assert base._cold_engine.dequant_backend == dequant
    got = base(hs, idx, wts)
    disable_cold_engine(base)
    # identical decoded values, fp32 math on both sides — only reduction order differs
    assert _relerr(got, ref) < 1e-5, _relerr(got, ref)
    assert torch.equal(base(hs, idx, wts), ref), "disable must restore the stock forward"


def test_all_cold_matches_reference_bf16():
    base, hs, idx, wts = _mk(torch.bfloat16)
    ref = base(hs.bfloat16(), idx, wts).float()
    enable_cold_engine(base, hot_sets=[torch.tensor([], dtype=torch.long)], device="cpu")
    got = base(hs.bfloat16(), idx, wts).float()
    disable_cold_engine(base)
    # decoded values round through bf16 on both sides; the engine then matmuls in
    # fp32 where the reference matmuls in bf16 — small, precision-only gap.
    assert _relerr(got, ref) < 2e-2, _relerr(got, ref)


def test_gptoss_epilogue_all_cold():
    gptoss_mod = pytest.importorskip("experts4bit_qlora.gptoss")
    torch.manual_seed(2)
    inter = 128
    gu_dense = torch.randn(E, HID, 2 * inter)   # input-major, interleaved
    gu_bias = torch.randn(E, 2 * inter)
    dn_dense = torch.randn(E, inter, HID)
    dn_bias = torch.randn(E, HID)
    try:
        mod = gptoss_mod.GptOssExperts4bit.from_gptoss(
            gu_dense, gu_bias, dn_dense, dn_bias, quant_type="nf4",
            compute_dtype=torch.float32)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bnb 4-bit quantize unavailable on this host: {e}")
    hs = torch.randn(T, HID)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    wts = torch.softmax(torch.randn(T, K), dim=-1)
    ref = mod(hs, idx, wts)
    n = enable_cold_engine(mod, hot_sets=[[]], device="cpu")
    assert n == 1
    got = mod(hs, idx, wts)
    disable_cold_engine(mod)
    assert _relerr(got, ref) < 1e-5, _relerr(got, ref)


def test_hot_sets_length_validated():
    base, *_ = _mk()
    with pytest.raises(ValueError, match="hot_sets has 2 entries"):
        enable_cold_engine(base, hot_sets=[[], []], device="cpu")


def test_mutual_exclusion_and_rebuild():
    base, hs, idx, wts = _mk()
    # [fast] wins if enabled first; cold engine must skip, not stack
    if enable_fast(base):
        assert enable_cold_engine(base, hot_sets=[[]], device="cpu") == 0
        disable_fast(base)
    assert enable_cold_engine(base, hot_sets=[[]], device="cpu") == 1
    st1 = base._cold_engine
    # re-enable rebuilds (never a stale cache) without stacking wrappers
    assert enable_cold_engine(base, hot_sets=[[]], device="cpu") == 1
    assert base._cold_engine is not st1
    ref_fwd = base._e4b_cold_ref
    assert disable_cold_engine(base) == 1
    assert base.forward == ref_fwd


def test_fp32_with_nonempty_hot_falls_back():
    # a non-empty hot set at fp32 compute must fall back to the reference (the
    # fused kernel takes bf16/fp16) — the wrapper's gate, not a crash. The state
    # build itself is CPU-side here (device="cpu"), so no CUDA is needed; the
    # kernel import requirement is what we bypass by monkeypatching the gate
    # check: without nf4_grouped installed, enable must raise ImportError.
    base, hs, idx, wts = _mk(torch.float32)
    try:
        import nf4_grouped  # noqa: F401
        have_kernel = True
    except ImportError:
        have_kernel = False
    if not have_kernel:
        with pytest.raises(ImportError, match="nf4_grouped"):
            enable_cold_engine(base, hot_sets=[[0, 1]], device="cpu")
        return
    enable_cold_engine(base, hot_sets=[[0, 1]], device="cpu")
    ref = base._e4b_cold_ref(hs, idx, wts)
    got = base(hs, idx, wts)  # fp32 + hot -> reference path
    disable_cold_engine(base)
    assert torch.equal(got, ref)
