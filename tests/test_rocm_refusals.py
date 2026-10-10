# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The vendor-first refusal map, on CPU with a mocked ROCm build.

A ROCm torch answers ``torch.cuda.get_device_capability`` with gfx numbers: MI300X (gfx942) is ``(9, 4)``, which passes
the sm_89 rule the fused FP8 KV append and the bucketed decode graphs were written against. Each gate must ask the
vendor first and refuse in words that name AMD. Every refusal is paired with the same call under an NVIDIA build at the
same capability tuple, which must not refuse on the vendor, so no test can pass because a gate refuses everything.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="engine imports need torch (CI installs CPU torch)")

from experts4bit_qlora import _rocm  # noqa: E402
from experts4bit_qlora import serve_paged  # noqa: E402
from experts4bit_qlora.engines import capture, fp8_paged_kv, paged_runner, spec_decode  # noqa: E402
from experts4bit_qlora.engines import triton_prebind as tp  # noqa: E402

GFX942 = (9, 4)  # MI300X, CDNA3


@pytest.fixture
def rocm(monkeypatch):
    """A ROCm build of torch on a gfx942 card, as far as every gate can see."""
    monkeypatch.setattr(torch.version, "hip", "6.4.43482", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: GFX942)
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    return monkeypatch


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: GFX942)
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    return monkeypatch


def test_rocm_helpers(rocm):
    why = _rocm.rocm_refusal("the widget")
    assert why and "AMD" in why and "the widget" in why and "6.4.43482" in why and "gfx number" in why
    assert _rocm.vendor_of(None) == "amd" and _rocm.vendor_of("nvidia") == "nvidia"
    with pytest.raises(ValueError, match="vendor"):
        _rocm.vendor_of("intel")
    rocm.setattr(torch.version, "hip", None)
    assert _rocm.rocm_refusal("the widget") is None and _rocm.vendor_of(None) == "nvidia"


def test_fused_append_refuses_amd_whatever_the_tuple():
    for cap in (GFX942, (9, 0), (11, 0), (12, 0), None):
        why = fp8_paged_kv.fused_append_unsupported(cap, vendor="amd")
        assert why and "AMD" in why, cap


def test_fused_append_control_nvidia_same_tuples_are_allowed():
    """Every tuple the AMD test above refuses, as an NVIDIA sm number: none is refused (and an sm_86 still is)."""
    for cap in (GFX942, (9, 0), (11, 0), (12, 0), None):
        assert fp8_paged_kv.fused_append_unsupported(cap, vendor="nvidia") is None, cap
    assert "sm_89" in fp8_paged_kv.fused_append_unsupported((8, 6), vendor="nvidia")


def test_fused_append_vendor_defaults_to_the_running_torch(rocm):
    assert "AMD" in fp8_paged_kv.fused_append_unsupported(GFX942)
    rocm.setattr(torch.version, "hip", None)
    assert fp8_paged_kv.fused_append_unsupported(GFX942) is None


def test_resolve_degrades_by_default_and_refuses_an_explicit_request_on_rocm(rocm):
    present = lambda: True  # noqa: E731
    assert fp8_paged_kv._resolve_fused_append(None, "cuda", present, GFX942) is False
    with pytest.raises(RuntimeError, match="E4B_FUSED_KV_APPEND=1 but AMD GPU"):
        fp8_paged_kv._resolve_fused_append("1", "cuda", present, GFX942)


def test_resolve_control_nvidia(nvidia):
    assert fp8_paged_kv._resolve_fused_append(None, "cuda", lambda: True, GFX942) is True
    assert fp8_paged_kv._resolve_fused_append("1", "cuda", lambda: True, GFX942) is True


def test_decode_graphs_auto_stays_eager_and_an_explicit_request_is_refused_on_rocm(rocm):
    assert serve_paged._graphs_env("auto", "cuda", "all-vram", GFX942) is False
    with pytest.raises(ValueError, match="AMD GPU"):
        serve_paged._graphs_env("1", "cuda", "all-vram", GFX942)
    assert serve_paged._graphs_env("0", "cuda", "all-vram", GFX942) is False


def test_decode_graphs_control_nvidia(nvidia):
    assert serve_paged._graphs_env("auto", "cuda", "all-vram", GFX942) is True
    assert serve_paged._graphs_env("1", "cuda", "all-vram", GFX942) is True


def _runner(**kw):
    fields = dict(device=torch.device("cuda", 0), linear_state=None, _verify_buckets=(), kv=SimpleNamespace(scratch=()))
    return SimpleNamespace(**{**fields, **kw})


def test_enable_decode_graphs_refuses_on_rocm(rocm):
    with pytest.raises(RuntimeError, match=r"AMD GPU .*bucketed decode graphs"):
        paged_runner.PagedModelRunner.enable_decode_graphs(_runner(), (1, 2))


def test_enable_decode_graphs_control_nvidia_reaches_its_own_checks(nvidia):
    with pytest.raises(ValueError, match="scratch_slots"):
        paged_runner.PagedModelRunner.enable_decode_graphs(_runner(), (1, 2))


def test_enable_prefill_graph_refuses_on_rocm(rocm):
    with pytest.raises(paged_runner.PrefillGraphRefused, match=r"AMD GPU .*prefill graph"):
        paged_runner.PagedModelRunner.enable_prefill_graph(_runner(), 16)


def test_enable_prefill_graph_control_nvidia_reaches_its_own_checks(nvidia, monkeypatch):
    from experts4bit_qlora.engines import hot_residency

    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [True])
    with pytest.raises(paged_runner.PrefillGraphRefused, match="linear-attention") as err:
        paged_runner.PagedModelRunner.enable_prefill_graph(_runner(linear_state=object()), 16)
    assert "AMD" not in str(err.value)


def test_capture_decode_refuses_on_rocm_before_touching_the_model(rocm):
    with pytest.raises(RuntimeError, match=r"AMD GPU .*capture_decode"):
        capture.capture_decode(None, None, 8)


class _Sentinel(Exception):
    """Raised by a mocked first CUDA-side step: the control reached past the vendor check without touching CUDA."""


def test_capture_decode_control_nvidia_reaches_its_first_step(nvidia, monkeypatch):
    def first_step(*a, **k):
        raise _Sentinel

    monkeypatch.setattr(capture, "_static_cache", first_step)
    with pytest.raises(_Sentinel):
        capture.capture_decode(None, None, 8)


def test_spec_post_stays_eager_on_rocm(rocm):
    drafter = SimpleNamespace(state={}, device=torch.device("cuda", 0), _post_graphs={})
    assert spec_decode.SpecDecoder.capture_post(drafter, 1) is None
    assert drafter._post_graphs == {}


def test_prebind_is_refused_on_rocm_even_when_forced(monkeypatch):
    if tp.triton is None:
        pytest.skip("triton is not installed: prebind returns the kernel unchanged regardless of vendor")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setattr(tp, "_triton_version", lambda: tp.SUPPORTED_TRITON[0])
    import triton
    import triton.language as tl

    @triton.jit
    def _k(x_ptr, N: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, N), tl.zeros((N,), tl.float32))

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert tp.prebind_unsupported() is None
    assert isinstance(tp.prebind(_k, force=True), tp.Prebound), "control: the same kernel IS prebound on NVIDIA"
    monkeypatch.setattr(torch.version, "hip", "6.4.43482")
    assert "AMD" in tp.prebind_unsupported()
    assert tp.prebind(_k, force=True) is _k
    monkeypatch.setenv("E4B_TRITON_PREBIND", "1")
    assert tp.prebind(_k) is _k


def test_spec_post_control_nvidia_reaches_its_capture(nvidia, monkeypatch):
    def first_step(*a, **k):
        raise _Sentinel

    monkeypatch.setattr(torch.cuda, "Stream", first_step)
    drafter = SimpleNamespace(state={}, device=torch.device("cuda", 0), _post_graphs={})
    with pytest.raises(_Sentinel):
        spec_decode.SpecDecoder.capture_post(drafter, 1)
