"""The fused training RMSNorm (E4B_FUSED_RMSNORM=1): within one bf16 ulp of the Hugging Face composite on ~all elements, forward
and dx; frozen-weight norms only; centered variants and trainable weights are left alone; a vacuous enable refuses."""
import pytest
import torch
import torch.nn as nn

CUDA = torch.cuda.is_available()
triton = pytest.importorskip("triton") if CUDA else None


class FakeRMSNorm(nn.Module):                         # the Hugging Face Llama / Qwen formula, by structure
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dt)


class CenteredRMSNorm(FakeRMSNorm):                   # x_norm * (1 + weight): shares the name, not the formula
    def forward(self, x):
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (h * (1.0 + self.weight.float())).to(dt)


@pytest.mark.skipif(not CUDA, reason="the fused kernel needs CUDA")
@pytest.mark.parametrize("shape", [(380, 2048), (2, 190, 32, 128), (2, 190, 4, 128), (7, 96)])
def test_fused_matches_the_composite_to_one_ulp(shape):
    from experts4bit_qlora.engines.rmsnorm_train import rmsnorm_frozen
    torch.manual_seed(0)
    N = shape[-1]
    ref = FakeRMSNorm(N).cuda().to(torch.bfloat16)
    with torch.no_grad():
        ref.weight.copy_(1 + 0.1 * torch.randn(N))
    ref.weight.requires_grad_(False)
    x = (torch.randn(*shape, device="cuda") * 3).to(torch.bfloat16)
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = ref(xa), rmsnorm_frozen(xb, ref.weight, ref.variance_epsilon)
    g = torch.randn_like(ya)
    ya.backward(g)
    yb.backward(g)
    for a, b in ((ya.detach(), yb.detach()), (xa.grad, xb.grad)):
        d = (a.float() - b.float()).abs()
        ulp = torch.finfo(torch.bfloat16).eps * a.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
        assert (d > 0).float().mean().item() < 1e-3                 # nearly every element identical
        assert bool((d <= 2 * ulp + 1e-6).all())                     # and the rest one rounding step away


@pytest.mark.skipif(not CUDA, reason="the fused kernel needs CUDA")
def test_patcher_takes_frozen_norms_and_leaves_the_rest():
    from experts4bit_qlora.engines import rmsnorm_train as rt
    m = nn.ModuleDict({"a_RMSNorm": FakeRMSNorm(64), "b": FakeRMSNorm(64), "c": CenteredRMSNorm(64)}).cuda().to(torch.bfloat16)
    m["a_RMSNorm"].weight.requires_grad_(False)
    m["c"].weight.requires_grad_(False)
    with torch.no_grad():
        m["c"].weight.zero_()
    # b stays trainable -> not patched; c is centered -> patched with ITS formula, never the plain one
    n = rt.enable_fused_rmsnorm_train(m)
    assert n == 2 and getattr(m["a_RMSNorm"], "_e4b_rmsnorm_train", False)
    assert m["a_RMSNorm"]._e4b_rmsnorm_variant == (0.0, False)
    assert m["c"]._e4b_rmsnorm_variant == (1.0, True)
    assert not hasattr(m["b"], "_e4b_rmsnorm_train")
    before = rt.RMSNORM_TRAIN_STATS["calls"]
    m["a_RMSNorm"](torch.randn(3, 64, device="cuda", dtype=torch.bfloat16))
    assert rt.RMSNORM_TRAIN_STATS["calls"] == before + 1
    m["a_RMSNorm"](torch.randn(3, 64, device="cuda", dtype=torch.float32))   # fp32 input falls through to the composite
    assert rt.RMSNORM_TRAIN_STATS["calls"] == before + 1


@pytest.mark.skipif(not CUDA, reason="the fused kernel needs CUDA")
def test_vacuous_enable_refuses():
    from experts4bit_qlora.engines import rmsnorm_train as rt
    m = nn.ModuleDict({"x": FakeRMSNorm(64)}).cuda()                 # trainable weight only
    with pytest.raises(RuntimeError, match="patched no RMSNorm"):
        rt.enable_fused_rmsnorm_train(m)


def test_requested_defaults_on_and_explicit_is_separate(monkeypatch):
    from experts4bit_qlora.engines.rmsnorm_train import fused_rmsnorm_explicit, fused_rmsnorm_requested
    monkeypatch.delenv("E4B_FUSED_RMSNORM", raising=False)
    assert fused_rmsnorm_requested() and not fused_rmsnorm_explicit()     # on by default since TC1 amendment 15
    monkeypatch.setenv("E4B_FUSED_RMSNORM", "0")
    assert not fused_rmsnorm_requested()
    monkeypatch.setenv("E4B_FUSED_RMSNORM", "1")
    assert fused_rmsnorm_requested() and fused_rmsnorm_explicit()


@pytest.mark.skipif(not CUDA, reason="the patcher's probe runs the module on CUDA")
def test_default_on_skips_a_model_without_frozen_norms_quietly():
    from experts4bit_qlora.engines import rmsnorm_train as rt
    m = nn.ModuleDict({"x": FakeRMSNorm(64)}).cuda()                 # trainable weight only
    assert rt.enable_fused_rmsnorm_train(m, strict=False) == 0


class GemmaRMSNorm(FakeRMSNorm):                      # Gemma-4: pow(ms, -0.5) and the weight multiply in fp32, rounded once
    def forward(self, x):
        h = x.float()
        h = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon, -0.5)
        return (h * self.weight.float()).type_as(x)


@pytest.mark.skipif(not CUDA, reason="the fused kernel needs CUDA")
@pytest.mark.parametrize("cls,stored,want", [(CenteredRMSNorm, 0.0, (1.0, True)), (GemmaRMSNorm, 1.0, (0.0, True)),
                                             (FakeRMSNorm, 1.0, (0.0, False))])
@pytest.mark.parametrize("shape", [(380, 2048), (2, 190, 4, 128), (7, 96)])
def test_each_variant_matches_its_own_composite(cls, stored, want, shape):
    """The probe names each family's formula (centered Qwen3.5/3.6, fp32-multiply Gemma-4, Llama rounding), and the fused
    forward and dx match that composite to one bf16 rounding step on nearly every element."""
    from experts4bit_qlora.engines import rmsnorm_train as rt
    torch.manual_seed(1)
    N = shape[-1]
    mod = cls(N).cuda().to(torch.bfloat16)
    with torch.no_grad():
        mod.weight.copy_(stored + 0.1 * torch.randn(N))
    mod.weight.requires_grad_(False)
    ref = cls(N).cuda().to(torch.bfloat16)
    ref.load_state_dict(mod.state_dict())
    ref.weight.requires_grad_(False)
    holder = nn.ModuleDict({"n_RMSNorm": mod})
    assert rt.enable_fused_rmsnorm_train(holder) == 1 and mod._e4b_rmsnorm_variant == want
    x = (torch.randn(*shape, device="cuda") * 3).to(torch.bfloat16)
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = ref(xa), mod(xb)
    g = torch.randn_like(ya)
    ya.backward(g)
    yb.backward(g)
    for a, b in ((ya.detach(), yb.detach()), (xa.grad, xb.grad)):
        d = (a.float() - b.float()).abs()
        ulp = torch.finfo(torch.bfloat16).eps * a.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
        assert (d > 0).float().mean().item() < 1e-2
        assert bool((d <= 2 * ulp + 1e-6).all())
