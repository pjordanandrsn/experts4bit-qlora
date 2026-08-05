"""Parity contract for any batched/fused TRAINING expert path.

What was already covered before this file, and what was not:

* ``test_fast_lora.py`` covers ``enable_fast`` — the INFERENCE fused path — and
  asserts that grad-enabled forwards fall back exactly. It never runs the
  training path.
* ``test_fast_v4.py::test_fast_train_keeps_the_clamps`` runs ``enable_fast_train``
  and compares the FORWARD against the reference, but on the DeepSeek-V4 fixture
  only, and its gradient assertion is ``gate_up_lora_A.grad is not None``.
* ``grouped-nf4-gemm``'s ``kernel/test_nf4_qlora_grad.py`` checks the kernel's
  backward against *its own* dequant reference — not the composition through
  ``ExpertsLoRA``, which adds the base's ``_epilogue`` hook, the top-k weighting,
  and the fp32 scatter.

So gradient *values* through the full composition were unverified. That is a
silent-wrong failure mode, not a loud one: a backward that is wrong by a
constant factor still trains, still descends, and produces an adapter that is
quietly worse than the reference would have been. Nothing raises.

This file states the contract once and applies it to whatever training paths are
installed, so a new one (a candidate batched forward, an opt-in batched backward)
is covered by adding a single entry to ``CANDIDATES``:

    forward values, dL/dx, and dL/d(every LoRA parameter) must match the
    reference ``ExpertsLoRA.forward`` within reference noise, with a
    load-bearing (non-zero) adapter.

``test_contract_detects_a_wrong_gradient`` is the control: it runs a deliberately
perturbed path through the same assertions and requires them to FAIL. A parity
harness that has never rejected anything is not evidence.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
E, HID, INTER, TOP_K = 8, 128, 192, 2

# bf16 compute with fp32 adapters (the shipped training configuration). The fused path
# sums experts in group-sorted order rather than ascending expert id, so exact equality
# is not the contract — "within reference noise" is. These are set from measurement on
# an A2000, NOT loosened until the suite went green.
#
# Noise floor, both paths scored against an fp32-compute run over the SAME quantized
# weights (so only arithmetic rounding differs):
#
#     quantity          ref vs fp32    fused vs fp32    fused vs ref
#     forward             6.44e-03        5.68e-03        6.29e-03
#     dL/dx               6.54e-03        6.26e-03        5.06e-03
#     dL/d(lora params)   4.9-5.4e-03     4.6-5.1e-03     3.0-4.4e-03
#
# The fused path is nearer fp32 truth than the reference on every quantity, so the
# fused-vs-reference gap is two roundings of one value, not a discrepancy. (Per-op
# accuracy is not model-level accuracy — see the `fast.py` docstring, where a path that
# measured more accurate per-op cost +0.023% perplexity through 16 layers. Do not read
# this table as "the fused path is better"; read it as "both sit at the bf16 floor".)
#
# Sensitivity, from an injected LoRA-scaling error (worst channel):
#
#     +0.5%  ->  5.99e-03   below the noise floor: NOT detectable
#     +1.0%  ->  1.13e-02   detectable, on a parameter gradient only
#     +2.0%  ->  2.19e-02
#
# So GRAD_TOL sits between the real measurement (<=5.1e-3) and a 1% bug (1.13e-2).
# FWD_TOL matches the repo's existing precedent for this comparison (test_fast_v4.TOL)
# because the forward channel is demonstrably NOT the discriminating one — see
# test_forward_parity_alone_would_miss_a_gradient_bug.
FWD_TOL = 1.5e-2
GRAD_TOL = 1.0e-2


def _rel(got, want):
    """Relative error in the norm — scale-free, and unlike a max-abs check it does
    not let a large tensor hide a wrong row."""
    want = want.float()
    return ((got.float() - want).norm() / want.norm().clamp_min(1e-12)).item()


def _build(seed=0, scaling_bug=1.0):
    """An ``ExpertsLoRA`` over a real NF4 base. Same seed => bit-identical modules,
    which is what makes a gradient comparison between two of them meaningful."""
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4",
                                      compute_dtype=torch.bfloat16)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).to(DEVICE)
    # B is zero-initialised, which makes the delta identically zero — the adapter
    # could be dropped entirely and every assertion here would still pass. Give it
    # real values so the LoRA term is load-bearing in both the forward and the grads.
    with torch.no_grad():
        for p in (mod.gate_up_lora_B, mod.down_lora_B):
            p.normal_(0, 0.02)
    mod.scaling *= scaling_bug     # 1.0 except in the negative control
    return mod.train()


def _inputs(n_tok=64, seed=1):
    torch.manual_seed(seed)
    hs = torch.randn(n_tok, HID, dtype=torch.bfloat16, device=DEVICE)
    idx = torch.randint(0, E, (n_tok, TOP_K), device=DEVICE)
    wts = torch.rand(n_tok, TOP_K, dtype=torch.bfloat16, device=DEVICE)
    # A fixed random loss weighting, not .sum(): summing lets sign errors and
    # per-row cancellation pass, which is most of what a wrong scatter looks like.
    torch.manual_seed(seed + 100)
    loss_w = torch.randn(n_tok, HID, device=DEVICE)
    return hs, idx, wts, loss_w


def _forward_backward(mod, hs, idx, wts, loss_w):
    hs = hs.clone().requires_grad_(True)
    for p in mod.parameters():
        p.grad = None
    out = mod(hs, idx, wts)
    assert out.requires_grad, "a training path that returns a detached tensor trains nothing"
    (out.float() * loss_w).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in mod.named_parameters() if p.grad is not None}
    assert grads, "no parameter received a gradient"
    return out.detach(), hs.grad.detach().clone(), grads


def assert_parity(candidate_mod, reference_mod, fwd_tol=FWD_TOL, grad_tol=GRAD_TOL):
    """THE CONTRACT. Any training-path candidate must satisfy this against the
    reference ``ExpertsLoRA.forward``. Returns the measured errors so a caller can
    report them; raises AssertionError on violation."""
    # Comparing gradients across two modules is only meaningful if the modules are
    # identical to begin with — check rather than trust the seed.
    ref_p = dict(reference_mod.named_parameters())
    for n, p in candidate_mod.named_parameters():
        assert torch.equal(p.detach(), ref_p[n].detach()), f"fixtures differ at {n}"

    args = _inputs()
    ref_out, ref_dx, ref_grads = _forward_backward(reference_mod, *args)
    got_out, got_dx, got_grads = _forward_backward(candidate_mod, *args)

    measured = {"forward": _rel(got_out, ref_out), "dL/dx": _rel(got_dx, ref_dx)}
    assert measured["forward"] < fwd_tol, f"forward {measured['forward']:.3e}"
    assert measured["dL/dx"] < grad_tol, f"dL/dx {measured['dL/dx']:.3e}"

    assert set(got_grads) == set(ref_grads), (
        f"different parameters received gradients: "
        f"{sorted(set(got_grads) ^ set(ref_grads))}")
    for n in ref_grads:
        measured[f"dL/d{n}"] = _rel(got_grads[n], ref_grads[n])
        assert measured[f"dL/d{n}"] < grad_tol, f"dL/d{n} {measured[f'dL/d{n}']:.3e}"
    return measured


def _enable_fused_train(mod):
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    from experts4bit_qlora import enable_fast_train
    if enable_fast_train(mod) != 1:
        pytest.skip("enable_fast_train declined this module (ineligible base)")


def _enable_batched_train(mod):
    from experts4bit_qlora import enable_batched_train
    assert enable_batched_train(mod) == 1, "batched path declined an eligible module"


# name -> callable that patches a freshly built module in place. Add an entry to
# put a new training path under the contract; nothing else needs to change.
CANDIDATES = {
    "enable_fast_train": _enable_fused_train,        # grouped-nf4-gemm kernel lane
    "enable_batched_train": _enable_batched_train,   # kernel-free lane
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused training path is CUDA-only")
@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_training_path_matches_reference_forward_and_gradients(name):
    """The gap this file was written for: gradient VALUES through the full
    ExpertsLoRA composition, not just `grad is not None`."""
    reference = _build(seed=0)
    candidate = _build(seed=0)
    CANDIDATES[name](candidate)
    measured = assert_parity(candidate, reference)
    print(f"\n{name}: " + "  ".join(f"{k}={v:.2e}" for k, v in measured.items()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused training path is CUDA-only")
def test_contract_detects_a_wrong_gradient():
    """The control. A 1% error in the LoRA scaling is a plausible batching bug —
    alpha/r applied once too often, or on the wrong side of the epilogue — and it is
    exactly the kind that trains without ever raising. The contract must reject it; if
    this passes, every other assertion in this file is decoration.

    Honest about the floor: +0.5% measures 5.99e-3, under the bf16 noise floor, and this
    contract cannot see it. 1% is the smallest scaling error it is entitled to claim."""
    reference = _build(seed=0)
    mutant = _build(seed=0, scaling_bug=1.01)
    with pytest.raises(AssertionError):
        assert_parity(mutant, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused training path is CUDA-only")
def test_forward_parity_alone_would_miss_a_gradient_bug():
    """Why this file exists rather than one more forward check.

    The pre-existing coverage style for the training path is forward parity plus
    ``grad is not None``. Run that style against the 1% mutant: the forward comparison
    PASSES (5.6e-3, inside a tolerance the real path needs 6.3e-3 of), and the mutant
    does produce gradients — they are simply wrong. Only comparing gradient VALUES
    separates them, and the separation lives on the parameter grads, not on dL/dx."""
    reference = _build(seed=0)
    mutant = _build(seed=0, scaling_bug=1.01)
    args = _inputs()
    ref_out, ref_dx, ref_grads = _forward_backward(reference, *args)
    got_out, got_dx, got_grads = _forward_backward(mutant, *args)

    assert _rel(got_out, ref_out) < FWD_TOL, "control invalid: forward already catches it"
    assert all(g is not None for g in got_grads.values())
    worst_param = max(_rel(got_grads[n], ref_grads[n]) for n in ref_grads)
    assert worst_param > GRAD_TOL, (
        f"parameter gradients must be what separates them, got {worst_param:.3e}")


@pytest.mark.parametrize(
    "shape,blocksize,quant_type",
    [
        # Both in_features dims must be divisible by the blocksize — the constructor
        # refuses otherwise, and that refusal is precisely what makes the flattened
        # absmax exact, so only admissible combinations are meaningful here.
        ((8, 128, 192), 64, "nf4"),
        ((8, 128, 192), 64, "fp4"),
        ((4, 64, 64), 64, "nf4"),
        ((16, 256, 128), 64, "nf4"),
        ((16, 256, 128), 128, "nf4"),
        ((8, 128, 256), 128, "nf4"),
    ],
)
def test_whole_stack_dequant_equals_per_expert_loop(shape, blocksize, quant_type):
    """The precondition for the OTHER batching strategy — issue #38's: dequantize the
    whole expert stack once with a flattened QuantState instead of per expert.

    It holds bit-for-bit, and the reason is structural rather than lucky:
    ``_quantize_stack`` quantizes each expert with ``compress_statistics=False``, and
    the constructor refuses shapes where the blocksize does not divide an expert's
    rows — so blocks never straddle an expert boundary and the flattened absmax is an
    exact concatenation. Pinned here because it is a property of the STORAGE layout:
    if double-quant is ever enabled, or a shape with straddling blocks is admitted,
    a caller batching on this assumption starts reading neighbouring experts' scales
    and gets shaped, plausible garbage. This test is what fails first."""
    import bitsandbytes.functional as F
    from bitsandbytes.functional import QuantState

    n_exp, hidden, inter = shape
    torch.manual_seed(0)
    gate_up = (torch.randn(n_exp, 2 * inter, hidden) * 0.02).to(DEVICE)
    down = (torch.randn(n_exp, hidden, inter) * 0.02).to(DEVICE)
    try:
        mod = Experts4bit.from_float(gate_up, down, quant_type=quant_type,
                                     compute_dtype=torch.bfloat16, blocksize=blocksize).to(DEVICE)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")

    for packed, absmax, pshape in ((mod.gate_up_proj, mod.gate_up_absmax, mod._gate_up_shape),
                                   (mod.down_proj, mod.down_absmax, mod._down_shape)):
        out, inn = pshape
        per_expert = torch.stack([
            mod._dequantize_expert(packed, absmax, pshape, e, torch.bfloat16)
            for e in range(n_exp)
        ])
        state = QuantState(
            absmax=absmax.reshape(-1), shape=torch.Size((n_exp * out, inn)),
            dtype=torch.bfloat16, blocksize=mod.blocksize, quant_type=quant_type,
            code=F.get_4bit_type(quant_type, device=DEVICE),
        )
        whole = F.dequantize_4bit(packed.reshape(-1, 1), quant_state=state).reshape(n_exp, out, inn)
        assert torch.equal(per_expert, whole), (
            f"{pshape}: whole-stack dequant diverged from the per-expert loop "
            f"(max {(per_expert.float() - whole.float()).abs().max():.3e})")
