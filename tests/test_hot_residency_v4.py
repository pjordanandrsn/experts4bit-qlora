"""DeepSeek-V4 hot-residency gate: the hot/cold fused path reproduces the bare
DeepseekV4Experts4bit forward (clamped SwiGLU, fp32 epilogue) at every hot/cold split.

`enable_hot_residency` allowlists V4's forward, which is only safe while
`_fused_over_stack` actually reproduces V4's epilogue — an allowlisted forward whose
epilogue the fused path does NOT reproduce is served plain SwiGLU, silently. gpt-oss got
this gate when it was allowlisted; V4 shipped without one. These are the missing tests,
and they are built to FAIL on the two ways the fused path can drift: dropping the clamps,
and taking gpt-oss's combination instead.

CUDA + nf4_grouped required.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import disable_hot_residency, enable_hot_residency  # noqa: E402

# hidden and intermediate must be multiples of the NF4 blocksize (64).
E, H, INTER, K, TOKENS = 8, 128, 64, 3, 24

# `limit` is deliberately far below the checkpoint's 10.0: with H=128 the pre-activation
# std is ~11*w_scale*x_scale ~= 5.9, so limit=2.0 clamps a large fraction of rows and the
# clamp becomes measurable. At the real 10.0 the clamps rarely bind on synthetic weights
# and a "clamps are live" assertion could not discriminate.
LIMIT = 2.0


def _make_v4(limit=LIMIT, seed=0, w_scale=0.35, x_scale=1.5):
    """Synthetic V4 expert stack (clean-concat [gate; up], no biases), NF4-quantized."""
    from experts4bit_qlora.deepseek_v4 import _DeepseekV4ForwardMixin
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, H, generator=g) * w_scale   # [E, 2I, hidden]
    down = torch.randn(E, H, INTER, generator=g) * w_scale          # [E, hidden, I]
    mod = _DeepseekV4ForwardMixin.from_deepseek_v4(
        gate_up, down, limit=limit, quant_type="nf4",
        compute_dtype=torch.bfloat16).cuda()
    x = (torch.randn(TOKENS, H, generator=g) * x_scale).to(torch.bfloat16).cuda()
    logits = torch.randn(TOKENS, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    w = torch.softmax(val, dim=-1).to(torch.bfloat16).cuda()
    return mod, x, idx.cuda(), w


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


# Same bar as the gpt-oss gate: both paths decode the SAME packed NF4 weights, so the
# only difference is arithmetic ordering and the grouped-GEMM kernel.
TOL = 1.5e-2


def test_v4_is_now_eligible():
    """enable_hot_residency skips custom forwards; V4's must be allowlisted, not skipped."""
    mod, x, idx, w = _make_v4()
    n = enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
    assert n == 1, "DeepSeek-V4 module should be patched, not skipped"
    disable_hot_residency(mod)


def test_v4_hot_matches_bare_reference():
    mod, x, idx, w = _make_v4(seed=1)
    with torch.no_grad():
        ref = mod(x, idx, w)
        enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
        got = mod(x, idx, w)
    assert _b_rel(got, ref) < TOL, _b_rel(got, ref)
    disable_hot_residency(mod)


def test_v4_all_hot_equals_all_cold():
    """K=E (all resident) and K=0 (all streamed) both reproduce the bare ref —
    isolates epilogue correctness from the residency split."""
    mod, x, idx, w = _make_v4(seed=2)
    with torch.no_grad():
        ref = mod(x, idx, w)
        for hot in (torch.arange(mod.num_experts), torch.tensor([], dtype=torch.long)):
            enable_hot_residency(mod, [hot], device="cuda")
            got = mod(x, idx, w)
            assert _b_rel(got, ref) < TOL, (hot.numel(), _b_rel(got, ref))
            disable_hot_residency(mod)


def test_v4_clamps_are_live_in_the_fused_path():
    """The discriminating test. A fused path that ignored `clamp_limit` would still
    match a NO-CLAMP reference — so assert against BOTH: close to clamped, far from
    unclamped. Mixed split, so hot and cold rows are both exercised."""
    mod, x, idx, w = _make_v4(seed=3, limit=LIMIT)
    unclamped, *_ = _make_v4(seed=3, limit=0.0)  # identical weights, clamping disabled
    with torch.no_grad():
        ref = mod(x, idx, w)
        ref_noclamp = unclamped(x, idx, w)
        # precondition: the clamp actually bites on these inputs, else nothing below
        # can discriminate and the test would pass vacuously
        assert _b_rel(ref, ref_noclamp) > 10 * TOL, (
            f"clamp barely binds (rel {_b_rel(ref, ref_noclamp)}) — raise x_scale/w_scale "
            f"or lower LIMIT, otherwise this test cannot detect a dropped clamp")
        enable_hot_residency(mod, [torch.tensor([0, 2, 5])], device="cuda")
        got = mod(x, idx, w)
    assert _b_rel(got, ref) < TOL, _b_rel(got, ref)
    assert _b_rel(got, ref_noclamp) > 10 * TOL, "fused path served UNCLAMPED SwiGLU"
    disable_hot_residency(mod)


def test_v4_does_not_take_the_gptoss_branch():
    """V4 shares gpt-oss's clamps but not its combination. The fused path selects the
    branch by `clamp_limit` vs `gptoss`; picking wrong yields (up+1)*gate*sigmoid(a*gate),
    which no assertion above would catch on its own."""
    mod, x, idx, w = _make_v4(seed=4)
    with torch.no_grad():
        enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
        got = mod(x, idx, w)
        disable_hot_residency(mod)

        # gpt-oss's GLU over the SAME clamped operands, built from the module's own
        # dequantized weights so only the combination differs.
        out = torch.zeros(TOKENS, H, dtype=torch.float32, device="cuda")
        for t in range(TOKENS):
            for j in range(K):
                e = int(idx[t, j])
                gu = mod._project(mod.gate_up_proj, mod.gate_up_absmax,
                                  mod._gate_up_shape, e, x[t], torch.bfloat16)
                gate, up = gu.float().chunk(2, dim=-1)
                gate = gate.clamp(max=LIMIT)
                up = up.clamp(min=-LIMIT, max=LIMIT)
                h = (up + 1) * (gate * torch.sigmoid(gate * 1.702))   # gpt-oss, not V4
                h = h * float(w[t, j])
                out[t] += mod._project(mod.down_proj, mod.down_absmax, mod._down_shape,
                                       e, h.to(torch.bfloat16), torch.bfloat16).float()
    assert _b_rel(got, out) > 10 * TOL, "fused path computed gpt-oss's GLU, not V4's"


def test_v4_epilogue_is_evaluated_in_fp32():
    """V4's reference promotes to fp32 for clamp+SiLU (`_apply_gate`) and only casts back
    for the down projection; the stock and gpt-oss references do NOT, which is why the
    other two branches of `_fused_over_stack` deliberately stay in compute dtype.

    Asserted STRUCTURALLY, by spying on the dtype the activation is handed. An end-to-end
    tolerance cannot see this: at bf16 compute the promotion moves the result far less
    than the output's own bf16 rounding, so a numerical version of this test measures
    noise and picks a winner at random (measured: 0.0067 vs 0.0058 with the promotion
    PRESENT). Same lesson as `test_nf4_tracks_the_reference` — prove exact properties
    exactly, and do not ask a tolerance to resolve below its own floor."""
    import torch.nn.functional as F
    mod, x, idx, w = _make_v4(seed=5)
    seen = []

    def spy(t):
        seen.append(t.dtype)
        return F.silu(t)

    with torch.no_grad():
        enable_hot_residency(mod, [torch.arange(mod.num_experts)], device="cuda")
        st = mod._hot_residency
        assert st.clamp_limit == LIMIT, "V4 module did not select the clamped branch"
        st.act_fn = spy
        mod(x, idx, w)
        assert seen and all(d is torch.float32 for d in seen), (
            f"clamped branch handed the activation {seen} — the fp32 promotion is gone")

        # And the promotion is tied to the clamped branch, not applied everywhere: with
        # clamp_limit cleared the same code must stay in compute dtype (bf16), which is
        # what keeps stock/gpt-oss modules faithful to THEIR references.
        seen.clear()
        st.clamp_limit = None
        mod(x, idx, w)
        assert seen and all(d is torch.bfloat16 for d in seen), (
            f"plain-SwiGLU branch promoted to {seen} — stock/gpt-oss fidelity changed")
    disable_hot_residency(mod)
