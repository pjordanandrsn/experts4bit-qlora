"""Reference-parity tests — does experts4bit compute the *same function* as the unquantized reference?

The rest of the suite checks **internal consistency**: freezing, gradient flow, offload==non-offload,
shapes, no-NaN. None of it anchors the **expert computation** to an independent reference — so a
layout / transpose / gate-up-order / routing-application bug that is *consistent between how the loader
writes and reads* is invisible: training loss still falls because the LoRA adapters simply adapt around
a scrambled base. These two tests close that gap.

Level 1 — ``test_primitive_forward_matches_float_reference`` (CPU-runnable, no transformers needed):
    the ``Experts4bit`` / ``ExpertsLoRA`` forward vs a hand-written float SwiGLU-MoE reference, with
    **negative controls** (gate/up swapped, routing weights dropped) that prove the check would *catch*
    those bugs, plus a per-expert **orientation** check (dequantized stored weight == source, in shape
    *and* value) that catches a transpose at the primitive level. Self-calibrating: the correct path
    must be ≥3× closer to the reference than any structural-bug control, so there is no magic tolerance.

Level 2 — ``test_loaded_model_matches_reference_forward`` (transformers, per architecture; runs on
    the GPU when present, else on CPU — bnb-CPU-4bit permitting — so CPU-only CI exercises the loader):
    the whole streaming-loader path vs the **real transformers model's** forward. This is the test that
    validates the loader's on-disk-layout assumption against each model's *actual* convention —
    transformers-v5 fused experts can be stored ``[E, in, out]`` for ``torch._grouped_mm``, while this
    package assumes ``[E, out, in]`` + ``F.linear`` (``x @ W.T``). A **rolled-expert** negative control
    (expert ``e`` gets expert ``e+1``'s weights — shape-safe, definitely wrong) proves the check has
    power. ``ExpertsLoRA`` is zero-delta at init (``B=0``), so the loaded model's logits isolate pure
    NF4-on-experts error; a transpose/layout bug destroys the correlation with the bf16 reference.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

import torch.nn.functional as F  # noqa: E402

from experts4bit_qlora import Experts4bit, ExpertsLoRA, ExpertsNbit  # noqa: E402

# bnb signals a missing/broken 4-bit backend in several ways depending on the build; catch them all so
# a host without a working bnb 4-bit path SKIPS cleanly rather than erroring (matches test_offload.py).
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _relerr(a, b):
    return (a - b).float().norm() / b.float().norm().clamp_min(1e-9)


def _ref_swiglu_moe(hs, idx, wts, gate_up, down, has_gate, *, swap=False, drop=False, act=F.silu):
    """A straightforward float SwiGLU-MoE forward in experts4bit's convention (gate_up=[E, out, hidden],
    down=[E, hidden, inter], x @ W.T via F.linear). ``swap``/``drop`` inject the two structural bugs the
    parity check must be able to see."""
    out = torch.zeros_like(hs)
    for e in range(gate_up.shape[0]):
        tok, pos = (idx == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        proj = F.linear(hs[tok], gate_up[e])
        if has_gate:
            g, u = proj.chunk(2, dim=-1)
            if swap:
                g, u = u, g
            h = act(g) * u
        else:
            h = act(proj)
        y = F.linear(h, down[e])
        if not drop:
            y = y * wts[tok, pos, None]
        out.index_add_(0, tok, y)
    return out


# --------------------------------------------------------------------------------------------------
# Level 1 — primitive forward vs float reference (CPU-runnable, self-calibrating, with orientation).
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "quant_type,blocksize",
    [("nf4", 64), ("fp4", 64), ("nf4", 128)],
    ids=["nf4-64", "fp4-64", "nf4-128"],
)
@pytest.mark.parametrize("has_gate", [True, False], ids=["swiglu", "plain-up"])
def test_primitive_forward_matches_float_reference(has_gate, quant_type, blocksize):
    torch.manual_seed(0)
    device = DEVICE
    E, HID, INTER, TOP_K, N_TOK = 6, 128, 256, 2, 40  # HID, INTER divisible by every blocksize used
    gate_up_out = 2 * INTER if has_gate else INTER
    # Small-magnitude, ~Gaussian weights (what NF4 is designed for). out != in on both projections, so
    # the orientation check's shape assertion actually bites on a transpose.
    gate_up = (torch.randn(E, gate_up_out, HID) * 0.1).to(device)
    down = (torch.randn(E, HID, INTER) * 0.1).to(device)
    hs = torch.randn(N_TOK, HID, device=device)
    idx = torch.randint(0, E, (N_TOK, TOP_K), device=device)
    wts = torch.rand(N_TOK, TOP_K, device=device)

    try:
        base = Experts4bit.from_float(
            gate_up, down, has_gate=has_gate, quant_type=quant_type, blocksize=blocksize, compute_dtype=torch.float32
        )
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit ({quant_type}/bs{blocksize}) unavailable on {device}: {e}")
    lora = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).to(device)

    got = lora(hs, idx, wts)

    # (a) Forward tracks the correct float reference far better than either structural-bug control.
    # fp4's coarser codebook compounds through the two SwiGLU projections (~2x nf4's forward error,
    # ~0.20 vs ~0.09 relerr here), so its margin is 2x rather than 3x — the separation from a
    # structural bug (which lands at or beyond the ~0.55+ controls) stays decisive either way.
    margin = 0.33 if quant_type == "nf4" else 0.5
    err_ok = _relerr(got, _ref_swiglu_moe(hs, idx, wts, gate_up, down, has_gate))
    controls = [_relerr(got, _ref_swiglu_moe(hs, idx, wts, gate_up, down, has_gate, drop=True))]
    if has_gate:
        controls.append(_relerr(got, _ref_swiglu_moe(hs, idx, wts, gate_up, down, has_gate, swap=True)))
    for c in controls:
        assert err_ok < margin * c, (
            f"forward relerr {err_ok:.3f} not ≥{1 / margin:.0f}x closer than bug control {c:.3f}"
        )

    # (b) Zero LoRA delta at init (B=0): the adapted forward is *bit-for-bit* the frozen base.
    assert torch.equal(got, base(hs, idx, wts))

    # (c) Orientation: each dequantized stored expert == its SOURCE weight, in shape AND value (within
    #     NF4). Compared against the original tensor (not a re-quant), so a transpose is caught here.
    for e in range(E):
        dq_gu = base._dequantize_expert(base.gate_up_proj, base.gate_up_absmax, base._gate_up_shape, e, torch.float32)
        dq_dn = base._dequantize_expert(base.down_proj, base.down_absmax, base._down_shape, e, torch.float32)
        assert dq_gu.shape == gate_up[e].shape and dq_dn.shape == down[e].shape
        assert _relerr(dq_gu, gate_up[e]) < 0.2 and _relerr(dq_dn, down[e]) < 0.2


# --------------------------------------------------------------------------------------------------
# Level 2 — loaded model vs the real transformers forward (GPU), per architecture, with a control.
# --------------------------------------------------------------------------------------------------
def _olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    return OlmoeForCausalLM(OlmoeConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, num_experts=8, num_experts_per_tok=2, vocab_size=128,
    ))


def _qwen3_moe():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
        vocab_size=128, decoder_sparse_step=1, mlp_only_layers=[], head_dim=16,
    ))


def _gemma4():
    pytest.importorskip("transformers.models.gemma4", reason="this transformers has no gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    return Gemma4ForCausalLM(Gemma4TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, head_dim=16, num_experts=8, top_k_experts=2,
        moe_intermediate_size=64, enable_moe_block=True,
    ))


def _write_ckpt(model, d, per_expert):
    """Same on-disk layouts the loader targets: per-expert Linears (OLMoE/Qwen3) or fused (Gemma-4)."""
    import json
    import os

    from safetensors.torch import save_file

    new = {}
    for k, v in model.state_dict().items():
        if per_expert and k.endswith("experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g.contiguous()
                new[f"{base}{e}.up_proj.weight"] = u.contiguous()
        elif per_expert and k.endswith("experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            new[k] = v
    # bf16 + .clone() to break any shared storage (e.g. Gemma-4 ties lm_head to embed_tokens) — else
    # safetensors refuses to save "tensors that share memory". (save_pretrained does the same unties.)
    new = {k: v.to(torch.bfloat16).contiguous().clone() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in new}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    model.config.save_pretrained(d)


def _roll_experts(model):
    """Negative control: give expert e the packed weights of expert e+1 (shape-safe, definitely wrong).
    The router still routes to e, so a correct load diverges from this by far more than NF4 error."""
    rolled = 0
    for m in model.modules():
        # ExpertsNbit, not Experts4bit: the base may be the parent class for non-4-bit schemes,
        # and a control that silently matches zero modules would nullify this test (the rolled
        # "corruption" would equal the clean model bit-for-bit).
        if isinstance(m, ExpertsNbit):
            rolled += 1
            with torch.no_grad():
                for name in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
                    t = getattr(m, name)
                    if t is None:  # passthrough schemes carry no absmax
                        continue
                    setattr_target = m._parameters if name in m._parameters else m._buffers
                    setattr_target[name] = t.roll(1, dims=0).clone()
    assert rolled, "negative control matched no expert modules — the control itself is broken"


@pytest.mark.parametrize(
    "build,per_expert", [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False)],
    ids=["olmoe", "qwen3_moe", "gemma4"],
)
def test_loaded_model_matches_reference_forward(build, per_expert, tmp_path):
    """The 4-bit-loaded model must compute the same function as the bf16 reference (within NF4 error),
    and must be *dramatically* closer to it than a rolled-expert corruption — which is only true if the
    loader mapped each model's on-disk expert layout into experts4bit's [E, out, in] convention right."""
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    ref_model = build().to(DEVICE, dtype=torch.bfloat16).eval()
    ref_model.config.use_cache = False
    _write_ckpt(ref_model, str(tmp_path), per_expert=per_expert)

    ids = torch.randint(0, ref_model.config.vocab_size, (1, 16), device=DEVICE)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=ids).logits

    try:
        model, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, torch.bfloat16, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit unavailable: {e}")
    model.config.use_cache = False
    model.eval()
    with torch.no_grad():
        q_logits = model(input_ids=ids).logits  # ExpertsLoRA is zero-delta at init => pure 4-bit base

    err_ok = _relerr(q_logits, ref_logits)
    _roll_experts(model)
    with torch.no_grad():
        rolled_logits = model(input_ids=ids).logits
    err_rolled = _relerr(rolled_logits, ref_logits)

    # Self-calibrating: NF4-on-experts must track the bf16 reference far better than a scrambled base.
    # A layout / transpose bug lands near (or worse than) the rolled control and trips this.
    assert err_ok < 0.5 * err_rolled, (
        f"{build.__name__}: 4-bit vs bf16 relerr {err_ok:.3f} not clearly better than "
        f"rolled-expert control {err_rolled:.3f} — suspect an expert-layout mismatch in the loader"
    )
    # And a direct high-correlation sanity check (a transpose destroys per-position cosine).
    cos = F.cosine_similarity(q_logits.flatten(0, 1), ref_logits.flatten(0, 1), dim=-1).mean()
    assert cos > 0.9, f"{build.__name__}: logits cosine to bf16 reference only {cos:.3f}"
