# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_SERVE_ATTN_INT8``'s kernel path through grouped-nf4-gemm K37's own ``dense_i8.linear_i8`` and grid: under
Triton's interpreter on CPU (tests/test_int8_attn.py runs this file in its own process, as tests/test_p149.py runs
P149's kernels), or compiled on a card with ``TRITON_INTERPRET=0``. Skipped when the installed grouped-nf4-gemm has no
``dense_i8`` (K37)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
transformers = pytest.importorskip("transformers")
# triton decorates its own @jit library at its first import, so the interpreter env must precede that import. In CI's
# full suite another module imports triton first; this file then skips itself here, and tests/test_int8_attn.py runs it
# in a fresh interpreted process.
if os.environ["TRITON_INTERPRET"] == "1" and type(triton.language.zeros).__name__ != "InterpretedFunction":
    pytest.skip("triton was imported compiled earlier in this process; tests/test_int8_attn.py runs this file in a "
                "fresh interpreted process", allow_module_level=True)
DEV = "cuda" if os.environ["TRITON_INTERPRET"] == "0" and torch.cuda.is_available() else "cpu"
if os.environ["TRITON_INTERPRET"] != "1" and DEV == "cpu":
    pytest.skip("compiled kernels need a CUDA card", allow_module_level=True)
pytest.importorskip("dense_i8", reason="grouped-nf4-gemm without dense_i8 (K37)")

from experts4bit_qlora.engines import int8_attn as ia  # noqa: E402

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


@pytest.fixture(autouse=True)
def routed(monkeypatch):
    monkeypatch.setattr(ia, "_on_device", lambda x: True)   # interpreted CPU calls take the kernel path too


def _mixtral(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.MixtralConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                                     num_experts_per_tok=2, max_position_embeddings=64)
    return transformers.MixtralForCausalLM(cfg).to(DEV, torch.bfloat16).eval()


@pytest.mark.parametrize("rows", [1, 7, 16])
def test_the_kernel_path_reads_the_dequantized_weights(rows):
    m = _mixtral()
    ia.enable_serve_attn_int8(m)                                # K37's linear_i8 and its grid
    mod = m.model.layers[0].self_attn.q_proj
    x = torch.randn(rows, mod.in_features, device=DEV).to(torch.bfloat16)
    out = mod(x).double()
    ref = x.double() @ (mod.q.double() * mod.s.double()[:, None]).T
    assert mod.routed == {rows: 1} and mod.passed == {}
    assert bool(((out - ref).abs() <= 2 ** -7 * ref.abs() + 1e-6).all())   # one bf16 step (the interpreter truncates)


def test_a_decode_sized_call_tracks_the_rounded_model():
    m, ref = _mixtral(), _mixtral()
    _, quantize = ia.kernel()
    with torch.no_grad():
        for layer in ref.model.layers:
            for name in PROJ:
                lin = getattr(layer.self_attn, name)
                q, s = quantize(lin.weight)
                lin.weight.copy_((q.float() * s[:, None]).to(torch.bfloat16))
    ia.enable_serve_attn_int8(m)
    ids = torch.randint(0, 128, (1, 5), device=DEV)
    with torch.no_grad():
        a, b = m(ids).logits.float(), ref(ids).logits.float()
    assert ia.engagement(m) == {"modules": 8, "routed_modules_by_rows": {"5": 8}, "passed_modules_by_rows": {}}
    assert (a - b).abs().max().item() <= 2 ** -5 * b.abs().max().item()   # the fp32 scale vs bf16(q * s), two layers
