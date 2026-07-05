"""Constructor-contract tests for ``Experts4bit``/``ExpertsNbit``: reject wrong layouts and wrong
scheme names loudly, at build time — and normalize the accepted scheme spellings in one place.

The documented layout is ``[num_experts, out, in]`` for both stacks. Numel-preserving mistakes — a
transposed ``down_proj``, or the grouped-GEMM ``[num_experts, in, out]`` convention some
transformers checkpoints use on disk — previously quantized cleanly and only surfaced as a
scrambled forward (when the 2D expert weight is square, e.g. OLMoE's 2048x2048 ``gate_up``) or as a
cryptic reshape error deep inside dequantize. ``from_float`` now cross-checks the two stacks and
raises a ``ValueError`` naming the expected layout.

Limitation (by construction): a transposed stack whose 2D expert weight is *square* has an
identical shape, so no shape check can see it — the value-level orientation check in
``tests/test_reference_parity.py`` is the anchor for that case.

All tests here are CPU-only and cheap: validation runs before any quantization, so no working
bitsandbytes 4-bit backend is needed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsNbit, normalize_quant_type  # noqa: E402

E, HID, INTER = 2, 128, 192  # non-square everywhere, so every transposition is shape-visible


def _stacks():
    torch.manual_seed(0)
    gate_up = torch.randn(E, 2 * INTER, HID) * 0.1
    down = torch.randn(E, HID, INTER) * 0.1
    return gate_up, down


def test_from_float_rejects_grouped_gemm_gate_up_layout():
    """gate_up stored [num_experts, in, out] (the torch._grouped_mm on-disk convention)."""
    gate_up, down = _stacks()
    with pytest.raises(ValueError, match="inconsistent expert stacks"):
        Experts4bit.from_float(gate_up.transpose(1, 2).contiguous(), down)


def test_from_float_rejects_transposed_down_proj():
    gate_up, down = _stacks()
    with pytest.raises(ValueError, match="inconsistent expert stacks"):
        Experts4bit.from_float(gate_up, down.transpose(1, 2).contiguous())


def test_from_float_rejects_expert_count_mismatch():
    gate_up, down = _stacks()
    with pytest.raises(ValueError, match="inconsistent expert stacks"):
        Experts4bit.from_float(gate_up[:1], down)


def test_from_float_rejects_gate_up_out_mismatch_for_plain_up():
    """has_gate=False expects gate_up_out == intermediate, not 2*intermediate."""
    gate_up, down = _stacks()
    with pytest.raises(ValueError, match="inconsistent expert stacks"):
        Experts4bit.from_float(gate_up, down, has_gate=False)


def test_from_float_rejects_non_3d():
    gate_up, down = _stacks()
    with pytest.raises(ValueError, match="3D"):
        Experts4bit.from_float(gate_up[0], down)


def test_init_rejects_bad_quant_type():
    with pytest.raises(ValueError, match="quant_type"):
        Experts4bit(E, HID, INTER, quant_type="int4")


def test_normalize_quant_type_function():
    """The single scheme-name validation path: canonical names pass through, spellings normalize
    (case/whitespace + the two torch-dtype longhands), everything else raises listing the allowed
    set — the same error users hit from the constructors and the loader."""
    assert normalize_quant_type("nf4") == "nf4"
    assert normalize_quant_type("NF4") == "nf4"
    assert normalize_quant_type(" BFloat16 ") == "bf16"
    assert normalize_quant_type("FLOAT16") == "fp16"
    assert normalize_quant_type("Fp8") == "fp8"
    for bad in ("int4", "nf3", "float8", ""):
        with pytest.raises(ValueError, match="quant_type must be one of"):
            normalize_quant_type(bad)
    with pytest.raises(ValueError, match="quant_type must be one of"):
        normalize_quant_type(torch.bfloat16)  # a torch.dtype is not a scheme name


def test_constructor_accepts_aliases():
    """Aliases normalize at construction — the module's own quant_type is always canonical (no
    quantize needed: constructor-time only, so this runs on any host)."""
    assert ExpertsNbit(E, HID, INTER, quant_type="bfloat16").quant_type == "bf16"
    assert ExpertsNbit(E, HID, INTER, quant_type=" FP16 ").quant_type == "fp16"
    assert Experts4bit(E, HID, INTER, quant_type="NF4", device="cpu").quant_type == "nf4"


def test_experts4bit_rejects_alias_of_16bit():
    """Experts4bit stays 4-bit-only through the alias path: 'float16' normalizes to fp16 and is
    then rejected against the subclass's allowed set — not silently accepted via the longhand."""
    with pytest.raises(ValueError, match=r"quant_type must be one of \('nf4', 'fp4'\)"):
        Experts4bit(E, HID, INTER, quant_type="float16")


def test_bare_constructor_default_device_is_consistent():
    """ExpertsNbit(...) with no device must build a single-device (CPU) module on any host:
    torch.empty(device=None) puts the packed buffers on CPU, and the codebook must follow — not
    take bitsandbytes' get_4bit_type default of "cuda", which crashed CUDA-less hosts and split
    the module across devices on CUDA ones."""
    m = Experts4bit(E, HID, INTER, quant_type="nf4")
    assert m.gate_up_proj.device.type == "cpu" and m.code.device.type == "cpu"


@pytest.mark.parametrize("hidden,inter", [(100, INTER), (HID, 100)], ids=["hidden", "intermediate"])
def test_init_rejects_blocksize_misaligned_dims(hidden, inter):
    """in_features must divide the blocksize so quantization blocks never straddle an expert."""
    with pytest.raises(ValueError, match="divisible by blocksize"):
        Experts4bit(E, hidden, inter, blocksize=64)
