"""Double-quantized ("nested") storage for the frozen expert absmax -- ``compress_expert_absmax_`` (E4B_ABSMAX_DQ=1).

What is pinned here, and why each matters:

* **Bookkeeping** (CPU): which stacks are compressed (every ExpertsLoRA-wrapped 4-bit stack; passthrough skipped), the
  count, idempotence, the stored layout (the fp32 buffers are GONE from the state_dict, four nested buffers replace each),
  and that a refusal compresses nothing -- a half-compressed model is the failure this switch must never produce.
* **The values are bitsandbytes' own** (CPU, and CUDA when present): the stored codes, scales, offset and code book are
  the bytes ``quantize_4bit(W, compress_statistics=True)`` produces for the same stack, the accessor equals
  ``dequantize_blockwise(q, state2) + offset`` exactly, and e4b's per-expert dequantized WEIGHTS equal bnb's nested
  ``dequantize_4bit`` of the whole stack, row for row. So "on" means "what Unsloth / bnb double-quant trains on", bit for bit.
* **Off is unchanged**: an uncompressed module's accessor is the old expression (a view of the same storage), and its
  per-expert operand is the stored buffer object itself.
* **Only the routed paths read it**: the reference loop and the fused training forward on a compressed module are
  BIT-IDENTICAL to the same code on an uncompressed module holding the dequantized absmax in fp32 -- the only difference
  the switch makes is the absmax values -- and within a stated tolerance of the fp32 run. Every other reader refuses by
  name, and the old attribute names raise rather than resolve to another buffer.
* **Memory**: the stored absmax bytes drop by the layout's exact ratio -- 4 / (1 + 4/256) less the code books, 3.9376x on a
  Qwen3-30B-A3B-sized layer -- and the CUDA allocator sees the drop.
"""
import copy
import math
import os
import pickle
import subprocess
import sys

import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

import torch.nn as nn  # noqa: E402

from experts4bit_qlora import (  # noqa: E402
    AbsmaxCompressedError,
    Experts4bit,
    ExpertsLoRA,
    ExpertsNbit,
    compress_expert_absmax_,
    expert_absmax_bytes,
    expert_absmax_fp32,
)
from experts4bit_qlora import absmax_dq  # noqa: E402
from experts4bit_qlora.absmax_dq import expert_absmax_rows, is_absmax_compressed  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stacks(E=4, H=128, inter=64, seed=0, device=DEVICE):
    g = torch.Generator().manual_seed(seed)
    gu = (torch.randn(E, 2 * inter, H, generator=g) * 0.05).to(device)
    dn = (torch.randn(E, H, inter, generator=g) * 0.05).to(device)
    return gu, dn


def _lora(E=4, H=128, inter=64, seed=0, device=DEVICE, quant_type="nf4", blocksize=64, compute_dtype=torch.float32,
          cls=Experts4bit, b_nonzero=True):
    require_quantize(device, quant_type)
    gu, dn = _stacks(E, H, inter, seed, device)
    base = cls.from_float(gu, dn, quant_type=quant_type, blocksize=blocksize, compute_dtype=compute_dtype)
    mod = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).to(device)
    if b_nonzero:                                    # a load-bearing adapter, so gradients through it are compared too
        g = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for p in (mod.gate_up_lora_B, mod.down_lora_B):
                p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(device))
    return mod


class _Model(nn.Module):
    """Two decoder layers, each with an ExpertsLoRA at `layers.<i>.mlp.experts` -- the streaming loader's shape."""

    def __init__(self, mods):
        super().__init__()
        self.layers = nn.ModuleList()
        for m in mods:
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = m
            self.layers.append(layer)


def _bases(model):
    return [m for m in model.modules() if isinstance(m, ExpertsNbit)]


# ----------------------------------------------------------------------------- bookkeeping
def test_compress_counts_wrapped_stacks_and_is_idempotent():
    model = _Model([_lora(seed=0), _lora(seed=1)])
    assert compress_expert_absmax_(model) == 2
    assert all(is_absmax_compressed(b) for b in _bases(model))
    assert compress_expert_absmax_(model) == 0, "a second call must leave compressed stacks alone and count none"


def test_passthrough_stacks_are_skipped_not_refused():
    model = _Model([_lora(quant_type="bf16", cls=ExpertsNbit)])
    assert compress_expert_absmax_(model) == 0
    assert not is_absmax_compressed(_bases(model)[0])


def test_stored_layout_replaces_the_fp32_buffers():
    E, H, inter = 4, 128, 64
    mod = _lora(E, H, inter)
    base = mod.base
    before = {k: v.clone() for k, v in base.state_dict().items() if k != "_extra_state"}
    compress_expert_absmax_(mod)
    sd = base.state_dict()
    assert "gate_up_absmax" not in sd and "down_absmax" not in sd, "the fp32 absmax must be gone, not shadowed"
    for which, (n, k) in (("gate_up", base._gate_up_shape), ("down", base._down_shape)):
        m = n * k // 64
        assert sd[f"{which}_absmax_q"].dtype == torch.uint8 and tuple(sd[f"{which}_absmax_q"].shape) == (E, m)
        assert sd[f"{which}_absmax_s"].dtype == torch.float32 and sd[f"{which}_absmax_s"].numel() == math.ceil(E * m / 256)
        assert sd[f"{which}_absmax_off"].dtype == torch.float32 and tuple(sd[f"{which}_absmax_off"].shape) == (1,)
        assert sd[f"{which}_absmax_code"].dtype == torch.float32 and sd[f"{which}_absmax_code"].numel() == 256
    # the packed weights are untouched
    assert torch.equal(sd["gate_up_proj"], before["gate_up_proj"]) and torch.equal(sd["down_proj"], before["down_proj"])
    # the extra-state contract (what a checkpoint validates against) is unchanged
    assert set(base.get_extra_state()) == {"schema", *type(base)._EXTRA_STATE_FIELDS}


def test_old_attribute_names_refuse_by_name():
    mod = _lora()
    compress_expert_absmax_(mod)
    guard = mod.base.gate_up_absmax
    assert guard is not None, "`is None` is the passthrough test; a compressed stack must not read as passthrough"
    for use in (lambda: guard.view(-1), lambda: guard.numel(), lambda: guard[0], lambda: guard.to("cpu"),
                lambda: guard.float(), lambda: mod.base.down_absmax.shape):
        with pytest.raises(AbsmaxCompressedError, match="E4B_ABSMAX_DQ=1"):
            use()
    # dunder lookups behave, so the module still copies and pickles
    assert pickle.loads(pickle.dumps(guard)).name == guard.name
    clone = copy.deepcopy(mod)
    assert is_absmax_compressed(clone.base)
    assert "E4B_ABSMAX_DQ" in repr(guard)


def test_uncompressed_accessor_is_the_old_expression():
    """Off: the accessor returns `<which>_absmax.view(E, N, K // 64).float()` -- a view of the same storage, no copy --
    and the per-expert operand is the buffer object itself."""
    base = _lora().base
    E = base.num_experts
    for which, (n, k) in (("gate_up", base._gate_up_shape), ("down", base._down_shape)):
        buf = getattr(base, f"{which}_absmax")
        got = expert_absmax_fp32(base, which)
        old = buf.view(E, n, k // 64).float()
        assert got.data_ptr() == buf.data_ptr() and got.shape == old.shape and torch.equal(got, old)
        assert expert_absmax_rows(base, which) is buf


@pytest.mark.parametrize("E,H,inter", [(4, 128, 64), (3, 64, 64), (5, 192, 64)])
def test_rows_equal_the_whole_layer_accessor(E, H, inter):
    """The per-expert form (the reference loop's operand) equals the whole-layer accessor row for row, bit for bit --
    with expert rows on a nested-block boundary (E*M with M % 256 == 0) and off it (M % 256 != 0)."""
    mod = _lora(E, H, inter)
    compress_expert_absmax_(mod)
    for which, (n, k) in (("gate_up", mod.base._gate_up_shape), ("down", mod.base._down_shape)):
        whole = expert_absmax_fp32(mod.base, which)
        assert tuple(whole.shape) == (E, n, k // 64) and whole.dtype == torch.float32
        rows = expert_absmax_rows(mod.base, which)
        for e in range(E):
            assert torch.equal(rows[e], whole[e].reshape(-1)), (which, e)
            assert torch.equal(rows[torch.tensor(e, device=DEVICE)], whole[e].reshape(-1)), (which, e, "0-d index")


def test_a_refusal_compresses_nothing():
    good = _lora(seed=0)
    bare = Experts4bit.from_float(*_stacks(seed=1), compute_dtype=torch.float32)
    model = _Model([good])
    model.extra = bare
    with pytest.raises(ValueError, match="BARE expert stack"):
        compress_expert_absmax_(model)
    assert not is_absmax_compressed(good.base), "a refusal must leave the whole model as it was"
    assert isinstance(good.base.gate_up_absmax, torch.Tensor)


@pytest.mark.parametrize("case", ["int8", "blocksize128", "offloaded", "engine_on_base", "batched_wrapper", "evicted"])
def test_ineligible_stacks_are_refused(case):
    if case == "int8":
        mod = _lora(quant_type="int8", cls=ExpertsNbit)
        match = "8-bit"
    elif case == "blocksize128":
        mod = _lora(H=128, inter=128, blocksize=128)    # both in-dims divisible by the blocksize, or the constructor refuses first
        match = "blocksize 128"
    else:
        mod = _lora()
        if case == "offloaded":
            mod._offload = object()                  # the attribute enable_expert_offload stashes its handle under
            match = "expert offload"
        elif case == "engine_on_base":
            mod.base.forward = lambda *a: None       # what enable_fast / hot / pipelined residency do to a base
            match = "rebound this stack's forward"
        elif case == "batched_wrapper":
            mod._e4b_batched_ref = mod.forward
            mod.forward = lambda *a: None
            match = "enable_batched_train"
        else:
            mod.base.gate_up_absmax = torch.empty(0, device=DEVICE)   # an offload-evicted placeholder
            match = "evicted"
    with pytest.raises(ValueError, match=match):
        compress_expert_absmax_(mod)
    assert not is_absmax_compressed(mod.base)


def test_fast_train_patch_is_allowed_before_or_after():
    """enable_fast_train patches the wrapper's forward; its forward reads through the accessor, so compressing after it
    is allowed (the selection test is structural: `_e4b_train_ref` marks that patch)."""
    mod = _lora()
    mod._e4b_train_ref = mod.forward
    mod.forward = lambda *a: None
    assert compress_expert_absmax_(mod) == 1


def _engine_calls():
    import experts4bit_qlora as e
    from experts4bit_qlora.engines.hybrid_train import enable_hybrid_train
    return {
        "enable_batched_train": lambda m: e.enable_batched_train(m),
        "enable_expert_offload": lambda m: e.enable_expert_offload(m.layers[0].mlp.experts, DEVICE, pin=False),
        "enable_hot_residency": lambda m: e.enable_hot_residency(m, [[0]]),
        "enable_pipelined_residency": lambda m: e.enable_pipelined_residency(m, [[0]]),
        "enable_cold_engine": lambda m: e.enable_cold_engine(m, [[]]),
        "enable_nvme_residency": lambda m: e.enable_nvme_residency(m, "/nonexistent", [[0]], hot_rows=1),
        "enable_mxfp4_nvme_residency": lambda m: e.enable_mxfp4_nvme_residency(m, "/nonexistent", k_slots=1, hot_rows=1),
        "enable_nvme_train_residency": lambda m: e.enable_nvme_train_residency(m, "/nonexistent", hot_rows=1),
        "enable_hybrid_tier": lambda m: e.enable_hybrid_tier(m, "/nonexistent", {}, hot_rows=1),
        "enable_hybrid_train": lambda m: enable_hybrid_train(m, "/nonexistent", {}, hot_rows=1),
    }


@pytest.mark.parametrize("entry", sorted(_engine_calls()))
def test_every_other_reader_refuses_a_compressed_model_by_name(entry):
    model = _Model([_lora()])
    compress_expert_absmax_(model)
    with pytest.raises(AbsmaxCompressedError, match=rf"{entry}.*E4B_ABSMAX_DQ=1"):
        _engine_calls()[entry](model)


def test_arch_forwards_refuse_a_compressed_stack():
    """The gpt-oss and DeepSeek-V4 per-expert forwards read the fp32 buffers themselves. compress_expert_absmax_ never
    compresses a bare stack, so this drives the forward guard directly on a stack compressed underneath it."""
    from experts4bit_qlora.arch.deepseek_v4 import _DeepseekV4ForwardMixin
    from experts4bit_qlora.arch.gptoss import _GptOssForwardMixin

    require_quantize(DEVICE, "nf4")
    gu, dn = _stacks()
    v4 = _DeepseekV4ForwardMixin.from_deepseek_v4(gu, dn, quant_type="nf4", compute_dtype=torch.float32)
    E, inter2, H = gu.shape
    go = _GptOssForwardMixin.from_gptoss(gu.transpose(1, 2).contiguous(), torch.zeros(E, inter2, device=DEVICE),
                                         dn.transpose(1, 2).contiguous(), torch.zeros(E, H, device=DEVICE),
                                         compute_dtype=torch.float32)
    x = torch.randn(3, H, device=DEVICE)
    idx = torch.tensor([[0, 1]] * 3, device=DEVICE)
    w = torch.full((3, 2), 0.5, device=DEVICE)
    for mod, entry in ((v4, "DeepseekV4Experts forward"), (go, "GptOssExperts forward")):
        mod(x, idx, w)                               # uncompressed: runs
        absmax_dq._compress_one(mod)
        with pytest.raises(AbsmaxCompressedError, match=entry):
            mod(x, idx, w)


def test_a_v4_stack_under_an_expertslora_is_compressed_and_trains():
    """DeepSeek-V4 loads INSIDE an ExpertsLoRA (its epilogue reaches the wrapper through `_apply_gate`), so its stack is
    wrapped and is compressed; training reaches it through the accessor, never through the V4 forward."""
    from experts4bit_qlora.arch.deepseek_v4 import _DeepseekV4ForwardMixin

    require_quantize(DEVICE, "nf4")
    gu, dn = _stacks()
    mod = ExpertsLoRA(_DeepseekV4ForwardMixin.from_deepseek_v4(gu, dn, quant_type="nf4", compute_dtype=torch.float32),
                      r=4, alpha=8, dtype=torch.float32).to(DEVICE)
    assert compress_expert_absmax_(mod) == 1
    x = torch.randn(3, gu.shape[2], device=DEVICE, requires_grad=True)
    mod(x, torch.tensor([[0, 1]] * 3, device=DEVICE), torch.full((3, 2), 0.5, device=DEVICE)).sum().backward()
    assert mod.gate_up_lora_A.grad is not None


# ----------------------------------------------------------------------------- the values are bitsandbytes' own
def test_values_are_bitsandbytes_nested_statistics():
    """(a) The stored nested statistics are the bytes bitsandbytes' quantize_4bit(W, compress_statistics=True) produces for
    the SAME stack; the accessor equals dequantize_blockwise(q, state2) + offset exactly; and e4b's per-expert dequantized
    weights equal bnb's own nested dequantize_4bit of the whole stack, row for row. Runs on CUDA when present."""
    import bitsandbytes.functional as BF

    E, H, inter = 4, 128, 64
    gu, dn = _stacks(E, H, inter)
    mod = _lora(E, H, inter)
    compress_expert_absmax_(mod)
    base = mod.base
    for which, W in (("gate_up", gu), ("down", dn)):
        q_bnb, st = BF.quantize_4bit(W.contiguous(), blocksize=64, compress_statistics=True, quant_type="nf4")
        assert st.nested
        q, s, off, code = (getattr(base, f"{which}_absmax{x}") for x in ("_q", "_s", "_off", "_code"))
        assert torch.equal(q.reshape(-1), st.absmax.reshape(-1)), f"{which}: 8-bit codes differ from bnb's"
        assert torch.equal(s, st.state2.absmax), f"{which}: nested scales differ"
        assert torch.equal(off, st.offset.reshape(1)), f"{which}: offset differs"
        assert torch.equal(code, st.state2.code), f"{which}: code book differs"
        assert torch.equal(getattr(base, f"{which}_proj").reshape(-1), q_bnb.reshape(-1)), f"{which}: packed bytes differ"
        # the accessor IS bnb's nested dequantize
        bnb_absmax = BF.dequantize_blockwise(st.absmax, st.state2) + st.offset
        assert torch.equal(expert_absmax_fp32(base, which).reshape(-1), bnb_absmax.reshape(-1))
        # and the weights e4b's per-expert loop dequantizes equal bnb's nested whole-stack dequantize
        whole = BF.dequantize_4bit(q_bnb, quant_state=st)
        rows = expert_absmax_rows(base, which)
        shape = base._gate_up_shape if which == "gate_up" else base._down_shape
        for e in range(E):
            got = base._dequantize_expert(getattr(base, f"{which}_proj"), rows, shape, e, torch.float32)
            assert torch.equal(got, whole[e].float()), (which, e)
        # lossy against fp32: bounded by the 8-bit dynamic code's resolution, and not zero
        fp32 = BF.quantize_4bit(W.contiguous(), blocksize=64, compress_statistics=False, quant_type="nf4")[1].absmax
        rel = ((bnb_absmax.reshape(-1) - fp32.reshape(-1)).abs() / fp32.reshape(-1)).max().item()
        assert 0 < rel < 0.02, rel


# ----------------------------------------------------------------------------- the reference loop
def _fwd_bwd(mod, x, idx, w, loss_w):
    x = x.clone().requires_grad_(True)
    for p in mod.parameters():
        p.grad = None
    out = mod(x, idx, w)
    (out.float() * loss_w).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in mod.named_parameters() if p.grad is not None}
    return out.detach(), x.grad.detach().clone(), grads


def _with_dequantized_fp32(compressed):
    """An UNCOMPRESSED twin of `compressed`'s ExpertsLoRA whose fp32 absmax buffers hold the dequantized nested values:
    the same code, the same kernels, the only difference being where the absmax values come from."""
    twin = copy.deepcopy(compressed)
    b = twin.base
    vals = {w: expert_absmax_fp32(b, w).reshape(b.num_experts, -1).clone() for w in ("gate_up", "down")}
    for w in ("gate_up", "down"):
        for sfx in ("_q", "_s", "_off", "_code"):
            delattr(b, f"{w}_absmax{sfx}")
        object.__delattr__(b, f"{w}_absmax")
        b.register_buffer(f"{w}_absmax", vals[w])
    object.__delattr__(b, "_e4b_absmax_dq")
    assert not is_absmax_compressed(b)
    return twin


def _inputs(H, E, n_tok=24, k=2, dtype=torch.float32, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_tok, H, generator=g).to(DEVICE, dtype)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(n_tok)]).to(DEVICE)
    w = torch.rand(n_tok, k, generator=g).to(DEVICE, dtype)
    loss_w = torch.randn(n_tok, H, generator=g).to(DEVICE)
    return x, idx, w, loss_w


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def test_reference_loop_on_compressed_equals_the_same_loop_on_the_dequantized_absmax():
    """(b, reference path) Forward, dL/dx and every LoRA gradient of the compressed module are BIT-IDENTICAL to the
    uncompressed loop holding the dequantized values in fp32; against the fp32-absmax run they move by the absmax's
    double-quant error only (stated tolerance 1e-2 relative, measured ~1e-3)."""
    E, H, inter = 4, 128, 64
    fp32 = _lora(E, H, inter).train()
    dq = copy.deepcopy(fp32)
    assert compress_expert_absmax_(dq) == 1
    twin = _with_dequantized_fp32(dq)
    args = _inputs(H, E)
    o_dq, dx_dq, g_dq = _fwd_bwd(dq, *args)
    o_tw, dx_tw, g_tw = _fwd_bwd(twin, *args)
    assert torch.equal(o_dq, o_tw) and torch.equal(dx_dq, dx_tw)
    assert set(g_dq) == set(g_tw) and all(torch.equal(g_dq[n], g_tw[n]) for n in g_dq)
    o_32, dx_32, g_32 = _fwd_bwd(fp32, *args)
    errs = {"forward": _rel(o_dq, o_32), "dL/dx": _rel(dx_dq, dx_32), **{n: _rel(g_dq[n], g_32[n]) for n in g_32}}
    assert all(v < 1e-2 for v in errs.values()), errs
    assert errs["forward"] > 0, "the switch changed nothing -- the absmax was not replaced"


def test_inference_decode_path_reads_the_rows():
    """The single-token decode loop indexes the absmax with a 0-d device tensor; the row view must take it."""
    E, H, inter = 4, 128, 64
    dq = _lora(E, H, inter).eval()
    compress_expert_absmax_(dq)
    twin = _with_dequantized_fp32(dq)
    x = torch.randn(1, H, device=DEVICE)
    idx = torch.tensor([[1, 3]], device=DEVICE)
    w = torch.tensor([[0.6, 0.4]], device=DEVICE)
    with torch.no_grad():
        assert torch.equal(dq(x, idx, w), twin(x, idx, w))


# ----------------------------------------------------------------------------- memory
def _nested_bytes(m):
    """The four nested buffers of one projection holding `m` absmax values: uint8 codes, an fp32 scale per 256, a 4-byte
    offset and the 256-entry fp32 code book."""
    return m + 4 * math.ceil(m / 256) + 4 + 4 * 256


def test_absmax_bytes_drop_by_the_nested_ratio():
    """(c) expert_absmax_bytes before / after equals the layout's exact ratio; for a Qwen3-30B-A3B-sized layer that ratio
    is 3.9375x (4 bytes per 64 weights -> 1 + 4/256, plus a 1 KB code book and a 4-byte offset per projection)."""
    E, H, inter = 4, 256, 128
    mod = _lora(E, H, inter)
    before = expert_absmax_bytes(mod)
    assert before == 4 * E * (2 * inter * H + H * inter) // 64
    compress_expert_absmax_(mod)
    after = expert_absmax_bytes(mod)
    m_gu, m_dn = E * 2 * inter * H // 64, E * H * inter // 64
    assert after == _nested_bytes(m_gu) + _nested_bytes(m_dn)
    # the real-model ratio, from the same layout (Qwen3-30B-A3B: 128 experts, gate_up 1536x2048, down 2048x768)
    m_gu, m_dn = 128 * 1536 * 2048 // 64, 128 * 2048 * 768 // 64
    real = 4 * (m_gu + m_dn) / (_nested_bytes(m_gu) + _nested_bytes(m_dn))
    assert abs(real - 3.9376) < 1e-4, real          # 4 / (1 + 4/256) = 3.9385x, less the 2 KB of code books and offsets


@CUDA
def test_cuda_allocator_sees_the_drop():
    """(c) on the device: memory_allocated falls by (fp32 bytes - nested bytes) once the old buffers are released."""
    E, H, inter = 8, 1024, 512
    mod = _lora(E, H, inter, device="cuda")
    torch.cuda.synchronize()
    before_bytes, alloc0 = expert_absmax_bytes(mod), torch.cuda.memory_allocated()
    compress_expert_absmax_(mod)
    torch.cuda.synchronize()
    after_bytes, alloc1 = expert_absmax_bytes(mod), torch.cuda.memory_allocated()
    ratio = before_bytes / after_bytes
    print(f"\nabsmax bytes {before_bytes} -> {after_bytes} ({ratio:.4f}x); allocated {alloc0} -> {alloc1}")
    m_gu, m_dn = E * 2 * inter * H // 64, E * H * inter // 64
    assert before_bytes == 4 * (m_gu + m_dn) and after_bytes == _nested_bytes(m_gu) + _nested_bytes(m_dn)
    assert 3.8 < ratio < 4.0, ratio
    # the caching allocator rounds each block up to 512 B; the drop is the byte difference to within that rounding
    assert abs((alloc0 - alloc1) - (before_bytes - after_bytes)) <= 8 * 512, (alloc0 - alloc1, before_bytes - after_bytes)


# ----------------------------------------------------------------------------- the fused training path (CUDA + grouped-nf4-gemm)
def _fused(mod):
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    from experts4bit_qlora import enable_fast_train
    assert enable_fast_train(mod) == 1
    return mod


@CUDA
@pytest.mark.parametrize("dgrad", [False, True])
def test_fused_training_on_compressed_equals_the_kernels_on_the_dequantized_absmax(dgrad):
    """(b) enable_fast_train's forward + backward on a compressed module is BIT-IDENTICAL to the same kernels fed the
    dequantized absmax from fp32 buffers, and within 1e-2 relative of the switch-off (fp32 absmax) run."""
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    from experts4bit_qlora import enable_fast_train

    E, H, inter = 8, 256, 128
    off = _lora(E, H, inter, device="cuda", compute_dtype=torch.bfloat16).train()
    on = copy.deepcopy(off)
    assert compress_expert_absmax_(on) == 1
    twin = _with_dequantized_fp32(on)
    for m in (off, on, twin):
        assert enable_fast_train(m, dgrad=dgrad) == 1
    args = _inputs(H, E, n_tok=64, dtype=torch.bfloat16)
    o_on, dx_on, g_on = _fwd_bwd(on, *args)
    o_tw, dx_tw, g_tw = _fwd_bwd(twin, *args)
    assert torch.equal(o_on, o_tw) and torch.equal(dx_on, dx_tw)
    assert set(g_on) == set(g_tw) and all(torch.equal(g_on[n], g_tw[n]) for n in g_on), \
        {n: _rel(g_on[n], g_tw[n]) for n in g_on}
    o_off, dx_off, g_off = _fwd_bwd(off, *args)
    errs = {"forward": _rel(o_on, o_off), "dL/dx": _rel(dx_on, dx_off), **{n: _rel(g_on[n], g_off[n]) for n in g_off}}
    print(f"\ndgrad={dgrad} on vs off: " + "  ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    assert all(v < 1e-2 for v in errs.values()), errs


@CUDA
def test_fused_training_can_be_enabled_before_compressing():
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    from experts4bit_qlora import enable_fast_train

    E, H, inter = 8, 256, 128
    a = _lora(E, H, inter, device="cuda", compute_dtype=torch.bfloat16).train()
    b = copy.deepcopy(a)
    assert enable_fast_train(a) == 1 and compress_expert_absmax_(a) == 1       # patch, then compress
    assert compress_expert_absmax_(b) == 1 and enable_fast_train(b) == 1       # compress, then patch
    args = _inputs(H, E, n_tok=32, dtype=torch.bfloat16)
    oa, dxa, ga = _fwd_bwd(a, *args)
    ob, dxb, gb = _fwd_bwd(b, *args)
    assert torch.equal(oa, ob) and torch.equal(dxa, dxb) and all(torch.equal(ga[n], gb[n]) for n in ga)


# ----------------------------------------------------------------------------- the trainer's switch
def test_trainer_refuses_the_switch_with_expert_offload():
    env = dict(os.environ, E4B_ABSMAX_DQ="1", OFFLOAD_EXPERTS="1", PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-c", "import experts4bit_qlora.train as t; t.main()"], capture_output=True, text=True,
                       timeout=300, env=env, cwd=REPO)
    assert p.returncode != 0 and "E4B_ABSMAX_DQ=1 is a RESIDENT-training switch" in p.stderr + p.stdout, (p.stdout + p.stderr)[-2000:]
    assert "loading" not in p.stdout, "the refusal must come before the model load"


# ----------------------------------------------------------------------------- the CLI trainer's default (TC1 amendments 28 / 31)
@pytest.mark.parametrize("env,want", [({}, (True, False)), ({"E4B_ABSMAX_DQ": "0"}, (False, False)), ({"E4B_ABSMAX_DQ": "1"}, (True, True)),
                                      ({"OFFLOAD_EXPERTS": "1"}, (False, False)), ({"TRAIN_ARENA": "/x"}, (False, False))])
def test_trainer_switch_is_on_by_default_for_resident_training(env, want):
    """Unset, the trainer double-quantizes the absmax on a resident run and not under expert offload or TRAIN_ARENA (those paths
    read the fp32 absmax); E4B_ABSMAX_DQ=0 turns it off; =1 requires it (and stays the switch that is refused with offload)."""
    base = {k: v for k, v in os.environ.items() if k not in ("E4B_ABSMAX_DQ", "OFFLOAD_EXPERTS", "TRAIN_ARENA")}
    full = dict(base, **env, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-c", "import experts4bit_qlora.train as t; print(t.ABSMAX_DQ, t.ABSMAX_DQ_REQUIRED)"],
                       capture_output=True, text=True, timeout=300, env=full, cwd=REPO)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip().splitlines()[-1] == f"{want[0]} {want[1]}", (env, p.stdout)


def test_trainer_default_keeps_the_fp32_absmax_where_the_compressor_refuses():
    """The default never turns a refusal into a failed run: a bare stack (no ExpertsLoRA wrapper; the compressor refuses it) or a
    passthrough model (nothing to compress) keeps its fp32 absmax; E4B_ABSMAX_DQ=1 (required) still refuses both. A wrapped NF4
    model is compressed in either mode."""
    from experts4bit_qlora.train import apply_absmax_dq

    require_quantize(DEVICE, "nf4")
    gu, dn = _stacks()
    bare = _Model([Experts4bit.from_float(gu, dn, quant_type="nf4", compute_dtype=torch.float32).to(DEVICE)])
    assert apply_absmax_dq(bare, required=False) == 0 and not any(is_absmax_compressed(m) for m in _bases(bare))
    with pytest.raises(ValueError):
        apply_absmax_dq(bare, required=True)
    passthrough = _Model([_lora(quant_type="bf16", cls=ExpertsNbit)])
    assert apply_absmax_dq(passthrough, required=False) == 0
    with pytest.raises(SystemExit):
        apply_absmax_dq(passthrough, required=True)
    wrapped = _Model([_lora(seed=1), _lora(seed=2)])
    assert apply_absmax_dq(wrapped, required=False) == 2 and all(is_absmax_compressed(m) for m in _bases(wrapped))


# ----------------------------------------------------------------------------- enable_fast_train's default (TC1 amendment 56)
@pytest.mark.parametrize("env,explicit,want", [
    ({}, None, (True, False)), ({"E4B_ABSMAX_DQ": "0"}, None, (False, False)), ({"E4B_ABSMAX_DQ": "1"}, None, (True, True)),
    ({"OFFLOAD_EXPERTS": "1"}, None, (False, False)), ({"TRAIN_ARENA": "/x"}, None, (False, False)),
    ({"E4B_ABSMAX_DQ": "1", "OFFLOAD_EXPERTS": "1"}, None, (True, True)),
    ({"E4B_ABSMAX_DQ": "0"}, True, (True, True)), ({}, False, (False, False))])
def test_fast_train_absmax_policy(monkeypatch, env, explicit, want):
    """Unset: on, except under expert offload or the training arena (the trainer's guards); 0 off; 1 required; an explicit
    argument wins over the environment."""
    import experts4bit_qlora.engines.fast as fast
    from experts4bit_qlora.engines.fast import absmax_dq_policy
    monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)                 # the production default (conftest pins it off)
    for k in ("E4B_ABSMAX_DQ", "OFFLOAD_EXPERTS", "TRAIN_ARENA"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert absmax_dq_policy(explicit) == want


def test_fast_train_default_compresses_a_resident_model(monkeypatch):
    from experts4bit_qlora.engines.fast import _default_absmax_dq
    import experts4bit_qlora.engines.fast as fast
    monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)
    monkeypatch.delenv("E4B_ABSMAX_DQ", raising=False)
    mod = _lora()
    rec = _default_absmax_dq(mod, None, patched=1, verbose=False)
    assert rec == {"compressed": 1, "requested": True, "required": False, "kept_fp32": None}
    assert is_absmax_compressed(mod.base)
    assert _default_absmax_dq(mod, None, patched=1, verbose=False)["compressed"] == 0     # idempotent: the trainer may have done it


def test_fast_train_default_off_keeps_fp32(monkeypatch):
    from experts4bit_qlora.engines.fast import _default_absmax_dq
    import experts4bit_qlora.engines.fast as fast
    monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)
    monkeypatch.setenv("E4B_ABSMAX_DQ", "0")
    mod = _lora()
    assert _default_absmax_dq(mod, None, patched=1, verbose=False)["kept_fp32"] == "off"
    assert not is_absmax_compressed(mod.base)
    monkeypatch.delenv("E4B_ABSMAX_DQ")
    assert _default_absmax_dq(mod, None, patched=0, verbose=False)["kept_fp32"] == "nothing patched"
    assert not is_absmax_compressed(mod.base)


def test_fast_train_default_keeps_fp32_where_the_compressor_refuses(monkeypatch):
    """A refusal leaves the model unchanged and is recorded under the default; required, it raises."""
    from experts4bit_qlora.engines.fast import _default_absmax_dq
    import experts4bit_qlora.engines.fast as fast
    monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)
    monkeypatch.delenv("E4B_ABSMAX_DQ", raising=False)
    mod = _lora()
    mod.base.gate_up_absmax = torch.empty(0, device=DEVICE)          # an offload-evicted placeholder: the compressor refuses
    rec = _default_absmax_dq(mod, None, patched=1, verbose=False)
    assert rec["compressed"] == 0 and "evicted" in rec["kept_fp32"]
    assert not is_absmax_compressed(mod.base)
    with pytest.raises(ValueError, match="evicted"):
        _default_absmax_dq(mod, True, patched=1, verbose=False)


def test_fast_train_required_with_nothing_to_compress_raises(monkeypatch):
    from experts4bit_qlora.engines.fast import _default_absmax_dq
    import experts4bit_qlora.engines.fast as fast
    monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)
    monkeypatch.delenv("E4B_ABSMAX_DQ", raising=False)
    m = nn.Sequential(nn.Linear(4, 4))
    assert _default_absmax_dq(m, None, patched=1, verbose=False)["kept_fp32"] == "no expert stack to compress"
    with pytest.raises(ValueError, match="no expert absmax could be compressed"):
        _default_absmax_dq(m, True, patched=1, verbose=False)


@CUDA
def test_fast_train_compresses_by_default(monkeypatch):
    """The default end to end: enable_fast_train patches, then compresses, and records it."""
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    from experts4bit_qlora import enable_fast_train
    from experts4bit_qlora.engines.fast import FAST_TRAIN_STATS
    import experts4bit_qlora.engines.fast as fast
    monkeypatch.setattr(fast, "ABSMAX_DQ_DEFAULT", True)
    monkeypatch.delenv("E4B_ABSMAX_DQ", raising=False)
    mod = _lora(8, 256, 128, device="cuda", compute_dtype=torch.bfloat16).train()
    assert enable_fast_train(mod) == 1
    assert is_absmax_compressed(mod.base) and FAST_TRAIN_STATS["absmax_dq"]["compressed"] == 1
    args = _inputs(256, 8, n_tok=32, dtype=torch.bfloat16)
    o, dx, g = _fwd_bwd(mod, *args)
    assert torch.isfinite(o).all() and torch.isfinite(dx).all()
